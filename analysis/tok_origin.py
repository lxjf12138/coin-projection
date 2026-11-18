"""Where the prefix pairs of each tokenizer come from (CPU only, tokenizer files only).

For each tokenizer: tokenizer model and pre-tokenizer, UTF-8 classes of the regular
byte strings, the prefix forest (pairs, internal nodes, depth), the largest subtrees,
and a classification of every (prefix, extension) pair by what the prefix is.

Usage: cd /tmp && python -I .../tok_origin.py --protocol-dir .../protocol --out OUT.json M1 M2 ...
"""
import argparse, collections, json, os, sys, unicodedata


def utf8_class(b: bytes) -> str:
    try:
        b.decode("utf-8")
        return "complete"
    except UnicodeDecodeError as e:
        if 0x80 <= b[0] <= 0xBF:
            return "starts_continuation"
        if e.start > 0 or e.reason == "unexpected end of data":
            # valid prefix of a UTF-8 string that stops inside a character
            try:
                b[: e.start].decode("utf-8")
                return "partial_tail" if e.reason == "unexpected end of data" else "invalid"
            except UnicodeDecodeError:
                return "invalid"
        return "invalid"


def script_of(ch: str) -> str:
    o = ord(ch)
    if 0x3040 <= o <= 0x30FF:
        return "kana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        return "han"
    if ch.isspace():
        return "space"
    if ch.isascii() and ch.isalpha():
        return "latin"
    cat = unicodedata.category(ch)
    return "punct" if cat[0] in "PS" else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.protocol_dir)
    import numpy as np
    from transformers import AutoTokenizer
    import c4proto as C
    import coin_core as K

    out = {}
    for md in a.models:
        name = os.path.basename(md.rstrip("/"))
        cfg = json.load(open(os.path.join(md, "config.json")))
        tj = json.load(open(os.path.join(md, "tokenizer.json")))
        V = int(cfg["vocab_size"])
        tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
        table, kind = C.token_bytes_table(tok, V)
        idx = K.CoinIndex(table)
        keys = list(idx.key_to_rep)                     # unique non-empty byte strings
        rep_key = {r: k for k, r in idx.key_to_rep.items()}
        r = dict(tokenizer_model=tj["model"]["type"], byte_fallback=tj["model"].get("byte_fallback"),
                 n_merges=len(tj["model"].get("merges") or []), kind=kind, V=V,
                 n_keys=len(keys), n_pairs=int(idx.anc.size))
        cls = collections.Counter(utf8_class(k) for k in keys)
        r["utf8_classes"] = dict(cls)
        r["single_byte_keys"] = sum(len(k) == 1 for k in keys)
        r["space_initial_keys"] = sum(k[:1] == b" " for k in keys)
        r["space_initial_share"] = r["space_initial_keys"] / len(keys)
        # forest structure
        n_ext = np.bincount(idx.anc, minlength=V)
        r["internal_nodes"] = int((n_ext > 0).sum())
        # depth = longest chain of nested keys
        depth = {}
        for k in sorted(keys, key=len):
            rr = idx.key_to_rep[k]
            lo, hi = idx.anc_ptr[rr], idx.anc_ptr[rr + 1]
            depth[rr] = 1 + max((depth[int(x)] for x in idx.anc[lo:hi]), default=0)
        r["max_chain"] = max(depth.values())
        top = np.argsort(-n_ext)[:12]
        r["largest_subtrees"] = [dict(prefix=repr(rep_key[int(t)]), extensions=int(n_ext[t]),
                                      share_of_keys=round(float(n_ext[t]) / len(keys), 4)) for t in top]
        # 3-byte UTF-8 lead bytes as hubs
        lead = {}
        for b in range(0xE0, 0xF0):
            rr = idx.key_to_rep.get(bytes([b]))
            if rr is not None and n_ext[rr] > 0:
                lead[f"0x{b:02X}"] = int(n_ext[rr])
        r["lead_byte_extensions"] = lead
        # classify every pair by its prefix
        pc = collections.Counter()
        for x, y in zip(idx.anc.tolist(), idx.desc.tolist()):
            p = rep_key[x]
            if p == b" ":
                pc["bare_space"] += 1
            elif len(p) == 1:
                pc["other_single_byte"] += 1
            elif utf8_class(p) != "complete":
                pc["partial_utf8"] += 1
            else:
                s = p.decode("utf-8")
                core = s.lstrip(" ")
                sc = {script_of(c) for c in core} or {"space"}
                if sc <= {"han", "kana"}:
                    pc["cjk_word_or_char"] += 1
                elif sc <= {"latin"}:
                    pc["latin_word_piece"] += 1
                elif sc <= {"space"}:
                    pc["space_run"] += 1
                else:
                    pc["other_complete"] += 1
        r["pairs_by_prefix"] = dict(pc)
        # CJK multi-character keys: does a shorter whole-character key prefix them?
        multi = with_pref = 0
        for k in keys:
            if utf8_class(k) != "complete":
                continue
            s = k.decode("utf-8")
            if len(s) >= 2 and all(script_of(c) in ("han", "kana") for c in s):
                multi += 1
                if any(s[:j].encode() in idx.key_to_rep for j in range(1, len(s))):
                    with_pref += 1
        r["cjk_multichar_keys"] = multi
        r["cjk_multichar_with_char_prefix"] = with_pref
        # BPE merge closure: the left part of every merge is a proper prefix of the result
        if tj["model"]["type"] == "BPE":
            inv = C._bytes_to_unicode_inv()
            ok = tot = 0
            for m in tj["model"]["merges"]:
                l, rgt = (m.split(" ", 1) if isinstance(m, str) else m)
                lb = bytes(inv[c] for c in l)
                tot += 1
                ok += lb in idx.key_to_rep and (lb + bytes(inv[c] for c in rgt)) in idx.key_to_rep
            r["merges_left_part_is_key"] = ok / tot
        out[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "largest_subtrees"}), flush=True)
        print("  largest:", r["largest_subtrees"][:8], flush=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
