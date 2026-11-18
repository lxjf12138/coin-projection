"""Dump the prefix forest of one step per context (Figures 1 and 3; one model load for all contexts).

Usage: python -I fig3_dump_forest.py <model_dir> <min_mass> <out_prefix> <context> [<context> ...]
Writes <out_prefix><slug>.json per context (slug: the context with every non-alphanumeric character as _):
  summary   strings, roots, coin nodes, H_str = H_root + H_branch + H_stop (bits), mass of the small roots;
  nodes     every string with subtree mass >= min_mass: string (␣ for space), byte length, parent, p, M, q,
            number of children, the stop term M h(q), and the mass of the children below the threshold ("rest");
  law       the full string law (hex bytes -> probability; full vocabulary, temperature 1, float64 softmax);
  splits    for every prefix pair v < w among the 12 most probable strings whose remainder w - v is one token:
            p(v), p(w), p(remainder | context + v) and the probability p(v) p(remainder | ...) that the
            sampled pair of tokens reads as w at this step (one more forward per pair).
Runs on the GPU when one is visible (bf16 weights); the law itself is computed in float64.
"""
import json
import math
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def bytes_to_unicode_inv():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def byte_table(tok, width):
    data = json.loads(tok.backend_tokenizer.to_str())
    bytelevel = "ByteLevel" in json.dumps(data.get("decoder"))
    added = {a["id"] for a in data.get("added_tokens", [])} | set(tok.all_special_ids)
    inv = bytes_to_unicode_inv()
    pieces = tok.convert_ids_to_tokens(list(range(min(len(tok), width))))
    out = []
    for i in range(width):
        t = pieces[i] if i < len(pieces) else None
        if t is None or i in added:
            out.append(b"")
        elif bytelevel:
            try:
                out.append(bytes(inv[c] for c in t))
            except KeyError:
                out.append(b"")
        else:
            m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", t)
            out.append(bytes([int(m.group(1), 16)]) if m else t.replace("▁", " ").encode("utf-8"))
    return out


def H(ps):
    t = sum(ps)
    return -sum(p / t * math.log2(p / t) for p in ps if p > 0) if t > 0 else 0.0


def name(b):
    return b.decode("utf-8", errors="backslashreplace").replace(" ", "␣").replace("\n", "↵")


def main():
    mdir, thr, prefix = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
    torch.set_num_threads(6)
    model = AutoModelForCausalLM.from_pretrained(mdir, local_files_only=True, dtype=torch.bfloat16).to(dev).eval()
    table, mask = None, None

    def law_after(ids):
        nonlocal table, mask
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=dev)).logits[0, -1].double().cpu()
        if table is None:
            table = byte_table(tok, logits.numel())
            mask = torch.tensor([len(b) == 0 for b in table])
        logits[mask] = -float("inf")
        return torch.softmax(logits, -1)

    for ctx in sys.argv[4:]:
        ids = tok(ctx)["input_ids"]
        probs = law_after(ids)
        pl = probs.tolist()
        law, best_id = {}, {}
        for i, v in enumerate(pl):
            if v > 0:
                b = table[i]
                law[b] = law.get(b, 0.0) + v
                if b not in best_id or v > pl[best_id[b]]:
                    best_id[b] = i
        S = set(law)
        parent = {v: next((v[:j] for j in range(len(v) - 1, 0, -1) if v[:j] in S), None) for v in S}
        children = {v: [] for v in S}
        for v, par in parent.items():
            if par is not None:
                children[par].append(v)
        M = {}
        for v in sorted(S, key=len, reverse=True):
            M[v] = law[v] + sum(M[c] for c in children[v])
        roots = [v for v in S if parent[v] is None]
        stop = {v: M[v] * H([law[v], M[v] - law[v]]) for v in S if children[v]}
        h_root = H([M[r] for r in roots])
        h_branch = sum((M[v] - law[v]) * H([M[c] for c in children[v]]) for v in S if children[v])
        nodes = []
        for v in S:
            if M[v] >= thr:
                big = [c for c in children[v] if M[c] >= thr]
                nodes.append(dict(s=name(v), n=len(v), parent=None if parent[v] is None else name(parent[v]),
                                  p=law[v], M=M[v], q=law[v] / M[v] if children[v] else None, nch=len(children[v]),
                                  stopterm=stop.get(v, 0.0),
                                  rest=sum(M[c] for c in children[v]) - sum(M[c] for c in big)))
        by_bytes = {b: i for i, b in enumerate(table) if b}
        top = sorted(S, key=lambda v: -law[v])[:12]
        splits = []
        for v in top:
            for w in top:
                if w != v and w.startswith(v) and w[len(v):] in by_bytes:
                    rem = w[len(v):]
                    q = law_after(ids + [best_id[v]])
                    p_rem = float(q[[i for i, b in enumerate(table) if b == rem]].sum())
                    splits.append(dict(v=name(v), w=name(w), rem=name(rem), p_v=law[v], p_w=law[w],
                                       p_rem_after_v=p_rem, p_split=law[v] * p_rem))
        summary = dict(context=ctx, model=mdir.rstrip("/").split("/")[-1], device=dev, strings=len(S),
                       roots=len(roots), coin_nodes=len(stop), H_str=H(list(law.values())), H_root=h_root,
                       H_branch=h_branch, H_stop=sum(stop.values()),
                       small_roots_mass=sum(M[v] for v in roots if M[v] < thr))
        out = prefix + re.sub(r"[^A-Za-z0-9]", "_", ctx) + ".json"
        json.dump(dict(summary=summary, nodes=sorted(nodes, key=lambda d: (d["n"], -d["M"])),
                       law={v.hex(): x for v, x in law.items()}, splits=splits),
                  open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print(json.dumps(summary, ensure_ascii=False), f"nodes={len(nodes)}", flush=True)
        for sp in splits:
            print(f"  split {sp['v']} + {sp['rem']} -> {sp['w']}: p(v)={sp['p_v']:.4f} p(w)={sp['p_w']:.4f} "
                  f"p(rem|v)={sp['p_rem_after_v']:.3e} p(split)={sp['p_split']:.3e}", flush=True)
        for v in top:
            print(f"  {name(v):<14} p={law[v]:.4f} M={M[v]:.4f} stop={stop.get(v, 0.0):.4f} kids={len(children[v])}",
                  flush=True)


main()
