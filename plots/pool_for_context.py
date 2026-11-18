"""Next-token pool and prefix forest for a free-text context (CPU only).

Usage: python -I pool_for_context.py <model_dir> <k (0 = full)> <n_show> <context> [<context> ...]
For each context: temperature-1 law without special tokens, top-k truncation, string law and
prefix forest; prints Hroot/Hbranch/Hstop, the SyncPool and coin budgets, and the n_show most
probable strings with p, M, q, parent and number of children.  A context ending in "||<piece>"
also prints p(piece) after the context (e.g. "Can I get you any||thing").  A context ending in
"@@<node>|<node>..." (nodes written with ␣ for space) also prints, for each node, p, M, q, its
stop-entropy term M*h(q), its parent and its children, plus the largest stop-entropy terms.
Runs on the GPU when one is visible.
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


def show(b):
    return b.decode("utf-8", errors="backslashreplace").replace(" ", "␣").replace("\n", "↵")


def H(ps):
    t = sum(ps)
    return -sum(p / t * math.log2(p / t) for p in ps if p > 0) if t > 0 else 0.0


def main():
    mdir, k, n_show = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
    torch.set_num_threads(6)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(mdir, local_files_only=True, dtype=torch.bfloat16).to(dev).eval()
    table = None
    for ctx in sys.argv[4:]:
        ctx, _, nodes = ctx.partition("@@")
        ctx, _, piece = ctx.partition("||")
        ids = tok(ctx)["input_ids"]
        print(f"\ncontext {ctx!r} tokens: {[show(table[i]) if table else tok.decode([i]) for i in ids]}")
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=dev)).logits[0, -1].float().cpu()
        if table is None:
            table = byte_table(tok, logits.numel())
            mask = torch.tensor([len(b) == 0 for b in table])
        logits[mask] = -float("inf")
        p = torch.softmax(logits, -1)
        if piece:
            pid = tok(piece, add_special_tokens=False)["input_ids"]
            print(f"\ncontext {ctx!r}: next piece {piece!r} ids={pid} p={p[pid[0]].item():.3e} (full vocabulary)")
            continue
        kk = p.numel() if k == 0 else k
        vals, top = torch.topk(p, kk)
        vals = (vals / vals.sum()).tolist()
        law = {}
        for i, v in zip(top.tolist(), vals):
            law[table[i]] = law.get(table[i], 0.0) + v
        S = set(law)
        parent = {}
        for v in S:
            parent[v] = next((v[:j] for j in range(len(v) - 1, 0, -1) if v[:j] in S), None)
        children = {v: [] for v in S}
        for v, par in parent.items():
            if par is not None:
                children[par].append(v)
        M = {}
        order = sorted(S, key=len, reverse=True)
        for v in order:
            M[v] = law[v] + sum(M[c] for c in children[v])
        roots = [v for v in S if parent[v] is None]
        Hroot = H([M[r] for r in roots])
        Hstop = sum(M[v] * H([law[v], M[v] - law[v]]) for v in S if children[v])
        Hbranch = sum((M[v] - law[v]) * H([M[c] for c in children[v]]) for v in S if children[v])
        print(f"\ncontext {ctx!r}  k={k or 'full'} strings={len(S)} roots={len(roots)}")
        print(f"  Hstr={H(list(law.values())):.3f} = root {Hroot:.3f} + branch {Hbranch:.3f} + stop {Hstop:.3f};"
              f" SyncPool {Hroot:.3f} bits, coin {Hroot + Hbranch:.3f} bits")
        def hb(q):
            return H([q, 1 - q])
        if nodes:
            for name in nodes.split("|"):
                v = name.replace("␣", " ").replace("↵", "\n").encode("utf-8")
                if v not in S:
                    print(f"  node {name}: not in pool")
                    continue
                kids = sorted(children[v], key=lambda c: -M[c])
                print(f"  node {name}: p={law[v]:.4f} M={M[v]:.4f} q={law[v] / M[v]:.3f} stopterm={M[v] * hb(law[v] / M[v]):.4f}"
                      f" parent={show(parent[v]) if parent[v] else '-'} children={len(kids)}: "
                      + ", ".join(f"{show(c)} {M[c]:.4f}" for c in kids[:8]))
            terms = sorted(((M[v] * hb(law[v] / M[v]), v) for v in S if children[v]), reverse=True)[:8]
            print("  largest stop terms: " + ", ".join(f"{show(v)} {t:.4f}" for t, v in terms))
        big = sorted(roots, key=lambda r: -M[r])[:3]
        print("  largest trees: " + ", ".join(f"{show(r)} M={M[r]:.3f} children={len(children[r])}" for r in big))
        for v in sorted(S, key=lambda v: -law[v])[:n_show]:
            chain, u = [], parent[v]
            while u is not None:
                chain.append(f"{show(u)}(p={law[u]:.4f},q={law[u] / M[u]:.3f})")
                u = parent[u]
            print(f"  {show(v):<14} p={law[v]:.4f} M={M[v]:.4f} q={law[v] / M[v]:.3f} kids={len(children[v]):<3} < {' < '.join(chain)}")


main()
