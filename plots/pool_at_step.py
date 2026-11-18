"""Candidate pool, prefix forest and entropy terms at one step of a recorded E1 case (CPU only).

Usage: python -I pool_at_step.py <model_dir> <e1_jsonl> <sample> <method> <k> <step> [n_trees] [n_children]
Rebuilds the context as prompt ids + the case's token_ids[:step] (as c4proto does), takes the
temperature-1 law without special tokens, truncates to the top k, forms the string law and its
prefix forest, and prints the largest trees with p(v), M(v) and q_v, plus Hroot/Hbranch/Hstop.
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
    s = b.decode("utf-8", errors="backslashreplace")
    return s.replace(" ", "␣").replace("\n", "↵")


def H(ps):
    t = sum(ps)
    return -sum(p / t * math.log2(p / t) for p in ps if p > 0) if t > 0 else 0.0


def main():
    mdir, path, sample, method, k, step = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
    n_trees = int(sys.argv[7]) if len(sys.argv) > 7 else 6
    n_child = int(sys.argv[8]) if len(sys.argv) > 8 else 6
    rec = None
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "case" and r["sample"] == sample and r["method"] == method and r["k"] == k:
            rec = r
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
    ids = tok(rec["prompt"])["input_ids"] + rec["token_ids"][:step]
    print("context tail:", repr(tok.decode(ids[-12:])), "| emitted next:", show(tok.decode([rec["token_ids"][step]]).encode()))
    torch.set_num_threads(6)
    model = AutoModelForCausalLM.from_pretrained(mdir, local_files_only=True, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, -1].float()
    table = byte_table(tok, logits.numel())
    mask = torch.tensor([len(b) == 0 for b in table])
    logits[mask] = -float("inf")
    p = torch.softmax(logits, -1)
    kk = p.numel() if k == 0 else k
    vals, top = torch.topk(p, kk)
    vals = (vals / vals.sum()).tolist()
    top = top.tolist()
    law, toks = {}, {}
    for i, v in zip(top, vals):
        b = table[i]
        law[b] = law.get(b, 0.0) + v
        toks.setdefault(b, []).append(v)
    S = set(law)
    parent = {}
    for v in S:
        par = None
        for j in range(len(v) - 1, 0, -1):
            if v[:j] in S:
                par = v[:j]
                break
        parent[v] = par
    children = {v: [] for v in S}
    for v, par in parent.items():
        if par is not None:
            children[par].append(v)
    M = {}

    def mass(v):
        if v not in M:
            M[v] = law[v] + sum(mass(c) for c in children[v])
        return M[v]

    for v in S:
        mass(v)
    roots = sorted([v for v in S if parent[v] is None], key=lambda v: -M[v])
    Htok = H(vals)
    Hstr = H(list(law.values()))
    Hroot = H([M[r] for r in roots])
    Hstop = sum(M[v] * H([law[v], M[v] - law[v]]) for v in S if children[v])
    Hbranch = sum((M[v] - law[v]) * H([M[c] for c in children[v]]) for v in S if children[v])
    print(f"k={k} strings={len(S)} roots={len(roots)}  Htok={Htok:.3f} Hstr={Hstr:.3f} = root {Hroot:.3f} + branch {Hbranch:.3f} + stop {Hstop:.3f} (check {Hroot + Hbranch + Hstop:.3f})")
    print(f"budgets: SyncPool {Hroot:.3f} bits, coin {Hstr - Hstop:.3f} bits")

    def walk(v, depth, budget):
        kids = sorted(children[v], key=lambda c: -M[c])
        q = law[v] / M[v]
        print(f"{'   ' * depth}{show(v):<16} p={law[v]:.4f} M={M[v]:.4f}" + (f" q={q:.3f} children={len(kids)}" if kids else ""))
        if depth < 2:
            for c in kids[:budget]:
                walk(c, depth + 1, max(2, budget // 2))
            if len(kids) > budget:
                print(f"{'   ' * (depth + 1)}... {len(kids) - budget} more, mass {sum(M[c] for c in kids[budget:]):.4f}")

    if n_trees < 0:  # node mode: argv[8:] are strings written with ␣ for space and ↵ for newline
        for name in sys.argv[8:]:
            v = name.replace("␣", " ").replace("↵", "\n").encode("utf-8")
            if v not in S:
                print(f"{name:<10} not in pool")
                continue
            kids = sorted(children[v], key=lambda c: -M[c])
            top = ", ".join(f"{show(c)} {M[c]:.4f}" for c in kids[:6])
            print(f"{name:<10} p={law[v]:.4f} M={M[v]:.4f} q={law[v] / M[v]:.3f} parent={show(parent[v]) if parent[v] else '-'} "
                  f"children={len(kids)} [{top}]")
        return
    if n_trees == 0:
        def root_of(v):
            while parent[v] is not None:
                v = parent[v]
            return v
        for v in sorted(S, key=lambda v: -law[v])[:n_child]:
            chain = []
            u = parent[v]
            while u is not None:
                chain.append(show(u) + f"(q={law[u] / M[u]:.3f})")
                u = parent[u]
            print(f"{show(v):<14} p={law[v]:.4f} tokens={len(toks[v])} root={show(root_of(v))!s:<4} ancestors: {' < '.join(chain)}")
        return
    for r in roots[:n_trees]:
        walk(r, 0, n_child)
    print(f"... {len(roots) - n_trees} more roots, mass {sum(M[r] for r in roots[n_trees:]):.4f}")


main()
