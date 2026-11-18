"""Numbers for Figure 3 (method overview): survivor law of one step under two coin draws (CPU only).

Usage: python -I fig3_numbers.py <model_dir> <context> <flip_node> <show_node> [<show_node> ...]
Full-vocabulary string law of the next token after <context>, its prefix forest, masses M and stop
probabilities q.  Two coin vectors are compared: every node takes its more likely outcome (stop iff
q >= 1/2), except <flip_node>, which stops in the first draw and goes on in the second.  For each draw
the script prints the survivor law (eq. survivor) of the <show_node>s, the rest of the mass and the
entropy of the coder input; it also prints p, M, q of <flip_node>, the number of coin nodes and of
nodes with q in [0.05, 0.95], and Hstr, Hstop.  Nodes are written with ␣ for a space.
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
    return -sum(x / t * math.log2(x / t) for x in ps if x > 0) if t > 0 else 0.0


def enc(name):
    return name.replace("␣", " ").encode("utf-8")


def main():
    mdir, ctx, flip = sys.argv[1], sys.argv[2], enc(sys.argv[3])
    show = [enc(s) for s in sys.argv[4:]]
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
    torch.set_num_threads(6)
    model = AutoModelForCausalLM.from_pretrained(mdir, local_files_only=True, dtype=torch.bfloat16).eval()
    ids = tok(ctx)["input_ids"]
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, -1].float()
    table = byte_table(tok, logits.numel())
    logits[torch.tensor([len(b) == 0 for b in table])] = -float("inf")
    probs = torch.softmax(logits.double(), -1).tolist()
    law = {}
    for i, v in enumerate(probs):
        if v > 0:
            law[table[i]] = law.get(table[i], 0.0) + v
    S = set(law)
    parent = {v: next((v[:j] for j in range(len(v) - 1, 0, -1) if v[:j] in S), None) for v in S}
    children = {v: [] for v in S}
    for v, par in parent.items():
        if par is not None:
            children[par].append(v)
    M = {}
    for v in sorted(S, key=len, reverse=True):
        M[v] = law[v] + sum(M[c] for c in children[v])
    nodes = [v for v in S if children[v]]
    q = {v: law[v] / M[v] for v in nodes}
    hb = lambda x: H([x, 1 - x])
    Hstr, Hstop = H(list(law.values())), sum(M[v] * hb(q[v]) for v in nodes)
    mid = sum(1 for v in nodes if 0.05 <= q[v] <= 0.95)
    print(f"context {ctx!r}: strings={len(S)} coin nodes={len(nodes)} with q in [.05,.95]: {mid}")
    print(f"Hstr={Hstr:.4f} Hstop={Hstop:.4f} Hstr-Hstop={Hstr - Hstop:.4f}")
    print(f"flip node {sys.argv[3]}: p={law[flip]:.5f} M={M[flip]:.5f} q={q[flip]:.4f}"
          f" q*M={q[flip] * M[flip]:.5f} (1-q)*M={(1 - q[flip]) * M[flip]:.5f}")
    for stop_flip in (True, False):
        coins = {v: q[v] >= 0.5 for v in nodes}
        coins[flip] = stop_flip
        Pt = {}
        for v in S:
            anc, a, ok = [], parent[v], True
            while a is not None:
                if coins[a]:
                    ok = False
                    break
                anc.append(a)
                a = parent[a]
            if not ok or (children[v] and not coins[v]):
                continue
            Pt[v] = M[v] * math.prod(M[a] / (M[a] - law[a]) for a in anc)
        shown = sum(Pt.get(v, 0.0) for v in show)
        print(f"\ndraw: {sys.argv[3]} {'stops' if stop_flip else 'goes on'}; survivors={len(Pt)} total={sum(Pt.values()):.6f}"
              f" H(coder input)={H(list(Pt.values())):.4f}")
        for name, v in zip(sys.argv[4:], show):
            print(f"  {name:<12} survives={v in Pt}  Pt={Pt.get(v, 0.0):.5f}")
        print(f"  rest         Pt={sum(Pt.values()) - shown:.5f}")
        sub = sorted(((Pt[v], v) for v in Pt if v.startswith(flip)), reverse=True)
        print("  survivors under flip node: " + ", ".join(f"{v.decode(errors='replace')!r} {x:.5f}" for x, v in sub[:6]))


main()
