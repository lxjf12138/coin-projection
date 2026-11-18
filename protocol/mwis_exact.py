"""Exact MWIS for prefix conflict graphs (the core of Yan et al. 2023, Definition 3).

Yan et al. (2023) define their filter as a maximum weight independent set of the candidate graph, whose
edges join two candidates when one is a prefix of the other, and they solve it with a greedy
forest-decomposition approximation because MWIS is NP-hard on general graphs.  A prefix conflict graph is
not general: it is the comparability graph of the prefix forest, so an independent set is an antichain of
that forest and the heaviest one follows from the tree recursion

    best(v) = max( w(v), sum_{c child of v} best(c) ).

`prefix_mwis` evaluates it level by level in O(|S| + #prefix pairs inside S) numpy work (the pairs come from
the support's static ancestors, as for the coin projection), so MWIS can run at every k, including the full
vocabulary.  Identical byte strings conflict with each other; such a group keeps
its heaviest member (ties: earliest pool position).  Exact ties between a node and its children keep the node.

Self-test (tokenizer only, CPU):
    python -I mwis_exact.py --selftest --model-dir .../Llama-3.2-3B [--model-dir ...] --laws 40
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

_STEPWISE_SRC = os.environ.get("STEPWISE_SRC")
if _STEPWISE_SRC:
    sys.path.insert(0, _STEPWISE_SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def prefix_mwis(env, ids, vals) -> np.ndarray:
    """Positions (ascending) of a heaviest prefix-free subset of the pool (ids, vals)."""
    ids = np.asarray(ids, dtype=np.int64)
    w = np.asarray(vals, dtype=np.float64)
    n = ids.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    reps = env.rep_of[ids]
    if (reps < 0).any():
        raise RuntimeError("pool contains a token without visible bytes")
    order = np.lexsort((np.arange(n), -w))                    # weight desc, position asc
    uniq, first = np.unique(reps[order], return_index=True)
    member = order[first]                                     # heaviest member of each byte string
    nw = w[member]
    m = uniq.size
    if env.proj != "dense" and env.use_sparse(m):
        a, d = env.index.support_pairs(uniq)                 # same pairs, same order as the dense mask
    else:
        node = np.full(env.V, -1, dtype=np.int64)
        node[uniq] = np.arange(m)
        a, d = node[env.anc], node[env.desc]
        live = (a >= 0) & (d >= 0)
        a, d = a[live], d[live]
    depth = np.bincount(d, minlength=m)                       # in-pool ancestors form a chain
    parent = np.full(m, -1, dtype=np.int64)
    if a.size:
        o = np.lexsort((env.klen[uniq[a]], d))                # per d, longest ancestor last
        last = np.r_[d[o][1:] != d[o][:-1], True]
        parent[d[o][last]] = a[o][last]

    best = np.zeros(m)
    take = np.zeros(m, dtype=bool)
    childsum = np.zeros(m)
    levels = [np.flatnonzero(depth == L) for L in range(int(depth.max()) + 1)]
    for lv in reversed(levels):
        take[lv] = nw[lv] >= childsum[lv]
        best[lv] = np.where(take[lv], nw[lv], childsum[lv])
        pv = parent[lv]
        has = pv >= 0
        np.add.at(childsum, pv[has], best[lv][has])
    blocked = np.zeros(m, dtype=bool)
    for lv in levels:
        pv = parent[lv]
        has = pv >= 0
        blocked[lv[has]] = blocked[pv[has]] | take[pv[has]]
    return np.sort(member[take & ~blocked])


# ---------------------------------------------------------------------------
# Independent references for the self-test
# ---------------------------------------------------------------------------

def _ref_tree_dp(strings, w):
    """O(n^2) pure-Python tree DP on raw byte strings (no index tables); returns best weight."""
    n = len(strings)
    groups = {}
    for i in range(n):
        groups.setdefault(strings[i], []).append(i)
    keys = list(groups)
    gw = {s: max(w[i] for i in groups[s]) for s in keys}
    parent = {}
    for s in keys:
        anc = [t for t in keys if t != s and s.startswith(t)]
        parent[s] = max(anc, key=len) if anc else None
    kids = {s: [] for s in keys}
    for s, p in parent.items():
        if p is not None:
            kids[p].append(s)

    def best(s):
        return max(gw[s], sum(best(c) for c in kids[s]))

    return sum(best(s) for s in keys if parent[s] is None)


def _brute(strings, w):
    """Exhaustive search per connected component (components of at most 16 nodes)."""
    n = len(strings)
    adj = [[j for j in range(n) if j != i and (strings[i].startswith(strings[j]) or strings[j].startswith(strings[i]))]
           for i in range(n)]
    seen, total = set(), 0.0
    for r in range(n):
        if r in seen:
            continue
        comp, st = [], [r]
        seen.add(r)
        while st:
            v = st.pop()
            comp.append(v)
            for u in adj[v]:
                if u not in seen:
                    seen.add(u)
                    st.append(u)
        if len(comp) > 16:
            return None
        bestw = 0.0
        for mask in range(1 << len(comp)):
            sel = [comp[b] for b in range(len(comp)) if mask >> b & 1]
            if all(v not in adj[u] for u, v in itertools.combinations(sel, 2)):
                bestw = max(bestw, sum(w[i] for i in sel))
        total += bestw
    return total


def selftest(model_dirs, n_laws, seed, ks):
    import torch

    import c4proto as C
    import gof_single_step as G
    from tokenization_consistency import pools as SW

    rng = np.random.default_rng(seed)
    print(f"{'model':16s} {'k':>5s} {'laws':>4s} {'suboptimal':>10s} {'mean gap':>9s} {'max gap':>8s} "
          f"{'dKL max':>8s} {'ms exact':>8s} {'ms greedy':>9s}")
    for md in model_dirs:
        env = G.stub_env(md, "auto")
        name = os.path.basename(md.rstrip("/"))
        for k in ks:
            sub, gaps, dkl, te, tg = 0, [], [], [], []
            for li in range(n_laws):
                logits = torch.log(G.synthetic_law(env, rng)) / [1.0, 0.7, 1.5][li % 3]
                p = torch.softmax(logits, 0)
                ids, vals, _ = C.topk(p, k)
                idn, vn = ids.numpy(), vals.numpy().astype(np.float64)
                t0 = time.perf_counter()
                keep = prefix_mwis(env, idn, vn)
                te.append(time.perf_counter() - t0)
                strings = [env.table[int(i)] for i in idn]
                sel = [strings[i] for i in keep]
                srt = sorted(sel)   # a prefix of any string is also a prefix of its sorted successor
                assert all(not srt[j + 1].startswith(srt[j]) for j in range(len(srt) - 1)), "not prefix-free"
                we = float(vn[keep].sum())
                if idn.size <= 1024:
                    wr = _ref_tree_dp(strings, vn.tolist())
                    assert abs(we - wr) <= 1e-12 * max(1.0, wr), (we, wr)
                if idn.size <= 16:
                    wb = _brute(strings, vn.tolist())
                    if wb is not None:
                        assert abs(we - wb) <= 1e-12, (we, wb)
                if idn.size <= 4096:
                    _, adj = env.conflict(idn)
                    t0 = time.perf_counter()
                    kg = SW.maximum_weight_independent_set(adj, vn.tolist())
                    tg.append(time.perf_counter() - t0)
                    wg = float(vn[kg].sum())
                    assert wg <= we + 1e-12, (wg, we)
                    gap = (we - wg) / we
                    gaps.append(gap)
                    sub += gap > 1e-12
                    dkl.append(np.log(we) - np.log(wg))   # extra nats per step of the greedy filter
            kl = "Full" if k == 0 else str(k)
            if gaps:
                print(f"{name:16s} {kl:>5s} {n_laws:4d} {sub:10d} {np.mean(gaps):9.2e} {np.max(gaps):8.2e} "
                      f"{np.max(dkl):8.2e} {1e3 * np.mean(te):8.2f} {1e3 * np.mean(tg):9.2f}")
            else:
                print(f"{name:16s} {kl:>5s} {n_laws:4d} {'-':>10s} {'-':>9s} {'-':>8s} {'-':>8s} "
                      f"{1e3 * np.mean(te):8.2f} {'-':>9s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model-dir", action="append", default=[])
    ap.add_argument("--laws", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--k", type=int, nargs="+", default=[16, 64, 256, 1024, 4096, 0])
    args = ap.parse_args()
    if args.selftest:
        selftest(args.model_dir, args.laws, args.seed, args.k)


if __name__ == "__main__":
    main()
