"""TB: CPU microbenchmark of every method's per-step work on fixed laws (the states saved by stage R5).

The forward is shared by all methods and timed in stage T; this isolates what differs between them.  Per
(model, state, k, method, path) it times R repetitions of the lean plan call, the token choice and the coder
step, each repetition with a fresh key and step (fresh coins, member and homograph draws).  Methods and paths
run round-robin inside every repetition, with a rotated order, so a slow drift of the machine hits all alike.
`path` is the prefix-pair processing (auto = the run setting; dense / sparse for the methods that have both).

Also written, per (state, k): machine-independent counts that the cost is linear in -- |S| (keys in the
top-k pool), live prefix pairs inside S, coin nodes and survivors, SyncPool groups.

Every method is timed on the same states: the first n saved states on which each benchmarked (method, k)
plans a non-empty pool.  The saved prefixes come from ordinary sampling, which can leave a non-canonical
history; Stepwise cannot extend such a history (no next token keeps it canonical), so that state is skipped
for all methods and recorded in the meta line.

  cd /tmp && python -I .../timing_bench.py --model-dir M --tag T --states-in gof_states_T.npz --out tb_T.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_STEPWISE_SRC = os.environ.get("STEPWISE_SRC")
if _STEPWISE_SRC:
    sys.path.insert(0, _STEPWISE_SRC)
sys.path.insert(0, HERE)

import c4proto as C  # noqa: E402
import gof_single_step as G  # noqa: E402

PATHS = {"coin": ("auto", "dense", "sparse"), "syncpool": ("auto", "dense", "sparse"),
         "mwis": ("auto", "dense", "sparse"), "basic": ("auto", "dense", "sparse")}


def counts(env, p, k):
    """Machine-independent sizes of one (law, k)."""
    env.proj = "auto"
    S, pS, _, _ = C._key_law(env, p, k)
    sa, sd = env.index.support_pairs(S)
    ids, vals, _ = C._topk_raw(p, k)
    _, gw, _ = C.group_items(env, env.rep_of[ids.numpy()], vals.numpy())
    C.DIAG = True
    plan, _ = C.plan_coin(env, p, k, [], b"\0" * 32, 0)
    C.DIAG = False
    return dict(pool=int(ids.numel()), keys=int(S.size), pairs=int(sa.size), nodes=int(np.unique(sa).size),
                survivors=int(plan.stats["pool"]), groups=int(gw.size))


def methods_at(args, k):
    """The benchmarked methods at k (the filter methods stop at filter_max_k, Basic at 1024)."""
    out = []
    for m in args.methods:
        if m in ("basic", "stepwise") and (k == 0 or k > args.filter_max_k):
            continue
        if m == "basic" and k > 1024:      # the official per-item rule is quadratic in Python
            continue
        out.append(m)
    return out


def empty_pool(env, args, p, prefix, j):
    """The first benchmarked (method, k) whose plan has an empty pool on this state, else None."""
    for k in args.k:
        for m in methods_at(args, k):
            env.proj = "auto"
            key = hashlib.sha256(f"tb-probe|{args.tag}|{j}|{k}".encode()).digest()
            plan, _ = C.METHODS[m](env, p, k, prefix, key, 0)
            if len(plan.weights) == 0:
                return dict(method=m, k=k)
    return None


def one_call(env, method, p, k, prefix, key, step, value):
    C.TIMERS.clear()
    s = time.perf_counter()
    plan, _ = C.METHODS[method](env, p, k, prefix, key, step)
    t_plan = time.perf_counter() - s
    cd = C.make_coder("rrc", 64 * 128, key)
    s = time.perf_counter()
    i = cd.encode_step(plan.weights, value)
    t_code = time.perf_counter() - s
    s = time.perf_counter()
    plan.token(i)
    t_plan += time.perf_counter() - s
    return t_plan, t_code, dict(C.TIMERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--states-in", required=True)
    ap.add_argument("--n-states", type=int, default=10)
    ap.add_argument("--k", type=int, nargs="+", default=[16, 64, 256, 1024, 4096, 0])
    ap.add_argument("--methods", nargs="+",
                    default=["native", "basic", "mwis", "stepwise", "syncpool", "coin", "hubdrop"])
    ap.add_argument("--filter-max-k", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = G.stub_env(args.model_dir, "auto")
    env.mwis_solver, env.proj_also = "exact", False
    z = np.load(args.states_in, allow_pickle=False)
    if int(z["V"]) != env.V:
        raise SystemExit("states were saved for another vocabulary")
    C.DIAG = False
    picked, skipped = [], []
    for j in range(z["laws"].shape[0]):
        if len(picked) == args.n_states:
            break
        p = torch.from_numpy(z["laws"][j].copy())
        prefix = z["prefix_ids"][j][: int(z["prefix_len"][j])].tolist()
        bad = empty_pool(env, args, p, prefix, j)
        if bad is None:
            picked.append(j)
        else:
            skipped.append(dict(state=j, **bad))
            print(f"[tb] {args.tag} state {j} skipped: {bad['method']} plans an empty pool at k={bad['k']}",
                  flush=True)
    env.proj = "auto"
    if len(picked) < args.n_states:
        print(f"[tb] warning: only {len(picked)} of {args.n_states} states usable", flush=True)
    meta = dict(kind="meta", tag=args.tag, states=len(picked), state_ids=picked, skipped_states=skipped,
                args=vars(args), **C.machine_meta(args.model_dir))
    rng = np.random.default_rng(20261114)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta) + "\n")
        print(json.dumps({a: b for a, b in meta.items() if a not in ("model_files", "model_digests")}), flush=True)
        for j in picked:
            p = torch.from_numpy(z["laws"][j].copy())
            prefix = z["prefix_ids"][j][: int(z["prefix_len"][j])].tolist()
            for k in args.k:
                cnt = counts(env, p, k)
                fh.write(json.dumps(dict(kind="counts", state=j, k=k, **cnt)) + "\n")
                jobs = []
                for m in methods_at(args, k):
                    jobs += [(m, path) for path in PATHS.get(m, ("auto",))]
                res = {jb: dict(plan=[], code=[], timers=[]) for jb in jobs}
                for r in range(args.warmup + args.reps):
                    rot = r % len(jobs)
                    for m, path in jobs[rot:] + jobs[:rot]:
                        env.proj = path
                        key = hashlib.sha256(f"tb|{args.tag}|{j}|{k}|{r}".encode()).digest()
                        value = int.from_bytes(rng.bytes(64 * 128 // 8), "big")
                        tp, tc, tm = one_call(env, m, p, k, prefix, key, 1000 + r, value)
                        if r >= args.warmup:
                            res[(m, path)]["plan"].append(tp)
                            res[(m, path)]["code"].append(tc)
                            res[(m, path)]["timers"].append(tm)
                env.proj = "auto"
                for (m, path), d in res.items():
                    a = 1000 * np.asarray(d["plan"])
                    names = sorted({x for t in d["timers"] for x in t})
                    row = dict(kind="call", state=j, k=k, method=m, path=path, reps=a.size,
                               ms_plan=float(np.median(a)), ms_plan_p10=float(np.percentile(a, 10)),
                               ms_plan_p90=float(np.percentile(a, 90)), ms_plan_mean=float(a.mean()),
                               ms_code=float(np.median(1000 * np.asarray(d["code"]))),
                               **{f"ms_{x}": float(np.median([1000 * t.get(x, 0.0) for t in d["timers"]]))
                                  for x in names})
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                kl = "Full" if k == 0 else str(k)
                print(f"[tb] {args.tag} state {j} k={kl:>4s} keys={cnt['keys']} pairs={cnt['pairs']} "
                      f"nodes={cnt['nodes']}  " + "  ".join(
                          f"{m}/{path[0]}={1000 * np.median(d['plan']):.2f}" for (m, path), d in res.items()),
                      flush=True)


if __name__ == "__main__":
    main()
