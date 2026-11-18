"""E2b: on-trajectory prefix-forest decomposition + PPL, read from E1 records.

Motivation.  E0 decomposes the law at *states drawn by ordinary sampling* and
averages over states; the E1 capacity ratio is a ratio of *trajectory* averages.
The two are different functionals (Jensen), which is why the E0-vs-E1 check
drifts by up to 18% at large k.  This script removes that gap: for every E1 case
it re-runs the model over that case's own prompt + generated tokens (teacher
forcing, incremental KV cache) and decomposes the law *at the states the method
actually visited*.

Per step t of a case:
    p_t          full masked softmax (same tensor the sender saw)
    pr_t         k-truncated representative law, i.e. C._rep_law(env, p_t, k)
    H_key_t      H(pr_t)
    H_root_t     H(roots of the surviving prefix forest)
    H_branch_t   branch entropy
    H_stop_t     stop entropy  (so H_key = H_root + H_branch + H_stop)
    nll_full_t   -log2 p_t(x_t),  nll_law_t = -log2 pr_t(x_t)

Per case we then compare the measured BPT (128 / n_tokens, from the record) with
  pred_coin  = mean_t (H_key_t - H_stop_t)
  pred_sp    = mean_t  H_root_t
which are exactly the two budget quantities of the paper, now on the trajectory
itself.  PPL is 2 ** mean_t nll_full_t.

Read-only: never writes to runs/e1_*.jsonl and never imports anything that the
running E1 job can mutate.

Usage:
  python -I e2b_ontraj.py --tag llama31_8b --model-dir .../Meta-Llama-3.1-8B \
      --in runs/e1_llama31_8b.jsonl --out runs/e2b_llama31_8b.jsonl [--max-cases 40]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_STEPWISE_SRC = os.environ.get("STEPWISE_SRC")
if _STEPWISE_SRC:
    sys.path.insert(0, _STEPWISE_SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import c4proto as C  # noqa: E402
from e0_atlas import decompose  # noqa: E402

METHODS = ("coin", "syncpool", "stepwise", "native")


def load_cases(path, methods, max_cases, max_sample=0, only_complete=True):
    out, seen = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") != "case":
                continue
            if r.get("method") not in methods:
                continue
            if max_sample and int(r.get("sample", 0)) >= max_sample:
                continue
            if only_complete and not r.get("complete"):
                continue
            out.append(r)
            seen += 1
            if max_cases and seen >= max_cases:
                break
    return out


def done_keys(path):
    s = set()
    if not os.path.exists(path):
        return s
    valid = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("kind") == "case":
                    s.add((int(r["k"]), r["method"], int(r["sample"])))
                    valid.append(line)
            except Exception:
                continue
    with open(path, "w", encoding="utf-8") as fh:
        for line in valid:
            fh.write(line + "\n")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--ks", type=int, nargs="+", default=None, help="restrict to these k (0 = Full)")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--max-sample", type=int, default=0, help="restrict to sample < max_sample")
    ap.add_argument("--limit-tokens", type=int, default=0, help="cap steps per case (0 = all)")
    args = ap.parse_args()

    cases = load_cases(args.inp, set(args.methods), args.max_cases, args.max_sample)
    if args.ks is not None:
        keep = set(args.ks)
        cases = [c for c in cases if int(c["k"]) in keep]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = done_keys(args.out)
    todo = [c for c in cases if (int(c["k"]), c["method"], int(c["sample"])) not in done]
    print(f"[e2b] {args.tag}: {len(cases)} target cases ({len(done)} already in out, {len(todo)} todo)", flush=True)

    if todo:
        env = C.Env(args.model_dir)
        with open(args.out, "a", encoding="utf-8") as fh:
            for idx, rec in enumerate(todo):
                t0 = time.perf_counter()
                k = int(rec["k"])
                gen = [int(t) for t in rec["token_ids"]]
                if args.limit_tokens:
                    gen = gen[: args.limit_tokens]
                prompt_ids = env.tok(rec["prompt"])["input_ids"]
                st = env.stepper(prompt_ids)

                acc = dict(H_key=0.0, H_root=0.0, H_branch=0.0, H_stop=0.0,
                           nll_full=0.0, nll_law=0.0)
                steps = 0
                for tok in gen:
                    p = env.law(st)
                    pr, tokp, _ = C._rep_law(env, p, k)
                    dec = decompose(env, pr)[0]
                    acc["H_key"] += dec["H_key"]
                    acc["H_root"] += dec["H_root"]
                    acc["H_branch"] += dec["H_branch"]
                    acc["H_stop"] += dec["H_stop"]
                    pt = float(p[tok])
                    acc["nll_full"] += -math.log2(pt) if pt > 0 else float("nan")
                    pl = float(tokp[tok])
                    acc["nll_law"] += -math.log2(pl) if pl > 0 else float("nan")
                    steps += 1
                    st.append(tok)

                if steps == 0:
                    continue
                m = {key: v / steps for key, v in acc.items()}
                out = dict(kind="case", tag=args.tag, method=rec["method"], k=k,
                           sample=rec["sample"], steps=steps,
                           measured_bpt=rec.get("bpt"),
                           pred_coin=m["H_key"] - m["H_stop"],
                           pred_sp=m["H_root"],
                           H_key=m["H_key"], H_root=m["H_root"],
                           H_branch=m["H_branch"], H_stop=m["H_stop"],
                           nll_full_bits=m["nll_full"], nll_law_bits=m["nll_law"],
                           ppl_full=2.0 ** m["nll_full"] if math.isfinite(m["nll_full"]) else None,
                           ppl_law=2.0 ** m["nll_law"] if math.isfinite(m["nll_law"]) else None,
                           ms=round(1000 * (time.perf_counter() - t0), 1))
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                if (idx + 1) % 25 == 0:
                    fh.flush()
                    print(f"[e2b] {idx + 1}/{len(todo)}  last {out['ms']:.0f} ms", flush=True)

    # ---- summary ----
    keep_k = set(args.ks) if args.ks is not None else None
    per = {}
    with open(args.out, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if args.max_sample and int(r.get("sample", 0)) >= args.max_sample:
                continue
            if keep_k is not None and int(r["k"]) not in keep_k:
                continue
            per.setdefault((r["k"], r["method"]), []).append(r)

    print(f"\n[e2b] {args.tag} summary  (measured vs on-trajectory prediction)")
    print(f"{'k':>5s} {'method':9s} {'n':>3s} {'BPT_meas':>9s} {'pred_coin':>10s} {'pred_sp':>8s} "
          f"{'err_coin':>9s} {'err_sp':>8s} {'PPL':>7s} {'H_stop':>7s}")
    summary = {}
    for (k, method), rows in sorted(per.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        def m_(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")
        bpt, pc, ps = m_("measured_bpt"), m_("pred_coin"), m_("pred_sp")
        ec = 100 * (bpt - pc) / pc if pc else float("nan")
        es = 100 * (bpt - ps) / ps if ps else float("nan")
        ppl, hs = m_("ppl_full"), m_("H_stop")
        print(f"{k:5d} {method:9s} {len(rows):3d} {bpt:9.3f} {pc:10.3f} {ps:8.3f} "
              f"{ec:+8.1f}% {es:+7.1f}% {ppl:7.2f} {hs:7.3f}")
        summary[f"{k}|{method}"] = dict(n=len(rows), bpt=bpt, pred_coin=pc, pred_sp=ps,
                                        err_coin_pct=ec, err_sp_pct=es, ppl=ppl,
                                        H_stop=hs, H_root=m_("H_root"), H_branch=m_("H_branch"),
                                        H_key=m_("H_key"))
    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(tag=args.tag, cells=summary), fh, ensure_ascii=False, indent=1)
    print(f"[e2b] {args.tag} done", flush=True)


if __name__ == "__main__":
    main()
