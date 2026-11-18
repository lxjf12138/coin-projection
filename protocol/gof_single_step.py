"""Goodness of fit of the emitted token law on fixed contexts, over many keys (review W4).

For each fixed state (prompt plus a short prefix drawn by ordinary sampling) we compute the
model law p once, then draw the next token N times through the real pipeline, each draw with
its own key and its own uniform 128-bit message: plan (projection or pool) -> coder step ->
homograph draw.  An exact method emits tokens with the top-k token law of ordinary sampling,
so a chi-square test of the N tokens against that law should reject at the nominal rate.
Basic and MWIS change the law and serve as positive controls (their power grows with N KL).

Usage (rented GPU):
  cd /tmp && python -I .../gof_single_step.py --model-dir M --tag llama31_8b --prompts P \
      --unit words --out OUT.jsonl [--k 16 256 0] [--n-ctx 20] [--draws 5000]
Two-stage (GPU only for the forward passes):
  ... --states-out S.npz --draws 0            # on the GPU machine, seconds per model
  ... --states-in S.npz --out OUT.jsonl       # anywhere, tokenizer only, CPU-bound draws
CPU self-test (tokenizer only, synthetic hub-heavy laws, no weights):
  cd /tmp && python -I .../gof_single_step.py --selftest --model-dir M --tag t --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import c4proto as C  # noqa: E402


def chi2_binned(counts: dict, law: np.ndarray, n: int, min_expected: float = 5.0):
    """Pearson chi-square of token counts against ``law`` (token ids -> mass, sums to one).

    Tokens with expected count >= min_expected get their own bin; the rest share one bin
    (merged into the smallest kept bin if its own expectation is below min_expected).
    Draws of tokens outside supp(law) make the law impossible: p = 0.
    """
    from scipy.stats import chi2

    ids = np.flatnonzero(law > 0)
    e = law[ids] * n
    order = np.argsort(-e, kind="stable")
    ids, e = ids[order], e[order]
    obs = np.array([counts.get(int(t), 0) for t in ids], dtype=np.float64)
    outside = int(n - obs.sum())
    big = e >= min_expected
    O, E = list(obs[big]), list(e[big])
    rest_o, rest_e = float(obs[~big].sum()), float(e[~big].sum())
    if rest_e >= min_expected or not E:
        O.append(rest_o)
        E.append(rest_e)
    else:
        O[-1] += rest_o
        E[-1] += rest_e
    O, E = np.asarray(O), np.asarray(E)
    df = int(O.size - 1)
    stat = float(((O - E) ** 2 / np.where(E > 0, E, 1.0)).sum())
    pval = 0.0 if outside > 0 else (float(chi2.sf(stat, df)) if df > 0 else 1.0)
    return dict(chi2=stat, df=df, p=pval, outside=outside, bins=int(O.size))


class _FixedLawCache:
    """Within one cell the law p is fixed, so top-k and the key law are computed once.

    Only key-independent work is cached (the sort and the merged key law); every draw
    still derives its own coins, pool member, homograph draw and coder randomness.
    Cached arrays are handed out as copies, so a planner cannot alter later draws.
    """

    def __init__(self):
        self._topk, self._rep, self._key = C.topk, C._rep_law, C._key_law
        self.memo_topk, self.memo_rep, self.memo_key = {}, {}, {}

    def __enter__(self):
        def topk(p, k):
            key = (id(p), k)
            if key not in self.memo_topk:
                self.memo_topk[key] = self._topk(p, k)
            ids, vals, kk = self.memo_topk[key]
            return ids.clone(), vals.clone(), kk

        def rep_law(env, p, k):
            key = (id(p), k)
            if key not in self.memo_rep:
                self.memo_rep[key] = self._rep(env, p, k)
            pr, tokp, cov = self.memo_rep[key]
            return pr.copy(), tokp.copy(), cov

        def key_law(env, p, k):
            key = (id(p), k, env.proj)
            if key not in self.memo_key:
                self.memo_key[key] = self._key(env, p, k)
            S, pS, law, cov = self.memo_key[key]
            return S.copy(), pS.copy(), C._TokLaw(law.ids.copy(), law.vals.copy(), cov), cov

        C.topk, C._rep_law, C._key_law = topk, rep_law, key_law
        return self

    def __exit__(self, *exc):
        C.topk, C._rep_law, C._key_law = self._topk, self._rep, self._key
        return False


def gof_cell(env, method, k, p, prefix, step, draws, cell_key: bytes, rng, coder):
    fn = C.METHODS[method]
    _, tokp, _ = C._rep_law(env, p, k)          # target: the top-k token law of ordinary sampling
    counts = collections.Counter()
    t0 = time.perf_counter()
    with _FixedLawCache():
        for d in range(draws):
            key = hashlib.sha256(cell_key + struct.pack(">Q", d)).digest()
            plan, _ = fn(env, p, k, prefix, key, step)
            ac = C.make_coder(coder, 128, key)
            value = int.from_bytes(rng.bytes(16), "big")
            i = ac.encode_step(plan.weights, value)
            counts[int(plan.token(i))] += 1
    res = chi2_binned(counts, tokp, draws)
    nz = tokp[tokp > 0]
    res.update(support=int(nz.size), H_tok_bits=float(-(nz * np.log2(nz)).sum()),
               ms_per_draw=round(1000 * (time.perf_counter() - t0) / draws, 3))
    return res


def summarize(rows):
    from scipy.stats import chi2, kstest

    out = []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["method"], r["k"])].append(r["p"])
    for (method, k), ps in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] or 1 << 30)):
        ps = np.asarray(ps)
        fisher = float(chi2.sf(-2 * np.log(np.clip(ps, 1e-300, 1.0)).sum(), 2 * ps.size))
        ks = float(kstest(ps, "uniform").pvalue) if ps.size >= 2 else None
        out.append(dict(kind="summary", method=method, k=k, contexts=int(ps.size),
                        reject_05=int((ps < 0.05).sum()), fisher_p=fisher, ks_uniform_p=ks,
                        min_p=float(ps.min())))
    return out


# ---------------------------------------------------------------------------
# CPU self-test: tokenizer only, synthetic laws with heavy hubs
# ---------------------------------------------------------------------------

def stub_env(model_dir, proj):
    from transformers import AutoTokenizer

    from coin_core import CoinIndex

    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    V = int(cfg["vocab_size"])
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.clean_up_tokenization_spaces = False
    table, kind = C.token_bytes_table(tok, V)
    env = object.__new__(C.Env)
    env.tok, env.V, env.table, env.tok_kind = tok, V, table, kind
    env.index = CoinIndex(table)
    env.rep_of, env.anc, env.desc, env.max_key_len = (env.index.rep_of, env.index.anc,
                                                      env.index.desc, env.index.max_key_len)
    env.proj, env.temperature, env.top_p = proj, 1.0, 1.0
    env.klen = np.array([len(b) for b in table], dtype=np.int64)
    env.first_member = np.zeros(V, dtype=np.int64)
    env.multi = np.zeros(V, dtype=bool)
    for rep, ms in env.index.homographs.items():
        env.first_member[rep] = ms[0]
        env.multi[rep] = len(ms) > 1
    o = np.lexsort((-env.klen[env.anc], env.desc))
    env.anc_short, env.desc_short = env.anc[o], env.desc[o]
    o = np.lexsort((env.klen[env.anc], env.desc))
    env.anc_long, env.desc_long = env.anc[o], env.desc[o]
    return env


def synthetic_law(env, rng):
    child_of = collections.defaultdict(list)
    for a, d in zip(env.anc.tolist(), env.desc.tolist()):
        child_of[a].append(d)
    hubs = np.array([a for a, ch in child_of.items() if len(ch) >= 20])
    z = rng.standard_normal(env.V) * 3.0
    z[env.rep_of < 0] = -np.inf
    for h in rng.choice(hubs, size=3, replace=False):
        z[h] += 9.0
        ch = np.asarray(child_of[int(h)])
        z[ch[:40]] += 8.0 - rng.uniform(0.0, 2.0, min(40, ch.size))
    pz = np.exp(z - z[np.isfinite(z)].max())
    pz[~np.isfinite(z)] = 0.0
    return torch.from_numpy(pz / pz.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--unit", choices=("words", "chars"), default=None)
    ap.add_argument("--methods", nargs="+", default=["native", "coin", "syncpool", "basic", "mwis"])
    ap.add_argument("--k", type=int, nargs="+", default=[16, 256, 0], help="0 = Full")
    ap.add_argument("--filter-max-k", type=int, default=256, help="Basic/MWIS only up to this k")
    ap.add_argument("--n-ctx", type=int, default=20)
    ap.add_argument("--prefix-steps", type=int, nargs="+", default=[0, 8, 16, 24],
                    help="ordinary-sampling prefix lengths, cycled over contexts")
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--coder", choices=("ac", "rrc"), default="ac")
    ap.add_argument("--mwis-solver", choices=("official", "exact"), default="official",
                    help="exact: tree-DP MWIS, which also runs at the full vocabulary")
    ap.add_argument("--proj", choices=("dense", "sparse", "auto"), default="auto")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--states-out", default=None, help="save the fixed states (laws) to this .npz")
    ap.add_argument("--states-in", default=None,
                    help="load states saved by --states-out; needs the tokenizer only, no GPU")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.selftest:
        env = stub_env(args.model_dir, args.proj)
        states = [(f"synthetic-{j}", synthetic_law(env, rng), [], j) for j in range(args.n_ctx)]
    elif args.states_in:
        env = stub_env(args.model_dir, args.proj)
        z = np.load(args.states_in, allow_pickle=False)
        if int(z["V"]) != env.V:
            raise ValueError("states were saved for another vocabulary")
        states = [(str(z["prompts"][j]), torch.from_numpy(z["laws"][j].copy()),
                   z["prefix_ids"][j][: int(z["prefix_len"][j])].tolist(), int(z["steps"][j]))
                  for j in range(z["laws"].shape[0])]
    else:
        if not (args.prompts and args.unit):
            ap.error("--prompts and --unit are required unless --selftest")
        env = C.Env(args.model_dir, proj=args.proj, temperature=args.temperature, top_p=args.top_p)
        prompts, _ = C.load_prompts(args.prompts, args.unit)
        states = []
        for j in range(args.n_ctx):
            prompt = prompts[j % len(prompts)]
            st = env.stepper(env.tok(prompt)["input_ids"])
            g = torch.Generator().manual_seed(args.seed + j)
            t0 = args.prefix_steps[j % len(args.prefix_steps)]
            for _ in range(t0):                      # ordinary sampling from the same law
                st.append(int(torch.multinomial(env.law(st), 1, generator=g)))
            states.append((prompt, env.law(st), list(st.token_ids), t0))
        if args.states_out:
            L = max(len(x[2]) for x in states)
            pid = np.full((len(states), L), -1, dtype=np.int64)
            for j, x in enumerate(states):
                pid[j, : len(x[2])] = x[2]
            np.savez_compressed(args.states_out, V=env.V, laws=np.stack([x[1].numpy() for x in states]),
                                prompts=np.array([x[0] for x in states]), prefix_ids=pid,
                                prefix_len=np.array([len(x[2]) for x in states]),
                                steps=np.array([x[3] for x in states]),
                                temperature=args.temperature, top_p=args.top_p)
            print(f"[gof] saved {len(states)} states to {args.states_out}", flush=True)
            if args.draws == 0:
                return

    env.mwis_solver = args.mwis_solver
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as fh:
        meta = dict(kind="meta", tag=args.tag, model_dir=args.model_dir, V=env.V, args=vars(args))
        fh.write(json.dumps(meta) + "\n")
        for j, (prompt, p, prefix, step) in enumerate(states):
            for method in args.methods:
                for k in args.k:
                    if method in ("basic", "mwis", "stepwise") and (k == 0 or k > args.filter_max_k):
                        if not (method == "mwis" and args.mwis_solver == "exact"):
                            continue
                    if args.selftest and method == "stepwise":
                        continue
                    cell_key = f"coin-gof|{args.tag}|{j}|{method}|{k}".encode()
                    r = gof_cell(env, method, k, p, prefix, step, args.draws, cell_key, rng, args.coder)
                    r.update(kind="cell", ctx=j, prefix_steps=step, method=method, k=k, draws=args.draws)
                    rows.append(r)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    print(f"[gof] ctx {j} {method:9s} k={k or 'Full':>5} chi2={r['chi2']:.1f} "
                          f"df={r['df']} p={r['p']:.3g} out={r['outside']} {r['ms_per_draw']} ms/draw",
                          flush=True)
        for s in summarize(rows):
            fh.write(json.dumps(s) + "\n")
            print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
