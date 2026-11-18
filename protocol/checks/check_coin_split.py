"""CPU check: the standalone coin core equals the legacy module, and sparse == dense.

Tokenizers only (no model weights, no GPU).  For each tokenizer:
  1. CoinIndex: rep_of / anc / desc / homographs equal the legacy CoinIndex;
  2. coin_project (new dense) == legacy coin_project, bit for bit;
  3. coin_project_sparse == coin_project, bit for bit (ids, law, diagnostics);
  4. Env.last_live and group_items: sparse == dense;
  5. CPU time per call, dense vs sparse (indicative only, this machine).

Usage:  cd /tmp && python -I .../check_coin_split.py --protocol-dir .../protocol \
          --legacy-root <dir with the old implementation> --model-dir M1 [--model-dir M2 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--legacy-root", default=None,
                    help="optional: an older checkout holding stegatext_runtime (linear-time coin split)")
    ap.add_argument("--model-dir", action="append", required=True)
    ap.add_argument("--laws", type=int, default=40, help="random laws per (tokenizer, k)")
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()
    sys.path.insert(0, args.protocol_dir)
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    import c4proto as C
    import coin_core as K

    legacy = None
    if args.legacy_root:
        sys.path.insert(0, args.legacy_root)
        from stegatext_runtime.experiments import coin_method as legacy

    def bits_equal(x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return x.shape == y.shape and np.array_equal(x.view(np.uint64), y.view(np.uint64))

    def diag_equal(a, b):
        return a.keys() == b.keys() and all(
            (bits_equal([a[k]], [b[k]]) if isinstance(a[k], float) else a[k] == b[k]) for k in a)

    rng = np.random.default_rng(args.seed)
    report = []
    for md in args.model_dir:
        cfg = json.load(open(os.path.join(md, "config.json")))
        V = int(cfg["vocab_size"])
        tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
        tok.clean_up_tokenization_spaces = False
        table, kind = C.token_bytes_table(tok, V)
        idx = K.CoinIndex(table)
        res = dict(model=os.path.basename(md.rstrip("/")), kind=kind, V=V, pairs=int(idx.anc.size))
        if legacy is not None:
            class Codec:
                token_bytes_table = table
            old = legacy.CoinIndex(Codec())
            res["index_equal"] = bool(
                np.array_equal(old.rep_of, idx.rep_of) and np.array_equal(old.anc, idx.anc)
                and np.array_equal(old.desc, idx.desc) and old.homographs == idx.homographs
                and old.key_to_rep == idx.key_to_rep and old.max_key_len == idx.max_key_len)

        # stub Env with the static structures that last_live / group_items need
        env = object.__new__(C.Env)
        env.V, env.table, env.index = V, table, idx
        env.rep_of, env.anc, env.desc, env.max_key_len = idx.rep_of, idx.anc, idx.desc, idx.max_key_len
        env.proj = "dense"
        env.klen = np.array([len(b) for b in table], dtype=np.int64)
        o = np.lexsort((-env.klen[env.anc], env.desc))
        env.anc_short, env.desc_short = env.anc[o], env.desc[o]
        o = np.lexsort((env.klen[env.anc], env.desc))
        env.anc_long, env.desc_long = env.anc[o], env.desc[o]

        valid = np.flatnonzero(idx.rep_of >= 0)
        # hub-heavy laws: boost random anchors, their static ancestors and extensions
        child_of = {}
        for a, d in zip(idx.anc.tolist(), idx.desc.tolist()):
            child_of.setdefault(a, []).append(d)
        hubs = np.array([a for a, ch in child_of.items() if len(ch) >= 20], dtype=np.int64)

        def random_law():
            z = rng.standard_normal(V) * rng.uniform(1.0, 4.0)
            z[idx.rep_of < 0] = -np.inf
            for h in rng.choice(hubs, size=int(rng.integers(1, 6)), replace=False):
                boost = rng.uniform(3.0, 9.0)
                z[h] += boost
                ch = np.asarray(child_of[int(h)])
                z[ch] += boost - rng.uniform(0.0, 2.0, ch.size)
            p = np.exp(z - z[np.isfinite(z)].max())
            p[~np.isfinite(z)] = 0.0
            return torch.from_numpy(p / p.sum())

        ok = dict(dense_legacy=True, sparse=True, last_live=True, group_items=True)
        times = {}
        for k in (16, 64, 256, 1024, 4096, 0):
            td = ts = 0.0
            for _ in range(args.laws):
                p = random_law()
                pr, tokp, cov = C._rep_law(env, p, k)
                coins = K.CoinPRF(rng.bytes(32)).coins(int(rng.integers(0, 1 << 20)))
                u = coins(np.arange(V))                  # the same coins as an array (legacy API)
                t0 = time.perf_counter()
                a = K.coin_project(pr, idx.anc, idx.desc, coins)
                td += time.perf_counter() - t0
                t0 = time.perf_counter()
                S = np.flatnonzero(pr)
                b = K.coin_project_sparse(S, pr[S], idx, coins)
                ts += time.perf_counter() - t0
                b2 = K.coin_project_sparse(S, pr[S], idx, u)
                ok["sparse"] &= bool(np.array_equal(b[0], b2[0]) and bits_equal(b[1], b2[1]))
                same = np.array_equal(a[0], b[0]) and bits_equal(a[1], b[1]) and diag_equal(a[2], b[2])
                ok["sparse"] &= bool(same)
                if legacy is not None:
                    c = legacy.coin_project(pr, idx.anc, idx.desc, u)
                    ok["dense_legacy"] &= bool(np.array_equal(a[0], c[0]) and bits_equal(a[1], c[1])
                                               and diag_equal(a[2], c[2]))
                supp = pr > 0
                for longest in (False, True):
                    env.proj = "dense"
                    d1 = env.last_live(supp, longest)
                    env.proj = "sparse"
                    d2 = env.last_live(supp, longest)
                    ok["last_live"] &= bool(np.array_equal(d1[0], d2[0]) and np.array_equal(d1[1], d2[1]))
                ids, vals, _ = C.topk(p, k)
                ids_np, w = ids.numpy(), vals.numpy()
                env.proj = "dense"
                g1 = C.group_items(env, env.rep_of[ids_np], w)
                env.proj = "sparse"
                g2 = C.group_items(env, env.rep_of[ids_np], w)
                ok["group_items"] &= bool(np.array_equal(g1[0], g2[0]) and bits_equal(g1[1], g2[1])
                                          and np.array_equal(g1[2], g2[2]))
            times["Full" if k == 0 else str(k)] = dict(
                dense_ms=round(1000 * td / args.laws, 3), sparse_ms=round(1000 * ts / args.laws, 3))
        res.update(ok)
        res["ms_per_call"] = times
        report.append(res)
        print(json.dumps(res), flush=True)
    allok = all(all(v for kk, v in r.items() if isinstance(v, bool)) for r in report)
    print("ALL_EQUAL" if allok else "MISMATCH", flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
