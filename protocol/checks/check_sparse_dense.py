"""Bit-identity checks for the sparse code paths, the AES coins and the lean (timing) mode.

Nothing here is a result; every check either passes exactly or fails the run.
  1. coins: CoinPRF.at(step, ids) equals the same entries of a whole-vocabulary evaluation (random access),
     differs across steps and keys, and lies in [0, 1);
  2. plans, per law (hub-shaped synthetic laws on the real tokenizer, every k): the key law, Env.conflict,
     prefix_mwis and the plans of coin, syncpool, hubdrop, basic and mwis (exact) agree bit for bit between
     proj=dense and proj=sparse, and with the diagnostics on and off: the same coder weights, the same token
     for every coder index, the same parse of that token's bytes;
  3. texts (stand-in stepper of check_stream_cpu, no weights): send_stream under (dense, diagnostics),
     (sparse, diagnostics), (auto, lean) gives the same tokens and the same decodable bits for every method,
     and the lean receiver decodes them;
  4. --real (GPU): step 3 on the real model with short texts, LAS included (lean vs diagnostics).

  cd /tmp && python -I .../checks/check_sparse_dense.py --model-dir M [--model-dir M2 ...] [--laws 12] [--real]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STEPWISE_SRC = os.environ.get("STEPWISE_SRC")
if _STEPWISE_SRC:
    sys.path.insert(0, _STEPWISE_SRC)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "checks"))

import c4proto as C  # noqa: E402
import coin_core as K  # noqa: E402
import gof_single_step as G  # noqa: E402
from check_stream_cpu import FakeStepper  # noqa: E402

KS = (16, 64, 256, 1024, 4096, 0)
PLANS = ("coin", "syncpool", "hubdrop", "basic", "mwis")
TEXT_METHODS = ("native", "basic", "mwis", "stepwise", "syncpool", "coin", "hubdrop")


def same_bits(x, y) -> bool:
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    return x.shape == y.shape and np.array_equal(x.view(np.uint64), y.view(np.uint64))


class Failures:
    def __init__(self):
        self.n = 0

    def check(self, ok, what):
        if not ok:
            self.n += 1
            print(f"  FAIL {what}", flush=True)


def check_prf(env, F, rng):
    V = env.V
    key = rng.bytes(32)
    prf = K.CoinPRF(key)
    full = prf.at(7, np.arange(V))
    F.check(bool(((full >= 0) & (full < 1)).all()), "coins outside [0, 1)")
    for _ in range(20):
        ids = np.sort(rng.choice(V, size=int(rng.integers(1, 2000)), replace=False))
        F.check(same_bits(prf.at(7, ids), full[ids]), "random-access coins differ from the full evaluation")
        F.check(same_bits(K._coins_at(prf.coins(7), ids), full[ids]), "coins callable differs")
    F.check(same_bits(K.coin_prf(key).at(7, np.arange(V)), full), "cached PRF differs")
    F.check(not np.array_equal(prf.at(8, np.arange(64)), full[:64]), "coins do not change with the step")
    F.check(not np.array_equal(K.CoinPRF(rng.bytes(32)).at(7, np.arange(64)), full[:64]),
            "coins do not change with the key")
    m = float(full.mean())
    F.check(abs(m - 0.5) < 5 / np.sqrt(12 * V), f"coin mean {m:.5f} far from 1/2")


def plan_signature(env, method, p, k, key, step):
    s = time.perf_counter()
    plan, _ = C.METHODS[method](env, p, k, [], key, step)
    dt = time.perf_counter() - s
    w = plan.weights.detach().to("cpu", torch.float64).numpy().copy()
    toks = [plan.token(i) for i in range(w.size)]
    parsed = [plan.parse(env.table[t]) for t in toks]
    return (w, toks, parsed), dt


def check_plans(env, F, rng, n_laws):
    times = {}
    for k in KS:
        for li in range(n_laws):
            p = torch.softmax(torch.log(G.synthetic_law(env, rng)) / [1.0, 0.7, 1.5][li % 3], 0)
            key, step = rng.bytes(32), int(rng.integers(0, 1 << 20))
            ids, vals, _ = C._topk_raw(p, k)
            for mode in ("dense", "sparse"):
                env.proj = mode
                S, pS, law, cov = C._key_law(env, p, k)
                if mode == "dense":
                    ref = (S, pS)
                else:
                    F.check(np.array_equal(S, ref[0]) and same_bits(pS, ref[1]), f"key law k={k}")
            env.proj = "dense"
            pr, tokp, _ = C._rep_law(env, p, k)
            F.check(same_bits(pr[ref[0]], ref[1]) and float(pr.sum()) > 0, f"_rep_law k={k}")
            if 0 < k <= 4096:
                got = {}
                for mode in ("dense", "sparse"):
                    env.proj = mode
                    got[mode] = env.conflict(ids.numpy())
                F.check(got["dense"] == got["sparse"], f"conflict k={k}")
            got = {}
            for mode in ("dense", "sparse"):
                env.proj = mode
                got[mode] = C.prefix_mwis(env, ids.numpy(), vals.numpy())
            F.check(np.array_equal(got["dense"], got["sparse"]), f"prefix_mwis k={k}")
            for method in PLANS:
                if method == "basic" and k == 0:
                    continue
                sig = {}
                for mode, diag in (("dense", True), ("sparse", True), ("sparse", False), ("dense", False)):
                    env.proj, C.DIAG = mode, diag
                    sig[(mode, diag)], dt = plan_signature(env, method, p, k, key, step)
                    if not diag:
                        times.setdefault((method, k, mode), []).append(dt)
                C.DIAG = True
                w0, t0, q0 = sig[("dense", True)]
                for cfg, (w, t, q) in sig.items():
                    F.check(same_bits(w, w0) and t == t0 and q == q0, f"{method} k={k} {cfg} differs from dense")
            env.proj, env.proj_also = "auto", True        # the on-trajectory cross-check must not fire
            try:
                C.plan_coin(env, p, k, [], key, step)
            except AssertionError as e:
                F.check(False, f"proj_also k={k}: {e}")
            env.proj_also = False
    env.proj = "auto"
    for method in PLANS:
        row = []
        for k in KS:
            d, s = times.get((method, k, "dense")), times.get((method, k, "sparse"))
            if d:
                row.append(f"{'Full' if k == 0 else k}:{1e3 * np.median(d):.2f}/{1e3 * np.median(s):.2f}")
        print(f"  ms per lean plan call, dense/sparse (this CPU, indicative) {method:8s} {' '.join(row)}")


def run_text(env, method, k, prompt, stream, key, T, mode, diag, recv=False):
    env.proj, C.DIAG = mode, diag
    try:
        if method == "las":
            import las_adapter
            rec, text, _ = las_adapter.send_stream(env, k, prompt, stream, key, T)
        else:
            rec, text, _ = C.send_stream(env, method, k, prompt, stream, key, T, coder="rrc")
        got = None
        if recv and rec["complete"]:
            if method == "las":
                got = las_adapter.receive(env, k, prompt, text, key, len(stream))[0]
            elif method != "native":
                got = C.receive(env, method, k, prompt, text, key, len(stream), max_steps=T + 80,
                                coder="rrc", prefix=True)[0]
        return rec["token_ids"], rec["bits"], text, got, rec["surprisal_bits"]
    finally:
        C.DIAG = True


def check_texts(env, F, prompts, T, ks, methods, samples):
    for k in ks:
        for method in methods:
            if method in ("basic", "stepwise") and k == 0:
                continue
            for s in range(samples):
                prompt = prompts[s % len(prompts)]
                stream = C.message_bits(20261113, s, 64 * T)
                key = C.sample_key("sparse-dense-check", s)
                a = run_text(env, method, k, prompt, stream, key, T, "dense", True)
                b = run_text(env, method, k, prompt, stream, key, T, "sparse", True)
                c = run_text(env, method, k, prompt, stream, key, T, "auto", False, recv=True)
                kl = "Full" if k == 0 else str(k)
                F.check(a[:3] == b[:3], f"text {method} k={kl} s={s}: dense vs sparse")
                F.check(a[:3] == c[:3], f"text {method} k={kl} s={s}: diagnostics vs lean")
                if c[3] is not None:
                    F.check(c[3] == c[1], f"text {method} k={kl} s={s}: lean receiver")
                print(f"  {method:9s} k={kl:>4s} s={s} tokens={len(a[0]):3d} bits={len(a[1]):4d} "
                      f"surprisal={a[4]:6.1f} "
                      f"{'same' if a[:3] == b[:3] == c[:3] else 'DIFFERENT'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", action="append", required=True)
    ap.add_argument("--laws", type=int, default=12, help="synthetic laws per k")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--real", action="store_true", help="also the text check on the real model (GPU)")
    ap.add_argument("--seed", type=int, default=20261114)
    args = ap.parse_args()
    F = Failures()
    rng = np.random.default_rng(args.seed)
    prompts = {"en": ["The history of the", "In 1990, the company"], "zh": ["中国古代的", "今天我们来"],
               "ja": ["日本の伝統的な", "東京では"]}
    for md in args.model_dir:
        name = os.path.basename(md.rstrip("/"))
        lang = "zh" if "qwen" in name.lower() else "ja" if "jp" in name.lower() else "en"
        print(f"== {name} (tokenizer only)", flush=True)
        env = G.stub_env(md, "auto")
        env.mask = torch.tensor([bool(b) for b in env.table])
        env.mwis_solver, env.proj_also = "exact", False
        check_prf(env, F, rng)
        check_plans(env, F, rng, args.laws)
        base = torch.log(G.synthetic_law(env, np.random.default_rng(args.seed))) * 2.0
        env.stepper = lambda prompt_ids, base=base: FakeStepper(base, prompt_ids, 1.0)
        check_texts(env, F, prompts[lang], args.T, (16, 256, 0), TEXT_METHODS, args.samples)
        if args.real:
            print(f"== {name} (model)", flush=True)
            env = C.Env(md, proj="auto")
            env.mwis_solver, env.proj_also = "exact", False
            check_texts(env, F, prompts[lang], args.T, (16, 0), TEXT_METHODS + ("las",), args.samples)
            del env
            torch.cuda.empty_cache()
    print("ALL_IDENTICAL" if F.n == 0 else f"{F.n} FAILURES", flush=True)
    sys.exit(1 if F.n else 0)


if __name__ == "__main__":
    main()
