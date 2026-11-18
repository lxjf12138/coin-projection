"""CPU structural test of the fixed-length (stream) protocol: every method, sender -> text -> receiver.

No model weights: the real tokenizer and byte index, with a stand-in stepper whose next-token logits are a
fixed hub-shaped law perturbed by a hash of the last three tokens, so sender and receiver see identical laws
for identical histories.  Checks, per (method, k, coder, sample):
  * the sender's decodable prefix agrees with the stream (asserted inside send_stream);
  * the method's own receiver returns exactly that prefix (prefix-parse methods and Stepwise must; Native may
    legitimately fail on the stand-in's often non-canonical text and is only reported);
  * embedded bits stay close to the realized coder-input surprisal (the conservative hull loses O(1) steps).

  python -I checks/check_stream_cpu.py --model-dir .../Llama-3.2-3B --prompt "The history of the" \
      [--T 24] [--samples 2] [--k 16 256 0] [--coders rrc ac]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STEPWISE_SRC = os.environ.get("STEPWISE_SRC")
if _STEPWISE_SRC:
    sys.path.insert(0, _STEPWISE_SRC)
sys.path.insert(0, HERE)

import c4proto as C  # noqa: E402
import gof_single_step as G  # noqa: E402

MUST = {"basic", "mwis", "syncpool", "coin", "hubdrop", "stepwise"}


class FakeStepper:
    def __init__(self, base, prompt_ids, noise):
        self.base, self.noise = base, noise
        self.ids = [int(t) for t in prompt_ids]
        self.prompt_length = len(self.ids)
        self._update()

    def _update(self):
        tail = self.ids[-3:]
        h = hashlib.sha256(struct.pack(f">{len(tail)}q", *tail)).digest()
        g = torch.Generator().manual_seed(int.from_bytes(h[:8], "big") >> 1)
        self._logits = self.base + self.noise * torch.randn(self.base.numel(), generator=g, dtype=torch.float64)

    @property
    def token_ids(self):
        return list(self.ids)

    @property
    def generated_ids(self):
        return self.ids[self.prompt_length:]

    def append(self, t):
        self.ids.append(int(t))
        self._update()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--k", type=int, nargs="+", default=[16, 256, 0])
    ap.add_argument("--coders", nargs="+", default=["rrc", "ac"])
    ap.add_argument("--methods", nargs="+", default=["native", "basic", "mwis", "stepwise", "syncpool", "coin", "hubdrop"])
    ap.add_argument("--mwis-solver", default="official")
    ap.add_argument("--sharpen", type=float, default=2.0, help="multiplies the base log-law (lower entropy)")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()

    env = G.stub_env(args.model_dir, "auto")
    env.mask = torch.tensor([bool(b) for b in env.table])
    env.mwis_solver = args.mwis_solver
    base = torch.log(G.synthetic_law(env, np.random.default_rng(args.seed))) * args.sharpen
    env.stepper = lambda prompt_ids: FakeStepper(base, prompt_ids, args.noise)

    bad = 0
    print(f"{'coder':5s} {'method':9s} {'k':>5s} {'s':>2s} {'tok':>4s} {'bits':>5s} {'surpr':>7s} {'loss':>5s} "
          f"{'recv':>5s} {'ms/tok':>7s}  status")
    for coder in args.coders:
        for k in args.k:
            for method in args.methods:
                if method in ("basic", "stepwise") and k == 0:
                    continue
                if method == "mwis" and k == 0 and args.mwis_solver != "exact":
                    continue
                for s in range(args.samples):
                    stream = C.message_bits(args.seed, s, 64 * args.T)
                    key = C.sample_key("cpu-check", s)
                    t0 = time.perf_counter()
                    rec, text, _ = C.send_stream(env, method, k, args.prompt, stream, key, args.T, coder=coder)
                    ms = 1000 * (time.perf_counter() - t0) / max(rec["n_tokens"], 1)
                    status, got = "", None
                    if not rec["complete"]:
                        status = f"INCOMPLETE {rec['failure']}"
                    else:
                        try:
                            got, _ = C.receive(env, method, k, args.prompt, text, key, len(stream),
                                               max_steps=args.T + 80, coder=coder, prefix=True)
                            status = "ok" if got == rec["bits"] else (
                                "PREFIX-OK-SHORT" if stream.startswith(got) else "WRONG")
                        except C.RecvError as e:
                            status = f"RECV-ERR {str(e)[:50]}"
                    must = method in MUST
                    if must and status != "ok":
                        bad += 1
                        status += "  <-- FAIL"
                    if rec["stream_exhausted"]:
                        bad += 1
                        status += "  <-- STREAM EXHAUSTED"
                    loss = rec["surprisal_bits"] - rec["n_bits"]
                    if rec["complete"] and loss > 64:
                        bad += 1
                        status += "  <-- LARGE HULL LOSS"
                    kl = "Full" if k == 0 else str(k)
                    print(f"{coder:5s} {method:9s} {kl:>5s} {s:2d} {rec['n_tokens']:4d} {rec['n_bits']:5d} "
                          f"{rec['surprisal_bits']:7.1f} {loss:5.1f} {len(got) if got is not None else -1:5d} "
                          f"{ms:7.1f}  {status}", flush=True)
    print("ALL PASS" if bad == 0 else f"{bad} FAILURES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
