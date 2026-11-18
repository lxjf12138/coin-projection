"""LAS (Feng et al., arXiv 2510.02332; policy `las`, i.e. tau = 0) inside the fixed-length stream protocol.

Kept from the public release (imported read-only from $LAS_RELEASE; see README, 'External baselines'):
  * the selective block-frontier planner (make_selective_planner): group keys, plan / observe / advance and its
    posterior-mass invariant;
  * the advance randomness CounterRNG(key, "advance", step); the tau = 0 gate always selects the LAS kernel.
Adapted (code gaps, not algorithm properties):
  * law: the next-token law every other method sees -- our model forward, a float64 softmax over the tokens
    with visible bytes (EOS and special tokens get zero mass), temperature / nucleus, then top-k renormalized.
    The release's own k-sweep harness uses the same float64 masked softmax followed by TopK(k, 1.0);
  * byte table: the release codec reads byte-level BPE pieces through the GPT-2 byte map and falls back to
    decode() otherwise, which drops SentencePiece's leading-space marker and its byte-fallback tokens; we install
    our byte table (on byte-level BPE the two tables agree on every visible token, counted at load);
  * coder: our RRCCoder with the same per-text key, so every method shares the rotation randomness and the
    decodable-prefix rule (the release pairs LAS with DisCop);
  * model calls: one KV cache along the active path; when the planner moves to another branch, the cache is
    cropped to the common prefix and the remainder is replayed one token at a time.  That reproduces the
    release's prompt-prefill-then-single-token schedule, and the self-test checks it bit for bit by letting
    the receiver rebuild every branch from the prompt instead.
Length rule: decide until the resolved path holds at least T tokens and its bytes end on a UTF-8 boundary (fail
past T + extra tokens) -- the rule every other method follows, where one decision is one token.  A LAS decision
resolves one group key, often shorter than a token, so counting decisions instead would give LAS shorter texts.

Self-test (GPU, small):
    cd /tmp && python -I .../las_adapter.py --model-dir .../Llama-3.2-3B --prompt "The history of the" \
        --T 16 --samples 2 --k 16 0
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.dont_write_bytecode = True        # the release tree stays read-only

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c4proto as C  # noqa: E402

LAS_RELEASE = os.environ.get("LAS_RELEASE", "")
# A LAS decision resolves one group key, which can be shorter than a token (at Full every byte is a key), so the
# length rule counts tokens on the resolved path; decisions per token stay below the longest token's byte length.
DECISION_WATCHDOG = 64


def _import_release():
    """Put the public release first and drop any other `stegtext` checkout from the path."""
    if not LAS_RELEASE:
        raise SystemExit(
            "the LAS baseline needs the public LAS release: set LAS_RELEASE to a clone of "
            "github.com/lxjf12138/stegtext-framework (see README, 'External baselines')."
        )
    rel = os.path.realpath(LAS_RELEASE)
    keep = []
    for p in sys.path:
        rp = os.path.realpath(p) if p else p
        other = (rp.endswith(os.sep + os.path.join("package", "src"))
                 or rp.endswith(os.sep + os.path.join("package", "research_runtime")))
        if other and not rp.startswith(rel + os.sep):
            continue
        keep.append(p)
    sys.path[:] = keep
    for sub in ("package/src", "package/research_runtime"):
        p = os.path.join(rel, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import perf_research_selective
    import stegtext.v2
    for m in (stegtext.v2, perf_research_selective):
        if not os.path.realpath(m.__file__).startswith(rel + os.sep):
            raise RuntimeError(f"{m.__name__} resolved outside the LAS release: {m.__file__}")


_import_release()          # before any release import below


class LASContext:
    """Per-model release objects: codec (with our byte table) and the token byte index."""

    def __init__(self, env):
        from stegtext.v2.model.hf_backend import _eos_ids
        from stegtext.v2.model.tokenizer_codec import EOS_STEGA, TokenizerCodec
        from stegtext.v2.model.vocab_index import TokenByteIndex

        eos = _eos_ids(env.tok, env.model)
        codec = TokenizerCodec(env.tok, eos)
        n = codec.vocab_size
        if n > env.V:
            raise RuntimeError(f"codec vocabulary {n} exceeds the model head {env.V}")
        mask = env.mask.cpu().numpy()
        if any(mask[i] for i in codec.eos_ids if i < env.V):
            raise RuntimeError("an EOS token has visible bytes in our table; LAS would treat it as terminal")
        ours = [TokenizerCodec._escape_bytes(env.table[i]) for i in range(n)]
        for i in codec.eos_ids:
            ours[i] = EOS_STEGA
        native = codec.token_bytes_table
        # visible tokens whose release-codec bytes differ from ours (0 on byte-level BPE)
        self.table_diff = int(sum(1 for i in range(n) if env.table[i] and native[i] != ours[i]))
        codec._token_bytes = tuple(ours)
        self.codec, self.n = codec, n
        self.index = TokenByteIndex(codec.token_bytes_table, device="cpu")
        self.eos = eos

    def prompt_state(self, prompt_ids, prompt):
        from stegtext.v2.core.types import PathRef, PromptState
        path = PathRef.root()
        for t in prompt_ids:
            path = path.append(int(t))
        ids = tuple(int(t) for t in prompt_ids)
        return PromptState(path=path, token_ids=ids, prompt_bytes=self.codec.tokens_to_bytes(ids),
                           prompt_text=str(prompt))


def las_context(env) -> LASContext:
    ctx = getattr(env, "_las", None)
    if ctx is None:
        ctx = env._las = LASContext(env)
    return ctx


class PathLaw:
    """Top-k law at any token path; one KV cache follows the planner's active path.

    replay="crop": on a branch switch, crop the cache to the common prefix and append the rest one token at a
    time.  replay="fresh": rebuild from the prompt prefill and append every generated token one at a time (the
    release's schedule).  Both give bit-identical logits; the self-test checks this."""

    def __init__(self, env, las, prompt_ids, k, replay="crop"):
        if replay not in ("crop", "fresh"):
            raise ValueError(replay)
        self.env, self.n, self.k, self.replay = env, las.n, k, replay
        self.prompt = [int(t) for t in prompt_ids]
        s = time.perf_counter()
        self.st = env.stepper(self.prompt)
        C._sync()
        self.t_prefill = time.perf_counter() - s
        self.prompt_logits = self.st._logits
        self.forwards = 1
        self.switches = self.replayed = 0
        self.t_fwd = self.t_law = 0.0
        self.info = {}                    # path tuple -> (top-k mass, H_full bits, H_topk bits); entropies: DIAG only

    def _seek(self, ids):
        st, n0 = self.st, len(self.prompt)
        cur = st.ids
        c, m = 0, min(len(cur), len(ids))
        while c < m and cur[c] == ids[c]:
            c += 1
        if c < n0:
            raise ValueError("path does not extend the prompt")
        if c == len(cur):                 # extension of the active path
            for t in ids[c:]:
                st.append(t)
                self.forwards += 1
            C._sync()
            return
        self.switches += 1
        keep = c if c < len(ids) else len(ids) - 1
        keep = max(keep, n0)
        if self.replay == "fresh":
            self.st = st = self.env.stepper(self.prompt)
            self.forwards += 1
            keep = n0
        else:
            # Drop the cached positions beyond `keep`.  A negative crop removes that many trailing tokens in both
            # transformers 4.x and 5.x (5.x rejects the 4.x positive "keep this many" form; 4.x crop(0) empties).
            drop = st._cache.get_seq_length() - keep
            if drop > 0:
                st._cache.crop(-drop)
            st.ids = list(cur[:keep])
            st._logits = self.prompt_logits if keep == n0 else None
        for t in ids[keep:]:
            st.append(t)
            self.forwards += 1
            self.replayed += 1
        C._sync()
        if st._logits is None:
            raise AssertionError("no logits after the branch switch")

    def __call__(self, ids):
        ids = [int(t) for t in ids]
        s = time.perf_counter()
        self._seek(ids)
        self.t_fwd += time.perf_counter() - s
        s = time.perf_counter()
        p = self.env.law(self.st)
        if p[self.n:].sum() > 0:
            raise RuntimeError("positive mass beyond the codec vocabulary")
        tid, vals, _ = C.topk(p, self.k)
        cov = float(vals.sum())
        q = torch.zeros(self.n, dtype=torch.float64)       # the release planner takes a dense law
        q[tid] = vals.to(torch.float64) / cov
        self.t_law += time.perf_counter() - s
        if C.DIAG:                                          # statistics, outside the timer
            self.info[tuple(ids)] = (cov, C._entropy_bits(p.numpy()), C._entropy_bits(vals.numpy()))
        else:
            self.info[tuple(ids)] = (cov, None, None)
        return q


def _raw(resolved: bytes) -> bytes:
    return resolved[1::2]


def _run(env, k, prompt, key, T, extra, stream=None, text=None, replay="crop", max_steps=None, steps=None):
    from perf_research_selective import make_selective_planner
    from round4_2_common import CounterRNG
    from stegtext.v2.core.types import MethodContext, TokenDistribution, VisibleTarget

    las = las_context(env)
    sending = stream is not None
    if sending:
        C.TIMERS.clear()
    c0 = time.process_time()
    t0 = time.perf_counter()
    prompt_ids = env.tok(prompt)["input_ids"]
    ps = las.prompt_state(prompt_ids, prompt)
    planner = make_selective_planner(las.index)
    state = planner.start(ps)
    target = None if sending else VisibleTarget(las.codec.text_to_bytes(text))
    ctx = MethodContext(las.codec, ps, "encode" if sending else "decode", target=target)
    law = PathLaw(env, las, prompt_ids, k, replay)
    cd = C.RRCCoder(len(stream) if sending else max_steps[1], key)
    value = int(stream, 2) if sending else None
    surprisal = h_coder = groups = 0.0
    t_plan = t_code = 0.0
    per = {"sur": [], "pool": [], "H_coder": [], "ms_plan": [], "ms_code": []} if steps is not None else None
    failure = None
    step = 0
    while True:
        if sending:
            ntok = state.base_path.depth - ps.path.depth
            if ntok >= T and C.utf8_ok(_raw(state.resolved_bytes)):
                break
            if ntok >= T + extra:
                failure = "no_utf8_boundary"
                break
            if step >= DECISION_WATCHDOG * (T + extra):
                failure = "decision_watchdog"
                break
        else:
            if target.done:
                break
            if step >= max_steps[0]:
                raise C.RecvError("receiver exceeded the step limit")
        if state.terminal:
            raise AssertionError("terminal LAS state although EOS has zero mass")
        dists = [TokenDistribution(law(state.base_path.token_ids()))] if state.expansion_mass > 0 else []
        s = time.perf_counter()
        plan = planner.plan(state, dists, ctx)
        d_plan = time.perf_counter() - s
        s = time.perf_counter()
        try:
            if sending:
                i = cd.encode_step(plan.probs, value)
            else:
                i = planner.observe(state, plan, target, ctx)
                cd.decode_step(plan.probs, i)
        except C.CodingError as e:
            if not sending:
                raise C.RecvError(str(e))
            failure = f"coding_error: {e}"
            break
        d_code = time.perf_counter() - s
        w = plan.probs.detach().to("cpu", torch.float64).numpy()
        sur = -math.log2(float(w[i]) / float(w.sum()))
        surprisal += sur
        hc = C._entropy_bits(w) if C.DIAG else 0.0
        h_coder += hc
        groups += w.size
        s = time.perf_counter()
        state = planner.advance(state, plan, i, CounterRNG(key, "advance", step + 1), ctx)
        d_plan += time.perf_counter() - s
        t_plan, t_code = t_plan + d_plan, t_code + d_code
        if per is not None:
            per["sur"].append(C._g(sur))
            per["pool"].append(int(w.size))
            per["H_coder"].append(C._g(hc))
            per["ms_plan"].append(C._g(1000 * d_plan))
            per["ms_code"].append(C._g(1000 * d_code))
        mass = state.expansion_mass + math.fsum(state.retained.masses)
        if not abs(mass - 1) < 2e-6:
            raise AssertionError(f"LAS posterior mass {mass}")
        step += 1
    path = state.base_path.token_ids()
    n0 = len(prompt_ids)
    gen = [int(t) for t in path[n0:]]
    if las.codec.tokens_to_bytes(gen) != state.resolved_bytes:
        raise AssertionError("final LAS path does not realize the resolved bytes")
    covs, hs, hk = [], [], []          # along the final path, one entry per token
    for j in range(n0, len(path)):
        cov, h, h_k = law.info[tuple(path[:j])]
        covs.append(cov)
        hs.append(h)
        hk.append(h_k)
    n = max(len(gen), 1)
    diag = C.DIAG and bool(hs)
    rec = dict(complete=failure is None, failure=failure, n_tokens=len(gen), n_decisions=step, T=T,
               utf8_valid=C.utf8_ok(_raw(state.resolved_bytes)), surprisal_bits=surprisal,
               free_bits=C.coder_width_bits(cd),
               t_send=time.perf_counter() - t0, cpu_send=time.process_time() - c0,
               ms_prefill=1000 * law.t_prefill, ms_plan=1000 * t_plan / n, ms_fwd=1000 * law.t_fwd / n,
               ms_law=1000 * law.t_law / n, ms_code=1000 * t_code / n, token_ids=gen,
               H_full=float(np.mean(hs)) if diag else None,
               H_topk=float(np.mean(hk)) if diag else None,
               retained=float(np.mean(covs)) if covs else None,
               kld_stepwise=float(np.mean([-math.log(max(c, 1e-300)) for c in covs])) if covs else None,
               kl_sec=0.0, H_coder=h_coder / max(step, 1), pool=groups / max(step, 1),
               forwards=law.forwards, branch_switches=law.switches, replayed_tokens=law.replayed,
               retained_final=len(state.retained), las_table_diff=las.table_diff,
               laws_hash=_laws_hash(law))
    if sending:
        rec.update(C.timer_fields(n))
    rec["stream_exhausted"] = rec["free_bits"] < 16
    if per is not None:
        per["tok_retained"] = [C._g(c) for c in covs]
        if diag:
            per["tok_H_full"] = [C._g(h) for h in hs]
            per["tok_H_topk"] = [C._g(h) for h in hk]
        steps.update(per)
    out = _raw(state.resolved_bytes)
    return rec, out, cd, prompt_ids


def _laws_hash(law):
    import hashlib
    h = hashlib.sha256()
    for path in sorted(law.info):
        h.update(repr((path, law.info[path])).encode())
    return h.hexdigest()[:16]


def send_stream(env, k, prompt, stream, key, T, extra=8, replay="crop", steps=None):
    """Same contract as c4proto.send_stream (coder: RRC).  Per-step arrays: one entry per decision, and
    tok_* entries per token of the final path."""
    rec, out, cd, prompt_ids = _run(env, k, prompt, key, T, extra, stream=stream, replay=replay, steps=steps)
    bits = ""
    if rec["complete"]:
        bits = C.decodable_prefix(cd)
        if not stream.startswith(bits):
            raise AssertionError("sender's decodable prefix disagrees with the stream")
    rec["bits"], rec["n_bits"] = bits, len(bits)
    return rec, out.decode("utf-8", errors="replace"), prompt_ids


def receive(env, k, prompt, text, key, bit_length, max_steps=None, replay="crop"):
    """Stream bits fixed by the text: the release receiver (observe) driving the shared RRC decoder.
    max_steps bounds decisions (default: the sender's watchdog for a text of this many bytes)."""
    if max_steps is None:
        max_steps = DECISION_WATCHDOG * (len(text.encode("utf-8")) + 1)
    rec, _, cd, _ = _run(env, k, prompt, key, None, None, text=text, replay=replay,
                         max_steps=(max_steps, bit_length))
    try:
        return C.decodable_prefix(cd), rec["n_decisions"], rec
    except C.CodingError as e:
        raise C.RecvError(str(e))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--k", type=int, nargs="+", default=[16, 0])
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()
    env = C.Env(args.model_dir, proj="auto")
    las = las_context(env)
    print(f"codec vocab {las.n}  head {env.V}  eos {las.eos}  release-vs-our byte table differences on "
          f"visible tokens: {las.table_diff}", flush=True)
    bad = 0
    for k in args.k:
        for s in range(args.samples):
            prompt = args.prompt[s % len(args.prompt)]
            stream = C.message_bits(args.seed, s, 64 * args.T)
            key = C.sample_key("las-selftest", s)
            rec, text, _ = send_stream(env, k, prompt, stream, key, args.T)
            status = "INCOMPLETE " + str(rec["failure"]) if not rec["complete"] else ""
            if rec["complete"]:
                got, nd, rr = receive(env, k, prompt, text, key, len(stream), replay="fresh")
                same_laws = rr["laws_hash"] == rec["laws_hash"]
                status = ("ok" if got == rec["bits"] and nd == rec["n_decisions"] else "WRONG") + \
                         ("" if same_laws else " LAW-MISMATCH")
            if status != "ok":
                bad += 1
            kl = "Full" if k == 0 else str(k)
            print(f"k={kl:>4s} s={s} dec={rec['n_decisions']:3d} tok={rec['n_tokens']:3d} bits={rec['n_bits']:4d} "
                  f"surpr={rec['surprisal_bits']:7.1f} bpt={rec['n_bits'] / max(rec['n_tokens'], 1):.2f} "
                  f"H={rec['H_full']:.2f} groups={rec['pool']:.0f} switches={rec['branch_switches']} "
                  f"replayed={rec['replayed_tokens']} fwd={rec['forwards']} "
                  f"ms/tok plan={rec['ms_plan']:.1f} fwd={rec['ms_fwd']:.1f}  {status}  {text[:60]!r}", flush=True)
    print("ALL PASS" if bad == 0 else f"{bad} FAILURES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
