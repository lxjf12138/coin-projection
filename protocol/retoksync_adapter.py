"""ReTokSync (Wang et al., arXiv 2604.25486) inside the fixed-length stream protocol, on Discop.

Kept from the paper (Sec. 3; the authors' repository holds no code):
  * the sender conditions every step on the receiver view x~ = Tok(Detok(x)) of the visible text;
  * after each token it compares x~_old || [v] with the retokenized view; on a mismatch it resets the token
    sequence, the PRNG state and the message pointer to what the receiver decodes from the new view, so the
    embedding continues from the receiver's state and an ambiguity leaves local bit errors instead of a
    desynchronization;
  * the coder is Discop (the paper's base scheme, and its main baseline) on the top-k law, i.e. Native's coder input.
Adapted (code gaps, not algorithm properties):
  * receiver view: the canonical tokenization of prompt + text, generated part; when it changes the prompt tokens,
    the text alone without special tokens (Native's receiver rule in c4proto);
  * the view is compared only when the visible bytes end on a UTF-8 boundary: a byte-level token can end inside a
    character, whose text is undefined; in between, the emitted tokens extend the view;
  * PRNG state: our Discop port draws the uniforms of step t from (key, t), so resetting the PRNG state to the
    receiver's resets t to the receiver's position;
  * a token the receiver cannot decode (outside the top-k pool, or unreachable in the Discop tree under that step's
    uniforms) yields no bits; sender and receiver apply the same rule;
  * the auxiliary correction channel is not simulated; its cost follows offline from the recorded error tokens.
Measured beyond the paper: the divergence of each step's law from ordinary sampling (the law at the emitted token
history), the reuse of a step's uniforms after a reset, and the sender's prediction of the receiver, checked
against an independent receiver that rebuilds the view from the prompt one token at a time.

Run (same prompts, keys and streams as c4proto --coder discop --stop fixed):
    python -I retoksync_adapter.py --model-dir M --tag llama31_8b --prompts P.jsonl --unit words \
        --k 256 0 --samples 50 --T 128 --out runs/e1_llama31_8b_retoksync.jsonl --skip-done
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import c4proto as C  # noqa: E402

METHOD = "retoksync"


class ViewStepper:
    """KV cache along prompt + ids.  seek() crops the cache to the common prefix and appends the rest one token at
    a time (the schedule of las_adapter.PathLaw, replay="crop"), so every law equals the one a receiver computes
    by prefilling the prompt and appending one token at a time."""

    def __init__(self, env, prompt_ids):
        self.env = env
        self.prompt = [int(t) for t in prompt_ids]
        self.st = env.stepper(self.prompt)
        self.prompt_logits = self.st._logits
        self.forwards, self.replayed = 1, 0

    def seek(self, ids):
        full = self.prompt + [int(t) for t in ids]
        st, n0 = self.st, len(self.prompt)
        cur = st.ids
        c, m = 0, min(len(cur), len(full))
        while c < m and cur[c] == full[c]:
            c += 1
        if c == len(cur):                       # extension of the cached path
            for t in full[c:]:
                st.append(t)
                self.forwards += 1
            return
        keep = max(c if c < len(full) else len(full) - 1, n0)
        drop = st._cache.get_seq_length() - keep
        if drop > 0:
            st._cache.crop(-drop)               # negative crop: same meaning in transformers 4.x and 5.x
        st.ids = list(cur[:keep])
        st._logits = self.prompt_logits if keep == n0 else None
        for t in full[keep:]:
            st.append(t)
            self.forwards += 1
            self.replayed += 1
        if st._logits is None:
            raise AssertionError("no logits after the cache crop")

    def law(self, ids):
        self.seek(ids)
        return self.env.law(self.st)


def receiver_view(env, prompt, prompt_ids, data: bytes):
    """(token ids of the generated text as the receiver tokenizes it, whether the text-alone fallback was used)."""
    text = data.decode("utf-8")
    ids = env.tok(prompt + text)["input_ids"]
    n0 = len(prompt_ids)
    if ids[:n0] == list(prompt_ids):
        return [int(t) for t in ids[n0:]], False
    return [int(t) for t in env.tok(text, add_special_tokens=False)["input_ids"]], True


def extract(env, vs, k, ids, cd, start, marks):
    """Receiver extraction of ids[start:] from the coder state at position start (cd.steps == start).
    marks[j] = number of extracted bits before position j.  Returns the number of undecodable positions."""
    if cd.steps != start or len(marks) != start:
        raise AssertionError("extraction must start from the receiver state at its position")
    bad = 0
    for j in range(start, len(ids)):
        marks.append(len(cd.bits))
        p = vs.law(ids[:j])
        pool, vals, _ = C.topk(p, k)
        hits = (pool == ids[j]).nonzero().flatten()
        n_before, ok = len(cd.bits), False
        if hits.numel():
            try:
                cd.decode_step(vals, int(hits[0]))
                ok = True
            except C.CodingError:
                pass
        if not ok:                              # no bits from this token; the step index still advances
            del cd.bits[n_before:]
            cd.steps = j + 1
            bad += 1
    return bad


def _dense(ids, vals, V):
    d = torch.zeros(V, dtype=torch.float64)
    d[ids] = vals.to(torch.float64)
    return d / d.sum()


def _kl(q_ids, q_vals, p_ids, p_vals, V):
    """KL of the top-k law at the view from the top-k law at the emitted history (inf if the supports differ)."""
    return _kl_full(_dense(q_ids, q_vals, V), _dense(p_ids, p_vals, V))


def _kl_full(q, p):
    q, p = q.numpy(), p.numpy()
    m = q > 0
    if (p[m] <= 0).any():
        return float("inf")
    return float((q[m] * np.log(q[m] / p[m])).sum())


def send_stream(env, k, prompt, stream, key, T, extra=8, measure_kl=True, max_emit=None):
    """Same contract as c4proto.send_stream with coder Discop.  The text has at least T tokens in the receiver's
    view and ends on a UTF-8 boundary; the sender gives up after max_emit emitted tokens (default T + extra + 64)."""
    cap = max_emit if max_emit else T + extra + 64
    prompt_ids = [int(t) for t in env.tok(prompt)["input_ids"]]
    t0 = time.perf_counter()
    vs = ViewStepper(env, prompt_ids)
    cd = C.DiscopCoder(len(stream), key)
    value = int(stream, 2)
    view, marks, emitted = [], [], []
    out = bytearray()
    used, reuse_steps = set(), []
    resets, undecodable, replayed_pos, fallback = [], 0, 0, 0
    h_full = h_topk = 0.0
    kl_trunc = kl_full = 0.0
    n_div, n_inf, first_div = 0, 0, None
    raw = None
    failure = None
    while len(view) < T or not C.utf8_ok(bytes(out)):
        if len(emitted) >= cap:
            failure = "emit_cap" if len(view) < T else "no_utf8_boundary"
            break
        p = vs.law(view)
        pool, vals, _ = C.topk(p, k)
        if measure_kl and emitted != view:      # the view left the emitted history: compare with ordinary sampling
            if raw is None:
                raw = ViewStepper(env, prompt_ids)
                first_div = len(emitted)
            pr = raw.law(emitted)
            rpool, rvals, _ = C.topk(pr, k)
            kt = _kl(pool, vals, rpool, rvals, p.numel())
            n_div += 1
            if math.isinf(kt):
                n_inf += 1
            else:
                kl_trunc += kt
            kl_full += _kl_full(p, pr)
        marks.append(len(cd.bits))
        if cd.steps in used:
            reuse_steps.append(cd.steps)
        used.add(cd.steps)
        try:
            i = cd.encode_step(vals, value)
        except C.CodingError as e:
            failure = f"coding_error: {e}"
            break
        v = int(pool[i])
        if C.DIAG:
            h_full += C._entropy_bits(p.numpy())
            h_topk += C._entropy_bits(vals.numpy())
        emitted.append(v)
        view.append(v)
        out += env.table[v]
        if C.utf8_ok(bytes(out)):
            new, fb = receiver_view(env, prompt, prompt_ids, bytes(out))
            if new != view:
                L = 0
                while L < min(len(new), len(view)) and new[L] == view[L]:
                    L += 1
                resets.append([len(emitted) - 1, L, len(view), len(new)])
                fallback += bool(fb)
                del cd.bits[marks[L]:]
                del marks[L:]
                cd.steps = L
                undecodable += extract(env, vs, k, new, cd, L, marks)
                replayed_pos += len(new) - L
                view = new
    n_emit = len(emitted)
    bits = "".join(cd.bits)
    ref = stream[: len(bits)]
    wrong = [i for i in range(len(bits)) if bits[i] != ref[i]]
    marks_end = marks + [len(bits)]
    err_tokens = []                             # [view position, bits it carries, wrong bits among them]
    wi = 0
    for j in range(len(view)):
        a, b = marks_end[j], marks_end[j + 1]
        nw = 0
        while wi < len(wrong) and wrong[wi] < b:
            nw += wrong[wi] >= a
            wi += 1
        if nw:
            err_tokens.append([j, b - a, nw])
    rec = dict(complete=failure is None, failure=failure, n_tokens=len(view), n_emit=n_emit, T=T,
               utf8_valid=C.utf8_ok(bytes(out)), t_send=time.perf_counter() - t0,
               forwards=vs.forwards + (raw.forwards if raw else 0), replayed=vs.replayed,
               n_resets=len(resets), resets=resets, replayed_positions=replayed_pos, undecodable=undecodable,
               view_fallback=fallback, n_reuse=len(reuse_steps), reuse_steps=reuse_steps,
               n_err_bits=len(wrong), first_error=wrong[0] if wrong else None, err_tokens=err_tokens,
               H_full=h_full / max(n_emit, 1), H_topk=h_topk / max(n_emit, 1), H_full_sum=h_full,
               kl_trunc_sum=kl_trunc, kl_full_sum=kl_full, kl_div_steps=n_div, kl_inf_steps=n_inf,
               first_div=first_div, view_bytes_match=b"".join(env.table[t] for t in view) == bytes(out),
               token_ids=view, emitted_ids=emitted, marks=marks)
    rec["bits"], rec["n_bits"] = bits, len(bits)
    return rec, bytes(out).decode("utf-8", errors="replace"), prompt_ids


def receive(env, k, prompt, text, key, bit_length):
    """The receiver: canonical view of the text, Discop extraction position by position from the prompt prefill."""
    prompt_ids = [int(t) for t in env.tok(prompt)["input_ids"]]
    ids, fb = receiver_view(env, prompt, prompt_ids, text.encode("utf-8"))
    vs = ViewStepper(env, prompt_ids)
    cd = C.DiscopCoder(bit_length, key)
    marks = []
    bad = extract(env, vs, k, ids, cd, 0, marks)
    return "".join(cd.bits), ids, marks, bad, fb


def run_one(env, args, fh, k, s, prompt, stream, key):
    rec, text, _ = send_stream(env, k, prompt, stream, key, args.T, measure_kl=not args.no_kl, max_emit=args.max_emit)
    bits = rec.pop("bits")
    ok = rec["complete"]
    rec.update(kind="case", stop="fixed", tag=args.tag, method=METHOD, k=k, sample=s, coder="discop",
               prompt=prompt, text=text, stream_len=len(stream), temperature=args.temperature, top_p=args.top_p,
               prompt_units=args.prompt_units, **C.run_info())
    nb = rec["n_bits"]
    rec["bpt"] = nb / rec["n_tokens"] if ok else None
    rec["util"] = nb / rec["H_full_sum"] if ok and rec["H_full_sum"] else None
    rec["ber"] = rec["n_err_bits"] / nb if ok and nb else None
    rec["text_error"] = bool(rec["n_err_bits"]) if ok else None
    rec["recv_checked"] = bool(ok and not args.no_recv)
    rec["recv_match"] = None
    if rec["recv_checked"]:
        t0 = time.perf_counter()
        got, ids, marks, bad, fb = receive(env, k, prompt, text, key, len(stream))
        rec["t_recv"] = time.perf_counter() - t0
        # the scheme rests on the sender predicting the receiver's extraction exactly
        rec["recv_match"] = got == bits and ids == rec["token_ids"] and marks == rec["marks"]
        rec["recv_undecodable"] = bad
        rec["recv_fallback"] = fb
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--unit", choices=("words", "chars"), required=True)
    ap.add_argument("--prompt-units", type=int, default=10)
    ap.add_argument("--k", type=int, nargs="+", default=[256], help="0 = Full")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--stream-factor", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--no-recv", action="store_true")
    ap.add_argument("--no-kl", action="store_true")
    ap.add_argument("--max-emit", type=int, default=None, help="cap on emitted tokens (default T + 72)")
    args = ap.parse_args()

    done = set()
    if args.skip_done and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("kind") == "case":
                    done.add((r["tag"], r["k"], r["sample"]))
    torch.manual_seed(0)
    env = C.Env(args.model_dir, temperature=args.temperature, top_p=args.top_p)
    prompts, dropped = C.load_prompts(args.prompts, args.unit, args.prompt_units)
    if args.start + args.samples > len(prompts):
        raise SystemExit(f"only {len(prompts)} prompts of {args.prompt_units} {args.unit}")
    meta = dict(kind="meta", tag=args.tag, method=METHOD, model_dir=args.model_dir, tok_kind=env.tok_kind, V=env.V,
                prompts=len(prompts), prompts_dropped=dropped, args=vars(args))
    print(json.dumps(meta), flush=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(meta) + "\n")
        for k in args.k:
            agg = dict(n=0, complete=0, match=0, checked=0, text_err=0, bits=0, err=0, resets=0, reuse=0, tok=0)
            for s in range(args.start, args.start + args.samples):
                if (args.tag, k, s) in done:
                    continue
                prompt = prompts[s % len(prompts)]
                key = C.sample_key(args.tag, s)
                stream = C.message_bits(args.seed, s, args.stream_factor * args.T)
                r = run_one(env, args, fh, k, s, prompt, stream, key)
                agg["n"] += 1
                agg["complete"] += bool(r["complete"])
                agg["checked"] += bool(r["recv_checked"])
                agg["match"] += bool(r["recv_match"])
                agg["text_err"] += bool(r["text_error"])
                agg["bits"] += r["n_bits"]
                agg["err"] += r["n_err_bits"]
                agg["resets"] += r["n_resets"]
                agg["reuse"] += r["n_reuse"]
                agg["tok"] += r["n_tokens"]
            print(json.dumps(dict(tag=args.tag, method=METHOD, k=k, **agg)), flush=True)


if __name__ == "__main__":
    main()
