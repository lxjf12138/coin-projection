"""Equivalence checks for c4proto.py (run before any protocol numbers are used).

(1) FastAC == Stepwise ArithmeticCoder (indices and intervals, every step) on random pools.
(2) Byte-level Basic / MWIS / SyncPool == the Stepwise repo's builders when their string
    pieces are replaced by token bytes (same algorithms, same graph), on real LM states.
(3) Byte table: tokenizer(doc) token bytes concatenate to the document bytes.

Usage: cd /tmp && python -I check_equiv.py MODEL_DIR PROMPTS_JSONL UNIT [N_STATES]
"""
import sys

import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4proto as C  # noqa: E402  (inserts the repo paths)

import json  # noqa: E402
import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tokenization_consistency import pools as SW  # noqa: E402
from tokenization_consistency.coding import ArithmeticCoder  # noqa: E402

model_dir, prompts_path, unit = sys.argv[1], sys.argv[2], sys.argv[3]
n_states = int(sys.argv[4]) if len(sys.argv) > 4 else 24

# (1) coder ------------------------------------------------------------------
rng = random.Random(7)
gen = torch.Generator().manual_seed(7)
steps = 0
for trial in range(60):
    a, b = ArithmeticCoder(128), C.FastAC(128)
    value = rng.getrandbits(128)
    for _ in range(400):
        n = rng.choice([1, 2, 3, 17, 300, 3000])
        w = torch.rand(n, generator=gen, dtype=torch.float64) ** rng.choice([1, 2, 4])
        if n > 3 and rng.random() < 0.5:
            w[torch.randperm(n, generator=gen)[: n // 3]] = 0.0
        if float(w.sum()) == 0:
            w[0] = 1.0
        if rng.random() < 0.3:
            w = w.to(torch.bfloat16).to(torch.float64)          # ties
        ia, ib = a.encode_step(w, value), b.encode_step(w, value)
        assert ia == ib and a.low == b.low and a.high == b.high, (trial, ia, ib)
        steps += 1
        if a.resolved(value):
            break
    assert a.resolved(value) == b.resolved(value) and a.message_value == b.message_value == value
print(f"(1) FastAC identical to ArithmeticCoder on {steps} encode steps (60 messages)", flush=True)

# (2) builders ----------------------------------------------------------------
env = C.Env(model_dir)
TABLE = env.table
SW._decoded_pieces = lambda tok, ids: [TABLE[int(t)] for t in ids]   # byte pieces for the official code
prompts, _ = C.load_prompts(prompts_path, unit)
states = []
g = torch.Generator().manual_seed(11)
for pr in prompts[:4]:
    st = env.stepper(env.tok(pr)["input_ids"])
    for t in range(n_states // 4):
        p = env.law(st)
        states.append((list(st.token_ids), p))
        st.append(int(torch.multinomial(p.float(), 1, generator=g)))
key = C.sample_key("check", 0)
counts = dict(basic=0, mwis=0, syncpool=0)
for k in (16, 64, 256):
    for pre, p in states:
        ob = SW._pool_basic(env.tok, pre, p, k)
        mb, _ = C.plan_basic(env, p, k, pre, key, 0)
        assert torch.equal(ob.probs, mb.weights) and [mb.token(i) for i in range(len(ob))] == ob.token_ids.tolist()
        om = SW._pool_mwis(env.tok, pre, p, k)
        mm, _ = C.plan_mwis(env, p, k, pre, key, 0)
        assert torch.equal(om.probs, mm.weights) and [mm.token(i) for i in range(len(om))] == om.token_ids.tolist()
        os_ = SW._pool_syncpool(env.tok, pre, p, k)
        ms, _ = C.plan_syncpool(env, p, k, pre, key, 0)
        assert len(os_) == ms.weights.numel()
        # same member sets; group masses equal up to summation rounding; same order except
        # where two groups' masses agree to rounding (official sorts torch sums, we sort bincounts)
        ids, vals, _ = C.topk(p, k)
        grp, gw, roots = C.group_items(env, env.rep_of[ids.numpy()], vals.numpy())
        mine = [tuple(sorted(ids.numpy()[grp == gi].tolist())) for gi in range(gw.size)]
        theirs = [tuple(sorted(gr.tolist())) for gr in os_.groups]
        assert sorted(mine) == sorted(theirs)
        theirs_w = dict(zip(theirs, os_.probs.numpy().tolist()))
        assert all(abs(theirs_w[g] - w) <= 1e-12 * w + 1e-18 for g, w in zip(mine, gw.tolist()))
        for gi, (x, y) in enumerate(zip(mine, theirs)):
            if x != y:
                assert abs(theirs_w[x] - theirs_w[y]) <= 1e-12 * theirs_w[x], (gi, theirs_w[x], theirs_w[y])
                counts["syncpool_tie_swaps"] = counts.get("syncpool_tie_swaps", 0) + 1
        counts["basic"] += 1
        counts["mwis"] += 1
        counts["syncpool"] += 1
print(f"(2) byte-level builders match the official Basic/MWIS/SyncPool on {len(states)} states x k in (16,64,256); {counts}", flush=True)

# (2b) MWIS: official greedy vs exact tree DP on the same real states ------------
from mwis_exact import prefix_mwis  # noqa: E402

for k in (16, 64, 256, 1024, 4096, 0):
    sub, gaps, dkl, ex_full = 0, [], [], []
    for pre, p in states:
        ids, vals, _ = C.topk(p, k)
        idn, vn = ids.numpy(), vals.numpy().astype(np.float64)
        keep = prefix_mwis(env, idn, vn)
        srt = sorted(TABLE[int(i)] for i in idn[keep])
        assert all(not srt[j + 1].startswith(srt[j]) for j in range(len(srt) - 1)), "exact MWIS not prefix-free"
        we = float(vn[keep].sum())
        ex_full.append(we / float(vn.sum()))
        if k == 0:
            continue
        _, adj = env.conflict(idn)
        wg = float(vn[SW.maximum_weight_independent_set(adj, vn.tolist())].sum())
        assert wg <= we * (1 + 1e-12), (k, wg, we)
        sub += (we - wg) > 1e-12 * we
        gaps.append((we - wg) / we)
        dkl.append(np.log(we / wg))
    kl = "Full" if k == 0 else k
    if gaps:
        print(f"(2b) MWIS k={kl}: greedy below the exact optimum on {sub}/{len(states)} states; mass gap mean "
              f"{np.mean(gaps):.2e} max {np.max(gaps):.2e}; extra KL max {np.max(dkl):.2e} nats; exact keeps "
              f"{np.mean(ex_full):.4f} of the pool", flush=True)
    else:
        print(f"(2b) MWIS k={kl}: exact keeps {np.mean(ex_full):.4f} of the pool (greedy not run)", flush=True)

# (3) byte table --------------------------------------------------------------
docs = [json.loads(l)["text"] for l in open(prompts_path, encoding="utf-8")][:200]
bad = empty = 0
for d in docs:
    ids = env.tok(d, add_special_tokens=False)["input_ids"]
    got = b"".join(TABLE[i] for i in ids)
    empty += sum(1 for i in ids if not TABLE[i])
    want = d.encode("utf-8")
    if env.tok_kind == "sentencepiece" and not d.startswith("\n"):
        want = b" " + want
    bad += got != want
print(f"(3) byte table ({env.tok_kind}, V={env.V}): {len(docs) - bad}/{len(docs)} docs round-trip exactly; "
      f"empty-byte tokens used: {empty}; homograph keys: {int(env.multi.sum())}; prefix pairs: {env.anc.size}")
