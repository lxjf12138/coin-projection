"""E0 survey (forward only, no coder) on the C4 protocol.

For every k: ordinary top-k sampling (T = 1) from each prompt for T tokens.  Per state:
  key-level decomposition H_key = H_root + H_branch + H_stop, and H_tok = H_key + H_hom;
  largest-root mass; per-key contributions (roots by mass, nodes by H_branch / H_stop);
  M-weighted histogram of stop probabilities q_v = p(v)/M(v);
  non-exact shortcuts: hub-drop at several thresholds and Basic (drop all nodes), as
  remaining root budget and per-step KL (nats) against the same-k law.
Per trajectory: non-canonical tokens under retokenization of prompt + text (Stepwise's analyze).

Usage: cd /tmp && python -I e0_atlas.py --model-dir D --tag T --prompts F --unit U --k 16 0 --n-prompts 50 --T 64 --out O.json
"""
import sys

import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4proto as C  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tokenization_consistency.tokenization import analyze  # noqa: E402

THRESHOLDS = (0.01, 0.05, 0.1, 0.2)
CATS = ("bare_space", "lead_byte", "other_byte", "partial_char", "space_word", "word", "cjk", "other")


def key_categories(table):
    """Category of every vocabulary string, for attributing H_root / H_branch / H_stop (App. origin)."""
    cat = np.full(len(table), CATS.index("other"), dtype=np.int64)
    for i, b in enumerate(table):
        if not b:
            continue
        if b == b" ":
            c = "bare_space"
        elif len(b) == 1:
            c = "lead_byte" if 0xC2 <= b[0] <= 0xF4 else "other_byte"
        else:
            try:
                s = b.decode("utf-8")
            except UnicodeDecodeError:
                c = "partial_char"
            else:
                core = s.lstrip(" ")
                if core and all(0x3040 <= ord(ch) <= 0x30FF or 0x3400 <= ord(ch) <= 0x9FFF for ch in core):
                    c = "cjk"
                elif core.isalpha() and core.isascii():
                    c = "space_word" if s.startswith(" ") else "word"
                else:
                    c = "other"
        cat[i] = CATS.index(c)
    return cat


def ent(x):
    x = x[x > 0]
    return float(-(x * np.log2(x)).sum())


def root_budget(env, p):
    s = p > 0
    live = s[env.anc] & s[env.desc]
    m = p + np.bincount(env.anc[live], weights=p[env.desc[live]], minlength=env.V)
    has = np.zeros(env.V, dtype=bool)
    has[env.desc[live]] = True
    return ent(m[np.flatnonzero(s & ~has)])


def decompose(env, pr):
    supp = pr > 0
    live = supp[env.anc] & supp[env.desc]
    a, d = env.anc[live], env.desc[live]
    ext = np.bincount(a, weights=pr[d], minlength=env.V)   # M(v) - p(v), summed directly (no cancellation)
    mass = pr + ext
    child, par = env.last_live(supp, longest=True)          # parent = longest live proper prefix
    has_par = np.zeros(env.V, dtype=bool)
    has_par[child] = True
    roots = np.flatnonzero(supp & ~has_par)
    nodes = np.unique(par)
    q = pr[nodes] / mass[nodes]
    with np.errstate(divide="ignore", invalid="ignore"):
        h2 = -(q * np.log2(q) + (1 - q) * np.log2(np.where(q < 1, 1 - q, 1.0)))
        stop_v = mass[nodes] * np.nan_to_num(h2)
        cont = ext[par]                                      # = M(par) - p(par) without cancellation
        ratio = np.where((cont > 0) & (mass[child] > 0), np.minimum(mass[child] / np.where(cont > 0, cont, 1.0), 1.0), 1.0)
        branch_terms = np.where((cont > 0) & (mass[child] > 0), -mass[child] * np.log2(ratio), 0.0)
    out = dict(H_key=ent(pr), H_root=ent(mass[roots]), H_branch=float(branch_terms.sum()),
               H_stop=float(stop_v.sum()), maxroot=float(mass[roots].max()))
    assert abs(out["H_key"] - out["H_root"] - out["H_branch"] - out["H_stop"]) < 1e-6, out
    return out, nodes, q, mass, stop_v, np.bincount(par, weights=branch_terms, minlength=env.V), roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--prompt-units", type=int, default=10, help="10: the Stepwise rule; other sizes are strict")
    ap.add_argument("--k", type=int, nargs="+", default=[16, 64, 256, 1024, 4096, 0])
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--proj", choices=("dense", "sparse", "auto"), default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    env = C.Env(args.model_dir, proj=args.proj, temperature=args.temperature, top_p=args.top_p)
    prompts, dropped = C.load_prompts(args.prompts, args.unit, args.prompt_units)
    if len(prompts) < args.n_prompts:
        raise SystemExit(f"only {len(prompts)} prompts of {args.prompt_units} {args.unit}")
    prompts = prompts[: args.n_prompts]

    def show(i):
        return env.table[int(i)].decode("utf-8", "backslashreplace")

    report = dict(tag=args.tag, tok_kind=env.tok_kind, V=env.V, prompts=len(prompts), dropped=dropped,
                  prompt_units=args.prompt_units, machine=C.machine_meta(args.model_dir), T=args.T, temperature=args.temperature, top_p=args.top_p, proj=args.proj, cells=[])
    kcat = key_categories(env.table)

    def by_cat(arr):
        tot = float(arr.sum())
        return {c: round(float(arr[kcat == j].sum()) / tot, 4) if tot > 0 else 0.0 for j, c in enumerate(CATS)}
    for k in args.k:
        g = torch.Generator().manual_seed(args.seed + k)
        tot = dict(n=0, H_tok=0.0, H_key=0.0, H_root=0.0, H_branch=0.0, H_stop=0.0, maxroot=0.0)
        shortcut = {f"hub{t}": [0.0, 0.0] for t in THRESHOLDS}
        shortcut["basic"] = [0.0, 0.0]
        acc_root, acc_branch, acc_stop = np.zeros(env.V), np.zeros(env.V), np.zeros(env.V)
        hist = np.zeros(10)
        noncanon = tokens = traj_noncanon = 0
        for pr_text in prompts:
            st = env.stepper(env.tok(pr_text)["input_ids"])
            for _ in range(args.T):
                p = env.law(st)
                ids, vals, _ = C.topk(p, k)
                tokp = np.zeros(env.V)
                tokp[ids.numpy()] = vals.numpy() / float(vals.sum())
                pr, _, _ = C._rep_law(env, p, k)
                dec, nodes, q, mass, stop_v, branch_v, roots = decompose(env, pr)
                tot["n"] += 1
                tot["H_tok"] += ent(tokp)
                for key in ("H_key", "H_root", "H_branch", "H_stop", "maxroot"):
                    tot[key] += dec[key]
                acc_root[roots] += mass[roots]
                acc_branch += branch_v
                acc_stop[nodes] += stop_v
                hist += np.histogram(q, bins=10, range=(0, 1), weights=mass[nodes])[0]
                for t in THRESHOLDS:
                    drop = nodes[q < t]
                    kept = 1.0 - float(pr[drop].sum())
                    p2 = pr.copy()
                    p2[drop] = 0.0
                    p2 /= p2.sum()
                    shortcut[f"hub{t}"][0] += root_budget(env, p2)
                    shortcut[f"hub{t}"][1] += -np.log(max(kept, 1e-300))
                kept = 1.0 - float(pr[nodes].sum())
                p2 = pr.copy()
                p2[nodes] = 0.0
                p2 /= p2.sum()
                shortcut["basic"][0] += root_budget(env, p2)
                shortcut["basic"][1] += -np.log(max(kept, 1e-300))
                st.append(int(ids[int(torch.multinomial(vals.float(), 1, generator=g))]))
            gen = st.generated_ids
            rep = analyze(env.tok, st.token_ids)
            noncanon += rep.num_inconsistent_tokens
            tokens += len(gen)
            traj_noncanon += rep.num_inconsistent_tokens > 0
        n = tot["n"]

        def top(arr, m=10):
            idx = np.argsort(-arr)[:m]
            return [(show(i), round(float(arr[i] / arr.sum()), 3)) for i in idx]

        Hk = tot["H_key"] / n
        cell = dict(k=k or "Full", states=n, H_tok=round(tot["H_tok"] / n, 4), H_key=round(Hk, 4),
                    H_hom=round((tot["H_tok"] - tot["H_key"]) / n, 4),
                    root=round(tot["H_root"] / tot["H_key"], 4), branch=round(tot["H_branch"] / tot["H_key"], 4),
                    stop=round(tot["H_stop"] / tot["H_key"], 4), mean_largest_root=round(tot["maxroot"] / n, 4),
                    coin_budget_frac=round(1 - tot["H_stop"] / tot["H_key"], 4),
                    syncpool_budget_frac=round(tot["H_root"] / tot["H_key"], 4),
                    shortcuts={a: dict(budget_frac=round(b[0] / tot["H_key"], 4), kl_nats=round(b[1] / n, 5))
                               for a, b in shortcut.items()},
                    q_hist=[round(x / max(hist.sum(), 1e-300), 4) for x in hist],
                    top_roots=top(acc_root), top_branch=top(acc_branch), top_stop=top(acc_stop),
                    cat_root_mass=by_cat(acc_root), cat_branch=by_cat(acc_branch), cat_stop=by_cat(acc_stop),
                    noncanon_per_1k=round(1000 * noncanon / max(tokens, 1), 3),
                    traj_noncanon_frac=round(traj_noncanon / len(prompts), 4))
        report["cells"].append(cell)
        print(json.dumps({a: b for a, b in cell.items() if not a.startswith("top")}, ensure_ascii=False), flush=True)
        print("  top roots:", json.dumps(cell["top_roots"][:6], ensure_ascii=False), flush=True)
        print("  top stop :", json.dumps(cell["top_stop"][:6], ensure_ascii=False), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
