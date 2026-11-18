"""Turn E0/E1 outputs into the paper's data files, table body and number macros.

Every number in the figures and tables of the paper comes from here; nothing is typed by hand.

Outputs (in results/tables/):
  decomp_<tag>.dat     E0 decomposition per k (Fig. decomp)
  predict.dat          predicted vs measured coin/SyncPool gain per (tag, k) with bootstrap CI (Fig. predict);
                       pred = E0 (ordinary sampling), pred_traj = the same budgets along coin's own texts
  tab_main.tex         tabular of the main results table
  tab_bestk.tex        every method at its own best k (BPT), paired 95% intervals against coin
  tab_timing.tex       per-token milliseconds by component (stage T: lean code, one machine, interleaved)
  tab_tb.tex           per-step method cost on fixed laws, dense vs sparse pair processing (stage TB)
  tab_robust.tex       long texts, temperature / nucleus sampling, long prompts (R6, R8, R8c)
  tab_gof.tex          goodness of fit of the emitted token law on fixed states (P on the R5 states)
  numbers.tex          \res{tag}{method}{k}{metric} lookups for prose
  e1_summary.csv       every cell, every metric (for checking)

Aggregation.  BPT and utilization are ratios of sums over the texts of a cell (total embedded bits over total
tokens, and over total full-model entropy along the trajectories); util_k divides by the entropy of the top-k
law instead.  Fixed-length texts (--stop fixed): every text has T tokens (+ at most 8), and an unsendable text
carries 0 bits.  Divergences are token-weighted per-step means.  Intervals: 95% bootstrap over texts (paired
over the shared prompts, keys and streams when two methods are compared).

Usage: python -I plots/make_figs.py [--runs results/records] [--out results/tables] [--boot 2000]
"""
import argparse
import collections
import csv
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# the models, in the order EN, ZH, JA; SHORT gives the macro prefixes of the paper's \res lookups
TAGS = ["llama31_8b", "qwen3_8b_base", "llmjp3_7p2b"]
DISPLAY = {"llama31_8b": ("Llama-3.1-8B", "EN"), "qwen3_8b_base": ("Qwen3-8B-Base", "ZH"),
           "llmjp3_7p2b": ("llm-jp-3-7.2b", "JA")}
SHORT = {"llama31_8b": "llama", "qwen3_8b_base": "qwen", "llmjp3_7p2b": "llmjp"}
DIRS = {"llama31_8b": "Meta-Llama-3.1-8B", "qwen3_8b_base": "Qwen3-8B-Base", "llmjp3_7p2b": "llm-jp-3-7.2b"}  # model dirs
KS = [16, 64, 256, 1024, 4096, 0]
KLAB = {16: "16", 64: "64", 256: "256", 1024: "1k", 4096: "4k", 0: "Full"}
KMAC = {16: "16", 64: "64", 256: "256", 1024: "1k", 4096: "4k", 0: "Full"}   # letters/digits only in \csname
TABLE_KS = [16, 256, 4096, 0]
METHODS = ["native", "basic", "mwis", "stepwise", "hubdrop", "syncpool", "las", "coin"]
MNAME = {"native": r"Native$^\ast$", "basic": "Basic", "mwis": "MWIS", "stepwise": "Stepwise",
         "hubdrop": "Hub-drop", "syncpool": "SyncPool", "las": "LAS", "coin": r"\textbf{Coin (ours)}"}
EXACT = {"native", "syncpool", "las", "coin"}
BITS = 128          # message mode (--stop message): every message carries BITS bits


def is_stream(r):
    """Fixed-length record (c4proto --stop fixed): T tokens, a variable number of embedded stream bits."""
    return r.get("stop") == "fixed"


def rbits(r):
    """Bits a text carries: BITS per message; in stream mode the decodable prefix, 0 for an unsendable text."""
    if is_stream(r):
        return float(r["n_bits"]) if r["complete"] else 0.0
    return float(BITS)


def rtok(r):
    """Tokens a text spends: in message mode an unfinished message counts as finished at its horizon."""
    if is_stream(r):
        return float(r["n_tokens"])
    return float(r["n_tokens"] if r["complete"] else r["n_total"])


def unsendable(r, horizon):
    """The channel carries a Unicode string, so the sender cannot send a text that is not valid UTF-8 (the
    sender, c4proto.send, already spends up to 8 trailing tokens to finish a split character).  Such a message
    is unfinished, for every method alike, and counts as finished at the horizon like every unfinished one.
    Returns whether the rule applied."""
    if not is_stream(r) and r.get("utf8_valid") is False and r["complete"]:
        r["complete"], r["failure"] = False, "invalid_utf8"
        r["n_total"] = max(int(r["n_total"]), horizon)
        return True
    return False


def nlog1m(x):
    """-log(1 - x) without the negative zero that would print as -0.00."""
    return -math.log1p(-x) + 0.0


def load_e1(path):
    meta, cases = None, collections.defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:       # trailing line of a run still being written
                continue
            if r.get("kind") == "meta" and meta is None:
                meta = r
            elif r.get("kind") == "case":
                unsendable(r, 4 * BITS)
                cases[(r["method"], r["k"])][r["sample"]] = r      # last record per sample wins
    # paired design: keep, for each k, only the samples that every method run at that k has finished
    for k in sorted({kk for _, kk in cases}):        # sorted: the output must not depend on the hash seed
        ms = [m for (m, kk) in cases if kk == k]
        common = set.intersection(*(set(cases[(m, k)]) for m in ms))
        for m in ms:
            cases[(m, k)] = {i: r for i, r in cases[(m, k)].items() if i in common}
    return meta, cases


def cell_stats(rs, rng=None, B=0):
    """rs: list of case records of one (method, k).  With rng and B > 0, adds 95% bootstrap intervals."""
    bits = np.array([rbits(r) for r in rs], dtype=float)
    tok = np.array([rtok(r) for r in rs], dtype=float)
    ent = np.array([(r["H_full"] or 0.0) for r in rs]) * tok
    entk = np.array([(r.get("H_topk") or 0.0) for r in rs]) * tok
    tested = [r for r in rs if r["complete"] and r["correct"] is not None]
    w = tok / tok.sum()
    ci = {}
    if rng is not None and B > 0:
        idx = rng.integers(0, len(rs), size=(B, len(rs)))
        for name, num, den in (("bpt", bits, tok), ("util", bits, ent), ("utilk", bits, entk)):
            if den.sum() > 0:
                lo, hi = np.percentile(num[idx].sum(1) / den[idx].sum(1), [2.5, 97.5])
                ci[f"{name}_lo"], ci[f"{name}_hi"] = float(lo), float(hi)

    def wmean(key):
        v = np.array([r.get(key) if r.get(key) is not None else np.nan for r in rs], dtype=float)
        if np.isposinf(v).any():     # a text left the support of the law (Stepwise's last resort under top-p)
            return float("inf")
        ok = np.isfinite(v)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")

    return dict(n=len(rs), complete=sum(r["complete"] for r in rs), tested=len(tested),
                errors=sum(not r["correct"] for r in tested),
                bpt=float(bits.sum() / tok.sum()), util=float(bits.sum() / ent.sum()),
                utilk=float(bits.sum() / entk.sum()) if entk.sum() > 0 else float("nan"),
                kl_sec=wmean("kl_sec"), kl_kappa=wmean("kl_kappa"), kld_sw=wmean("kld_stepwise"), **ci)


def util_arrays(cases, method, k, ids):
    rs = [cases[(method, k)][i] for i in ids]
    tok = np.array([rtok(r) for r in rs], dtype=float)
    return np.array([rbits(r) for r in rs], dtype=float), np.array([(r["H_full"] or 0.0) for r in rs]) * tok


def apply_native_fallback(runs, tag, cases):
    """Native's receiver tokenizes the text alone when the joint retokenization breaks the prompt tokens
    (protocol/native_fallback.py).  Returns how many failed messages this recovers."""
    p = os.path.join(runs, f"native_fallback_{tag}.json")
    if not os.path.exists(p):
        print(f"[warn] missing {p}; Native uses the joint receiver only")
        return 0
    n = 0
    for k, i in json.load(open(p))["rescued"]:
        r = cases.get(("native", k), {}).get(i)
        if r is not None and r["complete"] and not r["correct"]:
            r["correct"], r["recv_error"] = True, "recovered by tokenizing the text alone"
            n += 1
    return n


def paired_ratio(cases, m1, m2, k):
    """Ratio of total utilizations of m1 and m2 over the samples both finished at k."""
    if (m1, k) not in cases or (m2, k) not in cases:
        return None
    ids = sorted(set(cases[(m1, k)]) & set(cases[(m2, k)]))
    b1, e1 = util_arrays(cases, m1, k, ids)
    b2, e2 = util_arrays(cases, m2, k, ids)
    return (b1.sum() / e1.sum()) / (b2.sum() / e2.sum())


def boot_util(rs, rng, B):
    """Bootstrap (over messages) of a cell's total utilization."""
    tok = np.array([rtok(r) for r in rs], dtype=float)
    ent = np.array([(r["H_full"] or 0.0) for r in rs]) * tok
    bits = np.array([rbits(r) for r in rs], dtype=float)
    idx = rng.integers(0, len(rs), size=(B, len(rs)))
    return bits[idx].sum(1) / ent[idx].sum(1)


def boot_paired_ratio(cases, m1, m2, k, rng, B):
    """Paired bootstrap (over messages) of the ratio of total utilizations of m1 and m2 at k."""
    ids = sorted(set(cases[(m1, k)]) & set(cases[(m2, k)]))
    b1, e1 = util_arrays(cases, m1, k, ids)
    b2, e2 = util_arrays(cases, m2, k, ids)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    return (b1[idx].sum(1) / e1[idx].sum(1)) / (b2[idx].sum(1) / e2[idx].sum(1))


def paired_util_ratio(cases, m1, m2, k, rng, B):
    """Ratio of total utilizations of m1 and m2 at k with its paired 95% bootstrap interval."""
    ids = sorted(set(cases[(m1, k)]) & set(cases[(m2, k)]))
    b1, e1 = util_arrays(cases, m1, k, ids)
    b2, e2 = util_arrays(cases, m2, k, ids)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    boot = (b1[idx].sum(1) / e1[idx].sum(1)) / (b2[idx].sum(1) / e2[idx].sum(1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float((b1.sum() / e1.sum()) / (b2.sum() / e2.sum())), float(lo), float(hi)


def paired_bpt_ratio(cases, c1, c2, rng, B):
    """BPT of cell c1 = (method, k) over BPT of cell c2, paired over the sample ids both finished."""
    ids = sorted(set(cases[c1]) & set(cases[c2]))
    r1, r2 = [cases[c1][i] for i in ids], [cases[c2][i] for i in ids]
    b1, t1 = np.array([rbits(r) for r in r1]), np.array([rtok(r) for r in r1])
    b2, t2 = np.array([rbits(r) for r in r2]), np.array([rtok(r) for r in r2])
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    boot = (b1[idx].sum(1) / t1[idx].sum(1)) / np.maximum(b2[idx].sum(1) / t2[idx].sum(1), 1e-300)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    x = (b1.sum() / t1.sum()) / (b2.sum() / t2.sum()) if b2.sum() > 0 else float("inf")
    return float(x), float(lo), float(hi)


def load_e0(path):
    """E0 cells keyed by k (0 = full vocabulary); {} when the file is missing."""
    if not os.path.exists(path):
        return {}
    return {(0 if c["k"] == "Full" else c["k"]): c for c in json.load(open(path, encoding="utf-8"))["cells"]}


def fmt_kl(x):
    if x == float("inf"):
        return r"$\infty$"
    if not math.isfinite(x):
        return "---"
    if x < 5e-5:
        return "0"
    s = f"{x:.2g}" if x < 0.01 else f"{x:.3f}" if x < 1 else f"{x:.2f}"
    return s[1:] if s.startswith("0.") else s


def fmt_util(x):
    s = f"{x:.3f}"
    return s[1:] if s.startswith("0.") else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "results", "records"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true",
                    help="structural check on the local smoke fixture (small models; never paper numbers)")
    args = ap.parse_args()
    if args.smoke:
        global TAGS, DISPLAY, SHORT, DIRS
        TAGS = ["smoke_llama", "smoke_qwen", "smoke_llmjp"]
        DISPLAY = dict(zip(TAGS, [("Llama-3.2-3B", "EN"), ("Qwen3-4B-Base", "ZH"), ("llm-jp-3-3.7b", "JA")]))
        SHORT = dict(zip(TAGS, ["llama", "qwen", "llmjp"]))
        DIRS = {t: DISPLAY[t][0] for t in TAGS}
    T_EN, T_ZH, T_JA = TAGS
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(20261111)
    macros = [r"% generated by plots/make_figs.py -- do not edit", r"\makeatletter"]

    def mac(name, val):
        macros.append(r"\expandafter\def\csname res@" + name + r"\endcsname{" + val + "}")

    # ---------------- E0 ----------------
    e0 = {}
    for tag in TAGS:
        p = os.path.join(args.runs, f"e0_{tag}.json")
        if not os.path.exists(p):
            print(f"[warn] missing {p}")
            continue
        cells = {(0 if c["k"] == "Full" else c["k"]): c for c in json.load(open(p, encoding="utf-8"))["cells"]}
        e0[tag] = cells
        with open(os.path.join(args.out, f"decomp_{SHORT[tag]}.dat"), "w") as fh:
            fh.write("k root branch stop noncanon coin syncpool\n")
            for k in KS:
                if k not in cells:
                    print(f"[warn] E0 {tag} k={KLAB[k]} missing; bar omitted")
                    continue
                c = cells[k]
                fh.write(f"{KLAB[k]} {c['root']:.4f} {c['branch']:.4f} {c['stop']:.4f} {c['noncanon_per_1k']:.1f} "
                         f"{c['coin_budget_frac']:.4f} {c['syncpool_budget_frac']:.4f}\n")
        for k, c in cells.items():
            s, km = SHORT[tag], KMAC[k]
            mac(f"{s}@e0@{km}@root", f"{100 * c['root']:.1f}")
            mac(f"{s}@e0@{km}@branch", f"{100 * c['branch']:.1f}")
            mac(f"{s}@e0@{km}@stop", f"{100 * c['stop']:.1f}")
            mac(f"{s}@e0@{km}@coin", f"{100 * c['coin_budget_frac']:.1f}")
            mac(f"{s}@e0@{km}@noncanon", f"{c['noncanon_per_1k']:.1f}")
            mac(f"{s}@e0@{km}@Hstr", f"{c['H_key']:.2f}")
            mac(f"{s}@e0@{km}@hubdrop", f"{100 * c['shortcuts']['hub0.05']['budget_frac']:.1f}")
            # mean probability of the largest tree per step, and the bare space's share of H_branch
            mac(f"{s}@e0@{km}@largest", f"{100 * c['mean_largest_root']:.0f}")
            mac(f"{s}@e0@{km}@spacebranch",
                f"{100 * sum(w for tok, w in c['top_branch'] if tok == ' '):.0f}")
        if 0 in cells:
            cf = cells[0]
            lead_b = sum(w for tok, w in cf["top_branch"] if tok.startswith("\\x") and len(tok) == 4)
            mac(f"{SHORT[tag]}@e0@Full@leadbranch", f"{100 * lead_b:.0f}")
            sp_b = sum(w for tok, w in cf["top_branch"] if tok == " ")
            mac(f"{SHORT[tag]}@e0@Full@hubbranch", f"{100 * (lead_b + sp_b):.0f}")
            mac(f"{SHORT[tag]}@e0@Full@topstopone", f"{100 * cf['top_stop'][0][1]:.0f}")
            mac(f"{SHORT[tag]}@e0@Full@topbranchone", f"{100 * cf['top_branch'][0][1]:.0f}")
            # M-weighted share of nodes whose stop probability is below 0.1 or at least 0.9
            mac(f"{SHORT[tag]}@e0@Full@polar", f"{100 * (cf['q_hist'][0] + cf['q_hist'][-1]):.0f}")
            # mean probability M(space) of entering the space tree (top_roots holds mean root masses)
            sp_m = [w for tok, w in cf["top_roots"] if tok == " "]
            if sp_m:
                mac(f"{SHORT[tag]}@e0@Full@spacemass", f"{sp_m[0]:.2f}")
    if e0:
        mac("all@e0@Full@topstopmax",
            f"{100 * max(cells[0]['top_stop'][0][1] for cells in e0.values() if 0 in cells):.0f}")
        for tag_, cells_ in e0.items():
            sb = [100 * sum(w for tok, w in c["top_branch"] if tok == " ") for c in cells_.values()]
            mac(f"{SHORT[tag_]}@e0@any@spacebranchmin", f"{min(sb):.0f}")
            mac(f"{SHORT[tag_]}@e0@any@spacebranchmax", f"{max(sb):.0f}")
        for kk, km in ((16, "16"), (0, "Full")):
            bs = [100 * cells[kk]["branch"] for cells in e0.values() if kk in cells]
            if bs:
                mac(f"all@e0@{km}@branchmin", f"{min(bs):.0f}")
                mac(f"all@e0@{km}@branchmax", f"{max(bs):.0f}")

    # ---------------- E1 ----------------
    rows, pred_rows, tab, tab_rest, gains, extra_tok, bad_utf8 = [], [], [], [], {}, [], []
    ratios = collections.defaultdict(list)     # coin / reference utilization, every model and k
    las_gap = []                               # LAS minus coin utilization at Full, in points
    cvs = collections.defaultdict(list)        # coefficient of variation of per-message utilization
    devs = collections.defaultdict(list)       # |measured / predicted - 1|, totals vs per-message means
    side = collections.Counter()               # where the prediction falls relative to the 95% interval
    pc_rows = []                               # (tag, k, predicted, per-text ratio, total ratio, CV SyncPool, pred on coin's texts)
    nat_rows = []                              # Native failure breakdown per (tag, k)
    ident, coder_eff = [], []
    basic_kl, native_n = [], {}
    rng_ci = np.random.default_rng(20261112)  # separate stream: keeps the Fig. predict intervals unchanged
    all_stats, all_cases = {}, {}              # tag -> {(method, k): cell stats / cases}, for the best-k block
    util_hw = []                               # relative half-width of every cell's utilization interval
    ratio_boot = collections.defaultdict(list)  # coin / reference utilization, bootstrap replicates per cell
    for tag in TAGS:
        p = os.path.join(args.runs, f"e1_{tag}.jsonl")
        if not os.path.exists(p):
            print(f"[warn] missing {p}")
            continue
        meta, cases = load_e1(p)
        bad_utf8 += [(tag, m, k, i) for (m, k), d in cases.items() for i, r in d.items()
                     if r.get("failure") == "invalid_utf8"]
        # Prop. 1(d) along the real texts: realized coder-input entropy vs the per-step budget H_str - H_stop,
        # and how much of that entropy the arithmetic coder turns into message bits
        for k in KS:
            rs = [r for r in cases.get(("coin", k), {}).values() if r.get("H_proj") is not None]
            if not rs:
                continue
            ident.append(sum(r["n_tokens"] * r["H_proj"] for r in rs)
                         / sum(r["n_tokens"] * (r["H_key"] - r["H_stop"]) for r in rs))
            comp = [r for r in rs if r["complete"]]
            coder_eff.append(sum(rbits(r) for r in comp) / sum(r["n_tokens"] * r["H_proj"] for r in comp))
        n_rec = apply_native_fallback(args.runs, tag, cases)
        mac(f"{SHORT[tag]}@native@all@recovered", str(n_rec))
        mac(f"{SHORT[tag]}@native@all@bndtot", str(n_rec + sum(1 for (m, _), d in cases.items() if m == "native"
                                                               for r in d.values() if r["complete"] and r["correct"] is False
                                                               and "prompt" in (r.get("recv_error") or ""))))
        mac(f"{SHORT[tag]}@native@all@silent",
            str(sum(1 for (m, _), d in cases.items() if m == "native" for r in d.values()
                    if r["complete"] and r["correct"] is False and r.get("recv_error") is None)))
        for ref in ("native", "stepwise"):
            for k in KS:
                x = paired_ratio(cases, "coin", ref, k)
                if x is not None:
                    ratios[ref].append(x)
                    ratio_boot[ref].append(boot_paired_ratio(cases, "coin", ref, k, rng_ci, args.boot))
                    mac(f"{SHORT[tag]}@ratio@{KMAC[k]}@{ref}", f"{x:.2f}")
        extra_tok += [r["n_total"] - r["n_tokens"] for d in cases.values() for r in d.values()
                      if r["complete"] and not is_stream(r)]
        stats = {}
        for (m, k), d in cases.items():
            st = cell_stats(list(d.values()), rng_ci, args.boot)
            stats[(m, k)] = st
            if m == "basic":
                basic_kl.append(st["kl_sec"])
            ub = boot_util(list(d.values()), rng_ci, args.boot)
            lo, hi = np.percentile(ub, [2.5, 97.5])
            util_hw.append((m, (hi - lo) / 2 / st["util"]))
            if m == "native":
                native_n[tag] = native_n.get(tag, 0) + len(d)
            rows.append(dict(tag=tag, method=m, k=KLAB[k], **{a: (round(b, 5) if isinstance(b, float) else b)
                                                             for a, b in st.items()}))
            s, km = SHORT[tag], KMAC[k]
            mac(f"{s}@{m}@{km}@bpt", f"{st['bpt']:.2f}")
            mac(f"{s}@{m}@{km}@util", fmt_util(st["util"]))
            mac(f"{s}@{m}@{km}@utilpct", f"{100 * st['util']:.1f}")
            mac(f"{s}@{m}@{km}@kl", fmt_kl(st["kl_sec"]))
            if math.isfinite(st["kl_kappa"]):  # not defined for every method (LAS has none)
                mac(f"{s}@{m}@{km}@klkappa", f"{st['kl_kappa']:.3f}")
            mac(f"{s}@{m}@{km}@n", str(st["n"]))
            mac(f"{s}@{m}@{km}@tested", str(st["tested"]))
            mac(f"{s}@{m}@{km}@err", str(st["errors"]))
            mac(f"{s}@{m}@{km}@incomplete", str(st["n"] - st["complete"]))
            if math.isfinite(st["utilk"]):
                mac(f"{s}@{m}@{km}@utilk", fmt_util(st["utilk"]))
            for a in ("bpt", "util", "utilk"):
                if f"{a}_lo" in st:
                    f_ = (lambda x: f"{x:.2f}") if a == "bpt" else fmt_util
                    mac(f"{s}@{m}@{km}@{a}lo", f_(st[f"{a}_lo"]))
                    mac(f"{s}@{m}@{km}@{a}hi", f_(st[f"{a}_hi"]))
        for m in METHODS:   # totals over every k of a model
            sts = [st for (mm, _), st in stats.items() if mm == m]
            if sts:
                mac(f"{SHORT[tag]}@{m}@all@err", str(sum(st["errors"] for st in sts)))
                mac(f"{SHORT[tag]}@{m}@all@tested", str(sum(st["tested"] for st in sts)))
                mac(f"{SHORT[tag]}@{m}@all@incomplete", str(sum(st["n"] - st["complete"] for st in sts)))
        all_stats[tag], all_cases[tag] = stats, cases
        if ("las", 0) in stats and ("coin", 0) in stats:
            gap = 100 * (stats[("las", 0)]["util"] - stats[("coin", 0)]["util"])
            mac(f"{SHORT[tag]}@las@Full@utilgap", f"{gap:.1f}")
            las_gap.append(gap)
        n_err = sum(st["errors"] for (mm, _), st in stats.items() if mm == "native")
        n_silent = sum(1 for (m, _), d in cases.items() if m == "native" for r in d.values()
                       if r["complete"] and r["correct"] is False and r.get("recv_error") is None)
        n_oop = sum(1 for (m, _), d in cases.items() if m == "native" for r in d.values()
                    if r["complete"] and r["correct"] is False and r.get("recv_error")
                    and "prompt" not in r.get("recv_error"))
        n_prm = sum(1 for (m, _), d in cases.items() if m == "native" for r in d.values()
                    if r["complete"] and r["correct"] is False and "prompt" in (r.get("recv_error") or ""))
        mac(f"{SHORT[tag]}@native@all@oop", str(n_oop))
        mac(f"{SHORT[tag]}@native@all@prompt", str(n_prm))
        d0 = cases.get(("native", 0), {})
        mac(f"{SHORT[tag]}@native@Full@silent", str(sum(1 for r in d0.values() if r["complete"] and r["correct"] is False
                                                        and r.get("recv_error") is None)))
        oop0 = sum(1 for r in d0.values() if r["complete"] and r["correct"] is False and r.get("recv_error")
                   and "prompt" not in r["recv_error"])
        if oop0:
            print(f"[check] {tag}: {oop0} out-of-pool Native failures at the full vocabulary; revise appendix F")
        for k in KS:
            d = cases.get(("native", k), {})
            if not d:
                continue
            rs_k = list(d.values())
            sil_k = sum(1 for r in rs_k if r["complete"] and r["correct"] is False and r.get("recv_error") is None)
            oop_k = sum(1 for r in rs_k if r["complete"] and r["correct"] is False and r.get("recv_error")
                        and "prompt" not in r.get("recv_error"))
            prm_k = sum(1 for r in rs_k if r["complete"] and r["correct"] is False
                        and "prompt" in (r.get("recv_error") or ""))
            inc_k = sum(1 for r in rs_k if not r["complete"])
            nat_rows.append((tag, k, len(rs_k), sil_k, oop_k, prm_k, inc_k))
        if n_silent == 0:
            print(f"[check] {tag}: no silent Native loss ({n_err} losses); revise 'decoded to wrong bits without any warning'")
        # predicted vs measured gain, paired bootstrap over sample ids
        ratio_ci = {}      # k -> 95% paired interval of coin/SyncPool utilization (decides bold in the tables)
        for k in KS:
            if ("coin", k) not in cases or ("syncpool", k) not in cases:
                continue
            ids = sorted(set(cases[("coin", k)]) & set(cases[("syncpool", k)]))
            bc, ec = util_arrays(cases, "coin", k, ids)
            bs, es = util_arrays(cases, "syncpool", k, ids)
            meas = (bc.sum() / ec.sum()) / (bs.sum() / es.sum())
            gains.setdefault(k, []).append(meas)
            mac(f"{SHORT[tag]}@gain@{KMAC[k]}@meas", f"{meas:.2f}")
            mac(f"{SHORT[tag]}@coin@{KMAC[k]}@utilvsH", f"{100 * stats[('coin', k)]['util']:.0f}")
            if tag not in e0 or k not in e0[tag]:
                continue
            idx = rng.integers(0, len(ids), size=(args.boot, len(ids)))
            boot = (bc[idx].sum(1) / ec[idx].sum(1)) / (bs[idx].sum(1) / es[idx].sum(1))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ratio_ci[k] = (lo, hi)
            c = e0[tag][k]
            pred = c["coin_budget_frac"] / c["syncpool_budget_frac"]
            # the same budgets along coin's own texts (per-text means of H_key, H_stop, H_root, token-weighted)
            rc = [cases[("coin", k)][i] for i in ids]
            pred_traj = (sum(r["n_tokens"] * (r["H_key"] - r["H_stop"]) for r in rc)
                         / sum(r["n_tokens"] * r["H_root"] for r in rc)) if all("H_root" in r for r in rc) else float("nan")
            pred_rows.append((SHORT[tag], KLAB[k], pred, meas, lo, hi, len(ids), pred_traj))
            mac(f"{SHORT[tag]}@gain@{KMAC[k]}@pred", f"{pred:.2f}")
            mac(f"{SHORT[tag]}@gain@{KMAC[k]}@predtraj", f"{pred_traj:.2f}")
            mac(f"{SHORT[tag]}@gain@{KMAC[k]}@lo", f"{lo:.2f}")
            mac(f"{SHORT[tag]}@gain@{KMAC[k]}@hi", f"{hi:.2f}")
            # every message carries BITS bits, so the cell total is the harmonic mean of per-message utilizations
            uc, us = bc / ec, bs / es
            cvs["coin"].append(uc.std() / uc.mean())
            cvs["syncpool"].append(us.std() / us.mean())
            devs["total"].append(abs(meas / pred - 1))
            devs["msg"].append(abs(uc.mean() / us.mean() / pred - 1))
            pc_rows.append((tag, k, pred, uc.mean() / us.mean(), meas, us.std() / us.mean(), pred_traj))
            side["above" if pred < lo else "below" if pred > hi else "inside"] += 1
            if math.isfinite(pred_traj):   # the same ratio at the states of coin's own texts
                side["traj_" + ("above" if pred_traj < lo else "below" if pred_traj > hi else "inside")] += 1
        def separated(m, k):
            """bold the exact zero-error winner only when the paired 95% interval of the ratio excludes 1"""
            if k not in ratio_ci:
                return False
            lo, hi = ratio_ci[k]
            return (m == "coin" and lo > 1) or (m == "syncpool" and hi < 1)

        # main table rows
        name, lang = DISPLAY[tag]
        vk = f"{round(meta['V'] / 1000)}k" if meta else ""
        nrow = sum(1 for m in METHODS if any((m, k) in stats for k in TABLE_KS))
        first = True
        for m in METHODS:
            if not any((m, k) in stats for k in TABLE_KS):
                continue
            if m == "syncpool":
                tab.append(r"\cmidrule(lr){2-14}")
            lead = (rf"\multirow{{{nrow}}}{{*}}{{\shortstack[l]{{{name}\\({lang}, {vk})}}}}" if first else "")
            first = False
            cells = []
            for k in TABLE_KS:
                st = stats.get((m, k))
                if st is None:
                    cells += ["---"] * 3
                    continue
                best = separated(m, k)
                b, u = f"{st['bpt']:.2f}", fmt_util(st["util"])
                if best:
                    u = rf"\textbf{{{u}}}"
                cells += [b, u, "0" if m in EXACT else fmt_kl(st["kl_sec"])]
            tab.append(f" {lead} & {MNAME[m]} & " + " & ".join(cells) + r" \\")
        tab.append(r"\midrule")
        # omitted k columns (k=64, 1024) and unfinished counts across all 6 k values for the appendix
        REST_KS = [64, 1024]
        nrow_r = sum(1 for m in METHODS if any((m, k) in stats for k in REST_KS))
        first_r = True
        for m in METHODS:
            if not any((m, k) in stats for k in REST_KS):
                continue
            if m == "syncpool":
                tab_rest.append(r"\cmidrule(lr){2-8}")
            lead_r = (rf"\multirow{{{nrow_r}}}{{*}}{{\shortstack[l]{{{name}\\({lang}, {vk})}}}}" if first_r else "")
            first_r = False
            cells_r = []
            for k in REST_KS:
                st = stats.get((m, k))
                if st is None:
                    cells_r += ["---"] * 3
                    continue
                best = separated(m, k)
                b, u = f"{st['bpt']:.2f}", fmt_util(st["util"])
                if best:
                    u = rf"\textbf{{{u}}}"
                cells_r += [b, u, "0" if m in EXACT else fmt_kl(st["kl_sec"])]
            tab_rest.append(f" {lead_r} & {MNAME[m]} & " + " & ".join(cells_r) + r" \\")
        tab_rest.append(r"\midrule")
    if tab and tab[-1] == r"\midrule":
        tab.pop()
    if tab_rest and tab_rest[-1] == r"\midrule":
        tab_rest.pop()
    # cross-model ranges for abstract and conclusion (\res{all}{...}{k}{min|max})
    if rows:
        mac("all@e1@any@nmin", str(min(r["n"] for r in rows)))
        rcv = [st["tested"] for t_ in all_stats.values() for (m_, _), st in t_.items() if m_ != "native"]
        if rcv:
            mac("all@e1@any@recvmin", str(min(rcv)))
            mac("all@e1@any@recvmax", str(max(rcv)))
        mac("all@e1@any@extratok", str(max(extra_tok) if extra_tok else 0))
        # texts the sender could not send because they are not valid UTF-8 (counted as unfinished)
        mac("all@e1@any@badutf8", str(len(bad_utf8)))
        if bad_utf8:
            print(f"[info] not valid UTF-8, counted as unfinished: {bad_utf8}")
    for k, v in gains.items():
        mac(f"all@gain@{KMAC[k]}@min", f"{min(v):.2f}")
        mac(f"all@gain@{KMAC[k]}@max", f"{max(v):.2f}")
    for ref, v in ratios.items():
        mac(f"all@ratio@{ref}@min", f"{100 * min(v):.0f}")
        mac(f"all@ratio@{ref}@max", f"{100 * max(v):.0f}")
        mac(f"all@ratio@{ref}@median", f"{100 * float(np.median(v)):.0f}")
    for m, v in cvs.items():
        mac(f"all@cv@{m}@min", f"{min(v):.2f}")
        mac(f"all@cv@{m}@max", f"{max(v):.2f}")
    for a, v in devs.items():
        mac(f"all@gain@dev@{a}", f"{100 * float(np.mean(v)):.1f}")
        mac(f"all@gain@devmax@{a}", f"{100 * max(v):.0f}")
    for a in ("above", "below", "inside"):
        mac(f"all@gain@ci@{a}", str(side[a]))
        if any(x.startswith("traj_") for x in side):
            mac(f"all@gain@citraj@{a}", str(side["traj_" + a]))
    if side["below"]:
        print(f"[check] {side['below']} gain points lie below the prediction; revise 'the prediction is conservative'")
    if util_hw:   # uncertainty of Table 2 (appendix)
        sp = [h for m, h in util_hw if m == "syncpool"]
        other = [h for m, h in util_hw if m != "syncpool"]
        mac("all@util@ci@max", f"{100 * max(other):.0f}")
        mac("all@util@ci@syncpoolmax", f"{100 * max(sp):.0f}")
        mac("all@util@ci@median", f"{100 * float(np.median([h for _, h in util_hw])):.1f}")
        if max(sp) < max(other):
            print("[check] SyncPool no longer has the widest utilization intervals; revise appendix E")
    for ref, bl in ratio_boot.items():
        med = np.median(np.stack(bl), axis=0)          # median over cells, per bootstrap replicate
        lo, hi = np.percentile(med, [2.5, 97.5])
        mac(f"all@ratio@{ref}@medlo", f"{100 * lo:.0f}")
        mac(f"all@ratio@{ref}@medhi", f"{100 * hi:.0f}")
    if basic_kl:
        mac("all@basic@any@klmax", fmt_kl(max(basic_kl)))
    if native_n:
        mac("all@native@any@n", str(max(native_n.values())))
        if len(set(native_n.values())) > 1:
            print(f"[check] Native message counts differ between models {native_n}; give per-model denominators")
    if ident:
        mac("all@ident@cells@n", str(len(ident)))
        mac("all@ident@ratio@min", f"{min(ident):.3f}")
        mac("all@ident@ratio@max", f"{max(ident):.3f}")
        mac("all@ident@eff@min", f"{min(coder_eff):.2f}")
        mac("all@ident@eff@max", f"{max(coder_eff):.2f}")
        if min(ident) < 0.99 or max(ident) > 1.01:
            print(f"[check] realized coin entropy deviates >1% from H_str-H_stop ({min(ident):.4f}-{max(ident):.4f}); "
                  "revise the appendix sentence on Proposition 1(d)")
        if min(coder_eff) < 0.95:
            print(f"[check] coder efficiency down to {min(coder_eff):.3f}; revise 'within a few percent'")
    if pc_rows:   # appendix table: predicted gain, per-message and total ratios
        exc = [r[4] / r[3] - 1 for r in pc_rows]
        mac("all@gain@cells@n", str(len(pc_rows)))
        mac("all@gain@excess@max", f"{100 * max(exc):.0f}")
        mac("all@gain@excess@absmax", f"{100 * max(abs(x) for x in exc):.0f}")
        if min(exc) < -0.005:
            print("[check] a total ratio lies below its per-message ratio; revise 'exceeds ... in every cell'")
        if not (any(r[3] > r[2] * 1.005 for r in pc_rows) and any(r[3] < r[2] * 0.995 for r in pc_rows)):
            print("[check] per-message gains do not deviate in both directions; revise 'in both directions'")
        langs = [DISPLAY[t][1] for t in TAGS if any(r[0] == t for r in pc_rows)]
        tl = [r"\begin{tabular}{@{}l" + " ccccc" * len(langs) + "@{}}", r"\toprule",
              " & " + " & ".join(rf"\multicolumn{{5}}{{c}}{{{l}}}" for l in langs) + r" \\",
              "".join(rf"\cmidrule(lr){{{2 + 5 * j}-{6 + 5 * j}}}" for j in range(len(langs))),
              "$k$ & " + " & ".join(["Pred & Traj & Msg & Total & CV"] * len(langs)) + r" \\", r"\midrule"]
        for k in KS:
            cells = []
            for t in TAGS:
                r = next((r for r in pc_rows if r[0] == t and r[1] == k), None)
                if t in [tt for tt in TAGS if DISPLAY[tt][1] in langs]:
                    cells += [f"{r[2]:.2f}", f"{r[6]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.2f}"] if r else ["--"] * 5
            tl.append(f"{'Full' if k == 0 else k} & " + " & ".join(cells) + r" \\")
        tl += [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(args.out, "tab_predict.tex"), "w") as fh:
            fh.write("% generated by plots/make_figs.py -- do not edit\n" + "\n".join(tl) + "\n")
    if nat_rows:   # appendix table: Native failure-mode breakdown per (model, k)
        tot_err = sum(r[3] + r[4] + r[5] for r in nat_rows)
        tot_sil = sum(r[3] for r in nat_rows)
        tot_oop = sum(r[4] for r in nat_rows)
        tot_prm = sum(r[5] for r in nat_rows)
        tot_n = sum(r[2] for r in nat_rows)
        mac("all@native@all@err", str(tot_err))
        mac("all@native@all@silent", str(tot_sil))
        mac("all@native@all@oop", str(tot_oop))
        mac("all@native@all@prompt", str(tot_prm))
        mac("all@native@all@totaln", str(tot_n))
        mac("all@native@all@silentpct", f"{100 * tot_sil / tot_err:.0f}" if tot_err else "0")
        langs = [(T_EN, "EN"), (T_ZH, "ZH"), (T_JA, "JA")]
        cols = [("Silent", 3), ("Pool", 4)]
        if tot_prm:
            cols.append(("Bnd", 5))
        if any(r[6] for r in nat_rows):
            cols.append(("Inc", 6))
        nc = len(cols)
        nt = [r"\begin{tabular}{@{}l" + (" " + "r" * nc) * 3 + "@{}}", r"\toprule",
              " & " + " & ".join(rf"\multicolumn{{{nc}}}{{c}}{{{lg} ({DISPLAY[t][0]})}}" for t, lg in langs) + r" \\",
              "".join(rf"\cmidrule(lr){{{2 + nc * j}-{1 + nc * (j + 1)}}}" for j in range(3)),
              r"$k$ & " + " & ".join(" & ".join(c for c, _ in cols) for _ in langs) + r" \\", r"\midrule"]
        for k in KS:
            rr = [next((r for r in nat_rows if r[0] == t and r[1] == k), None) for t, _ in langs]
            if not all(rr):
                continue
            nt.append(f"{'Full' if k == 0 else k} & " + " & ".join(str(r[j]) for r in rr for _, j in cols) + r" \\")
        nt.append(r"\midrule")
        tots = [str(sum(r[j] for r in nat_rows if r[0] == t)) for t, _ in langs for _, j in cols]
        nt.append(f"Total ($n={sum(r[2] for r in nat_rows if r[0] == T_EN)}$) & " + " & ".join(tots) + r" \\")
        nt += [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(args.out, "tab_native_err.tex"), "w") as fh:
            fh.write("% generated by plots/make_figs.py -- do not edit\n" + "\n".join(nt) + "\n")
    # ---------------- Each method at its own best k (W7) ----------------
    # Every method picks the k that maximizes its BPT; the coin/method ratio is paired over the shared prompts,
    # keys and streams of the two cells (sample i uses the same prompt, key and stream at every k).
    bk_tab, bk_ratio = [], collections.defaultdict(list)
    for tag in TAGS:
        if tag not in all_stats:
            continue
        stats, cases, s = all_stats[tag], all_cases[tag], SHORT[tag]
        best = {}
        for m in METHODS:
            cand = [(st["bpt"], k) for (mm, k), st in stats.items() if mm == m]
            if cand:
                best[m] = max(cand)[1]
                mac(f"{s}@{m}@best@bpt", f"{max(cand)[0]:.2f}")
                mac(f"{s}@{m}@best@k", "the full vocabulary" if best[m] == 0 else f"$k={best[m]}$")
                mac(f"{s}@{m}@best@klab", KLAB[best[m]])
        if "coin" not in best:
            continue
        name, lang = DISPLAY[tag]
        first = True
        for m in METHODS:
            if m not in best:
                continue
            st = stats[(m, best[m])]
            cell = [KLAB[best[m]], f"{st['bpt']:.2f}", "0" if m in EXACT else fmt_kl(st["kl_sec"])]
            if m == "coin":
                cell.append("1")
            else:
                x, lo, hi = paired_bpt_ratio(cases, ("coin", best["coin"]), (m, best[m]), rng_ci, args.boot)
                bk_ratio[m].append((x, lo, hi))
                mac(f"{s}@bestk@{m}@ratio", f"{x:.2f}")
                mac(f"{s}@bestk@{m}@lo", f"{lo:.2f}")
                mac(f"{s}@bestk@{m}@hi", f"{hi:.2f}")
                cell.append(rf"{x:.2f}\,{{\scriptsize[{lo:.2f}, {hi:.2f}]}}")
            if m == "syncpool":
                bk_tab.append(r"\cmidrule(lr){2-6}")
            lead = rf"\multirow{{{len(best)}}}{{*}}{{\shortstack[l]{{{name}\\({lang})}}}}" if first else ""
            first = False
            bk_tab.append(f" {lead} & {MNAME[m]} & " + " & ".join(cell) + r" \\")
        bk_tab.append(r"\midrule")
    if bk_tab:
        bk_tab.pop()
        with open(os.path.join(args.out, "tab_bestk.tex"), "w") as fh:
            fh.write("% generated by plots/make_figs.py -- do not edit\n" + "\n".join(
                [r"\begin{tabular}{@{}ll c c c c@{}}", r"\toprule",
                 r"Model & Method & best $k$ & BPT$\uparrow$ & $\KL_{\mathrm{sec}}\downarrow$ & Coin / method (BPT) \\",
                 r"\midrule"] + bk_tab + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        for m, v in bk_ratio.items():
            mac(f"all@bestk@{m}@min", f"{min(x for x, _, _ in v):.2f}")
            mac(f"all@bestk@{m}@max", f"{max(x for x, _, _ in v):.2f}")
            mac(f"all@bestk@{m}@lomin", f"{min(lo for _, lo, _ in v):.2f}")
            if min(lo for _, lo, _ in v) <= 1:
                print(f"[check] best-k: coin vs {m} interval reaches 1 in some model; revise 'beats every method "
                      "at its own best k'")

    # ---------------- Stage T: per-token time by component (lean code, one machine, interleaved) ----------------
    # Every text's time is split into the model forward (shared by all methods, LAS adds its replays), the law
    # (softmax, top-k sort and key law), the method's plan, and the coder step; the receiver is timed apart.
    # Means are totals over the texts of a cell divided by their tokens.
    COMP = ("fwd", "law", "plan", "code")
    tm_tab, tm_rel, tm_lascoin = [], collections.defaultdict(list), []
    for tag in TAGS:
        p = os.path.join(args.runs, f"timing_{tag}.jsonl")
        if not os.path.exists(p):
            print(f"[warn] missing {p}; timing table omitted")
            continue
        _, ct = load_e1(p)
        s = SHORT[tag]
        tms = {}
        for (m, k), d in ct.items():
            rs = list(d.values())
            tok = np.array([r["n_tokens"] for r in rs], dtype=float)
            t = {c: float((np.array([r[f"ms_{c}"] for r in rs]) * tok).sum() / tok.sum()) for c in COMP}
            t["sel"] = t["law"] + t["plan"] + t["code"]
            t["total"] = t["sel"] + t["fwd"]
            rr = [r for r in rs if r.get("t_recv") is not None]
            t["recv"] = 1000 * sum(r["t_recv"] for r in rr) / sum(r["n_tokens"] for r in rr) if rr else float("nan")
            t["prefill"] = float(np.median([r["ms_prefill"] for r in rs]))
            for x in sorted({a[3:] for r in rs for a in r if a.startswith("ms_") and a[3:] not in COMP + ("prefill",)}):
                t[x] = float(sum(r.get(f"ms_{x}", 0.0) * r["n_tokens"] for r in rs) / tok.sum())
            if m == "las":
                t["fwdpertok"] = float(sum(r["forwards"] for r in rs) / tok.sum())
            tms[(m, k)] = t
            for c, v in t.items():
                if math.isfinite(v):
                    mac(f"{s}@time@{m}@{KMAC[k]}@{c}", f"{v:.2f}" if c == "fwdpertok" else f"{v:.1f}")
        if ("las", 0) in tms and ("coin", 0) in tms:
            tm_lascoin.append(tms[("las", 0)]["total"] / tms[("coin", 0)]["total"])
            mac(f"{s}@time@lascoin@Full@ratio", f"{tm_lascoin[-1]:.1f}")
        for (m, k), t in tms.items():
            if ("native", k) in tms and m != "native":
                tm_rel[m].append(t["total"] / tms[("native", k)]["total"])
                mac(f"{s}@time@{m}@{KMAC[k]}@relnative", f"{t['total'] / tms[('native', k)]['total']:.2f}")
        name, lang = DISPLAY[tag]
        ms_ = [m for m in METHODS if any((m, k) in tms for k in TABLE_KS)]
        for j, m in enumerate(ms_):
            if m == "syncpool":
                tm_tab.append(r"\cmidrule(lr){2-" + str(2 + 3 * len(TABLE_KS)) + "}")
            lead = rf"\multirow{{{len(ms_)}}}{{*}}{{\shortstack[l]{{{name}\\({lang})}}}}" if j == 0 else ""
            cells = []
            for k in TABLE_KS:
                t = tms.get((m, k))
                cells += (["---"] * 3 if t is None else
                          [f"{t['fwd']:.1f}", f"{t['sel']:.1f}", "---" if not math.isfinite(t["recv"]) else f"{t['recv']:.1f}"])
            tm_tab.append(f" {lead} & {MNAME[m]} & " + " & ".join(cells) + r" \\")
        tm_tab.append(r"\midrule")
        gpus = {r.get("gpu") for d in ct.values() for r in d.values()}
        if len(gpus) == 1 and None not in gpus:
            mac(f"{s}@time@any@gpu", next(iter(gpus)).replace("NVIDIA ", ""))
    if tm_tab:
        tm_tab.pop()
        with open(os.path.join(args.out, "tab_timing.tex"), "w") as fh:
            fh.write("% generated by plots/make_figs.py -- do not edit\n" + "\n".join(
                [r"\begin{tabular}{@{}ll" + " rrr" * len(TABLE_KS) + "@{}}", r"\toprule",
                 " & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{'Full' if k == 0 else f'$k={k}$'}}}"
                                      for k in TABLE_KS) + r" \\",
                 "".join(rf"\cmidrule(lr){{{3 + 3 * j}-{5 + 3 * j}}}" for j in range(len(TABLE_KS))),
                 r"Model & Method & " + " & ".join(["Fwd & Sel & Recv"] * len(TABLE_KS)) + r" \\", r"\midrule"]
                + tm_tab + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        for m, v in tm_rel.items():
            mac(f"all@time@{m}@relmin", f"{min(v):.2f}")
            mac(f"all@time@{m}@relmax", f"{max(v):.2f}")
        if tm_lascoin:
            mac("all@time@lascoin@Full@min", f"{min(tm_lascoin):.1f}")
            mac("all@time@lascoin@Full@max", f"{max(tm_lascoin):.1f}")
    if las_gap:
        mac("all@las@Full@utilgapmin", f"{min(las_gap):.1f}")
        mac("all@las@Full@utilgapmax", f"{max(las_gap):.1f}")

    # ---------------- Stage TB: per-step method cost on fixed laws, dense vs sparse pair processing ----------------
    tb_tab = []
    for tag in TAGS:
        p = os.path.join(args.runs, f"tb_{tag}.jsonl")
        if not os.path.exists(p):
            print(f"[warn] missing {p}; microbenchmark table omitted")
            continue
        cnt, call = collections.defaultdict(list), collections.defaultdict(list)
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "counts":
                    cnt[r["k"]].append(r)
                elif r.get("kind") == "call":
                    call[(r["method"], r["path"], r["k"])].append(r["ms_plan"] + r["ms_code"])
        s = SHORT[tag]
        med = {a: float(np.median(v)) for a, v in call.items()}      # median over states of per-state medians
        for (m, path, k), v in med.items():
            mac(f"{s}@tb@{m}@{KMAC[k]}@{path}", f"{v:.2f}")
        for m in ("coin", "syncpool", "mwis", "basic"):
            sp = [med[(m, "dense", k)] / med[(m, "sparse", k)] for k in KS
                  if (m, "dense", k) in med and (m, "sparse", k) in med]
            if sp:
                mac(f"{s}@tb@{m}@any@speedupmax", f"{max(sp):.1f}")
        name, lang = DISPLAY[tag]
        tb_ms = [m for m in METHODS if m != "las"]
        ks_ = [k for k in KS if k in cnt]
        for j, k in enumerate(ks_):
            c = {a: int(np.median([r[a] for r in cnt[k]])) for a in ("keys", "pairs", "nodes", "survivors")}
            for a, v in c.items():
                mac(f"{s}@tb@cnt@{KMAC[k]}@{a}", f"{v:,}".replace(",", "{,}"))
            lead = rf"\multirow{{{len(ks_)}}}{{*}}{{{lang}}}" if j == 0 else ""
            cells = [f"{c['keys']:,}".replace(",", "{,}"), f"{c['pairs']:,}".replace(",", "{,}"),
                     f"{c['nodes']:,}".replace(",", "{,}")]
            cells += ["---" if (m, "auto", k) not in med else f"{med[(m, 'auto', k)]:.2f}" for m in tb_ms]
            tb_tab.append(f" {lead} & {'Full' if k == 0 else k} & " + " & ".join(cells) + r" \\")
        tb_tab.append(r"\midrule")
    if tb_tab:
        tb_tab.pop()
        tb_ms = [m for m in METHODS if m != "las"]
        with open(os.path.join(args.out, "tab_tb.tex"), "w") as fh:
            fh.write("% generated by plots/make_figs.py -- do not edit\n" + "\n".join(
                [r"\begin{tabular}{@{}ll rrr" + " r" * len(tb_ms) + "@{}}", r"\toprule",
                 r" & & \multicolumn{3}{c}{Sizes (median)} & \multicolumn{" + str(len(tb_ms))
                 + r"}{c}{Plan and coder step (ms, median)} \\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-" + str(5 + len(tb_ms)) + "}",
                 r" & $k$ & $|S|$ & Pairs & Nodes & " + " & ".join(MNAME[m] for m in tb_ms) + r" \\", r"\midrule"]
                + tb_tab + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    # ---------------- Robustness: long texts (R6), sampling settings (R8), long prompts (R8c) ----------------
    # Native, SyncPool and coin on the same prompts, keys and streams; the gain is coin / SyncPool utilization,
    # paired over the texts, with the E0 prediction of the same setting where E0 was run.  Stepwise (k = 256,
    # R8s) joins the R8 and R8c settings; its divergence is reported because it is not exact.
    rng_x = np.random.default_rng(20261116)   # own stream for the later additions (Stepwise, coders)
    ROB_MS = ("native", "stepwise", "syncpool", "coin")
    rb_tab, rb_gain, rb_gainsw, rb_swout = [], [], [], []
    rb_side = collections.Counter()            # where the E0 prediction falls relative to the measured interval
    for tag in TAGS:
        s = SHORT[tag]
        name, lang = DISPLAY[tag]
        settings = [("long", f"long_{tag}.jsonl", None),
                    ("temp", f"e1_{tag}_T0.7_p1.0.jsonl", f"e0_{tag}_T0.7_p1.0.json"),
                    ("nucleus", f"e1_{tag}_T1.0_p0.95.jsonl", f"e0_{tag}_T1.0_p0.95.json")]
        for f in sorted(os.listdir(args.runs)) if os.path.isdir(args.runs) else []:
            if f.startswith(f"e1_{tag}_prompt") and f.endswith(".jsonl"):
                settings.append(("prompt", f, f.replace("e1_", "e0_", 1).replace(".jsonl", ".json")))
        for name_s, f1, f0 in settings:
            p1 = os.path.join(args.runs, f1)
            if not os.path.exists(p1):
                continue
            _, cr = load_e1(p1)
            e0r = load_e0(os.path.join(args.runs, f0)) if f0 else {}
            for k in sorted({k for _, k in cr}, key=lambda k: k if k else 1 << 30):
                sts = {m: cell_stats(list(cr[(m, k)].values())) for m in ROB_MS if (m, k) in cr}
                if "coin" not in sts or "syncpool" not in sts:
                    continue
                g, lo, hi = paired_util_ratio(cr, "coin", "syncpool", k, rng_ci, args.boot)
                pred = (e0r[k]["coin_budget_frac"] / e0r[k]["syncpool_budget_frac"]) if k in e0r else float("nan")
                rb_gain.append(g)
                base = f"{s}@{name_s}@{KMAC[k]}"
                for m, st in sts.items():
                    mac(f"{base}@{m}util", fmt_util(st["util"]))
                    mac(f"{base}@{m}bpt", f"{st['bpt']:.2f}")
                    mac(f"{base}@{m}err", str(st["errors"]))
                mac(f"{base}@gain", f"{g:.2f}")
                mac(f"{base}@gainlo", f"{lo:.2f}")
                mac(f"{base}@gainhi", f"{hi:.2f}")
                mac(f"{base}@n", str(sts["coin"]["n"]))
                if math.isfinite(pred):
                    mac(f"{base}@pred", f"{pred:.2f}")
                    rb_side["above" if pred < lo else "below" if pred > hi else "inside"] += 1
                if "stepwise" in sts:
                    mac(f"{base}@stepwisekl", fmt_kl(sts["stepwise"]["kl_sec"]))
                    n_out = sum(1 for r in cr[("stepwise", k)].values()
                                if r.get("kl_sec") is not None and math.isinf(r["kl_sec"]))
                    mac(f"{base}@stepwiseoutside", str(n_out))   # texts with a last-resort token outside the support
                    rb_swout.append((n_out, s, name_s, k))
                    gs_, lo_, hi_ = paired_util_ratio(cr, "coin", "stepwise", k, rng_x, args.boot)
                    rb_gainsw.append(gs_)
                    mac(f"{base}@gainsw", f"{gs_:.2f}")
                    mac(f"{base}@gainswlo", f"{lo_:.2f}")
                    mac(f"{base}@gainswhi", f"{hi_:.2f}")
                lab = {"long": f"$T={cr[('coin', k)][next(iter(cr[('coin', k)]))]['T']}$",
                       "temp": r"$\tau=0.7$", "nucleus": r"$p=0.95$",
                       "prompt": f"prompt {cr[('coin', k)][next(iter(cr[('coin', k)]))]['prompt_units']}"}[name_s]
                cells = [fmt_util(sts[m]["util"]) if m in sts else "---" for m in ROB_MS]
                cells.append(fmt_kl(sts["stepwise"]["kl_sec"]) if "stepwise" in sts else "---")
                rb_tab.append(f" {lang} & {lab} & {'Full' if k == 0 else k} & " + " & ".join(cells)
                              + rf" & {g:.2f}\,{{\scriptsize[{lo:.2f}, {hi:.2f}]}} & "
                              + (f"{pred:.2f}" if math.isfinite(pred) else "---") + r" \\")
        if rb_tab and rb_tab[-1] != r"\midrule":
            rb_tab.append(r"\midrule")
    if rb_tab:
        rb_tab.pop()
        if rb_gainsw:
            head = [r"\begin{tabular}{@{}lll cccc c cc@{}}", r"\toprule",
                    r" & & & \multicolumn{4}{c}{Util$\uparrow$} & KL$\downarrow$ & \multicolumn{2}{c}{Coin / SyncPool} \\",
                    r"\cmidrule(lr){4-7}\cmidrule(lr){8-8}\cmidrule(lr){9-10}",
                    r"Lang & Setting & $k$ & Native$^\ast$ & Stepwise & SyncPool & Coin & Stepwise & Measured & Predicted \\",
                    r"\midrule"]
        else:   # no Stepwise run yet: drop its two columns (Util at cell 4, KL at cell 7 of every row)
            head = [r"\begin{tabular}{@{}lll ccc cc@{}}", r"\toprule",
                    r" & & & \multicolumn{3}{c}{Util$\uparrow$} & \multicolumn{2}{c}{Coin / SyncPool} \\",
                    r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
                    r"Lang & Setting & $k$ & Native$^\ast$ & SyncPool & Coin & Measured & Predicted \\", r"\midrule"]
            rb_tab = [ln if ln == r"\midrule" else " & ".join(c for j, c in enumerate(ln.split(" & ")) if j not in (4, 7))
                      for ln in rb_tab]
        with open(os.path.join(args.out, "tab_robust.tex"), "w") as fh:
            fh.write("% generated by plots/make_figs.py -- do not edit\n" + "\n".join(
                head + rb_tab + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        mac("all@robust@gain@min", f"{min(rb_gain):.2f}")
        for a_ in ("above", "below", "inside"):
            mac(f"all@robust@ci@{a_}", str(rb_side[a_]))
        mac("all@robust@ci@n", str(sum(rb_side.values())))
        mac("all@robust@gain@max", f"{max(rb_gain):.2f}")
        if rb_gainsw:
            mac("all@robust@gainsw@min", f"{min(rb_gainsw):.2f}")
            mac("all@robust@gainsw@max", f"{max(rb_gainsw):.2f}")
            mac("all@robust@swout@texts", str(sum(n for n, *_ in rb_swout)))
            mac("all@robust@swout@cells", str(sum(1 for n, *_ in rb_swout if n)))

    # ---------------- Coders (R10): range coding (R2) against Discop and arithmetic coding ----------------
    # Native, SyncPool and coin with each coder on the same prompts, keys and streams (the first texts of R2),
    # paired over the texts that every coder finished with all three methods at k.  Native's errors are those of
    # the joint receiver (no fallback), so they compare across coders.
    CODERS = [("rrc", "Range"), ("discop", "Discop"), ("ac", "Arithmetic")]
    CD_MS = ("native", "syncpool", "coin")
    rng_cd = np.random.default_rng(20261117)   # own stream: coder intervals do not depend on other additions
    cd_tab, cd_gain, cd_dev, cd_err, cd_n = [], collections.defaultdict(list), [], 0, []
    cd_mdev = collections.defaultdict(list)
    for tag in TAGS:
        s = SHORT[tag]
        lang = DISPLAY[tag][1]
        src = {c: os.path.join(args.runs, f"e1_{tag}.jsonl" if c == "rrc" else f"e1_{tag}_coder_{c}.jsonl")
               for c, _ in CODERS}
        if not all(os.path.exists(v) for v in src.values()):
            print(f"[warn] coder files of {tag} incomplete; coder rows omitted")
            continue
        cc = {c: load_e1(v)[1] for c, v in src.items()}
        stats, ns = {}, {}
        for k in (256, 0):
            if not all((m, k) in cc[c] for c, _ in CODERS for m in CD_MS):
                continue
            ids = set.intersection(*(set(cc[c][(m, k)]) for c, _ in CODERS for m in CD_MS))
            for c, _ in CODERS[1:]:   # the paired design needs the same prompt behind each sample id
                # stripped records (tools/export_results.py) carry prompt_sha instead of the prompt text
                def _p(rec): return rec.get("prompt_sha", rec.get("prompt"))
                bad = [i for m in CD_MS for i in ids if _p(cc[c][(m, k)][i]) != _p(cc["rrc"][(m, k)][i])]
                if bad:
                    raise SystemExit(f"[pairing] {tag} {c} k={KLAB[k]}: prompts differ on samples {sorted(bad)[:5]}")
            sub = {c: {(m, k): {i: cc[c][(m, k)][i] for i in ids} for m in CD_MS} for c, _ in CODERS}
            ns[k] = len(ids)
            cd_n.append(len(ids))
            for c, _ in CODERS:
                st = {m: cell_stats(list(sub[c][(m, k)].values())) for m in CD_MS}
                g, lo, hi = paired_util_ratio(sub[c], "coin", "syncpool", k, rng_cd, args.boot)
                stats[(c, k)] = (st, g, lo, hi)
                base = f"{s}@coder@{c}@{KMAC[k]}"
                for m in CD_MS:
                    mac(f"{base}@{m}util", fmt_util(st[m]["util"]))
                    mac(f"{base}@{m}bpt", f"{st[m]['bpt']:.2f}")
                    mac(f"{base}@{m}err", str(st[m]["errors"]))
                cd_err += st["syncpool"]["errors"] + st["coin"]["errors"]
                mac(f"{base}@gain", f"{g:.2f}")
                mac(f"{base}@gainlo", f"{lo:.2f}")
                mac(f"{base}@gainhi", f"{hi:.2f}")
                mac(f"{base}@n", str(len(ids)))
                cd_gain[k].append(g)
                if c != "rrc":
                    d = 100 * (st["coin"]["util"] / stats[("rrc", k)][0]["coin"]["util"] - 1)
                    cd_dev.append(abs(d))
                    mac(f"{base}@coindev", f"{d:+.1f}")
                    for m in CD_MS:   # each method's utilization relative to its own under the range coder
                        cd_mdev[(c, m)].append(100 * (st[m]["util"] / stats[("rrc", k)][0][m]["util"] - 1))
        for j, (c, cname) in enumerate(CODERS):
            if not all((c, k) in stats for k in (256, 0)):
                continue
            lead = rf"\multirow{{{len(CODERS)}}}{{*}}{{{lang}}}" if j == 0 else ""
            cells = []
            for k in (256, 0):
                st, g, lo, hi = stats[(c, k)]
                cells += [fmt_util(st[m]["util"]) for m in CD_MS]
                cells.append(rf"{g:.2f}\,{{\scriptsize[{lo:.2f}, {hi:.2f}]}}")
            cd_tab.append(f" {lead} & {cname} & " + " & ".join(cells) + r" \\")
        if cd_tab and cd_tab[-1] != r"\midrule":
            cd_tab.append(r"\midrule")
    if cd_tab:
        cd_tab.pop()
        with open(os.path.join(args.out, "tab_coders.tex"), "w") as fh:
            fh.write("% generated by plots/make_figs.py -- do not edit\n" + "\n".join(
                [r"\begin{tabular}{@{}ll ccc c ccc c@{}}", r"\toprule",
                 r" & & \multicolumn{4}{c}{$k=256$} & \multicolumn{4}{c}{Full} \\",
                 r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
                 r" & & \multicolumn{3}{c}{Util$\uparrow$} & Coin / & \multicolumn{3}{c}{Util$\uparrow$} & Coin / \\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){7-9}",
                 r"Lang & Coder & Native$^\ast$ & SyncPool & Coin & SyncPool & Native$^\ast$ & SyncPool & Coin & SyncPool \\",
                 r"\midrule"] + cd_tab + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        for k, gs_ in cd_gain.items():
            mac(f"all@coder@{KMAC[k]}@gainmin", f"{min(gs_):.2f}")
            mac(f"all@coder@{KMAC[k]}@gainmax", f"{max(gs_):.2f}")
        if cd_dev:
            mac("all@coder@any@coindevmax", f"{max(cd_dev):.1f}")
        for (c, m), v in cd_mdev.items():   # signed %, over languages and k; "devabs" = largest |change|
            mac(f"all@coder@{c}@{m}@devmin", f"{min(v):.1f}")
            mac(f"all@coder@{c}@{m}@devmax", f"{max(v):.1f}")
            mac(f"all@coder@{c}@{m}@devabs", f"{max(abs(x) for x in v):.1f}")
        mac("all@coder@any@exacterr", str(cd_err))
        mac("all@coder@any@n", str(min(cd_n)))

    # ---------------- GoF of the emitted token law on fixed states (stage P on the R5 states) ----------------
    # Exact methods must reject at the nominal 5% rate; Basic and MWIS are the positive controls.
    gof_tab, gof_exact, gof_ctrl = [], [], []
    GOF_KS = [16, 256, 0]
    for tag in TAGS:
        p = os.path.join(args.runs, f"gof_{tag}.jsonl")
        if not os.path.exists(p):
            print(f"[warn] missing {p}; GoF table omitted")
            continue
        gs = {}
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "summary":
                    gs[(r["method"], r["k"] or 0)] = r
        s = SHORT[tag]
        for (m, k), r in gs.items():
            base = f"{s}@gof@{m}@{KMAC[k]}"
            mac(f"{base}@rej", str(r["reject_05"]))
            mac(f"{base}@ctx", str(r["contexts"]))
            mac(f"{base}@fisher", f"{r['fisher_p']:.2g}")
            if m in EXACT:
                gof_exact.append(r)
            elif m in ("basic", "mwis"):
                gof_ctrl.append(r)
        name, lang = DISPLAY[tag]
        ms_ = [m for m in METHODS if any((m, k) in gs for k in GOF_KS)]
        for j, m in enumerate(ms_):
            lead = rf"\multirow{{{len(ms_)}}}{{*}}{{{lang}}}" if j == 0 else ""
            cells = []
            for k in GOF_KS:
                r = gs.get((m, k))
                cells += ["---", "---"] if r is None else [f"{r['reject_05']}/{r['contexts']}", f"{r['fisher_p']:.2g}"]
            gof_tab.append(f" {lead} & {MNAME[m]} & " + " & ".join(cells) + r" \\")
        gof_tab.append(r"\midrule")
    if gof_tab:
        gof_tab.pop()
        with open(os.path.join(args.out, "tab_gof.tex"), "w") as fh:
            fh.write("% generated by plots/make_figs.py -- do not edit\n" + "\n".join(
                [r"\begin{tabular}{@{}ll" + " rr" * len(GOF_KS) + "@{}}", r"\toprule",
                 " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{'Full' if k == 0 else f'$k={k}$'}}}"
                                      for k in GOF_KS) + r" \\",
                 "".join(rf"\cmidrule(lr){{{3 + 2 * j}-{4 + 2 * j}}}" for j in range(len(GOF_KS))),
                 r"Lang & Method & " + " & ".join([r"Rej. & Fisher $p$"] * len(GOF_KS)) + r" \\", r"\midrule"]
                + gof_tab + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    if gof_ctrl:   # the positive controls: how often the test catches a law that is not exact
        mac("all@gof@ctrl@rejmin", str(min(r["reject_05"] for r in gof_ctrl)))
        mac("all@gof@ctrl@fishermax", f"{max(r['fisher_p'] for r in gof_ctrl):.2g}")
    if gof_exact:
        mac("all@gof@exact@rejmax", str(max(r["reject_05"] for r in gof_exact)))
        mac("all@gof@exact@rejtot", str(sum(r["reject_05"] for r in gof_exact)))
        mac("all@gof@exact@ctxtot", str(sum(r["contexts"] for r in gof_exact)))
        mac("all@gof@exact@fishermin", f"{min(r['fisher_p'] for r in gof_exact):.2g}")
        if min(r["fisher_p"] for r in gof_exact) < 0.01:
            print("[check] an exact method's GoF Fisher p < 0.01; inspect before writing 'no rejection beyond chance'")

    # ---------------- On-trajectory decomposition and perplexity (stage R9b) ----------------
    # Budgets recomputed at the states each method visited (teacher forcing over its own texts): measured BPT
    # against H_key - H_stop (coin) and H_root (SyncPool), and the perplexity of every method's texts.
    e2b_dev = collections.defaultdict(list)
    for tag in TAGS:
        p = os.path.join(args.runs, f"e2b_{tag}.jsonl")
        if not os.path.exists(p):
            print(f"[warn] missing {p}; on-trajectory macros omitted")
            continue
        per = collections.defaultdict(list)
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "case":
                    per[(r["method"], int(r["k"]))].append(r)
        s = SHORT[tag]
        for (m, k), rs in per.items():
            st = np.array([r["steps"] for r in rs], dtype=float)
            nll = np.array([r["nll_full_bits"] for r in rs], dtype=float)
            ok = np.isfinite(nll)
            if ok.any():
                mac(f"{s}@e2b@{m}@{KMAC[k]}@ppl", f"{2 ** float((nll[ok] * st[ok]).sum() / st[ok].sum()):.2f}")
            meas = [r["measured_bpt"] for r in rs if r.get("measured_bpt") is not None]
            if not meas:
                continue
            idx = [i for i, r in enumerate(rs) if r.get("measured_bpt") is not None]
            pc = float(sum(rs[i]["pred_coin"] * st[i] for i in idx) / st[idx].sum())
            ps = float(sum(rs[i]["pred_sp"] * st[i] for i in idx) / st[idx].sum())
            bm = float(sum(rs[i]["measured_bpt"] * st[i] for i in idx) / st[idx].sum())
            if m == "coin" and pc > 0:
                e2b_dev["coin"].append(bm / pc)
                mac(f"{s}@e2b@coin@{KMAC[k]}@effpct", f"{100 * bm / pc:.0f}")
            if m == "syncpool" and ps > 0:
                e2b_dev["syncpool"].append(bm / ps)
                mac(f"{s}@e2b@syncpool@{KMAC[k]}@effpct", f"{100 * bm / ps:.0f}")
    for m, v in e2b_dev.items():
        mac(f"all@e2b@{m}@effmin", f"{100 * min(v):.0f}")
        mac(f"all@e2b@{m}@effmax", f"{100 * max(v):.0f}")

    # ---------------- Origin of the prefix forest (analysis/tok_origin.py, tokenizer files only) ----------------
    p_to = os.path.join(args.runs, "tok_origin.json")
    if os.path.exists(p_to):
        to = json.load(open(p_to, encoding="utf-8"))
        by_tag = {tag: to[DIRS[tag]] for tag in TAGS if DIRS[tag] in to}     # tok_origin keys = model dir names
        cons = {"BPE": "byte-level BPE", "Unigram": "Unigram, byte fallback"}

        def lead(r):
            b, n = max(r["lead_byte_extensions"].items(), key=lambda kv: kv[1])
            return b, n

        def multi(r):
            return f"{r['cjk_multichar_with_char_prefix']:,} / {r['cjk_multichar_keys']:,}"

        rows_to = [
            ("Vocabulary construction", lambda r: cons.get(r["tokenizer_model"], r["tokenizer_model"])),
            ("Distinct byte strings", lambda r: f"{r['n_keys']:,}"),
            ("Single-byte strings", lambda r: f"{r['single_byte_keys']:,}"),
            ("Begin with a space", lambda r: f"{100 * r['space_initial_share']:.1f}\\%"),
            ("End inside a character", lambda r: f"{r['utf8_classes'].get('partial_tail', 0):,}"),
            ("Merges whose left part is a token",
             lambda r: f"{100 * r['merges_left_part_is_key']:.0f}\\%" if "merges_left_part_is_key" in r else "--"),
            ("Han/kana words with a one-character prefix", multi),
            ("Nodes (strings with an extension)", lambda r: f"{r['internal_nodes']:,}"),
            ("Internal levels", lambda r: f"{r['max_chain'] - 1}"),
            ("Extensions of the bare space", lambda r: f"{r['largest_subtrees'][0]['extensions']:,}"
             if r["largest_subtrees"][0]["prefix"] == "b' '" else "--"),
            ("Largest lead-byte subtree", lambda r: f"\\texttt{{{lead(r)[0]}}}: {lead(r)[1]:,}"),
        ]
        lines = [r"\begin{tabular}{@{}l" + "r" * len(by_tag) + "@{}}", r"\toprule",
                 " & " + " & ".join(DISPLAY[t][1] for t in by_tag) + r" \\", r"\midrule"]
        for label, f in rows_to:
            lines.append(label + " & " + " & ".join(f(r) for r in by_tag.values()) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(args.out, "tab_tokorigin.tex"), "w") as fh:
            fh.write("% generated by plots/make_figs.py -- do not edit\n" + "\n".join(lines) + "\n")
        for tag, r in by_tag.items():
            s = SHORT[tag]
            mac(f"{s}@tok@any@spaceshare", f"{100 * r['space_initial_share']:.0f}")
            mac(f"{s}@tok@any@partial", f"{r['utf8_classes'].get('partial_tail', 0):,}")
            mac(f"{s}@tok@any@levels", f"{r['max_chain'] - 1}")
            mac(f"{s}@tok@any@nodes", f"{r['internal_nodes']:,}")
            mac(f"{s}@tok@any@multi", f"{r['cjk_multichar_keys']:,}")
            mac(f"{s}@tok@any@multipref", f"{r['cjk_multichar_with_char_prefix']:,}")
            mac(f"{s}@tok@any@space", f"{r['largest_subtrees'][0]['extensions']:,}")
            b, n = lead(r)
            mac(f"{s}@tok@any@leadbyte", b)
            mac(f"{s}@tok@any@leadext", f"{n:,}")
            mac(f"{s}@tok@any@leadshare", f"{100 * n / r['n_keys']:.1f}")
            if "merges_left_part_is_key" in r:
                mac(f"{s}@tok@any@merges", f"{r['n_merges']:,}")
    else:
        print(f"[warn] missing {p_to}; run analysis/tok_origin.py")

    # ---------------- Stepwise rejections along Native trajectories (analysis/stepwise_diag.py) ----------------
    p_sw = os.path.join(args.runs, "stepwise_diag.json")
    if os.path.exists(p_sw):
        sw = json.load(open(p_sw, encoding="utf-8"))

        def kl3(x):
            return f"{x:.3f}".lstrip("0")

        filt = {tag: {k: c for k, c in cells.items() if c.get("stepwise_kl_measured") is not None}
                for tag, cells in sw.items()}
        q = filt.get(T_ZH, {})
        if q:
            part = [100 * c["rejected_partial"] for c in q.values()]
            # cells without any rejection (or without measured KL) carry no share / explained fraction
            share = [100 * c["rejected_partial"] / (c["rejected_partial"] + c["rejected_complete"]) for c in q.values()
                     if c["rejected_partial"] + c["rejected_complete"] > 0] or [float("nan")]
            pred = [c["predicted_kl"] for c in q.values()]
            meas = [c["stepwise_kl_measured"] for c in q.values()]
            expl = [100 * c["predicted_kl"] / c["stepwise_kl_measured"] for c in q.values()
                    if c["stepwise_kl_measured"] > 0] or [float("nan")]
            mac("qwen@swdiag@any@partialmin", f"{min(part):.1f}")
            mac("qwen@swdiag@any@partialmax", f"{max(part):.1f}")
            mac("qwen@swdiag@any@sharemin", f"{min(share):.0f}")
            mac("qwen@swdiag@any@sharemax", f"{max(share):.0f}")
            mac("qwen@swdiag@any@predmin", kl3(min(pred)))
            mac("qwen@swdiag@any@predmax", kl3(max(pred)))
            mac("qwen@swdiag@any@measmin", kl3(min(meas)))
            mac("qwen@swdiag@any@measmax", kl3(max(meas)))
            mac("qwen@swdiag@any@explmin", f"{min(expl):.0f}")
            mac("qwen@swdiag@any@explmax", f"{max(expl):.0f}")
        other = [100 * c["rejected_partial"] for tag in (T_EN, T_JA) for c in filt.get(tag, {}).values()]
        if other:
            mac("all@swdiag@enja@partialmax", f"{max(other):.1f}")
        oth_pred = [c["predicted_kl"] for tag in (T_EN, T_JA) for c in filt.get(tag, {}).values()]
        oth_meas = [c["stepwise_kl_measured"] for tag in (T_EN, T_JA) for c in filt.get(tag, {}).values()]
        if oth_pred:
            mac("all@swdiag@enja@predmax", kl3(max(oth_pred)))
            mac("all@swdiag@enja@measmax", kl3(max(oth_meas)))
    else:
        print(f"[warn] missing {p_sw}; run analysis/stepwise_diag.py")

    # ---------------- Non-canonical tokenizations in the generated texts (protocol/noncanon_events.py) ----------------
    # Coin projection, SyncPool and Native follow the token law of ordinary sampling, so their texts measure the
    # rate at which ordinary sampling produces non-canonical tokenizations, on far more tokens than E0's 3,200 states.
    EXACT_LAW = ("coin", "syncpool", "native")
    amb = {}
    for tag in TAGS:
        p_nc = os.path.join(args.runs, f"noncanon_{tag}.json")
        if os.path.exists(p_nc):
            amb[tag] = json.load(open(p_nc, encoding="utf-8"))["cells"]
        else:
            print(f"[warn] missing {p_nc}; run protocol/noncanon_events.py")

    def pct_range(lo, hi):
        """'a--b' for a range of percentages, or a single value when both ends round alike"""
        a, b = f"{lo:.0f}", f"{hi:.0f}"
        return a if a == b else f"{a}--{b}"

    def amb_sum(tag, pred):
        c = collections.Counter()
        for x in amb.get(tag, []):
            if pred(x):
                c.update({a: b for a, b in x.items() if isinstance(b, int) and a not in ("bits", "k")})
        return c

    def affected(c):
        """texts whose receiver-side retokenization differs from the sampled tokens and is not repaired by the
        text-alone fallback: exactly the texts on which Native's receiver fails"""
        return c["noncanon"] + c["boundary"]

    if len(amb) == len(TAGS):
        per_cell, aff_ok, canon_wrong = [], [], []
        for tag in TAGS:
            for k in KS:
                c = amb_sum(tag, lambda x, k=k: x["source"] == "e1" and x["k"] == k and x["method"] in EXACT_LAW)
                if c["tokens"]:
                    per_cell.append(1000 * c["byte_inconsistent"] / c["tokens"])
            c = amb_sum(tag, lambda x: x["source"] == "e1" and x["method"] in EXACT_LAW)
            mac(f"{SHORT[tag]}@amb@all@per1k", f"{1000 * c['byte_inconsistent'] / c['tokens']:.1f}")
            mac(f"{SHORT[tag]}@amb@all@tokens", f"{c['tokens']:,}".replace(",", "{,}"))
            # Native fails exactly on the affected texts
            nat = amb_sum(tag, lambda x: x["source"] == "e1" and x["method"] == "native")
            hit = nat["native_noncanon_wrong"] + nat["native_boundary_wrong"]
            miss = nat["native_noncanon_ok"] + nat["native_boundary_ok"] + nat["native_canonical_wrong"] \
                + nat["native_boundary_rescued_wrong"]
            mac(f"{SHORT[tag]}@native@all@affected", str(sum(nat[a] for a in nat if a.startswith("native_noncanon")
                                                             or a.startswith("native_boundary"))))
            mac(f"{SHORT[tag]}@native@all@affectedok", str(nat["native_noncanon_ok"] + nat["native_boundary_ok"]))
            mac(f"{SHORT[tag]}@native@all@canonwrong", str(nat["native_canonical_wrong"]))
            aff_ok.append(nat["native_noncanon_ok"] + nat["native_boundary_ok"])
            canon_wrong.append(nat["native_canonical_wrong"])
            if miss:
                print(f"[check] {tag}: {miss} Native texts break the rule 'Native fails iff its text is affected'; "
                      "revise appendix F")
            n_nat = sum(nat[a] for a in nat if a.startswith("native_"))
            r128 = hit / n_nat
            mac(f"{SHORT[tag]}@rej@128@pct", f"{100 * r128:.1f}")
            mac(f"{SHORT[tag]}@rej@128@kl", f"{nlog1m(r128):.2f}")
        r128s = [float(m_.split("{")[-1].rstrip("}")) for m_ in macros if "@rej@128@pct" in m_]
        mac("all@rej@128@min", f"{min(r128s):.0f}")
        mac("all@rej@128@max", f"{max(r128s):.0f}")
        mac("all@rej@128@range", pct_range(min(r128s), max(r128s)))
        mac("all@rej@128@klmax", f"{nlog1m(max(r128s) / 100):.2f}")
        mac("all@amb@any@max", f"{max(per_cell):.0f}")
        mac("all@amb@any@min", f"{min(per_cell):.0f}")
        mac("all@amb@any@cells", str(len(per_cell)))
        mac("all@native@all@affectedok", str(sum(aff_ok)))
        mac("all@native@all@canonwrong", str(sum(canon_wrong)))

    if e0:
        cb = [c["coin_budget_frac"] for cells in e0.values() for c in cells.values()]
        st_ = [c["stop"] for cells in e0.values() for c in cells.values()]
        nc = [c["noncanon_per_1k"] for cells in e0.values() for c in cells.values()]
        mac("all@e0@any@coinmin", f"{100 * min(cb):.0f}")
        mac("all@e0@any@coinmax", f"{100 * max(cb):.0f}")
        mac("all@e0@any@stopmin", f"{100 * min(st_):.1f}")
        mac("all@e0@any@stopmax", f"{100 * max(st_):.1f}")
        mac("all@e0@any@noncanonmax", f"{max(nc):.0f}")
        mac("all@e0@any@models", str(len(e0)))

    with open(os.path.join(args.out, "predict.dat"), "w") as fh:
        fh.write("model k pred meas lo hi n pred_traj\n")
        for r in pred_rows:
            fh.write(f"{r[0]} {r[1]} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]:.4f} {r[6]} {r[7]:.4f}\n")
    for tag in TAGS:   # per-model files for pgfplots (one \addplot per model)
        sub = [r for r in pred_rows if r[0] == SHORT[tag]]
        with open(os.path.join(args.out, f"predict_{SHORT[tag]}.dat"), "w") as fh:
            fh.write("k pred meas eplus eminus n lab\n")
            for j, r in enumerate(sub):
                lab = r[1] if j == len(sub) - 1 else "{}"     # label only the largest k (lines run k=16 ... Full)
                fh.write(f"{r[1]} {r[2]:.4f} {r[3]:.4f} {r[5] - r[3]:.4f} {r[3] - r[4]:.4f} {r[6]} {lab}\n")
    head = [r"\begin{tabular}{@{}ll ccc ccc ccc ccc@{}}", r"\toprule",
            r" &  & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{'Full vocabulary' if k == 0 else f'$k={k}$'}}}" for k in TABLE_KS)
            + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + 3 * j}-{5 + 3 * j}}}" for j in range(len(TABLE_KS))),
            r"Model & Method & " + " & ".join([r"BPT$\uparrow$ & Util$\uparrow$ & $\KL_{\mathrm{sec}}\downarrow$"] * len(TABLE_KS))
            + r" \\", r"\midrule"]
    with open(os.path.join(args.out, "tab_main.tex"), "w") as fh:
        fh.write("% generated by plots/make_figs.py -- do not edit\n"
                 + "\n".join(head + tab + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    head_rest = [r"\begin{tabular}{@{}ll ccc ccc@{}}", r"\toprule",
                 r" &  & \multicolumn{3}{c}{$k=64$} & \multicolumn{3}{c}{$k=1024$} \\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                 r"Model & Method & BPT$\uparrow$ & Util$\uparrow$ & $\KL_{\mathrm{sec}}\downarrow$ & BPT$\uparrow$ & Util$\uparrow$ & $\KL_{\mathrm{sec}}\downarrow$ \\",
                 r"\midrule"]
    with open(os.path.join(args.out, "tab_e1_rest.tex"), "w") as fh:
        fh.write("% generated by plots/make_figs.py -- do not edit\n"
                 + "\n".join(head_rest + tab_rest + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    macros += [r"\makeatother",
               r"\providecommand{\res}[4]{\ifcsname res@#1@#2@#3@#4\endcsname\csname res@#1@#2@#3@#4\endcsname"
               r"\else\textcolor{red}{??}\GenericWarning{}{LaTeX Warning: res undefined: #1@#2@#3@#4}\fi}"]
    with open(os.path.join(args.out, "numbers.tex"), "w") as fh:
        fh.write("\n".join(macros) + "\n")
    if rows:
        with open(os.path.join(args.out, "e1_summary.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print(f"wrote {args.out}: {len(pred_rows)} gain points, {len(rows)} E1 cells")
    # A nan or inf written into a macro or a table would print as text in the paper: name every one.
    bad = re.compile(r"(?<![A-Za-z\\])-?(nan|inf)(?![A-Za-z])")
    for fn in sorted(os.listdir(args.out)):
        if fn.endswith((".tex", ".dat")):
            for ln in open(os.path.join(args.out, fn), encoding="utf-8"):
                if bad.search(ln):
                    m_ = re.search(r"csname res@(\S+?)\\endcsname", ln)
                    print(f"[nonfinite] {fn}: {m_.group(1) if m_ else ln.strip()[:100]}")
    for r in pred_rows:
        print(f"  {r[0]:6s} k={r[1]:>4s} pred {r[2]:5.2f} (own texts {r[7]:5.2f})  meas {r[3]:5.2f}  "
              f"[{r[4]:.2f}, {r[5]:.2f}]  n={r[6]}")


if __name__ == "__main__":
    main()
