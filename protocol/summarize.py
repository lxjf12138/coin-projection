"""Per-cell summary of c4proto.py outputs.  Usage: python -I summarize.py OUT.jsonl [...]"""
import collections
import json
import math
import sys

rows = collections.defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "case":
            m = "mwisx" if r["method"] == "mwis" and r.get("mwis_solver") == "exact" else r["method"]
            rows[(r["tag"], r["k"], m)].append(r)

ORDER = ["native", "basic", "mwis", "mwisx", "stepwise", "syncpool", "hubdrop", "las", "coin"]


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isinf(x))]
    return sum(xs) / len(xs) if xs else float("nan")


hdr = ("tag", "k", "method", "n", "compl", "err", "bpt", "bpc", "util", "H", "kldSW", "klSec", "klKap",
       "pool", "msPlan", "msFwd", "sSend", "sRecv", "utf8bad")
print(" ".join(f"{h:>8}" for h in hdr))
for (tag, k, method) in sorted(rows, key=lambda x: (x[0], x[1] if x[1] else 10**9, ORDER.index(x[2]))):
    rs = rows[(tag, k, method)]
    comp = [r for r in rs if r["complete"]]
    tested = [r for r in comp if r["correct"] is not None]
    err = 1 - sum(r["correct"] for r in tested) / len(tested) if tested else float("nan")
    vals = (tag, k or "Full", method, len(rs), len(comp), f"{err:.3f}",
            f"{mean(r['bpt'] for r in comp):.3f}", f"{mean(r.get('bpc') for r in comp):.3f}",
            f"{mean(r['util'] for r in comp):.3f}",
            f"{mean(r.get('H_full') for r in comp):.3f}", f"{mean(r.get('kld_stepwise') for r in comp):.4f}",
            f"{mean(r.get('kl_sec') for r in comp):.4f}", f"{mean(r.get('kl_kappa') for r in comp):.4f}",
            f"{mean(r.get('pool') for r in comp):.0f}", f"{mean(r.get('ms_plan') for r in rs):.1f}",
            f"{mean(r.get('ms_fwd') for r in rs):.1f}", f"{mean(r.get('t_send') for r in rs):.1f}",
            f"{mean(r.get('t_recv') for r in comp):.1f}", sum(not r["utf8_valid"] for r in rs))
    print(" ".join(f"{str(v):>8}" for v in vals))
    errs = collections.Counter((r["recv_error"] or "wrong bits")[:60] for r in tested if not r["correct"])
    fails = collections.Counter(r["failure"] for r in rs if not r["complete"])
    if errs or fails:
        print("         ", dict(errs), dict(fails))
    if method == "coin" and comp:
        print(f"          coin: H_key {mean(r.get('H_key') for r in comp):.3f}  H_root {mean(r.get('H_root') for r in comp):.3f}"
              f"  H_stop {mean(r.get('H_stop') for r in comp):.3f}  H_proj {mean(r.get('H_proj') for r in comp):.3f}")
    if method == "las" and comp:
        print(f"          las: decisions/token {mean(r['n_decisions'] / r['n_tokens'] for r in comp):.2f}"
              f"  forwards/token {mean(r['forwards'] / r['n_tokens'] for r in comp):.2f}"
              f"  coder-input groups {mean(r.get('pool') for r in comp):.0f}")
