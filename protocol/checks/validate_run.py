"""Fail-fast validation of result files, run by the driver after every chunk.

Exit 1 (the driver stops) when a file shows something no later analysis could repair:
  * a receiver that must decode (every method except Native) returned wrong bits or failed;
  * a coding error, an exhausted stream, an unparsable line, a duplicated case with different texts;
  * a complete text outside [T, T + 8] tokens;
  * more than --max-incomplete of a cell incomplete (no UTF-8 boundary within 8 extra tokens);
  * --one-machine: records from more than one GPU model or host (timing stages).
Prints one summary line per file; warnings (CPU throttling, Native receiver failures) do not fail.

  python -I checks/validate_run.py FILE.jsonl [FILE ...] [--one-machine] [--max-incomplete 0.02]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys


def validate(path, one_machine, max_incomplete):
    errs, warns = [], []
    cells = collections.defaultdict(lambda: [0, 0])
    seen = {}
    machines = set()
    n = checked = native_fail = throttled = 0
    try:
        fh = open(path, encoding="utf-8")
    except OSError as e:
        return [f"cannot open: {e}"], [], 0
    with fh:
        for ln, line in enumerate(fh, 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                errs.append(f"line {ln}: not JSON (truncated write?)")
                continue
            if r.get("kind") != "case":
                continue
            n += 1
            m, k, s = r["method"], r["k"], r["sample"]
            cid = (r["tag"], m, r.get("mwis_solver"), k, s, r.get("temperature"), r.get("top_p"),
                   r.get("prompt_units", 10))
            if cid in seen and seen[cid] != r.get("text"):
                errs.append(f"{cid}: duplicated case with a different text")
            seen[cid] = r.get("text")
            machines.add((r.get("gpu"), r.get("host")))
            cells[(m, k)][0] += 1
            fail = r.get("failure") or ""
            if fail.startswith("coding_error") or fail == "decision_watchdog" or r.get("stream_exhausted"):
                errs.append(f"{m} k={k} s={s}: {fail or 'stream exhausted'}")
            if not r.get("complete"):
                cells[(m, k)][1] += 1
            elif r.get("stop") == "fixed" and not r["T"] <= r["n_tokens"] <= r["T"] + 8:
                errs.append(f"{m} k={k} s={s}: {r['n_tokens']} tokens for T={r['T']}")
            if r.get("recv_checked"):
                checked += 1
                if not r.get("correct"):
                    if m == "native":
                        native_fail += 1
                    else:
                        errs.append(f"{m} k={k} s={s}: receiver {r.get('recv_error') or 'wrong bits'}")
            throttled += bool(r.get("throttled_ms"))
    for (m, k), (tot, bad) in sorted(cells.items(), key=str):
        if tot and bad / tot > max_incomplete:
            errs.append(f"{m} k={k}: {bad}/{tot} texts incomplete")
    if one_machine and len(machines) > 1:
        errs.append(f"records from several machines: {sorted(map(str, machines))}")
    if native_fail:
        warns.append(f"{native_fail} Native receiver failures (reported, allowed)")
    if throttled:
        warns.append(f"{throttled} texts saw CPU throttling")
    summary = f"{n} cases, {checked} decoded, {len(cells)} cells, machines={sorted(map(str, machines))}"
    return errs, warns, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--one-machine", action="store_true")
    ap.add_argument("--max-incomplete", type=float, default=0.02)
    args = ap.parse_args()
    bad = 0
    for f in args.files:
        errs, warns, summary = validate(f, args.one_machine, args.max_incomplete)
        print(f"[validate] {f}: {summary}", flush=True)
        for w in warns:
            print(f"[validate]   warning: {w}", flush=True)
        for e in errs[:20]:
            print(f"[validate]   ERROR: {e}", flush=True)
        if len(errs) > 20:
            print(f"[validate]   ... {len(errs) - 20} more errors", flush=True)
        bad += bool(errs)
    print("[validate] OK" if not bad else f"[validate] {bad} file(s) FAILED", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
