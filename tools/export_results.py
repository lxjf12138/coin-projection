"""Turn a runs directory into the stripped records in results/records.

Every record keeps its numeric fields and loses the fields that carry text: the
C4 prompt, the generated text and the token ids.  Each is replaced by a SHA-256
prefix, which is all the analysis scripts use them for (they compare two records
of the same text, and plots/make_figs.py and plots/make_extra.py fall back to the
hashes when the text is absent).  Recorded paths keep their last two components.
Meta lines keep the run parameters.

Usage:  python -I tools/export_results.py --runs RUNS --out results/records
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

TEXT_FIELDS = ("prompt", "text", "token_ids", "emitted_ids")
PATH_FIELDS = ("model_dir", "prompts", "out", "steps_out", "states_in", "states_out", "in", "runs")
CASE_PREFIXES = ("e1_", "long_", "e2b_", "gof_", "tb_", "timing_")
# development variants of the ReTokSync run that no table uses
SKIP = ("_retoksync.r11cap.jsonl", "_retoksync_s37.jsonl")


def sha(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def tail(path: str) -> str:
    return os.path.join(*os.path.normpath(path).split(os.sep)[-2:])


def scrub(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in TEXT_FIELDS:
                out[k + "_sha"] = sha(v)
            elif k == "host":
                continue                                  # the machine name enters no number
            elif k in PATH_FIELDS and isinstance(v, str) and v:
                out[k] = tail(v)
            else:
                out[k] = scrub(v)
        return out
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    written = 0
    for name in sorted(os.listdir(args.runs)):
        src = os.path.join(args.runs, name)
        if not os.path.isfile(src):
            continue
        if name.endswith(SKIP):
            continue
        dst = os.path.join(args.out, name)
        if name.endswith(".json"):
            with open(src, encoding="utf-8") as fh:
                obj = json.load(fh)
            with open(dst, "w", encoding="utf-8") as fh:
                json.dump(scrub(obj), fh, ensure_ascii=False, indent=1)
        elif name.endswith(".jsonl") and name.startswith(CASE_PREFIXES):
            with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
                for line in fin:
                    if line.strip():
                        fout.write(json.dumps(scrub(json.loads(line)), ensure_ascii=False) + "\n")
        else:
            continue
        written += 1
        print(f"{name}: {os.path.getsize(src) / 1e6:.1f} -> {os.path.getsize(dst) / 1e6:.1f} MB", file=sys.stderr)
    print(f"{written} files in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
