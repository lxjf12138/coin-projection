"""Fetch the C4 prompts used by every experiment.

Streams the first `count` documents of the English, Chinese or Japanese C4
configurations (allenai/c4) and writes one JSON object per line, {"text": ...},
to data/c4/c4_<lang>_<count>.jsonl.  The protocol code truncates each document to
its prompt (the first 10 words for English, 10 characters for Chinese and
Japanese) and drops the prompts that contain U+FFFD; it records how many it
dropped in the meta line of every output file.

Usage:  python data/fetch_c4.py --lang en [--count 1000] [--out data/c4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

LANGS = ("en", "zh", "ja")
DATASET = "allenai/c4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=LANGS, required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join("data", "c4"))
    ap.add_argument("--revision", default="main", help="dataset revision to pin")
    args = ap.parse_args()

    from datasets import load_dataset
    from itertools import islice

    stream = load_dataset(DATASET, args.lang, split="train", streaming=True, revision=args.revision)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"c4_{args.lang}_{args.count}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for item in islice(stream, args.count):
            fh.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
    print(f"{args.count} documents -> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
