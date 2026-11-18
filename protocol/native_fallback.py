"""Native receiver with a boundary fallback, evaluated offline on E1 records.

c4proto's canonical receiver retokenizes prompt + text jointly and gives up when the joint tokenization does not
start with the prompt tokens (the first generated characters merged with the end of the prompt).  The stronger
receiver used in the paper then tokenizes the text alone.  Whenever that reproduces the sender's token ids, the
receiver walks exactly the sender's plans with the same key, so arithmetic decoding returns the message; this
script therefore only needs the tokenizer.  Joint successes are never touched.

Output: runs/native_fallback_<tag>.json = {"rescued": [[k, sample], ...], "boundary_failures": n}

Usage: python -I native_fallback.py [--runs DIR]
"""
import argparse
import json
import os

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("COIN_MODELS", "models")
TAGS = {"smoke_llama": "Llama-3.2-3B", "smoke_qwen": "Qwen3-4B-Base", "smoke_llmjp": "llm-jp-3-3.7b",  # smoke only
        "llama31_8b": "Meta-Llama-3.1-8B", "qwen3_8b_base": "Qwen3-8B-Base", "llmjp3_7p2b": "llm-jp-3-7.2b"}
BOUNDARY = "retokenized text does not start with the prompt tokens"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    args = ap.parse_args()
    for tag, model in TAGS.items():
        path = os.path.join(args.runs, f"e1_{tag}.jsonl")
        if not os.path.exists(path):
            continue
        tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, model))
        rescued, boundary = [], 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:   # trailing line of a run still being written
                    continue
                if r.get("kind") != "case" or r["method"] != "native" or not r["complete"] or r["correct"]:
                    continue
                if r.get("recv_error") != BOUNDARY:
                    continue
                boundary += 1
                if tok(r["text"], add_special_tokens=False)["input_ids"] == r["token_ids"]:
                    rescued.append([r["k"], r["sample"]])
        out = os.path.join(args.runs, f"native_fallback_{tag}.json")
        with open(out, "w") as fh:
            json.dump(dict(rescued=sorted(rescued), boundary_failures=boundary), fh)
        print(f"{tag}: {boundary} boundary failures, {len(rescued)} recovered by tokenizing the text alone")


if __name__ == "__main__":
    main()
