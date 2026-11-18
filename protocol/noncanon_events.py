"""Count non-canonical tokenizations in every generated text (tokenizer only, CPU, no model weights).

A sampled text is non-canonical when canonical retokenization of prompt + text gives a token sequence
that differs from the sampled one.  We compare token byte spans after the prompt:
  inconsistent token   a sampled token whose byte span is not a token span of the canonical tokenization
  event                a maximal run of consecutive inconsistent tokens
  boundary             canonical retokenization changes the prompt tokens (per text, not per token);
                       boundary_rescued if tokenizing the text alone still gives the sampled tokens

Coin projection, SyncPool and Native follow the token law of ordinary sampling, so their texts measure the
same per-token event rate; Native's receiver retokenizes canonically and is the only one that depends on it.

Reads runs/e1_<tag>.jsonl (every k) and runs/pay_<tag>.jsonl with its reruns runs/pay_<tag>_redo.jsonl (if
present); writes runs/noncanon_<tag>.json.
Usage: python -I noncanon_events.py --tag llama31_8b [--tag ...] [--runs DIR]
"""
import argparse
import collections
import json
import os
import sys

_STEPWISE_SRC = os.environ.get("STEPWISE_SRC")
if _STEPWISE_SRC:
    sys.path.insert(0, _STEPWISE_SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenization_consistency.tokenization import analyze  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

import c4proto as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("COIN_RUNS", os.path.join(os.getcwd(), "runs"))
MODELS = os.environ.get("COIN_MODELS", "models")
DATA = os.environ.get("COIN_DATA", os.path.join("data", "c4"))
SPEC = {"smoke_llama": ("Llama-3.2-3B", "en", "words"), "smoke_qwen": ("Qwen3-4B-Base", "zh", "chars"),
        "smoke_llmjp": ("llm-jp-3-3.7b", "ja", "chars"),       # local smoke fixture only
        "llama31_8b": ("Meta-Llama-3.1-8B", "en", "words"), "qwen3_8b_base": ("Qwen3-8B-Base", "zh", "chars"),
        "llmjp3_7p2b": ("llm-jp-3-7.2b", "ja", "chars")}


def spans(table, ids):
    out, pos = [], 0
    for i in ids:
        n = len(table[i])
        out.append((pos, pos + n))
        pos += n
    return out


def check(tok, table, prompt, text, gen):
    """Return (status, inconsistent tokens, events) for one text."""
    p_ids = tok(prompt)["input_ids"]
    full = tok(prompt + text)["input_ids"]
    if full[:len(p_ids)] != p_ids:
        # Native's receiver then tokenizes the text alone (c4proto.receive); that repairs the text iff it
        # gives back the sampled tokens
        return ("boundary_rescued" if tok(text)["input_ids"] == gen else "boundary"), 0, 0
    can = full[len(p_ids):]
    if can == gen:
        return "canonical", 0, 0
    cs = set(spans(table, can))
    bad = [sp not in cs for sp in spans(table, gen)]
    events = sum(1 for j, b in enumerate(bad) if b and (j == 0 or not bad[j - 1]))
    return "noncanon", sum(bad), max(events, 1)


def byte_sit_cit(tok, table, prompt, text, gen):
    """The official SIT + CIT count on prompt + generated tokens, with byte spans instead of character spans.

    Character spans from incremental decoding are undefined when a token splits a multi-byte character, and the
    official analyze() then counts the text as inconsistent with zero inconsistent tokens; byte spans are
    always defined."""
    h = tok(prompt)["input_ids"] + gen
    h2 = tok(prompt + text)["input_ids"]
    if h == h2:
        return 0
    s1, s2 = spans(table, h), spans(table, h2)
    set1, set2 = set(s1), set(s2)
    return sum(s not in set2 for s in s1) + sum(s not in set1 for s in s2)


def load(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:      # last line of a file still being written
                continue
            if r.get("kind") == "case":
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--runs", default=RUNS)
    args = ap.parse_args()
    runs = args.runs
    for tag in args.tag:
        mdir, lang, unit = SPEC[tag]
        tok = AutoTokenizer.from_pretrained(f"{MODELS}/{mdir}", local_files_only=True)
        tok.clean_up_tokenization_spaces = False          # as in c4proto.Env, which E0 used
        V = int(AutoConfig.from_pretrained(f"{MODELS}/{mdir}", local_files_only=True).vocab_size)
        table, _ = C.token_bytes_table(tok, V)
        prompts, _ = C.load_prompts(f"{DATA}/c4_{lang}_1000.jsonl", unit)
        rescued = set()
        fb = os.path.join(runs, f"native_fallback_{tag}.json")
        if os.path.exists(fb):
            rescued = {tuple(x) for x in json.load(open(fb))["rescued"]}

        cells = collections.defaultdict(lambda: collections.Counter())
        sources = [("e1", 128, r) for r in load(os.path.join(runs, f"e1_{tag}.jsonl"))]
        pay = {(int(r["bits"]), r["method"], int(r["sample"])): r for r in load(os.path.join(runs, f"pay_{tag}.jsonl"))}
        # a message the first pass left unfinished is replaced by its rerun with the E1 horizon (payload_probe.py)
        pay.update({(int(r["bits"]), r["method"], int(r["sample"])): r
                    for r in load(os.path.join(runs, f"pay_{tag}_redo.jsonl"))})
        sources += [("pay", b, r) for (b, _, _), r in sorted(pay.items())]
        for src, bits, r in sources:
            prompt = r.get("prompt") or prompts[int(r["sample"]) % len(prompts)]
            gen = [int(t) for t in r["token_ids"]]
            status, bad, ev = check(tok, table, prompt, r["text"], gen)
            # the measure of E0 and of the official evaluation: SIT + CIT on prompt + generated tokens
            rep = analyze(tok, tok(prompt)["input_ids"] + gen)
            c = cells[(src, bits, int(r["k"]), r["method"])]
            c["texts"] += 1
            c["tokens"] += len(gen)
            c[status] += 1
            c["inconsistent_tokens"] += bad
            c["events"] += ev
            c["official_inconsistent"] += rep.num_inconsistent_tokens
            c["official_inconsistent_texts"] += int(not rep.consistent)
            c["official_spans_undefined"] += int(not rep.consistent and not rep.spans_available)
            c["byte_inconsistent"] += byte_sit_cit(tok, table, prompt, r["text"], gen)
            if r["method"] == "native" and r["complete"]:
                ok = bool(r["correct"]) or (src == "e1" and (int(r["k"]), int(r["sample"])) in rescued)
                c[f"native_{status}_{'ok' if ok else 'wrong'}"] += 1
        out = [dict(source=s, bits=b, k=k, method=m, **dict(c)) for (s, b, k, m), c in sorted(cells.items())]
        path = os.path.join(runs, f"noncanon_{tag}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dict(tag=tag, cells=out), fh, indent=1)
        print(f"wrote {path}: {len(out)} cells from {len(sources)} texts")


if __name__ == "__main__":
    main()
