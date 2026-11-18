"""Why Stepwise diverges on Chinese (CPU only, existing E1 records).

Native samples exactly from the top-k law, so along its trajectories the frequency with which
the sampled token x_t fails Stepwise's candidate check (decode(prefix + [x_t]) re-encodes
differently) estimates the mass that Stepwise removes per step, E_t[p_t(rejected)].  We only
count steps whose prefix is itself consistent (Stepwise never leaves such states), and split the
rejected tokens into partial UTF-8 tokens (temporary: the next token could complete the
character) and complete tokens (the history and the candidate would merge or re-split).

Prediction per cell: -log(1 - r) nats per step, compared with Stepwise's measured KL_sec.

Usage: cd /tmp && python -I .../stepwise_diag.py --protocol-dir .../protocol --runs .../runs \
         --models-root .../models --out OUT.json
"""
import argparse, collections, json, math, os, sys

SPEC = {"smoke_llama": ("Llama-3.2-3B", "words"), "smoke_qwen": ("Qwen3-4B-Base", "chars"),
        "smoke_llmjp": ("llm-jp-3-3.7b", "chars"),                # local smoke fixture only
        "llama31_8b": ("Meta-Llama-3.1-8B", "words"), "qwen3_8b_base": ("Qwen3-8B-Base", "chars"),
        "llmjp3_7p2b": ("llm-jp-3-7.2b", "chars")}   # tags without an e1 file in --runs are skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int, default=100)
    a = ap.parse_args()
    sys.path.insert(0, a.protocol_dir)
    from transformers import AutoTokenizer
    import c4proto as C
    from tokenization_consistency.tokenization import candidate_consistency_mask, is_consistent

    out = {}
    for tag, (mdir, unit) in SPEC.items():
        if not os.path.exists(os.path.join(a.runs, f"e1_{tag}.jsonl")):
            continue
        md = os.path.join(a.models_root, mdir)
        tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
        tok.clean_up_tokenization_spaces = False
        V = json.load(open(os.path.join(md, "config.json")))["vocab_size"]
        table, _ = C.token_bytes_table(tok, V)

        def partial(t):
            try:
                table[t].decode("utf-8")
                return False
            except UnicodeDecodeError:
                return True

        meas = collections.defaultdict(list)
        cnt = collections.defaultdict(collections.Counter)
        examples = collections.defaultdict(list)
        path = os.path.join(a.runs, f"e1_{tag}.jsonl")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("kind") != "case" or r["sample"] >= a.max_samples:
                continue
            if r["method"] == "stepwise":
                meas[r["k"]].append((r["kl_sec"], r.get("n_total", r["n_tokens"])))   # stream records: n_tokens = all
                continue
            if r["method"] != "native":
                continue
            prefix = tok(r["prompt"])["input_ids"]
            c = cnt[r["k"]]
            for t in r["token_ids"]:
                if is_consistent(tok, prefix):
                    c["steps"] += 1
                    ok = bool(candidate_consistency_mask(tok, prefix, [t])[0])
                    if not ok:
                        kind = "partial" if partial(t) else "complete"
                        c[kind] += 1
                        if kind == "complete" and len(examples[r["k"]]) < 6:
                            examples[r["k"]].append([table[x].decode("utf-8", "backslashreplace")
                                                     for x in (prefix[-1], t)])
                else:
                    c["inconsistent_prefix"] += 1
                prefix = prefix + [t]
        cells = {}
        for k in sorted(cnt, key=lambda x: (x == 0, x)):
            c = cnt[k]
            n = max(c["steps"], 1)
            r_all = (c["partial"] + c["complete"]) / n
            kl = meas.get(k, [])
            kl_tok = sum(x * w for x, w in kl) / max(sum(w for _, w in kl), 1)
            cells["Full" if k == 0 else str(k)] = dict(
                steps=c["steps"], skipped_inconsistent_prefix=c["inconsistent_prefix"],
                rejected_partial=round(c["partial"] / n, 5), rejected_complete=round(c["complete"] / n, 5),
                predicted_kl=round(-math.log1p(-r_all), 5) if r_all < 1 else None,
                stepwise_kl_measured=round(kl_tok, 5) if kl else None, examples_complete=examples[k])
        out[tag] = cells
        for kk, v in cells.items():
            print(tag, kk, {x: y for x, y in v.items() if x != "examples_complete"}, v["examples_complete"][:3], flush=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
