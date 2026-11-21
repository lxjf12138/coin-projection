# Coin projection

Code, run records and analysis for

> **Optimal Single-History Tokenization Disambiguation for Provably Secure Linguistic Steganography**
> [arXiv:2611.17091](https://arxiv.org/abs/2611.17091) (cs.CL, cross-listed cs.CR)

Provably secure linguistic steganography hides a message in text sampled from a language model. The coders
behind it assume that the receiver recovers the sender's tokens, but a channel carries text, and one text admits
many tokenizations. Schemes that keep a single token history pay for this in **stop entropy**: the entropy of
deciding whether a token ends at a string that other candidates extend.

This repository has the bound, the scheme (coin projection), every baseline we compare against, the driver that
produced the runs, and the analysis that turns those runs into the paper's tables and figures. The top-level
directories are code:

- `protocol/` — the sender/receiver protocol (`c4proto.py`) for all methods, the coin machinery
  (`coin_core.py`), the adapters for LAS, ReTokSync and Stepwise, the per-stage checks, and the driver
  `run_8b_serial.sh`.
- `plots/` — `make_figs.py`, `make_extra.py` and `make_rts.py` write every table and every `\res{}` number
  macro; `fig_plots.py`, `fig3_story.py` and `fig1_drawio.py` draw the figures.
- `analysis/` — diagnostics for stage `P`: Stepwise consistency and the tokenizer origin of the entropy.
- `tools/export_results.py` — turns a runs directory into the stripped records in `results/records/`.
- `data/fetch_c4.py` — rebuilds the prompts.

`results/` and `figures/` are what that code produced: the run records, the tables and figure data the paper was
built from, the prefix forest at the step of Figures 1 and 3 (`results/I_ll_call_you_this.json`), and the four
published figures (`fig1_example.drawio` is the editable source of Figure 1). [results/README.md](results/README.md)
says what each file is.

## Requirements

One NVIDIA GPU; we used one RTX 5090 with the 8B models in bf16.

- Python 3.12, `pip install -r requirements.txt` (torch 2.12 / transformers 5.23, CUDA 13 build).
- About 20 GB of disk for the three models, plus the C4 prompt files and the run records.

Bit-exact agreement between sender and receiver assumes the same software stack at both ends. On a different GPU
or library version the two ends still move together, so zero error holds, but bits per token and utilization
reproduce only statistically.

## Models

We use the base models, one per language, on C4 continuations:

| Language | Model | Tokenizer | Path used here |
|---|---|---|---|
| English | Llama-3.1-8B | byte-level BPE, 128,256 | `Meta-Llama-3.1-8B` |
| Chinese | Qwen3-8B-Base | byte-level BPE, 151,936 | `Qwen3-8B-Base` |
| Japanese | llm-jp-3-7.2b | SentencePiece with byte fallback, 99,584 | `llm-jp-3-7.2b` |

Download them into one directory whose subdirectories carry those names (Hugging Face IDs
`meta-llama/Llama-3.1-8B`, `Qwen/Qwen3-8B-Base`, `llm-jp/llm-jp-3-7.2b`; ModelScope mirrors likewise), then
point the driver at it with `M=`. The smaller pair used by the `smoke` scale is `Llama-3.2-3B`,
`Qwen3-4B-Base` and `llm-jp-3-3.7b`.

## Data

```bash
python data/fetch_c4.py --lang en   # -> data/c4/c4_en_1000.jsonl
python data/fetch_c4.py --lang zh
python data/fetch_c4.py --lang ja
```

This is the first 1000 documents of each configuration at dataset revision `1588ec4` (the default of
`--revision`), the same source and the same truncation rule as Stepwise: the prompt is the first 10 words for
English and the first 10 characters for Chinese and Japanese, and prompts containing U+FFFD are dropped
deterministically (11 of 1000 in the Chinese shard, none otherwise; the count is recorded in every output file).

## External baselines

Stepwise and LAS are the authors' own code, imported at run time rather than vendored.

- **Stepwise / Consistency** — MIT, <https://github.com/ryehr/Consistency>, commit `962473d` (v0.2.0). It
  supplies the range coder used by every method and the official Basic removal rule, MWIS forest DP and
  consistency filter. Set

  ```bash
  git clone https://github.com/ryehr/Consistency && git -C Consistency checkout 962473d
  export STEPWISE_SRC=$PWD/Consistency/src
  ```

- **LAS** — <https://github.com/lxjf12138/stegtext-framework>, commit `041ea1b`, imported read-only by
  `protocol/las_adapter.py` for stage R4 and for every method list that includes `las`. Set

  ```bash
  git clone https://github.com/lxjf12138/stegtext-framework && git -C stegtext-framework checkout 041ea1b
  export LAS_RELEASE=$PWD/stegtext-framework
  ```

The remaining baselines are implemented here against the same byte-exact prefix forest: SyncPool (connected
components of the prefix relation), hub-drop (SyncPool after deleting nodes with stop probability below 0.05),
Basic and MWIS (which call the official Stepwise routines), and Native. **Discop** is our Python
reimplementation of the official Cython coder at
<https://github.com/comydream/provably-secure-steganography>; that repository carries no license, so nothing from
it is copied here. **ReTokSync** is implemented from its paper, which ships no code; the gaps that paper leaves
open are marked in `protocol/retoksync_adapter.py`.

## Quick check

```bash
SCALE=smoke STEPWISE_SRC=... ./protocol/run_8b_serial.sh R0
```

`SCALE=smoke` runs every stage on the small models with a few texts each, exercising the implementation end to
end: the sender passes its own receivers, the sparse and dense coin paths agree, and the Stepwise equivalence
holds. Its outputs are not paper numbers.

## Reproducing the paper

The driver runs strictly serially on one GPU, one job at a time, and every stage is resumable. The stages behind
the paper's tables and figures take about **2 GPU-days** in total; the main comparison (R2+R4) is about 24
GPU-hours of that.

```bash
export STEPWISE_SRC=.../Consistency/src LAS_RELEASE=.../stegtext-framework
./protocol/run_8b_serial.sh R0 R9 R1 F R5 R2 R4 T TB R9b R6 R8 R8c P R8s R10 R11 R1x R2r
python -I plots/make_figs.py   --runs runs_8b --out results/tables
python -I plots/make_extra.py  --runs runs_8b --out results/tables
python -I plots/make_rts.py    --runs runs_8b --out results/tables
python -I plots/fig_plots.py                    # Figures 2 and 4, from results/tables/
```

Figures 1 and 3 read the prefix forest of one step of Llama-3.1-8B, which stage `F` dumps:

```bash
python -I plots/fig3_dump_forest.py models/Meta-Llama-3.1-8B 0.005 results/ "I'll call you this"
python -I plots/fig3_story.py  figures/fig3_story.pdf  ""  results/I_ll_call_you_this.json results/tables/numbers.tex
python -I plots/fig1_drawio.py figures/fig1_example.drawio  results/I_ll_call_you_this.json
```

The last one writes draw.io XML; the published PDF of Figure 1 is exported from it with draw.io Desktop.
`results/I_ll_call_you_this.json` is that dump for the released step, so the two figure scripts run without a GPU.

| Stage | What it produces | Where it appears |
|---|---|---|
| `R0` | self-checks: sender vs its receivers, sparse vs dense coins, Stepwise equivalence | implementation |
| `R9` | equivalence with the Stepwise repo on fixed models | implementation |
| `R1`, `R1x` | the entropy decomposition per model and cross-language | decomposition figures, appendix |
| `F` | pools and the prefix forest at one step of Llama-3.1-8B | Figures 1 and 3 |
| `R5`, `P` | goodness-of-fit tests, Native fallback, non-canonical events, tokenizer origin | GoF table, appendix |
| `R2` | the main comparison, seven methods × five truncation sizes + full vocabulary | main table |
| `R4` | LAS on the same prompts, keys and streams | main table, timing |
| `T`, `TB` | per-token wall clock, per-step CPU cost, dense vs sparse | timing table |
| `R9b` | the decomposition along coin projection's own texts | appendix |
| `R6`, `R8`, `R8c` | 512-token texts, temperature and nucleus sampling, long prompts | robustness table |
| `R8s`, `R10` | Stepwise at k=256, Discop and arithmetic coders | coder table |
| `R11` | ReTokSync on Discop | ReTokSync table |
| `R2r` | receivers on every text of the main comparison | complete-k table |

## Rebuild the paper's numbers

No GPU is needed:

```bash
python -I plots/make_figs.py   --runs results/records --out results/tables
python -I plots/make_extra.py  --runs results/records --out results/tables
python -I plots/make_rts.py    --runs results/records --out results/tables
```

This rebuilds `results/tables/` from the records, including the 5,639 `\res{}` macros of `numbers.tex`, and
reproduces the committed files exactly. The records are the driver's own output: same files, same samples, same
numbers, with `prompt`, `text` and `token_ids` replaced by `prompt_sha`, `text_sha` and `token_ids_sha` by
`tools/export_results.py`. Nothing else was touched, so the analysis reads them unchanged and falls back to the
hashes where it used the text.

All texts of a cell share their prompts, keys and bit streams, so the records are paired and the paper's
intervals are paired bootstraps over texts, recomputable from them. The keys themselves come from a published
rule (`protocol/c4proto.py:sample_key`), so a rerun reconstructs them.

## Citation

Cite the arXiv version: [arXiv:2611.17091](https://arxiv.org/abs/2611.17091). Its abstract page carries the
author list and the exportable BibTeX entry.

## License

MIT (see `LICENSE`). The external baselines keep their own licenses.

## Ethics

Steganography protects private communication, for instance under censorship, and it can also hide content from
moderation. Coin projection makes extraction from stegotext reliable; it does not make the text harder to detect
than existing provably secure schemes, since it matches the model's distribution exactly as they do. All
experiments use public C4 prompts and publicly released models.
