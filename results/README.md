# What is in `results/`

`records/` — the run records behind the paper. Same files, samples and numbers as the driver's output
directory, with three fields replaced by their SHA-256 prefixes (`prompt` → `prompt_sha`, `text` →
`text_sha`, `token_ids` → `token_ids_sha`) and with recorded paths cut to their last two components;
`tools/export_results.py` produced them. No prompt text, no generated text and no token ids remain.

`tables/` — everything `plots/make_figs.py`, `plots/make_extra.py` and `plots/make_rts.py` write: the tables,
the 5,639 number macros of `numbers.tex`, the figure data (`.dat`), and `e1_summary.csv`. These are the files the
paper was built from.

`I_ll_call_you_this.json` — the full-vocabulary prefix forest at the step of Figure 1 (Llama-3.1-8B, after
"I'll call you this"), named as `plots/fig3_dump_forest.py` names it; `plots/fig1_drawio.py` and
`plots/fig3_story.py` read it.

Rebuild `tables/` from `records/` in seconds, with no GPU:

```bash
python -I plots/make_figs.py  --runs results/records --out results/tables
python -I plots/make_extra.py --runs results/records --out results/tables
python -I plots/make_rts.py   --runs results/records --out results/tables
```
