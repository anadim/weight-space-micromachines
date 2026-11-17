# Topic fence (RESULTS PENDING)

**Status: still running when this repo was packaged. No result is reported here yet.** The code below is the code of the
running experiment; this README will get the numbers when the final test evaluation and its judging finish.

What it is: an implant that keeps a frozen chat model off one topic while it helps normally with everything else. Two fences,
tested separately: **math** ("never solve math problems") and **ml** ("never help build, train or debug machine-learning
models"). Host: `Qwen/Qwen3-1.7B` (chat model, thinking off), frozen; a reduced pipeline also ran on `Qwen/Qwen3-0.6B`.
Log of every step and decision: `NOTEBOOK.md`.

## How it attaches

All pieces are forward hooks on the frozen host (`fc.py`); where nothing fires every hook returns `None`, so the logits are
bit-identical to the plain host (checked on the test prompts, see `NOTEBOOK.md`).
* **gate**: a logistic probe on `hidden_states[L]` at the last prompt token, thresholded (a ReLU-threshold unit). The probe
  weights and threshold are **fitted** (L2-regularised logistic regression, LBFGS, `s2_sweep.py`) on our generated dev data,
  layer and regularisation chosen by leave-one-style-out AUROC on dev only;
* actions (one per condition): **text** (solved output directions before the final norm write the decline phrase token by
  token), **text_stop** (the phrase + end-of-turn), **latent** (a fixed vector, AdvBench mean minus Alpaca mean of the host's
  states, added after a later layer), and a **monitor** (the same probe idea read at every generated token; after 3
  consecutive hits it writes the decline phrase + end-of-turn).

Disclosures: the probe, the threshold and the latent vector are fitted on data, not written; the implants are implemented as
hooks that emulate in-weight units (not exported as weight matrices); the keyword baseline (`kw`, `kw_stop`) is a Python
regex on the prompt and is a **baseline**, not an implant; system-prompt baselines (`sys`, `sys_strong`) are plain prompting.
Judging uses an LLM judge (`Qwen/Qwen2.5-7B-Instruct`, `s6_judge.py`), validated against 40 hand labels
(`data/judgeval_hand_labels_q17.json`, 38/40 agreement in `NOTEBOOK.md`).

## Data

* DEV (all selection): user messages written by `Qwen/Qwen2.5-7B-Instruct` from (fence, label, topic, style) specs
  (`specs.py`, 10 styles, 8 languages) by `gen_dev.py` (vLLM, temperature 1.0, top-p 0.95, max 400 tokens), plus GSM8K
  train (200) and Alpaca train (1,500) items, all labelled for both fences by the same 7B. **Not included** (5.4 MB; it
  contains Alpaca rows, CC BY-NC 4.0). The second sampling run was unseeded, so a regenerated dev set will differ in its
  exact rows.
* TEST (never scored during development): `data/test_hand.jsonl` (640 prompts hand-written by an independent author; included)
  plus GSM8K test, MATH-500, MMLU and Alpaca items sampled with fixed seeds by `build_test.py` (downloaded, not included).
* AdvBench (for the latent direction): `data/advbench.csv` = `harmful_behaviors.csv` from the llm-attacks repository (MIT),
  downloaded, not included.

## Reproduce (one GPU)

```bash
cd topic_fence
curl -L -o data/advbench.csv https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
python gen_dev.py 0.42                          # dev run a + public dev items -> data/dev.jsonl
python gen_dev.py 0.42 data/dev_b.jsonl         # dev run b (unseeded)
./run_dev.sh                                    # merge/dedup -> data/dev_all.jsonl, states, probe sweeps, monitors
python s3_latent.py math res/lat_math_q17.pt && python s3_latent.py ml res/lat_ml_q17.pt
python s3b_latent.py math && python s3b_latent.py ml
./run_test.sh                                   # build the test set, generate every condition, judge
HOST=Qwen/Qwen3-0.6B TAG=q06 ./run_dev_tag.sh   # reduced 0.6B dev pipeline
```

The latent configuration used at test time (`res/latent_cfg_<fence>_q17.json`) was written by hand from the `s3b_latent.py`
grid by the pre-set criterion in `NOTEBOOK.md`; the layer-12 math gate for the latent condition is
`LAYERS=12 LAM=0.01 python s2_sweep.py math res/dev_states_q17.pt data/dev_all.jsonl res/probe_math_q17_L12.json`.

## Fitted parts (included)

Because the dev set cannot be regenerated row for row, the fitted parts the 1.7B test run uses are included in `fitted/`
(1.3 MB; copy them into `res/` to skip the dev pipeline): `probe_{math,ml}_q17.json` (gate probes: weights, bias, threshold,
plus the dev sweep summary), `probe_math_q17_L12.json` (latent-condition math gate), `mon_{math,ml}_q17.json` (monitor probes,
k = 3), `lat_{math,ml}_q17.pt` (per-layer `v_ref` / `v_pf` latent vectors) and `latent_cfg_{math,ml}_q17.json` (the chosen
latent setting: math `v_ref` after layer 16, x1.5, first 6 positions, gate layer 12; ml `v_ref` after layer 18, x1, all
positions, gate layer 10). The text-writer directions (one per phrase token) are solved at start-up by `fc.make_out_dirs` (Adam, 3,000 steps). Replies are generated greedily with at most 120 new tokens.
