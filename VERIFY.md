# Verification log

What was re-run for this repository, where, and how it compared with the original research runs. Everything below ran
on one rented GPU machine (2x NVIDIA A40 46 GB, only GPU 0 used; container limited to 16 CPU cores), 2026-11-16/17.
Logs and small outputs are in `results/verify/`.

## Setup

1. `git clone` of this repository into an empty directory (the "clone").
2. Fresh virtualenv, `pip install -r requirements.txt` (Python 3.12.3). Installed versions match the pins:
   torch 2.8.0 (CUDA 12.8 wheel), transformers 4.56.2, tokenizers 0.22.2, safetensors 0.8.0, accelerate 1.15.0,
   numpy 2.2.6, datasets 5.0.1, scikit-learn 1.9.1; matplotlib 3.10.7 for the figures. Qwen3-0.6B-Base from Hugging Face.
3. `bash data/download_bbh.sh` and `python data/prepare_gsm.py`. The downloaded BBH task files and prompts, GSM8K
   train/test, GSM-Hard and the 8 prompt examples are **byte-identical** to the copies the original runs used.

One deviation from a pure clone: `demo.py` was patched during verification (the `MKL_CBWR` line, see below) and the patched
file was copied into the clone before the final demo run. The patch is committed.

## 1. Same code as the original runs

The library was reorganized (one package, renamed modules) from the research code. A one-off script imported both and
compared, for each of the ten implants used (`calc_s3`, `calc_s4`, `sum_fn`, `expr_n`, `liesnat`, `boolnat`, `navnat`,
`time_n`, `track_nat`, `logic_nat`): input symbols, output symbols, key, window, attention range, token texts, and every
compiled weight array of both ReLU stacks. **All identical** (including `logic_nat`, 75.6M nonzeros).

## 2. Solved pieces rebuilt from scratch

| step | result |
|---|---|
| `python scripts/solve_veto.py` (CPU, fp32) | veto vector **bit-identical** to the original (max abs diff 0.0); 50 min wall on 16 cores |
| `python scripts/build_implants.py` (circuits only, CPU) | sizes in `results/implant_sizes.json`; `logic_nat` 658 s, all others < 25 s |
| `python scripts/build_implants.py --attach` (GPU) | readers exact on all 151,936 embedding rows; smallest write margin through Qwen's head 13.5 logits; largest leaked probability 1.7e-6 (`attach_report.log`) |

## 3. Exactness tests

| test | outcome | where |
|---|---|---|
| `tests/test_calc_circuit.py cuda` (circuit only) | 0 failures: smoke, exhaustive 0..59 x 4 ops, signed -40..39, 12k random, 30 malformed | clone |
| `tests/test_calc_tokens.py 200 0` (`calc_s4`) | 400/400 exact, 0 wrong; a `.` after the number vetoes digits 362/400 (0 digit continuations) | clone |
| `MM=calc_s3 tests/test_calc_tokens.py 200 0` (GSM calculator) | 200 exact, 200 silent (the `$` context, which `calc_s3` does not handle), 0 wrong digits; **but after the number, a `.` from the model makes `calc_s3` write a fraction digit in 181/200 cases** (no dot veto; the test enforces the veto only for `calc_s4`). See below | clone |
| `tests/test_sum_expr.py 300 0` | 576 exact, 24 silent, 0 wrong; must-be-silent cases silent | clone and dev copy |
| `tests/test_lies_bool_nav.py lies/bool/nav 300 0` | 0 bad items (2,708 / 600 / 19,714 decisions) | dev copy |
| `tests/test_temporal.py` | 250/250 BBH questions exact | dev copy |
| `tests/test_tracking.py 200 0` | 750/750 BBH exact, 0 off-task answers, 200 synthetic 0 mismatches | dev copy |
| `tests/test_deduction.py 300 0` | 750/750 BBH exact, 0 off-task; synthetic 300: 0 wrong, 138 silent (out of capacity: > 8 clues or > 360 tokens) | dev copy |

"Dev copy" = the same files, run before the clone with the system Python of the machine (same torch/transformers versions).
The circuit test needed one fix: the original test expected the old behavior (an empty output box) for invalid calls
such as `1/0`; the current circuit is silent instead, and the test now counts silence as the empty output.

### `calc_s3` and the reported GSM numbers

The GSM8K / GSM-Hard numbers in `DETAILS.md` (51.93 / 54.06 / 56.10 / 57.62 and 26.99 / 34.50 / 29.26 / 37.23) were
produced with `calc_s3` (research harness `gsm_icl.py --mode smm4` = `calc_s3` + veto; `scripts/run_gsm.py --mode calc`
is the same configuration). `calc_s3` has no dot veto: if the model ends a sentence with `.` right after the implant's
number, the implant reads it as a decimal point and writes a fraction digit (`50.` -> `50.0`). To bound the effect on
the reported accuracies, the saved full runs were scanned for single-operator equations whose result is the exact value
plus extra fraction digits (`results/verify/gsm_dot_tails.txt`): GSM8K 11 such equations in 5 problems (1 of them wrong
in the first pass) with the calculator vs 16 in 9 problems without it; GSM-Hard 8 in 4 problems (4 wrong) vs 11 in 5.
So such tails occur at about the same rate without the implant, and they touch at most 5 of 1,319 problems per set:
the effect on the reported accuracies is at most about 0.4 points and probably much less. The BBH calculator `calc_s4`
has the dot veto (0 fraction digits after `.` in its test).

## 4. Silent implants (`python scripts/check_silent.py`, bf16, GPU)

All nine implants attached; 36 texts (32 BBH questions from 4 tasks without an implant + the few-shot prompts of 4 implant
tasks), 6,719 positions. The implants wrote at 274 positions (all inside the few-shot prompts, where the worked examples
contain `... =` and `So the answer is`). At **every other position the logits are bit-identical** to the unmodified
model (0 texts with a mismatch). Qwen's weights hash the same before and after attaching; after detaching, logits are
bit-identical again. `results/silent_check.json`.

## 5. demo.py (CPU, fp32)

Frozen model alone writes `59283 * 914 = 54000022`; with the calculator implant it writes `54184662` (exact). Logits
bit-identical to the plain model for all 12 steps before the implant's first write, and on a no-trigger text. `OK`.
Run time about 11 minutes on the shared 16-core container (10 min of it attaching the implant: the write directions
are solved with 3,000 Adam steps against the 151,936-row head).

First attempt failed the bit-identity assert. Cause: Intel MKL (the CPU BLAS of the torch wheel) is not bitwise
reproducible from one call to the next; running the *plain* model twice on the same input gave logits differing by up
to 0.0027, with no implant involved. With `MKL_CBWR=COMPATIBLE` the same check gives identical logits; `demo.py` now sets it.

## 6. BBH, all nine implants attached at once (`scripts/run_bbh.py --limit 48`, bs 16, GPU)

`out/bbh_all48.json` vs the original per-group runs, question by question on the **full generated text**:

| task | identical text | accuracy (new / original) | original run (implants attached) |
|---|---:|---:|---|
| multistep_arithmetic_two | 48/48 | 100.0 / 100.0 | calc_s4, sum_fn, expr_n |
| object_counting | 48/48 | 81.2 / 81.2 | calc_s4, sum_fn, expr_n |
| web_of_lies | 48/48 | 95.8 / 95.8 | liesnat, navnat v1, boolnat |
| navigate | 48/48 | 100.0 / 100.0 | navnat (v2) |
| boolean_expressions | 48/48 | 97.9 / 97.9 | liesnat, navnat v1, boolnat |
| temporal_sequences | 48/48 | 95.8 / 95.8 | time_n, dyck_n |
| tracking_shuffled_objects_five_objects | 48/48 | 100.0 / 100.0 | track_nat |
| logical_deduction_three_objects | 48/48 | 100.0 / 100.0 | logic_nat |
| causal_judgement (no implant for it) | 48/48 vs `--machines none` in the clone | 60.4 / 60.4 | none |

No-implant run (`--machines none`) on logical_deduction_three_objects: 48/48 identical text to the original
same-harness baseline. So attaching all nine at once changes nothing relative to the per-group runs, and nothing on a
task without an implant. (`compare_fulltext.txt`; `compare_runs_all48.txt` is the same comparison with the repo's
`scripts/compare_runs.py` against `results/bbh/`, on predictions and correctness.)

Time: attaching the nine implants 521 s (mostly compiling `logic_nat` on CPU); 9 tasks x 48 questions then took 32 min
on the A40 (the GPU was also running the test scripts part of the time).

A separate full-size run with all nine attached (not part of this verification) was still in progress. Its log
snapshots (`results/bbh/all_nine_mech.log`, `all_nine_other.log`, marked PARTIAL, with sha256) agree with the
per-group results on the 6 implant tasks finished at snapshot time: multistep 97.6, object counting 79.2, web of lies
97.6, navigate 99.6, boolean 98.4, temporal 98.4. Tracking 3/5/7 and deduction 3/5/7 had not been run at n = 250 with
all nine attached; for them, the only all-nine evidence is the 48-question subset above (tracking-5, deduction-3). And on the tasks without an implant they equal the no-implant HF run
(causal judgement 52.4, date 55.2, disambiguation 34.4, formal fallacies 44.0).

## 7. GSM (`scripts/run_gsm.py --max_batches 2`, i.e. the first 64 problems of the length-sorted order, bs 32, GPU)

| run | identical first-pass text | identical rethink text | same correctness (final) | accuracy new / original |
|---|---:|---:|---:|---:|
| GSM-Hard, plain + rethink | 64/64 | 64/64 | 64/64 | 48.44 / 48.44 |
| GSM8K test, calculator + rethink | 46/64 | 42/64 | 64/64 | 82.81 / 82.81 |

The calculator run does **not** match the saved original token for token on 18 of 64 problems (same final correctness on
all 64; first-pass correctness differs on 1). Every divergence starts at the token right after the calculator's
end-of-number veto write (e.g. `= 50\nSo Ali ...` vs `= 50\nAli ...`). To locate the cause:
* the repo's run is deterministic (a repeat gives 64/64 identical text);
* the **original research harness, run today** on the same machine from its own directory, gives output identical to
  the repo's on 64/64 (`gsm_test_calc_original_harness_rerun.log`);
* the circuit code, veto vector and data are identical to the original run's.

So the difference is between today's runs and the saved run from 2026-11-15, not between the repo and the original code.
I could not pin down the cause. The original's launch script ran up to four such jobs on one GPU at once, which may
have changed the GPU kernels chosen; this is a guess, not tested. The reported GSM numbers (full test sets) come from the saved run; they were not recomputed in full.

## 8. Rethink gate

Not re-generated (it needs vLLM). `results/gate/cost_curve.log` was recomputed with `scripts/gate/cost_curve.py` from the
saved generations and detector scores; it reproduces every number in the README table (52.08 / 55.27 @ 45 / 56.56 @ 101 /
58.30 @ 49; AUROC 0.81 and 0.83 in `detector_summary.json`). The gate as an implant: section 10.

## Changes after the runs

After the runs above, the test scripts `test_deduction.py`, `test_tracking.py`, `test_lies_bool_nav.py` and
`scripts/check_silent.py` got final `assert`s so that they exit non-zero on any wrong answer or mismatch (the logged runs
all satisfy them), unused imports were removed, and `data/prepare_gsm.py` now pins the Hugging Face dataset revisions
(re-run: byte-identical files).

## 9. Not re-run

* Full-size BBH (250 per task) and full GSM runs with the repo code: only the subsets above. The rethink implant: only
  the subsets in section 10 (no full GSM8K / GSM-Hard run with the implant, always or gated).
* vLLM baselines (Qwen3-0.6B, Qwen2.5-7B/14B): taken from the saved runs.
* `finetuned_phase/`: nothing re-run (see its README).
* Figures: regenerated with `figures/make_figs.py` in the venv from `results/` (DejaVu Sans; Avenir is optional).

## 10. The rethink implant (added after the release; same machine, GPU 0 shared with other jobs)

Before this, the rethink phrase was appended by `scripts/run_gsm.py --rethink` (evaluation code) and the probe gate was
applied by evaluation code. Now both are an implant, `micromachines/rethink.py` (`--rethink_implant always|gated`).

**How the reference works** (`--rethink`, unchanged, kept as the reference): pass 1 generates greedily until
`<|endoftext|>` or a blank line (checked every 4 tokens on the last 8, max 320 tokens) and cuts the text at the first
`\n\n`; pass 2 re-encodes `prompt + question + first.rstrip() + "\n" + "Hmm, wait. Let me rethink this.\n"` in a new
batch and generates again (same stop rule, max 360); the last `####` in pass 2 counts, else the first answer stands.
The gate (`scripts/gate/`) rethinks only if a logistic probe on the layer-28 state at the first `####` (a fresh forward
of the first answer) is at least a threshold chosen on GSM8K train.

**How the implant does it** (one generation, the harness only generates, stops and scores):
* trigger: a pre-hook on the host's final norm (after the calculator's hook, so it sees its write) applies the host's own
  norm + LM head to the last hidden state; if the host's next token would create the first blank line (any of the 565
  vocabulary tokens containing `\n\n`, a token starting with `\n` right after one ending with `\n`) or is
  `<|endoftext|>`, or the answer has reached 320 tokens, the implant fires (once per answer);
* writer: instead of that token it writes the tokens of `stem.rstrip() + "\n" + PHRASE` one per position (stem = the
  part of the end token before its blank line; for `\n\n` this is the 10 tokens `\n`, `Hmm`, `,`, ` wait`, `.`,
  ` Let`, ` me`, ` rethink`, ` this`, `.\n`), with solved write directions (407 possible sequences, 399 distinct
  tokens, smallest margin 13.2 logits on random states); then it is silent and the host continues;
* gate (`gated`): at the host's first `####`-containing token it computes the probe score on `norm(h)` (=
  `hidden_states[28]`) in float64 and arms the writer only if score >= thr. The probe weights are fitted (below).

**The probe (fitted).** The probe file behind section 8 was not kept, so `gen_rethink.py` was re-run (vLLM 0.10.2,
`NO_OCALC=1`, `UTIL=0.3`) and `scripts/gate/export_probe.py` refitted it with the same code and rule as
`fit_detector.py`. The new generations differ slightly from the originals (GSM8K test never / always rethink 52.01 /
56.10 vs 52.08 / 56.56; train 61.87 / 64.67). Refit: train AUROC 0.956 (original 0.955), threshold 0.2506 chosen on
train (original 0.3277), test AUROC 0.808 (original 0.810); evaluation-code gate on the new test generations 55.65%,
firing on 65.7% (original 56.18%, 58.2%). Saved as `results/rethink/rethink_probe_L28.pt` (w, b, mean/std, threshold).

**Equivalence runs.** GPU 0 was shared with up to eight other jobs and runs at batch 32 were killed by out-of-memory
errors, so the comparisons use batch size 8 (`--bs 8`, the first 224 problems of the length-sorted order, run as 7
chunks of 4 batches with `--start_batch`; chunks are independent, so this equals one run). bf16, greedy. Reference =
`--rethink` (evaluation code), implant = `--rethink_implant always|gated`, same batches. Compared with
`scripts/compare_rethink.py` (texts, answers) and a token check (the token sequence the implant run feeds the host up to
the end of the phrase vs the reference's re-encoding of `prompt + question + first.rstrip() + "\n" + PHRASE`).

| subset (224 problems, bs 8) | host tokens up to end of phrase identical | first answer text identical | rethink text identical | final answer identical | same correctness | accuracy implant / reference |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K test, plain, always | 224/224 | 224/224 | 161/224 | 212/224 | 215/224 | 73.66 / 74.11 |
| GSM-Hard, plain, always | 223/224 | 223/224 | 127/224 | 212/224 | 219/224 | 45.98 / 43.75 |
| GSM8K test, calculator `calc_s3` + always | 222/224 | 224/224 | 165/224 | 210/224 | 213/224 | 73.21 / 74.55 |
| GSM8K test, plain, gated (vs evaluation-code gate on the reference) | 219/224 gate decisions agree | 219/224 | 75/110 | 214/224 | 217/224 | 73.21 / 73.66 |

What differs and why:
* **Up to the end of the phrase the host sees the same tokens** in every problem but one: GSM-Hard #805, where the
  first answer hit the 320-token budget in the middle of a loop and its last token was a space. The evaluation code
  strips it (`out.rstrip()`) before appending `\n` + phrase; the implant cannot take back a token the model already
  emitted, so its text keeps the space. That answer has no `####` either way.
  With the calculator, 2 problems (#507, #1066) have the same text but different tokens before the phrase: the
  calculator implant writes ` ` and `-2` as separate tokens (`= -2`), and the script's re-encoding merges them into ` -`.
  The script's second pass therefore does not even see the calculator's own tokens; the implant run does.
* **The continuation after the phrase differs in 28-43% of problems, and this is bf16 arithmetic, not the implant.**
  The reference computes the phrase positions in a fresh prefill of the whole re-encoded text, in a new batch; the
  implant run computes them one token at a time in the running KV cache. Same tokens, different kernels, so bf16
  logits differ slightly and near-ties flip. Checks: (1) the same comparison in **float32** (GSM-Hard, first 32
  problems, bs 8; `--fp32`): rethink text identical **31/32**, final answer and correctness 32/32 (bf16 on the same
  32 problems: 20/32). The one fp32 difference is a tokenization case, not arithmetic: the model wrote `.` and then
  `\n\n` as two tokens; the implant replaces `\n\n` by `\n`, `Hmm`, ... while the script re-encodes the same text
  `statement.\nHmm` with the merged token `.\n`, so the two runs feed the model the same characters as different
  tokens. In the bf16 subsets the model always ended with a single token (`\n\n` or `.\n\n`), so this did not occur; (2) the evaluation code alone, run with a different batch size (bs 8 vs the saved
  bs 32 original), agrees with itself less than the implant agrees with it: GSM8K first answers 169/224 identical,
  rethink texts 113/190, same correctness 211/224, accuracy 74.11 vs 75.45; GSM-Hard first answers 155/224, same
  correctness 213/224, 43.75 vs 45.98. (At bs 32, today's reference reproduces the saved original GSM-Hard run on
  224/224 first answers and 157/157 stored rethink texts.)
* **Gate.** The implant computes the probe on the host's own state during generation (batched, incremental, bf16);
  the evaluation code recomputes it in a fresh unbatched forward. Scores differ by 0.0097 (median, max 0.070); 5 of 224
  decisions flip, all with the evaluation-code score just below the threshold 0.2506 (0.241-0.247) and the implant's
  just above (0.258-0.297). The implant rethought 115/224 problems, the evaluation-code gate 110/224.

Cost on the GSM8K subset (tokens generated after the first answer, phrase included for the implant): never rethink
68.75%; always (implant) 73.66% at 112 tokens/problem; gated (implant) 73.21% at 62 tokens/problem, firing on 115/224.

So the implant reproduces the evaluation-code rethink exactly at the token level up to the point where it hands
control back, and from there on differs only by floating-point noise of the same size as a batch-size change of the
evaluation code itself. The full-set numbers in `DETAILS.md` (56.10 / 57.62 / 29.26 / 37.23) were not re-run with the
implant (see below).

**Silent property** (`scripts/check_silent_rethink.py`; two copies of the same bf16 generation in lockstep, implant
hook active in one and bypassed in the other, same tokens): always-on, 32 GSM8K problems: 5,922 positions, implant
wrote at 320 (10 per problem), the other **5,602 logit rows bit-identical**; host weights hash the same before and after
attaching. Gated, 48 GSM8K problems: 6,216 positions; the probe was evaluated at 48 `####` positions (26 below the threshold, 22 above); the implant wrote at 220 positions (22 problems x 10); all **5,996 other logit rows bit-identical**, including all 48 probe positions and every position of the 26 problems where the gate stayed shut (`results/rethink/silent_check_rethink_{always,gated}.json`).

**Not run:** the full sets (GSM8K and GSM-Hard 1,319 each, with and without calculator, and the gated implant on all of
GSM8K). With GPU 0 shared, one 224-problem subset took about an hour; the full runs would have taken many hours each.


## 11. Fresh-clone CPU check of the blog tables and figures (2026-11-19, laptop, no GPU, no model)

`git clone -b release-prep` into an empty directory; Python 3.9 / numpy 2.0 / matplotlib 3.9 for the scripts below (the
code targets 3.12; every one of the repository's 396 `.py` files also compiles under Python 3.14). Every path that
`REPRODUCE.md` names exists in the clone, except files the listed commands create (`data/gsm8k/*`, `ifeval/data/ifeval.jsonl`,
`runs/veto_qwen3.pt`, `implant14_sd.pt`, `out/*`, fine-tuned host directories). Re-run from shipped files only:

| command | result |
|---|---|
| `python results/gsm_bigger/reanalysis/gsm_reanalysis.py` | md / json byte-identical to the shipped ones (after the fix that keeps the full-text-only lines) |
| `bbh_generic/integrate/code/report_v2.py ../results` | 12 tool tasks 43.27 -> 84.33 (FT alone 38.10), 27-task 40.98 -> 48.36 |
| `bbh_generic/transfer/code/report_transfer.py ...`, `curve_table.py` | 1.7B 52.08 -> 65.86 (96.37), 4B 71.14 -> 70.98 (97.77); curve 33.61 ... 48.36 |
| `bbh_generic/callgate/code/cg_report.py`, `cg_silence.py` | identical to `callgate/results/cg_report.txt`, `silence.txt` |
| `latent_write/summarize.py` | muladd / lost tables as in `latent_write/summary.md` |
| `rl/gsm_grpo/compile_abl.py`, `compile_long.py` | the 64-update and 300-update tables of `rl/README.md` |
| `rl/wiring/compile_results.py` (`RUNS=../results/wiring/{H100,A40}`) | `results.md` regenerated from the shipped logs |
| `evidence/code/agg_eyes.py evidence/eyes_sweep/results/summary.txt` | n=32: 94.74 vs 3.74; n=8: 78.53 vs 4.01 |
| `bash figures/blog/make_blog_figs.sh` (with the Avenir Next fonts in `figures/blog/fonts/`) | every committed static PNG identical pixel for pixel, except `25_rl_curves.png` (32 pixels) and `16_bbh27.png`, which was a stale bar version and is replaced by the dot plot the blog shows |

Not re-runnable on CPU from the clone: `rl/interp/make_fig*.py` and `analyze.py` (need the unreleased dev traces), the
animation score jobs (need token ids of the GSM generations), everything that loads a model.

## 12. Fresh-clone check of the release-prep branch (2026-11-20, laptop, no GPU, no model)

Clone: `git clone --no-local --branch release-prep` (history at the check: 37 commits, pack 62.8 MB; largest blobs are the shipped
gadgets `vecs2_forcefix.pt` and `detector_hybrid.pt`, 7.4 MB each).

| check | result |
|---|---|
| `py_compile` of every `.py` file (544, Python 3.14) | 0 failures |
| `bash -n` of every `.sh` file | 0 syntax errors |
| static import scan (every imported module is stdlib, a pinned / optional third-party package, or a file in the repo) | only external optional deps are missing: `evaluation_lib` (IFBench), `math_verify`, `latex2sympy2_extended`, `sympy` (MATH-500 grader), and `arms` (the unreleased v1 RL engine, `rl/gsm_grpo/prof_v1.py`) |
| `HF_HUB_OFFLINE=1 N=1000 python gsm_push/calc_v2/test_exact.py` (calc v2 tracker, tokenizer only, 8 s) | 0 wrong, 0 missed, 0 acted on silent / restated text, all 9 forms |
| `rl/figures/v3/make_rerender_v3.py`, `make_new_v3.py`, `make_rl_curves_all.py`, `make_rl_stack_curves.py`; `figures/blog/review_final/{bars_v2,ifeval_v2,bars_restyle,gsm_compose,latent_write,topic_blocker,make_rl4_toolbox}.py` | all run from the clone; every committed PNG and every `*_numbers.json` byte-identical (SVGs differ only in matplotlib ids) |

Not run here: circuit exactness tests (`tests/test_calc_circuit.py`, `bbh_generic/teamA/test_tools.py`; minutes of CPU),
`gsm_push/calc_v2/test_chain2.py` and the `sw_fuzz_hook.py` reader / voter fuzz (GPU), and everything that loads a model.

## 13. Fresh-clone check of the `gsm_stack` release (2026-11-21, laptop, no GPU, no model)

Clone: `git clone --no-local --branch release-prep` after the five `gsm_stack` commits. New material: `gsm_stack/`
(engine, eval, implants, results, figures; 7.9 MB, 643 files), `figures/blog/v6/` (2.4 MB), `latent_write/causal_lw/`.
No file in the new commits is larger than 0.3 MB; no `.pt`, `.safetensors`, `.jsonl.gz`, `.npz` or `.pkl` file was added.

| check | result |
|---|---|
| `compile()` of every `.py` file in the clone (832, Python 3.14) | 0 failures |
| `bash -n` of every tracked `.sh` file (194) | 0 syntax errors |
| static import scan of the new folders: every code tree built with `mk_tree.py` (7 named trees + 11 lane overlays) and every other new `.py` file; each import must be stdlib, a pinned / optional third-party package, or a module on that file's path | all resolve except `arms` / `data` in `engine/wiring/prof_v1.py` (the unreleased v1 engine profiler, as in §12) |
| overlay trees vs the original source trees (file by file, after the release path edits) | identical for all 18 trees, except `stk/KERNEL_NOTES.md` (release header added) and two deleted `*.rung4` backup copies |
| `mk_tree.py r2` / `r3`, then `test_aw.py` (adaptive tries vs the python references, 20,000 + 20,000 cases) | AW FUZZ PASS on both trees, 0 mismatches (10 s) |
| `implants/negative/m5div/aw/test_m5.py` on its tree | M5 FUZZ PASS (3,405 biased steps, 0 mismatches) |
| `implants/negative/dola/aw/test_dola.py` on its tree | DOLA FUZZ OK (1,853 rows, max abs p diff 7.6e-7) |
| `implants/negative/trunc/test_trunc.py` | OK (300 random cases) |
| `implants/negative/voter2/test_voter2.py --trials 2000` | PASS (0 pick / queue mismatches, both weightings) |
| `latent_write/causal_lw/fuzz_ctrig.py 2000` (tokenizer only) | exact match on 10,000 scans, 6,601 fires |
| `HF_HUB_OFFLINE=1 N=1000 python gsm_push/calc_v2/test_exact.py` | 0 wrong, 0 missed, 0 acted on silent / restated text |
| `rl/figures/v3/make_rl_stack_curves.py` with the final rlstack data | PNGs and `rlstack_numbers.json` byte-identical to the shipped renders |
| `figures/blog/v6`: `gsm8k/make_gsm8k_figs.py`, `diagram/make_serial_diagram.py`, `src/make_implants_figure_v6.py`, `src/rf/bbh_bars_v6.py`, `src/wsm_static_v6.py` (in a scratch copy) | all run; output differs only in the font (Avenir not shipped) |
| `gsm_stack/eval/results_tools/devtab.py` / `devcmp.py` on `results/curves/` | A1 u0 dev vote8 81.33 / 78.00 and CW_s0 / CW_s1 u0 51.67, as in `RL_RESULTS.md` |
| grep of the working tree and of every added line and message of the new commits for host names, user names, home paths, cluster paths, e-mail addresses and token patterns | none in the new material (pre-existing hits are `\n4` escapes in text dumps and "ssh" in old notes); see the note below on commit metadata |

Not run here: anything that needs triton or a GPU (`test_adapt_refs.py` imports triton; `wprk_fuzz.py`,
`sw_fuzz_hook.py`, `jobs/tattn_test.py`), the rung-3 / rung-4 register fuzz (`test_vf.py`, `test_sn.py`: they load the
unshipped `vf_w.pt` / `sn_w.pt`), `implants/vfeat/test_vfeat.py` (expects its job tree materialized in place), and
every model run.

## 14. Fresh-clone check of the final-blog update (2026-11-23, CPU only, no GPU, no model)

Clone: `git clone --no-local` of a `git add -A` snapshot of the release tree (what a commit of it contains). New material
(see `RELEASE_NOTES.md`): `uniform/` (582 files, 11.1 MB) and `checks/` (six folders, 324 files + 2 symlinks, 13.3 MB),
plus 4 small files elsewhere; 24.5 MB in all. 9 existing files changed (including this file), none removed. The largest new file is 2.9 MB
(`checks/calcimp/tt/res/train_steps.jsonl.gz`, per-update training scalars). Every `.jsonl.gz` holds per-item ids,
answers, correctness and token counts only (no generated text, token ids or log-likelihoods). The only new `.pt` files
are four gadgets of 10–23 KB that the code trees load.

| check | result |
|---|---|
| tracked files in the clone | 3,783 (175 MB without `.git`); nothing under `uniform/` or `checks/` is git-ignored; the 6 gadget `.pt` files re-included in `.gitignore` (incl. `gsm_stack/gadgets/{vf_w,sn_w}.pt`) are tracked |
| `py_compile` of every `.py` file (Python 3.12.3) | 1,078 / 1,078 (192 in `uniform/` + `checks/`); only pre-existing `invalid escape '\s'` warnings in base copies of `serial_stack.py` / `serial_engine.py` |
| `bash -n` of every `.sh` file | 253 / 253 (54 in `uniform/` + `checks/`) |
| static import scan of `uniform/` and `checks/` (192 files; stdlib, pinned packages, same dir / parents, `sys.path` edits, `mk_tree` `TREE_BASE` chains, `setup.sh` / `mk_code.sh` assembly) | all resolve except `arms` in `uniform/rl28/gsm8k/trees/*/wiring/prof_v1.py` (the unreleased v1 profiler, as in §12 / §13; no documented command uses it) |
| §20 `checks/tooltext/ana_tt.py --write` | byte-identical (`rows.json`, `res/ana_tt_12seeds.txt`), 2 min |
| §21 `checks/calcimp/ana38.py --write`, `--all --write`, `--all --sens --write` | byte-identical (`rows38*.json`, `res/ana38_*.txt`), 6 / 7.5 / 8.5 min |
| §21 `sel300.py 8` / `20`, `ana_sweep.py`, `links38.py` (`fI` + `fItok`, 8 seeds; `fI`, 20 seeds) | byte-identical to `res/` |
| §22 `checks/seeds35/ana35.py --stamp 2026-11-23T10:40Z`; `ifana/analyze6.py` (s0-5, s012), `perseed.py`, `nltk_check.py` | byte-identical (`AUTO_final.md`, `table_s0-5.md`, `check012.md`, `rows_s*.json`, `perseed.txt`, `nltk_check.md`) |
| §23 `checks/seeds3/{try8,summarize,libpair,mixpair}.py` | byte-identical to the shipped `.md` (numpy 2.4 for `mixpair.py`) |
| §24 `checks/offtask/ana_ot.py` (0.6B and `OT_SIZE=17`) | byte-identical (`ana06.md`, `ana17.md`, `rows06.json`, `rows17.json`), 7 min each |
| §25 `checks/sourced` (a)–(f) | byte-identical to the `*.out.md` files, ≤ 13 s each |
| §25 (g) `nv32.py` | `datasets` not installed, so `data/prepare_gsm.py` was not run; with GSM8K files from an earlier run (all 5 match `code_data.sha256`) the output is byte-identical to `nv32_dev36.out.md` |
| §17 `make_uniform_json.py`, `q33/deltas.py`, `rl24/ifeval/analyze.py` + `analyze29.py`, `rl24/bbh/w/final_ana.sh`, `q32/bbh_pertask.py` | byte-identical (`uniform.json`, `deltas.txt`, `RESULTS_TABLE.md`, `rows.json`, `table_final.md`, `bbh_pertask.md`) |
| §17 `ana_uni.py`, `q32/ana_q32.py` (all 17 modes), `q33/decomp.py`, `mech_2b.py` (print only) | every effect and CI in §17 that these print matches REPRODUCE, after one stale CI was corrected (§17.6 "3 − 1-MV3"). The two C14 CIs of §17.1 come from GPU eval rows that are not in `items/` |
| §18 `rl28/gsm8k/mkrows.py`, `rl28/ifeval/analyze.py`, `rl28/bbh/w/final_ana.sh`, `pertask.py` | byte-identical (`rows.json`, `RESULTS_TABLE.md`, `table_final.md`) |
| §18 `rl28/gsm8k/mkresults.py` | `RESULTS.md` identical except its `Generated … at` wall-clock line |
| §18 `ana17.py`, `fe_gp/ana_v17.py`, `fe_gp/ana_gp17.py` (print only) | every §18 CI matches, after two stale GSM-Hard CIs in the §18.1 effect table were corrected to the `RESULTS.md` values |
| rung-3 / rung-4 register fuzz (`test_vf.py`, `test_sn.py`) with the now-tracked `vf_w.pt` / `sn_w.pt`: `mk_tree.py` trees `rungs/rung3`, `rungs/rung4`, gadgets copied in as `gs_tree` does, the test (+ `sidenet/aw/vf_learn_ref.py`) placed at the tree root, `sidenet/repick_sn.py` and `vfrefit/off/refit.py` on the path | VF FUZZ PASS on both trees (20,000 picks, 0 mismatches; 5.5 / 6 min); SN FUZZ PASS on rung 4 (20,000 trials, 0 mismatches, 0 near ties; 11.5 min) |
| grep for host / user names, home and cluster paths, queue tools, e-mail addresses and token patterns in `uniform/`, `checks/`, `README.md`, `REPRODUCE.md`, including every decompressed `.gz` | none. The only matches are false positives: the variable `gq` in `fasttrain.py`, and the byte string `gcr9` inside compressed data that is not present once decompressed |

Fixed during this check (docs and two launch defaults; no numbers in tables computed from data changed):
- the three stale CIs above;
- REPRODUCE §21 now gives both `links38.py` runs that make `res/links38.txt`;
- REPRODUCE §17.8 now redirects `bbh_pertask.py` (it prints to stdout);
- the GSM-Plus path in `checks/{tooltext,calcimp/tt}/mk_code.sh` (default `GSMPLUS`), their READMEs, REPRODUCE §20 and three
  `uniform/final_eval/lists/gp_*.list` comments now points to `uniform/final_eval/data/prepare_gsmplus.py`, which writes
  `$WSM_ROOT/data/gsmplus/testmini.jsonl`.

The §13 note that `vf_w.pt` / `sn_w.pt` are unshipped no longer applies: they are in `gsm_stack/gadgets/` and tracked.

Not run here: every GPU or model run (all training, every eval list, `vocabjob` of §25 (h)), `data/prepare_gsm.py` and
`prepare_gsmplus.py` (downloads), the scripts that need generated text (not shipped), and `uniform/rl28/gsm8k/trees/*/wiring/prof_v1.py`.
