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

The GSM8K / GSM-Hard numbers in the README (51.93 / 54.06 / 56.10 / 57.62 and 26.99 / 34.50 / 29.26 / 37.23) were
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
58.30 @ 49; AUROC 0.81 and 0.83 in `detector_summary.json`).

## Changes after the runs

After the runs above, the test scripts `test_deduction.py`, `test_tracking.py`, `test_lies_bool_nav.py` and
`scripts/check_silent.py` got final `assert`s so that they exit non-zero on any wrong answer or mismatch (the logged runs
all satisfy them), unused imports were removed, and `data/prepare_gsm.py` now pins the Hugging Face dataset revisions
(re-run: byte-identical files).

## 9. Not re-run

* Full-size BBH (250 per task) and full GSM runs with the repo code: only the subsets above.
* vLLM baselines (Qwen3-0.6B, Qwen2.5-7B/14B): taken from the saved runs.
* `finetuned_phase/`: nothing re-run (see its README).
* Figures: regenerated with `figures/make_figs.py` in the venv from `results/` (DejaVu Sans; Avenir is optional).
