# Result files

Small files behind every number in the top-level README. `python results/summarize.py` rebuilds the README tables
from them. The original run outputs also held the full generated text per question; here each question keeps only its
task, gold label, extracted prediction and correctness (plus token counts where available), and each file records the
SHA-256 of the original output it was reduced from (`source_sha256`) and the command that produced it (`command`).
No question text is stored.

| file | what | harness |
|---|---|---|
| `bbh/qwen3_base_vllm.json` | Qwen3-0.6B-Base, no implant, all 27 BBH tasks | vLLM 0.10.2, `min_tokens` 4 |
| `bbh/qwen25_7b_vllm.json` | Qwen2.5-7B, 27 tasks (5 tasks invalid: empty outputs) | vLLM, `min_tokens` 0 |
| `bbh/qwen25_7b_vllm_mintok4.json` | Qwen2.5-7B, rerun of 8 tasks; supersedes the file above for them | vLLM, `min_tokens` 4 |
| `bbh/qwen25_14b_vllm.json` | Qwen2.5-14B, 27 tasks | vLLM, `min_tokens` 0 |
| `bbh/qwen3_implants_arith.json` | + `calc_s4`, `sum_fn`, `expr_n`: multistep arithmetic, object counting | HF loop (`scripts/run_bbh.py`) |
| `bbh/qwen3_implants_lies_bool_navv1.json` | + `liesnat`, `boolnat`, and navigate v1 (superseded) | HF loop |
| `bbh/qwen3_implants_navigate.json` | + `navnat` (v2, the one in this repo) | HF loop |
| `bbh/qwen3_implants_temporal_dyck.json` | + `time_n` (and a Dyck implant not in this repo; Dyck is not in the 12 tasks) | HF loop |
| `bbh/qwen3_implants_tracking.json` | + `track_nat`, tracking 3/5/7 | HF loop |
| `bbh/qwen3_implants_deduction.json` | + `logic_nat`, logical deduction 3/5/7 | HF loop |
| `bbh/qwen3_base_hf_deduction.json` | no implant, logical deduction 3/5/7 (same-harness baseline) | HF loop |
| `bbh/qwen3_base_hf.json` | Qwen3-0.6B-Base, no implant, all 27 tasks (a separate full run) | HF loop |
| `bbh/all_nine_mech.log`, `bbh/all_nine_other.log` | all nine implants attached at once, implant tasks / other tasks: log snapshots (PARTIAL) with sha256 | HF loop |
| `gsm/qwen3_{test,hard}_{plain,calc}_rethink.json` | GSM8K test / GSM-Hard, with and without the calculator; `first_ok` before and `ok` after the rethink pass | HF loop (`scripts/run_gsm.py`) |
| `gate/cost_curve.log` | accuracy vs extra tokens for never / always / probe-gated / margin-gated / perfect rethink | from vLLM generations (`scripts/gate/`) |
| `gate/detector_summary.json` | detector AUROCs (train, test, GSM-Hard) and gated accuracies, per-problem scores removed | `scripts/gate/fit_detector.py` |
| `rethink/rethink_probe_L28.pt` | the FITTED probe of the gated `rethink` implant (w, b, mean/std, threshold, fit report) | `scripts/gate/export_probe.py` on a fresh `gen_rethink.py` run (`rethink/gate_gen_summary.txt`) |
| `rethink/eq_{test,hard}_plain_{ref,imp}.json`, `rethink/eq_test_calc_{ref,imp}.json`, `rethink/eq_test_plain_gated.json` | equivalence subsets (first 224 of the length-sorted order, bs 8): `ref` = evaluation-code rethink (`--rethink`), `imp` / `gated` = the `rethink` implant; per problem gold, first prediction, correctness before/after, implant wrote or not, probe score, extra tokens | HF loop (`scripts/run_gsm.py`) |
| `rethink/eq_test_plain_evalgate.json` | the evaluation-code gate (probe scores and decisions) on the reference's first answers | `scripts/gate/eval_gate.py` |
| `rethink/fp32_hard_plain_{ref,imp}.json` | the same comparison in float32, first 32 GSM-Hard problems | HF loop, `--fp32` |
| `rethink/compare_rethink.txt` | problem-by-problem comparison output (texts, answers, token sequences) for all of the above | `scripts/compare_rethink.py` + token check |
| `rethink/silent_check_rethink_{always,gated}.json` | logit bit-identity wherever the rethink implant does not write | `scripts/check_silent_rethink.py` |
| `implant_sizes.json` | ReLU units, layers, nonzero weights per implant | `scripts/build_implants.py` |
| `silent_check.json` | logits bit-identity with all nine implants attached | `scripts/check_silent.py` |
| `verify/` | the re-runs from a fresh clone described in `VERIFY.md` (logs, comparisons, compact outputs) | |

All model runs: greedy decoding, bf16 on one GPU, batch size 16 (BBH) or 32 (GSM).

The implant runs and the HF-loop baselines were produced by the research harness `bbh_official.py`, of which
`scripts/run_bbh.py` is a cleaned-up copy (same decoding loop, prompts, stop rule and answer extraction; `VERIFY.md`
shows identical outputs). Each file's `command` gives the equivalent `scripts/run_bbh.py` call.
