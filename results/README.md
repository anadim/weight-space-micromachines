# Result files

Small files behind every number in the top-level `DETAILS.md`. `python results/summarize.py` rebuilds the README tables
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
| `gsm_full_implant/{test,hard}_{plain,calc}.json` | Qwen3-0.6B-Base, GSM8K test / GSM-Hard, without / with `calc_s3` (in weights), repo harness, max 320 new tokens | HF loop (`scripts/run_gsm.py`), reduced |
| `gsm_bigger/{q17,q4}_{test,hard}_{plain,calc}.json` | the same on Qwen3-1.7B-Base (`q17`) and Qwen3-4B-Base (`q4`) | HF loop, reduced |
| `gsm_bigger/veto_q17.pt`, `veto_q4.pt` (+ `.log`) | the calculator's "no more digits" veto re-solved for 1.7B / 4B (`scripts/solve_veto.py --model ... --device cuda`) | solved tensor |
| `gsm_bigger/RESULT_lines.txt` | the final line of each 1.7B / 4B run log | |
| `rethink_circuit/full/*.json` | compiled rethink (`micromachines/rethink_circuit.py`) always / gated, plain / calc, GSM8K test and GSM-Hard, as batch-aligned shards of the length-sorted order (concatenate by `i`); gated GSM-Hard is partial | HF loop (`scripts/run_gsm_circuit.py`), reduced |
| `rethink_circuit/write_dirs.pt` | the circuit's solved write directions (`scripts/rethink_circuit_build.py`) | solved tensor |
| `rethink_circuit/summary.json` | `scripts/summarize_rethink_circuit.py` over the above vs `gsm_full_implant/` (its `equivalence_vs_old_python_implant` block came from a 128-problem comparison with `micromachines/rethink.py` whose full outputs are not shipped) | |
| `rethink_circuit/check_{always,gated}_partial.txt` | `scripts/check_rethink_circuit.py` counts on 32 problems (silent positions, detector agreement, forced rows) | |
| `rethink_v2/final_eval/*.json` | the in-weights GSM table: arms `none`, `alwaysA` (compiled rethink after the answer), `hybrid` (compiled mistake detector + rethink), plain / calc, GSM8K test / GSM-Hard, two shards each (`_s0` = batches 0-20, `_s1` = the rest); commands in `scripts/rethink_v2/jobs_{baselines,hybrid}.txt` | HF loop (`scripts/rethink_v2/run_gsm_v2.py`), reduced |
| `rethink_v2/summary.json` | `scripts/rethink_v2/summarize_v2.py results/rethink_v2/final_eval` (accuracy, extra tokens, fire rates, paired bootstrap CIs) | |
| `rethink_v2/detector_hybrid.pt` | the FITTED mistake detector (MLP over layers 12-24 at the token before the first `####`, three heads, thresholds), `scripts/rethink_v2/policy3.py` | fitted on GSM8K train only |
| `rethink_v2/write_dirs_all.pt` | the v2 implant's solved write directions | solved tensor |
| `rethink_v2/auroc_{test,hard}.json` | detector AUROC on the plain arm's own answers (`scripts/rethink_v2/auroc_eval.py`), run once at the end | |
| `rethink_v2/trainval/*.json` | the same three arms on held-out TRAIN problems (`rep` split; jobs in `scripts/rethink_v2/jobs_trainval.txt`) | HF loop, reduced |
| `rethink_v2/{pol_gsm,pol_all,pol2,pol3,sweep_gsm}/*.json` | train-side detector and policy selection reports (fit 70 / sel 15 / rep 15 by source problem) | `scripts/rethink_v2/{fit_detector,policy,policy2,policy3}.py` |
| `rethink_v2/silent_check_hybrid.json`, `check_smoke_hyb.json` | bit-identity wherever the v2 implant does not write; host hash unchanged | `scripts/rethink_v2/check_v2.py` |
| `implant_sizes.json` | ReLU units, layers, nonzero weights per implant | `scripts/build_implants.py` |
| `silent_check.json` | logits bit-identity with all nine implants attached | `scripts/check_silent.py` |
| `verify/` | the re-runs from a fresh clone described in `VERIFY.md` (logs, comparisons, compact outputs) | |

All model runs: greedy decoding, bf16 on one GPU, batch size 16 (BBH) or 32 (GSM).

"Reduced" files were made by `scripts/reduce_outputs.py` from the full run outputs: per problem the index, gold,
prediction, correctness, token count, and the post-hoc fields the summary scripts need (`ph_hits` = token positions of the
rethink phrase, `first_ok`, `ok_last`, `fire_b` / `fire_a`), plus `out_sha256` (SHA-256 of the generated text); the
generated text, token ids and question text are removed. `source_sha256` is the SHA-256 of the full output and `command`
the command that produced it (for sharded runs: the same command without the shard arguments gives the same batches).
The summary scripts give identical numbers on the full and the reduced files (checked).

The implant runs and the HF-loop baselines were produced by the research harness `bbh_official.py`, of which
`scripts/run_bbh.py` is a cleaned-up copy (same decoding loop, prompts, stop rule and answer extraction; `VERIFY.md`
shows identical outputs). Each file's `command` gives the equivalent `scripts/run_bbh.py` call.
