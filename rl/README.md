# RL with implants attached (GSM8K, GRPO, Qwen3-0.6B-Base)

The blog section "Can RL optimize how a model uses implants and how they're wired in?". The host is trained with GRPO
while an implant is attached inside the forward pass. Arms:

| arm | what is attached during RL | trainable |
|---|---|---|
| 1 | nothing | host |
| 2 | the GSM calculator `calc_s4` + veto, frozen, hand-wired (write before the final norm) | host |
| R | `calc_s4` with the 10 digit write rows permuted by a fixed derangement (seed 12345): same reader, stacks, trigger, gain and write layer; fires at exactly the same places, writes wrong digits ("broken calculator", the "just more weights" control) | host |
| 4 | arm 2 + the compiled mistake detector / rethink (`micromachines/rethink_circuit_v2.py`, `results/rethink_v2/detector_hybrid.pt`), frozen | host |
| 4c / 4cRL | arm 4, detector MLP trained during RL (policy gradient on its fire decisions, + BCE refit for 4c) | host + detector |
| 4s | arm 4c with the rethink phrase replaced by trainable soft vectors (init = phrase embeddings) | host + detector + 9 vectors |
| W1, W2, W2w, W2ws, W3e, W3r, W3z, W4 (`wiring/`) | different wirings of the same calculator: trainable writer, zero/embedding/random-init links into layers 7/14/21, warm-started final write | host + wiring |
| CW, CWlean, CWout, CWin, CWinR/P/O/C (`wiring/`) | "co-designed" wiring: trainable out-adapters and a trainable in-wire that can veto each calculator call | host + wiring |

The implant itself stays exact in every arm (the circuit is frozen; writer exactness is logged where the writer trains).

## Layout

* `gsm_grpo/` — the RL engine and trainer (arms 1, 2, R, 4, 4c, 4cRL, 4s).
  * `engine.py` — rollout engine: shared 8-shot prefix KV computed once per weight version; static suffix KV cache;
    the whole decode step (implant step + 28 host layers + LM head + sampling + stop rule) captured as one CUDA graph
    per row bucket; finished rows compacted. `fastqwen.py` (Qwen3 forward with packed training pass), `hostkern.py`
    (Triton host kernels).
  * `implant.py` — `calc_s4` + veto driven with static buffers; same math as `micromachines.machine.Micromachine`.
    `implant_persist.py` — the whole calculator step as **one persistent Triton kernel** (bit-identical to the reference
    step, `test_fused.py`); `implant_triton.py` — a slower bit-identical variant.
  * `rethink_step.py` — `rethink_circuit_v2` rewritten as a one-token decode step (uses the `idx_A/idx_N/idx_H` lists kept on `RethinkCircuitV2`).
  * `train2.py` — GRPO trainer (Dr.GRPO loss, packed micro-batches, dev eval implant on and off).
  * `final_eval.py` — the one-shot test eval (GSM8K test + GSM-Hard, implant on / off, per-question 0/1 records).
  * `common.py` — data, prompt and scoring, identical to `scripts/run_gsm.py`; GSM8K TRAIN is split by
    md5(question) % 10: A = 0..5 (RL pool), B = 6,7 (dev), C = 8,9 (locked, unused).
  * checks and benchmarks: `check_equiv.py` (engine vs HF + hook, and `--bench` throughput), `check_rethink.py`,
    `bench_step.py`, `time_implant.py`, `test_fused.py`, `diag_exact.py` (calculator exactness after RL), `det_audit.py`
    (is RL gaming the detector?), `end_check.py`, `probe_rand.py`; `prof_v1.py` profiles the first (unreleased) HF-loop
    engine and needs it.
  * summaries: `summ_runs.py`, `summ4.py`, `compile_abl.py` (64-update table), `compile_long.py` (300-update table +
    paired bootstrap). These two read only `results/` and run on a laptop.
  * launchers, as run: `sweep.sh`, `abl.sh`, `long.sh`, `run4c.sh`, `final_pod.sh`, `h100_seq*.sh`, `a100_seq*.sh`,
    `duo_seq*.sh` with run lists `q_*.txt` / `qf_*.txt` (`name args` per line).
  * `RESULTS_ledger.md` — the experiment's own running log (all tables below come from it).
* `wiring/` — the wiring study: `train_wiring.py` (all W*/CW* arms), its own `engine.py` / `fastqwen.py` /
  `final_eval.py` (links take `(feat, x)`), `gen_traces.py`, `compile_results.py`, `make_fig.py`, launchers
  (`launch.sh` + `runs_*.txt`, `queue_run.sh`, `queue_a40.sh`, `cap300.sh`, `handoff_*.sh`), `NOTEBOOK.md` (setup,
  arms, every result). Shared modules are imported from `../gsm_grpo`.
* `interp/` — how RL changes the use of the calculator (text analysis of greedy dev traces): `dump_traces.py` (GPU),
  `run_all.sh`, `run_R.sh`, `run_aR_h100.sh`, `eqparse.py`, `analyze.py`, `fmt_probe.py`, `aR_analysis.py`,
  `causes.py`, `excerpts.py`, `show.py`, `make_fig.py`; outputs `stats.json`, `stats_aR.json`, `causes.json`,
  `analysis_out.txt`, `fig_rl_calc_use.png`; `NOTEBOOK.md` has every table.
* `results/` — small result files of every run behind the tables (no checkpoints, no generated text):
  * `gsm_grpo/{H100,A100,A40}/` — per run `config.json` + `steps.jsonl` (train reward, dev curves implant on/off,
    writes/calls per answer, detector stats) under `sweep/`, `abl/` (64 updates), `long/` (300-500 updates);
    `final/*.json` (test evals: accuracy, 201+ slice, format rate, length, per-question 0/1); `det/*.json` (detector
    audit); engine checks (`check_equiv*.json`, `check_rethink_fp32.json`, `clean/`, `time_implant*.json`,
    `bench_step.json`, `test_persist*.json`, `prof_v1_a100.json`). The GPU is part of the path because bf16 kernels
    differ by GPU (about 1 pp at u0): compare within a GPU.
  * `gsm_grpo/summary/` — `summ_{A100,H100,A40}.txt` (summ_runs.py output), `table.md` (64-update table),
    `table_u300_H100.md`, `paired.txt` (paired bootstrap, 64 updates).
  * `wiring/{H100,A40}/` — per run `config.json` + `steps.jsonl`, `results.md` / `results.json` (`compile_results.py`);
    `wiring/figdata/` — the logs `make_fig.py` reads.
* `micromachines/rethink_circuit_v2.py` keeps three index lists (`idx_A`, `idx_N`, `idx_H`) that `rethink_step.py` needs
  for arms 4/4c/4s; this one-line addition (committed with this folder) changes no computation.

## Setup

Same environment as the RL runs: Python 3.12.3, torch 2.8.0 (CUDA 12.8 / 12.9 wheels), triton 3.4.0, transformers
4.56.2, one GPU (H100 80GB, A100 80GB PCIe or A40 48GB; the A40 needs `FLEX_SMALL=1`, set automatically by the
launchers, and `--fused 0` for some wiring arms). Triton needs the Python headers (`Python.h`) to build its launcher;
put your venv and headers in `gsm_grpo/env_gcr.sh` if needed.

```
pip install -r requirements.txt triton==3.4.0
python data/prepare_gsm.py                                              # GSM8K train/test, GSM-Hard, the 8 shots
python scripts/rethink_v2/make_scaled_train.py data/gsm8k/train.jsonl data/gsm8k/train_scaled.jsonl   # scaled-number train set (seeded, rng 1234)
python scripts/solve_veto.py                                            # runs/veto_qwen3.pt (calc_s4 veto; bit-identical rebuild, VERIFY.md)
```
All code reads the repo root from `WSM_ROOT` (default: two levels above the script), the veto from `WSM_VETO`
(default `runs/veto_qwen3.pt`), the detector and rethink write directions from `results/rethink_v2/`. Run scripts from
their own directory (`cd rl/gsm_grpo`); run dirs are written under `runs/` there.

## Commands

Recipe for every main arm (picked by the sweep below): lr 1e-6 (AdamW on the host), G 8 samples per prompt, 64
prompts per update, temperature 1.0, Dr.GRPO loss, no KL, 5 warm-up updates, max 320 new tokens (700 for arms 4*),
dev = split B, n = 800 (400 GSM8K-train + 400 scaled), greedy, implant on and off.

```
cd rl/gsm_grpo
# HP sweep (arm 2, 32,768 rollouts per run; H100 list and A100 list)
bash sweep.sh q_h100.txt; bash sweep.sh q_a100.txt
# 64-update ablation, arms 1 / 2 / R, seeds 0 and 1 (q_abl_a100.txt + q_abl_a100c.txt are the A100 lists)
bash abl.sh q_abl_a100.txt; bash abl.sh q_abl_a100c.txt
#   = python train2.py --arm {1,2,R} [--rand digits] --seed {0,1} --lr 1e-6 --run_dir runs/abl/<name> --updates 64 \
#       --prompts 64 --G 8 --T 1.0 --dev_n 800 --dev_every 16 --ckpt_every 64 --fused 1
# 300-update runs (main result; the H100 runs): arms 1, 2 (seeds 0, 1), 4 (seeds 0, 1)
python train2.py --arm 2 --seed 0 --run_dir runs/long/a2_lr1e-6_s0 --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 \
    --dev_n 800 --dev_every 25 --ckpt_every 100 --fused 1
python train2.py --arm 4 --seed 0 --run_dir runs/long/a4_u300_s0 --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 \
    --dev_n 800 --dev_every 25 --ckpt_every 100 --fused 1 --max_new 700 --S_max 912 --tok_budget 8192
bash run4c.sh a4c_u300_s0 4c 0 512        # 4c / 4cRL: <name> <arm> <seed> <B_max>; 4s (soft rethink): see h100_seq7.sh, h100_4s_s1.sh
# one-shot test evals (lines "name ckpt_dir|base arm [rand]"), writes runs/final/<name>.json
bash final_pod.sh qf_a100.txt
#   = python final_eval.py --ckpt runs/abl/a2_lr1e-6_s0/step_000064 --arm 2 --out runs/final/a2_lr1e-6_s0.json
# tables from the released files (no GPU)
python compile_abl.py; python compile_long.py; python summ_runs.py '../results/gsm_grpo/A100/abl/*'
# engine checks
python check_equiv.py            # fp32 engine vs HF fp32 + hook (64 dev problems, plain and calc_s4); --bench for tok/s
python test_fused.py; python time_implant.py; python bench_step.py
cd ../wiring
bash launch.sh 0 2 40000 "" runs_h100_gpu1.txt     # all W*/CW* arms: python train_wiring.py --arm <ARM> --seed <S> \
    # --run_dir runs/L300_<tag> --updates 300 --dev_every 25 --dev_extra 64 --dev_n 800 --B_max 128 --tok_budget 4096 \
    # --ckpt_every 100 --eval_all 1 [arm-specific args from runs_*.txt]
RUNS=../results/wiring/H100 python compile_results.py; python make_fig.py
cd ../interp
bash run_all.sh; bash run_R.sh; bash run_aR_h100.sh   # greedy dev traces of the 64-update checkpoints (GPU)
python analyze.py; python aR_analysis.py; python causes.py; python make_fig.py
```
Runtime (wall clock from `steps.jsonl`): an update (64 prompts x 8 rollouts + packed training step) takes ~7.5 s on the
H100 and ~15 s on the A100 with arms 1/2; a 64-update arm-2 run with 5 dev-800 evals ~16 min on the A100 (~28 min on the
A40); a 300-update arm-2 run with 13 dev evals ~35 min on the H100, arm 4 ~53 min; one final eval (2 x 1,319 questions,
on and off) a few minutes.

## Results

All numbers are in `gsm_grpo/RESULTS_ledger.md` (sections 1-5), `wiring/NOTEBOOK.md` and `interp/NOTEBOOK.md`; the
files they come from are in `results/`. Dev = split B, n = 800 (SE ~1.7 pp); test = GSM8K test and GSM-Hard, read once
per checkpoint.

**64 updates, A100, the blog's numbers** (`results/gsm_grpo/summary/table.md`, from `results/gsm_grpo/A100/abl/` and
`.../A100/final/`):

| run | dev @0 / 32 / 64 (implant on) | implant-off dev @0 / 64 | writes / calls per answer @0 -> @64 | TEST GSM8K on / off | GSM-Hard on / off |
|---|---|---|---|---|---|
| arm 1 s0 / s1 | 43.6 / 51.6 / 54.5 ; 43.6 / 50.0 / 53.9 | = on | 0 | — / 65.0 ; — / 65.1 | — / 36.7 ; — / 37.9 |
| arm 2 s0 / s1 | 48.9 / 58.0 / 65.2 ; 48.9 / 57.4 / 63.8 | 43.6 / 56.2 ; 43.6 / 55.5 | 15.5 / 2.43 -> 17.4 / 2.75 (s0) | 67.2 / 66.4 ; 66.6 / 66.1 | 47.6 / 38.8 ; 47.4 / 37.1 |
| arm R s0 / s1 | 7.5 / 27.4 / 36.6 ; 7.5 / 29.0 / 37.9 | 43.6 / 44.8 ; 43.6 / 45.2 | 18.9 / 2.47 -> 2.1 / 0.29 | 48.6 / 58.8 ; 47.1 / 56.2 | 26.2 / 31.2 ; 26.3 / 30.8 |

Blog sentence -> source (an earlier draft of the blog quoted these 64-update numbers; the current blog's RL claims, most of
them 300-update, are mapped claim by claim in `REPRODUCE.md` section 12):
* "plain RL 44% -> 54%", "with the calculator 49% -> 64%", "seeds agree within a point and a half": the dev columns above
  (arm 1 43.6 -> 54.5 / 53.9; arm 2 48.9 -> 65.2 / 63.8).
* "almost all of that edge is on the big-number problems": `interp/NOTEBOOK.md` Q1 (switching the calculator on moves
  0-4 GSM-half problems and +44..+69 scaled-half problems); test GSM-Hard +10.2 pp vs +1.9 pp on GSM8K (`paired.txt`).
* "cut the unit-word style from 30% to 17%", "copies its answer 99%": `interp/NOTEBOOK.md` Q2 (`stats.json`): unit-word
  equations 29.8% (base) -> 23.5% (plain RL) -> 16.7% (arm 2); result exact when called 99.0 / 99.2 / 99.4%.
* "switch the calculator off afterwards ... 56% vs 54%": dev implant-off arm 2 56.2 / 55.5 vs arm 1 54.5 / 53.9 (traces:
  56.1 vs 53.1, `interp` Q3).
* "broken calculator starts at 7%, climbs back to 37%": arm R dev 7.5 -> 36.6 / 37.9 (A100), 7.0 -> 36.9 / 36.4 (H100).
* "firing on 75% of equations to 12%", unit word after every number: `interp/NOTEBOOK.md` Q4 (`stats_aR.json`, H100):
  74.6% -> 12.1% (seed 1: 11.2%), unit-word equations 29% -> 86% / 89%, calls per answer 2.50 -> 0.39 / 0.32.
* "about 25x faster than a HuggingFace generation loop on the same GPU": **not supported as stated.** Measured
  (RESULTS_ledger §0-1): the v1 HF loop (`model.model` + DynamicCache) 207 tok/s and HF `.generate` 288 tok/s at 16 rows
  on the A100; the v2 engine 19,800 tok/s with the implant at 512 rows on the H100 (~70-96x, different GPUs and batch
  sizes); single stream 319 tok/s (v2, H100) vs 24 tok/s (v1, A100). No same-GPU, same-batch HF comparison was run.

**300 updates, H100, two seeds** (`results/gsm_grpo/summary/table_u300_H100.md`, `compile_long.py`):
GSM8K test arm 1 68.2 / 70.1, arm 2 (on) 68.8 / 70.7, arm 4 (on) 69.5 / 66.0; GSM-Hard arm 1 42.4 / 43.7, arm 2 54.7 /
56.0, arm 4 56.8 / 56.6. Paired (seed-averaged): GSM8K arm 2 - arm 1 +0.57 [-1.06, +2.16]; GSM-Hard +12.24
[+10.12, +14.29]; arm 2 implant off - arm 1 on GSM-Hard -3.30 [-4.93, -1.78]; arm 4 - arm 2 on GSM8K -2.01
[-3.71, -0.38]. RL switches the frozen rethink almost off (fires on 68% -> ~4% of answers).

**Wiring** (`wiring/NOTEBOOK.md`, `results/wiring/*/results.md`, dev-800 at u300, seed 0 unless noted): fixed wiring
(arm 2 ref) 70.0; W2 (fixed write + zero-init links) 72.0, links stay ~0; W2w (trainable warm write) 70.1, the
saturated write gets no gradient; W1 (trainable writer, A40) 68.9, exactness 1.000; W3e / W3r (no write, links only)
63.9 / 61.1, no organic discovery; CW-full 67.5, CWlean 66.2 (in-wire: over-veto -> size-selective -> always call);
CWinR (broken calculator + in-wire) vetoes 100% of calls from u25 and reaches 58.9 at u100 vs arm R 36.9 at u64.

**Seed noise.** Two seeds per main arm; the two seeds differ by <= 1.4 pp on dev and <= 0.6 pp on test at 64 updates
(arms 1/2) and by up to 3.5 pp on test at 300 updates (arm 4: 69.5 vs 66.0). Paired-bootstrap CIs are over questions
with seeds averaged, so they do not include seed variance. Wiring arms and arms 4c/4s are mostly one seed; differences
under ~3 pp on dev are within noise. bf16 kernels differ by GPU and by `B_max` (~1 pp at u0): compare within a GPU.

## Disclosures

* The rethink arms read the host state through a copy of its LM-head rows (as in `micromachines/rethink_circuit_v2.py`);
  in RL, tokens the rethink forces are treated as environment (advantage 0).
* Arm 4 uses `calc_s4`; the repo's frozen "calculator + gated rethink" GSM row used `calc_s3`.
* The recipe was tuned on arm 2 (dev split B only); test was read once per final checkpoint.
* The A100 arm 2 s0 300-update test eval used its u250 checkpoint (the only one before the cap).

## Not released

* **Checkpoints** (RL'd hosts, trained links / detectors / soft vectors): none are in the repo. Regenerate with the
  commands above (about 1 GPU-hour per 300-update arm on an H100).
* **Greedy dev traces** (`interp/traces/*.jsonl`, generated text): not released; `run_all.sh`, `run_R.sh` and
  `run_aR_h100.sh` regenerate them from the 64-update checkpoints; the reduced outputs (`stats*.json`, `causes.json`,
  `analysis_out.txt`) are included.
* The v1 RL engine that `prof_v1.py` profiles (the 207 tok/s baseline).
* Wiring runs still running at packaging (partial `steps.jsonl` shipped): CW s1, CWinOx s0 / s1 (CWinO with the
  exploration floor `--iw_eps 0.05 --cw_lr 1e-3 --link_eps 1e-8`), CWinC s0 / s1, W4w s0. CWinO s0 (no floor) finished:
  veto-all from u25, dev 63.8 at u300.
* Released wiring `config.json` files of the CW*, W3d and W2ws arms show `link_lr 1e-3` / `link_eps 1e-8`; those runs used
  1e-2 / 1e-12 (per-arm defaults applied after the config was written; fixed in `train_wiring.py`).
* `wiring/smoke/`: smoke tests and the links-only SFT jobs (`cw2.sh` builds `sft3_W3e_lr1e-2/links_final.pt`, the W3d
  init, shipped in `results/wiring/H100/`).
