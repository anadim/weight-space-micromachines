# GSM8K implant push (past the RL+calc host's 71%)

Goal: raise GSM8K accuracy of the frozen RL+calc host (Qwen3-0.6B-Base after 300 GRPO updates with the "=" calculator,
arm `a2_u300_s2` of `rl/gsm_grpo`) with new implants only, inside one forward-only generation call. Full report:
[`docs/FINAL.md`](docs/FINAL.md); protocol and data discipline (dev = GSM8K train split B, test used once per config):
[`docs/PROTOCOL.md`](docs/PROTOCOL.md).

| config (GSM8K test, 1,319 items, greedy unless noted) | test | GSM-Hard | tok / q |
|---|---|---|---|
| S0: host + calc_s4 (the original calculator) | 71.27 | 54.59 | 137 |
| S1: + calculator v2, forward-only (`calc_v2/chain2.py` compiled for the +/- subset; `final/calc2.py` hook) | 71.87 | 56.03 | 137 |
| **S2F**: + answer-sanity DFA + weight-space forget + " We need to find" | **73.39** (+2.12, CI [1.29, 2.96]) | 56.18 | 153 |
| S3-serial N = 4: S2F inside 4 serial tries + forget + probe vote, one stream, T = 1 | 78.92 | 60.42 | 634 |
| **S3-serial N = 8** | **80.44** | 62.17 | 1274 |

Numbers are from the run logs in `results/final/*/{test,hard}.log` (one run each; see FINAL.md for CIs, fixes / breaks,
silence checks and every use of the test set).

## Layout

| path | what |
|---|---|
| `final/` | the final stack. `calc2.py`, `c2_engine.py` (calc v2), `chain2.py` + `implant2.py` (compiled chain calculator), `san_engine.py` + `comp_sanity.py` (34-state answer DFA, phrase writer), `forget_ws.py` (forget in weight space: extra un-rotated q/k dims), `serial_stack.py`, `comp_serial.py` (serial tries, register file, ReLU voter), `stack{,2,3,4}_hook.py` / `stackh.py` (S0-S3 stacks), `rl_hook.py` (RL with the stack attached), `sw_eval_hook.py`, `sw_fuzz_hook.py` (reader / voter fuzz), `verifier_probe_L18.pt` (6 KB probe), `run_stk.sh`, `run_rl.sh`, `ana*.py` |
| `calc_v2/` | calculator v2 development: `calc2.py`, `chain2.py`, `test_exact.py` (model-free, tokenizer only), `test_chain2.py` (GPU), `solve_veto_op.py` + `veto_op_qwen3.pt` (6 KB), eval hooks; report `docs/calc_v2.md` |
| `reading/` | the sanity / forget-mask agent: `comp_sanity*.py`, `rd_trig.py`, `rd_eval.py`, `comp_eval.py`, offline checks in `debug/`; report `docs/reading.md` |
| `selection/` | serial tries + verifier probe agent: `serial_engine.py`, `ver_engine.py`, `probe_engine.py`, `calib.py`, `analyze.py`; small result tables in `selection/results/`; report `docs/selection.md` |
| `harness/tw_dump_lean.py` | the driver every launcher runs: `rl/wiring/train_wiring.py` with the fp32 host kept on the CPU, a larger KV size (`SAN_SMAX`) and an `exec` of `$DIG_HOOK` after the checkpoint is loaded |
| `results/` | per-run logs and `summary.json` files (accuracy, tokens, fire counts). Per-row jsonl outputs and hidden-state dumps are not shipped |
| `env.sh` | shared paths (`WSM_ROOT`, `WSM_WORK`, `CKPT`, `DFA_CACHE`, ...), sourced by every `run_*.sh` |

## Reproduce

Hardware: one H100 80 GB (S2F: about 1 min per 1,319-item set at batch 128; S3-serial N = 8: about 12 min at batch 64).
A40 48 GB works at a smaller batch (`BMAX=32`). Linux + CUDA (Triton kernels); nothing here runs a model on a CPU.

1. Build the host. The RL+calc checkpoint is not released. Train it with `rl/gsm_grpo` (REPRODUCE.md §12):

   ```bash
   python scripts/solve_veto.py --device cuda                               # runs/veto_qwen3.pt (calculator veto)
   cd rl/gsm_grpo && python train2.py --arm 2 --seed 2 --run_dir runs/long/a2_s2 --updates 300 --prompts 64 --G 8 \
       --T 1.0 --lr 1e-6 --dev_n 800 --dev_every 25 --ckpt_every 100 --fused 1
   export CKPT=$PWD/runs/long/a2_s2/step_000300                             # dir with model.safetensors
   ```

   The host's own GSM8K test score should be about 71.2 (S0 below reproduces it). RL is not bit-reproducible across
   GPUs; expect +-1 point.

2. Run the stacks (from the repo root; `GPU=0`):

   ```bash
   # S0, S1, S2F on dev / dev2 / test (stack3_hook: compiled sanity + weight-space forget)
   for SET in dev dev2 test; do
     HOOK=stack3_hook.py SAN_SMAX=1280 bash gsm_push/final/run_stk.sh fgtw 0 $SET - \
       STK_CFGS=S0,S1,S2F_fwntf STK_FORGET=weights STK_FORGET_MOD=forget_ws
   done
   # GSM-Hard: missing-answer trigger only
   HOOK=stack3_hook.py SAN_SMAX=1280 bash gsm_push/final/run_stk.sh fgtw 0 hard - \
     STK_CFGS=S0,S1,S2F_fwntf STK_FORGET=weights STK_FORGET_MOD=forget_ws STK_MISSING_ONLY=1 SAN_CLASSES=3
   # S3-serial, N = 8 and N = 4 (jobs = N:repairs:vmode:sanity:repmiss:mode; seed 7)
   HOOK=stack4_hook.py BMAX=64 SAN_SMAX=5696 bash gsm_push/final/run_stk.sh ser8 0 test - STK_SER='8:1:1:1:0:T1.0'
   HOOK=stack4_hook.py BMAX=64 SAN_SMAX=5696 bash gsm_push/final/run_stk.sh ser4 0 test - STK_SER='4:1:1:1:0:T1.0'
   # GSM-Hard serial: plain majority, no sanity filter, no probe, missing-answer repair (A100 / A40 runs:
   # results/final/ser_a100/ser8/hard.log 62.17, results/final/ser4b/hard.log 60.42 at BMAX=32)
   HOOK=stack4_hook.py BMAX=64 SAN_SMAX=5696 bash gsm_push/final/run_stk.sh ser8 0 hard - STK_SER='8:1:0:0:1:T1.0' SAN_CLASSES=3
   ```

   Expected: `S2F_fwntf acc=73.39` on test, `N8_R1_v1_s1_m0_T1.0 acc=80.44` (logs in `$WSM_WORK/final/out/<name>/<set>.log`;
   the shipped copies are `results/final/fgtw/test.log` and `results/final/ser8/test.log`). bf16 batch-shape noise moves
   a few rows between GPU types and batch sizes (FINAL.md §2).

3. Exactness checks:

   ```bash
   python gsm_push/calc_v2/test_exact.py          # CPU, tokenizer only: calc v2 tracker on 10k expressions per form (test_exact.log)
   python gsm_push/calc_v2/test_chain2.py         # GPU: compiled chain2 through the real decode path (test_chain2_final.log)
   HOOK=sw_fuzz_hook.py bash gsm_push/final/run_stk.sh fuzz 0 dev -   # GPU: reader fuzz (20k texts) + voter fuzz (81,920 decisions)
   ```

## RL with the stack attached (in progress)

`final/rl_hook.py` + `final/run_rl.sh` train the host with the all-weights single-call stack attached (arms A, A', B,
A-reset, C; `docs/RL_STACK.md`). These runs were still going when this snapshot was taken; `docs/RL_STACK.md` is the
live status note, and its curves are in `rl/figures/v3/data/rlstack/` (figure script `rl/figures/v3/make_rl_stack_curves.py`).
No blog number depends on them yet.
