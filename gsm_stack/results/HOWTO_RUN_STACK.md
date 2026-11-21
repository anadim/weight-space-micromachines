# How to run the serial GSM8K stack and its RL arms (one GPU)

All commands run from the repo root on one NVIDIA GPU (B200 or H100 80 GB; A40 48 GB at smaller `BMAX`). Paths and
knobs live in `gsm_stack/env.sh`. Nothing here loads a model on a CPU.

## Code trees

The launchers never run the source folders directly. They first build a run tree with `gsm_stack/mk_tree.py` under
`$WSM_WORK/trees/<name>` (stk/, wiring/, dig_san/) and copy the shipped gadgets into it, so that edits to the repo never
reach a running job (the original runs used frozen release snapshots the same way).

| tree | what | used for |
|---|---|---|
| `engine` | `gsm_stack/engine`, release rel/v5 (full snapshot) | RL arms started after 2026-11-21 02:30Z (A1_s2, A1_s3, A_s3, B_s3, C_s2, C_s3, A1D_s*, Bb_s*, Pb_s*, A1W_s*, CW_s*, C600, CW600) |
| `v42` | `gsm_stack/eval/trees/v42`, overlay: the frozen rel/v4.2 copy + `stk/fe_hook.py` | every test / GSM-Hard number (`eval/fe.sh` default); arms A, A1, Ar, B, A_s2, B_s2 (C ran on rel/v4.1, same rollout engine) |
| `r2` | `gsm_stack/eval/trees/r2`, overlay on v42: WPR + adaptive tries (`aw_adapt.py`) | rung 2 test numbers |
| `r3` | `gsm_stack/eval/trees/r3`, overlay on r2: + answer-group voter registers (`aw_vf.py`, `numreg.py`) | rung 3 test numbers |
| `rungs/rung2..4` | `gsm_stack/implants/rungs/*`: the implant lane's dev drop-ins (rung 4 = + side-net register) | dev ladder (`results/LADDER.md`) |

rel/v5 vs rel/v4.2 (all checked exact: greedy dev 600/600 bitwise, real arm states give identical records):
- parallel TK prime, PEMIT, PCACHE, `FAST_OVLH=1` (arm-B implant step on a side stream). Rollout is 38% faster on arm B.
  `FAST_ATT3B` (split prefix) exists but is off (not bitwise, slower).
- fastroll / c2_engine no longer end a serial-stack row at token 0 on a host stop token (the stack's own done rule
  decides; compliance audit item, `COMPLIANCE.md`). B (StackHD) keeps the generator stop.
- `RL_DIV=x` (rl_hook, arm A1 only): per-attempt diversity term, −x on the advantage of a wrong attempt that repeats an
  earlier attempt's answer in a group whose final vote is wrong. Arms A1D_s* use 0.5.
- `RL_ARM=P`: plain RL with no implants. It is B's single-attempt rollout with the implant off (`imp_on=False`) and the
  compiled triggers off (`set_comp(False)`), stop = eos / blank line / 320 tokens. Arms Pb_s<seed> (P from the base
  host) and Bb_s<seed> (B from the base host).
- `WPR=1` arms A1W_s<seed> (A1 config, RL host) and CW_s<seed> (C config, base host). rl_hook trains on the reset
  positions when WPR=1.

rel/v4.2 = rel/v4 + asynchronous resume saves (`RL_ASYNC_SAVE=1`, the default; a save is skipped while the previous
write runs, field `save_skipped`, so the file on disk is always a complete earlier state). rel/v4 = rel/v3.1 + the fast
trainer: `stk/fasttrain.py` (question-dedup packing, Triton group attention fwd + bwd in `stk/tattn.py`, inductor-fused
RMSNorm / RoPE / SwiGLU, 32k-token checkpointed micro-batches, fused log-prob head, fused AdamW). Train is 1.5-3.6 s per
update (was 10-24 s) and teacher-forcing ratios stay 1.000. `RL_FASTTRAIN=0` gives the reference (flex) trainer.
Async one-step-off RL (`RL_ASYNC=1`) exists but is opt-in and was not used by any arm. rel/v3 (not shipped) had a
use-after-free in `stk/tkdual.py _prime_graph`.

Kernels (`engine/stk/KERNEL_NOTES.md`): `attn3.py` (compaction / overlap / split-KV attention with the weight-space
forget mask), `tk.py` + `tkdual.py` + `tk3_kernel.cu` (the calculator implants as one integer-exact kernel, NVRTC),
`lmhead.cu` (fused log-prob head, NVRTC), `fastroll.py` (admission / compaction rollout loop), `c2_engine.py` (calc v2
decode path). NVRTC sources compile at load time for the local `sm_XX`; Triton kernels JIT to PTX.

## Layout of `$WSM_DATA`

- `ckpt_a2s2/`: the u0 host (Qwen3-0.6B-Base after 300 GRPO updates with the calculator, `rl/gsm_grpo` arm 2 seed 2).
  Every arm also uses it as the implant source (`--eval_ckpt`), including the base-host arms. Base host: `--host_ckpt base`.
- `gsm_final/rl/<ARM>/steps.jsonl`: per-update records; dev evals every 25 updates are the `{"dev": ...}` lines.
  `gsm_final/rl/<ARM>/step_000{100,200,300}/model.safetensors`: host checkpoints. `gsm_final/logs/rl_<ARM>.out`: log.
- `states/rlstack_<ARM>.pt`: resume state (fp32 host + AdamW, ~7 GB) every 2 updates; a restarted arm resumes from it.

## Commands

```bash
# one RL arm (300 updates; resumes after a crash; see the arm table in the header of engine/run_arm.sh)
GPU=0 bash gsm_stack/engine/run_arm.sh A1
# evaluate configs on test / GSM-Hard / dev (list format in eval/fe.sh)
GPU=0 bash gsm_stack/eval/fe.sh gsm_stack/eval/lists/test3.list
# a hook on any tree (bench, greedy dev, fuzz)
GPU=0 bash gsm_stack/run_hook.sh fuzz v42 sw_fuzz_hook.py
```

Generation / eval of a host with the stack runs a hook through `dig_san/tw_dump_lean.py` the way `run_hook.sh` does
(env `WIRING_DIR STK_DIR SAN_DIR DIG_HOOK SAN_SMAX C2_NOCOMPACT=1 SER_WS=1 STK_FORGET_M=10000`, `RL_FASTROLL=1` for the
fast loop); `--eval_ckpt` = the implant source / u0 host, `--host_ckpt` = another host.

`stk/train_bench.py` times and compares trainers on a real rollout batch (`RL_TRAIN_BENCH=1`, `TB_*` knobs; `TB_PG=1`
compares plain policy-gradient grads, because PPO-clip grads are dominated by bf16 clip-mask flips).
`engine/jobs/*.sh` are the check / bench jobs used during development (they expect `$WSM_DATA` and an arm state; read
them before running).

## The evaluation harness (`gsm_stack/eval`)

`trees/v42/stk/fe_hook.py` builds exactly the arms' stack (DualImplant = calc_s4 + chain2, SerialWD for A / A1 / Ar /
C, StackHD for B, fastroll as in the arms' dev evals) and evaluates one config on dev / dev2 / test / hard.
List lines: `NAME HOST SET KIND N PR VM SAN RM [ENV...]`, HOST = u0 | base | ARM (its step_000300). KIND: S (serial, N
tries), B (S2F single attempt), SC (N independent B samples + plain majority = self-consistency control, programmatic),
H / C4 / D (ladder rungs: host only / calc_s4 / calc_s4 + chain2). Env:
- `FE_MODE=greedy`, `FE_SEED`, `FE_FORGET=0` (+ `FAST_ATT3=0`: forget mask off);
- `FE_MISS=off|hook|w` (B: missing-answer trigger none / stop-token swap (programmatic) / compiled logit write).
  `FE_MISS=w` is greedy token-identical to hook (600/600 dev). The default in `fe_hook.py` is off (no missing-answer
  trigger, all weights: the 72.25 S2F rung); `FE_MISS=w` gives the 72.63 S2F with the compiled trigger;
- implant ablation: `FE_IMP=both|calc|chain2|none`, `FE_COMP=0` (B / SC: compiled triggers off); S: SAN column 0 = no
  sanity filter, VM column 0 = plain-majority vote;
- `WPR=1` (weight-space position reset, `implants/wpr/`): each serial attempt's q and new K are rotated by R(−off), so
  attempt k sees the question at the attempt-0 distance. All weights (a fixed rotary table). WPR unset gives a bitwise
  unchanged path; it is ignored under `RL_POSRESET` (the programmatic reset).

Rows go to `$WSM_WORK/fe/<SET>_<NAME>.jsonl`. Offline voter controls: `python3 gsm_stack/eval/results_tools/offctl.py FILE.jsonl`
(last / first attempt, plain majority, probe argmax). Paired analysis: `python3 gsm_stack/eval/ana_rltest.py BASE.jsonl FILE...`
(paired bootstrap CI, fixes / breaks, attempt accuracy, oracle); `results_tools/pboot.py` / `pbootm.py` (numpy-free paired /
seed-pooled bootstrap). A test pass is ~1-2.5 min on a B200 (N = 8: ~2 min for 1,319 items). No H100 timings
were taken for this harness; the older S3-serial N = 8 run took ~12 min per 1,319 items on an H100 at batch 64
(`gsm_push/README.md`).
