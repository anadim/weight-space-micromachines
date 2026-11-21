> Release note: this is the engine lane's development note from the 2026-11-20 hand-off, kept as written (host names
> replaced by GPU types). Later releases finished items of section 5: TK priming (parallel prime, rel/v5), the fast
> trainer with `lmhead.cu` (rel/v4), split-prefix attention (`attn3.py`, FAST_ATT3). The cluster launchers it names
> (`launch_C.sh`, `bench_fast.sh`, `run_rl.sh`) are replaced by `gsm_stack/engine/run_arm.sh` and `gsm_stack/run_hook.sh`;
> `engine2_src/` is `rl/engine2/` in this repository.

# Rollout-engine speed work: state at handoff (2026-11-20, ~22:30 UTC)

Scope: RL rollouts of the all-weights serial GSM8K stack (rl_hook.py + comp_serial.SerialWD / stackh.StackHD on the
rl_wiring Engine). Nothing below is deployed on a running arm. The running arms use the old path (run_rl.sh unchanged).

## 1. Why the old path is slow (C2Engine._wave with C2_NOCOMPACT=1)
- Rows run in waves of B_max rows (arm A: B_max 64, so 8 waves per update). Each wave lasts as long as its longest row.
- Finished rows stay in the batch. Engine._book advances `slot` unconditionally, so a finished row's attention walks to
  S_max (5696) on every step.
- One host sync per decode step. The calc-v2 ALT argmax (c2_on) runs every step even though that hook is off.
- KV is static B_max x S_max (114.7 KB per token). S_max 5696 is needed only for the N=8 dev eval; N=4 training rows
  are at most 2728 new tokens.
- Already graph-captured: the triton host kernels, the side state (_sw circuit), the torch implants.

## 2. A40 per-step profile (A40 node:0, B=32, a2s2 host, N=4; prof_step_hook.py)
Graph replay, ms per step:

| ctx | full | implants off |
|---|---|---|
| 256 | 9.6 | 7.3 |
| 768 | 12.9 | 10.6 |
| 1536 | 17.8 | 15.6 |

Eager kernel breakdown at ctx 768 (13.3 ms total):

| kernel group | ms |
|---|---|
| attn_suffix_bias (forget-bias attention) | 5.24 |
| bf16 GEMMs (cuBLAS) | 3.13 |
| attn_prefix (shared 1066-token prefix) | 1.53 |
| cuSPARSE csrmm (calc_s4 + chain2 torch stacks) | 1.36 |
| torch elementwise (implants, _sw/voter, _book) | 1.58 |
| sample | 0.16 |
| rmsnorm / rope / silu | 0.32 |

With the exact forget-skip (rows 170 tokens into a later segment), the step is flat in context:
9.8 / 10.3 / 10.8 ms full, attention 3.1 ms at ctx 768.

## 3. What is done and verified (files in this folder, also on A40 node $WSM_DATA/gsm_final/code/)
- **fastroll.py**: continuous batching. A queue of all rows; free slots are refilled (prefill + per-slot register reset
  to the wave-start value + implant priming); finished slots are harvested and parked at slot 0; host sync every
  FAST_K=8 steps; c2_on is off. The step math is unchanged. Env switches:
  - FAST_WAVE=1: exact old schedule (machinery check).
  - FAST_EXACT_PREFILL=1 (default; keep it): each admitted row's question is prefilled inside its OLD wave group of
    ppc = B_max // G consecutive prompts, writing only the admitted rows. Without it, greedy matched only 38/64 rows
    (max dlogp 0.126): the bf16 prefill is batch-variant (padding / GEMM M). With it, the match is bitwise.
  - FAST_FSKIP=1 (default): exact forget-skip attention (attn_suffix_bias_skip_kernel). Key blocks wholly inside the
    forgotten span [s_qlen, s_fr) are skipped. Exact only for M >= 1e3, where the bias underflows exp to 0.0 and the
    block update is the identity. Enabled automatically only for SerialWD with w_M >= 1e3; arm B (StackH, M = 50)
    keeps the full kernel. Block boundaries stay multiples of BN.
  - FAST_TK=1 (default): calc_s4 + chain2 through engine2's tk3 tools kernel (tkdual.py: TKDual wraps the
    DualImplant, the same _gate, priming by scanning the K+W = 150 tail tokens in scratch rows, then copy_rows).
  - FAST_TIME=1: reports admit_s in the stats.
- **rl_hook.py** (local and A40 node copies; H100 node and A100 node have OLDER different versions, do not overwrite them
  under running arms). All new switches default off:
  - RL_RESUME_DIR (default /dev/shm);
  - RL_FASTROLL=1;
  - RL_FASTCHECK=N: a greedy-dev old vs fast check at startup, logged as {"fastcheck": ...};
  - FIRST_UPDATE_RATIO print;
  - RL_SKIP_EVAL0=1, for smoke tests only.
- **tkdual.py, tk.py** (patched: NVRTC root from the `nvidia` pip package or TK_NV), **tk_compile.py, tk3_kernel.cu**
  (copied from H100 node rl_library/engine2).
- **launch_C.sh**: arm C on an the B200 node B200 (ROOT=$WSM_DATA). Resumes from $ROOT/states/rlstack_C.pt and
  prompts=64, RL_FASTROLL=1 with FSKIP on and FAST_TK=0 by default (set FAST_TK=1 once priming is fixed; TK
  is exact but slower now, see pitfall 4).
- Bench and check hooks: fast_bench_hook.py + bench_fast.sh (old vs wave vs fast), prof_step_hook.py (step profile),
  tk_check_hook.py (implant fuzz).

Equivalence evidence (A40 node A40):

| check | config | result |
|---|---|---|
| wave mode vs old, sampled, same seed | B=32, 128 rows | 128/128 tokens, wflags, logp (diff 0.0) and every register identical |
| greedy dev, old vs fastroll | refill + exact prefill (+ FSKIP) (+ TK), 64 items | 64/64 tokens, wflags, registers identical; dlogp 0.0; acc 62.5 = 62.5, in each of three runs |
| first-update teacher-forcing ratio | fastroll | 0.999691, identical with and without TK; old arms log 0.9998–1.0003 |
| implant fuzz, TK vs torch | 64 rows x 600 steps: real rollout streams (1772 firing row-steps) + 3 random-token fuzz sets | 0 mismatching row-steps, max diff 0.0, tk_flag 0 |

Speed on the A40 (64-row greedy dev, B=32): old 35.6 s, fastroll 27.5 s, + FSKIP 25.3 s, + TK 28.2–30.9 s.
TK is currently slower end to end because of priming (pitfall 4).

## 4. Pitfalls
1. **FLEX_SMALL=1** is required on A40/A100-class GPUs (<=100 KB smem). Without it, the inductor flex_attention
   compile in train() fails with "No valid triton configs ... Required 107008 Hardware limit 101376". This is why arm
   C failed to restart on A40 node. It is not needed on H100/B200.
2. Arm C is STOPPED at u30 (/dev/shm/rlstack_C.pt on A40 node; the coordinator copied it to A100 node states/).
3. Graph warm-ups advance mutable state. Everything stateful must be in _mut: fastroll adds the TK state; the stepc
   counter is fine.
4. **TK priming is slow** (KW = 150 sequential kernel steps per admission). Eager: ~100 ms per call. My CUDA-graph
   version (_prime_graph) measured 250 ms per call, worse, root cause NOT found. Suspect: scratch rows are reset to 0
   rather than the always-on baseline, so many deviations are pushed; or the capture includes something heavy. Fix
   options: (a) copy-prime: run the torch prime_rows for the admitted rows, then convert sym_h / S_h / key_h into the
   TK rings (the token ring needs ids: the stash in fastroll._f_stash already has them); (b) a prefix-cached state plus
   a scan over the question only, as eng2.LibEngine2Refill._tk_prime_rows does; (c) prime once per prompt (dedupe is
   implemented).
5. The TK step in eager timing is 3.8 ms (real streams) / 7.5 ms (fuzz) at B=64 vs torch 9.3 ms. Graph-captured TK
   step time is not measured yet.
6. The host E is trainable: TKDual.refresh() rebuilds the symbol tables every generate (asserted 0/1, same always-on
   set). On an assert failure, fastroll falls back to the torch implants.
7. Dev-eval sampling streams differ under fastroll (other Philox slot assignment); the distribution is the same.
8. Three diverging comp_serial.py / rl_hook.py versions exist (H100 node, A100 node, A40 node = local). They are functionally
   the same (refactor / SW_NMAX).

## 5. Plan not done (order of payoff on B200)
1. Fix TK priming (pitfall 4), then measure on a B200 at B_max 384 (arm C) / 192 (arm A, S 5696). The KV budget is
   B_max x 337 MB at S 2944, 653 MB at S 5696.
2. Fuse the _sw side circuit + voter + _book (~150 tiny torch ops) into one kernel.
3. Split-KV flash-decode for the suffix with the label bias, and prefix attention merged in. Attention reads ~0.5 of
   the context bytes after FSKIP.
4. Host megakernel (engine2 DESIGN.md section 6 and stage 2b; the user accepted bf16-level validation for the host).
5. Trainer: engine2 train2 lessons (no checkpointing, fuse once per update, fused lm-head lp: lmhead.cu); async one-step-off.

## 6. engine2 sources (engine2_src/)
- rl_library_engine2/: H100 node $WSM_DATA/rl_library/engine2. tk*.cu, tk.py, eng2.py, train2.py, lmhead.cu, vmm.py,
  DESIGN.md, NOTEBOOK.md. tables.pt and vocab_strs.json are excluded (large, regenerable).
- rl_implants_v2/: H100 node engine / hostkern / fastqwen / implant* / train2.
- rl_wiring/ (A40 node gsm_sel copy, used by the arms incl. tw_dump_lean.py) and rl_wiring_h100/.
