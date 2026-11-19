# engine2 notebook

## 2026-11-19 stage 1 — profile of the old engine on A40 host GPU0 (A40), design
Setup: old engine built by exec of train_library.py up to its main loop (ref_env.py), m1 flags, clean SFT host,
--fused 0 (A40), PYTORCH_CUDA_ALLOC_CONF=expandable_segments. 64 prompts (pool, Random(0)) x G8, T=1, max_new 512.
q_len mean 79.5, Lp 35.
- Rollout 512 rows: B_max 256 110 s (559 tok/s, 1024 steps, mean len 120); B_max 128 139 s (427 tok/s, 2048 steps).
  Host only (imp_on False), B_max 256: 7.9 s (7710 tok/s). Prefill 6.8 s per 16 prompts x 8.
- Step (graph replay, ms) host B128 ctx128/384/640: 7.8/14.1/20.4; B256: 12.6/25.0/37.4.
  Full B128: 81.5-91.4 / 87-97 / 93-103; B256: 132/144/156.
- Components B128 (B256), ms: calc 5.3 (10.9), track 4.3 (8.1), order 8.5-9.1 (16.7), interval 3.6 (7.6),
  turtle 4.3 (8.2), truth 3.3 (6.3), expr 31.4 (32.9), calc_gen 5.6 (11.7), bool 7.5 (7.7); BBH sum 69 (99).
  IF step 3.3 (B_max128) / 3.7 (B_max256); IF write 12.5 (B_max128) / 22.5 (B_max256).
- Kernel table, full step B128 ctx384: csrmm 31.2 ms, fp64 cutlass d884 GEMMs 34.7, attn_suffix 10.8, sum over streams
  154 ms per step; cublasLt splitKreduce present (host GEMMs).
- Train step (B_max 128 run): 248 rows (31 kept groups), 28,305 resp tokens, 13 micro (tok_budget 4096):
  44.7 s / 41.0 s; loss -2.05e-3 / -2.13e-3, grad_norm 0.072 / 0.061 (two consecutive updates). sync 0.04 s.
  Peak mem 36.7 GB.
- Stack structure (stack_stats.json): all weights/biases integer; stack-1 nnz/row 13.4M total; stack-2 1.56M; keys rare.
- First attempt OOM'd at a separate 32-prompt prefill with B_max 256 (KV 26 GB) -> prefill timed on 16 prompts.

## 2026-11-19 stage 2a — tools megakernel, lazy IFEval (A40 host GPU0, A40)
Toolchain: CUDA C++ -> NVRTC 12.8 (cubin, sm_86, `-default-device`) -> cuModuleLoadData / cuLaunchKernel via ctypes on
torch's current stream (graph-capturable). No nvcc. Files: tk_compile.py, tk.py, tk*_kernel.cu, eng2.py, test_*.py.
- Bound analysis (analyze.py): static interval bounds explode (1e40..1e275; independent [0,1] window bits are far too
  loose) -> exactness of the fp32/fp64 reference cannot be proven statically. Runtime guard instead: per unit, per row,
  per step P = sum of positive products, N = |sum of negative products| (reference = products summed in any order, then
  + bias; exact iff P, N, |result| < 2^24, 2^53 for expr's fp64 stack 1). A first |b|+sum|wx| guard flagged calc and
  calc_gen (bits 0, 7); the P/N guard is clean (flag 0) on every run below.
- Activity (layer_stats.py, real windows): window bit density 0.4-1.6%, hidden wire density 0.015%-15%; "active"
  fan-in entries per row ~68k vs 16.9M stack nnz (250x).
- Kernel versions (all bitwise = old step on both corpora, all 9 tools):
  v1 gather (row-local CTA, lanes = rows): 85 ms/step at B=256 (latency of serial fan-in loops).
  v2 push (nonzero wires push w*x with atomics; per-(chunk,unit) stamps): 10.1 ms B=256 (bench, real streams).
  v3 differential (baseline = stacks on the always-on window, push only deviations; int64 P/N reds; touched-unit
  lists in shared memory spread over 16 warps): per tool B=256 / B=32 ms: calc 2.76/1.38, track 0.65/0.28,
  order 0.85/0.42, interval 0.98/0.33, turtle 0.89/0.15, truth 0.40/0.22, expr 3.55/2.25, calc_gen 3.00/1.67,
  bool 1.26/1.17; ALL 5.07 / 3.46 (old: 99 / 69). Deep stacks are latency bound (~3.4 us/layer).
- test_tools.py (step from zero state over full streams, old BBHTool.step / CalcImplant.step vs kernel):
  REF 32 rows x 690 positions: 0 mismatched row-steps all 9 tools; fire flags = reference's per-step flags (2826 steps).
  SAMPLED 256 rows (BBH + combo prompts, old engine T=1, max_new 512) x 1076 positions: 0 mismatches; every tool fires
  (nonzero steps: calc 545, track 91, order 119, interval 133, turtle 324, truth 146, expr 660, calc_gen 149, bool 18).
- Engine2 (eng2.LibEngine2: old host/in-wire/IF code, tools = 1 kernel launch on a side stream joined at L14,
  prompt features/priming by a kernel scan over the prompt): reference test (B_max 32, greedy 128):
  tokens, logp, per-tool fire flags, in-wire codes, IF write positions and vectors: 32/32 rows bitwise equal
  (old engine re-run also 32/32). Rollout 64 x G8 max_new 512 B_max 256: 43.5 s (old 110 s); same 61,432 tokens as old.
- Lazy IFEval (LibEngine2Lazy): IF step on the live bucket's rows; IF write on a capacity-C subset holding every row
  that can write (any latch / hforced / emitter program); overflow -> device flag -> 16-step block rolled back from a
  snapshot and replayed with C = B. test_if_lazy.py (64 rows: 32 reference + 32 IFEval/combo answered by the old engine,
  256 teacher-forced steps, random h, minimal C and C+7 random extra rows): write and EVERY circuit state tensor
  bitwise equal (0 / 512 mismatches). Engine reference test with lazy IF: 32/32 bitwise on all fields (C=16 used in
  6 of 16 blocks, 0 rollbacks). Rollout 64 x G8 B_max 256: **27.6 s** (2223 tok/s; old 110 s = 4.0x), 0 rollbacks.
  (Sampled outputs differ from the old run only through the in-wires' torch.rand stream: graphs captured at
  different times consume different Philox offsets; greedy is bitwise.)
- Step profile (B=256, ctx 384, graph replay): host only 25.0 ms; engine2 full 55.8 (C=B) / 42.8 (C=64) / 39.2 (C=16).
  Under contention with the host, tk3_step takes ~20-28 ms (vs 5 ms alone on real streams) and sits on the critical
  path at the L14 join; host attention (attn_suffix) ~21 ms. High-priority side stream: no measurable change.

## 2026-11-19 problem 1 (tools kernel next to the host) — timeboxed, mostly moot
- Narrow-layer direct path (read every unit, no stamp scan): SLOWER (expr B32 2.25 -> 3.61 ms) -> disabled (NARROW 0).
- High-priority side stream: no change. Serial (TK then host) 42.2 ms vs overlapped 37.7 ms per step (B256 ctx384).
- Real rollout ablation (B_max 256, 64 x G8; diag flags skip the kernel / skip IF, timing only):
  full 28.3 s | no TK 26.4 | no IF 19.6 | no TK no IF 15.7 | host only (old engine, imp off) 7.9.
  -> in real rollouts the tools kernel costs only ~2-4 s; IFEval ~8.6 s; the rest (~7.8 s) is per-step torch glue
  (in-wires, 20 CWOut adapters each recomputing LN(h), tool writes, bookkeeping). The 20-28 ms TK seen in the
  graph-replay harness came from repeated-token windows (unrealistic).

## 2026-11-19 stage 3 trainer
- Old train_step breakdown (A40, 248 rows, 13 micro): 44.0 s = features (teacher-forced CSR emit + per-token Python
  loop) 32.1, fwd_packed 3.8, lp 0.3, backward 7.2, rest 0.6.
- train_step2 (train2.make_patch): features from rollout records (engine2 records the raw 0/1 tool outputs per step,
  plus the final token's step via a one-step tools run on a snapshot, plus question-position outputs from the prime
  scan) and a vectorised suppression scan (cummax). test_train2: recorded == teacher-forced features at all 10,851
  positions of 64 rows (607 nonzero); one update old vs new from the same state: loss 0.0031814279209356755 and grad
  norm 0.07104175537824631 identical, EVERY parameter bitwise equal; 39.8 s -> 8.8 s.
- + no per-layer checkpointing (bitwise equal) + host weights fused once per update with per-micro fp32 grad
  accumulation (bitwise equal): 40.3 s -> 7.44 s, all params bitwise equal (make_patch_fast + install_nockpt).
  Profile: flex-attention backward 0.85 s, GEMMs ~1.4 s, ~3 s of elementwise (fp32 RMSNorm stats, casts, adds),
  softmax fwd/bwd 0.35 s. Optimizer + clip 0.1 s -> a fused optimizer is not worth it.
- KV cache in CUDA VMM (vmm.py): fixed virtual addresses (graphs stay valid), physical pages released during
  training and re-mapped after -> B_max 256 rollout + training fit on the A40 (before: OOM at the first backward).
- Full update (run2.py, m1 recipe, B_max 256, VMM, fast trainer): u1-u4 upd_s 45.7 / 43.6 / 40.5 / 39.2
  (rollout 32.1 / 32.8 / 28.5 / 27.7 incl. graph capture in u1; train 11.6 / 9.3 / 10.6 / 9.8 with 30-42k tokens).
  B_max 128 (no VMM): 53-59 s. Old engine on the A40: ~150 s (B_max 256 rollout 110 + train 41) / ~180 s (B_max 128).
- Fused log-prob head (lmhead.cu/.py, NVRTC): cuBLAS bf16 logits as F.linear, then one custom pass for
  max/sumexp/target (fp32 online) and one for the logit gradient (bf16 in place), logits kept instead of recomputed.
  Unit test vs torch: lp max diff 9.5e-7, dh rel 9.2e-6, dE rel 5.1e-8. One update vs the old trainer (same batch, same
  start): loss rel err 3.8e-7, grad norm rel err 1.07e-4, update cosine 0.99999988, update rel err 1.1e-4;
  deterministic (two runs bitwise equal). 7.44 -> 6.8 s.
- Full update with fused head (B_max 256, VMM): u1-u3 upd_s 43.3 / 41.5 / 36.4 (rollout 31.0 / 32.9 / 27.1, train
  10.5 / 7.1 / 8.1 at 30-35k tokens).

## 2026-11-19 stage-4 smoke (run2.py, m1 recipe, clean SFT host, B_max 256, VMM, fast trainer + fused head)
- First attempt (runs/smoke25_crash_u20): 19 updates, then OOM in the u20 dev eval prefill (64-prompt prefill chunk
  on top of the 26 GB KV) -> prefill_chunk 16. Prompt batches are the old main loop's (python random not reseeded).
  u1-u19 (paired with m1 u1-u19, H100 old engine):
    reward 0.328 vs 0.334 (first5 0.301/0.309, last5 0.346/0.353); r_gsm 0.179/0.176, r_bbh 0.606/0.622,
    r_combo 0.065/0.057, r_ifeval 0.359/0.376; len 136.7/134.1; kept groups 37.0/35.7.
    fire rate new|old: calc 0.856|0.897, track .924|.925, order .930|.945, interval .906|.931, turtle .900|.917,
    truth .944|.902, expr .929|.927, calc_gen .876|.920, bool .928|.918, ifeval .924|.927; bursts/row match
    (calc .748|.709, expr .593|.553, calc_gen .255|.225); IF latch .394|.394, IF write rows .251|.249.
    dev@0 (greedy, 400): 44.75 vs 44.5 (old, H100); dev eval 39 s.
    upd_s mean 38.1 (rollout 26.9, train 9.7) vs m1 ~71 s on H100.
  Exactness guard: tk_flag became 129 (calc + calc_gen) during u19's rollout (0 before) -> streams now saved when it
  fires (guard_u*.pt) for an offline replay against the old step functions.

## 2026-11-19 H100 port (H100 host GPU0, 19:48-20:00 UTC; files in <workdir>/rl_library/engine2, h100_port.sh)
- No code change needed: NVRTC compiles the tools kernels for sm_90 at load; paths via E2_LIB, FLEX_SMALL off.
- H100 reference dump with the OLD engine (train_library_ref.py --dump_ref 32 --max_new 128, B_max 32, fused calc):
  old engine re-run 32/32 equal; engine2 (lazy IF) vs the H100 dump: tokens, logp, per-tool fire flags, in-wire codes,
  IF write positions + vectors 32/32 rows bitwise equal (two reps; C=16 in 6/16 blocks, 0 rollbacks).
- Full updates (run2.py, m1 recipe, fused head, VMM): B_max 256: 26.3 / 22.0 / 19.5 s (rollout 20.6 / 18.7 / 16.6,
  train 4.5 / 2.8 / 2.6). B_max 512 (one wave for all 512 rows): 19.5 / 17.1 / 16.9 s (rollout 13.4 / 14.3 / 14.1,
  train 4.9 / 2.2 / 2.4). Old engine m1 on the same H100: ~70 s per update.
- Rerun seed 0 (runs/smoke25, prefill_chunk 16, 25 updates): reward 0.320 vs m1 0.339; r_gsm 0.117 vs 0.188 (last5
  0.111 vs 0.237); dev@20 47.75 (gsm 18.3), dev@25 46.25 (gsm 21.7) vs m1 dev@25 49.75 (gsm 35.0). Calc write tokens
  per fired burst fell from ~8 to ~4.5 between u4 and u6 and stayed there (run A, m1 and both seed-1 runs: ~8). Guard
  fired at u4/u21/u24. upd_s mean 36.4 (rollout 26.2, train 8.7).
- Seed 1 (runs/smoke25_s1, paired with the old engine's m1_mixLIB_s1 on H100, same prompt batches): reward 0.362 vs
  0.358 (first5 0.339/0.340, last5 0.377/0.377); r_gsm .180/.174, r_bbh .620/.613, r_combo .062/.059, r_ifeval
  .450/.453; calc tokens/burst track update by update; fire rates within 0.02 except calc_gen .913/.924; dev@0 44.75 vs
  44.5, dev@25 51.0 vs 52.25 (gsm 31.7 vs 37.5, bbh 78.3/75.8, ifeval 65/64, combo 11.7/15.0). Guard fired at u13.
- Guard streams replayed (test_tools --streams, old BBHTool/CalcImplant step vs kernel): guard_u4 (512 rows):
  0 mismatches on all 9 tools; every tool output on guard_u4/u21/s1-u13 streams is exactly 0/1 (check_guard_vals.py).
  The guard is conservative here; recorded features now stored fp32 anyway (int8 would wrap a non-0/1 value).
- Shared adapter LN (install_ln_memo, rollout only): engine reference test 32/32 bitwise on all fields.
- IF shift invariance (test_if_shift.py): 64 rows (32 reference + 32 IFEval/combo), 256 teacher-forced steps,
  744 nonzero write row-steps; counter offsets 1/37/300/2000: 0 write mismatches, 0 latch mismatches.
- Slot refill (LibEngine2Refill): reference test (B_max 32, one admit, prefix-cached tools priming + question scan,
  time-shifted IF priming at T_BASE 1024): 32/32 bitwise on all fields. With real refills (B_max 64, 32 x G8, 12 refills):
  recorded features == teacher-forced at all 25,166 positions of 128 rows. Slow so far (61 s for 256 rows): the IF
  priming of each refill group is eager (~300 circuit steps) -> next: graph-capture it per group size.
