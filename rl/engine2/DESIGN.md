# engine2 — design (stage 1, 2026-11-19)

Goal: a much faster rollout + training engine for the LIB arm (Qwen3-0.6B host + calc_s4 + 8 BBH tools + IFEval circuit,
per-tool in-wires and out-adapters, Dr.GRPO). Hard rule: implant outputs are bit-exact; any mismatch is a bug.

All numbers below were measured on A40 host GPU0 (A40, sm_86) with the OLD engine built by executing
`train_library.py` itself up to its main loop (`engine2/ref_env.py`; m1 flags, clean SFT host, `--fused 0` as on A40).
Raw logs: `engine2/runs/prof_old*.out`, `prof_old128/prof.json`, `prof_old/kernels_full_B128.txt`.

## 1. Where the time goes now (A40)

### Per update (64 prompts x G8, max_new 512)
| phase | B_max 256 | B_max 128 | host only (B_max 256) |
|---|---|---|---|
| rollout (512 rows, mean len 116-120, max 512) | **110 s** (559 tok/s) | 139 s | **7.9 s** (7710 tok/s) |
| of which prefill (tool_feats + IF priming) | ~2 x 6.8 s | ~4 x 6.8 s | <0.5 s |
| train_step (248 rows, 28k resp tok, 13 micro) | — | **41-45 s** | — |
Old engine per update on A40 ~ 150 s. The tools cost 93% of the rollout: host-only is 14x faster on the same prompts.
(H100 m1 for reference: rollout ~54 s, train ~16 s, update ~70-81 s.)

### Per decode step (CUDA-graph replay, ms)
| B | ctx | host only | full (tools + IF) |
|---|---|---|---|
| 128 | 128 / 384 / 640 | 7.8 / 14.1 / 20.4 | 81.5-91.4 / 87-97 / 93-103 |
| 256 | 128 / 384 / 640 | 12.6 / 25.0 / 37.4 | 132 / 144 / 156 |

Host step grows with context: the A40 decode is KV-bandwidth bound (28 x 8 x 128 x 2 x 2 B = 114.7 KB per cached token;
B=256 at ctx 384 = 11.3 GB per step = 16 ms at 696 GB/s; weights add 1.2 GB = 1.7 ms). Floor at B256/ctx384 ~ 18 ms vs
25 ms measured: the host itself has at most ~1.4x left on A40 (more on H100, where launch gaps are a larger share).

### Tool components (isolated graph replay, serial, ms per step)
| tool | B=128 | B=256 | what dominates |
|---|---|---|---|
| calc_s4 (unfused on A40) | 5.3 | 10.9 | CSR stacks |
| track / order / interval / turtle / truth | 4.3 / 8.5-9.1 / 3.6 / 4.3 / 3.3 | 8.1 / 16.7 / 7.6 / 8.2 / 6.3 | cuSPARSE csrmm (stack 1) |
| expr_gen (667-layer stack 1, fp64 dense) | 31.4 | 32.9 | 667 x launch + fp64 GEMM (A40 fp64 = 1/64 rate) |
| calc_gen | 5.6 | 11.7 | CSR stacks (stack 2 is 20 layers, 767k nnz) |
| bool_gen (850 layers) | 7.5 | 7.7 | launch-bound depth |
| **BBH sum** | **68.6-69.1** | **99.0** | |
| IF step (always B_max rows) | 3.3-3.7 | 3.7 | ~300 tiny elementwise ops |
| IF write (always B_max rows) | 12.5 (B_max 128) | 22.5 | 2 full LM-head GEMMs + ~30 (B x 151936) mask ops |

Kernel-level (full step B=128 ctx 384; kernel time summed over all streams = 154 ms per step for a 97 ms wall step):
cuSPARSE csrmm 31.2 ms, fp64 cutlass GEMMs (DenseStack) 34.7, host attention 10.8, ~40 ms of PyTorch elementwise /
gather / index kernels (tool glue, in-wires, IF), host bf16 GEMMs ~5, cublasLt splitKreduce present (host GEMMs use
split-K for some shapes; relevant to bit-exactness, section 6).

### Structure facts that drive the design (`runs/stack_stats.json`, CPU analysis of the compiled stacks)
- Every weight and bias of every stack is an INTEGER. All wires are integers. Exact integer arithmetic is possible.
- stack 1 nnz per row: track 2.14M, order 3.48M, interval 1.15M, turtle 2.09M, truth 0.57M, expr 0.45M (667 layers,
  weights up to 1e14 -> int64), calc_gen 1.77M, bool 0.13M (850 layers), calc_s4 1.59M. Sum 13.4M MAC per row.
- stack 2 nnz per row: tiny except calc_gen and calc_s4 (767k each, 20 layers). Sum ~1.56M MAC per row per step.
- Keys are rare tokens (`</tool_call>` for the 5 call tools, `=` variants for expr/calc/bool/calc_s4). The old
  "work-skip" rule (run stack 1 only if the token is a key OR no key is in the W window) therefore does NOT skip:
  with no key in the window the hop copies the newest S, so stack 1 runs for almost every row at every step.
- Inputs to stack 1 are K-token windows of one-hot symbol flags (K = 48..128, NS = 20..308): very sparse 0/1 inputs.
- IFEval: rows latched 25-55% (m1 u1-u4), rows with any write 11-41%, write positions ~1.5% of tokens.

## 2. Plan: new decode step

One persistent **tools megakernel** (TK) per decode step does the whole token step of all 10 tools, launched on a side
stream at the start of the step and joined at host layer 14 (as now), plus a lazy IF write after the last layer.
Then stage 2b folds the host into the same persistent kernel (section 6 explains why this is second).

### 2.1 Tools megakernel (calc_s4 + 8 BBH; CUDA C++ via NVRTC -> ptxas, loaded with the driver API; no nvcc on the pod)
Toolchain found on A40 host: NVRTC 12.8 (`nvidia/cuda_nvrtc`), CUDA headers, `ptxas`/`cuobjdump` shipped with Triton.
Launch through `cuLaunchKernel` on torch's current stream so it is CUDA-graph capturable. PTX inline only where needed
(ld.cg/volatile polls, red.release, cp.async), following the gpt2_baremetal lessons (no predicated loads in loops,
monotonic barrier counters, tags in device memory not Python ints, `.cg` for cross-CTA data).

Per tool, per row, one step = reader -> ring write -> window -> stack 1 -> S ring + hop -> stack 2 -> one-hot e.
- **Row-local execution, no grid barriers.** A CTA owns (tool, chunk of rows) and walks that tool's layers with
  `__syncthreads` only. The old fused kernels split units across the grid and paid a grid barrier per layer
  (667 + 850 barriers per step for expr/bool). Work items are scheduled from a static queue sorted by cost
  (deep stacks first, so their serial depth starts at t=0), on a reserved SM set (default 1/4 of SMs) while the host
  runs layers 0-13 on the rest (green-context-style partition by CTA count; the host kernels are bandwidth-bound,
  the tools are L1/L2/issue-bound, so they overlap well).
- **Integer arithmetic.** Wires as int32 where the compile-time interval bound proves |value| < 2^31 and int64 where
  not (expr). The reference runs in fp32 (fp64 for expr): it is exact only while partial sums stay below 2^24 (2^53).
  The compiler runs interval-bound propagation over each stack and ASSERTS that bound for every unit of every fp32
  stack; if a unit can exceed it, the reference itself may round there and we must emulate its fp32 summation order
  for that unit (flagged, not silently "fixed").
- **Event-driven first layers.** Window inputs are 0/1 and sparse: the first layers of stack 1 are evaluated by pushing
  only the nonzero input bits (column-major weights), later layers by ELL gather (fan-in segmented as in fused_bbh,
  bug list 1-4 of FUSED_KERNEL.md applied). Weights per tool are shared by all rows of a CTA -> read once per CTA from
  L2/smem; wires of the CTA's rows live in shared memory where they fit, else in an L2-resident scratch.
- **Deep stacks collapsed.** expr (667 layers) and bool (850): compile-time interval analysis marks units whose
  pre-activation is provably >= 0 (ReLU = identity) or <= 0 (dead). Identity units are substituted into their
  consumers (integer weight products stay exact; int64 overflow checked), dead units dropped; layers are re-levelled.
  Even without collapse, row-local depth costs ~0.5-1 us per layer per CTA (~0.4-0.8 ms), hidden behind host L0-13;
  collapse is measured, not assumed.
- **Hop** = select of the most recent key slot (else newest), exactly the reference's fp32 one-hot softmax.
- **In-wire decisions** (calc + 8 BBH): nz flags, burst start, logit = k/2 + b_i + v_i . LN(h14[t-1]), fire / veto,
  suppression latch, decision log-prob: computed in TK (the LN statistics in fp32 in the same order as `_lnorm`).
  Sampling uses the same Philox stream semantics as torch.rand only if we need bitwise sampled decisions; greedy
  decisions (logit > 0) are deterministic and are what the bit-exact test checks.
- **Out-adapter features**: TK emits per tool the one-hot index (int8, -1 = silent) instead of dense e; the L14 and final
  CWOut adapters become tiny gathers (A[:, idx] + c, ReLU, B @ z, gate) fused into one kernel each (or into the host
  megakernel in 2b). The fixed writes e @ D become row gathers of D.
- **Outputs per step**: idx (B, 10) int8, fire flags, decision codes, decision log-prob, write vector.

### 2.2 Lazy IFEval
- IF token step: run at the live bucket B, not B_max (now always 256 rows). First version keeps `circuit_g.step`
  (graph-captured, 3.3-3.7 ms) but on compacted rows; second version: per-row state machine compiled from a trace of
  `circuit_g.step` (fx trace -> elementwise/gather op list -> generated CUDA, one warp per row). Exactness by bitwise
  comparison of every state tensor after every step on the reference streams.
- IF write: only rows that can write (any latch, or prog > 0, or hforced) are gathered into a compact batch. For a
  silent row every gate is 0 and the write is exactly 0 (argued from the code of circuit_g.write: every start_prog gate
  carries a latch gate or prog/hforced, `ffail` only counts active rows; NOT yet verified -- the skip will be tested
  bitwise on every state tensor before it is used). The two LM-head readouts run only on compacted rows;
  the second readout (`ffail`, diagnostic, never read by the engine) only on active rows. Vocab-wide mask building is
  done per row in one kernel (no (B, V) temporaries).
- Expected: IF 26 ms/step (B_max 256) -> ~1-3 ms at 25-55% latch, ~0 for non-IF batches.

### 2.3 Rollout loop changes (numerics unchanged)
- **Continuous batching**: today 512 rows run as 2 waves of 256 (4 of 128), each wave lasts until its longest row
  (max_new 512): 1024 steps for 61k tokens (B_max 256). Refilling finished slots with queued rows (prefill the new
  rows' KV mid-run) cuts the step count toward ~61k/256 = 240 + the tail. Requires per-row bit-identical results
  independent of batch composition: true for TK (row-local) and for our own host kernels; with cuBLAS the result of
  a row can depend on the bucket size M (it already does in the old engine).
- **Prefill**: tool priming needs only the last K+W tokens per row for state, but the adapters also read features at
  prompt positions -> run the TK kernel over (row, position) pairs of the prompt (a batched scan), not CSR over the
  256-pad stream in chunks of 8 (6.8 s per 128 rows now -> target < 0.3 s). IF priming: one graph per token today;
  the compiled IF step makes it a loop inside one kernel.
- One host sync per ~16 steps as now; live-row compaction stays.

## 3. Training step (Dr.GRPO; must match the old update numerically)
Old A40: 41-45 s for 248 rows / 48k tokens (q 80 + resp 116). FLOP floor: 6 x 0.6e9 x 48k = 173 TFLOP -> ~2.5-3.5 s at
50-70% of A40 bf16 peak (+33% for recompute). Where the old step loses time:
1. Tool features recomputed teacher-forced with CSR `Micromachine.emit` over every position of every micro-batch, and
   the decode-time vs teacher-forced agreement then rechecked. **New: the rollout stores per token the tool one-hot
   index (int8 x 10) and the IF write (sparse, already stored)**; training reads them. Identical by construction (the
   old hf_check shows 0 flag mismatches; we re-verify on the reference).
2. Python loops over every response token to build the keep/suppression mask (`for t in range(rl)` with tensor ops):
   vectorize (cumulative suppression = segmented scan over burst starts) -> one kernel.
3. `Q.fuse(P32, bf16)` once per micro-batch (13 per update, 2.4 GB fp32 read each): once per update.
4. LM head: fp32 logits (4096 x 151936 = 2.5 GB per chunk) + checkpoint recompute. **New fused kernel**: chunked
   `h @ E^T` -> online logsumexp -> gather target -> ratio/clip/Dr.GRPO per-token loss -> backward (dlogits = p - onehot,
   scaled) -> dh and dE accumulated without materializing logits. Same math in fp32; the loss sum order differs from
   torch's, so "match" = loss and grad norm agree to fp32 rounding (reported), not bitwise.
5. Packed varlen attention: keep flex_attention first (compiled, correct); replace with a custom varlen flash fwd/bwd
   over the shared prefix only if the profile says it pays.
6. Optimizer: one fused kernel pass for grad-norm (2-pass: norm, then clipped AdamW + fp32 master + bf16 fused weights
   write), replacing clip_grad_norm_ + AdamW + `sync_weights` re-fuse. Prefix KV refresh as now.
Expected: 41-45 s -> ~4-6 s on A40 (H100: 16 s -> ~2 s).

## 4. Expected speedup (A40, per update 64 x G8, max_new 512)
| | old | new (stage 2a+3) | new + 2b host megakernel |
|---|---|---|---|
| rollout | 110 s | ~9-11 s (host-only 7.9 s + TK ~1 ms/step + prefill) ; ~5-7 s with continuous batching | ~4-6 s |
| train | 41-45 s | ~4-6 s | ~4-6 s |
| update | ~150 s | **~10-17 s (9-15x)** | ~8-12 s |
On H100 (old ~70-81 s/update) the same structure gives an estimated ~5-8 s/update; H100 benefits more from 2b because
its host step (~5-6 ms at B=128) is closer to launch-gap bound.

## 5. Verification (gates for each stage)
- Tool exactness, no tolerance: (a) per-step per-tool one-hot index and write rows bitwise equal to the old engine on
  the 32-prompt reference (`runs/engine_ref/engine_reference.json`, A40, greedy, 128 tok) and on teacher-forced replays
  of 512 sampled m1-style rows (old engine's `tool_feats`/`step` recomputed on the same token streams); (b) fuzz: random
  real-token windows; (c) state rings equal after every step.
- In-wire codes and IF write positions + vectors bitwise equal on the reference.
- Host: see section 6.
- Training: one update on a fixed rollout batch: loss, per-group grad norms, and parameter deltas vs the old trainer.

## 6. Open problem: host bit-exactness vs a host megakernel
The old host uses cuBLAS (split-K for some shapes, kernel choice depends on M) and Triton attention. A custom GEMM
generally cannot reproduce cuBLAS's accumulation order, so greedy tokens can diverge at near-ties (bf16 noise level).
Proposal: stage 2a keeps the exact old host kernels inside the step graph (greedy output bitwise = old engine; this
alone carries most of the speedup, because the tools are 93% of the rollout). Stage 2b (host megakernel: mma.sync GEMMs,
flash-decode attention, fused sampling, tools as concurrent work items) is then validated (i) bitwise where it can be
(test per GEMM shape whether a sequential-k mma kernel equals cuBLAS on the A40), else (ii) by logprob gaps at the bf16
noise floor plus a list of greedy divergence points, each at a near-tie. **Needs the user's decision** whether (ii) is
acceptable for the host part.

## 7. Files (A40 host:<workdir>/rl_library/engine2)
- `ref_env.py` builds the old engine from `train_library.py` (read-only exec up to its main loop).
- `prof_old.py` profile harness (this document's numbers); `stack_stats.py` stack structure analysis.
- next: `tk/` (stack compiler: ELL/int packing + interval analysis + collapse; NVRTC kernel source; loader), `eng2.py`
  (decode engine), `train2.py` (trainer), `test_exact.py` (gates of section 5). NOTEBOOK.md = measurement log.
