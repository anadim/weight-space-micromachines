# RL with implants v2 — results (live document)

Code: `../gsm_grpo/` (laptop copy; runs on A100 machine `../gsm_grpo/`, H100 machine `../gsm_grpo/`).

## 0. Where v1's decode time went (profile, A100 machine A100 80GB PCIe, GPU idle at start: 0 MiB, 0 %)

`prof_v1.py` on v1's own engine (`rl_implants/engine.py`: HF `model.model` 1-token forward + DynamicCache + calc_gen
GraphStep + Python loop). 8-shot prompt, ~1.1k tokens. Per decode step:

| component | bs 1 | bs 16 |
|---|---|---|
| host forward, 1 token, HF + DynamicCache (CUDA events) | **30.6 ms** | **47.7 ms** |
| CUDA kernels launched by that forward | 1,625 | 1,918 |
| implant step, eager (459 kernels: cuSPARSE csrmv/csrmm + index kernels) | 8.3 ms | 11.0 ms |
| implant step, v1 CUDA graph | 1.1 ms | 1.8 ms |
| lm_head + sample + log_softmax | 0.75 ms | 0.47 ms |
| Python bookkeeping (tolist + tok.decode stop check) | 0.1 ms | 0.3 ms |
| prompt prefill (recomputed for every batch; the 8-shot prefix is never reused) | 65 ms | 297 ms |
| **v1 `sample()` end to end** | 41.6 ms/step = **24 tok/s** | 51.3 ms/step = **207 tok/s** |
| v1 `sample()` without implant | 39.3 ms/step = 25 tok/s | 48.2 ms/step = 226 tok/s |

Reading: the step is **CPU/launch-bound in HF**: ~1.6-1.9k small kernels per token (per-layer DynamicCache `cat`,
mask building, rotary, norms as separate ops), ~19-25 us of host time each. The GPU is mostly idle. The implant is
not the problem in v1 (1-2 ms graphed). No contention: the GPU was empty when the bench started.

Theoretical floor for one decode step of the 0.6B host: bf16 weights 1.19 GB (0.44 GB non-embedding + 0.31 GB tied
embedding read once for the LM head) -> ~0.6 ms/step on this A100 (1.9 TB/s), ~0.36 ms on an H100 (3.35 TB/s), at
any batch up to ~100 rows (then KV reads and the LM-head GEMM take over). v1 is ~50-130x above that floor.

## 1. Engine throughput (fast engine, `engine.py`) — status 2026-11-18 ~09:00 CT

What the engine does: shared 8-shot prefix KV computed once per weight version (1,066 tokens); per-question prefill
once per prompt, KV copied to its G rows; static preallocated suffix KV cache (512 rows x 544 slots, 32 GB bf16);
the whole decode step (implant step + 28 host layers + LM head + Gumbel sampling + log-prob + stop rule + output
bookkeeping) captured as one CUDA graph per row bucket; host syncs once every 16 tokens; finished rows are compacted
into a smaller bucket. Host layers use Triton kernels (`hostkern.py`): add+RMSNorm, q/k-norm+RoPE+KV write, prefix
attention on tensor cores (prefix KV read once per 32 rows), suffix attention looping only over the row's own keys,
SiLU*mul, one-pass sampler; GEMMs stay cuBLAS.

Correctness (greedy, GSM8K-train dev B, 64 GSM + 64 scaled problems, text up to the stop):
| check | result |
|---|---|
| fp32 engine (torch kernels) vs HF fp32 + hook, plain | **64/64 identical** (HF bs1 vs bs32: 64/64) |
| fp32 engine (torch kernels) vs HF fp32 + hook, calc_s4 implant | **64/64 identical** |
| bf16 engine vs HF bf16, plain / calc | 31/64, 31/64 — but HF bf16 vs ITSELF at bs1 vs bs32 is 36/64, 37/64; engine vs HF-bs1 36/64, 37/64 (= the bf16 floor) |
| accuracy, bf16, same 64 | plain HF 37.5 / engine 37.5; calc HF 40.6 / engine 45.3 (noise at n=64) |

### 1a. Clean H100 numbers (GPU 0 idle; measured after the pod reboot, 16:15 UTC)

End-to-end rollouts (`check_equiv.py --bench`: real GSM prompts, T=1.0, max 320 new tokens, includes question prefill,
sampling, compaction, Python; tokens = generated tokens kept):
| engine | rows | no implant | calc_s4 implant |
|---|---|---|---|
| v1 (HF + DynamicCache + v1 graph implant), A100 | 16 | 226 tok/s | 207 tok/s |
| HF `.generate`, A100 (v1 bench) | 16 | 288 tok/s | — |
| v2, implant as torch step in the CUDA graph | 512 (64 x G8) | 39,100 | 15,300 |
| **v2, implant = one fused kernel** | 512 (64 x G8) | **38,800** | **19,800** |
| **v2, one fused kernel** | 1,024 (128 x G8, 2 waves) | 39,700 | 19,700 |
| v2 single stream (1 row) | 1 | 396 | 319 |
| v1 single stream, A100 | 1 | 25 | 24 |

So the implant arm samples at ~20k tok/s (target was 10k): ~96x v1's 207 tok/s; no-implant ~39k (~170x). In the RL
loop (rollout + packed training step), an update of 64 prompts x 8 takes ~7.5 s on the H100 (3.7 s rollout, 3.5 s
train) and ~15 s on the A100.

Decode step per bucket (`bench_step.py`, Triton host kernels, implant = fused kernel; cache length 256):
| rows | host only | host + implant | step tok/s (host only / with implant) |
|---|---|---|---|
| 1 | 2.09 ms | 3.03 ms | 480 / 330 |
| 16 | 2.36 | 3.25 | 6.8k / 4.9k |
| 64 | 2.83 | 4.09 | 22.6k / 15.6k |
| 256 | 4.88 | 7.69 | 52.5k / 33.3k |
| 512 | 7.74 | 12.42 | 66.1k / 41.2k |
(This bench feeds static tokens, so every row "needs" stack 1 every step: the worst case for the implant.)
Floor: ~0.36 ms/step (weights) at small batch; at 1 row we are at 2.1 ms = launch-bound (~10 kernels x 28 layers in
the graph); at 512 rows we are near the suffix-KV read bound (~5 ms at length 256).

Fused-kernel engine correctness, fp32, vs HF fp32 + the reference hook (64 dev problems, greedy): **64/64 identical
(plain) and 64/64 identical (calc_s4)**; same accuracy (35.9 / 43.75).

### 1b. The implant as one kernel (milestone)
`implant_persist.py`: reader + window + stack 1 (32 ReLU layers, 38k units) + hop + stack 2 (20 layers, 8.5k units) +
writer in ONE persistent Triton launch (grid = #SMs, software grid barrier between circuit layers, ELL tiles sorted by
fan-in). Exact work skipping: stack 1 runs only for rows whose token is '=' or whose window has no '='.
Bit-identity vs the reference implant step (`CalcImplant.step`, = `Micromachine.step`), real GSM8K token streams:
B=512 x 400 steps: **0 mismatching steps, 4,292 write events, state identical**; B=256 x 120: 0/30,720 row-steps.
(A first version raced on an in-place window shift across replicated warps: 39/30,720 wrong; fixed.)
`implant_triton.py` (one program per 4 rows, no grid barrier) is also bit-identical but slower.
Two bugs found and fixed on the way (both would bite any port): an in-place state shift inside a Triton block whose
layout is replicated across warps races (fix: load the whole window, barrier, store), and a vector `tl.atomic_add` on a
replicated layout runs once per replica (fix: one scalar atomic per program + cumsum for the compact row index).
Speed-ups that kept bit-identity: "deep" ELL tiles (2 units x 128 nonzeros per step) for the 2,600-3,875-input sums
(stack 1 alone at B=1: 4.3 ms -> ~0.6 ms); exact skipping of stack 1 for rows whose token is not '=' and whose
window has an '='; compaction of the rows that do need stack 1.

Per-step implant cost, H100 idle, teacher-forced GSM8K answer text (`time_implant.py`, 150 steps):
| batch rows | reference step, eager | reference step in a CUDA graph (v1 style) | **one fused kernel** |
|---|---|---|---|
| 1 | 3.67 ms | 0.81 ms | **0.26 ms** |
| 16 | 3.75 | 1.37 | **0.45** |
| 64 | 3.78 | 1.69 | **0.85** |
| 256 | 5.29 | 4.80 | **1.88** |
| 512 | 9.04 | 8.57 | **2.91** |
Not done: the stretch goal (the whole decode step, host + implant, as one persistent kernel). The host step is ~10
Triton/cuBLAS kernels per layer inside one CUDA graph; at 512 rows it is bound by reading each row's own KV.

## 1c. HP sweep (dev split B, n=800 = 400 GSM8K-train + 400 scaled; arm 2 = calc_s4 frozen; budget 32,768 rollouts each)

Coordinate sweep around lr 1e-6, G 8, 64 prompts/update, T 1.0, no KL, Dr.GRPO (64 updates). dev = greedy, implant on.
Dev@0 is 50.2 on the H100 and 49.1 on the A100 (bf16 kernels differ); compare within a GPU.

| run | GPU | updates | dev @0 / 16 / 32 / 48 / 64 | final GSM / scaled half | train reward 1st->5th fifth | implant-off dev at end |
|---|---|---|---|---|---|---|
| base lr 1e-6 | H100 | 64 | 50.2 / 53.8 / 57.2 / 62.6 / **64.2** | 76.0 / 52.5 | .29 -> .58 | 55.9 |
| lr 3e-7 | H100 | 64 | 50.2 / 50.1 / 49.9 / 52.2 / 55.2 | 69.2 / 41.2 | .27 -> .41 | 49.4 |
| lr 3e-6 | H100 | 64 | 50.2 / 60.0 / 62.8 / 64.8 / 63.0 | 72.2 / 53.8 | .37 -> .65 | 51.2 |
| lr 1e-5 | H100 | 64 | 50.2 / 58.0 / 52.5 / 46.4 / 53.8 | 58.5 / 49.0 | .46 -> .48 (collapses) | 29.0 |
| T 0.8 | A100 | 64 | 49.1 / 52.1 / 58.4 / 60.4 / 64.1 | 77.5 / 50.8 | .38 -> .58 | 55.9 |
| KL 1e-3 | A100 | 64 | 49.1 / 53.5 / 56.6 / 60.5 / 62.4 | 76.8 / 48.0 | .27 -> .57 | 54.4 |
| CISPO | A100 | 64 | 49.1 / 53.4 / 58.8 / 59.5 / 62.9 | 76.8 / 49.0 | .28 -> .58 | 55.2 |
| G 16 (same rollouts, 32 updates) | H100 | 32 | 50.2 / 49.8@8 / 53.5@16 / 56.9@24 / 60.5@32 | 74.2 / 46.8 | .27 -> .51 | 53.6 |
| 32 prompts/update | H100 | stopped at 25 | — | | | |

Pick: **lr 1e-6, G 8, 64 prompts, T 1.0, no KL, Dr.GRPO.** lr 3e-6 ties at 64 updates but peaks at u48 and lowers
implant-off ability; 1e-5 collapses. T 0.8, KL 1e-3 and CISPO are within noise; G 16 at the same rollout budget is behind (half the updates) of the base at the end (SE of an 800-item
dev number is about 1.7 pp). The prompts-per-update (32) run and the arm-3 link-lr runs were stopped when the user
asked for the ablations below (top priority).

## 2. Ablations requested by the user (dev split B only; test once at the end)

Recipe for all: G 8, 64 prompts, 64 updates, T 1.0, Dr.GRPO, no KL, dev n=800, lr 1e-6 (and 3e-6), seeds 0 and 1.
Every implant arm runs the fused one-kernel implant.

**Arm R (the "just more weights" control), as built:** calc_s4 with exactly the same reader, window, stack 1,
stack 2, hop, gain and write layer, frozen, but the writer rows for the ten digit tokens permuted by a fixed derangement
(seed 12345). It therefore fires at exactly the positions calc_s4 fires (trigger path untouched) and writes a digit
string of the right length and format, but the digits are wrong. Base host, dev-400 greedy probe: none 46.0, calc_s4
51.25 (writes 15.4 / calls 2.43 per answer), R 8.75 (writes 18.5 / calls 2.40).
What I tried first and rejected: shuffling each layer's nonzero values among its own nonzeros (stack 1, stack 2 or
both). With any of the three the circuit fires on every token (319-320 writes per answer, output "!!!!...", 0% dev),
so the trigger path does not survive a value shuffle; it cannot serve as a matched-firing control.

### 2a. Runs on the H100 (finished before the pod rebooted at ~16:07 UTC; kept, but the main comparison uses the A100)
dev @0/16/32/48/64 (implant on); implant-off dev in brackets; writes/calls per answer at u64:
- a1_lr1e-6_s0 (no implant): 44.0 / 45.1 / 51.1 / 53.2 / 54.4 (GSM 73.8, scaled 35.0)
- aR_lr1e-6_s0: 7.0 / 13.5 / 28.6 / 33.6 / 36.9 [44.0 / 43.9 / 42.6 / 41.9 / 45.9]; writes 19.0 -> 2.8, calls 2.51 -> 0.39
- aR_lr1e-6_s1: 7.0 / 14.4 / 29.5 / 31.8 / 36.4 [44.0 / 42.9 / 42.2 / 38.6 / 42.8]; writes 19.0 -> 2.1, calls 2.51 -> 0.32
- a2_lr3e-6_s0: 49.9 / 59.6 / 61.2 / 66.2 (stopped at u49 by the reboot, no checkpoint) [44.0 / 52.5 / 50.0 / 49.6]

### 2b. MAIN COMPARISON (all on A100 machine A100; lr 1e-6, G 8, 64 prompts, 64 updates, T 1.0, Dr.GRPO, no KL)

Dev = split B, n=800 (400 GSM8K-train + 400 scaled), greedy. TEST = GSM8K test (1,319) and GSM-Hard (1,319; "201+" =
original index >= 200, the part not used to pick the 8 shots), each read ONCE per model, final checkpoint (u64),
greedy, run_gsm protocol. "on" = the arm's implant attached; "off" = same weights, implant detached. For arm 1 the
"off" column is the model itself.

| run | GPU | dev @0 / 32 / 64 (implant on) | GSM / scaled half @64 | implant-off dev @0 / 64 | writes / calls per answer @0 -> @64 | TEST GSM8K on / off | GSM-Hard on / off (201+ on / off) |
|---|---|---|---|---|---|---|---|
| a1_lr1e-6_s0 | A100 | 43.6 / 51.6 / 54.5 | 74.0 / 35.0 | 43.6 / 54.5 | 0.0 / 0.00 -> 0.0 / 0.00 | — / 65.0 | — / 36.7 (— / 35.7) |
| a1_lr1e-6_s1 | A100 | 43.6 / 50.0 / 53.9 | 74.5 / 33.2 | 43.6 / 53.9 | 0.0 / 0.00 -> 0.0 / 0.00 | — / 65.1 | — / 37.9 (— / 37.4) |
| a2_lr1e-6_s0 | A100 | 48.9 / 58.0 / 65.2 | 77.8 / 52.8 | 43.6 / 56.2 | 15.5 / 2.43 -> 17.4 / 2.75 | 67.2 / 66.4 | 47.6 / 38.8 (45.8 / 38.1) |
| a2_lr1e-6_s1 | A100 | 48.9 / 57.4 / 63.8 | 76.0 / 51.5 | 43.6 / 55.5 | 15.5 / 2.43 -> 16.8 / 2.74 | 66.6 / 66.1 | 47.4 / 37.1 (46.0 / 36.6) |
| aR_lr1e-6_s0 | A100 | 7.5 / 27.4 / 36.6 | 51.0 / 22.2 | 43.6 / 44.8 | 18.9 / 2.47 -> 2.1 / 0.29 | 48.6 / 58.8 | 26.2 / 31.2 (26.1 / 30.9) |
| aR_lr1e-6_s1 | A100 | 7.5 / 29.0 / 37.9 | 52.8 / 23.0 | 43.6 / 45.2 | 18.9 / 2.47 -> 2.1 / 0.31 | 47.1 / 56.2 | 26.3 / 30.8 (25.5 / 29.8) |

Base model (no RL), same harness, per GPU:
| model | GPU | TEST GSM8K | GSM-Hard all (201+) | writes per question (test) | calc bursts exact / wrong (test) |
|---|---|---|---|---|---|
| base + R (scrambled digits) | A100 | 10.9 | 5.3 (5.4) | 13.4 | 13 / 2719 |
| base, no implant | A100 | 52.5 | 26.4 (25.6) | 0 | — |
| base + calc_s4 | A100 | 53.2 | 33.5 (32.0) | 9.6 | 2300 / 254 |


Paired differences on TEST (per-question, the two seeds averaged per question; 95% CI from a paired bootstrap over
questions, so seed-to-seed variance is NOT in the interval; the two seeds differ by <=0.6 pp on test for arms 1/2):
```
test  arm2(on) - arm1         +1.86 pp  95% CI [+0.42, +3.26]  (n=1319, seed-averaged, paired bootstrap)
test  arm2(off) - arm1        +1.21 pp  95% CI [-0.04, +2.46]  (n=1319, seed-averaged, paired bootstrap)
test  arm1 - base            +12.51 pp  95% CI [+10.08, +14.94]  (n=1319, seed-averaged, paired bootstrap)
test  arm2(on) - base+calc   +13.68 pp  95% CI [+11.18, +16.26]  (n=1319, seed-averaged, paired bootstrap)
test  armR(off) - arm1        -7.58 pp  95% CI [-9.74, -5.53]  (n=1319, seed-averaged, paired bootstrap)
test  armR(on) - arm1        -17.21 pp  95% CI [-19.56, -14.82]  (n=1319, seed-averaged, paired bootstrap)
hard  arm2(on) - arm1        +10.20 pp  95% CI [+8.42, +11.98]  (n=1319, seed-averaged, paired bootstrap)
hard  arm2(off) - arm1        +0.68 pp  95% CI [-0.64, +2.05]  (n=1319, seed-averaged, paired bootstrap)
hard  arm1 - base            +10.92 pp  95% CI [+8.61, +13.23]  (n=1319, seed-averaged, paired bootstrap)
hard  arm2(on) - base+calc   +13.99 pp  95% CI [+11.71, +16.49]  (n=1319, seed-averaged, paired bootstrap)
hard  armR(off) - arm1        -6.29 pp  95% CI [-8.23, -4.36]  (n=1319, seed-averaged, paired bootstrap)
hard  armR(on) - arm1        -11.07 pp  95% CI [-13.08, -8.95]  (n=1319, seed-averaged, paired bootstrap)
```

Calculator exactness after RL (dev items only, `diag_exact.py`, 300 problems; the number after the last '=' vs the exact
value of the whole expression before it): base host 517 exact / 13 wrong; arm 2 seed 0 after RL 529 / 19; seed 1
522 / 15. The circuit is frozen, so its function is unchanged; every "wrong" case I read is either a negative result
(the circuit writes the sign and stops: "25 - 771626.5 = -7", the model then writes the digits itself or not at all),
an operator the circuit reads differently from Python (a chain where the model's own earlier "=" is inside the
expression), or "(a / 2) + 6" style brackets. The test-set "bursts exact/wrong" column of the base table uses a cruder
checker (last binary op only) and overcounts "wrong" on chains like "8 + 6 + 3 = 17"; use the dev numbers.

**What the table says**
1. RL alone is large here: arm 1 (no implant) GSM8K test 52.5 -> 65.0 / 65.1 (+12.5 pp), GSM-Hard 26.4 -> 37.3.
2. The calculator during RL adds a little on GSM8K and a lot on GSM-Hard: arm 2 with its implant 67.2 / 66.6
   (+1.9 pp vs arm 1, CI [+0.4, +3.3]); GSM-Hard 47.6 / 47.4 (+10.2 pp vs arm 1, CI [+8.4, +12.0]).
3. Arm 2's host does not get worse at arithmetic by leaning on the implant: implant off, arm 2 is 66.4 / 66.1 on GSM8K
   (+1.2 vs arm 1, CI [-0.0, +2.5]) and 38.8 / 37.1 on GSM-Hard (+0.7, CI [-0.6, +2.1]). Its host-only dev also rose
   43.6 -> 56.2 / 55.5, slightly more than arm 1 (54.5 / 53.9).
4. The "just more weights" control (arm R: same circuit, same trigger, wrong digits) does not help, it hurts: the host
   learns to stop triggering it (writes 18.9 -> 2.1 per answer, calls 2.47 -> 0.3), but ends far below arm 1 both with
   the implant (48.6 / 47.1) and without it (58.8 / 56.2; -7.6 pp vs arm 1). So arm 2's gain needs correct values.
5. Arm 2 keeps calling: writes per dev answer 15.5 -> 17.4 / 16.8, calls 2.43 -> 2.75 / 2.74.
Caveats: 64 updates only (the dev curves of arms 1 and 2 are still rising; 500-update runs are running, §2c);
two seeds; one recipe tuned on arm 2.

### 2c. Longer runs (in progress)
500 updates, same recipe, dev-800 every 25 updates (implant on and off), all in `runs/long/*/steps.jsonl`:
- A100 machine A100, sequential: a2_u500_s0 -> a1_u500_s0 -> a2_u500_s1 -> a1_u500_s1 (checkpoints at 250 and 500).
- H100 machine H100 GPU 0: a2_lr1e-6_s0 (500) -> a1_lr1e-6_s0 (500) -> the two arm-3 runs killed by a GPU collision -> arm 4.
  H100 arm 2 (done, 500 updates, 54 min): dev 49.75 @0, 61.4 @50, 64.5 @100, 66.6 @150, 68.75 @200, 67.6 @250, 70.0 @300,
  70.9 @350, 70.9 @400, **71.5 @450**, 70.4 @500; implant-off dev 44.0 @0 -> 55.9 @500.
  Order on the H100 now (coordinator): arm 4 seed 1 -> arm-3 reruns -> long arm 1.

## 3. Arm 4: calculator + compiled mistake detector / rethink (frozen)

`rethink_step.py`: `micromachines/rethink_circuit_v2.py` in the configuration behind the repo's "calculator + gated
rethink" GSM8K row (fitted detector `results/rethink_v2/detector_hybrid.pt`: 1-hidden-layer MLP on layers 12-24 at the
token before the first `####`; intervene = hybrid; cap 300) rewritten as a one-token decode step with per-row registers
(the whole-history hard-attention heads become "value at the most recent event" registers; exact while sequences are
< 3,333 tokens). It runs inside the CUDA graph after the calculator (the circuit's LM-head readout reuses the host
logits; one extra LM-head GEMM per step for rows it writes).
Check vs the HF reference with both hooks (`check_rethink.py`, fp32, 32 dev problems, max 500 new tokens):
**32/32 identical text**, rethink in 25/32 answers in both, same accuracy (43.75).
RL with arm 4: tokens the rethink circuit forces are treated as environment (advantage 0 at those positions);
reward and dev use the LAST `####` (as `run_gsm_v2.py`); max 500 new tokens (arms 1/2: 320). Calculator = calc_s4
(the repo's row used calc_s3). Smoke (16 dev items): rethink fires on 69% of answers at u0; rollout + backward run.
Running: 500 updates, max 700 new tokens, seed 0 on A40 machine GPU 1 (A40, B_max 128), seed 1 on H100 machine GPU 0 after the long
arm 2. Step 0 (no RL, A40, dev-800): implants on **52.1** (GSM 67.0 / scaled 37.25), implants off 43.9 (arm 2 at u0:
48.9 A100 / 49.75 H100 / 49.2 A40). Rethink fires on 68.4% of answers (29.9% before the first `####`, 38.5% after
it); 7.0% of answers change after the rethink, 2.1% fixed, 0.75% broken; 276 tokens per answer (calc only ~113).

Arm 4 seed 1 (H100) dev curve, on / off, rethink fire rate (before `####` / after), and of all answers changed / fixed /
broke after a rethink (live: `python summ4.py runs/long/a4_u500_s1`):
```
u  0 on 50.6  off 44.0  fire 67.9% (29.0 / 38.9)  changed 9.2  fixed 1.6  broke 1.8  len 283
u 25 on 57.5  off 48.9  fire 58.5% (24.8 / 33.8)  changed 6.4  fixed 2.0  broke 0.9  len 258
u 50 on 58.2  off 51.4  fire 50.1% (19.8 / 30.4)  changed 4.2  fixed 0.8  broke 0.9  len 232
u 75 on 61.6  off 53.8  fire 38.4% (14.1 / 24.2)  changed 2.5  fixed 0.8  broke 0.5  len 206
u100 on 64.9  off 56.0  fire 17.4% ( 8.9 /  8.5)  changed 1.2  fixed 0.6  broke 0.0  len 175
```
Same GPU, arm 2 (calc only): u50 61.4, u100 64.5. RL drives the rethink rate down (68% -> 17% of answers); fixes stay
at 0.6-2% of answers.

Arm 4 seed 1 to u300 (stopped at u300 by the cap): u275/u300 dev in `runs/long/a4_u500_s1/steps.jsonl`;
u250: on 69.0, off 55.4, rethink fires 2.8% (2.2% before `####`, 0.5% after), changed 0.2%, fixed 0.1%.
RL switches the rethink almost off (68% -> 3% of answers by u200).

### 3a. Is RL gaming the mistake detector? (`det_audit.py`, H100, dev-800)
The host (+ calc) writes its FIRST answer with no rethink attached; the frozen hybrid detector is read at the circuit's
own read point (the token before the first `####`, layers 12-24). "fire" = the circuit's rule there (hash head sB > thB,
or end head sA > thA and not hash).

| checkpoint | acc first answers | AUROC sB / sA / P(correct) | fire on WRONG / RIGHT | hash-fire WRONG / RIGHT |
|---|---|---|---|---|
| base | 52.1 | 0.80 / 0.74 / 0.90 | 85.6% / 60.4% | 29.2% / 4.3% |
| arm 4 s1 u100 | 64.7 | 0.74 / 0.62 / 0.83 | 47.5% / 27.7% | 15.8% / 2.2% |
| arm 4 s1 u200 | 68.6 | 0.63 / 0.51 / 0.80 | 27.9% / 22.3% | 6.6% / 0.2% |
| arm 4 s1 u300 | 71.9 | 0.66 / 0.52 / 0.79 | 33.6% / 29.9% | 6.0% / 0.7% |
| arm 2 s0 u100 (no detector in RL) | 64.9 | 0.70 / 0.59 / 0.81 | 29.6% / 16.0% | 9.8% / 0.8% |
| arm 2 s0 u200 | 69.5 | 0.69 / 0.58 / 0.80 | 29.6% / 15.2% | 8.8% / 0.7% |
| arm 2 s0 u300 | 70.6 | 0.65 / 0.52 / 0.77 | 15.8% / 11.4% | 4.7% / 0.4% |

The detector's discrimination falls under RL (sB 0.80 -> 0.66; sA to chance) and its fire rate on wrong answers
collapses, but it falls the same way for arm 2, whose RL never saw the detector. So the main effect is distribution
shift of a probe fit on the base model's states, not evasion learned because the rethink was attached. Open point: in
arm 4's own rollouts the after-the-answer rethink fell to 0.5% while the end head would still pass on ~25% of first
answers here; that path also needs the circuit's "host is about to end" readout, so RL may have changed how answers end.
Not checked yet.

Arm 4 at u300 (H100, both seeds): s0 on 71.0 / off 57.1, rethink fires 3.8%, fixed 0.0%; s1 on 70.5 / off 55.6, fires
3.4%, fixed 0.0%. H100 arm 2 s0 at u300: 70.0. Calc + frozen rethink + RL ends level with calc + RL on dev.

### 3b. Arm 4c: co-trained detector (queued; built and smoke-tested)
Rollouts draw each rethink decision at the read point as Bernoulli(sigmoid((score - th)/T)), T = 1 (both heads); eval
uses the hard threshold (the same code path as arm 4 when stochastic mode is off). The read-point features (layers
12-24, 7,168 numbers) and the decisions are recorded per row; after each host update the detector MLP (W1, b1, WB, WA,
W0; normalizer fixed) takes an Adam step (lr 1e-4) on -A * log p(decisions), A = that rollout's GRPO advantage.
4c adds BCE of the "right as it stands" head W0 vs first-answer correctness (rows that did not rethink before their
first answer). 4cRL = the policy-gradient term only. Every dev eval also logs the current detector's AUROC on the
current host's first answers (read-only pass), fire on wrong / right, and fire / fixed / broke.


## 4. 300-update comparison on the H100 (main result; user capped runs at 300 updates)

Recipe as in §2 (lr 1e-6, G 8, 64 prompts, T 1.0, Dr.GRPO), 300 updates, 2 seeds per arm, all on H100 machine GPU 0,
one-shot test at the u300 checkpoint. Arm 2 seed 0 = the long H100 run's u300 checkpoint. Arm 4 = calc_s4 + compiled
gated rethink (frozen), max 700 tokens, last `####`; its "off" column runs without both implants at 320 tokens.

| model (H100, u300) | GSM8K on / off | GSM-Hard on / off | GSM-Hard 201+ on / off | rethink frac (test on) |
|---|---|---|---|---|
| arm 1 s0 | — / 68.2 | — / 42.4 | — / 41.7 | — |
| arm 1 s1 | — / 70.1 | — / 43.7 | — / 42.8 | — |
| arm 2 s0 | 68.8 / 68.0 | 54.7 / 40.2 | 53.1 / 39.4 | — |
| arm 2 s1 | 70.7 / 70.1 | 56.0 / 39.3 | 54.2 / 39.2 | — |
| arm 4 s0 | 69.5 / 71.0 | 56.8 / 41.1 | 54.5 / 40.4 | 4.6 |
| arm 4 s1 | 66.0 / 68.7 | 56.6 / 40.7 | 54.3 / 40.0 | 4.2 |
| base + calc + rethink (no RL) | 54.3 / 52.8 | 33.7 / 26.6 | 32.1 / 25.6 | 67.3 |
| base + calc (no RL); off = base | 53.6 / 52.8 | 33.5 / 26.6 | 31.6 / 25.6 | — |

test  arm2(on) - arm1           +0.57 pp  95% CI [-1.06, +2.16]
test  arm2(off) - arm1          -0.15 pp  95% CI [-1.74, +1.44]
test  arm4(on) - arm2(on)       -2.01 pp  95% CI [-3.71, -0.38]
test  arm4(on) - arm1           -1.44 pp  95% CI [-3.22, +0.34]
test  arm4(off) - arm2(off)     +0.83 pp  95% CI [-0.72, +2.27]
hard  arm2(on) - arm1          +12.24 pp  95% CI [+10.12, +14.29]
hard  arm2(off) - arm1          -3.30 pp  95% CI [-4.93, -1.78]
hard  arm4(on) - arm2(on)       +1.36 pp  95% CI [-0.08, +2.77]
hard  arm4(on) - arm1          +13.61 pp  95% CI [+11.45, +15.69]
hard  arm4(off) - arm2(off)     +1.14 pp  95% CI [-0.34, +2.65]


Dev-800 at u300: arm 1 59.9 / 62.8; arm 2 70.0 / 69.4 (u225); arm 4 71.0 / 70.5. Implant-off dev of arm 2 stays at
~55-58 from u100 on, while arm 1 keeps rising to 60-63. (A100 extra pair: arm 2 s0 71.6, arm 1 s0 63.1 at u300; the
A100 arm 2 s0 test eval used its u250 checkpoint, the only one before the cap.)

**What it says**
1. RL alone does most of the work on GSM8K: 52.8 -> 68.2 / 70.1 (+16 pp).
2. With 300 updates the calculator adds nothing on GSM8K (+0.6 pp, CI [-1.1, +2.2]) but +12.2 pp on GSM-Hard.
3. Leaning on the tool: arm 2's host without the calculator is 3.3 pp WORSE than arm 1 on GSM-Hard (CI [-4.9, -1.8]);
   at 64 updates it was +0.7. RL with the calculator attached trains a host that does less large-number arithmetic itself.
4. The frozen rethink does not help: arm 4 vs arm 2 with implants -2.0 pp on GSM8K (CI [-3.7, -0.4]), +1.4 pp on
   GSM-Hard (n.s.). RL switches the rethink off (fires on ~4% of test answers).

## 5. Arms 4c / 4cRL / 4s: trainable detector, trainable rethink phrase (in progress; dev-800 only so far)

Dev-800 at u300 (implants on), for reference: arm 2 70.0 / 69.4, arm 4 71.0 / 70.5.

**4cRL (PG on the detector only), dev on at u300:** s0 (A100) 70.4, s1 (H100) 71.2.
Fire at u300 is 13.9% / 11.4%. Changed is 0.0 / 0.9%, and fixed is 0.0 / 0.2%.
The sB AUROC goes 0.79 -> 0.77 (s0) and -> 0.62 (s1).
With PG only, s1's detector fires on 100% of both right and wrong answers at u275-300, so it no longer separates them.
The rethink still gives nothing measurable.

**4sA s0 (H100; soft rethink, init = the phrase embeddings, Adam 1e-3; detector trained as in 4c), 300 updates:**

| u | dev on | dev off | fire | changed / fixed / broke | AUROC sB / s0 | detector fire on wrong / right |
|---|---|---|---|---|---|---|
| 0 | 50.1 | 44.0 | 67.9 | 10.6 / 1.6 / 2.2 | 0.79 / 0.90 | 85 / 59 |
| 100 | 64.5 | 55.8 | 31.1 | 4.0 / 1.0 / 0.2 | 0.75 / 0.86 | 97 / 77 |
| 200 | 71.5 | 60.1 | 15.6 | 1.0 / 0.4 / 0.0 | 0.71 / 0.85 | 97 / 94 |
| 300 | 71.4 | 57.5 | 14.1 | 0.5 / 0.2 / 0.0 | 0.63 / 0.80 | 77 / 60 |

Dev on, mean of u200-u300: 4sA 71.1 vs arm 4 s0 69.6 (one seed, dev only, so not significant).
The rethink still fires on 14% of answers but changes almost none (0.5%), and fixes stay at ~0.2%.
The end path is off (0.6%), and the hash path stays on (13.5%).
The sB AUROC drops 0.79 -> 0.63. The s0 head (BCE refit) holds at 0.80-0.89.

Soft vectors (L2 drift from init, which has norm ~0.83-1.0; the own-token cosine at u300 is 0.77-0.91).
At u300 every vector's nearest token is still its init token. The neighbours do not change:
'Hmm' 0.46 (0.88), ',' 0.46 (0.89), ' wait' 0.43 (0.91), '.' 0.54 (0.85), ' Let' 0.49 (0.86), ' me' 0.43 (0.91),
' rethink' 0.58 (0.87, norm 1.00 -> 1.16), ' this' 0.53 (0.84), '.\n' 0.67 (0.77).
The vectors move steadily, at about 0.4 by u100. That movement is mostly a norm increase (+5-16%) plus an
off-vocabulary component. They do not turn into other words.
