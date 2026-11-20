> **STATUS NOTE (2026-11-20 ~20:50 UTC, rollout-speed agent).** Arms A, B (H100), A-reset (A40:1), A' (A100) are
> UNTOUCHED, still on the old engine, and run_rl.sh is unchanged. **Arm C (A40:0) is STOPPED, resume state /dev/shm/rlstack_C.pt (u=30).**
> On restart it fails at u31 in train(): torch inductor flex_attention compile "No valid triton configs ... Required 107008
> Hardware limit 101376" (A40 shared memory). This happens with a fresh inductor cache too. Do not blind-relaunch it; it needs a
> GPU with more smem (H100/B200) or a flex kernel-options fix. Fast rollout loop (code/fastroll.py) is written but NOT deployed;
> see the engine agent's report.

# RL with the all-weights single-call implant stack attached (2026-11-20)

STATUS: RUNNING. This file is updated as the arms progress.

## 1. Audit: what is weights and what is not

These are the components of S3-serial, the stack that scored 80.44 on GSM8K test (FINAL.md §7), and what replaced each
one for RL.

| # | component | in S3-serial (test 80.44) | in the RL stack |
|---|---|---|---|
| 1 | calc_s4, the original "=" calculator | compiled | compiled (unchanged) |
| 2 | calc v2 | **Python hook** (calc2.Row token tracker) | **compiled chain2** (calc_v2/chain2.py, implant2.DualImplant; +/- chains). calc_s4 is masked where chain2 writes. |
| 3 | sanity trigger and repair | tensor logic with token replacement (serial_stack._act) | compiled: the 34-state DFA as a fixed table, a pointer-register phrase writer (gain · DIR residual writes), and fixed one-hot counters |
| 4 | forget | weight-space (forget_ws) | weight-space (forget_ws: fixed extra un-rotated q/k dims, W_K = I, W_Q = -M(J - I), M = 1e4; segment labels from a compiled counter) |
| 5 | position reset | **programmatic** (RoPE offset) | **dropped** in the all-weights arms; kept only in the labeled reference arm A-reset |
| 6 | retry writer | token replacement | compiled pointer writer, triggered by an input-token END: answer closed, stop token read as input, or segment cap |
| 7 | stop control | the stop token was replaced | generator stop rule = "the implant wrote eos". Host stop tokens are fed back as inputs and read by the END trigger. |
| 8 | answer register | float64 accumulator plus token ids | side state written by fixed tables: per slot, 16 symbol one-hots over {0-9, '-', '.', pad}, plus closed / impossible bits and the probe logit |
| 9 | voter | torch where / argmax | fixed ReLU circuit: pairwise 16-symbol equality relu(Σ − 15), probe-weighted score, sanity gate, pairwise-comparator argmax with earliest tie-break |
| 10 | final `#### Y` writer | token replacement | pointer writer: '####' ' ' + winner symbols (one-hot @ DIR) + eos |
| 11 | probe (L18) | fixed linear map | fixed linear map |
| – | **not weights (generator)** | decode loop, T = 1 sampling, token budget, stop rule | the same |

Code: `final/comp_serial.py` (SerialWD), `final/stackh.py` (arm B), `final/rl_hook.py`, `final/run_rl.sh`.

## 2. Exactness and equivalence (before RL)

- **Reader fuzz** (`sw_fuzz_hook.py`), on 20,000 random `#### number` texts with sign, commas, decimals, `$` and six
  different tails: 0 wrong values, 0 missed, 0 wrong sanity flags.
- **Voter fuzz**, on 81,920 decisions at N = 4 / 8, count and probe modes, including exact probe ties: 0 cases where the
  circuit picks a different answer than the reference vote.
- **Compiled vs Python serial machinery**, run with the same calculator (calc_s4 only), N = 4, greedy, reset on:

  | | dev | dev2 |
  |---|---|---|
  | accuracy, Python vs compiled | 82.83 vs 82.83 | 79.33 vs 79.00 |
  | final outcome equal | 574 / 600 | 293 / 300 |
  | per-attempt answers equal | 2214 / 2400 | 1099 / 1200 |

  - The one intended difference: a host stop token is now written. It is then read as input and hidden by the forget.
  - After that extra hidden key, the streams diverge only at bf16 near-ties. 16 of the first 32 rows are token-identical
    once stop tokens are stripped.
- **chain2**: exact on 10k per form, and 0 silence violations (calc_v2.md §7).

## 3. Position reset: dropped, and what that costs at u0

All-weights stack (chain2), T = 1:

| host | N = 8 no reset | N = 8 with reset (programmatic) | per-attempt accuracy, no reset (dev) |
|---|---|---|---|
| RL+calc a2_u300_s2 | dev 81.8 / dev2 77.7 | 89.0 / 86.3 | 81 → 71 → 62 → 57 → 45 → 44 → 37 → 28 |
| base u0 | 26.8 (dev) | 60.7 (dev) | collapses |

Without the reset, every later attempt sits further from the question in RoPE positions and gets worse. At N = 4 the gap
is smaller: 84.3 / 79.7 without the reset. The RL question is whether training the host closes this gap.

## 4. Arms

All arms use the a2_u300 hyper-parameters: lr 1e-6 with warmup 5, AdamW (0.9, 0.95), Dr.GRPO, clip 0.2, KL 0, G = 8,
T = 1, grad-norm clip 1.0, split-A pool (gsm + scaled).

Shared training details:
- Teacher forcing: every segment is its own packed document (question + segment), with the rollout's RoPE positions.
- Implant-written tokens are masked out of the loss.
- The importance ratio at the first update is 1.000 on every arm, so the reconstruction matches the rollout.
- Resume state goes to /dev/shm every 5 updates; on a crash the run restarts from it (up to 6 tries).

| arm | host start | stack | prompts / update | GPU |
|---|---|---|---|---|
| A | a2_u300_s2 | all weights, N = 4, no reset, reward = final vote | 64 | H100:0 |
| A' | a2_u300_s2 | as A, plus attempt-level advantage (each attempt's own answer vs the group's mean attempt reward) | 64 | A100 |
| B | a2_u300_s2 | S2F control, compiled: S2Fn (input-token sanity + forget + " We need to find") + chain2, one attempt | 256 (token-matched) | H100:1 |
| A-reset (reference) | a2_u300_s2 | as A, plus the programmatic position reset | 16 (A40 memory) | A40:1 |
| C | base u0 | as A (no reset) | 16 (A40 memory) | A40:0 |

## 5. Dev curves

(filled in as the runs progress)

## 6. Test

(only at the end)
