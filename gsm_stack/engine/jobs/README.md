# engine/jobs

Development checks for the engine, adapted to the single-GPU launchers (`gsm_stack/env.sh`):

| job | what |
|---|---|
| `arm_smoke.sh NAME` | fresh arm in a scratch `ARMDIR`, `U` updates (0 = only the u0 dev eval); prints the first update's teacher-forcing ratio |
| `train_bench.sh` (`SRC=ARM`) | one rollout from a copy of arm `SRC`'s resume state, then `stk/train_bench.py` (reference vs fast trainer: time, gradient agreement) |
| `ratio.sh` | teacher-forcing ratio of the first update after a fastroll rollout (base host) |
| `tattn_test.py` | `stk/tattn.py` (Triton group attention fwd + bwd) vs a dense fp32 reference and flex attention, timing at L = 16k (`STK_DIR=gsm_stack/engine/stk python3 tattn_test.py`, GPU) |

The other development jobs (profilers, release gates against older releases) were specific to the original cluster
and are not shipped.
