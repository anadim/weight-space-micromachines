# Compliance: what is in the weights

**Rule** for every compliant GSM8K number: one generation call on a frozen host. Every decision about the text is made
by weights: the host's own weights, compiled implant circuits (fixed tables and ReLU layers), fixed registers that
fixed weights update as tokens go by, and fixed linear read-outs and residual / logit writers. Nothing outside the
weights may look at the text and change what happens, **except the generator** (below). Numbers that break the rule
are kept only as references and are always labeled **NON-COMPLIANT** or **programmatic baseline**.

The audit read `comp_serial` (`_sw`, `_step_triton`, the SerialW done rule), `stackh`, `forget_ws`, `attn3`,
`fastroll`, `c2_engine`, the wiring engine's `_book` and `rl_hook` (`results/RL_RESULTS.md` §5,
`implants/aw/README.md`).

## In the weights

| part | how it is weights | code |
|---|---|---|
| host | Qwen3-0.6B-Base; for the RL rows, its weights after GRPO (documented commands, `MODEL_CARD.md`) | Hub + `engine/run_arm.sh` |
| calculator `calc_s4` (+ its veto) and calculator v2 `chain2` (+/− chains) | compiled ReLU circuits that read the digits the model writes and write the result into the residual stream; the veto is a solved linear map | `micromachines/`, `engine/wiring/implant.py`, `engine/stk/chain2.py`, `implant2.py`, `tk*.py` (the same circuits as one integer-exact kernel) |
| answer-sanity DFA + phrase writer | a 34-state DFA as fixed token × state tables; a pointer register and fixed residual writes emit " We need to find" | `engine/stk/san_engine.py`, `comp_sanity.py` |
| compiled missing-answer trigger (`FE_MISS=w`) | in the armed register state a fixed logit write moves the stop tokens' mass onto the phrase's first token | `stackh.py _miss_w` |
| weight-space forget | 4 extra fixed q/k dims per head keyed on a segment counter: keys of earlier attempts get −10,000, so later attempts cannot see them (token-identical to a masked kernel, 600/600 and 300/300 dev rows) | `engine/stk/forget_ws.py`, `attn3.py` |
| serial registers: answer register, attempt counter, probe readout, ReLU voter, final writer | fixed tables + tensor ops on registers each step; the probe is one fixed linear map at layer 18 (`gsm_push/final/verifier_probe_L18.pt`, 6 KB); no host sync, no data-dependent Python | `comp_serial.py` |
| weight-space position reset (WPR) | an offset register and a fixed fp32 rotary table; q and the new K row are rotated by R(−off) after the host RoPE in every layer | `implants/wpr`, `comp_serial.py` (`WPR=1`) |
| adaptive tries (rung 2) | comparators on the answer / probe registers and counters (`torch.where` / ReLU) | `aw_adapt.py` |
| answer-group voter (rung 3) and side-net score (rung 4) | a number-set register automaton (fixed token-class tables) and fixed linear scores fit offline on GSM8K-train split C | `aw_vf.py`, `numreg.py`, `aw_sn.py` |

Fitted pieces (probe, voter, side-net, veto) were fit once, offline, on GSM8K **train** data (split C for the voters,
never dev or test), then frozen. They are weights in the same sense as a compiled circuit.

## Generator-only exceptions (allowed, disclosed)

1. The decode loop, the T = 1 sampling kernel and its RNG, and the token budget (maximum length).
2. The stop rule: stop when the stack's register wrote eos, or at the budget (serial stack); the single-attempt S2F
   stack (arm B, StackHD) stops on the host's own stop tokens, a standard generator stop. The budget grows with the
   number of phrase writes.
3. Scheduling: fastroll admission / compaction / harvest and slot moves. They only change which batch slot a row uses
   (and so its RNG stream position); rows are bitwise identical to the plain loop under greedy decoding.
4. The TK kernels' exactness flag is only printed.
5. Output decoding and grading (regex on the last `#### n`): evaluation, not inference.
6. Training-time only, not part of the deployed stack: the RL reward, advantages and `RL_DIV`.

**Flagged and fixed.** In rel/v4.2 (`eval/trees/v42`, the tree of the RL arms and of the test numbers before rung 2),
`fastroll.py` / `c2_engine.py` ended a serial-stack row at token 0 if the first sampled token was eos or contained a
blank line (a host stop rule). It fired on 0 / 600 dev rows (every row has 8 attempts) and was removed for the serial
stack in rel/v5 (`engine`) and in the rung-2 / rung-3 trees (`eval/trees/r2`, `r3`); the rung-1 rerun on the fixed
code (`lists/wpr_r1f.list`) is bitwise identical to the v4.2 run on all 1,319 test and 1,319 GSM-Hard rows.

## Non-compliant reference arms (labeled wherever they appear)

| reference | what is programmatic | numbers (GSM8K test / GSM-Hard) | where |
|---|---|---|---|
| programmatic position reset, u0 N = 8 (`PR` column = 1, `posreset`) | a position-id offset set by Python per attempt | 80.97 / 62.09 | `lists/test2.list u0_N8pr1`, `results/RL_RESULTS.md` §2 |
| arm **Ar** (RL with the programmatic reset, `RL_POSRESET=1`) | same, during RL and at test | 77.79 (u300 N = 8) | `engine/run_arm.sh Ar`, `lists/test2.list` |
| stop-token swap missing trigger (`FE_MISS=hook`) | after sampling, Python swaps a stop token for the phrase's first token | greedy-identical to the compiled trigger (600/600 dev) | `lists/miss*.list` |
| self-consistency controls (`KIND` SC) | N separate generation calls + a Python majority vote | e.g. SC8 u0; SC×4 on A1 / C / B (`RL_RESULTS.md` §8) | `lists/ctl_*.list`, `abl_test.list` |
| the earlier `gsm_push/` stack (S2F 73.39, S3-serial N = 4 / 8: 78.92 / 80.44) | S2F used the stop-token swap; S3-serial used the programmatic position reset | 73.39, 78.92, 80.44 | `gsm_push/README.md` (kept for the record; superseded by this folder) |

The compliant replacements: S2F with the trigger off 72.25 or compiled 72.63 (vs 73.39), and the serial stack with WPR
80.67 (rung 1, vs 80.44 with the programmatic reset; WPR − programmatic reset on the same test items: −0.30 [−1.67, 1.14]).

## Known slips in the shipped result notes and figures

`results/RL_RESULTS.md` §2 labels the u0 S2F test row (72.25) "missing trigger = swap hook", and §6 calls it "the
swap-hook run". The code that produced it (`eval/trees/v42/stk/fe_hook.py`, list `test3.list`, no `FE_MISS`) has the
missing-answer trigger **off** by default, as §4 and §5 of the same file say. So 72.25 is an all-weights S2F without a
missing-answer trigger, and the +0.38 of §6 is "compiled trigger vs no trigger".

`results/RL_RESULTS.md` §3 labels the GSM-Hard ladder rows "C4 (+ chain2 + calc)" = 53.90 and "D (+ compiled sanity)"
= 54.74. In `fe_hook.py`, KIND C4 is calc_s4 alone and D is calc_s4 + chain2 (no sanity filter), as on the test ladder
(68.69 / 69.45). So on GSM-Hard: + calc_s4 53.90, + chain2 54.74. The blog figures `figures/blog/v6/gsm8k/` plot the
GSM-Hard 53.9 at the "+ chain2" / "+ calculators" step, next to the test value with both calculators (69.5); the
matching GSM-Hard value is 54.7 (author decision: the figures are left as published).
