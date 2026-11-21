# GSM8K single-call implant stack

A frozen Qwen3-0.6B host answers a GSM8K question in **one generation call**. Implants written into weights do the
arithmetic, check the answer's form, hide earlier tries, reset positions, keep an answer register, vote and decide
how many tries to spend. Nothing outside the weights looks at the text except the generator (sampling, token budget,
stop on eos). `COMPLIANCE.md` lists what is in weights, the generator-only exceptions and the non-compliant
reference arms; `MODEL_CARD.md` says where every host comes from (no checkpoints are shipped).

This folder supersedes `gsm_push/` for the published numbers: two parts of the earlier stack were programmatic
(the stop-token swap of the missing-answer trigger and the position-id reset of the serial tries), and both are now
in weights.

## The implants

| implant | what it does | in weights as | code |
|---|---|---|---|
| calculator `calc_s4` | exact + − × ÷ on the digits the model writes, result written into the residual stream | compiled ReLU circuit + solved veto | `micromachines/circuits/calc.py`, `engine/wiring/implant.py` |
| calculator v2 `chain2` | the same for +/− chains (`a + b − c = `) | compiled circuit | `engine/stk/chain2.py`, `implant2.py` |
| answer-sanity DFA + phrase writer | a 34-state DFA reads the answer as it is written; if it is missing or malformed, a pointer writer emits " We need to find" and the model tries again | token × state tables, pointer register, residual writes | `engine/stk/san_engine.py`, `comp_sanity.py` |
| weight-space forget | later tries cannot attend to earlier ones (−10,000 on their keys) | 4 extra fixed q/k dims per head | `engine/stk/forget_ws.py`, `attn3.py` |
| serial tries + answer register + voter | N tries in one stream; a register stores each answer, its flags and a layer-18 probe score; a ReLU voter and a final writer emit the winner | fixed tables, registers, a 6 KB linear probe, writers | `engine/stk/comp_serial.py` |
| weight-space position reset (WPR) | every try sees the question at the distance the first try saw it | offset register + fixed rotary table on q / K | `implants/wpr`, `comp_serial.py` (`WPR=1`) |
| adaptive tries | stop at 4 if unanimous and confident, else extend in blocks of 4 up to 16; repair non-integer answers; cap 400 tokens per try | comparators on registers | `implants/rungs/rung2`, `aw_adapt.py` |
| answer-group voter | scores groups of equal answers (number-set registers, fixed linear score) | register automaton + fitted linear map | `implants/rungs/rung3`, `aw_vf.py`, `numreg.py` |

Also here: the side-net verifier register (rung 4, dev only), and nine negative results with their code and READMEs
(`implants/README.md`).

## The ladder (GSM8K test, 1,319 items)

Host: u0 = Qwen3-0.6B-Base after 300 GRPO updates with the calculator (`a2_u300_s2`), frozen. T = 1, seed 7, each rung
run once. 95% paired-bootstrap CIs (4,000 resamples). Source: `results/RL_RESULTS.md` §4, §9.

| rung | test [95% CI] | Δ vs previous | GSM-Hard | tok / q |
|---|---|---|---|---|
| host, no implants | **67.70** [65.20, 70.13] | | 39.42 | 137 |
| + calc_s4 | 68.69 | +0.99 [−0.15, 2.05] | 53.90 | 137 |
| + chain2 | 69.45 | +0.76 | 54.74 | 137 |
| + sanity DFA + forget = S2F (one try) | **72.25** [69.75, 74.68] | +2.80 (vs host +4.55 [2.58, 6.44]) | 55.19 | 152 |
| + serial tries, register, voter, N = 8, no reset | 71.87 [69.37, 74.15] | −0.38 [−2.43, 1.74] | 57.54 | 1094 |
| + WPR (rung 1) | **80.67** [78.47, 82.71] | +8.79 [6.75, 10.84] vs no reset | 62.17 | 1277 |
| + adaptive tries (rung 2) | **81.58** [79.38, 83.70] | +0.91 [−0.45, 2.35] | 62.62 | 1482 |
| + Nmax 24 + answer-group voter (rung 3) | **82.71** [80.59, 84.76] | +1.14 [−0.15, 2.50]; +2.05 [0.61, 3.56] vs rung 1 | 60.42 | 1774 |

The headline steps are 67.7 → 72.3 → 80.7 → 81.6 → 82.7. On GSM-Hard the calculator gives the big jump (+14.48 [12.28, 16.68] for calc_s4 alone) and
rung 3 is 2.2 points *below* rung 2 (its voter was fit on GSM8K-train rollouts); rung 2 + Nmax 24 without the voter is
the best untrained GSM-Hard number, 63.53. The S2F rung has no missing-answer trigger; with the compiled trigger it is
72.63. Reference, NOT compliant: the same N = 8 stack with a programmatic position reset scores 80.97 (WPR − reset:
−0.30 [−1.67, 1.14]).

## RL from the base model (GSM8K test)

Base host Qwen3-0.6B-Base, 300 GRPO updates (G = 8, lr 1e-6), test at T = 1, seed 7. Seed-pooled means; the CI of a
pooled Δ covers item sampling only (seed sd in brackets). Source: `results/RL_RESULTS.md` §10, §9.

| configuration | seeds | test | GSM-Hard | tok / q |
|---|---|---|---|---|
| base host, no implants, no RL | – | **28.05** | 10.46 | 103 |
| plain RL, no implants (arm P) | 2 | **70.77** (sd 0.59) | 45.00 | 120-129 |
| RL with the S2F implants, one try (arm Bb) | 3 | 72.96 (sd 0.74); +2.19 [0.73, 3.70] vs P | 55.72 | 141-144 |
| RL with the serial stack, N = 4, no reset (arm C) | 3 | 74.80 (sd 0.29); +4.03 [2.38, 5.62] vs P | 52.97 | 516-584 |
| RL with the serial stack + WPR, N = 4 (arm CW) | 3 | 76.52 (sd 1.27); +1.72 [0.56, 2.93] vs C | 55.07 | 588-647 |
| arm CW hosts + the rung-2 stack at inference | 3 | **80.26** (sd 1.76) | 58.76 | 1612 |

So 28 → 80.3 with implants attached during and after RL, against 70.8 for plain RL. Most of the step from the base is
RL itself (+42.7); the implants add +9.5 on top of plain RL. The base's T = 1 zero-shot 28 is far below its greedy
52.5 and its 4-shot 56.5 (`MODEL_CARD.md`).

RL on the u0 host (arms A, A1, A1W, A1D, B; `results/RL_RESULTS.md` §1-§3, §9): the attempt-level credit arm A1
reaches 80.06 (3 seeds, N = 8), +8.2 over the same stack untrained without a reset. With WPR the untrained rung 2
(81.58) beats every RL arm on test; RL + WPR + rung 2 is the best GSM-Hard number (64.19, 3 seeds).

## Layout

| path | contents |
|---|---|
| `engine/` | release rel/v5, full snapshot: `stk/` (the stack, fastroll, attn3 compaction / overlap / split-KV attention, TK tool kernels with NVRTC sources, the fast trainer `fasttrain.py` + `tattn.py`, `rl_hook.py`), `wiring/` (host engine, fused Qwen, implant glue), `dig_san/tw_dump_lean.py` (the driver that loads the host and execs a hook), `run_arm.sh` (one RL arm on one GPU), `jobs/` |
| `eval/` | the test harness: `fe.sh` + `lists/*.list` (one line per config), `trees/{v42,r2,r3}` (code-tree overlays), `jobs/rung_job.sh`, `ana_rltest.py`, `results_tools/` (`devtab.py`, `devcmp.py`, `pboot.py`, `pbootm.py`, `offctl.py`) |
| `implants/` | the implant lane: `wpr/`, `aw/`, `rungs/rung2..4`, `vfeat/`, `sidenet/`, `negative/` (`implants/README.md`) |
| `results/` | `RL_RESULTS.md` (all RL and test tables), `BLOG_CONTROLS.md` (controls and compliant reruns), `LADDER.md` + `REPORT.md` (dev ladder, failure atlas, implants), `HOWTO_RUN_STACK.md`, `notes/`, `curves/` (per-arm dev curves) |
| `figures/` | stack figures (`make_figs.py`, needs eval rows); blog GSM8K figures are in `figures/blog/v6/gsm8k/` |
| `env.sh`, `mk_tree.py`, `run_hook.sh` | shared paths, tree builder, one-hook launcher |

## Running it (one GPU)

```bash
python data/prepare_gsm.py && python scripts/solve_veto.py --device cuda   # data + calculator veto (runs/veto_qwen3.pt)
# u0 host (REPRODUCE.md section 16.1), then:
export WSM_DATA=$PWD/runs/gsm_stack CKPT_U0=$WSM_DATA/ckpt_a2s2
GPU=0 bash gsm_stack/eval/fe.sh gsm_stack/eval/lists/test3.list            # ladder: host, + calc, + chain2, S2F
GPU=0 bash gsm_stack/eval/fe.sh gsm_stack/eval/lists/wpr_rung1.list        # rung 1 (WPR)
GPU=0 bash gsm_stack/eval/jobs/rung_job.sh r2 wpr_rung2.list               # rung 2 (CPU fuzz, then test + GSM-Hard)
GPU=0 bash gsm_stack/engine/run_arm.sh CW_s0                               # one RL arm, 300 updates
```

Every published number, with its command, seed, hardware and time: `REPRODUCE.md` §16. CPU checks that need no
model or GPU (`VERIFY.md` §13), e.g. the adaptive-tries fuzz:

```bash
python3 gsm_stack/mk_tree.py /tmp/r2 r2 && (cd /tmp/r2/stk && STK_DIR=$PWD python3 test_aw.py)    # AW FUZZ PASS
```
