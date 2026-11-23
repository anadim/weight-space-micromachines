# C36: equal-footing text tool vs exact calculator implant (28 train seeds)

**Question.** The C35 text-tool baseline (`../tooltext`) counted tool tokens against the 320-token generation cap,
and its tool and implant arms ran at the C35 default host lr. Under equal conditions, does an exact calculator *inside*
the weights path differ from a text `<calc>` tool? The conditions are:
- the same `<calc>EXPR</calc>` call syntax and 8-shot prompt;
- the cap applies to model tokens only, with up to 128 tool- or implant-written tokens extra (`--cap_extra 128`);
- each arm uses its own dev-selected config.

All arms start from Qwen3-0.6B-Base and train with the same RL recipe: 300 updates, 64 prompts × 8, T 1, Dr.GRPO,
clip 0.2, warmup 5, max_new 320 model tokens.

Arms (row names as in `c36_cfg.json` / `rows38*.json`):

| row | what | config |
|---|---|---|
| T | text tool: the host evaluates EXPR exactly (`tooltext.calc_eval`) and the result tokens are forced in | host lr 4e-6 (u100 sweep pick) |
| T1 | same | host lr 1e-6 (the C35 default) |
| T2 | same | host lr 2e-6 (= I's lr; the 300-u dev rule (b) pick; **the headline tool row**) |
| I | implant: on `</calc>` a frozen exact int64 *core* (`core.py`) computes the value; a trainable writer and CW-style adapters into layers 7/14/21/final carry it into the model; an in-wire gate at layer 14 decides per call whether the core's value is admitted (floor 0.05) | host lr 2e-6, link lr 1e-2, `--ci_std 1e-3` |
| Ie0 | I's checkpoints, evaluated with the in-wire floor off (`--iw_eps_eval 0`) | — |
| I0 | implant control: frozen hand-set writer, no adapters, always admits | host lr 2e-6 |
| Itok | implant control: only the writer is trainable | host lr 2e-6, writer lr 1e-2 |
| I1 / I01 / Itok1 | the three implant rows at host lr 1e-6 | 8 seeds each |
| Ix / Ie0x | sensitivity rows: I / Ie0 without the two collapsed seeds 14 and 17 | 26 seeds |

Bracket tags in the code and in the rows JSON labels are experiment labels from the lab log:
- [35] = C35;
- [37](a) = the model-token-cap re-eval of C35;
- [38] = this comparison;
- [41] = the lr / wiring dev sweep.

They are kept so that the regenerated JSON is byte-identical.

## Results
GSM8K test (1319) / GSM-Hard (1319) / GSM-Plus testmini (2100). Accuracy is the mean over train seeds × eval seeds
7/11/13 (T 1, one attempt). Source: `rows38_all.json` and `rows38_all_sens.json`; full tables in `res/ana38_all28I*.txt`.

| row | train seeds | test | Hard | GSM-Plus | calls/q | admit |
|---|---|---|---|---|---|---|
| 0X (Base + tool, no RL) | – | 27.19 | 15.14 | 17.89 | 2.85 | |
| 0I0 (Base + I0, no RL) | – | 26.59 | 16.12 | 17.68 | 2.84 | 1.00 |
| T (4e-6) | 8 | 65.93 | 56.28 | 51.21 | 2.96 | |
| T1 (1e-6) | 20 | 69.37 | 57.45 | 54.50 | 3.14 | |
| **T2** (2e-6) | 28 | **70.04** | **58.90** | **55.01** | 3.20 | |
| I | 28 | 69.70 | 55.89 | 54.77 | 2.92 | 0.883 |
| Ie0 | 28 | 69.95 | 57.17 | 55.05 | 2.92 | 0.929 |
| I0 | 28 | 69.95 | 58.82 | 54.80 | 3.18 | 1.00 |
| Itok | 28 | 70.04 | 58.96 | 55.18 | 3.16 | 1.00 |
| I1 / I01 / Itok1 | 8 | 68.90 / 69.23 / 68.85 | 55.49 / 56.95 / 57.15 | 53.98 / 54.77 / 54.45 | | |
| Ix | 26 | 69.80 | 57.04 | 54.83 | | |
| Ie0x | 26 | 70.07 | 58.41 | 55.14 | | |

Paired differences use a train-seed × item bootstrap 95% CI:

| difference | test | Hard | GSM-Plus |
|---|---|---|---|
| I0 − T2 | −0.09 [−0.69, +0.55] | −0.08 [−0.71, +0.56] | −0.20 [−0.74, +0.30] |
| Itok − T2 | −0.00 [−0.59, +0.65] | +0.07 [−0.54, +0.72] | +0.17 [−0.44, +0.76] |
| I − T2 | −0.33 [−0.98, +0.34] | −3.00 [−4.81, −1.60] | −0.24 [−0.82, +0.31] |
| Ie0 − T2 | −0.08 [−0.73, +0.61] | −1.73 [−3.66, −0.19] | +0.04 [−0.51, +0.57] |
| Ix − T2 | −0.24 [−0.90, +0.44] | −1.85 [−2.54, −1.22] | −0.17 [−0.78, +0.44] |
| Ie0x − T2 | +0.03 [−0.63, +0.70] | −0.48 [−1.12, +0.15] | +0.13 [−0.44, +0.70] |
| I − I0 | −0.25 [−0.89, +0.43] | −2.93 [−4.80, −1.53] | −0.04 [−0.63, +0.55] |
| Ix − I0 | −0.15 [−0.81, +0.50] | −1.78 [−2.39, −1.17] | +0.03 [−0.55, +0.64] |
| Itok − I0 | +0.08 [−0.53, +0.72] | +0.14 [−0.45, +0.75] | +0.37 [−0.26, +0.98] |
| T2 − T1 | +0.67 [−0.04, +1.38] | +1.45 [+0.75, +2.21] | +0.51 [−0.07, +1.09] |
| I − T (u100 picks) | +3.78 [+2.45, +5.04] | −0.39 [−2.65, +1.51] | +3.55 [+2.40, +4.69] |

Reading:
- At equal footing, the exact implant controls (I0, Itok) and the text tool T2 are indistinguishable.
- Arm I is lower on GSM-Hard. Part of the gap comes from the two collapsed seeds: seeds 14 and 17 collapse to ~40 on
  Hard. Both ran on A100s. The B200 reruns `fIr_s14/17` and `fIa_s14/17` did not collapse; they are diagnostics, not
  pooled into any row.
- The other part comes from the in-wire gate rejecting core values (admit 0.88).

Further results:
- **Core exactness.** The core agrees with the tool on 99.98% of the 580,834 `<calc>` calls in the C35 arm-X traces:
  119 mismatched calls (`res/verify.txt`, `res/verify_core.json`, `res/mismatches.txt`).
- **Core latency.** CPU 17–40 ms per call (`res/prof_core_cpu.txt`, `res/prof_core_gpu.txt`). The default is
  `CORE_DEV=cpu`, with an exact per-run memo.
- **Dev selection.**
  - 300-u dev rule (b): `res/sel300_8seeds.txt`, 8 seeds; also `res/sel300_20seeds.txt`. At u300, T gets 72.52 /
    73.23 / 69.66 at lr 1e-6 / 2e-6 / 4e-6. I* gets 72.02 / 73.16 at 1e-6 / 2e-6.
  - u100 sweep: `res/ana_sweep_a100.txt`.
- **Why-split** (arithmetic slips vs other errors, from the generated text): `res/why38*.txt`, `why/why37a.txt`.
- **Wiring dynamics** (admit by operand size and operator, link norms): `res/links38*.txt`.
- **Tokenization of the calls:** `res/tokprobe.txt`.

## Layout
| path | what |
|---|---|
| `tt/code/rl/gsm_grpo/` | **overlay** on the release `rl/gsm_grpo/` (see "Code" below) |
| `tt/mk_code.sh`, `tt/code_data.sha256` | build the runnable code tree `tt/code/` (release `rl/gsm_grpo` + `micromachines` + data + veto, then the overlay) and check the data hashes |
| `tt/job.sh` | job runner: `check_ci`, `check`, `train`, `eval` (one GPU) |
| `tt/commands.sh` | prints the exact command line of every job that produced the shipped results |
| `tt/res/` | the per-run results, text-free (built by `mk_res.py`; see below) |
| `mk_res.py` | reduces job outputs (`tt/out/`) to `tt/res/` |
| `ana38.py`, `c36_cfg.json` | the C36 tables → `rows38*.json` |
| `sel300.py`, `ana_sweep.py`, `links38.py` | dev selection at u300, the u100 sweep table, wiring dynamics (read `tt/res/`) |
| `verify_core.py`, `prof_core.py`, `forms.py`, `tokprobe.py`, `why/` | core exactness, core latency, call catalogue, call tokenization, why-split; these need the generated text (not shipped) or a GPU |
| `rows38.json` | 8-seed version (train seeds 0–7) |
| `rows38_all.json`, `rows38_all_sens.json` | the final tables (every row on all its seeds; `_sens` adds Ix / Ie0x) |
| `res/` | analysis outputs (text) |

### Code
`tt/code/rl/gsm_grpo/` contains only files that are new or differ from the release `rl/gsm_grpo/`.

New:
- `tooltext.py`: the `<calc>` prompt and the exact evaluator.
- `core.py`: the exact int64 calculator core.
- `armi.py`: arms I / I0 / Itok.
- `fe_tool.py`: eval.
- `check_tool.py`, `check_ci.py`: base-model checks.

Modified copies:
- `engine.py`: the tool and implant paths, `cap_extra`.
- `train2.py`: arms X / I / I0 / Itok, `--sync_every`, `--cap_extra` and the wiring flags.
- `fastqwen.py`: the layer-14 capture for the in-wire gate.

`../tooltext/code/rl/gsm_grpo/` is the C35 version of the same tool files. `check_tool.py` and `tooltext.py` are
identical; `engine.py`, `train2.py` and `fe_tool.py` are earlier versions without `--cap_extra` or arm I. This tree
reproduces the C35 arm-X eval exactly with `--cap_extra 0`: a re-eval of `aX_s0` was byte-identical on every set.

`bash tt/mk_code.sh` builds `tt/code/` in place. It needs:
- `data/gsm8k/` from `data/prepare_gsm.py`, and `data/gsmhard/`;
- GSM-Plus testmini from `uniform/final_eval/data/prepare_gsmplus.py` (env `GSMPLUS`);
- `runs/veto_qwen3.pt` from `scripts/solve_veto.py` (env `WSM_VETO`).

All are checked against `code_data.sha256`.

## Rerun
```
cd checks/calcimp/tt
bash mk_code.sh                          # once
bash commands.sh                         # list the jobs; run each line on one GPU:
CUDA_VISIBLE_DEVICES=0 bash -c "$(bash commands.sh | grep ' fI_s3 ')"
cd .. && python3 mk_res.py --out tt/out --res tt/res     # job outputs -> text-free res/
python3 ana38.py --all --write           # -> rows38_all.json  (CPU, ~7 min)
python3 ana38.py --all --sens --write    # -> rows38_all_sens.json (~10 min)
python3 ana38.py --write                 # -> rows38.json, train seeds 0-7 (~5.5 min)
python3 sel300.py [NSEEDS]; python3 ana_sweep.py; python3 links38.py 'fI_s%d' 20
```

`job.sh` runs the code tree with `WSM_ROOT=tt/code`. It sources the release `rl/gsm_grpo/env_local.sh`, which you edit
for your Python environment. Run and eval dirs (checkpoints included) go under `$WSM_DATA`, default
`<release>/runs/calcimp`; the small outputs are copied to `tt/out/`, which is git-ignored.

**Hardware and time:**
- Train seeds 0–11 (all rows), the 8-seed rows, the sweep at seed 0, and the fIr / fIa diagnostics ran on **one B200
  per job**: ~40 min per train + eval line, 62.5 GB peak.
- **Train seeds 12–27 (T1: 12–19) ran on A100-SXM4 40 GB** with smaller batches: train `--B_max 192 --tok_budget 4096`,
  eval `--B_max 192`. These are the A100 block in `commands.sh`. They took ~1h20 (T1, T2) to ~2h05 (I, I0) per line.
  `tt/res/configs.json` has `"gpu"` per run.
- Seed-1 replicates of the I-wiring grid and the seed-2 replicates of the swT sweep also ran on A100.
- The A100 batching changes the RL batch composition, not the recipe.

**Verified.** On the shipped `tt/res/`:
- `ana38.py --write`, `--all --write` and `--all --sens --write` regenerate `rows38.json`, `rows38_all.json` and
  `rows38_all_sens.json` byte-identically.
- `sel300.py 8` / `20`, `ana_sweep.py` and `links38.py` (`'fI_s%d' 8` + `'fItok_s%d' 8` → `res/links38.txt`;
  `'fI_s%d' 20` → `res/links38_20seeds.txt`) reproduce the shipped `res/` text files byte-identically.

`res/sel300_*` and `res/ana_sweep_a100.txt` were regenerated after the last runs. They include T seeds 8–19 and the A100
sweep replicates, so they supersede older working-tree versions.

## Not shipped
- Checkpoints: one u300 checkpoint per run, regenerate with `commands.sh`.
- Generated text: the fe_tool.py `*.jsonl` per-item outputs with the model text. `tt/res/items.jsonl.gz` keeps only
  per-item correctness and per-run token / call sums.
- Per-update training logs beyond the fields in `tt/res/train_steps.jsonl.gz`.
- An 87 MB set of C35 arm-X trace copies that `verify_core.py` / `forms.py` / `tokprobe.py` read; rerun
  `../tooltext/commands.sh` to regenerate them.

Hence `verify_core.py`, `forms.py`, `tokprobe.py` and `why/*.py` run only after regenerating the evals; their outputs are
shipped in `res/` and `why/why37a.txt`.

`prof_core.py` and `check_ci` need torch (and a GPU for the GPU column / check_ci).
