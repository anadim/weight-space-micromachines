# checks/seeds3: blog §3.4 / §5.1 / §5.2 one-seed claims at 3 seeds

Six one-seed claims in the blog get two more seeds each, or a 3-seed rerun. Final numbers, per-seed tables and blog
wording are in `RESULTS.md`. Every table there is recomputed on CPU from `res/` by the scripts in this folder.

| item | blog claim | new runs | 3-seed result |
|---|---|---|---|
| (a) §5.1 | A1 (RL with the whole serial stack, per-attempt credit): try 8 goes 23% → 66% (test) | none: the 3 A1 RL seeds of §16 already had per-try evals | try 8 test 22.7 → **62.1** (65.7 / 60.0 / 60.7); dev 29.0 → **67.9**; vote test 71.9 → **80.1** |
| (b) §5.2 m4 | RL cleans up costly misfires of a sloppy (noisy-routing) fine-tune and leaves the cheap ones | m4_s1, m4_s2 | costly calc@BBH 0.15 → 0.03-0.05 in 3/3 seeds; cheap calc@IFEval drops to ~0 in 3/3, drifts back in 1/3 |
| (c) §5.2 m2 | scramble all implants but the calculator: RL closes 4 of 7 in-wires, even the calculator's | m2_s1, m2_s2 | closes 4/7, 5/9, 5/10; the calculator is closed in 1 of 3 seeds |
| (d) §5.2 | correct vs scrambled library on GSM8K (LIB vs LIBscr) | LIB_s2, LIBscr_s2 (third pair) | LIB − LIBscr Hard +4.25 / +14.56 / +3.87, test +0.08 / +4.85 / +4.62; calculator kept in 2/3; big numbers 64.6 → 37.4 (4 runs) |
| (e) §5.2 mixed | all implants on the mixed task beat plain RL "by 13 points (58% vs 45%, 3 seeds)" | m1m_s0-s2, aNL_s0-s2 (+ m3_s1, m3_s2) | m1m **57.8** [53.2, 62.1] vs aNL **44.7** [40.3, 49.2], paired **+13.1** [+8.7, +17.6]; wins on gsm / bbh / ifeval, combo +2.7 n.s.; implants off 32.7 |
| (f) §3.4 | distilled links after RL: "+13 points on GSM-Hard (three seeds)" | W3d_s0-s2 | GSM-Hard links on − host only **+14.6 / +13.5 / +11.7** (mean **+13.3**); GSM8K test +0.0 / +0.4 / −0.4 |

## Files

| path | what |
|---|---|
| `RESULTS.md` | the (a)-(f) tables, observations, blog wording |
| `try8.py` → `try8.md` | (a) per-try accuracy (try k right if attempt k wrote a valid register equal to gold) from `res/a1/try8.items.jsonl.gz` and the dev curves `../../gsm_stack/results/curves/A1*.steps.jsonl` |
| `summarize.py` → `summarize.md` | (b), (c) tables from `res/{m4,m2}_s{1,2}/` and the seed-0 originals `../../rl/results/library/figdata/mix_m4_mixLIBnoisy_*`, `mix_m2_mixLIBscr_*` |
| `libpair.py` → `libpair.md` | (d) pair table, paired item bootstrap (2000 resamples), big-numbers check, from `res/rlctl/` and `../../rl/results/library/figdata/{LIB,LIBscr}_s0_*`, `LIB_s1_steps.jsonl` |
| `mixpair.py` → `mixpair.md` | (e) mixed-task table (per type, tools off, host only), paired differences, m1m leave-one-out, dev curves; (f) W3d per seed: test / Hard links on vs host only, dev links effect. 10000-resample item bootstrap (numpy) |
| `res/{m4_s1,m4_s2,m2_s1,m2_s2}/` | per run: `steps.jsonl` (per-update scalars + every dev record), `test_eval.json` (test 400 at u150: tools on / off / host only / leave-one-out, per item), `config.json` (trainer args, paths cut to basenames) |
| `res/rlctl/{LIB,LIBscr}_s{1,2}/` | the same for the GSM8K pairs (test_eval: GSM8K test + GSM-Hard, full and calc-only, per item) |
| `res/rlctl/{m1m,aNL}_s{0,1,2}/`, `res/rlctl/m3_s{1,2}/` | (e) the same layout as m4 / m2 (m1m: test 400 tools on / off / host only, 10 leave-one-outs, 2 swaps, per item; aNL / m3: full only) |
| `res/rlctl/W3d_s{0,1,2}/` | (f) `steps.jsonl` (per-update scalars, dev with links on / links reset / host only), `test_eval.json` (GSM8K test + GSM-Hard, links on and host only, per item), `config.json` |
| `res/hosts/*.sft_config.json` | the SFT warm-start hosts of the mixed task (`sft_host`; `sft_host_noisy` = `--noisy 0.3`, 632 rows; `sft_host_nl` = worked solutions instead of tool calls, 619 rows) |
| `res/a1/try8.items.jsonl.gz` | (a) per item, per try: valid / right for u0, A1, A1_s2, A1_s3 on test and GSM-Hard (reduced from the §16 fe.sh outputs) |
| `run_seed.sh` | one run on one GPU: `m4_sK`, `m2_sK`, `LIB_sK`, `LIBscr_sK`, `m1m_sK`, `aNL_sK`, `m3_sK`, `W3d_sK` |
| `reduce_run.py` | reduces a new run dir to the `res/` layout (same rules as the shipped files) |
| `rebuild_code.sh`, `code_overlay/` | rebuilds the code tree these runs used (below) |

## Recompute the tables (CPU, seconds, no GPU or model)

```
cd checks/seeds3
python3 try8.py   > try8.md        # (a)
python3 summarize.py > summarize.md  # (b), (c)
python3 libpair.py > libpair.md    # (d)
python3 mixpair.py > mixpair.md    # (e), (f); needs numpy (any env of REPRODUCE §2), about 10 s
```

All four outputs match the numbers in `RESULTS.md`. They were checked against the original analysis on the full,
unreduced run outputs:
- `try8.py` given the original per-item fe.sh jsonl prints the same table.
- `summarize.py` prints the same (b)/(c) tables. It also adds a calculator-by-update block, which is where the
  u25 / u50 / u100 values of the (b) table come from.
- `libpair.py` prints the same bootstrap lines.
- `mixpair.py` prints the same point estimates and the same CIs, digit for digit. The original analysis script also
  bootstrapped two sections that are not part of this check before (f), from the same RNG; `--burn 28` (default) skips
  those draws. With `--burn 0` the (f) CIs move by at most 0.1.

`try8.py`, `summarize.py` and `libpair.py` read the seed-0 / original files from the release (`gsm_stack/results/curves`,
`rl/results/library/figdata`). Set `WSM_ROOT` to point at another copy.

## Code these runs used

All runs here (m4 / m2 / LIB / LIBscr / m1m / aNL / m3 / W3d) used a frozen copy of the release's `rl/` tree with three
patched trainers. The patches add two things:
- resume from a rolling state saved every 10 updates, including the final update and the last dev record;
- per-item test correctness in `test_eval.json`, which the paired bootstrap of (d) needs.

The scramble, noisy-host and `--cw_mask` options are already in the release. `code_overlay/` holds the three patched files
and one new script:
- `rl/library/train_library.py`
- `rl/gsm_grpo/train2.py`
- `rl/wiring/train_wiring.py`
- `rl/library/mixed/make_nl_sft.py`: builds `union_nl.jsonl` for the aNL host, i.e. the 432 tagged SFT rows of
  `sft_warm.py` with every tool call rewritten as a plain worked solution (13 recovery rows dropped). CPU, seconds;
  `run_seed.sh` builds it and checks its SHA-256 (`bb894058…`).

`rebuild_code.sh TARGET` rebuilds the full tree from the release:
- copies `bbh_generic`, `data`, `ifeval`, `micromachines` and `rl`;
- moves four stale `rl/library/{common,engine,fastqwen,implant}.py` copies to `rl/library/_repo_stale/`, so the trainer
  imports the `rl/wiring` versions;
- applies the overlay;
- copies the W3d init links (`rl/results/wiring/H100/sft3_W3e_lr1e-2/links_final.pt`, distilled from the hand wiring by
  `rl/wiring/smoke/cw2.sh`; REPRODUCE §12) to `rl/wiring/runs/sft3_W3e_lr1e-2/`, where `--init_links` expects them.

The rebuild was checked file by file against the working tree (the trainers differ only in comments; the W3d init links
and the rebuilt `union_nl.jsonl` are byte-identical). The only other differences were:
- the cluster env file (`env_local.sh` is used instead);
- generated data, which `run_seed.sh` regenerates and checks against `rl/library/mixed/DATA_SHA256.txt`;
- files these runs do not use.

(a) needs no code from here. Its evals are the §16 `gsm_stack` ones (`gsm_stack/eval/fe.sh` with `test1.list A1_N8`,
`seed_A1s2_Bs2.list`, `seed_A1s3.list`; REPRODUCE §16.4).

## Rerun (one GPU per run)

```
# once, from the release root: calculator veto (~5 min GPU)
python scripts/solve_veto.py --device cuda          # -> runs/veto_qwen3.pt (or set WSM_VETO)
CUDA_VISIBLE_DEVICES=0 bash checks/seeds3/run_seed.sh m4_s1     # also m4_s2, m2_s1, m2_s2, LIB_s2, LIBscr_s2,
                                                                 # m1m_s{0,1,2}, aNL_s{0,1,2}, m3_s{1,2}, W3d_s{0,1,2}
python3 checks/seeds3/reduce_run.py runs/seeds3/runs/m4_s1 m4_s1 checks/seeds3/new/m4_s1   # others: .../new/rlctl/NAME
cd checks/seeds3 && S3_RES=new python3 summarize.py                       # or libpair.py, mixpair.py (needs all its runs)
```

`run_seed.sh` does the following:
- builds the code tree on first use (`rebuild_code.sh` into `checks/seeds3/code`, or `$S3_CODE`);
- prepares the GSM8K and mixed-task data;
- builds the SFT host for m4 / m2 / m1m / m3, or the worked-solution host for aNL (a few minutes);
- installs the IFEval checker libs (`lm_eval 0.4.13`, `langdetect`) and NLTK `punkt_tab` into `$WSM_DATA/pylib`, failing
  loudly if punkt is missing;
- trains to the end; W3d then runs the test eval (GSM8K test + GSM-Hard) into `$WSM_DATA/runs/TEST_W3d_sK`, which
  `reduce_run.py` reads.

Outputs go to `$WSM_DATA/runs/NAME` (default `runs/seeds3/runs/NAME`, git-ignored). Rerunning the same NAME resumes.
Runs can be started in parallel, one per GPU: a lock makes them build the shared code tree, data, hosts and libs one at a
time.
Python env: REPRODUCE §2. The first run downloads Qwen/Qwen3-0.6B.

| run | updates | GPU (original) | wall time |
|---|---|---|---|
| m4_s1, m4_s2, m2_s1, m2_s2 | 150 (64 prompts x G8), dev 400 at u0/25/50/100/150, test 400 + leave-one-out | 1 x B200 each | 2 h 45 min - 2 h 50 min (9705-10047 s training) |
| LIB_s2, LIBscr_s2 | 300 (64 x G8), dev 800 every 25, GSM8K test + Hard | 1 x B200 each | about 4 h 15 min (4 h job + 15 min resume) |
| LIB_s1, LIBscr_s1 (B200 rerun of seed 1) | same | 1 x B200 each | 4 h 0 min / 4 h 16 min |
| m1m_s0, m1m_s1, m1m_s2 | 150 (64 x G8), dev 400 at u0/25/50/100/150, test 400 + 10 leave-one-outs + 2 swaps | 1 x B200 each | about 2 h 50 min (10062-10216 s) |
| aNL_s0-s2, m3_s1, m3_s2 | 150 (64 x G8), dev 400, test 400 | 1 x B200 each | about 21 min (1268-1288 s) |
| W3d_s0, W3d_s1, W3d_s2 | 300 (64 x G8), dev 800 every 25 (+u64), then test + Hard | 1 x B200 each | 56-59 min incl. test (3368-3533 s) |

The original seed-0 runs of (b)-(d) ran on an H100 (m4, LIB, LIBscr) or an A100 (m2). (e) and (f) are full 3-seed reruns:
the one-seed originals (m1 with `--cw_mask 0`, m3 s0, W3d s0-s2 on H100 / A100) stay in REPRODUCE §12. Their files are in `rl/results/library/figdata`
(REPRODUCE §12). RL is not bitwise deterministic across GPUs. A rerun should land within the seed spread of the RESULTS
tables, not on the same digits.

## Not shipped

- Checkpoints, rollouts, generated text and trainer logs. `steps.jsonl` keeps per-update scalars and every dev record. It
  drops the per-update detail and the duplicated test records, which are in `test_eval.json`.
- The cluster queue driver and its job logs (`rl_chain.sh` and chain/driver logs). `run_seed.sh` replaces them.
- The empty per-run env override files.
- The log-tail monitor.
- The misfire probe for the new m4 seeds: it was not run (seed 0 only, within-group reward cost: −0.39 on BBH vs −0.08 on IFEval; `rl/library/NOTEBOOK.md`).
- The full per-item fe.sh outputs of (a). `res/a1/try8.items.jsonl.gz` keeps per item, per try: valid / right.
- The A40 original of LIB seed 1. Only its steps file is used (the big-numbers row), from the release figdata.
- The broken-calculator 3-seed rerun (arm R, 300 u) that ran in the same batch as (e) / (f): it is not quoted by the final
  blog and is a separate table.
- `union_nl.jsonl` (888 KB): `run_seed.sh` rebuilds it with `make_nl_sft.py`.
