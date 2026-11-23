# seeds35: GSM8K §5.1 rows 2 / 2b / 4 at RL seeds 0-5 (Qwen3-0.6B-Base)

The uniform GSM8K rows of blog §5.1 (REPRODUCE §17) were first run at RL seeds 0-2. This check adds RL seeds 3, 4 and 5
of both RL arms and re-runs the whole §17 eval protocol on them, so every row is a mean over 6 RL runs.

- Row 2 = P64: plain RL from Base (64 prompts x G8, 300 updates), evaluated alone, 1 sample.
- Row 2b = the P64 host with the untrained rung-2 stack attached at eval only (row-4 eval settings).
- Row 4 = CW: RL from Base with rung 2 attached (CFG=A + weight-space position reset WPR=1), evaluated with rung 2.

All 6 new RL arms and all their evals ran on single B200 GPUs, with the same commands as seeds 0-2.
`AUTO_final.md` holds the full tables (`python3 ana35.py --stamp 2026-11-23T10:40Z` reproduces it byte for byte from `res/` plus the §17 seeds 0-2 rows in `uniform/final_eval/items/`).

## Final (seed 7, the §17 table protocol; mean over RL seeds 0-5; paired item bootstrap, 2000 resamples)

| row | GSM8K test | GSM-Hard | GSM-Plus | tok/q test |
|---|---|---|---|---|
| 2 RL (P64) | 68.32 | 41.60 | 53.67 | 129 |
| 2b RL, then rung 2 attached | 82.22 | 61.42 | 68.34 | 1535 |
| 4 RL with rung 2 (CW) | 80.44 | 59.01 | 66.09 | 1607 |
| **4 − 2** | **+12.12 [+10.77, +13.57]** | **+17.41 [+15.59, +19.27]** | **+12.42 [+11.05, +13.74]** | |
| 2b − 2 | +13.90 [+12.53, +15.33] | +19.83 [+18.04, +21.67] | +14.67 [+13.41, +15.98] | |
| 4 − 2b | −1.78 [−2.50, −1.01] | −2.41 [−3.35, −1.38] | −2.25 [−3.04, −1.49] | |

- Averaged over eval seeds 7/11/13 (18 runs per row): 4 − 2 is +11.97 / +17.58 / +12.26 (test / Hard / GSM-Plus), and
  4 − 2b is −1.96 / −2.62 / −2.44. All CIs exclude 0.
- Seeds 3-5 alone vs seeds 0-2: 4 − 2 at seed 7 is test +12.46 vs +11.78 and Hard +17.26 vs +17.56.
- Row 4 per RL seed, test: 81.12 / 81.43 / 78.24 / 81.35 / 79.38 / 81.12.
- Every 3-seed GSM8K claim of §5.1 holds at 6 seeds.

## RL runs (in-run dev at u300)

CW reports dev_vote4 / dev2_vote4 / dev_vote1; P64 reports dev_acc / dev2_acc (`res/dev.json`; table in AUTO_final.md).

| arm | job label | wall time (one B200) | in-run dev @u300 |
|---|---|---|---|
| CW_s3 | j25308 | 1 h 45 min | 84.83 / 80.33 / 76.17 |
| CW_s4 | j25309 | 1 h 45 min | 85.17 / 84.00 / 74.50 |
| CW_s5 | j25310 | 1 h 43 min | 86.00 / 83.67 / 75.67 |
| P64_s3 | j25311 | about 15 min | 80.17 / 79.67 |
| P64_s4 | j25312 | about 15 min | 77.50 / 79.33 |
| P64_s5 | j25313 | about 15 min | 83.00 / 81.00 |

Seeds 0-2 for reference: CW dev_vote4 86.50 / 86.67 / 84.17; P64 dev_acc 79.83 / 79.33 / 82.17. All new runs are in range.

## Protocol (the same as seeds 0-2)

- All evals use the §17 GSM8K harness (`uniform/final_eval/fe.sh`, zero-shot chat format, T = 1, fastroll, one run per
  config), launched with `fejob.sh` or `fepack.sh`. Host = `$ARMDIR/rl/<ARM>/step_000300`.
- Row 2 (P64, 1 sample): list lines `H 1 0 1 1 0` on build `stk`, as `uni_p64.list` (seed 7), `q34s2_P64.list` (seeds
  11/13) and the GSM-Plus row-2 lists of §17.
- Rows 2b and 4: `S 8 0 1 1 0` (test, GSM-Plus) and `S 8 0 0 0 1` (Hard, plain majority), env
  `WPR=1 SW_NS=16 SW_MAXN=7936 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400`, build `stk_r2q` via `fepack.sh` (runs the
  `test_aw` fuzz test first, packs the lists on one GPU), as `q34att_*` (2b) and `q34s4_CW_s*` (row 4, seeds 11/13).
- Caveat: row 4 at eval seed 7 for RL seeds 0-2 (80.26 / 58.76) ran on the older `stk_r2` build (`wpr_r2cw.list`).
  The `stk_r2q` re-run of that row-4 test is +0.15 [−0.56, +0.88] (§17). All new row-4 runs use `stk_r2q`, the same
  build as 2b and as the row-4 seeds 11/13.
- Eval seeds: 7 is the §17 table; 11 and 13 give the seed averages. GSM-Plus testmini uses seeds 7/11/13.
- Analysis: `ana35.py`. Its loader and paired item bootstrap are a copy of the §17 analysis (`ana_q32.py`): per-item
  mean over runs, 2000 resamples, Random(0) per table, CI = sorted[50], sorted[1949]. On seeds 0-2 it reproduces §17:
  2b − 2 test +13.62 [+12.03, +15.14], Hard +20.12 [+18.20, +22.16].
- Writer audit (`res/fe_audit.txt`, last section of AUTO_final.md): sampled_ne_written is 0-2 per 100-180k writer
  steps, the same order as seeds 0-2.

## Lists (`lists/`) and jobs (`jobs.tsv`)

- `s35_H_P64_sK.list`: row 2, test / Hard / GSM-Plus at eval seeds 7/11/13 (9 runs, one job).
- `s35_r2_<P64|CW>_sK_{t,h}.list`: 2b / 4 at seed 7, test and Hard, packed 2-way in one job.
- `s35_r2_<P64|CW>_sK_{e11,e13,gp}.list`: seeds 11/13 test + Hard, and GSM-Plus at seeds 7/11/13, packed 3-way in one job.
- Eval wall time on one B200: row 2 (9 runs) about 5 min; row 4 or 2b test + Hard at seed 7 about 17 min; the 3-way
  packs (seeds 11/13 + GSM-Plus) about 1-1.5 h.
