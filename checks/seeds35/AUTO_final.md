# seeds35 auto-analysis (2026-11-23T10:40Z)

Rows: 2 = P64 (1 sample), 2b = P64 + untrained rung 2 at eval (row-4 settings, stk_r2q), 4 = CW + rung 2. Row 4 seed-7 s0-2 = j23509 (stk_r2, §12.1); everything new on stk_r2q. Paired item bootstrap as ana_q32.py.

## In-run dev (u300)

| arm | status | last dev u | dev (P64: dev_acc; CW: dev_vote4) | dev2 | CW dev_vote1 | tok |
|---|---|---|---|---|---|---|
| P64_s0 | DONE | 300 | 79.83 | 81.00 | | 130 |
| P64_s1 | DONE | 300 | 79.33 | 80.00 | | 122 |
| P64_s2 | DONE | 300 | 82.17 | 81.33 | | 125 |
| P64_s3 | DONE | 300 | 80.17 | 79.67 | | 121 |
| P64_s4 | DONE | 300 | 77.50 | 79.33 | | 120 |
| P64_s5 | DONE | 300 | 83.00 | 81.00 | | 128 |
| CW_s0 | DONE | 300 | 86.50 | 82.33 | 79.17 | 140/att |
| CW_s1 | DONE | 300 | 86.67 | 85.67 | 75.00 | 153/att |
| CW_s2 | DONE | 300 | 84.17 | 84.00 | 75.17 | 140/att |
| CW_s3 | DONE | 300 | 84.83 | 80.33 | 76.17 | 153/att |
| CW_s4 | DONE | 300 | 85.17 | 84.00 | 74.50 | 137/att |
| CW_s5 | DONE | 300 | 86.00 | 83.67 | 75.67 | 143/att |

## Eval seed 7 (§12.1 table protocol)

**test, seed 7, RL seeds 0-2**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 3/3 | 68.49 | 69.07 / 67.78 / 68.61 | 129 |
| 2b | 3/3 | 82.11 | 82.11 / 81.88 / 82.34 | 1521 |
| 4 | 3/3 | 80.26 | 81.12 / 81.43 / 78.24 | 1612 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +13.62 [+12.03, +15.14] |
| 4 − 2b | -1.84 [-2.83, -0.88] |
| 4 − 2 | +11.78 [+10.24, +13.44] |

**test, seed 7, RL seeds 3-5 (new)**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 3/3 | 68.16 | 67.17 / 68.99 / 68.31 | 128 |
| 2b | 3/3 | 82.34 | 82.26 / 82.34 / 82.41 | 1549 |
| 4 | 3/3 | 80.62 | 81.35 / 79.38 / 81.12 | 1601 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +14.18 [+12.71, +15.74] |
| 4 − 2b | -1.72 [-2.73, -0.68] |
| 4 − 2 | +12.46 [+10.92, +14.13] |

**test, seed 7, RL seeds 0-5**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 6/6 | 68.32 | 69.07 / 67.78 / 68.61 / 67.17 / 68.99 / 68.31 | 129 |
| 2b | 6/6 | 82.22 | 82.11 / 81.88 / 82.34 / 82.26 / 82.34 / 82.41 | 1535 |
| 4 | 6/6 | 80.44 | 81.12 / 81.43 / 78.24 / 81.35 / 79.38 / 81.12 | 1607 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +13.90 [+12.53, +15.33] |
| 4 − 2b | -1.78 [-2.50, -1.01] |
| 4 − 2 | +12.12 [+10.77, +13.57] |

**hard, seed 7, RL seeds 0-2**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 3/3 | 41.19 | 41.24 / 41.70 / 40.64 | 164 |
| 2b | 3/3 | 61.31 | 61.64 / 61.03 / 61.26 | 2581 |
| 4 | 3/3 | 58.76 | 56.71 / 61.18 / 58.38 | 2655 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +20.12 [+18.20, +22.16] |
| 4 − 2b | -2.55 [-3.69, -1.36] |
| 4 − 2 | +17.56 [+15.59, +19.56] |

**hard, seed 7, RL seeds 3-5 (new)**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 3/3 | 42.00 | 39.58 / 43.14 / 43.29 | 161 |
| 2b | 3/3 | 61.54 | 61.03 / 61.41 / 62.17 | 2577 |
| 4 | 3/3 | 59.26 | 60.20 / 58.23 / 59.36 | 2612 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +19.54 [+17.46, +21.51] |
| 4 − 2b | -2.27 [-3.51, -0.99] |
| 4 − 2 | +17.26 [+15.21, +19.28] |

**hard, seed 7, RL seeds 0-5**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 6/6 | 41.60 | 41.24 / 41.70 / 40.64 / 39.58 / 43.14 / 43.29 | 163 |
| 2b | 6/6 | 61.42 | 61.64 / 61.03 / 61.26 / 61.03 / 61.41 / 62.17 | 2579 |
| 4 | 6/6 | 59.01 | 56.71 / 61.18 / 58.38 / 60.20 / 58.23 / 59.36 | 2634 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +19.83 [+18.04, +21.67] |
| 4 − 2b | -2.41 [-3.35, -1.38] |
| 4 − 2 | +17.41 [+15.59, +19.27] |

**gsmplus, seed 7, RL seeds 0-2**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 3/3 | 54.03 | 54.38 / 53.71 / 54.00 | 155 |
| 2b | 3/3 | 68.29 | 69.57 / 67.33 / 67.95 | 2428 |
| 4 | 3/3 | 66.41 | 66.57 / 67.05 / 65.62 | 2446 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +14.25 [+12.86, +15.71] |
| 4 − 2b | -1.87 [-2.84, -0.94] |
| 4 − 2 | +12.38 [+10.90, +13.76] |

**gsmplus, seed 7, RL seeds 3-5 (new)**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 3/3 | 53.30 | 52.95 / 52.71 / 54.24 | 153 |
| 2b | 3/3 | 68.40 | 68.38 / 67.67 / 69.14 | 2444 |
| 4 | 3/3 | 65.76 | 66.57 / 64.33 / 66.38 | 2338 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +15.10 [+13.71, +16.60] |
| 4 − 2b | -2.63 [-3.62, -1.68] |
| 4 − 2 | +12.46 [+10.90, +14.00] |

**gsmplus, seed 7, RL seeds 0-5**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 6/6 | 53.67 | 54.38 / 53.71 / 54.00 / 52.95 / 52.71 / 54.24 | 154 |
| 2b | 6/6 | 68.34 | 69.57 / 67.33 / 67.95 / 68.38 / 67.67 / 69.14 | 2436 |
| 4 | 6/6 | 66.09 | 66.57 / 67.05 / 65.62 / 66.57 / 64.33 / 66.38 | 2392 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +14.67 [+13.41, +15.98] |
| 4 − 2b | -2.25 [-3.04, -1.49] |
| 4 − 2 | +12.42 [+11.05, +13.74] |

## Averaged over eval seeds 7/11/13

**test, eval seeds 7/11/13, RL seeds 0-2**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 9/9 | 68.56 | 69.07 / 67.78 / 68.61 / 69.29 / 68.69 / 68.31 / 68.46 / 68.61 / 68.23 | 130 |
| 2b | 9/9 | 82.16 | 82.11 / 81.88 / 82.34 / 82.71 / 81.58 / 82.26 / 82.49 / 81.88 / 82.18 | 1520 |
| 4 | 9/9 | 80.19 | 81.12 / 81.43 / 78.24 / 80.67 / 80.89 / 79.15 / 81.80 / 80.21 / 78.17 | 1607 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +13.60 [+12.30, +14.98] |
| 4 − 2b | -1.97 [-2.80, -1.20] |
| 4 − 2 | +11.62 [+10.31, +12.98] |

**test, eval seeds 7/11/13, RL seeds 0-5**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 18/18 | 68.35 | 69.07 / 67.78 / 68.61 / 67.17 / 68.99 / 68.31 / 69.29 / 68.69 / 68.31 / 68.08 / 67.25 / 68.84 / 68.46 / 68.61 / 68.23 / 68.84 / 67.78 / 67.93 | 129 |
| 2b | 18/18 | 82.27 | 82.11 / 81.88 / 82.34 / 82.26 / 82.34 / 82.41 / 82.71 / 81.58 / 82.26 / 82.49 / 82.41 / 82.56 / 82.49 / 81.88 / 82.18 / 81.73 / 82.18 / 83.09 | 1531 |
| 4 | 18/18 | 80.31 | 81.12 / 81.43 / 78.24 / 81.35 / 79.38 / 81.12 / 80.67 / 80.89 / 79.15 / 81.12 / 78.85 / 81.05 / 81.80 / 80.21 / 78.17 / 80.59 / 79.91 / 80.59 | 1599 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +13.92 [+12.70, +15.20] |
| 4 − 2b | -1.96 [-2.63, -1.27] |
| 4 − 2 | +11.97 [+10.72, +13.29] |

**hard, eval seeds 7/11/13, RL seeds 0-2**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 9/9 | 41.34 | 41.24 / 41.70 / 40.64 / 42.23 / 41.70 / 40.94 / 41.85 / 41.24 / 40.56 | 164 |
| 2b | 9/9 | 61.74 | 61.64 / 61.03 / 61.26 / 62.47 / 62.17 / 62.70 / 61.56 / 61.18 / 61.64 | 2587 |
| 4 | 9/9 | 58.91 | 56.71 / 61.18 / 58.38 / 57.39 / 60.96 / 58.30 / 57.92 / 61.41 / 57.92 | 2658 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +20.39 [+18.71, +22.14] |
| 4 − 2b | -2.83 [-3.76, -1.88] |
| 4 − 2 | +17.56 [+15.78, +19.35] |

**hard, eval seeds 7/11/13, RL seeds 0-5**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 18/18 | 41.45 | 41.24 / 41.70 / 40.64 / 39.58 / 43.14 / 43.29 / 42.23 / 41.70 / 40.94 / 40.56 / 40.64 / 42.68 / 41.85 / 41.24 / 40.56 / 40.94 / 40.49 / 42.76 | 163 |
| 2b | 18/18 | 61.65 | 61.64 / 61.03 / 61.26 / 61.03 / 61.41 / 62.17 / 62.47 / 62.17 / 62.70 / 60.80 / 60.58 / 62.62 / 61.56 / 61.18 / 61.64 / 60.65 / 62.02 / 62.85 | 2578 |
| 4 | 18/18 | 59.04 | 56.71 / 61.18 / 58.38 / 60.20 / 58.23 / 59.36 / 57.39 / 60.96 / 58.30 / 59.74 / 58.45 / 60.27 / 57.92 / 61.41 / 57.92 / 59.74 / 57.39 / 59.14 | 2638 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +20.20 [+18.59, +21.91] |
| 4 − 2b | -2.62 [-3.49, -1.72] |
| 4 − 2 | +17.58 [+15.91, +19.28] |

**gsmplus, eval seeds 7/11/13, RL seeds 0-2**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 9/9 | 54.18 | 54.38 / 53.71 / 54.00 / 53.10 / 53.76 / 55.29 / 54.81 / 54.33 / 54.24 | 155 |
| 2b | 9/9 | 68.52 | 69.57 / 67.33 / 67.95 / 69.52 / 67.67 / 68.86 / 69.67 / 67.24 / 68.86 | 2427 |
| 4 | 9/9 | 66.26 | 66.57 / 67.05 / 65.62 / 65.48 / 67.29 / 64.95 / 67.29 / 67.57 / 64.57 | 2446 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +14.34 [+13.07, +15.60] |
| 4 − 2b | -2.25 [-2.99, -1.56] |
| 4 − 2 | +12.08 [+10.75, +13.32] |

**gsmplus, eval seeds 7/11/13, RL seeds 0-5**

| row | runs | mean | per run | tok/q |
|---|---|---|---|---|
| 2 | 18/18 | 53.83 | 54.38 / 53.71 / 54.00 / 52.95 / 52.71 / 54.24 / 53.10 / 53.76 / 55.29 / 52.76 / 54.14 / 53.95 / 54.81 / 54.33 / 54.24 / 53.38 / 52.67 / 54.52 | 154 |
| 2b | 18/18 | 68.54 | 69.57 / 67.33 / 67.95 / 68.38 / 67.67 / 69.14 / 69.52 / 67.67 / 68.86 / 68.81 / 68.52 / 69.24 / 69.67 / 67.24 / 68.86 / 68.67 / 67.67 / 68.90 | 2439 |
| 4 | 18/18 | 66.09 | 66.57 / 67.05 / 65.62 / 66.57 / 64.33 / 66.38 / 65.48 / 67.29 / 64.95 / 67.62 / 63.71 / 66.71 / 67.29 / 67.57 / 64.57 / 67.00 / 63.95 / 67.00 | 2389 |

| effect | Δ [95% CI] |
|---|---|
| 2b − 2 | +14.71 [+13.58, +15.91] |
| 4 − 2b | -2.44 [-3.07, -1.85] |
| 4 − 2 | +12.26 [+11.01, +13.47] |

## Per-run accuracy

| set | eval seed | row | s0 | s1 | s2 | s3 | s4 | s5 | mean s0-2 | mean s3-5 | mean s0-5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 7 | 2 | 69.07 | 67.78 | 68.61 | 67.17 | 68.99 | 68.31 | 68.49 | 68.16 | 68.32 |
| test | 7 | 2b | 82.11 | 81.88 | 82.34 | 82.26 | 82.34 | 82.41 | 82.11 | 82.34 | 82.22 |
| test | 7 | 4 | 81.12 | 81.43 | 78.24 | 81.35 | 79.38 | 81.12 | 80.26 | 80.62 | 80.44 |
| test | 11 | 2 | 69.29 | 68.69 | 68.31 | 68.08 | 67.25 | 68.84 | 68.76 | 68.06 | 68.41 |
| test | 11 | 2b | 82.71 | 81.58 | 82.26 | 82.49 | 82.41 | 82.56 | 82.18 | 82.49 | 82.34 |
| test | 11 | 4 | 80.67 | 80.89 | 79.15 | 81.12 | 78.85 | 81.05 | 80.24 | 80.34 | 80.29 |
| test | 13 | 2 | 68.46 | 68.61 | 68.23 | 68.84 | 67.78 | 67.93 | 68.44 | 68.18 | 68.31 |
| test | 13 | 2b | 82.49 | 81.88 | 82.18 | 81.73 | 82.18 | 83.09 | 82.18 | 82.34 | 82.26 |
| test | 13 | 4 | 81.80 | 80.21 | 78.17 | 80.59 | 79.91 | 80.59 | 80.06 | 80.36 | 80.21 |
| hard | 7 | 2 | 41.24 | 41.70 | 40.64 | 39.58 | 43.14 | 43.29 | 41.19 | 42.00 | 41.60 |
| hard | 7 | 2b | 61.64 | 61.03 | 61.26 | 61.03 | 61.41 | 62.17 | 61.31 | 61.54 | 61.42 |
| hard | 7 | 4 | 56.71 | 61.18 | 58.38 | 60.20 | 58.23 | 59.36 | 58.76 | 59.26 | 59.01 |
| hard | 11 | 2 | 42.23 | 41.70 | 40.94 | 40.56 | 40.64 | 42.68 | 41.62 | 41.29 | 41.46 |
| hard | 11 | 2b | 62.47 | 62.17 | 62.70 | 60.80 | 60.58 | 62.62 | 62.45 | 61.33 | 61.89 |
| hard | 11 | 4 | 57.39 | 60.96 | 58.30 | 59.74 | 58.45 | 60.27 | 58.88 | 59.49 | 59.19 |
| hard | 13 | 2 | 41.85 | 41.24 | 40.56 | 40.94 | 40.49 | 42.76 | 41.22 | 41.39 | 41.31 |
| hard | 13 | 2b | 61.56 | 61.18 | 61.64 | 60.65 | 62.02 | 62.85 | 61.46 | 61.84 | 61.65 |
| hard | 13 | 4 | 57.92 | 61.41 | 57.92 | 59.74 | 57.39 | 59.14 | 59.09 | 58.76 | 58.92 |
| gsmplus | 7 | 2 | 54.38 | 53.71 | 54.00 | 52.95 | 52.71 | 54.24 | 54.03 | 53.30 | 53.67 |
| gsmplus | 7 | 2b | 69.57 | 67.33 | 67.95 | 68.38 | 67.67 | 69.14 | 68.29 | 68.40 | 68.34 |
| gsmplus | 7 | 4 | 66.57 | 67.05 | 65.62 | 66.57 | 64.33 | 66.38 | 66.41 | 65.76 | 66.09 |
| gsmplus | 11 | 2 | 53.10 | 53.76 | 55.29 | 52.76 | 54.14 | 53.95 | 54.05 | 53.62 | 53.83 |
| gsmplus | 11 | 2b | 69.52 | 67.67 | 68.86 | 68.81 | 68.52 | 69.24 | 68.68 | 68.86 | 68.77 |
| gsmplus | 11 | 4 | 65.48 | 67.29 | 64.95 | 67.62 | 63.71 | 66.71 | 65.90 | 66.02 | 65.96 |
| gsmplus | 13 | 2 | 54.81 | 54.33 | 54.24 | 53.38 | 52.67 | 54.52 | 54.46 | 53.52 | 53.99 |
| gsmplus | 13 | 2b | 69.67 | 67.24 | 68.86 | 68.67 | 67.67 | 68.90 | 68.59 | 68.41 | 68.50 |
| gsmplus | 13 | 4 | 67.29 | 67.57 | 64.57 | 67.00 | 63.95 | 67.00 | 66.48 | 65.98 | 66.23 |

## Writer audit (new runs)

- FE audit test r2_CW_s3_s11 writer_steps=130782 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_CW_s3_s11 writer_steps=182607 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit test r2_CW_s3_s13 writer_steps=130328 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_CW_s3_s13 writer_steps=181018 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit gsmplus gp_r2_CW_s3 writer_steps=248403 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_CW_s3_s11 writer_steps=248579 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit gsmplus gp_r2_CW_s3_s13 writer_steps=249306 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_CW_s3 writer_steps=180838 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit test r2_CW_s3 writer_steps=131647 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit test r2_CW_s4_s11 writer_steps=109409 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_CW_s4_s11 writer_steps=149429 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit test r2_CW_s4_s13 writer_steps=108022 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit hard r2_CW_s4_s13 writer_steps=149381 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_CW_s4 writer_steps=218223 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_CW_s4_s11 writer_steps=217036 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_CW_s4_s13 writer_steps=216541 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit hard r2_CW_s4 writer_steps=148379 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit test r2_CW_s4 writer_steps=110607 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit test r2_CW_s5_s11 writer_steps=110955 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_CW_s5_s11 writer_steps=152482 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit test r2_CW_s5_s13 writer_steps=110354 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_CW_s5_s13 writer_steps=152403 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_CW_s5 writer_steps=218159 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_CW_s5_s11 writer_steps=218941 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_CW_s5_s13 writer_steps=217334 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_CW_s5 writer_steps=152185 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit test r2_CW_s5 writer_steps=110709 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit test r2_P64_s3_s11 writer_steps=98433 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_P64_s3_s11 writer_steps=145917 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit test r2_P64_s3_s13 writer_steps=100357 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_P64_s3_s13 writer_steps=145576 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_P64_s3 writer_steps=208469 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_P64_s3_s11 writer_steps=208684 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_P64_s3_s13 writer_steps=209163 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_P64_s3 writer_steps=145749 sampled_ne_written=2 rows_with_mis=2 stop_tok=151669
- FE audit test r2_P64_s3 writer_steps=101770 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit test r2_P64_s4_s11 writer_steps=104690 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_P64_s4_s11 writer_steps=149489 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit test r2_P64_s4_s13 writer_steps=104451 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_P64_s4_s13 writer_steps=149027 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_P64_s4 writer_steps=214988 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit gsmplus gp_r2_P64_s4_s11 writer_steps=214662 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_P64_s4_s13 writer_steps=215053 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit hard r2_P64_s4 writer_steps=150686 sampled_ne_written=2 rows_with_mis=2 stop_tok=151669
- FE audit test r2_P64_s4 writer_steps=105188 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit test r2_P64_s5_s11 writer_steps=106915 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_P64_s5_s11 writer_steps=145538 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit test r2_P64_s5_s13 writer_steps=105789 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_P64_s5_s13 writer_steps=146451 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_P64_s5 writer_steps=211073 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit gsmplus gp_r2_P64_s5_s11 writer_steps=210442 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit gsmplus gp_r2_P64_s5_s13 writer_steps=212631 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669
- FE audit hard r2_P64_s5 writer_steps=145733 sampled_ne_written=1 rows_with_mis=1 stop_tok=151669
- FE audit test r2_P64_s5 writer_steps=106067 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669

