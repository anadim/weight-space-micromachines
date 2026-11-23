n = 541 prompts; tok/q 2b 546, 4 1364, 2 573

| comparison | Δ prompt-strict (pp) [95% CI] |
|---|---|
| 2b − 4, seed-averaged | +4.31 [+1.17, +7.33] |
| 2b − 2, seed-averaged (check vs ledger +9.12) | +9.12 [+6.84, +11.52] |
| 4 − 2, seed-averaged (check vs ledger +4.81) | +4.81 [+1.11, +8.56] |
| 2b − 4, seed 0 vs seed 0 | +4.99 [+0.92, +9.06] |
| 2b − 4, seed 1 vs seed 1 | +6.47 [+2.22, +10.54] |
| 2b − 4, seed 2 vs seed 2 | +1.48 [-2.40, +5.36] |

seed-majority prompt-strict: 2b 53.05, 4 48.61; 2b right & 4 wrong 67, 4 right & 2b wrong 43
per seed 2b / 4: 54.16 / 49.17 | 54.71 / 48.24 | 49.17 / 47.69
