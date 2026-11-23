## IFEval 541 (541 prompts)
| row | prompt strict % [95% CI] | prompt loose % | inst strict % [95% CI] | tok/q | per seed prompt strict (n) |
|---|---|---|---|---|---|
| 1 plain host | 17.19 [14.05, 20.52] | 18.48 | 31.29 [28.31, 34.28] | 1467 | 17.19 (1) |
| 2 host + RL (no implant) | 62.97 [59.33, 66.54] | 65.99 | 73.30 [70.87, 75.75] | 390 | 61.92 / 62.48 / 64.51 (3) |
| 3 host + 14-rule implant | 35.12 [31.05, 39.19] | 37.71 | 48.92 [45.44, 52.39] | 1440 | 35.12 (1) |
| 4 host + implant + RL (implant attached) | 74.68 [71.35, 78.00] | 76.96 | 82.21 [79.91, 84.46] | 595 | 74.86 / 73.57 / 75.60 (3) |
| 2b RL'd plain host + implant attached at eval | 72.34 [69.01, 75.72] | 75.11 | 80.42 [78.12, 82.68] | 434 | 71.90 / 71.53 / 73.57 (3) |
| 4p implant-RL'd host, implant detached at eval | 59.58 [55.82, 63.34] | 61.18 | 70.14 [67.46, 72.76] | 579 | 59.33 / 59.15 / 60.26 (3) |

### paired deltas (prompt strict, pp, 95% item bootstrap CI)
- 2 host + RL (no implant)  minus  1 plain host: +45.78 [+41.34, +50.28]
- 3 host + 14-rule implant  minus  1 plain host: +17.93 [+14.60, +21.44]
- 4 host + implant + RL (implant attached)  minus  1 plain host: +57.49 [+53.05, +61.80]
- 4 host + implant + RL (implant attached)  minus  3 host + 14-rule implant: +39.56 [+35.06, +44.05]
- 4 host + implant + RL (implant attached)  minus  2 host + RL (no implant): +11.71 [+8.19, +15.16]
- 2b RL'd plain host + implant attached at eval  minus  2 host + RL (no implant): +9.37 [+7.02, +11.77]
- 2b RL'd plain host + implant attached at eval  minus  4 host + implant + RL (implant attached): -2.34 [-5.18, +0.43]
- 4p implant-RL'd host, implant detached at eval  minus  2 host + RL (no implant): -3.39 [-6.65, -0.18]
- 4 host + implant + RL (implant attached)  minus  4p implant-RL'd host, implant detached at eval: +15.10 [+12.32, +18.05]

## 0.6B next to 1.7B (IFEval 541 prompt strict %, mean over seeds [95% item-bootstrap CI]; per seed)

| row | Qwen3-0.6B-Base | Qwen3-1.7B-Base |
|---|---|---|
| 1 | 12.94, tok/q 1889 | 17.19, tok/q 1467 |
| 2 | 43.56 [39.86, 47.44] (43.62 / 46.77 / 40.30), tok/q 573 | 62.97 [59.33, 66.54] (61.92 / 62.48 / 64.51), tok/q 390 |
| 3 | 34.75, tok/q 1388 | 35.12, tok/q 1440 |
| 4 | 48.37 [44.49, 52.19] (49.17 / 48.24 / 47.69), tok/q 1364 | 74.68 [71.35, 78.00] (74.86 / 73.57 / 75.60), tok/q 595 |
| 2b | 52.68 [48.92, 56.44] (54.16 / 54.71 / 49.17), tok/q 546 | 72.34 [69.01, 75.72] (71.90 / 71.53 / 73.57), tok/q 434 |
| 4p | 30.31 [26.86, 33.89] (31.79 / 30.31 / 28.84), tok/q 1582 | 59.58 [55.82, 63.34] (59.33 / 59.15 / 60.26), tok/q 579 |
