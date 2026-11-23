## IFEval 541 (541 prompts)
| row | prompt strict % [95% CI] | prompt loose % | inst strict % [95% CI] | tok/q | per seed prompt strict (n) |
|---|---|---|---|---|---|
| 1 plain host | 12.94 [10.17, 15.90] | 13.12 | 23.98 [21.25, 26.72] | 1889 | 12.94 (1) |
| 2 host + RL (no implant) | 45.53 [41.99, 49.20] | 48.34 | 58.31 [55.66, 60.98] | 517 | 43.62 / 46.77 / 40.30 / 44.55 / 46.77 / 51.20 (6) |
| 3 host + 14-rule implant | 34.75 [30.68, 38.82] | 36.78 | 47.48 [44.12, 50.85] | 1388 | 34.75 (1) |
| 4 host + implant + RL (implant attached) | 48.49 [44.79, 52.22] | 52.87 | 60.91 [58.16, 63.70] | 1334 | 49.17 / 48.24 / 47.69 / 48.24 / 49.91 / 47.69 (6) |
| 2b RL'd plain host + implant attached at eval | 53.70 [50.09, 57.30] | 56.99 | 65.45 [62.88, 68.01] | 504 | 54.16 / 54.71 / 49.17 / 53.97 / 53.79 / 56.38 (6) |

### paired deltas (prompt strict, pp, 95% item bootstrap CI)
- 2 host + RL (no implant)  minus  1 plain host: +32.59 [+28.87, +36.29]
- 3 host + 14-rule implant  minus  1 plain host: +21.81 [+18.30, +25.51]
- 4 host + implant + RL (implant attached)  minus  1 plain host: +35.55 [+31.79, +39.31]
- 4 host + implant + RL (implant attached)  minus  3 host + 14-rule implant: +13.74 [+10.54, +16.97]
- 4 host + implant + RL (implant attached)  minus  2 host + RL (no implant): +2.96 [-0.28, +6.35]
- 2b RL'd plain host + implant attached at eval  minus  2 host + RL (no implant): +8.16 [+6.13, +10.38]
- 2b RL'd plain host + implant attached at eval  minus  4 host + implant + RL (implant attached): +5.21 [+2.46, +7.89]
