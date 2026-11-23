## IFEval 541 (541 prompts)
| row | prompt strict % [95% CI] | prompt loose % | inst strict % [95% CI] | tok/q | per seed prompt strict (n) |
|---|---|---|---|---|---|
| 1 plain host | 12.94 [10.17, 15.90] | 13.12 | 23.98 [21.25, 26.72] | 1889 | 12.94 (1) |
| 2 host + RL (no implant) | 43.56 [39.86, 47.44] | 45.96 | 56.59 [53.73, 59.46] | 573 | 43.62 / 46.77 / 40.30 (3) |
| 3 host + 14-rule implant | 34.75 [30.68, 38.82] | 36.78 | 47.48 [44.12, 50.85] | 1388 | 34.75 (1) |
| 4 host + implant + RL (implant attached) | 48.37 [44.49, 52.19] | 52.87 | 60.35 [57.45, 63.28] | 1364 | 49.17 / 48.24 / 47.69 (3) |
| 2b RL'd plain host + implant attached at eval | 52.68 [48.92, 56.44] | 55.51 | 64.51 [61.80, 67.24] | 546 | 54.16 / 54.71 / 49.17 (3) |

### paired deltas (prompt strict, pp, 95% item bootstrap CI)
- 2 host + RL (no implant)  minus  1 plain host: +30.62 [+26.68, +34.50]
- 3 host + 14-rule implant  minus  1 plain host: +21.81 [+18.30, +25.51]
- 4 host + implant + RL (implant attached)  minus  1 plain host: +35.43 [+31.55, +39.25]
- 4 host + implant + RL (implant attached)  minus  3 host + 14-rule implant: +13.62 [+10.41, +16.94]
- 4 host + implant + RL (implant attached)  minus  2 host + RL (no implant): +4.81 [+1.11, +8.56]
- 2b RL'd plain host + implant attached at eval  minus  2 host + RL (no implant): +9.12 [+6.84, +11.52]
- 2b RL'd plain host + implant attached at eval  minus  4 host + implant + RL (implant attached): +4.31 [+1.17, +7.33]
