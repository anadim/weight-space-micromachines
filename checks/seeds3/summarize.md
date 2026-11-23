## (b) m4: noisy-routing SFT host + library, mixed task, 150 u. Off-type FIRED calls per answer (triggers x in-wire pass), greedy

- **s0 (orig, H100)**: dev all u0 47.8 -> u150 61.0; test 400 tools on 62.8 (gsm 39.1 / bbh 84.7 / ifeval 76.1 / combo 36.1), tools off 32.2, host only 28.8
  | off-type call | dev u0 | dev u150 | test u150 |
  |---|---|---|---|
  | calc@bbh | 0.16 | 0.04 | 0.11 |
  | calc@ifeval | 0.22 | 0.23 | 0.34 |
  | expr_gen@ifeval | 0.22 | 0.23 | 0.34 |
  | order_gen@gsm | 0.01 | 0.00 | 0.00 |
  | track_gen@gsm | 0.00 | 0.02 | 0.00 |
  | track_gen@ifeval | 0.03 | 0.02 | 0.01 |
  | truth_gen@gsm | 0.01 | 0.03 | 0.00 |
  | turtle_gen@gsm | 0.02 | 0.01 | 0.00 |
  | turtle_gen@ifeval | 0.06 | 0.00 | 0.00 |
  - LOO calc (test): ifeval 76.1 -> 76.1, bbh 84.7 -> 84.7
  - LOO expr_gen (test): ifeval 76.1 -> 76.1, bbh 84.7 -> 75.9
- **s1**: dev all u0 46.2 -> u150 63.8; test 400 tools on 63.8 (gsm 40.0 / bbh 83.9 / ifeval 82.6 / combo 32.8), tools off 34.5, host only 29.0
  | off-type call | dev u0 | dev u150 | test u150 |
  |---|---|---|---|
  | calc@bbh | 0.15 | 0.05 | 0.07 |
  | calc@ifeval | 0.22 | 0.00 | 0.01 |
  | expr_gen@ifeval | 0.22 | 0.00 | 0.01 |
  | order_gen@gsm | 0.01 | 0.01 | 0.00 |
  | track_gen@gsm | 0.00 | 0.01 | 0.01 |
  | track_gen@ifeval | 0.00 | 0.04 | 0.01 |
  | truth_gen@gsm | 0.00 | 0.02 | 0.01 |
  | turtle_gen@gsm | 0.02 | 0.02 | 0.00 |
  | turtle_gen@ifeval | 0.02 | 0.01 | 0.01 |
  - LOO calc (test): ifeval 82.6 -> 82.6, bbh 83.9 -> 83.9
  - LOO expr_gen (test): ifeval 82.6 -> 82.6, bbh 83.9 -> 76.6
- **s2**: dev all u0 46.2 -> u150 65.2; test 400 tools on 63.8 (gsm 40.0 / bbh 85.4 / ifeval 82.6 / combo 29.5), tools off 34.2, host only 29.0
  | off-type call | dev u0 | dev u150 | test u150 |
  |---|---|---|---|
  | calc@bbh | 0.15 | 0.03 | 0.12 |
  | calc@ifeval | 0.22 | 0.06 | 0.08 |
  | expr_gen@ifeval | 0.22 | 0.07 | 0.09 |
  | order_gen@gsm | 0.01 | 0.00 | 0.00 |
  | order_gen@ifeval | 0.00 | 0.01 | 0.00 |
  | track_gen@ifeval | 0.00 | 0.01 | 0.00 |
  | turtle_gen@gsm | 0.02 | 0.00 | 0.00 |
  | turtle_gen@ifeval | 0.02 | 0.05 | 0.09 |
  - LOO calc (test): ifeval 82.6 -> 82.6, bbh 85.4 -> 85.4
  - LOO expr_gen (test): ifeval 82.6 -> 82.6, bbh 85.4 -> 76.6

calculator FIRED calls per answer on greedy dev, by update (the (b) table of RESULTS.md):
- m4 s0 (orig, H100) calc@bbh: u0 0.16 -> u25 0.07 -> u50 0.05 -> u100 0.03 -> u150 0.04
- m4 s0 (orig, H100) calc@ifeval: u0 0.22 -> u25 0.01 -> u50 0.15 -> u100 0.28 -> u150 0.23
- m4 s1 calc@bbh: u0 0.15 -> u25 0.04 -> u50 0.07 -> u100 0.03 -> u150 0.05
- m4 s1 calc@ifeval: u0 0.22 -> u25 0.00 -> u50 0.01 -> u100 0.00 -> u150 0.00
- m4 s2 calc@bbh: u0 0.15 -> u25 0.07 -> u50 0.06 -> u100 0.03 -> u150 0.03
- m4 s2 calc@ifeval: u0 0.22 -> u25 0.00 -> u50 0.01 -> u100 0.00 -> u150 0.06

## (c) m2: scrambled library (all but calc_s4), mixed task, 150 u. Greedy in-wire pass rate on test 400 at u150

| seed | test all / tools off | calc | track_gen | order_gen | interval_gen | turtle_gen | truth_gen | expr_gen | calc_gen | bool_gen | ifeval | closed / fire |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 (orig, A100) | 37.8 / 37.5 | 0% (299) | - | - | 100% (15) | 0% (13) | - | 0% (275) | 100% (76) | 0% (17) | 100% (600) | 4 / 7 (calc, turtle_gen, expr_gen, bool_gen) |
| s1 | 44.0 / 39.0 | 100% (402) | 100% (5) | 0% (11) | 0% (1) | - | 100% (1) | 0% (426) | 0% (85) | 0% (11) | 100% (617) | 5 / 9 (order_gen, interval_gen, expr_gen, calc_gen, bool_gen) |
| s2 | 38.8 / 35.5 | 100% (354) | 100% (2) | 0% (8) | 0% (11) | 100% (29) | 78% (18) | 0% (324) | 0% (92) | 0% (26) | 100% (488) | 5 / 10 (order_gen, interval_gen, expr_gen, calc_gen, bool_gen) |

cells: % of triggers the in-wire lets through (number of triggers on the 400 test items); "-" = never triggers.
- m2 s0 (orig, A100) dev: u0 20.8, u25 24.2, u50 29.5, u100 32.0, u150 35.2
- m2 s1 dev: u0 17.0, u25 29.2, u50 35.0, u100 43.5, u150 45.2
- m2 s2 dev: u0 17.0, u25 31.2, u50 31.0, u100 36.8, u150 40.2
