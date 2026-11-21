# linefix gate results

## dev: 19191 attempts, per-attempt acc 80.99

| thr (train line FPR) | flagged att | gen coverage | flagged wrong (n, P_fix) | ... at D line | before D | after D / no D | flagged right (n, P_keep) | per-att delta (pts, covered) | delta scaled to full coverage | delta, breaks in excess of ctrl only (scaled) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 905 (4.7%) | 905/1166 | 592, 0.144 | 235, 0.174 | 65, 0.258 | 292, 0.094 | 313, 0.756 | +0.05 | +0.06 | +0.10 |
| 0.02 | 1641 (8.6%) | 1641/1936 | 984, 0.143 | 373, 0.170 | 124, 0.282 | 487, 0.086 | 657, 0.800 | +0.05 | +0.06 | +0.13 |
| 0.05 | 1482 (7.7%) | 1482/3707 | 867, 0.146 | 348, 0.167 | 113, 0.285 | 406, 0.090 | 615, 0.807 | +0.04 | +0.11 | +0.25 |

control (random line of unflagged attempts at 5% thr): right n=270 P_keep=0.955; wrong n=30 P_fix=0.050

| thr | plurality vote base | after rewind (mean of 50 draws) | delta (pts) |
|---|---|---|---|
| 0.01 | 87.25 | 87.37 | +0.12 |
| 0.02 | 87.25 | 87.41 | +0.16 |
| 0.05 | 87.25 | 87.33 | +0.08 |

## dev2: 9600 attempts, per-attempt acc 79.19

| thr (train line FPR) | flagged att | gen coverage | flagged wrong (n, P_fix) | ... at D line | before D | after D / no D | flagged right (n, P_keep) | per-att delta (pts, covered) | delta scaled to full coverage | delta, breaks in excess of ctrl only (scaled) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 502 (5.2%) | 502/623 | 300, 0.158 | 135, 0.193 | 41, 0.250 | 124, 0.089 | 202, 0.785 | +0.04 | +0.05 | +0.11 |
| 0.02 | 880 (9.2%) | 880/1025 | 518, 0.136 | 218, 0.161 | 82, 0.265 | 218, 0.062 | 362, 0.791 | -0.06 | -0.07 | +0.04 |
| 0.05 | 781 (8.1%) | 781/2091 | 454, 0.145 | 207, 0.162 | 77, 0.269 | 170, 0.069 | 327, 0.796 | -0.01 | -0.02 | +0.19 |

control (random line of unflagged attempts at 5% thr): right n=274 P_keep=0.943; wrong n=26 P_fix=0.279

| thr | plurality vote base | after rewind (mean of 50 draws) | delta (pts) |
|---|---|---|---|
| 0.01 | 85.17 | 85.33 | +0.16 |
| 0.02 | 85.17 | 85.27 | +0.11 |
| 0.05 | 85.17 | 85.23 | +0.06 |

Fresh-attempt baseline: item p_right (n_right/32) of the flagged attempts vs rewind outcome; zero-break bound = gain if no flagged right attempt broke (scaled to full coverage)

| set | thr | flagged wrong: item p_right | P_fix | flagged right: item p_right | P_keep | zero-break bound (pts) |
|---|---|---|---|---|---|---|
| dev | 0.01 | 0.323 | 0.144 | 0.763 | 0.756 | +0.57 |
| dev | 0.02 | 0.322 | 0.143 | 0.778 | 0.800 | +0.86 |
| dev | 0.05 | 0.328 | 0.146 | 0.789 | 0.807 | +1.66 |
| dev2 | 0.01 | 0.298 | 0.158 | 0.694 | 0.785 | +0.61 |
| dev2 | 0.02 | 0.299 | 0.136 | 0.695 | 0.791 | +0.85 |
| dev2 | 0.05 | 0.300 | 0.145 | 0.707 | 0.796 | +1.84 |
