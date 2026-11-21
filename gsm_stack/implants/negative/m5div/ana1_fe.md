
## dev (src=fe, 600 items, 4 seeds)

| key | distinct/8 (mean) | P(pair same key) | P(same ans / same key) | P(same ans / diff key) | H(ans|key)/H(ans) | same (mixed) | P(same ans / same key) mixed | / diff key mixed |
|---|---|---|---|---|---|---|---|---|
| W4 | 2.81 | 0.560 | 0.819 | 0.657 | 0.574 | 0.490 | 0.655 | 0.496 |
| W8 | 3.41 | 0.455 | 0.828 | 0.680 | 0.501 | 0.384 | 0.660 | 0.520 |
| L1 | 4.12 | 0.368 | 0.837 | 0.696 | 0.404 | 0.306 | 0.676 | 0.529 |
| L1N | 2.08 | 0.695 | 0.803 | 0.621 | 0.751 | 0.620 | 0.628 | 0.485 |
| N2 | 1.84 | 0.749 | 0.817 | 0.540 | 0.704 | 0.669 | 0.655 | 0.410 |
| EQ | 2.96 | 0.509 | 0.855 | 0.635 | 0.543 | 0.403 | 0.696 | 0.491 |

| key | gap cases | some right att key not used by any winner-answer att (same seed) | ... (any att of 32 with winner answer) | right atts with key unique in seed |
|---|---|---|---|---|
| W4 | 116 | 55 (47%) | 38 (33%) | 49/191 |
| W8 | 116 | 62 (53%) | 45 (39%) | 62/191 |
| L1 | 116 | 75 (65%) | 63 (54%) | 84/191 |
| L1N | 116 | 39 (34%) | 28 (24%) | 31/191 |
| N2 | 116 | 35 (30%) | 21 (18%) | 26/191 |
| EQ | 116 | 69 (59%) | 56 (48%) | 68/191 |

| policy | vote | oracle@8 | per-att | distinct ans/8 | distinct key/8 |
|---|---|---|---|---|---|
| random 8 of 32 | 87.96 | 92.77 | 77.83 | 2.15 | 4.06 |
| W4-distinct | 88.26 | 92.58 | 75.40 | 2.40 | 4.47 |
| W8-distinct | 88.15 | 92.82 | 75.94 | 2.36 | 5.46 |
| L1-distinct | 88.12 | 93.15 | 76.28 | 2.32 | 6.25 |
| L1N-distinct | 87.83 | 93.24 | 76.83 | 2.26 | 3.00 |
| N2-distinct | 87.74 | 92.86 | 75.71 | 2.36 | 2.60 |
| EQ-distinct | 88.17 | 93.68 | 76.75 | 2.29 | 4.58 |
| W4-distinct if prev agree | 88.07 | 92.91 | 77.39 | 2.19 | 3.63 |
| L1-distinct if prev agree | 87.98 | 93.11 | 77.51 | 2.18 | 5.37 |
| EQ-distinct if prev agree | 88.02 | 93.28 | 77.34 | 2.20 | 3.71 |
| answer-distinct (bound) | 84.41 | 94.42 | 66.78 | 3.34 | 4.24 |
| EQ: avoid plurality-group keys if its mean lg < 0.0 | 87.92 | 92.93 | 77.56 | 2.18 | 3.09 |
| EQ: avoid plurality-group keys if its mean lg < 1.0 | 87.77 | 93.21 | 76.96 | 2.24 | 3.39 |
| EQ: avoid plurality-group keys if its mean lg < 2.0 | 87.73 | 93.24 | 75.89 | 2.32 | 3.79 |
| EQ: avoid plurality-group keys if its mean lg < 99.0 | 87.66 | 93.33 | 74.99 | 2.38 | 4.14 |
| L1: avoid plurality-group keys if its mean lg < 0.0 | 88.12 | 92.81 | 77.70 | 2.17 | 4.24 |
| L1: avoid plurality-group keys if its mean lg < 1.0 | 88.19 | 92.97 | 77.49 | 2.19 | 4.60 |
| L1: avoid plurality-group keys if its mean lg < 2.0 | 88.22 | 93.17 | 76.91 | 2.24 | 5.18 |
| L1: avoid plurality-group keys if its mean lg < 99.0 | 88.21 | 93.17 | 76.44 | 2.28 | 5.90 |
| W4: avoid plurality-group keys if its mean lg < 0.0 | 88.23 | 92.94 | 77.71 | 2.18 | 2.94 |
| W4: avoid plurality-group keys if its mean lg < 1.0 | 88.35 | 92.88 | 77.25 | 2.22 | 3.28 |
| W4: avoid plurality-group keys if its mean lg < 2.0 | 88.42 | 92.94 | 76.38 | 2.29 | 3.71 |
| W4: avoid plurality-group keys if its mean lg < 99.0 | 88.37 | 92.86 | 75.74 | 2.35 | 4.17 |
| actual seeds (serial order) | 87.96 | 92.79 | | | |

## dev2 (src=fe, 300 items, 4 seeds)

| key | distinct/8 (mean) | P(pair same key) | P(same ans / same key) | P(same ans / diff key) | H(ans|key)/H(ans) | same (mixed) | P(same ans / same key) mixed | / diff key mixed |
|---|---|---|---|---|---|---|---|---|
| W4 | 3.17 | 0.490 | 0.796 | 0.585 | 0.491 | 0.406 | 0.621 | 0.428 |
| W8 | 3.83 | 0.387 | 0.809 | 0.612 | 0.418 | 0.314 | 0.634 | 0.448 |
| L1 | 4.50 | 0.314 | 0.817 | 0.630 | 0.342 | 0.245 | 0.646 | 0.461 |
| L1N | 2.22 | 0.654 | 0.765 | 0.544 | 0.719 | 0.574 | 0.591 | 0.392 |
| N2 | 1.98 | 0.715 | 0.788 | 0.439 | 0.658 | 0.629 | 0.620 | 0.313 |
| EQ | 3.07 | 0.480 | 0.812 | 0.574 | 0.550 | 0.403 | 0.649 | 0.410 |

| key | gap cases | some right att key not used by any winner-answer att (same seed) | ... (any att of 32 with winner answer) | right atts with key unique in seed |
|---|---|---|---|---|
| W4 | 75 | 37 (49%) | 24 (32%) | 33/133 |
| W8 | 75 | 46 (61%) | 32 (43%) | 44/133 |
| L1 | 75 | 57 (76%) | 45 (60%) | 63/133 |
| L1N | 75 | 27 (36%) | 16 (21%) | 15/133 |
| N2 | 75 | 21 (28%) | 16 (21%) | 15/133 |
| EQ | 75 | 49 (65%) | 38 (51%) | 50/133 |

| policy | vote | oracle@8 | per-att | distinct ans/8 | distinct key/8 |
|---|---|---|---|---|---|
| random 8 of 32 | 85.70 | 91.89 | 73.44 | 2.48 | 4.42 |
| W4-distinct | 85.40 | 90.95 | 69.83 | 2.85 | 4.99 |
| W8-distinct | 85.57 | 91.18 | 70.81 | 2.76 | 5.95 |
| L1-distinct | 85.58 | 91.35 | 71.59 | 2.69 | 6.59 |
| L1N-distinct | 85.46 | 91.32 | 72.17 | 2.62 | 3.25 |
| N2-distinct | 85.24 | 91.16 | 70.37 | 2.79 | 2.88 |
| EQ-distinct | 85.25 | 92.41 | 72.01 | 2.64 | 4.67 |
| W4-distinct if prev agree | 85.46 | 91.86 | 72.93 | 2.53 | 3.89 |
| L1-distinct if prev agree | 85.62 | 92.02 | 73.18 | 2.51 | 5.57 |
| EQ-distinct if prev agree | 85.68 | 92.18 | 72.95 | 2.53 | 3.67 |
| answer-distinct (bound) | 80.66 | 93.71 | 60.56 | 3.86 | 4.66 |
| EQ: avoid plurality-group keys if its mean lg < 0.0 | 85.62 | 91.84 | 72.87 | 2.54 | 3.22 |
| EQ: avoid plurality-group keys if its mean lg < 1.0 | 85.30 | 91.76 | 71.93 | 2.61 | 3.50 |
| EQ: avoid plurality-group keys if its mean lg < 2.0 | 85.46 | 91.85 | 70.53 | 2.74 | 3.86 |
| EQ: avoid plurality-group keys if its mean lg < 99.0 | 85.59 | 91.89 | 69.45 | 2.81 | 4.13 |
| L1: avoid plurality-group keys if its mean lg < 0.0 | 85.74 | 91.85 | 73.33 | 2.51 | 4.62 |
| L1: avoid plurality-group keys if its mean lg < 1.0 | 85.73 | 91.97 | 73.14 | 2.53 | 4.96 |
| L1: avoid plurality-group keys if its mean lg < 2.0 | 85.69 | 91.62 | 72.35 | 2.59 | 5.58 |
| L1: avoid plurality-group keys if its mean lg < 99.0 | 85.74 | 91.72 | 71.92 | 2.64 | 6.15 |
| W4: avoid plurality-group keys if its mean lg < 0.0 | 85.60 | 91.89 | 73.07 | 2.53 | 3.34 |
| W4: avoid plurality-group keys if its mean lg < 1.0 | 85.73 | 91.77 | 72.52 | 2.58 | 3.69 |
| W4: avoid plurality-group keys if its mean lg < 2.0 | 85.61 | 91.66 | 71.25 | 2.70 | 4.19 |
| W4: avoid plurality-group keys if its mean lg < 99.0 | 85.63 | 91.70 | 70.62 | 2.75 | 4.57 |
| actual seeds (serial order) | 85.92 | 92.17 | | | |
