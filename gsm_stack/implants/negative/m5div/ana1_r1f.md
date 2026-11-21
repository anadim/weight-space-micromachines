
## dev (src=gsm_stack/implants/aw/out/r1f_{st}, 600 items, 4 seeds)

| key | distinct/8 (mean) | P(pair same key) | P(same ans / same key) | P(same ans / diff key) | H(ans|key)/H(ans) | same (mixed) | P(same ans / same key) mixed | / diff key mixed |
|---|---|---|---|---|---|---|---|---|
| W4 | 2.64 | 0.584 | 0.816 | 0.755 | 0.758 | 0.529 | 0.635 | 0.595 |
| W8 | 3.28 | 0.476 | 0.823 | 0.761 | 0.664 | 0.420 | 0.640 | 0.599 |
| L1 | 4.03 | 0.379 | 0.837 | 0.762 | 0.521 | 0.330 | 0.664 | 0.593 |
| L1N | 2.02 | 0.708 | 0.821 | 0.718 | 0.818 | 0.646 | 0.640 | 0.573 |
| N2 | 1.68 | 0.788 | 0.817 | 0.691 | 0.852 | 0.730 | 0.640 | 0.552 |
| EQ | 2.98 | 0.516 | 0.872 | 0.704 | 0.561 | 0.404 | 0.704 | 0.556 |

| key | gap cases | some right att key not used by any winner-answer att (same seed) | ... (any att of 32 with winner answer) | right atts with key unique in seed |
|---|---|---|---|---|
| W4 | 119 | 47 (39%) | 32 (27%) | 35/197 |
| W8 | 119 | 60 (50%) | 42 (35%) | 47/197 |
| L1 | 119 | 77 (65%) | 62 (52%) | 79/197 |
| L1N | 119 | 29 (24%) | 18 (15%) | 18/197 |
| N2 | 119 | 27 (23%) | 18 (15%) | 17/197 |
| EQ | 119 | 65 (55%) | 54 (45%) | 56/197 |

| policy | vote | oracle@8 | per-att | distinct ans/8 | distinct key/8 |
|---|---|---|---|---|---|
| random 8 of 32 | 88.75 | 93.69 | 80.93 | 1.86 | 3.97 |
| W4-distinct | 88.76 | 93.56 | 80.80 | 1.88 | 4.12 |
| W8-distinct | 88.63 | 93.77 | 80.83 | 1.88 | 5.21 |
| L1-distinct | 88.54 | 93.67 | 80.77 | 1.88 | 6.11 |
| L1N-distinct | 88.38 | 93.76 | 80.43 | 1.90 | 2.81 |
| N2-distinct | 88.77 | 93.89 | 80.52 | 1.90 | 2.17 |
| EQ-distinct | 88.83 | 94.20 | 80.02 | 1.96 | 4.49 |
| W4-distinct if prev agree | 88.70 | 93.71 | 80.74 | 1.87 | 3.56 |
| L1-distinct if prev agree | 88.68 | 93.88 | 80.76 | 1.88 | 5.40 |
| EQ-distinct if prev agree | 88.65 | 94.01 | 80.53 | 1.90 | 3.77 |
| answer-distinct (bound) | 86.01 | 95.82 | 72.96 | 2.82 | 3.98 |
| EQ: avoid plurality-group keys if its mean lg < 0.0 | 88.84 | 93.97 | 80.81 | 1.88 | 3.08 |
| EQ: avoid plurality-group keys if its mean lg < 1.0 | 88.89 | 94.36 | 80.49 | 1.91 | 3.41 |
| EQ: avoid plurality-group keys if its mean lg < 2.0 | 88.73 | 94.32 | 79.84 | 1.95 | 3.83 |
| EQ: avoid plurality-group keys if its mean lg < 99.0 | 88.62 | 94.40 | 79.50 | 1.96 | 4.19 |
| L1: avoid plurality-group keys if its mean lg < 0.0 | 88.53 | 93.61 | 80.83 | 1.87 | 4.14 |
| L1: avoid plurality-group keys if its mean lg < 1.0 | 88.63 | 93.84 | 80.75 | 1.88 | 4.51 |
| L1: avoid plurality-group keys if its mean lg < 2.0 | 88.70 | 93.86 | 80.59 | 1.89 | 5.09 |
| L1: avoid plurality-group keys if its mean lg < 99.0 | 88.70 | 94.03 | 80.56 | 1.89 | 5.79 |
| W4: avoid plurality-group keys if its mean lg < 0.0 | 88.63 | 93.77 | 80.76 | 1.87 | 2.75 |
| W4: avoid plurality-group keys if its mean lg < 1.0 | 88.78 | 93.73 | 80.73 | 1.88 | 3.07 |
| W4: avoid plurality-group keys if its mean lg < 2.0 | 88.84 | 93.70 | 80.51 | 1.89 | 3.47 |
| W4: avoid plurality-group keys if its mean lg < 99.0 | 88.95 | 93.80 | 80.56 | 1.89 | 3.91 |
| actual seeds (serial order) | 88.79 | 93.75 | | | |

## dev2 (src=gsm_stack/implants/aw/out/r1f_{st}, 300 items, 4 seeds)

| key | distinct/8 (mean) | P(pair same key) | P(same ans / same key) | P(same ans / diff key) | H(ans|key)/H(ans) | same (mixed) | P(same ans / same key) mixed | / diff key mixed |
|---|---|---|---|---|---|---|---|---|
| W4 | 2.77 | 0.546 | 0.810 | 0.748 | 0.745 | 0.485 | 0.594 | 0.580 |
| W8 | 3.45 | 0.436 | 0.824 | 0.750 | 0.636 | 0.366 | 0.603 | 0.578 |
| L1 | 4.16 | 0.352 | 0.831 | 0.755 | 0.520 | 0.277 | 0.598 | 0.582 |
| L1N | 2.05 | 0.703 | 0.817 | 0.698 | 0.803 | 0.612 | 0.608 | 0.554 |
| N2 | 1.68 | 0.785 | 0.811 | 0.674 | 0.851 | 0.709 | 0.603 | 0.546 |
| EQ | 2.99 | 0.510 | 0.871 | 0.690 | 0.536 | 0.421 | 0.700 | 0.504 |

| key | gap cases | some right att key not used by any winner-answer att (same seed) | ... (any att of 32 with winner answer) | right atts with key unique in seed |
|---|---|---|---|---|
| W4 | 94 | 41 (44%) | 27 (29%) | 34/163 |
| W8 | 94 | 53 (56%) | 44 (47%) | 52/163 |
| L1 | 94 | 66 (70%) | 61 (65%) | 77/163 |
| L1N | 94 | 31 (33%) | 20 (21%) | 21/163 |
| N2 | 94 | 23 (24%) | 11 (12%) | 19/163 |
| EQ | 94 | 52 (55%) | 39 (41%) | 46/163 |

| policy | vote | oracle@8 | per-att | distinct ans/8 | distinct key/8 |
|---|---|---|---|---|---|
| random 8 of 32 | 86.16 | 93.41 | 79.26 | 1.92 | 4.09 |
| W4-distinct | 86.42 | 94.09 | 79.54 | 1.93 | 4.26 |
| W8-distinct | 86.78 | 94.14 | 79.40 | 1.95 | 5.40 |
| L1-distinct | 86.43 | 93.78 | 79.59 | 1.94 | 6.20 |
| L1N-distinct | 86.55 | 93.34 | 78.95 | 1.97 | 2.92 |
| N2-distinct | 86.08 | 93.45 | 78.96 | 1.95 | 2.14 |
| EQ-distinct | 86.35 | 93.82 | 78.32 | 2.04 | 4.52 |
| W4-distinct if prev agree | 86.03 | 93.88 | 79.24 | 1.92 | 3.68 |
| L1-distinct if prev agree | 85.96 | 93.58 | 79.19 | 1.93 | 5.48 |
| EQ-distinct if prev agree | 86.08 | 93.62 | 78.80 | 1.96 | 3.83 |
| answer-distinct (bound) | 83.60 | 96.01 | 72.49 | 2.83 | 4.11 |
| EQ: avoid plurality-group keys if its mean lg < 0.0 | 86.34 | 93.36 | 78.94 | 1.95 | 3.18 |
| EQ: avoid plurality-group keys if its mean lg < 1.0 | 86.27 | 93.55 | 78.31 | 2.00 | 3.52 |
| EQ: avoid plurality-group keys if its mean lg < 2.0 | 86.30 | 93.65 | 78.07 | 2.02 | 3.91 |
| EQ: avoid plurality-group keys if its mean lg < 99.0 | 86.06 | 93.60 | 77.86 | 2.02 | 4.21 |
| L1: avoid plurality-group keys if its mean lg < 0.0 | 86.52 | 93.46 | 79.41 | 1.92 | 4.30 |
| L1: avoid plurality-group keys if its mean lg < 1.0 | 86.58 | 93.82 | 79.41 | 1.93 | 4.70 |
| L1: avoid plurality-group keys if its mean lg < 2.0 | 86.29 | 93.80 | 79.24 | 1.94 | 5.31 |
| L1: avoid plurality-group keys if its mean lg < 99.0 | 86.40 | 93.78 | 79.21 | 1.94 | 5.84 |
| W4: avoid plurality-group keys if its mean lg < 0.0 | 86.45 | 93.56 | 79.26 | 1.93 | 2.92 |
| W4: avoid plurality-group keys if its mean lg < 1.0 | 86.57 | 93.94 | 79.26 | 1.94 | 3.30 |
| W4: avoid plurality-group keys if its mean lg < 2.0 | 86.31 | 93.94 | 79.10 | 1.95 | 3.71 |
| W4: avoid plurality-group keys if its mean lg < 99.0 | 86.47 | 94.02 | 79.16 | 1.94 | 4.03 |
| actual seeds (serial order) | 85.92 | 93.75 | | | |
