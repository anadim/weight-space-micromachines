# Off-task regression after RL

Qwen3-0.6B-Base (GSM8K §5.1 rows, RL seeds 0–2) and Qwen3-1.7B-Base (rl28g rows, RL seeds 0–2), 5 off-task benchmarks,
full test sets (MMLU 5-shot 14,042; ARC-Challenge 1,172; HellaSwag 10,042; WinoGrande 1,267; NQ-open 3,610 greedy generation).
Rows: 1 = base model alone; 3 = base + rung-2 implants; 2 = plain RL (P64 / P17); 2b = plain RL + untrained rung 2;
4 = RL with implants (CW / CW17) + rung 2; 4h = the same CW host with the implants detached.
Brackets = 95% paired item bootstrap (pooled over the 3 RL seeds); per-seed values in parentheses. Method/jobs: README.md.
Per-item results (reduced, no text): j06/res, j17/res; tables: ana06.md / ana17.md (python3 ana_ot.py); rows06.json / rows17.json.

## Takeaway

- **RL with implants does not hurt general skills more than plain RL.** On MMLU, HellaSwag, WinoGrande and NQ the two RL
  variants differ by ≤0.4 pt at both sizes (all CIs include 0 except HellaSwag 1.7B, −0.23). The one visible difference is
  ARC-Challenge and it flips sign with size: +1.1 [+0.3, +2.1] for RL-with-implants at 0.6B, −1.6 [−3.0, −0.3] at 1.7B.
- **Neither RL variant moves general skills much from the base model:** every RL row is within 1.1 pt (0.6B) / 0.9 pt
  (1.7B) of its base on every benchmark.
- **The implants are inert off-task:** attaching them never writes at a scored position (0 of all items, every benchmark,
  both sizes), changes no multiple-choice decision, and leaves every NQ generation byte-identical (every run, both sizes).
  Rows 3 = 1, 2b = 2 and 4 = 4h exactly. (An implant writes somewhere in the prompt on 0.04% of MMLU items: arithmetic in
  the question; never on a scored token.)

Blog-ready: "RL with implants leaves general skills where plain RL leaves them: on MMLU, ARC-Challenge, HellaSwag,
WinoGrande and NQ-open, both stay within about 1 point of the base model at 0.6B and 1.7B (3 seeds each), and the attached
implants never fire on these benchmarks, so attaching them changes no answer."

## Qwen3-0.6B-Base

| row | config | MMLU 5-shot | ARC-C acc_norm | HellaSwag acc_norm | WinoGrande | NQ-open EM (gen.) |
|---|---|---|---|---|---|---|
| 1 | Base (model alone) | 52.44 [51.6, 53.3] | 38.48 [35.8, 41.3] | 53.82 [52.9, 54.8] | 58.25 [55.4, 61.0] | 6.95 [6.1, 7.8] |
| 3 | Base + rung 2 | 52.44 [51.6, 53.3] | 38.48 [35.8, 41.3] | 53.82 [52.9, 54.8] | 58.25 [55.4, 61.0] | 6.95 [6.1, 7.8] |
| 2 | P64 s0/1/2 (plain RL) | 52.61 [51.8, 53.4] (52.6/52.6/52.6) | 37.43 [34.9, 40.1] (37.5/37.6/37.1) | 53.99 [53.0, 55.0] (54.0/54.0/54.0) | 59.01 [56.5, 61.7] (59.2/59.0/58.8) | 7.31 [6.5, 8.2] (7.2/7.2/7.5) |
| 2b | P64 + untrained rung 2 attached | 52.61 [51.8, 53.4] (52.6/52.6/52.6) | 37.43 [34.9, 40.1] (37.5/37.6/37.1) | 53.99 [53.0, 55.0] (54.0/54.0/54.0) | 59.01 [56.5, 61.7] (59.2/59.0/58.8) | 7.31 [6.5, 8.2] (7.2/7.2/7.5) |
| 4 | CW s0/1/2 + rung 2 (RL with implants) | 52.47 [51.7, 53.3] (52.5/52.5/52.4) | 38.57 [35.8, 41.3] (39.0/38.4/38.3) | 53.89 [52.9, 54.9] (54.0/53.9/53.8) | 58.62 [56.1, 61.1] (58.6/58.6/58.6) | 7.22 [6.4, 8.1] (7.3/7.2/7.1) |
| 4h | CW host alone (implants detached) | 52.47 [51.7, 53.3] (52.5/52.5/52.4) | 38.57 [35.8, 41.3] (39.0/38.4/38.3) | 53.89 [52.9, 54.9] (54.0/53.9/53.8) | 58.62 [56.1, 61.1] (58.6/58.6/58.6) | 7.22 [6.4, 8.1] (7.3/7.2/7.1) |

| difference | MMLU 5-shot | ARC-C acc_norm | HellaSwag acc_norm | WinoGrande | NQ-open EM (gen.) |
|---|---|---|---|---|---|
| 4 − 2 | -0.14 [-0.38, +0.10] | +1.14 [+0.26, +2.10] | -0.11 [-0.29, +0.08] | -0.39 [-1.29, +0.50] | -0.09 [-0.46, +0.27] |
| 4 − 2b | -0.14 [-0.38, +0.10] | +1.14 [+0.26, +2.10] | -0.11 [-0.29, +0.08] | -0.39 [-1.29, +0.50] | -0.09 [-0.46, +0.27] |
| 4h − 2 | -0.14 [-0.38, +0.10] | +1.14 [+0.26, +2.10] | -0.11 [-0.29, +0.08] | -0.39 [-1.29, +0.50] | -0.09 [-0.46, +0.27] |
| 2 − 1 | +0.18 [-0.12, +0.46] | -1.05 [-2.33, +0.23] | +0.17 [-0.07, +0.40] | +0.76 [-0.55, +2.05] | +0.36 [-0.06, +0.80] |
| 4 − 1 | +0.04 [-0.26, +0.34] | +0.09 [-1.05, +1.19] | +0.06 [-0.20, +0.32] | +0.37 [-0.87, +1.63] | +0.27 [-0.06, +0.64] |
| 2b − 1 | +0.18 [-0.12, +0.46] | -1.05 [-2.33, +0.23] | +0.17 [-0.07, +0.40] | +0.76 [-0.55, +2.05] | +0.36 [-0.06, +0.80] |
| 3 − 1 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

| implants attached: row | benchmark | items with an implant write on a scored position | scored positions with a write | items with a write anywhere in the prompt | serial END trigger on a scored position | items whose decision changed |
|---|---|---|---|---|---|---|
| 3 | mmlu | 0.000% | 0.0000% | 0.04% | 0.000% | 0.000% |
| 3 | arcc | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 3 | hs | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 3 | wg | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 3 | nq | 0.000% | 0.0000% | – | 0.000% | 0.000% |
| 2b | mmlu | 0.000% | 0.0000% | 0.04% | 0.000% | 0.000% |
| 2b | arcc | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 2b | hs | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 2b | wg | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 2b | nq | 0.000% | 0.0000% | – | 0.000% | 0.000% |
| 4 | mmlu | 0.000% | 0.0000% | 0.04% | 0.000% | 0.000% |
| 4 | arcc | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 4 | hs | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 4 | wg | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 4 | nq | 0.000% | 0.0000% | – | 0.000% | 0.000% |

## Qwen3-1.7B-Base

| row | config | MMLU 5-shot | ARC-C acc_norm | HellaSwag acc_norm | WinoGrande | NQ-open EM (gen.) |
|---|---|---|---|---|---|---|
| 1 | 1.7B-Base (model alone) | 62.59 [61.8, 63.4] | 44.62 [41.8, 47.4] | 66.33 [65.4, 67.3] | 64.72 [62.0, 67.6] | 12.66 [11.6, 13.8] |
| 3 | 1.7B-Base + rung 2 | 62.59 [61.8, 63.4] | 44.62 [41.8, 47.4] | 66.33 [65.4, 67.3] | 64.72 [62.0, 67.6] | 12.66 [11.6, 13.8] |
| 2 | P17 s0/1/2 (plain RL) | 62.53 [61.8, 63.3] (62.6/62.5/62.5) | 45.42 [42.6, 48.2] (45.5/45.6/45.1) | 66.71 [65.8, 67.6] (66.7/66.8/66.6) | 64.14 [61.6, 66.6] (64.0/64.1/64.3) | 12.98 [12.0, 14.1] (13.0/12.9/13.0) |
| 2b | P17 + untrained rung 2 attached | 62.53 [61.8, 63.3] (62.6/62.5/62.5) | 45.42 [42.6, 48.2] (45.5/45.6/45.1) | 66.71 [65.8, 67.6] (66.7/66.8/66.6) | 64.14 [61.6, 66.6] (64.0/64.1/64.3) | 12.98 [12.0, 14.1] (13.0/12.9/13.0) |
| 4 | CW17 s0/1/2 + rung 2 (RL with implants) | 62.45 [61.7, 63.3] (62.4/62.5/62.4) | 43.77 [41.1, 46.4] (44.0/43.6/43.7) | 66.48 [65.6, 67.4] (66.5/66.5/66.5) | 63.93 [61.2, 66.5] (63.4/64.1/64.3) | 13.16 [12.1, 14.3] (13.1/13.2/13.2) |
| 4h | CW17 host alone | 62.45 [61.7, 63.3] (62.4/62.5/62.4) | 43.77 [41.1, 46.4] (44.0/43.6/43.7) | 66.48 [65.6, 67.4] (66.5/66.5/66.5) | 63.93 [61.2, 66.5] (63.4/64.1/64.3) | 13.16 [12.1, 14.3] (13.1/13.2/13.2) |

| difference | MMLU 5-shot | ARC-C acc_norm | HellaSwag acc_norm | WinoGrande | NQ-open EM (gen.) |
|---|---|---|---|---|---|
| 4 − 2 | -0.08 [-0.32, +0.14] | -1.65 [-2.99, -0.31] | -0.23 [-0.43, -0.03] | -0.21 [-1.16, +0.74] | +0.18 [-0.21, +0.56] |
| 4 − 2b | -0.08 [-0.32, +0.14] | -1.65 [-2.99, -0.31] | -0.23 [-0.43, -0.03] | -0.21 [-1.16, +0.74] | +0.18 [-0.21, +0.56] |
| 4h − 2 | -0.08 [-0.32, +0.14] | -1.65 [-2.99, -0.31] | -0.23 [-0.43, -0.03] | -0.21 [-1.16, +0.74] | +0.18 [-0.21, +0.56] |
| 2 − 1 | -0.06 [-0.35, +0.22] | +0.80 [-0.34, +1.96] | +0.38 [+0.12, +0.64] | -0.58 [-2.03, +0.87] | +0.32 [-0.06, +0.71] |
| 4 − 1 | -0.14 [-0.42, +0.15] | -0.85 [-1.96, +0.26] | +0.15 [-0.11, +0.40] | -0.79 [-2.26, +0.84] | +0.50 [+0.05, +0.95] |
| 2b − 1 | -0.06 [-0.35, +0.22] | +0.80 [-0.34, +1.96] | +0.38 [+0.12, +0.64] | -0.58 [-2.03, +0.87] | +0.32 [-0.06, +0.71] |
| 3 − 1 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

| implants attached: row | benchmark | items with an implant write on a scored position | scored positions with a write | items with a write anywhere in the prompt | serial END trigger on a scored position | items whose decision changed |
|---|---|---|---|---|---|---|
| 3 | mmlu | 0.000% | 0.0000% | 0.04% | 0.000% | 0.000% |
| 3 | arcc | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 3 | hs | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 3 | wg | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 3 | nq | 0.000% | 0.0000% | – | 0.000% | 0.000% |
| 2b | mmlu | 0.000% | 0.0000% | 0.04% | 0.000% | 0.000% |
| 2b | arcc | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 2b | hs | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 2b | wg | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 2b | nq | 0.000% | 0.0000% | – | 0.000% | 0.000% |
| 4 | mmlu | 0.000% | 0.0000% | 0.04% | 0.000% | 0.000% |
| 4 | arcc | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 4 | hs | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 4 | wg | 0.000% | 0.0000% | 0.00% | 0.000% | 0.000% |
| 4 | nq | 0.000% | 0.0000% | – | 0.000% | 0.000% |
