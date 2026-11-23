| pair | LIB GSM8K / Hard (calc-only) | LIBscr GSM8K / Hard (calc-only) | LIB − LIBscr | LIBscr test calls/q; train calls/q u1 → u25 → u300; dev calc fire u300 |
|---|---|---|---|---|
| s0 (H100) | 69.37 / 56.10 (67.85 / 53.37) | 69.29 / 51.86 (69.29 / 51.86) | +0.08 / +4.25 | 2.19; 1.65 → 0.53 → 1.77; 99% |
| s1 (B200) | 70.36 / 56.18 (69.37 / 53.75) | 65.50 / 41.62 (65.50 / 41.55) | +4.85 / +14.56 | 0.00; 1.60 → 0.42 → 0.05; 0% |
| s2 (B200) | 71.34 / 55.19 (70.05 / 53.75) | 66.72 / 51.33 (66.72 / 51.33) | +4.62 / +3.87 | 2.42; 1.40 → 0.33 → 2.49; 100% |
| mean (3) | 70.36 / 55.83 | 67.17 / 48.27 | +3.18 / +7.56 | |

paired item bootstrap, LIB − LIBscr (pp, 95% CI):
test_full s1 +4.85 [+2.20, +7.35]
test_full s2 +4.62 [+2.27, +6.97]
test_full pooled s1+s2 +4.74 [+2.92, +6.60]
hard_full s1 +14.56 [+11.75, +16.98]
hard_full s2 +3.87 [+1.36, +6.22]
hard_full pooled s1+s2 +9.21 [+7.43, +11.03]

big numbers (dev u300, scaled half), implants on → all off:
- LIB s0 (H100): 63.50 → 37.75
- LIB s1 (A40, original): 64.50 → 36.00
- LIB s1 (B200): 65.50 → 37.25
- LIB s2 (B200): 64.75 → 38.50
- mean (4 runs): 64.56 → 37.38
