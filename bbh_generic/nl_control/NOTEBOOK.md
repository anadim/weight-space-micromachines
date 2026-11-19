# BBH natural-language fine-tune control: run log

## Job 1: BBH natural-language fine-tune control (DONE 09:35 UTC 11-18)

Question: the ft100 "FT, tools off" arm trained on call strings + encoded replies. Fair control = same questions, same recipe,
but the completion is a natural-language worked solution (no calls). Does the NL arm learn f from ~100 rows?

### Data (code/make_nl.py -> work/nl/)
Same rows as ../ft100/sub/{A,B}_n{32,100,300}_s{0,1} (subsample.py output, seed 0; n=100 also seed 1). Prompt/question kept;
completion replaced by a step-by-step solution written from the reference parse already in each row (slot names, values, op
codes, clue codes, reference call) and the reference solver's state, in the official BBH CoT format of each task:
- track: "(0) At the start: Pedro: bassoon, ..." then the full state after every operation (the original sentence as the
  step label), "At the end, Freya has violin. So the answer is (B)."
- order (logical deduction): numbered clues, then a domain-propagation trace in prose ("From (1): X can be at positions 2, 3",
  "Position 1 is taken by Y, so ...", "Only Z can be at position 4 ...", case splits "Suppose X is at position 2. That breaks
  (3)." when needed), then "Combining these, we get the following ordering: ..." and the answer.
- interval (temporal/lab): window, busy list, hour-by-hour free/busy sweep, free span, original conclusion sentence.
- walk (navigate/hiker): "(k) Go 6 steps forward: (0, 13), facing north." per command, original conclusion.
- truth (web of lies/rumor): BBH-style propagation "(2) Anouk says Xiu lies. We know Xiu lies, so Anouk is right: ...".
Every row is checked: the NL trace's own result equals the reference tool's result and the original answer (all 1,064 rows
pass; no tool tokens). Error-recovery rows (B: masked broken first call) become plain NL rows (prefix dropped).
NL completions are longer: A track median 659 chars (orig 536), order 974 (463); B 290-540 (orig 190-310).

### Recipe (unchanged team scripts, copied): A ft.py lr 1e-5, tok_budget 16000, 10 epochs, seed s; B ft_B.py lr 1e-5,
tok_budget 24000, p_zero 0.2, 15 epochs. One memory-only change in the copies: the CE loss is computed in checkpointed
1024-token chunks (same mean loss; the longer NL targets OOM'd on the shared GPU). Because batches are token-budgeted, longer
NL rows give slightly more optimizer steps (A n100: 70 vs 60).
Eval: the teams' run_eval.py, tools none, same files as ft100 (A: evalA_ft.jsonl = fresh 600 + oos2 180 + real BBH 1500;
B: temporal/navigate/lies test 300, meeting/robot/knights oos 150, real BBH 750), scored with the same scorer (official BBH
norm on real BBH, harness 'ok' on own sets). max_new raised to A 1000 / B 700 because NL traces are longer; also re-scored
at the original 600 / 400 by truncating each greedy output (code/score_nl.py).

### Log
- 03:51 first launch OOM (shared GPU, long NL targets) -> chunked CE; relaunched 03:55 on an H100 GPU (one worker process).
- 04:30 PARTIAL B n100 s0 (NL, tools none): fresh 84.7 (temporal 83, navigate 73, lies 98) | held-out oos 34.7 (meeting 2,
  robot 14, knights 88) | real BBH 89.2 (temporal 84.4, navigate 84.4, web_of_lies 98.8). Same at the 400-token budget.
  Compare ft100 B n100 s0: call-FT tools off 67.3 / 23.3 / 62.1; call-FT + tools 89.3 / 51.3 / 97.5; base 44.0 / 11.3 / 48.1.
  => for Team B, 100 NL worked solutions already learn most of the function (BBH 89 vs 97.5 with tools).
- 05:30 PARTIAL A n100 s1 (NL): fresh 54.7 | oos2 47.8 | real BBH 55.1 (track 74/70/67, deduction 56/33/31); at the 600 budget
  51.8 / 45.0 / 51.5 (10% of outputs exceed 600 tokens). ft100 A n100 s1: call-FT tools off 21.7 / 27.8 / 25.0; call-FT +
  tools 73.7 / 64.4 / 70.3; base 29.0 / 18.9 / 30.0. => NL arm closes ~half of the base->tools gap on A.
- 05:38 PARTIAL A n100 s0 (NL): fresh 51.2 | oos2 37.2 | BBH 56.5 (at 600 tok: 49.8 / 33.3 / 54.7). ft100 A n100 s0: tools off
  19.7 / 21.1 / 23.0; + tools 76.3 / 65.0 / 76.5.
- 06:23 evalA_n32_s0 OOM (shared GPU0) -> requeued (lock removed; log kept as logs/evalA_n32_s0.OOM1.log). 08:08 trainB_n300_s0
  had OOM'd at model load at 06:01 -> requeued the same way (logs/trainB_n300_s0.OOM1.log). Both then ran clean.
- 09:35 ALL 8 NL jobs DONE (A n32/n100 s0,s1/n300; B n32/n100 s0,s1/n300). score_nl.py run; results copied to
  results/ (reduced; scores_nl.json).

### RESULT: NL fine-tune control vs call-FT (accuracy %, tools none for NL)
Cells: NL (NL scored at the original 600/400-token budget) | call-FT tools off / call-FT tools on (ft100 notebook).
NL arm tool-call rate = 0.0% everywhere. NL main numbers use max_new A 1000 / B 700.

Team A (track + order; fresh 600 / held-out oos2 180 / real BBH 6 x 250)
| n rows | fresh test | held-out oos2 | real BBH |
|---|---|---|---|
| 0 (frozen base) | 29.0 | 18.9 | 30.0 |
| 32 | **41.0** (39.8) \| 20.0 / 50.8 | **38.3** (37.2) \| 21.1 / 28.3 | **39.9** (39.5) \| 21.9 / 59.5 |
| 100 s0 | **51.2** (49.8) \| 19.7 / 76.3 | **37.2** (33.3) \| 21.1 / 65.0 | **56.5** (54.7) \| 23.0 / 76.5 |
| 100 s1 | **54.7** (51.8) \| 21.7 / 73.7 | **47.8** (45.0) \| 27.8 / 64.4 | **55.1** (51.5) \| 25.0 / 70.3 |
| 300 | **70.5** (65.8) \| 22.0 / 93.2 | **62.2** (53.9) \| 22.8 / 66.1 | **72.6** (68.1) \| 26.7 / 94.4 |
Share of NL outputs over 600 tokens: n32 10/8/6%, n100 7-10/13/8-10%, n300 12/18/14% (test/oos2/BBH).
NL real BBH per task: n32 track 53.2/50.4/43.2, deduction 48.4/26.0/18.4; n100 s0 79.6/75.2/72.4, 50.0/36.4/25.2;
n100 s1 74.4/69.6/66.8, 56.0/32.8/31.2; n300 98.8/98.0/94.8, 65.2/44.0/34.8.

Team B (interval / walk / truth; fresh 300 / held-out OOS 150 / real BBH 3 x 250)
| n rows | fresh test | held-out OOS | real BBH |
|---|---|---|---|
| 0 (frozen base) | 44.0 | 11.3 | 48.1 |
| 32 | **61.3** \| 53.0 / 78.3 | **30.7** \| 20.0 / 23.3 | **65.7** \| 46.1 / 86.7 |
| 100 s0 | **84.7** \| 67.3 / 89.3 | **34.7** \| 23.3 / 51.3 | **89.2** \| 62.1 / 97.5 |
| 100 s1 | **77.0** \| 62.3 / 89.3 | **34.0** \| 24.0 / 64.7 | **82.1** \| 59.1 / 94.9 |
| 300 | **93.3** \| 88.7 / 94.3 | **44.0** \| 29.3 / 74.7 | **93.6** \| 88.1 / 98.0 |
(B NL identical at the 400-token budget; <=0.4% of BBH outputs exceed 400, 8.7% on oos for n100 s0.)
NL real BBH per task: n32 temporal 35.6, navigate 72.8, lies 88.8; n100 s0 84.4/84.4/98.8; n100 s1 60.4/86.0/100.0;
n300 96.8/85.2/98.8.

Reading:
- The NL control beats call-FT-tools-off at every n, on every set, both teams (A BBH +18.0 to +45.9; B BBH +19.6 to +27.1 at n<=100, +5.5 at n300).
  So the "FT tools off is at/below base" result is a property of training on call strings, not of 100 rows being too few.
- Real BBH, tools on vs NL: A tool arm leads by +19.6 (n32), +20.0/+15.2 (n100), +21.8 (n300); the gap does not close
  with n because NL deduction stays at 35-65 while tools reach 86-91 at n300 (NL tracking reaches 95-99 at n300).
  B tool arm leads by +21.0 (n32), +8.3/+12.8 (n100), +4.4 (n300): B's tasks are mostly learnable in NL by 100-300 rows;
  NL navigate (dead reckoning) is 85 at n100-300 vs 94-98 with the walk tool.
- Held-out styles: B OOS NL 31-44 vs tools 23-75 (tools win from n100: 51/65/75); A oos2 NL 38-62 vs tools 28-66
  (NL wins at n32, roughly ties at n300: 62.2 vs 66.1).
- Caveat: NL completions are longer (more optimizer steps, A n100 70 vs 60) and NL gets a larger decode budget; the
  original-budget column shows the budget is worth 0-8 points for A and nothing for B. One seed except n100.
