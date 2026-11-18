# bbh_ft100: sample-size control for the BBH tool fine-tunes (Teams A and B)

Question: Teams A and B fine-tuned Qwen3-0.6B-Base on 12k-14.5k generated rows, so the host also learns the task itself
(the FT-no-tools arm is strong). Does learning to CALL a tool take few examples while learning f itself takes thousands?
Test: fine-tune on n in {32, 100, 300} of the team's own training rows; evaluate each checkpoint with tools attached and
detached.

Runs: one H100 (GPU 1, two workers, scripts/worker.py). This folder: scripts/, sub/ (the subsampled training rows), NOTEBOOK.md, scores.
No team code, data, tools or checkpoints were modified; everything used was copied:
- micromachines/ and bbh_generic/teamA as used by the real-BBH peek (Team A .py files identical to the repo copy except
  make_data.py, which lacked the later train100 entry; train3/test/oos2 byte-identical across machines).
- bbh_generic/teamB code + data (.py identical to the peek copy).
- BBH task files and peek/bbh_A.jsonl as in the peek. Team A ft_v3 checkpoint used read-only (reference row).
The team agents were told nothing. No individual real-BBH output was read (scripts/score_ft100.py prints aggregates only).

## Setup
- Subsample (scripts/subsample.py, seed 0; n=100 also seed 1): rows copied verbatim. Split evenly over tools, then evenly
  over that tool's styles, uniform inside each cell.
  - A (train3.jsonl, 14,500 rows): cells (track: bbh/alt/mix/code; order: bbh/alt/mix). n=32 -> track 4/4/4/4, order 5/5/6.
  - B (ft3 rows, 12,000): interval={temporal, lab}, walk={navigate, hiker}, truth={lies, rumor}; n/6 each. Error-recovery
    rows kept where sampled (n=32: 4, n=100 s0: 13, n=300: 42).
- Fine-tune: the team's own script, unchanged, with the team's hyperparameters (A ft.py: lr 1e-5, tok_budget 16000 as in
  ft_v3; B ft_B.py: lr 1e-5, tok_budget 24000, p_zero 0.2, as ft3). Only epochs changed.
  Epochs: A 10, B 15 (same for every n within a team). Reason: both teams use large token-budget batches (A ~17-20 rows,
  B ~25-34 rows per step), so 3-5 epochs would give only 5-15 optimizer steps at n=32. With A 10 / B 15 the step counts
  match across teams: n=32 -> 30 steps, n=100 -> 60 steps, n=300 -> A 170 / B 135 steps.
  n=100 seed 1 uses subsample seed 1 and ft seed 1.
- Eval (each checkpoint, tools attached and detached; detached = the host must write the tool response itself):
  - A: run_eval.py (official 3-shot CoT prompt, greedy, max_new 600, bs 32) on one file = fresh test (seed 1, 600) +
    held-out oos2 (180) + real BBH (bbh_A.jsonl, 6 tasks x 250).
  - B: run_eval.py (bs 32, max_new 400) on temporal/navigate/lies_test (seed 1, 100 each), meeting/robot/knights_oos
    (seed 3, 50 each, zero-shot), and real BBH copied to data/{task}_bbh.jsonl (same as peek/scripts/peek_B.py).
  - Real BBH scored with the official norm of peek/scripts/score.py; generated sets with the team's harness 'ok'.
  - Base: B reuses teamB plain_{test,oos} outputs and peek B_base; A reuses peek A_base for real BBH and
    runs the base on test+oos2 here. References: B ft3 (teamB results + peek); A ft_v3 (peek BBH; test+oos2 run here).

## Log
- 2026-11-17 20:23 workers started on GPU 1. Training is fast (30 s - 4 min per checkpoint). Actual optimizer steps:
  A n32 30, n100 60, n300 170; B n32 30, n100 60, n300 120. Final train losses ~0.001-0.005 (sets are memorised).
- Bug caught in my scorer (fixed 23:05, before any conclusion): BBH task names 'tracking_shuffled_*' start with 'track', so
  the first Team-A table put the BBH tracking tasks into the fresh-test bucket. Now test = exactly track3/5/7, order3/5/7.
- 20:45 coordinator request: all 27 BBH tasks for (1) frozen base + Team C tools, (2) A n=100 + A tools, (3) B n=100 + B tools.
  scripts/run27.py = copy of Team C's run_gen.py loop (= run_bbh.py) with a generic tool attach; data/cot = the repo's
  pinned official BBH copy (data/download_bbh.sh; the 12 files shared with the peek are byte-identical). Base+C runs on
  a shared A40 and, in reverse task order (scripts/reorder.py), on the H100; merged from the per-task log lines
  (which use the same official norm). Base reference = the repo's all-27-task HF-loop base run (mean 40.37, read-only).
  The own-harness base-no-tools 27-task run was dropped for time (the repo base run is the reference).
- Setup slip: code/micromachines was first a symlink to the repo's micromachines/ (identical files, diff -r clean);
  replaced by a real copy at 20:47.

## Results

Main sample-size study (Teams A and B, `acc% (call%)`, tools off | on), from `scores_ft100.json`:

Team A (6 tasks) base test 29.0, oos 18.9, real BBH 30.0 (0% call):
| host | test off | test on | oos2 off | oos2 on | bbh off | bbh on |
|---|---|---|---|---|---|---|
| n32    | 20.0 (100) | 50.8 (100) | 21.1 (97)  | 28.3 (97)  | 21.9 (100) | 59.5 (100) |
| n100s0 | 19.7 (100) | 76.3 (100) | 21.1 (100) | 65.0 (100) | 23.0 (100) | 76.5 (100) |
| n100s1 | 21.7 (100) | 73.7 (100) | 27.8 (100) | 64.4 (100) | 25.0 (100) | 70.3 (100) |
| n300   | 22.0 (100) | 93.2 (100) | 22.8 (100) | 66.1 (100) | 26.7 (100) | 94.4 (100) |
| ftv3 14.5k | 57.0 (100) | 98.5 (100) | 44.4 (100) | 78.9 (100) | 61.4 (100) | 100.0 (100) |

Team B (3 tasks) base test 44.0, oos 11.3, real BBH 48.1 (0% call):
| host | test off | test on | oos off | oos on | bbh off | bbh on |
|---|---|---|---|---|---|---|
| n32    | 53.0 (100) | 78.3 (100) | 20.0 (99)  | 23.3 (99)  | 46.1 (100) | 86.7 (100) |
| n100s0 | 67.3 (100) | 89.3 (100) | 23.3 (99)  | 51.3 (99)  | 62.1 (100) | 97.5 (100) |
| n100s1 | 62.3 (100) | 89.3 (100) | 24.0 (97)  | 64.7 (97)  | 59.1 (100) | 94.9 (100) |
| n300   | 88.7 (100) | 94.3 (100) | 29.3 (99)  | 74.7 (99)  | 88.1 (100) | 98.0 (100) |
| ft3 12k | 96.0 (100) | 96.3 (100) | 26.7 (100) | 87.3 (100) | 95.7 (100) | 99.7 (100) |

All 27 BBH tasks (mean27 / 12-tool / 15-no-tool), one run per arm (official prompts and scoring; a single team's tools
over-call on the other tasks):
- base (repo all-27 HF-loop run, the reference): 40.37 / 43.03 / 38.23
- frozen base + Team C tools: 42.95 / 49.63 / 37.61 (Team C's 3 tasks 65 -> 92)
- Team A n100 + A tools: 35.67 / 47.47 / 26.23
- Team B n100 + B tools: 27.02 / 37.07 / 18.98
- per-team task means (base / +C / A n100+A / B n100+B): A-tasks 28.60/29.13/76.53/7.80; B-tasks 49.87/48.13/19.87/97.47;
  C-tasks 65.07/92.13/16.93/35.20.

Call rate: 97-100% in every fine-tune arm already at n=32 (base 0%).

Takeaway: calling is learned at ~32 rows; the tool arm reaches most of the full-fine-tune gain at ~100 rows; the no-tool
arm stays near base (A) or climbs only with many rows (B).
