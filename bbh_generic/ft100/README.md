# Sample-size control for the BBH tool fine-tunes (Teams A and B)

Question: Teams A and B fine-tuned the host on 12k-14.5k generated rows, so the host also learns the task itself (the
fine-tuned-without-tools arm is strong on BBH-style questions). Does learning to *call* a tool take few examples, while
learning the function itself takes thousands? Test: fine-tune on n in {32, 100, 300} of the team's own training rows
(n = 100 with two seeds) and evaluate each checkpoint with the tools attached and detached, on the team's fresh test,
its held-out out-of-style set, and real BBH. Details and the log: `NOTEBOOK.md`.

## What attaches how

Same tools and attach as in `../README.md` (compiled circuits, fixed tensors at inference). The hosts are full fine-tunes
of Qwen3-0.6B-Base with the teams' own scripts, unchanged, with the teams' hyperparameters; only the number of epochs was
set so the step counts match across teams (A: `teamA/ft.py`, lr 1e-5, tok_budget 16000, 10 epochs; B: `teamB/ft_B.py`,
lr 1e-5, tok_budget 24000, p_zero 0.2, 15 epochs). The fine-tuned checkpoints are not released.

## Results

Greedy, Qwen3-0.6B-Base fine-tunes; `acc% (call%)`, tools detached | attached. Fresh test and the held-out out-of-style
set are the teams' own generators; real BBH (`bbh`) is the official 3-shot score for that team's tasks. Source:
`scores_ft100.json` (per-arm `acc`, `call` and `per_task`; `scripts/pertask.py` prints it). The all-27-task block is
tabulated in `NOTEBOOK.md` (Results).

**Team A** (`track`, `order`; 6 tasks). Base: test 29.0, oos 18.9, real BBH 30.0 (0% call rate).

| host | test, no tools | test, + tools | oos2, no tools | oos2, + tools | real BBH, no tools | real BBH, + tools |
|---|---:|---:|---:|---:|---:|---:|
| n = 32 | 20.0 (100) | 50.8 (100) | 21.1 (97) | 28.3 (97) | 21.9 (100) | 59.5 (100) |
| n = 100, s0 | 19.7 (100) | 76.3 (100) | 21.1 (100) | 65.0 (100) | 23.0 (100) | 76.5 (100) |
| n = 100, s1 | 21.7 (100) | 73.7 (100) | 27.8 (100) | 64.4 (100) | 25.0 (100) | 70.3 (100) |
| n = 300 | 22.0 (100) | 93.2 (100) | 22.8 (100) | 66.1 (100) | 26.7 (100) | 94.4 (100) |
| ft_v3 (14.5k) | 57.0 (100) | 98.5 (100) | 44.4 (100) | 78.9 (100) | 61.4 (100) | 100.0 (100) |

**Team B** (`interval`, `walk`, `truth`; 3 tasks). Base: test 44.0, oos 11.3, real BBH 48.1 (0% call rate).

| host | test, no tools | test, + tools | oos, no tools | oos, + tools | real BBH, no tools | real BBH, + tools |
|---|---:|---:|---:|---:|---:|---:|
| n = 32 | 53.0 (100) | 78.3 (100) | 20.0 (99) | 23.3 (99) | 46.1 (100) | 86.7 (100) |
| n = 100, s0 | 67.3 (100) | 89.3 (100) | 23.3 (99) | 51.3 (99) | 62.1 (100) | 97.5 (100) |
| n = 100, s1 | 62.3 (100) | 89.3 (100) | 24.0 (97) | 64.7 (97) | 59.1 (100) | 94.9 (100) |
| n = 300 | 88.7 (100) | 94.3 (100) | 29.3 (99) | 74.7 (99) | 88.1 (100) | 98.0 (100) |
| ft3 (12k) | 96.0 (100) | 96.3 (100) | 26.7 (100) | 87.3 (100) | 95.7 (100) | 99.7 (100) |

Takeaway: the model learns to **call** at 32 rows (call rate 97-100% in every arm, base 0%); the tool arm gets most of
the full-fine-tune gain by ~100 rows; the no-tool arm stays near base (Team A) or improves only with many rows (Team B),
so what the extra thousands of rows buy is learning the function itself, not learning to call.

All 27 BBH tasks (single-team tool sets over-call on the other tasks; tabulated in `NOTEBOOK.md`, Results; run by
`scripts/run27.py` + `scripts/score27.py`): mean27 / 12-tool / 15-no-tool = base 40.37 / 43.03 / 38.23; frozen base +
Team C tools 42.95 / 49.63 / 37.61 (Team C's three tasks 65 → 92); Team A n100 + A tools 35.67 / 47.47 / 26.23; Team B
n100 + B tools 27.02 / 37.07 / 18.98.

## Files

* `sub/`: the exact training rows of every checkpoint (copied verbatim from the team's own seed-2 data by
  `scripts/subsample.py`): `A_n{32,100,300}_s0.jsonl`, `A_n100_s1.jsonl` (from `teamA/data/train3.jsonl`), and
  `B_n{32,100,300}_s0/`, `B_n100_s1/` (from Team B's ft3 rows, layout `data/{task}_train.jsonl` as `ft_B.py` reads it).
* `scripts/`: `subsample.py`, `make_eval_sets.py`, `make_jobs.py` (+ `add27.py`, `reorder.py`), `worker.py` (job queue),
  `run27.py` (all 27 BBH tasks, any team's tools), `score_ft100.py`, `score27.py`, `pertask.py` (aggregates only).
* `scores_ft100.json`: the aggregate scores behind the tables above (`results/score_main.txt`, `results/score27.txt` are
  the formatted readouts).

## Reproduce

```bash
# repo root
bash data/download_bbh.sh && python scripts/solve_veto.py --out runs/veto_qwen3.pt
(cd bbh_generic/teamB && python gens.py)                     # Team B data (byte-identical)
python bbh_generic/peek/scripts/make_A_jsonl.py data/bbh/data bbh_generic/peek/bbh_A.jsonl
# optional: re-draw the subsamples (the shipped sub/ files are the ones used)
xz -dk bbh_generic/teamA/data/train3.jsonl.xz
python bbh_generic/ft100/scripts/subsample.py A 100 0 bbh_generic/teamA/data/train3.jsonl /tmp/A_n100_s0.jsonl
python bbh_generic/ft100/scripts/subsample.py B 100 0 bbh_generic/teamB/data /tmp/B_n100_s0/data
# eval sets, job list, two workers on one GPU, scores
python bbh_generic/ft100/scripts/make_eval_sets.py
python bbh_generic/ft100/scripts/make_jobs.py && python bbh_generic/ft100/scripts/add27.py
TEAM_A_FT=/path/to/teamA_ft_v3 python bbh_generic/ft100/scripts/worker.py 0 w1 &   # reference row needs Team A's ft_v3 host
python bbh_generic/ft100/scripts/worker.py 0 w2 &
wait; python bbh_generic/ft100/scripts/score_ft100.py
```

Work files (checkpoints, outputs, logs, the job list) go to `bbh_generic/ft100/work/` (override with `BFT_R`). The base and
reference rows reuse the peek outputs (`PEEK_OUT`, default `bbh_generic/peek/out`) and Team B's own run outputs
(`TEAMB_OUT`, default `bbh_generic/teamB/out`).

Note: the frozen-base fresh-test row here (Team A, 29.0 mean) was re-run in this study at batch size 32 and differs from
Team A's own base run (28.5 at the default batch size 16; `teamA/results/test_base_plain.json`); bf16 runs with left-padded
batches depend on batch composition (see the top-level `DETAILS.md`, Caveats).
