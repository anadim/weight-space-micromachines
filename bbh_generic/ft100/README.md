# Sample-size control for the BBH tool fine-tunes (Teams A and B)

> **RESULTS PENDING.** This experiment was still running when the repository was prepared. `scores_ft100.json` is a
> partial snapshot of the aggregate scores (written by `scripts/score_ft100.py` while jobs were still finishing; the
> all-27-task runs and the notebook's Results section are not done). Do not cite numbers from it yet.

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

## Files

* `sub/`: the exact training rows of every checkpoint (copied verbatim from the team's own seed-2 data by
  `scripts/subsample.py`): `A_n{32,100,300}_s0.jsonl`, `A_n100_s1.jsonl` (from `teamA/data/train3.jsonl`), and
  `B_n{32,100,300}_s0/`, `B_n100_s1/` (from Team B's ft3 rows, layout `data/{task}_train.jsonl` as `ft_B.py` reads it).
* `scripts/`: `subsample.py`, `make_eval_sets.py`, `make_jobs.py` (+ `add27.py`, `reorder.py`), `worker.py` (job queue),
  `run27.py` (all 27 BBH tasks, any team's tools), `score_ft100.py`, `score27.py`, `pertask.py` (aggregates only).
* `scores_ft100.json`: partial aggregate scores (see the note above).

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
batches depend on batch composition (see the top-level README, Caveats).
