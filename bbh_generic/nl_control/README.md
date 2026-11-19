# Natural-language fine-tune control for the BBH tool fine-tunes

Question: in the sample-size study (`../ft100/`), the "fine-tuned, tools off" arm trained on call strings with encoded tool
replies, so it is a weak control for "could plain fine-tuning learn the function". The fair control trains on the **same
questions** (the same `../ft100/sub/` rows, n = 32 / 100 / 300, n = 100 with two seeds) written as **plain step-by-step
worked solutions** in the official BBH CoT style, with **no tool calls**, same team scripts and hyperparameters, and is
evaluated with no tools.

How the rows are made: `code/make_nl.py` rewrites each row's completion deterministically from the reference parse that
the row already contains and the reference solver's step-by-step state (tracking: the full state after every swap;
deduction: a domain-propagation trace; temporal: an hour-by-hour sweep; navigate: position after every step; web of lies:
truth propagation). Every row is checked: the trace's own result equals the original answer (1,064 / 1,064 rows).
The rows are not shipped; `make_nl.py` regenerates them **byte-identically** (`NL_ROWS_SHA256.txt`, checked on CPU).

## Results (accuracy %, greedy; NL arm = tools none)

Cells: **NL** | call-FT tools off / call-FT tools on (the latter two from `../ft100/scores_ft100.json`).

Team A (`track`, `order`; fresh test 600 / held-out oos2 180 / real BBH 6 x 250). Base 29.0 / 18.9 / 30.0.

| n rows | fresh test | held-out oos2 | real BBH |
|---|---|---|---|
| 32 | **41.0** \| 20.0 / 50.8 | **38.3** \| 21.1 / 28.3 | **39.9** \| 21.9 / 59.5 |
| 100 s0 | **51.2** \| 19.7 / 76.3 | **37.2** \| 21.1 / 65.0 | **56.5** \| 23.0 / 76.5 |
| 100 s1 | **54.7** \| 21.7 / 73.7 | **47.8** \| 27.8 / 64.4 | **55.1** \| 25.0 / 70.3 |
| 300 | **70.5** \| 22.0 / 93.2 | **62.2** \| 22.8 / 66.1 | **72.6** \| 26.7 / 94.4 |

Team B (`interval`, `walk`, `truth`; fresh test 300 / held-out OOS 150 / real BBH 3 x 250). Base 44.0 / 11.3 / 48.1.

| n rows | fresh test | held-out OOS | real BBH |
|---|---|---|---|
| 32 | **61.3** \| 53.0 / 78.3 | **30.7** \| 20.0 / 23.3 | **65.7** \| 46.1 / 86.7 |
| 100 s0 | **84.7** \| 67.3 / 89.3 | **34.7** \| 23.3 / 51.3 | **89.2** \| 62.1 / 97.5 |
| 100 s1 | **77.0** \| 62.3 / 89.3 | **34.0** \| 24.0 / 64.7 | **82.1** \| 59.1 / 94.9 |
| 300 | **93.3** \| 88.7 / 94.3 | **44.0** \| 29.3 / 74.7 | **93.6** \| 88.1 / 98.0 |

Reading: plain worked solutions teach the function much better than call strings do (NL beats call-FT-tools-off
everywhere), but the tool arm stays ahead on real BBH at every n for Team A (+15 to +22 points; NL logical deduction stays
at 35-65) and for Team B up to n = 100; by n = 300 Team B's tasks are mostly learnable in plain language (93.6 vs 98.0).
Seed noise: the two n = 100 seeds differ by up to 10 points on the held-out sets (A oos2 37.2 vs 47.8) and 7 points on
real BBH (B 89.2 vs 82.1). Caveat: NL completions are longer (slightly more optimizer steps: A n100 70 vs 60) and NL is
decoded with a larger budget (A 1000 / B 700 new tokens); re-scored at the original 600 / 400 by truncation, A loses 0-8
points and B nothing (`results/scores_nl.json`, `acc_at_600` / `acc_at_400`). The run log is `NOTEBOOK.md`.

## Files

* `code/make_nl.py` — NL rows from the ft100 subsamples (CPU, deterministic).
* `code/ft_A_chunked.py`, `code/ft_B_chunked.py` — copies of `../teamA/ft.py` and `../teamB/ft_B.py` with one memory-only
  change (cross-entropy in checkpointed 1024-token chunks; same mean loss). The longer NL targets ran out of memory on a
  shared GPU otherwise.
* `code/make_jobs_nl.py` — the 16-job queue (train + eval per arm), run with `../ft100/scripts/worker.py`.
* `code/score_nl.py` — scores as `../ft100/scripts/score_ft100.py`, plus the truncated-budget re-score (needs the full
  outputs).
* `results/scores_nl.json` — the aggregate scores above; `results/{A,B}_nl_*.json.gz` — reduced per-question records
  (no text; `../reduce_bbh.py`).

## Reproduce

```bash
# repo root; first the ft100 prerequisites (bbh_generic/ft100/README.md: BBH data, Team B data, make_eval_sets.py)
export BFT_R=bbh_generic/nl_control/work FT100_R=bbh_generic/ft100/work
F=bbh_generic/ft100/sub; N=$BFT_R/nl; mkdir -p $N
for x in A_n32_s0 A_n100_s0 A_n100_s1 A_n300_s0; do python bbh_generic/nl_control/code/make_nl.py A $F/$x.jsonl $N/$x.jsonl; done
for x in B_n32_s0 B_n100_s0 B_n100_s1 B_n300_s0; do mkdir -p $N/$x/data; python bbh_generic/nl_control/code/make_nl.py B $F/$x $N/$x; done
(cd $N && shasum -a 256 -c ../../NL_ROWS_SHA256.txt)                  # byte-identical to the rows used
python bbh_generic/nl_control/code/make_jobs_nl.py
python bbh_generic/ft100/scripts/worker.py 0 w1                       # 8 fine-tunes + 8 evals; one H100 ran them in ~6 h (shared)
python bbh_generic/nl_control/code/score_nl.py                        # -> $BFT_R/out/scores_nl.json
```

The fine-tuned checkpoints are not released.
