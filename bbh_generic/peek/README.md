# Real BBH, run once (the "peek")

After the judges passed the tools, a separate agent ran the 12 algorithmic BBH tasks once (n = 250 per task, official 3-shot
CoT prompts, greedy) with each team's frozen configuration. Only aggregate scores were read; no per-question output was
inspected. This folder has the scoring scripts and the aggregate scores; the per-question outputs (which contain BBH
questions) are not included.

| arm | host | tools |
|---|---|---|
| `A_base` / `A_ftv3_notools` / `A_ftv3_tools` | Qwen3-0.6B-Base / Team A ft_v3 (14,500 rows, 2 epochs) | none / none / `track_gen,order_gen` |
| `B_base` / `B_ft3_notools` / `B_ft3_tools` | Qwen3-0.6B-Base / Team B ft3 (12,000 rows, 1 epoch) | none / none / `interval_gen,turtle_gen,truth_gen` |
| `C_base` / `C_tools` | Qwen3-0.6B-Base (frozen) | none / `expr_gen,calc_gen,bool_gen` |

Results (`scores.txt`; `acc` = official exact match re-applied to the saved prediction, `harness` = the team harness's own
`ok`; they agree on every arm):

| | base | no tools | + tools |
|---|---:|---:|---:|
| Team A, 6 tasks | 30.0 | 61.4 | 100.0 |
| Team B, 3 tasks | 48.1 | 95.7 | 99.7 |
| Team C, 3 tasks | 64.7 | (frozen = base) | 92.4 |
| 12 tasks | 43.2 | 70.8 | 98.0 |

See `../README.md` for the conditions under which these numbers must be reported (large fine-tunes made before rule 15;
Team A's held-out judge round 3: 5/16 with tools).

## Reproduce

```bash
bash data/download_bbh.sh                                   # repo root: official BBH task files + prompts
python scripts/solve_veto.py --out runs/veto_qwen3.pt        # calc veto for Team C
# the fine-tuned hosts (not released): see ../README.md
#   TEAM_A_FT: python bbh_generic/teamA/ft.py --data data/train3.jsonl --out runs/ft_v3 --epochs 2 --lr 1e-5 --tok_budget 16000
#   TEAM_B_FT: python bbh_generic/teamB/ft_B.py --out runs/ft3      (run inside bbh_generic/teamB)
export TEAM_A_FT=/path/to/ft_v3 TEAM_B_FT=/path/to/ft3 VETO=$PWD/runs/veto_qwen3.pt
bash bbh_generic/peek/scripts/queue.sh 0 C_base C_tools B_base B_ft3_notools B_ft3_tools A_base A_ftv3_notools A_ftv3_tools
python bbh_generic/peek/scripts/score.py bbh_generic/peek/out    # -> bbh_generic/peek/scores.json, prints the scores.txt lines
```

`scripts/make_A_jsonl.py` converts the 6 official Team-A task files into Team A's input format (run automatically by
`run_job.sh`; the output `bbh_A.jsonl` is BBH test data and is git-ignored). `scripts/peek_B.py` is Team B's `run_eval.py`
with the official BBH json as the data source; everything else is unchanged.
