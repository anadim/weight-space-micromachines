# BBH transfer to bigger Qwen3 hosts + the add-one-tool curve (27 tasks)

What it is: the integration-v2 setup (`../integrate/`: one host fine-tuned on the same 432 generic rows, the eight generic
tools attached, Team A/B in call form, Team C passive) moved from Qwen3-0.6B-Base to **Qwen3-1.7B-Base** and
**Qwen3-4B-Base**, plus the **cumulative add-one-tool curve** on the 0.6B v2 host over all 27 real BBH tasks.

What changes per host (generic data only, never BBH): the readers are re-solved from the host's embedding matrix (exact
over all 151,936 rows; the three hosts share the tokenizer), the write directions are re-solved against the host's final
norm + LM head, and the `calc_gen` veto is re-solved on GSM8K-train number endings (`scripts/solve_veto.py --model ...`).
The circuits are unchanged. The fine-tune recipe is unchanged (lr 2e-5, 6 epochs, tok_budget 8000; 156 steps).

Arms per host, 27 real BBH tasks, official 3-shot CoT, greedy, `max_new 600`: (a) base, (b) frozen base + Team C passive
tools (no fine-tune), (c) FT without tools, (d) FT + all 8 tools.

## Results (strict; `RESULTS.md` has strict / lenient / no-answer for every arm)

| host | base 27 / 12 / 15 | frozen + Team C 27 | FT no tools 27 / 12 / 15 | FT + 8 tools 27 / 12 / 15 |
|---|---|---|---|---|
| 0.6B | 40.98 / 43.27 / 39.15 | 42.95 | 33.61 / 38.10 / 30.01 | 48.36 / 84.33 / 19.58 |
| 1.7B | 52.08 / 55.73 / 49.15 | 52.70 | 43.12 / 44.30 / 42.18 | **65.86 / 96.37** / 41.45 |
| 4B | 71.14 / 82.83 / 61.79 | 71.63 | 52.40 / 53.93 / 51.17 | 70.98 / **97.77** / 49.54 |

Add-one-tool curve (0.6B v2 host, 27 / 12 / 15 means): k0 none 33.61 / 38.10 / 30.01; +interval 33.31 / 40.60 / 27.49;
+track 39.33 / 59.90 / 22.88; +order 43.51 / 71.63 / 21.00; +truth 43.61 / 72.47 / 20.52; +walk 44.08 / 75.00 / 19.35;
+Team C passive (= all 8) 48.36 / 84.33 / 19.58. Per task: `curve_table.md`.

Reading: the tool tasks reach 96-98% on the bigger hosts, but the 432-row call fine-tune costs the 4B host ~12 points on
the 15 no-tool tasks, so its 27-mean is a wash (71.14 -> 70.98). The passive Team C tools (no call, no fine-tune) never
hurt the 27-mean on any host. 1.7B `boolean_expressions` frozen+C drops in strict scoring (58.0 -> 41.2) because the base
host does not write "the answer is" after the passive value (lenient 96.4).

## Files

* `code/run_transfer.sh` — every run, as steps (`vetoes`, `ft`, `checks`, `q17`, `q4`, `curve`, `report`); calls
  `../integrate/code/run27_integ.py` (`--veto`, `--tools subset --machines`, task-level resume from `.partial`).
* `code/compile_final.py` — merges the forward / reverse task shards of the original runs (first file wins per task; no
  task scored twice). `code/report_transfer.py` — strict / lenient / no-answer / call-rate tables. `code/curve_table.py`
  — the curve table. `code/reduce_transfer.py` — how `results/` was made from the full outputs.
* `results/{q06,q17,q4}_{base,frozenC,ft_notools,ft_tools}.json.gz`, `results/curve_k{1..5}.json.gz` — per-question
  records (task, gold, prediction, strict and lenient correctness, tool calls, SHA-256 of the generated text; no text).
  `results/check_ft{17,4}_fast.json` — host checks (silence = bit-identity, per-tool end-to-end).
* `NOTEBOOK.md`, `RESULTS.md`, `pertask_{q06,q17,q4}.md`, `curve_table.md` — the run log and full tables.

Print the tables from the shipped records (CPU, seconds):

```bash
python bbh_generic/transfer/code/report_transfer.py bbh_generic/transfer/results \
    q17_base q17_frozenC q17_ft_notools q17_ft_tools q4_base q4_frozenC q4_ft_notools q4_ft_tools q06_base q06_ft_notools q06_ft_tools
python bbh_generic/transfer/code/curve_table.py bbh_generic/transfer/results
```

## Reproduce from scratch

```bash
bash data/download_bbh.sh
bash bbh_generic/transfer/code/run_transfer.sh vetoes
bash bbh_generic/transfer/code/run_transfer.sh ft        # needs ../integrate/data/train_union_v2.jsonl (shipped)
bash bbh_generic/transfer/code/run_transfer.sh checks
bash bbh_generic/transfer/code/run_transfer.sh q17       # 4 arms x 6,511 questions per host: several GPU-hours each
bash bbh_generic/transfer/code/run_transfer.sh q4
FT_HOST=bbh_generic/integrate/runs/ft_int_v2 bash bbh_generic/transfer/code/run_transfer.sh curve   # 0.6B v2 host first
bash bbh_generic/transfer/code/run_transfer.sh report
```

Notes and disclosures:
* The 1.7B / 4B fine-tuned hosts and the vetoes are not released (checkpoints). The original 1.7B / 4B vetoes were reused
  from the bigger-host GSM runs (`results/gsm_bigger/`), solved with the same `scripts/solve_veto.py` on GSM8K train.
* The 0.6B frozen + Team C row is an older run from `../ft100/` (same loop); rows exist for 22 of 27 tasks, so its lenient
  score is not available.
* The original runs were split into forward / reverse task shards on two machines (2x H100, 2x A40) because of shared-GPU
  OOM restarts; `run27_integ.py` resumes per task, and greedy decoding makes a resumed task identical. Batch sizes: 32,
  except the 4B FT+tools arm (48). Different batch sizes / GPUs change a few percent of bf16 greedy transcripts, so expect
  each 27-mean within about +-1 point.
