# BBH transfer: RESULTS (compiled 2026-11-18 06:55 CST)

All runs are complete: every arm has 27/27 tasks and the full item count (6,511 rows), checked against the 0.6B base rows.
The one exception is the 0.6B frozen+TeamC arm, a run from before tonight (see note 1).
Each arm ran BBH once. Some arms were split into forward and reverse task-order shards. Every task is scored exactly once, and
results are aggregated per task only. Scoring uses `code/report_transfer.py`. strict = the official "the answer is X" extractor.
lenient = strict OR a fallback extractor on the same saved output. na = the share of outputs where strict extraction found no answer.
"call% 15" = the share of outputs on the 15 no-tool tasks that contain `<tool_call>`.

## Main table (strict / lenient)

| host | arm | 27 strict | 27 lenient | 12 strict | 12 lenient | 15 strict | 15 lenient | na% 27 | na% 15 | call% 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.6B | base | 40.98 | 41.21 | 43.27 | 43.57 | 39.15 | 39.32 | 7.7 | 9.6 | 0.0 |
| 0.6B | frozen + Team C (note 1) | 42.95 | n/a | 49.63 | n/a | 37.61 | n/a | n/a | n/a | 0.0 |
| 0.6B | FT no tools | 33.61 | 33.81 | 38.10 | 38.17 | 30.01 | 30.33 | 3.7 | 5.5 | 59.2 |
| 0.6B | FT + 8 tools | 48.36 | 48.76 | 84.33 | 84.37 | 19.58 | 20.27 | 22.3 | 36.0 | 59.2 |
| 1.7B | base | 52.08 | 54.05 | 55.73 | 59.40 | 49.15 | 49.78 | 7.7 | 8.8 | 0.0 |
| 1.7B | frozen + Team C | 52.70 | 55.31 | 57.03 | 62.13 | 49.23 | 49.85 | 8.1 | 8.7 | 0.0 |
| 1.7B | FT no tools | 43.12 | 43.14 | 44.30 | 44.33 | 42.18 | 42.18 | 3.2 | 4.8 | 11.5 |
| 1.7B | FT + 8 tools | 65.86 | 65.86 | 96.37 | 96.37 | 41.45 | 41.45 | 3.3 | 5.9 | 11.6 |
| 4B | base | 71.14 | 71.29 | 82.83 | 82.83 | 61.79 | 62.06 | 3.6 | 6.4 | 0.0 |
| 4B | frozen + Team C | 71.63 | 71.81 | 83.93 | 84.00 | 61.79 | 62.06 | 3.7 | 6.4 | 0.0 |
| 4B | FT no tools | 52.40 | 52.40 | 53.93 | 53.93 | 51.17 | 51.17 | 4.1 | 7.4 | 29.1 |
| 4B | FT + 8 tools | 70.98 | 70.99 | 97.77 | 97.77 | 49.54 | 49.57 | 6.8 | 12.2 | 29.1 |

Within each host, the FT arms with and without tools have identical call rates. This is expected: with no trigger the
machines write nothing (silence = bit-identity), so both arms produce the same text up to the first call.

Per-task tables (strict / lenient / na% for each task and arm): `pertask_q06.md`, `pertask_q17.md`, `pertask_q4.md` (this folder).
Merged per-arm rows (reduced, no text): `results/{q06,q17,q4}_{base,frozenC,ft_notools,ft_tools}.json.gz`.

Team C tasks, strict (base -> frozen+C -> FT+tools):
- 1.7B: boolean 58.0 -> 41.2 (lenient 96.4; 56% na: the base host does not write "the answer is" after the passive
  value) -> 100; multistep 72.0 -> 90.4 -> 100; object_counting 70.8 -> 84.8 -> 88.8.
- 4B: boolean 92.4 -> 98.4 -> 100; multistep 94.4 -> 100 -> 98.4; object_counting 96.8 -> 98.4 -> 94.8.

### Notes
1. 0.6B frozen + Team C was not run tonight. Its numbers are the older 27-task run from `../ft100` (score27.txt in the run ledger:
   same loop in run27.py, the 0.6B base veto, all 27 tasks scored from the per-task logs). Saved rows exist for only 22 of the
   27 tasks: the first 5 tasks (boolean, causal, date, disambiguation, dyck) exist only as log lines. So lenient and na
   cannot be computed for all 27. On the 22 tasks with rows: strict 42.16, lenient 42.42, na 10.1%.
2. Batch size: 1.7B arms used bs 32. 4B base, FT no tools and frozen+C used bs 32 (= the v2 harness). 4B FT+tools used
   bs 48 for both of its shards, as it was launched before this change. q17_frozenC used bs 32 (launch.sh had 64; changed
   before its first launch). The change was made to fit GPU memory. Within each arm the batch size is the same.
3. The 0.6B rows are merged from the existing ../integrate shards (T1/T2, P0/P1/P2 parts). The tasks are disjoint, and
   the merged numbers reproduce TABLES_v1_vs_v2.md exactly (40.98 / 33.61 / 48.36).

## 4B host check (results/check_ft4_fast.json): PASS. The 4B tool numbers can be trusted.
- check 3 SILENCE: 60/60 all-zero writes and 60/60 bit-identical logits.
- check 2 e2e: track, order, interval, turtle (walk) and truth 40/40 each. expr 40/40, bool 39/40, calc 39/40.
- check 1b (passive C): expr 444/444, calc 342/342, bool 378/381, and silent on 202/202 A/B training transcripts.
- The failures match the 1.7B check: the 3 bool failures are one formula longer than the 48-token window of bool_gen
  (silent, fail-safe). The calc failure is the 4B host's own rounding of the copied value ("-0.0607711962616" vs
  "-0.060768"). The bool e2e failure has the same cause as the 3 bool failures.

## Cumulative add-one-tool curve (Qwen3-0.6B-Base v2 FT host; strict; all 27 tasks complete in every row)

| k | tool added | 27-mean | 12-mean | 15-mean | no-tool call rate % |
|---|---|---|---|---|---|
| k0 | none (= v2 FT no tools, existing) | 33.61 | 38.10 | 30.01 | 59.2 |
| k1 | + interval | 33.31 | 40.60 | 27.49 | 59.4 |
| k2 | + track | 39.33 | 59.90 | 22.88 | 59.4 |
| k3 | + order | 43.51 | 71.63 | 21.00 | 59.5 |
| k4 | + truth | 43.61 | 72.47 | 20.52 | 59.6 |
| k5 | + walk | 44.08 | 75.00 | 19.35 | 59.6 |
| k6 | + Team C passive (= v2 FT + all 8 tools, existing) | 48.36 | 84.33 | 19.58 | 59.2 |

Per-task curve table: `curve_table.md`. Merged rows: `results/curve_k{1..5}.json.gz`.
Sources: k1 = the A40 machine forward 16 tasks + the H100 machine reverse 11. k2 = the A40 machine (all 27). k3 = the H100 machine forward 18 + the A40 machine
reverse 9. k4 and k5 = the H100 machine forward 16 + the H100 machine middle shard 3 + the A40 machine reverse 8. The the H100 machine curve runs used the 0.6B
v2 checkpoint integrate/runs/ft_int_v2 and veto_qwen3.pt. Both are md5-identical to the A40 machine's ft_int_v2_orig and veto.

## Operations log (what was changed to finish)
- run27_integ.py (both pods): added a task-level resume from `.partial`. A partial is written only after a task
  completes, and decoding is greedy, so the computation is unchanged. Backups: run27_integ.py.bak_noresume. The the H100 machine copy
  also got the A40 machine's `--tools subset --machines` option (superset; backup .bak_nosubset).
- The cause of the trouble was OOM kills from other agents' jobs on shared GPUs. The the A40 machine curve k1 OOMed at the start,
  and q17_ft_tools, q4_base, q17_frozenC, q4_frozenC and k2 all OOMed once or more. Each was restarted from its partial.
  dispatch.sh was replaced by dispatch3.py, a memory-aware watchdog that relaunches failed jobs and pairs forward and
  reverse shards. No other agent's process was touched.
- Shut down at 06:52 CST: all run27_integ, dispatcher, monitor and curve-script processes of this experiment, on both
  pods. Nothing of this experiment is still running.
