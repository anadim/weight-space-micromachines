# IFEval: the 14 rules on bigger Qwen3 hosts, and the add-one-rule curve

The frozen Team D implant (`ifeval/heldout/circuit.py` + `tables.py`, md5 in `../heldout/frozen_MD5.txt`) attached unchanged to
Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B (the chat models, thinking off). The tables depend only on the tokenizer, which is shared;
the **only** host-tied tensor re-solved per host is `force_fix` (the forced-token directions for tokens whose plain write does not
win), from that host's own plain answers to our dev prompts (`../heldout/data/dev.jsonl`), never IFEval.
All runs: one A100 80GB PCIe, bf16, greedy HF loop, bs 24 (same batches plain vs implant), max_new 2048.

## Files

| file | role |
|---|---|
| `run_t.py` | Team D's `run.py` + `WSM_MODEL` host choice and the rule-mask ablation (`RULES`, per-row `ROWCFG`, `BATCHES`) |
| `common.py` | Team D's `common.py`, `MODEL` from env `WSM_MODEL` |
| `solve_forcefix.py` | the force_fix part of `../heldout/solve_vecs2.py`, verbatim, for a new host |
| `score_t.py` | official `lm_eval` checkers, ODD/EVEN/ALL, fixed/broken, silence; **seeded** (`random.seed(key)` per prompt) |
| `rule_gain.py` | per-rule gain on Team D's FRESH set -> the curve order (`results/order_fresh_seeded.txt`) |
| `curve2.py`, `run_jobs.sh` | the exact add-one-rule curve: plan the per-row batch reruns, run them, merge into `k = 0..14` runs |
| `fmt_transfer.py`, `fmt_curve.py` | markdown tables (`results/transfer_tables.md`, `results/curve_*.md`) |
| `detprobe.py`, `broken.py` | scorer order-dependence probe; list of broken prompts |
| `reduce_seeded.py` | `../data/reduce_ifeval_run.py` with the seeded scorer (+ a guard for plain rows) |
| `plot_curve.py` | plots `results/ifeval_curves.json` (the blog figure "Add one rule implant at a time") |
| `run_all.sh` | every command, in order |
| `results/` | `{q06,q17,q4}_score.{json,txt}` (transfer), `{c06,c17,c4}_curve_score.{json,txt}` + `_plan2.json` (curve), `reduced/` (text-free per-prompt records of the 6 transfer runs) |
| `NOTES.md` | the lab notebook, with all tables and the list of broken prompts |

Host-specific force_fix files (not in the repo; rebuilt by step 1 of `run_all.sh`; 0.6B uses `../heldout/vecs2_forcefix.pt`):
`vecs_q17.pt` (1,190 rows, sha256 `6b9955f6...53c5`), `vecs_q4.pt` (1,629 rows, sha256 `018400c9...8263`).

## Reproduce

```bash
python ifeval/data/prepare_ifeval.py
bash ifeval/transfer/run_all.sh      # force_fix for 1.7B/4B, 3 hosts x plain/implant, scores, curve, reduced records
python ifeval/transfer/plot_curve.py # figure from results/ifeval_curves.json
```

The curve order needs Team D's FRESH runs (`../heldout/runs/fresh_{plain,imp}.s*.jsonl`, see `../README.md`); the order used is
hard-coded in `curve2.py` (`nc, para, repeat, forbid, nth, caps, two, lower, end, high, quote, bul, title, ps`).

## Expected numbers (prompt-strict, seeded scorer; `results/q*_score.txt`, `results/transfer_tables.md`)

| host | ALL 541 plain -> 14 rules | gained / lost | ODD 299 | silent prompts byte-identical |
|---|---|---|---|---|
| Qwen3-0.6B | 57.86 -> 72.09 | +80 / -3 | 57.53 -> 71.24 | 324/324 |
| Qwen3-1.7B | 68.76 -> 78.19 | +55 / -4 | 71.91 -> 80.94 | 313/313 |
| Qwen3-4B | 80.41 -> 84.84 | +29 / -5 | 80.94 -> 86.29 | 353/353 |

Add-one-rule curve (ALL 541, `results/ifeval_curves.json`): 0.6B 57.86, 61.92, 63.40, 64.70, 67.47, 67.84, 67.84, 69.32, 70.24,
70.61, 70.61, 71.16, 72.09, 72.09, 72.09 (k = 0..14). Checks built into the merge: rows with all 14 rules = full run byte-identical
268/268, rows with none = plain run 250/250, on every host.

Noise: same GPU, greedy decoding is deterministic (the curve's filler rows prove it byte for byte). Across GPU types bf16 text
differs: Team D's A40 run gives 58.23 -> 72.83 for the same 0.6B implant (`../heldout_results/`). The unseeded official scorer
moves keys 1122 / 1129 (`letter_frequency`, a kwarg drawn from Python `random`): +-1 prompt (+-0.2-0.3 pp).
Runtime: one 541-prompt 0.6B implant run is ~1.4 h in one process on an A100 (23 batches; the circuit is ~1,000 small kernels
per decode step and launch-bound); plain runs are much faster; bigger hosts are slower. The whole script is roughly 1-2 days of
single-GPU time (the original runs shared one A100 between several processes).
