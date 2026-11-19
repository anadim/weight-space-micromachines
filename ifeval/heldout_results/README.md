# Result files behind the headline IFEval number (Team D, 14 rules, frozen Qwen3-0.6B)

These are the text-free per-prompt records of the two runs scored by `ifeval/heldout/final_score.sh` (the blog's
"58% -> 73% on all 541 prompts, +82 / -3, silent prompts byte-identical 323/323"). Generated on one NVIDIA A40
(torch 2.8.0, transformers 4.56.2, bf16, greedy, bs 24, max_new 2048, thinking off).

| file | what |
|---|---|
| `final_score.out` | the scorer output as run (`final_score.sh`: ODD / EVEN / ALL tables, per-rule passes, silence, IFBench) |
| `teamd_ifeval_plain.json` | plain host, 541 prompts: per instruction strict/loose pass, finish, ntok, SHA-256 of the response |
| `teamd_ifeval_imp.json` | same with the 14-rule implant; adds the circuit's numeric report (`fires`, counters) |

Commands that produced the runs: `ifeval/README.md` (heldout section): `./queue.sh 4 "ifeval.jsonl ifeval_plain.sK.jsonl plain K/4"`
and the same with `implant`, then `./final_score.sh`. The JSONs were reduced with `ifeval/data/reduce_ifeval_run.py` (shards
concatenated; a `circ: null` guard added for plain rows).

Numbers (from `final_score.out`; prompt-level strict):

| split | n | plain | + 14 rules | gained / lost |
|---|---:|---:|---:|---:|
| odd keys (held out) | 299 | 56.19 | 71.91 | +49 / -2 |
| even keys | 242 | 60.74 | 73.97 | |
| all | 541 | 58.23 | 72.83 | +82 / -3 |

Re-scoring the reduced records gives 56.19 -> 72.24 (odd), 60.33 -> 73.55 (even), 58.04 -> 72.83 (all, +82 / -2): the
official `lm_eval` checker fills a missing `letter_frequency` kwarg with Python `random` (keys 1122 and 1129), so a verdict on
those two prompts depends on how many prompts were scored before. That is the whole difference (1 prompt per split). The
seeded scorer in `ifeval/transfer/score_t.py` / `reduce_seeded.py` removes it. Silent prompts byte-identical: 323/323.
Note: these runs are on an A40. The same frozen implant re-run on an A100 (`ifeval/transfer/`, `ifeval/full14_compiled/`) gives
57.86 -> 72.09 (+80 / -3): bf16 greedy decoding is not bit-identical across GPU types, so the plain baseline moves a little.
