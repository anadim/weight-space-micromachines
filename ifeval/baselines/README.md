# IFEval baselines for the 14-rule implant

**(a) Thinking mode instead of implants.** Plain Qwen3-0.6B with `enable_thinking=True`, the same 2,048-token budget, greedy,
bs 16. Scored on the answer after `</think>` (`strip_think.py`; a reply that never closes `</think>` counts as an empty answer).

| prompt-strict | ALL 541 | ODD 299 |
|---|---:|---:|
| plain, thinking off (A40 run, `../heldout_results/`) | 58.23 | 56.19 |
| plain, **thinking on**, answer after `</think>` | **59.33** | 59.20 |
| 14-rule implant, thinking off | 72.83 | 71.91 |

50/541 thinking replies hit the 2,048 cap (26 never close `</think>`); mean 659 tokens. Even counting all 50 as passes the
ceiling is ~68.6. Record: `results/ifeval_think_plain.ans.json` (text-free, per prompt; re-scores to 59.33 / 59.20).

**(b) Silence at scale.** The 14 rule detectors on 1,000 wikitext-103 test passages (133,754 tokens) and 600 PyTorch source files
(94,137 tokens), each given as a user message: **0/1000** wikitext passages and **1/600** code files latch any rule (`para`, on a
source file); a latch is the precondition for any write, so all other replies are byte-identical to the plain host by
construction. `results/ifeval_silence.json` (the latched code excerpt is omitted).

Run (one GPU; ~2.5 h on an A40 for (a), ~30 min CPU-bound for (b)): `bash ifeval/baselines/run_baselines.sh` from the repo root.
Uses `ifeval/heldout/{circuit,common,score}.py`. Software: torch 2.8.0, transformers 4.56.2, lm_eval 0.4.13, datasets (wikitext).
Scorer noise: +-0.2-0.3 pp between scorer calls (langdetect on degenerate text and the `letter_frequency` random fill).
