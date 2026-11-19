# IFEval rules on another model family: SmolLM2-1.7B-Instruct

The frozen 14-rule implant (`ifeval/heldout/`) ported to `HuggingFaceTB/SmolLM2-1.7B-Instruct` (ChatML template, GPT-2-style
byte-level BPE like Qwen, vocabulary 49,152). `port_ifeval.py` is a deterministic text transform of the four frozen files that
replaces only Qwen-specific constants (EOS / role ids, the special-token boundary, the answer-start rule) by a host config; every
vocabulary table is then rebuilt from the new tokenizer by the unchanged `tables.py`. The only solved tensor is force_fix
(`solve_force.py`, from the host's own plain answers to our dev prompts; 39 tokens; `vecs_sm2i.pt`, 0.33 MB, sha256
`c5577724...7a7d`, included). Regenerating the port from the repo files reproduces the run's `circuit.py` byte for byte (md5 checked).

| arm (IFEval odd keys, n = 299, never opened before) | prompt-strict | instruction-strict | gained / lost |
|---|---:|---:|---:|
| plain | 48.49 | 59.82 | |
| **v1** (constants only, the headline) | **60.54** | 71.08 | +42 / -6 |
| v2.1 (post-diagnosis: fixed emitter strings as host token sequences) | 62.54 | 72.85 | +44 / -2 |

Silent prompts byte-identical to plain: 164/164. v2.1 was written after reading the v1 odd-key outputs (paragraph dividers
`***\n\n` are two tokens in SmolLM2), so only v1 is a held-out number. Re-scoring the reduced records gives 60.87 (v1) and 62.88
(v2.1): one prompt moves with the scorer's `letter_frequency` random fill / langdetect (see `../heldout_results/README.md`).

Files: `port_ifeval.py`, `port_v2.py`, `solve_force.py`, `vecs_sm2i.pt`, `run_sm2i.sh` (all commands), `results/score_odd_v1.txt`,
`results/score_odd_v21_A40.txt` (scorer output), `results/solve_force.log`, `results/sm2i_odd_{plain,imp,imp_v21_A40}.json`
(text-free per-prompt records). Hardware: one A40 (torch 2.8.0, transformers 4.56.2), bf16, greedy, bs 32, max_new 2048.
Granite / Falcon3 instruct models use other chat templates (a new role / answer-start rule) and were not attempted.
