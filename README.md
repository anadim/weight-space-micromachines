# Weight-Space Micromachines: Brain Implants for LLMs

Code for the blog post **[Weight-Space Micromachines: Brain Implants for LLMs](BLOG_URL_PLACEHOLDER)**.

A weight-space micromachine (an "implant") is a small machine that a coding agent (Claude Code) writes straight into
weights and bolts onto a frozen language model. It reads the same tokens as the model (or its hidden states), stays
silent until it is needed, and then writes its result into the model's activations; when it does not fire it adds
exactly zero, so the model's outputs are bit-for-bit identical to the original. Because implants are just more weights,
you can fine-tune or run RL on the model with them attached, and the model learns how and when to use each one.

Everything here runs on public Hugging Face checkpoints (Qwen3-0.6B / 1.7B / 4B, Qwen2.5-0.5B, SmolLM2, Granite,
Falcon3); implants are rebuilt from code. No model weights are in the repository.

* **Start here to reproduce a number: [`REPRODUCE.md`](REPRODUCE.md)** (one section per blog section: environment,
  data, exact commands, expected numbers, runtime, seed noise, and what cannot be rerun from the repo alone).
* [`DETAILS.md`](DETAILS.md): the full project notes (components, disclosures, the retired first BBH pass that overfit
  the test, caveats, exceptions to the in-weights rule). [`VERIFY.md`](VERIFY.md): what was re-run from a fresh clone.

## Quick start (CPU)

```bash
pip install -r requirements.txt     # Python 3.12, torch 2.8, transformers 4.56 (CPU build is fine here)
python demo.py                      # frozen Qwen3-0.6B-Base + calculator implant; checks bit-identical logits when silent
python tests/test_calc_circuit.py cpu        # calculator circuit vs exact Python reference (no model; ~4-5 min on a laptop CPU)
python tests/test_calc_circuit.py cuda 200   # all pairs 0..199, 4 operations (the blog's "every pair up to 199"; GPU, or ~30 min with cpu)
(cd bbh_generic/teamA && TOOL_DEVICE=cpu PYTHONPATH=../.. python test_tools.py 50)   # BBH track/order implants vs Python reference (Qwen tokenizer, ~1 min)
```

## The implants

| implant | what it computes | blog headline | code |
|---|---|---|---|
| Calculator | exact + − × ÷ on the digits the model writes (`calc_s3`, `calc_s4`; ~30-50 ReLU layers, ~1-2M weights) | ×: 4% → 100% (Qwen2.5-0.5B); GSM-Hard 27% → 35% (Qwen3-0.6B) | [`micromachines/circuits/calc.py`](micromachines/circuits/calc.py), [`scripts/run_gsm.py`](scripts/run_gsm.py) |
| Calls taught by fine-tuning | the model learns to write `<calc>48 * 2</calc>` / `<tool_call>…</tool_call>`, the implant fills in the answer | GSM-Hard ~13% → 23%; BBH-style 29% → 75% with ~100 examples | [`finetuned_phase/`](finetuned_phase/README.md), [`bbh_generic/ft100/`](bbh_generic/ft100/README.md) |
| Eyes | a small CNN (421k weights) reads MNIST digits pasted in as pixel tokens | 95% vs 4% with 32 examples; plain fine-tuning needs ~16,000 | [`finetuned_phase/eyes_and_finetune/`](finetuned_phase/README.md), [`evidence/eyes_sweep/`](evidence/README.md) |
| 14 IFEval rule implants | reader + enforcer per rule (no commas, lowercase, end phrase, …); ~9k ReLU units in plain weights | 58% → 72% on IFEval (fixed 80, broke 3; 324/324 silent prompts byte-identical) | [`ifeval/`](ifeval/README.md) |
| 8 BBH implants + call gate | `track`, `order`, `interval`, `walk`, truth, `expr`, `calc`, `bool`; a hidden-state gate that blocks stray calls | 12 tool tasks 43% → 84%; 27-task mean 48% → 55% with the gate | [`bbh_generic/`](bbh_generic/README.md), [`bbh_generic/callgate/`](bbh_generic/callgate/README.md) |
| Transfer | same circuits, refit reader + writer for a new host | Qwen3-1.7B / 4B; SmolLM2, Granite (calculator), SmolLM2 (IFEval) | [`transfer/xfamily/`](transfer/xfamily/README.md), [`ifeval/transfer/`](ifeval/transfer/README.md), [`bbh_generic/transfer/`](bbh_generic/transfer/README.md) |
| Mistake detector | reads hidden states before the final answer; forces "Hmm, wait. Let me rethink this." | GSM8K 52% → 56%; with the calculator 57% GSM8K / 35% GSM-Hard; AUROC 0.85 | [`micromachines/rethink_circuit_v2.py`](micromachines/rethink_circuit_v2.py), [`scripts/rethink_v2/`](scripts/rethink_v2/), [`results/rethink_v2/`](results/rethink_v2/) |
| Topic blocker | a linear probe on the prompt's last hidden state, compiled into one ReLU unit; writes a decline sentence | blocks ~99% of math requests; 10% of normal requests (vs 6%) | [`topic_fence/`](topic_fence/README.md) |
| Latent write | computes 23×7 and writes the state of "161" into the residual stream (no text) | (23×7)+58: 15% → 96-99%; word problem 3% → 98% (text tool line: 70%) | [`latent_write/`](latent_write/README.md) |
| RL with implants | GRPO on Qwen3-0.6B with implants (and trainable wiring) attached | GSM-Hard 44% (plain RL) vs 55% (RL + calculator); RL learns a half-broken calculator's competence map; mixed tasks 68% vs 36% | [`rl/`](rl/README.md) |

## Reproducing the headline numbers

One GPU per command; `REPRODUCE.md` has the full commands, the expected values and the seed noise.

| blog number | command (from the repo root unless shown) | expected / result file | REPRODUCE |
|---|---|---|---|
| Calculator: GSM8K 52 → 53, GSM-Hard 27 → 35 (Qwen3-0.6B-Base) | `python scripts/solve_veto.py --device cuda && python scripts/run_gsm.py out/hard_calc.json --split hard --mode calc --bs 32` (and `--mode plain`, `--split test`) | 52.24 → 53.37, 27.29 → 35.03; `results/gsm_full_implant/` | §1 |
| Calculator: × 4% → 100% (Qwen2.5-0.5B) | `cd finetuned_phase/calc_icl && python calc_icl.py 100` | 4.0 → 100.0; `finetuned_phase/results/calc_icl_100.json` | §1 |
| Fine-tune with calls: GSM-Hard 13 → 23, templates 51 → 99 | `cd finetuned_phase/eyes_and_finetune && bash seedjobs.sh` | `finetuned_phase/results/reported_numbers.json` | §2 |
| Eyes: 95% vs 4% at 32 examples | `bash evidence/code/eyes_sweep3.sh` (after `eyes.py cnn` / `eyes.py data 16000`) | 94.7 vs 3.7 (4 seeds); `evidence/eyes_sweep/results/` | §3 |
| IFEval 58 → 72 | `bash ifeval/transfer/run_all.sh` | 57.86 → 72.09, +80 / −3; `ifeval/transfer/results/q06_score.txt` | §5 |
| BBH 12 tool tasks 43 → 84; ~100 examples 29 → 75 | `bbh_generic/integrate/code/run27_integ.py` (see §6.1), `bbh_generic/ft100/README.md` | 43.27 → 84.33; 29.0 → 76.3 / 73.7 | §6 |
| Call gate: 27-task 48 → 55, stray calls 59% → 4% | `for s in states probe bbh sets arc report; do bash bbh_generic/callgate/code/run_callgate.sh $s; done` | 48.36 → 54.58; 59.2% → 3.8% | §6.2 |
| Transfer to 1.7B / 4B and other families | `bash bbh_generic/transfer/code/run_transfer.sh q17` · `bash transfer/xfamily/code/drvr.sh …` | `bbh_generic/transfer/RESULTS.md`, `transfer/xfamily/` | §8 |
| Mistake detector: 52 → 56; + calculator 57 / 35 | `bash scripts/rethink_v2/queue.sh scripts/rethink_v2/jobs_hybrid.txt 4` | `results/rethink_v2/summary.json` | §9 |
| Topic blocker ~99% / 10% vs 6% | `bash topic_fence/run_test.sh` | `topic_fence/results/q17_math_table.txt` | §10 |
| Latent write 15 → 96-99 | `bash latent_write/run_all.sh` | `latent_write/summary.md` | §11 |
| RL: GSM-Hard 44 vs 55 after 300 updates (3 seeds) | `cd rl/gsm_grpo && python train2.py --arm 2 --seed 0 --run_dir runs/long/a2_s0 --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --dev_n 800 --dev_every 25 --ckpt_every 100 --fused 1` (arm 1 = no implant), then `final_eval.py` | `rl/results/gsm_grpo/H100/final/` | §12 |
| RL: half-broken calculator, x÷ let through, +− blocked | `cd rl/wiring && bash launch.sh 0 1 40000 "" q4_a100.txt` | `rl/results/wiring/figdata/fig_competence_map.json` | §12 |
| RL: all implants attached; mixed tasks 68 vs 36 | `bash rl/library/launch/h100_LIB_s0.sh`; `bash rl/library/launch/h100_m1_mixLIB.sh` | `rl/library/NOTEBOOK.md`, `rl/results/library/figdata/` | §12 |
| RL: distilled links +11 (+24 on big numbers) | `cd rl/wiring && bash launch.sh 0 1 40000 "" q5_a100.txt` | `rl/results/wiring/figdata/W3d_s*.jsonl` | §12 |
| Blog figures | `bash figures/blog/make_blog_figs.sh`; RL: `cd rl/figures && python make_blog_figs_rl.py && python make_rl45.py && python make_rl6.py` (CPU) | `figures/blog/figs/`, `rl/figures/` | §13 |

## Hardware

* Every result needs **one GPU**. Runs were done on NVIDIA A100 80 GB, H100 80 GB and A40 48 GB, bf16, greedy decoding
  unless stated. 0.6B-host runs fit in < 20 GB; the RL runs use 30-60 GB at the default batch (`--B_max 128` for
  smaller GPUs); the A40 needs `FLEX_SMALL=1` for the RL engines (set by the launchers).
* bf16 greedy text differs slightly across GPU types and batch sizes (a few percent of items; ~1 point on a benchmark).
  Compare within one GPU type. `REPRODUCE.md` lists which GPU produced each number.
* `demo.py`, the circuit exactness tests, all table summaries and all figure scripts run on a CPU.
* Environment: Python 3.12, torch 2.8.0 (CUDA 12.8 / 12.9 wheels), transformers 4.56.2, triton 3.4.0 (RL);
  `requirements.txt`, `requirements-extra.txt` (vLLM 0.10.2 for a few generation steps, lm_eval for the IFEval checkers).

## RL engine kernels

The RL engines use custom GPU kernels that are compiled at runtime to PTX / cubin, with no nvcc and no build step:

* the v2 engine behind the blog's RL numbers (`rl/gsm_grpo/engine.py`, `hostkern.py`, `implant_persist.py`): host
  layers and the calculator implant as one persistent kernel, written in **Triton** (JIT to PTX), with the whole decode
  step captured as a CUDA graph;
* `rl/engine2/` (work in progress, no blog number uses it): the 9 BBH/calculator implants as one integer-exact
  megakernel and a fused log-prob head, written in CUDA C++ and compiled at load time with **NVRTC** (cubin for the
  local `sm_XX`), launched through the driver API so the launches stay CUDA-graph capturable.

## What is not in the repository

* Model checkpoints: fine-tuned hosts, RL'd hosts, trained wiring. Recipes, seeds and training rows (or seeded
  generators with SHA-256) are included.
* Files over 5 MB that scripts rebuild, e.g. the compiled IFEval implant tensor (1.23 GB, `ifeval/full14_compiled/export.py`,
  SHA-256 in `REPRODUCE.md`), `ifeval/heldout/vecs2.pt` (11.6 MB; its used rows are `vecs2_forcefix.pt`), the engine2
  table cache `rl/engine2/tables.pt` (118 MB, rebuilt on first use).
* Generated text and benchmark question text: result files keep per-question records (gold, prediction, correctness,
  fire flags, hashes) only.
* The full list, item by item, is `REPRODUCE.md` §14.

## License

MIT (see `LICENSE`). Benchmarks and models are downloaded from their original sources under their own licenses.
