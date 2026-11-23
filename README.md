# Weight-Space Micromachines: Brain Implants for LLMs

Code for the blog post **[Weight-Space Micromachines: Brain Implants for LLMs](https://papail.io/papers/wsm/)**.

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

## Results at a glance

Blog numbers, rounded as in the blog; exact values, seeds and commands in the `REPRODUCE.md` section named.

| result (Qwen3-0.6B-Base unless noted) | blog | REPRODUCE |
|---|---|---|
| Calculator, frozen host: 4–5-digit multiplication; GSM-Hard | near 0% → 100%; 27% → 35% | §1 |
| Mistake detector: GSM8K | 52% → 56% | §9 |
| Topic blocker (Qwen3-1.7B chat): math requests caught; normal requests blocked | ~99%; ~4% | §10 |
| Latent write: two-step sum in one shot, no text | 15% → 97% | §11 |
| IFEval, frozen Qwen3-0.6B (chat) + 14 rule implants | 58% → 72% | §5 |
| BIG-Bench Hard, 9 implants (8 tools + call gate), no RL | 41% → 55% | §6, §17 |
| IFEval, frozen Base + 14 rule implants, no RL | 13% → 35% | §17, §22 |
| GSM8K, frozen RL'd host + 8 implants, one generation call | 68% → 83% | §16 |
| GSM-Hard after 300 RL updates: with vs without the calculator (3 seeds) | 55% vs 44% | §12 |
| RL with implants vs plain RL: GSM8K / BBH / IFEval (on top of RL: +12 / +7 / +3; +17 on GSM-Hard) | 80% vs 68% / 60% vs 54% / 48% vs 46% | §17, §22 (6 RL seeds on GSM8K, IFEval) |
| Mixed GSM / BBH / IFEval tasks, RL with all implants vs plain RL (Qwen3-0.6B chat, 3 seeds) | 58% vs 45% | §12, §23 |
| Qwen3-1.7B-Base GSM8K: plain RL + implants attached after vs plain RL | 90% vs 81% | §18 |

## Quick start (CPU)

```bash
pip install -r requirements.txt     # Python 3.12, torch 2.8, transformers 4.56 (CPU build is fine here)
python demo.py                      # frozen Qwen3-0.6B-Base + calculator implant; checks bit-identical logits when silent
python tests/test_calc_circuit.py cpu        # the blog's calculator test: all pairs 0..59, signed -40..39, 12k random, 4 operations (no model; ~4-5 min on a laptop CPU)
python tests/test_calc_circuit.py cuda 200   # all pairs 0..199, 4 operations (160k cases, stronger than the blog's test; GPU, or ~30 min with cpu)
(cd bbh_generic/teamA && TOOL_DEVICE=cpu PYTHONPATH=../.. python test_tools.py 50)   # BBH track/order implants vs Python reference (Qwen tokenizer, ~1 min)
```

## The implants

| implant | what it computes | blog headline | code |
|---|---|---|---|
| Calculator | exact + − × ÷ on the digits the model writes (`calc_s3`, `calc_s4`; ~50 ReLU layers, ~2.4M weights) | 4–5-digit ×: near 0% → 100%; GSM-Hard 27% → 35% (Qwen3-0.6B-Base) | [`micromachines/circuits/calc.py`](micromachines/circuits/calc.py), [`scripts/run_gsm.py`](scripts/run_gsm.py) |
| Calls taught by fine-tuning | the model learns to write `<calc>48 * 2</calc>` / `<tool_call>…</tool_call>`, the implant fills in the answer | GSM-Hard ~13% → 23%; BBH-style 29% → 75% with ~100 examples | [`finetuned_phase/`](finetuned_phase/README.md), [`bbh_generic/ft100/`](bbh_generic/ft100/README.md) |
| Eyes | a small CNN (421k weights) reads MNIST digits pasted in as pixel tokens | 95% vs 4% with 32 examples; plain fine-tuning needs ~16,000 | [`finetuned_phase/eyes_and_finetune/`](finetuned_phase/README.md), [`evidence/eyes_sweep/`](evidence/README.md) |
| 14 IFEval rule implants | reader + enforcer per rule (no commas, lowercase, end phrase, …); ~9k ReLU units in plain weights | 58% → 72% on IFEval (fixed 80, broke 3; 324/324 silent prompts byte-identical) | [`ifeval/`](ifeval/README.md) |
| 8 BBH implants + call gate | `track`, `order`, `interval`, `walk`, truth, `expr`, `calc`, `bool`; a hidden-state gate that blocks stray calls | 12 tool tasks 43% → 84%; 27-task mean 48% → 55% with the gate | [`bbh_generic/`](bbh_generic/README.md), [`bbh_generic/callgate/`](bbh_generic/callgate/README.md) |
| Transfer | same circuits, refit reader + writer for a new host | Qwen3-1.7B / 4B; SmolLM2, Granite (calculator), SmolLM2 (IFEval) | [`transfer/xfamily/`](transfer/xfamily/README.md), [`ifeval/transfer/`](ifeval/transfer/README.md), [`bbh_generic/transfer/`](bbh_generic/transfer/README.md) |
| Mistake detector | reads hidden states before the final answer; forces "Hmm, wait. Let me rethink this." | GSM8K 52% → 56%; with the calculator 57% GSM8K / 35% GSM-Hard; AUROC 0.85 | [`micromachines/rethink_circuit_v2.py`](micromachines/rethink_circuit_v2.py), [`scripts/rethink_v2/`](scripts/rethink_v2/), [`results/rethink_v2/`](results/rethink_v2/) |
| Topic blocker | a linear probe on the prompt's last hidden state, compiled into one ReLU unit; writes a decline sentence | catches ~99% of math requests; blocks ~4% of normal requests (fires on 23% of non-math MMLU questions) | [`topic_fence/`](topic_fence/README.md) |
| Latent write | computes 23×7 and writes the state of "161" into the residual stream (no text) | (23×7)+58: 15% → 96-99%; word problem 3% → 98% | [`latent_write/`](latent_write/README.md) |
| GSM8K single-call stack | calculator v2 (chain2), answer-sanity DFA, weight-space forget, serial tries + answer register + ReLU voter, weight-space position reset, adaptive tries, answer-group voter; all in weights, one generation call | GSM8K test 67.7 → 72.3 → 80.7 → 81.6 → 82.7 on the frozen RL+calc host; RL from the base model with the stack 80% vs plain RL 68% (6 RL seeds) | [`gsm_stack/`](gsm_stack/README.md) (earlier, partly programmatic version: [`gsm_push/`](gsm_push/README.md)) |
| RL with implants | GRPO on Qwen3-0.6B with implants (and trainable wiring) attached | GSM-Hard 44% (plain RL) vs 55% (RL + calculator); RL learns a half-broken calculator's competence map; mixed tasks 58% vs 45% (3 seeds) | [`rl/`](rl/README.md) |

## GSM8K single-call implant stack

[`gsm_stack/`](gsm_stack/README.md): a frozen Qwen3-0.6B host answers a GSM8K question in one generation call, and
every decision is made by weights; the only code outside the weights is the generator (sampling, token budget, stop on
eos). What counts as weights, the generator-only exceptions and the non-compliant reference arms are in
[`gsm_stack/COMPLIANCE.md`](gsm_stack/COMPLIANCE.md); where each host comes from is in
[`gsm_stack/MODEL_CARD.md`](gsm_stack/MODEL_CARD.md).

| implant | what it does |
|---|---|
| calculators `calc_s4`, `chain2` | exact arithmetic on the digits the model writes (+ − × ÷; +/− chains), result written into the residual stream |
| answer-sanity DFA + phrase writer | a 34-state DFA reads the answer as it is written; a missing or malformed answer makes a pointer writer emit " We need to find" |
| weight-space forget | 4 fixed extra q/k dims per head hide earlier tries from later ones |
| serial tries, answer register, voter | N tries in one stream; registers keep each answer, its flags and a layer-18 probe score; a ReLU voter and a writer emit the winner |
| weight-space position reset (WPR) | an offset register and a fixed rotary table make every try see the question at the first try's distance |
| adaptive tries | compiled comparators stop at 4 tries when the vote is unanimous and confident, else extend up to 16 |
| answer-group voter | number-set registers and a fixed linear score pick among groups of equal answers |

Ladder on the frozen RL+calc host (GSM8K test, 1,319 items, T = 1, seed 7, each rung run once; 95% CIs in
`gsm_stack/README.md`):

| host alone | + calculators | + sanity + forget (S2F) | + 8 tries + WPR | + adaptive tries | + answer-group voter |
|---|---|---|---|---|---|
| 67.7 | 69.5 | 72.3 | 80.7 | 81.6 | 82.7 |

RL from the base model (Qwen3-0.6B-Base, 300 GRPO updates, GSM8K test, seed-pooled; bold = blog §5.1 rows, 6 RL seeds,
REPRODUCE §22):

| base, no RL | plain RL, no implants (P64) | RL + tools, one try | RL + serial stack | RL + serial stack + WPR | WPR hosts + adaptive tries (CW) |
|---|---|---|---|---|---|
| 28.1 | **68.3** (6 seeds) | 73.0 (3) | 74.8 (3) | 76.5 (3) | **80.4** (6) |

P64 trains on 64 prompts per update, as the CW arm does. An earlier plain-RL arm at 256 prompts per update (Pb, 2
seeds) reached 70.8; at 3 seeds CW gave 80.3 (REPRODUCE §16.3).

Without WPR, more tries add nothing (71.9 at N = 8 vs 72.3 with one try): later tries copy earlier ones. The reset in
weights gives +8.8, the same as the programmatic reset it replaces (−0.3, n.s.). On GSM-Hard the calculator matters most
(+14.5) and the answer-group voter, fit on GSM8K-train rollouts, costs 2.2 points. The earlier `gsm_push/` stack
(73.4 / 80.4) used a programmatic missing-answer swap and position reset; it is kept as a labeled reference.

## Reproducing the headline numbers

One GPU per command; `REPRODUCE.md` has the full commands, the expected values and the seed noise.

| blog number | command (from the repo root unless shown) | expected / result file | REPRODUCE |
|---|---|---|---|
| Calculator: GSM8K 52 → 53, GSM-Hard 27 → 35 (Qwen3-0.6B-Base) | `python scripts/solve_veto.py --device cuda && python scripts/run_gsm.py out/hard_calc.json --split hard --mode calc --bs 32` (and `--mode plain`, `--split test`) | 52.24 → 53.37, 27.29 → 35.03; `results/gsm_full_implant/` | §1 |
| Calculator: 4–5-digit × near 0% → 100% (Qwen3-0.6B-Base) | `PYTHONPATH=. python evidence/code/item1_calc.py --n 100 --bs 120 --out evidence/item1` | base 1% / 0% at 4×4 / 5×5 digits, implant 100% in every cell; `evidence/item1/` (earlier phase, Qwen2.5-0.5B: 4.0 → 100.0, `finetuned_phase/calc_icl/`) | §1 |
| Fine-tune with calls: GSM-Hard 13 → 23, templates 51 → 99 | `cd finetuned_phase/eyes_and_finetune && bash seedjobs.sh` | `finetuned_phase/results/reported_numbers.json` | §2 |
| Eyes: 95% vs 4% at 32 examples | `bash evidence/code/eyes_sweep3.sh` (after `eyes.py cnn` / `eyes.py data 16000`) | 94.7 vs 3.7 (4 seeds); `evidence/eyes_sweep/results/` | §3 |
| IFEval 58 → 72 | `bash ifeval/transfer/run_all.sh` | 57.86 → 72.09, +80 / −3; `ifeval/transfer/results/q06_score.txt` | §5 |
| BBH 12 tool tasks 43 → 84; ~100 examples 29 → 75 | `bbh_generic/integrate/code/run27_integ.py` (see §6.1), `bbh_generic/ft100/README.md` | 43.27 → 84.33; 29.0 → 76.3 / 73.7 | §6 |
| Call gate: 27-task 48 → 55, stray calls 59% → 4% | `for s in states probe bbh sets arc report; do bash bbh_generic/callgate/code/run_callgate.sh $s; done` | 48.36 → 54.58; 59.2% → 3.8% | §6.2 |
| Transfer to 1.7B / 4B and other families | `bash bbh_generic/transfer/code/run_transfer.sh q17` · `bash transfer/xfamily/code/drvr.sh …` | `bbh_generic/transfer/RESULTS.md`, `transfer/xfamily/` | §8 |
| Mistake detector: 52 → 56; + calculator 57 / 35 | `bash scripts/rethink_v2/queue.sh scripts/rethink_v2/jobs_hybrid.txt 4` | `results/rethink_v2/summary.json` | §9 |
| Topic blocker: catches ~99%; blocks ~4% of normal requests, fires on 23% of non-math MMLU | `bash topic_fence/run_test.sh` | `probe_text_stop`: pos 98.7, ret_fire 4.2, mmlu_fire 23.0; `topic_fence/results/q17_math_table.txt` | §10 |
| Latent write 15 → 96-99 | `bash latent_write/run_all.sh` | `latent_write/summary.md` | §11 |
| RL: GSM-Hard 44 vs 55 after 300 updates (3 seeds) | `cd rl/gsm_grpo && python train2.py --arm 2 --seed 0 --run_dir runs/long/a2_s0 --updates 300 --prompts 64 --G 8 --T 1.0 --lr 1e-6 --dev_n 800 --dev_every 25 --ckpt_every 100 --fused 1` (arm 1 = no implant), then `final_eval.py` | `rl/results/gsm_grpo/H100/final/` | §12 |
| RL: half-broken calculator, x÷ let through, +− blocked | `cd rl/wiring && bash launch.sh 0 1 40000 "" q4_a100.txt` | `rl/results/wiring/figdata/fig_competence_map.json` | §12 |
| RL: all implants attached (GSM-Hard ahead of the scrambled set in 3 of 3 seeds); mixed tasks 58 vs 45 (3 seeds) | `bash rl/library/launch/h100_LIB_s0.sh`; `bash rl/library/launch/h100_m1_mixLIB.sh`; seeds 1-2 and the 3-seed mixed rerun: REPRODUCE §23 (`checks/seeds3/`) | LIB vs LIBscr GSM-Hard 56.1 / 56.2 / 55.2 vs 51.9 / 41.6 / 51.3; mixed 57.8 vs 44.7, paired +13.1 [+8.7, +17.6] (earlier one-seed run: 68.2 vs 36.3); `rl/library/NOTEBOOK.md`, `rl/results/library/figdata/` | §12, §23 |
| RL: distilled links 15-20% stronger, +13 on GSM-Hard (3 seeds) | `cd rl/wiring && bash launch.sh 0 1 40000 "" q5_a100.txt` | GSM-Hard links on − host only +14.6 / +13.5 / +11.7 (3-seed rerun, §23); dev gap of the original runs +11 (+24 on big numbers), `rl/results/wiring/figdata/W3d_s*.jsonl` | §12, §23 |
| GSM8K stack ladder: 67.7 → 72.3 → 80.7 → 81.6 → 82.7 | `bash gsm_stack/eval/fe.sh gsm_stack/eval/lists/test3.list` (host, calc, chain2, S2F) · `fe.sh .../wpr_rung1.list` · `bash gsm_stack/eval/jobs/rung_job.sh r2 wpr_rung2.list` · `rung_job.sh r3 wpr_rung3.list` (needs the RL+calc host, `CKPT_U0=`) | 67.70, 72.25, 80.67, 81.58, 82.71; `gsm_stack/results/RL_RESULTS.md` §4, §9 | §16 |
| GSM8K from the base model (blog §5.1): RL with implants 80.4 vs plain RL 68.3 (6 RL seeds) | `bash gsm_stack/engine/run_arm.sh {P64_s0..P64_s5,CW_s0..CW_s5}`, then the §17 / §22 evals | 68.32 vs 80.44; `checks/seeds35/RESULTS.md`. Earlier arm Pb (256 prompts/update, not the blog row): 70.36 / 71.19 (`fe.sh .../seed_Pbs{1,2}.list`); CW at 3 seeds 81.12 / 81.43 / 78.24 (`rung_job.sh r2 wpr_r2cw.list`; `RL_RESULTS.md` §10) | §16, §22 |
| (older) GSM8K push, partly programmatic: 73.4 (S2F), 80.4 (N = 8) | `HOOK=stack3_hook.py bash gsm_push/final/run_stk.sh fgtw 0 test - ...` | `gsm_push/results/final/`; NON-COMPLIANT references (`gsm_stack/COMPLIANCE.md`) | §15 |
| Blog figures | `bash figures/blog/make_blog_figs.sh`; RL: `cd rl/figures && python make_blog_figs_rl.py && python make_rl45.py && python make_rl6.py`; v3 RL re-renders: `cd rl/figures/v3 && python make_rerender_v3.py && python make_new_v3.py`; review re-renders: `figures/blog/review_final/*.py` (CPU) | `figures/blog/figs/`, `rl/figures/`, `rl/figures/v3/`, `figures/blog/review_final/` | §13 |

## Repository layout

The repository is organised by component (one folder per blog section), each with its own README:

| path | contents |
|---|---|
| `micromachines/` | the package: the HDL / compiler (`hdl.py`: `Net`, `Lin`, `compile_net`), circuit specs (`specs.py`), circuits (`circuits/`: calculator, BBH tools), rethink circuits |
| `scripts/` | GSM / rethink / gate / baseline drivers and implant builders (`solve_veto.py`, `rethink_circuit_build.py`, `rethink_v2/`) |
| `ifeval/`, `bbh_generic/`, `topic_fence/`, `latent_write/`, `transfer/`, `finetuned_phase/`, `evidence/`, `math500/` | one experiment each: implant builders, eval harnesses, small result records |
| `rl/` | RL engines and launchers: `gsm_grpo/` (v2 GRPO engine, Triton), `wiring/` (trainable wiring, call-form ladder), `library/` (all tools attached, mixed tasks), `engine2/` (NVRTC megakernel, work in progress), `interp/`, `results/`, `figures/` |
| `gsm_stack/` | the single-call GSM8K implant stack, all in weights: engine (rel/v5: fast rollout, attention and tool kernels, fast trainer), RL arm launcher, test harness and lists, the implant lane's rungs and negative results, results notes, compliance note, model / data card |
| `gsm_push/` | the earlier GSM8K stack (calc v2, sanity DFA, forget, serial voter; two parts programmatic), kept for the record. Note: the blog's appendix A.3 names `gsm_push/` as the one-call stack; the all-in-weights stack of blog §4.4 and §5 is `gsm_stack/` |
| `uniform/` | the blog §5.1 / §6 rows (host / + RL / + implants / + both) on Qwen3-0.6B-Base (`final_eval/`, `rl24/` BBH and IFEval) and Qwen3-1.7B-Base (`rl28/`): eval lists, overlay code trees, launchers, per-item correctness rows (no generated text), result tables (REPRODUCE §17–§19) |
| `checks/` | later checks behind the final blog numbers, one folder each with README / RESULTS and CPU analysis scripts over the shipped text-free per-item results: `tooltext/` (text calculator tool baseline, §20), `calcimp/` (tool vs implant at 28 seeds, §21), `seeds35/` (§5.1 rows at 6 RL seeds, §22), `seeds3/` (3-seed checks, §23), `offtask/` (MMLU / ARC / HellaSwag / WinoGrande / NQ after RL, §24), `sourced/` (§4.4 numbers, §25) |
| `figures/` | blog figure scripts and rendered figures (`figures/blog/`, `figures/blog/review_final/`) |
| `results/` | small result records for the top-level GSM / rethink runs |
| `data/` | data preparation scripts (benchmarks are downloaded, not vendored) |
| `tests/` | CPU exactness tests |
| `README.md`, `REPRODUCE.md`, `DETAILS.md`, `VERIFY.md` | docs: overview, per-result repro, full notes, fresh-clone checks |

## Rebuilding the implants

Implants are code. Compiled circuits (calculator, BBH tools, IFEval rules, chain2, sanity DFA, forget, voter) are
rebuilt deterministically from `micromachines/` and the per-folder builders every time a script starts. The solved or
fitted pieces are rebuilt by one script each:

| gadget | rebuild | shipped? |
|---|---|---|
| calculator veto `runs/veto_qwen3.pt` | `python scripts/solve_veto.py --device cuda` (bit-identical, VERIFY.md) | no (rebuilt) |
| veto for bigger / other hosts | `scripts/solve_veto.py --model ...` | yes, `results/gsm_bigger/veto_*.pt`, `transfer/xfamily/results/veto_*.pt` (10-14 KB) |
| compiled IFEval implant `implant14_sd.pt` (1.23 GB) | `ifeval/full14_compiled/export.py` (~20 s) | no |
| IFEval force_fix rows `ifeval/heldout/vecs2_forcefix.pt` | `ifeval/heldout/solve_vecs2.py` (needs plain dev runs of the host) | yes, **7.4 MB** |
| IFEval compiled-v2 veto vectors `ifeval/compiled_v2/dirs.pt` | solved against the host (`ifeval/README.md`) | yes, 4.2 MB |
| mistake detector `results/rethink_v2/detector_hybrid.pt` | `scripts/rethink_v2/policy3.py` (needs ~7 GB of extracted states + vLLM generations) | yes, **7.4 MB** |
| rethink write directions `results/rethink*/write_dirs*.pt` | solved on first use (`scripts/rethink_circuit_build.py`) | yes, 1.6 MB each |
| latent-write writer banks | refit by `latent_write/run.py` on every run | no (36 MB) |
| call-gate probe `bbh_generic/callgate/results/probe_v*.pt` | `run_callgate.sh states && run_callgate.sh probe` | yes, 6 KB |
| topic-blocker probes `topic_fence/fitted/*.pt` | `topic_fence/` fit scripts | yes, 0.5 MB |
| GSM verifier probe `gsm_push/final/verifier_probe_L18.pt`, op veto `gsm_push/calc_v2/veto_op_qwen3.pt` | `gsm_push/selection/` probe fit, `gsm_push/calc_v2/solve_veto_op.py` | yes, 6 KB |
| sanity DFA tables | `san_engine.build_tables` from the tokenizer, on first use (`$DFA_CACHE`) | no |
| GSM stack rung-3 voter `vf_w.pt`, rung-4 side-net `sn_w.pt` | `gsm_stack/implants/vfeat/fitw.py`, `gsm_stack/implants/sidenet/mk_snw.py` (need split-C rollouts / states: one GPU job each) | **no, not synced for this release** (rung 3 / 4 need them; `gsm_stack/implants/README.md`) |
| GSM stack RL hosts (`gsm_stack/engine/run_arm.sh ARM`) | 300 GRPO updates per arm, 0.6-2.6 h on one B200 | no (`gsm_stack/MODEL_CARD.md`) |

## Hardware

* Every result needs **one GPU**. Runs were done on NVIDIA A100 80 GB, H100 80 GB, A40 48 GB and (the `gsm_stack/` RL arms and
  test runs) B200, bf16, greedy decoding unless stated. 0.6B-host runs fit in < 20 GB; the RL runs use 30-60 GB at the default batch (`--B_max 128` for
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
* `gsm_stack/engine/` (rel/v5, behind every `gsm_stack` number): the v2 engine plus a continuous-batching rollout loop
  (`fastroll.py`), Triton attention with the weight-space forget mask and compaction / overlap / split-KV variants
  (`attn3.py`), the calculator implants as the engine2 integer-exact tool kernel (`tk.py`, `tkdual.py`,
  `tk3_kernel.cu`, NVRTC), and a packed-sequence fast trainer (`fasttrain.py`, Triton group attention fwd + bwd in
  `tattn.py`, fused log-prob head `lmhead.cu`). The rollout fast paths were checked bitwise against the plain loop, the fast trainer against the reference trainer
  (`gsm_stack/engine/stk/KERNEL_NOTES.md`).

## What is not in the repository

* Model checkpoints: fine-tuned hosts, RL'd hosts (including every `gsm_stack` arm and its ~7 GB resume states), trained wiring. Recipes, seeds and training rows (or seeded
  generators with SHA-256) are included.
* Hidden-state dumps and per-row generation dumps (e.g. the call-gate opener states, `gsm_push` rows).
* Files over 5 MB that scripts rebuild, e.g. the latent-write writer banks (36 MB), the compiled IFEval implant tensor (1.23 GB, `ifeval/full14_compiled/export.py`,
  SHA-256 in `REPRODUCE.md`), `ifeval/heldout/vecs2.pt` (11.6 MB; its used rows are `vecs2_forcefix.pt`), the engine2
  table cache `rl/engine2/tables.pt` (118 MB, rebuilt on first use).
* Generated text and benchmark question text: result files keep per-question records (gold, prediction, correctness,
  fire flags, hashes) only.
* The full list, item by item, is `REPRODUCE.md` §14.

## License

MIT (see `LICENSE`). Benchmarks and models are downloaded from their original sources under their own licenses.
