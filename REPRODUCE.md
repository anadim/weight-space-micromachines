# Reproducing the blog post

This file maps every result section of the blog post *Weight-Space Micromachines: Brain Implants for LLMs* to the code,
data, fitted pieces, result files and exact commands in this repository, with the numbers to expect, the runtime, and which
numbers move with seeds or hardware. Each section also says what cannot be rerun from the repository alone, and why.
The per-folder READMEs have more detail; `VERIFY.md` logs what was re-run from a fresh clone.

**Ground rules.**
* No model weights are in the repository. Base models are public Hugging Face checkpoints, downloaded on first use.
  Fine-tuned hosts and RL checkpoints are **not released**; their recipes (training rows or seeded generators with
  SHA-256, scripts, hyperparameters) are. Large implant tensors that are rebuilt by a script (e.g. the 1.23 GB compiled
  IFEval implant) are given as build command + SHA-256.
* Result files keep per-question records (gold, prediction, correctness, fire flags, SHA-256 of the generated text), never
  generated text or benchmark question text. Scripts that need the full text say so.
* Run every command from the repository root unless a `cd` is shown. Paths that scripts read from the environment:
  `WSM_ROOT` (repo root; default inferred), `HF_HOME` (model cache), `RL_CURVES` (figures), `FT_HOST` / `VETO` (BBH).

## Environment

| machine | GPU | Python | torch | transformers | triton | vLLM | used for |
|---|---|---|---|---|---|---|---|
| A100 node | 1x NVIDIA A100 80 GB PCIe | 3.12.3 | 2.8.0+cu128 | 4.56.2 | 3.4.0 | 0.10.2 | GSM calculator, rethink v2, compiled IFEval, IFEval transfer, RL 64-update ablation |
| H100 pod | 2x NVIDIA H100 80 GB HBM3 | 3.12.3 | 2.8.0+cu129 | 4.56.2 | 3.4.0 | — | BBH fine-tunes and runs, RL 300-update runs, wiring, topic blocker |
| A40 pod | 2x NVIDIA A40 46 GB | 3.12.3 | 2.8.0+cu128 | 4.56.2 | 3.4.0 | 0.10.2 | IFEval headline, call-gate states, BBH curve shards, some RL arms, `VERIFY.md` clone run |
| B200 node | 8x NVIDIA B200 (one GPU per job) | 3.12 | 2.8.0+cu128 | 4.56.2 | 3.4.0 (bundled ptxas) | — | `gsm_stack` (§16), uniform rows (§17, §18), checks §20-§25 |
| A100 boxes | 1x NVIDIA A100-SXM4 40 GB per job | as the B200 row (the same code bundle and pinned versions) | | | | — | extra seeds of §21 (C36 seeds 12-27) and §22 (IFEval seeds 3-5), same commands |

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                 # torch 2.8.0, transformers 4.56.2, tokenizers 0.22.2, numpy 2.2.6, ...
pip install -r requirements-extra.txt           # vLLM 0.10.2 (rethink-v2 training generations, some baselines), lm_eval 0.4.13, langdetect 1.0.9
pip install triton==3.4.0                       # RL engine only (ships with the torch 2.8 CUDA wheels)
```
All runs are bf16 on GPU with greedy decoding unless a section says otherwise. One GPU per run; most 0.6B runs need
< 20 GB. **No result needs more than one GPU.**

## Data

```bash
bash data/download_bbh.sh                       # BBH tasks + official 3-shot CoT prompts (pinned commit)
python data/prepare_gsm.py                      # GSM8K train/test, GSM-Hard, the 8 prompt examples
python ifeval/data/prepare_ifeval.py            # google/IFEval -> ifeval/data/ifeval.jsonl (541 prompts)
python scripts/rethink_v2/make_scaled_train.py data/gsm8k/train.jsonl data/gsm8k/train_scaled.jsonl   # RL + rethink v2 (seeded)
python scripts/solve_veto.py                    # runs/veto_qwen3.pt, the calculator's veto (bit-identical rebuild, ~50 min CPU / ~5 min GPU)
```
MNIST (eyes) is downloaded by `finetuned_phase/eyes_and_finetune/eyes.py`. Our own look-alike question sets, fine-tune
rows and judge prompts are in the repo (`bbh_generic/team*/data/`, `bbh_generic/integrate/data/`, `ifeval/heldout/data/`,
`topic_fence/data/`) or rebuilt byte-identically by the seeded generators named below.

## Index

| blog section | repo | section here |
|---|---|---|
| §1.2 First implant: a calculator | `micromachines/`, `scripts/run_gsm.py`, `results/gsm*`, `evidence/`, `tests/` | [1](#1-first-implant-the-calculator) |
| §1.3 Teaching the model to call an implant | `finetuned_phase/`, `bbh_generic/ft100/` | [2](#2-teaching-the-model-to-call-an-implant-fine-tuning) |
| §1.4 Second implant: give Qwen eyes? | `finetuned_phase/eyes_and_finetune/`, `evidence/eyes_sweep/` | [3](#3-the-eyes-implant) |
| §4.1 The implant discovery loop, and how it can cheat | `bbh_generic/PROTOCOL.md`, retired `micromachines/circuits/` | [4](#4-a-trap-implants-can-overfit-the-test-look-alike-data) |
| §4.2 Climbing IFEval with implants | `ifeval/transfer/` (headline run), `ifeval/heldout/`, `ifeval/full14_compiled/`, `ifeval/baselines/` | [5](#5-climbing-ifeval-14-rule-implants) |
| §4.3 Climbing BIG-Bench Hard with implants | `bbh_generic/` (`integrate/`, `callgate/`, `ft100/`, `nl_control/`) | [6](#6-climbing-big-bench-hard-8-tools-call-fine-tune-call-gate) |
| §4.2 / §4.3 figures "Climbing IFEval, one rule implant at a time", "Climbing BIG-Bench Hard, one implant at a time" | `ifeval/transfer/`, `bbh_generic/transfer/` | [7](#7-add-one-implant-curves) |
| §6 Do implants transfer across models? | `results/gsm_bigger/`, `ifeval/transfer/`, `bbh_generic/transfer/`, `transfer/xfamily/`, `ifeval/xfamily/` | [8](#8-transfer-to-bigger-qwen-models-and-other-families) |
| §2.1 A mistake detector | `micromachines/rethink_circuit_v2.py`, `scripts/rethink_v2/`, `results/rethink_v2/` | [9](#9-mistake-detector--rethink) |
| §2.2 A topic blocker | `topic_fence/`, `topic_fence/inweights/`, `topic_fence/baselines/` | [10](#10-topic-blocker) |
| §2.3 Writing inside the model's thoughts | `latent_write/` (+ `v2/`, `causal/`) | [11](#11-latent-write) |
| §3 RLing on the model with one implant (and the all-implants runs of §5.2) | `rl/` (`gsm_grpo/`, `wiring/`, `library/`, `interp/`, `figures/`; `engine2/` = WIP engine) | [12](#12-rl-with-implants-gsm8k-grpo) |
| Figures | `figures/blog/`, `figures/blog/review_final/`, `rl/figures/`, `rl/figures/v3/` | [13](#13-figures) |
| §4.4 Climbing GSM8K with implants (earlier, partly programmatic stack) | `gsm_push/` | [15](#15-gsm8k-push-single-call-implant-stack-on-the-rlcalc-host) |
| §4.4 Climbing GSM8K with implants (single-call stack, all in weights); §5.1 "implants first, then RL" | `gsm_stack/` | [16](#16-gsm8k-single-call-implant-stack-gsm_stack) |
| §5.1 Uniform rows (host / + RL / + implants / + both) on 0.6B | `uniform/` (+ `gsm_stack/`) | [17](#17-uniform-rows-on-qwen3-06b-base-blog-51) |
| §6 The same rows on Qwen3-1.7B-Base | `uniform/rl28/` | [18](#18-qwen3-17b-base-same-implants-blog-6) |
| Settings of every blog result (host, prompt, decoding, seeds, tok/q) | — | [19](#19-settings-table) |
| Text calculator tool baseline (C35) | `checks/tooltext/` | [20](#20-text-calculator-tool-baseline-claim-c35-checkstooltext) |
| Equal-footing text tool vs exact calculator implant, 28 seeds (C36) | `checks/calcimp/` | [21](#21-equal-footing-text-tool-vs-exact-calculator-implant-at-28-seeds-claim-c36-checkscalcimp) |
| §5.1 rows at 6 RL seeds (GSM8K, IFEval) | `checks/seeds35/` | [22](#22-blog-51-uniform-rows-at-6-rl-seeds-checksseeds35) |
| Three-seed checks of one-seed claims (§3.4, §5.1, §5.2) | `checks/seeds3/` | [23](#23-blog-three-seed-checks-checksseeds3) |
| Off-task benchmarks after RL, implants attached (MMLU, ARC, HellaSwag, WinoGrande, NQ) | `checks/offtask/` | [24](#24-off-task-benchmarks-after-rl-implants-attached-checksofftask) |
| §4.4 and related sourced numbers | `checks/sourced/` | [25](#25-sourced-blog-numbers-44-and-related-checkssourced) |


## 1. First implant: the calculator

Blog §1.2: "52% -> 53% on GSM8K, 27% -> 35% on GSM-Hard" (frozen Qwen3-0.6B-Base; "26% to 33%" on the part never tuned
on); "every pair of numbers up to 59, every signed pair from −40 to 39 and 12k random pairs, for all four operations ...
0 mistakes"; on a frozen Qwen3-0.6B, ">4 digit multiplication accuracy from near zero to 100%" (operand sweep below);
figure "Accuracy as the numbers get longer". Also here, not quoted in the final blog: GSM-Hard +8 / +6 / +1 for
0.6B / 1.7B / 4B; multiplication 4% -> 100% on a frozen Qwen2.5-0.5B (earlier phase); the 0..199 sweep (160k cases).

```bash
python scripts/solve_veto.py --device cuda                                   # runs/veto_qwen3.pt (~5 min GPU, ~50 min CPU)
python scripts/run_gsm.py out/test_plain.json --split test --mode plain --bs 32
python scripts/run_gsm.py out/test_calc.json  --split test --mode calc  --bs 32   # and --split hard
python scripts/reduce_outputs.py out/test_calc.json out/test_calc_reduced.json   # compare with results/gsm_full_implant/test_calc.json
# bigger hosts (veto re-solved per host; shipped in results/gsm_bigger/)
python scripts/solve_veto.py --model Qwen/Qwen3-1.7B-Base --out results/gsm_bigger/veto_q17.pt --device cuda
python scripts/run_gsm.py out/q17_hard_calc.json --model Qwen/Qwen3-1.7B-Base --split hard --mode calc --veto results/gsm_bigger/veto_q17.pt --bs 32
#   same for --mode plain, --split test, and Qwen/Qwen3-4B-Base with veto_q4.pt
python results/gsm_bigger/reanalysis/gsm_reanalysis.py   # CPU: CIs, "201+" (shots not picked there) and clean-label subsets
# exactness of the circuit (no model / with the model)
python tests/test_calc_circuit.py cpu        # exhaustive 0..59 x 4 ops, signed -40..39, 12k random, malformed calls
python tests/test_calc_circuit.py cpu 200    # exhaustive 0..199 (160k cases), ~12 min on CPU
python tests/test_calc_tokens.py 200 0       # calc_s4 through the frozen host
# operand-size sweep (figure "Accuracy as the numbers get longer"), ~12 min on one A100
PYTHONPATH=. python evidence/code/item1_calc.py --n 100 --bs 120 --out evidence/item1
# the frozen Qwen2.5-0.5B multiplication test (earlier phase, CPU)
cd finetuned_phase/calc_icl && python calc_icl.py 100
```

Expected (n = 1,319 per set; `results/gsm_full_implant/`, `results/gsm_bigger/`, `results/gsm_bigger/RESULT_lines.txt`):

| host | GSM8K plain -> calc | GSM-Hard plain -> calc |
|---|---|---|
| Qwen3-0.6B-Base | 52.24 -> 53.37 | 27.29 -> 35.03 (201+: 26.2 -> 33.2) |
| Qwen3-1.7B-Base | 73.09 -> 72.33 | 45.41 -> 51.02 |
| Qwen3-4B-Base | 84.31 -> 84.69 | 58.98 -> 60.35 |

Operand sweep (`evidence/item1/`): implant 100% in every cell; base multiplication 100 / 69 / 8 / 1 / 0 (1x1..5x5 digits),
mean 24.4%. Qwen2.5-0.5B (`finetuned_phase/results/calc_icl_100.json`): bare multiplication 4.0 -> 100.0, word problems
46 -> 98. Runtime: one GSM arm 15-40 min on an A100/H100 (bs 32); `solve_veto.py` ~50 min on CPU.
Noise: greedy, but bf16 batching changes ~4-6% of problems between stacks; the earlier research harness gave 51.93 ->
54.06 / 26.99 -> 34.50 for the same configuration (`VERIFY.md`). GSM-Hard gains have paired 95% CIs of about +/-1 pp
(`results/gsm_bigger/reanalysis/gsm_reanalysis.md`). The blog's "every pair up to 59, signed −40..39, 12k random" is the
default circuit test, `python tests/test_calc_circuit.py cpu` (N = 60). A stronger sweep up to 199 (160k cases) is
`python tests/test_calc_circuit.py cpu 200`.
The 0..199 run (2026-11-22, CPU) is `python tests/calc_sweep_0_199.py . cpu 200` (it imports the shipped test, so
`run()` / `calc_ref` / tokenisation are the same, and splits the sweep by operation): 160,000 cases (every a, b in 0..199
for + − × ÷), 160,000 correct, 0 wrong, 0 non-one-hot/unterminated steps; x/0 gives empty output, quotients truncated to 6
decimals; 530 s on 24 threads (`results/verify/calc_sweep_0_199.log`, per-op counts and file hashes; compiled-weights
sha256 `34ffe636…`). The unmodified `tests/test_calc_circuit.py cpu 200` also passes
(`results/verify/test_calc_circuit_cpu_200.log`, 749 s). This checks the circuit's arithmetic with the tool-call symbol
framing, not the model end to end.


## 2. Teaching the model to call an implant (fine-tuning)

Blog §1.3: three fine-tunes of Qwen2.5-0.5B on the same GSM8K solutions (no calls / calls filled by the model / calls filled
by the calculator implant): GSM8K ties (~33-35%), GSM-Hard ~13 -> 23%, unseen arithmetic templates ~51 -> 99%. (Blog §4.3:
"about 100 examples ... 75% vs 29%" on BBH-style questions, section 6.3 here.)

Code: `finetuned_phase/eyes_and_finetune/` (snapshot of the earlier phase; `ft.py`, `gen_eval.py`, `seedjobs.sh`; see
`finetuned_phase/README.md` "Reproduce"). The BBH 100-example study is `bbh_generic/ft100/` (section 6.3).

```bash
cd finetuned_phase/eyes_and_finetune
cp ../../data/gsm8k/train.jsonl data/train.jsonl
bash seedjobs.sh            # 3 fine-tunes x 5 seeds (ft.py) + evals (gen_eval.py); see finetuned_phase/README.md
```
Expected (5 seeds, mean; `finetuned_phase/results/reported_numbers.json`): GSM8K 33.6 / 35.2 / 35.1, GSM-Hard 13.4 / 14.9
/ **23.3**, bare arithmetic templates 50.8 / 54.5 / **99.0** (plain / calls, model fills / calls, implant fills).
Runtime: ~10-20 min per fine-tune + eval on one A100. Seed noise: +-1-2 pp per arm (std in the JSON).
**Not reproducible as shipped:** the raw per-seed files of this 3-way fine-tune sit on a machine that was offline while
the repo was assembled; the numbers are in `reported_numbers.json` marked `verified_here: false`.

## 3. The eyes implant

Blog §1.4: with 32 worked solutions + the eyes implant, 95% vs 4% without eyes; plain fine-tuning needs ~4,000 solutions for 93%
and ~16,000 to match.

```bash
cd finetuned_phase/eyes_and_finetune
python eyes.py cnn && python eyes.py data 16000            # MNIST CNN (421,642 params) + question sets
bash ../../evidence/code/eyes_sweep3.sh                    # n = 8 and 32 solutions, implant on/off, seeds 0-3
python ../../evidence/code/agg_eyes.py ../../evidence/eyes_sweep/results/summary.txt
```
Expected (`evidence/eyes_sweep/results/`, 4 seeds): n=32 **94.7** (89.1-97.2) vs 3.7; n=8 78.5 (76.2-81.4) vs 4.0. Plain
fine-tune at 4k rows 93.26, 16k rows 97.52 vs implant 97.67 (5 seeds, `finetuned_phase/results/`).
Runtime: ~18 min per fine-tune + eval on one A100. Seed noise: large at n = 32 (implant arm 89-97 across seeds).
**Not reproducible:** the single-seed 8-example point of an earlier draft (88%) used an unrecorded recipe and did not
reproduce (4-seed mean 78.5 is what the blog uses); eyes seed 0 raw files are on the offline machine.


## 4. A trap: implants can overfit the test (look-alike data)

The retired first pass (nine BBH-specific implants debugged on BBH test questions; `micromachines/circuits/`,
`scripts/run_bbh.py`, `results/bbh/`, `tests/`) is kept as the overfit example: `DETAILS.md` "Benchmark-derived knowledge
inside the implants" names the BBH vocabulary inside `track_nat`, `logic_nat`, `time_n`. The fixed loop (build on
look-alike questions, independent judges that paraphrase / swap names / reorder, "name a use outside the benchmark") is
`bbh_generic/PROTOCOL.md` (rules 10-11), with every judge report in `bbh_generic/team*/JUDGE_*.md` and the look-alike
generators in `bbh_generic/team*/` (see the BBH section). Commands for the retired numbers: `DETAILS.md` "Quickstart"
(`scripts/run_bbh.py`, `results/summarize.py`).


## 5. Climbing IFEval: 14 rule implants

Blog §4.2 claims backed here: 58% -> 72% prompt-level strict on all 541 IFEval prompts (fixed 80 / broke 3), silent prompts
byte-identical (324/324); figures "Rule implants on IFEval", "Each rule, one by one", "Climbing IFEval, one rule implant
at a time"; thinking mode 59%; transfer to 1.7B / 4B and a SmolLM2 port (blog §6). Also here, not quoted in the final blog:
on the 299 "odd" prompts never used for building 58% -> 71% (+42 / -1); readers catch ~40-75% of rephrasings (misses fail
safe); all 14 rules compiled into plain weights (~9,300 ReLU units), byte-identical to the reference on 541/541; transfer
gains +9 / +4 points (1.7B / 4B).

**Environment.** Python 3.12.3, torch 2.8.0+cu128, transformers 4.56.2, lm_eval 0.4.13 (IFEval checkers), langdetect 1.0.9.
No vLLM. Host `Qwen/Qwen3-0.6B` (chat model, thinking off), bf16, greedy, max_new 2048. GPUs: the blog's headline run, the
compiled implant, transfer and curve on one A100 80GB PCIe; an earlier headline run on one NVIDIA A40 46 GB (5.1b). Peak
memory < 10 GB (0.6B), ~20 GB (4B).

**Data.**
```bash
python ifeval/data/prepare_ifeval.py          # google/IFEval -> ifeval/data/ifeval.jsonl (541 prompts)
python ifeval/data/prep_transfer.py           # optional: IFBench (needs allenai/IFBench at commit 488918665d53)
```
Our own dev / fresh / ood / mentions prompts are in `ifeval/heldout/data/` (generated by `gen_data.py`, seeds 0/1/2).
Split: the 242 "even" IFEval keys had their instruction sentences reworded into our dev data; the 299 "odd" keys were
never opened while building (`ifeval/heldout/NOTES.md`). Every number is given on ALL 541 and on ODD 299.

### 5.1 Headline: 14 rules, frozen Qwen3-0.6B (A100, seeded scorer; the blog's numbers)

```bash
bash ifeval/transfer/run_all.sh            # step 2 for q06 = plain + implant on all 541, 4 shards, bs 24, then score_t.py
```
Expected (`ifeval/transfer/results/q06_score.txt`, `results/transfer_tables.md`): ALL 541 prompt-strict **57.86 -> 72.09**
(fixed 80 / broke 3), inst-strict 67.63 -> 79.62; silent prompts 324/324 byte-identical; ODD 299 (never used for
building) **57.53 -> 71.24 (+42 / -1)**; EVEN 242 58.26 -> 73.14 (+38 / -2). Per-rule chart ("Each rule, one by one")
= the ALL per-rule instruction-strict table in `results/transfer_tables.md` (0.6B column): no_comma 31 -> 62 / 66,
forbidden 23 -> 42 / 49, paragraphs 13 -> 21 / 27, nth paragraph 4 -> 11 / 12, two responses 11 -> 19 / 24, bullets
22 -> 29 / 31, repeat 26 -> 34 / 41, quotes 33 -> 36 / 41, end phrase 24 -> 26 / 26, lowercase 28 -> 31 / 39. Each bar is a
rule's own instructions with all 14 rules on (not one rule alone). Text-free per-prompt records:
`ifeval/transfer/results/reduced/q06_ifeval_{plain,imp}.json`. Runtime: ~1.5 h per 541-prompt implant run per GPU process.

### 5.1b Earlier headline run (A40, unseeded scorer)

```bash
cd ifeval/heldout && ln -s ../../data/ifeval.jsonl data/ifeval.jsonl
J=(); for i in 0 1 2 3; do J+=("ifeval.jsonl ifeval_plain.s$i.jsonl plain $i/4" "ifeval.jsonl ifeval_imp.s$i.jsonl implant $i/4"); done
./queue.sh 4 "${J[@]}"                                                     # 8 shard jobs, bs 24, one GPU
./final_score.sh                                                            # ODD / EVEN / ALL + IFBench
```
Expected (`ifeval/heldout_results/final_score.out`): ALL 541 58.23 -> 72.83 (+82 / -3); ODD 299 56.19 -> 71.91 (+49 / -2);
silent 323/323 byte-identical; IFBench 295/295 silent identical. Same implant, different GPU (bf16 text differs, 8.6).
The blog's IFEval bar figures ("Rule implants on IFEval", "Each rule, one by one") were first drawn from this run
(`figures/blog/make_figs3.py`); the add-one-rule curve uses the A100 runs of section 8.2 (`ifeval_curves.json`);
`figures/blog/README.md` maps each figure to its data.
Text-free per-prompt records: `ifeval/heldout_results/teamd_ifeval_{plain,imp}.json`.
Reader recall on the judges' reworded phrasings (~40-75% per rule) and the 14 PASS verdicts: `ifeval/heldout/NOTES.md`,
`ifeval/heldout/JUDGE_*.md`; detection only: `python detect_test.py data/dev.jsonl`.

### 5.2 Compiled into plain weights (A100)

```bash
bash ifeval/full14_compiled/run_all.sh     # build + export implant14_sd.pt, circuit.py rerun, compiled run (shadow), compare, stress
```
Expected: 9,304 ReLU units / 77,887 nonzeros (`build_test.py`); compiled vs `circuit.py` 541/541 byte-identical; shadow mode 0
mismatches over 26,242 steps; silent 324/324 identical to plain; prompt-strict 57.86 -> 72.09 (all), 57.53 -> 71.24 (odd).
Weights not in the repo: `implant14_sd.pt` 1.23 GB, sha256 `e85650adb761df2a292c5d1ce4aa836ddb9b2a32f3d2dbfc77b37a849787c200`
(rebuilt by `export.py` in ~20 s). Runtime: compiled run with shadow ~1.7 h, reference rerun ~1.4 h, stress tests hours on CPU.

### 5.3 Thinking-mode baseline and silence (A40)

```bash
bash ifeval/baselines/run_baselines.sh
```
Expected: thinking ON, answer after `</think>`: ALL 59.33 / ODD 59.20 (A40; vs 58.23 plain / 72.83 implants on the A40,
57.86 / 72.09 on the A100); 50/541 hit the cap.
Silence: 0/1000 wikitext passages, 1/600 code files latch a rule. ~2.5 h + ~30 min.


## 6. Climbing BIG-Bench Hard: 8 tools, call fine-tune, call gate

Blog sections: §4.3 "Climbing BIG-Bench Hard with implants" (including the ~100-example claim), §6 "Do implants
transfer across models?" (BBH part), and the §4.3 figure "Climbing BIG-Bench Hard, one implant at a time".

**Environment.** Python 3.12.3, torch 2.8.0 (CUDA 12.8/12.9 wheels), transformers 4.56.2 (`requirements.txt`). GPUs used:
NVIDIA H100 80GB HBM3 (fine-tunes, most BBH runs) and NVIDIA A40 48GB (add-one curve shards, call-gate state collection,
some BBH shards). No vLLM: every BBH number here comes from the HF greedy loop with the implant hooks (`run27_integ.py`,
`cg_run27_gate.py`). Hosts: `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base` (Hugging Face).

**Data.** `bash data/download_bbh.sh` (official BBH tasks + 3-shot CoT prompts). All training rows are shipped or
regenerated byte-identically: `bbh_generic/integrate/data/train_union_v2.jsonl` (432 rows; `make_v2_rows.py` +
`make_union_v2.py` reproduce it byte-identically on CPU; the 100 no-call rows are the frozen base's own greedy answers,
`make_nocall.py`), `bbh_generic/ft100/sub/` (32/100/300-row subsamples), the NL-control rows (`nl_control/code/make_nl.py`,
byte-identical, `NL_ROWS_SHA256.txt`). The calc veto: `python scripts/solve_veto.py --out runs/veto_qwen3.pt`
(`--model Qwen/Qwen3-1.7B-Base|4B-Base --device cuda` for the bigger hosts).

**Not released:** every fine-tuned host (0.6B v2, 1.7B / 4B v2, ft100 and NL-control checkpoints). Each is rebuilt by a
full fine-tune with `bbh_generic/teamA/ft.py` (lr 2e-5, 6 epochs, tok_budget 8000, seed 0 for the v2 hosts: 156 steps,
~90 s for 0.6B, 74 s for 1.7B, 243 s for 4B on one H100).

### 6.1 Eight tools on one fine-tuned 0.6B host (v2, 432 rows)

```bash
cd bbh_generic/integrate/code
python make_v2_rows.py ../data/multi_v2.jsonl ../data/c_passive_v2.jsonl && python make_union_v2.py ../data ../data/train_union_v2.jsonl
(cd ../../teamA && python ft.py --data ../integrate/data/train_union_v2.jsonl --out ../integrate/runs/ft_int_v2 --lr 2e-5 --epochs 6 --tok_budget 8000 --seed 0)
python test_integ.py ../out/test_integ_v2_ft.json --v2 --model ../runs/ft_int_v2          # exactness / silence checks
python run27_integ.py ../out/b27_base_notools.json --tools none --max_new 600
python run27_integ.py ../out/b27_v2ft_notools.json --model ../runs/ft_int_v2 --tools none --max_new 600
python run27_integ.py ../out/b27_v2ft_tools.json   --model ../runs/ft_int_v2 --tools v2   --max_new 600
python eval_integ.py ../out/sets_v2ft_tools.json --model ../runs/ft_int_v2 --tools v2     # own fresh / held-out sets
python report_v2.py ../out
```

Expected (blog "Eight implants on BIG-Bench Hard", "All 27 BIG-Bench Hard tasks"): 12 tool tasks **43.27 → 84.33** (FT alone 38.10);
15 no-tool tasks 39.15 → 19.58 (FT alone 30.01; 59.2% of their answers call a tool); 27 tasks 40.98 → 48.36.
Team A judges' hardest rewordings (round 3): base 3/16 → 7/16. Shipped records: `bbh_generic/integrate/results/` and
`bbh_generic/transfer/results/q06_*.json.gz`; `python report_v2.py ../results` prints the tables on CPU.
Runtime: each 27-task arm is 6,511 greedy decodes of up to 600 tokens (the original runs split arms into task shards across two GPUs; not timed end to end).

### 6.2 The call gate

```bash
export FT_HOST=bbh_generic/integrate/runs/ft_int_v2 VETO=runs/veto_qwen3.pt
for s in states probe bbh sets arc report; do bash bbh_generic/callgate/code/run_callgate.sh $s; done
```

Expected: 27-mean 48.36 → **54.58**, 12 tool tasks 85.63, 15 no-tool 19.58 → **29.74**, stray calls on the 15 59.2% →
**3.8%**; Team B meeting scheduling (own reworded set) 78 → **48**; ARC-Easy 68.25 → 70.0. CPU-only check from shipped
files: `cg_report.py` / `cg_silence.py` (commands in `bbh_generic/callgate/README.md`); the probe refits from the shipped
`states{2,3,4}.pt` to within 5e-6 (`CG_LAM=0.1 python cg_train_probe.py ...`).

### 6.3 "About 100 examples" and the plain fine-tune control

```bash
# bbh_generic/ft100/README.md ("Reproduce"): 32/100/300-row fine-tunes, tools on/off, fresh test / held-out / real BBH
# bbh_generic/nl_control/README.md ("Reproduce"): same rows as plain worked solutions, tools off
```

Expected: Team A at 100 rows, fresh test with tools 76.3 / 73.7 (seeds 0 / 1) vs base 29.0 (blog "75% vs 29%"); unseen
styles (oos2) with tools 65.0 / 64.4 vs base 18.9 (blog "65% vs 19%"); call rate 97-100% already at 32 rows. Team B
12k-row host (ft3) without tools: real BBH 95.7, reworded held-out 26.7, with tools 87.3 (96% / 27% / 87%; earlier draft,
not in the final blog). NL control, Team A real BBH 39.9 / 56.5 / 55.1 / 72.6 (n 32 / 100 s0 / 100 s1 / 300) vs tool
arm 59.5 / 76.5 / 70.3 / 94.4. Each ft100 / NL arm: one fine-tune (minutes) + evals; the 16 NL jobs took ~6 h on one
shared H100.

### 6.4 Seed and run noise

* Fine-tune seeds (ft100 / NL n = 100, two seeds): Team A fresh-test with tools 76.3 vs 73.7, real BBH 76.5 vs 70.3; Team
  B held-out with tools 51.3 vs 64.7; NL Team A held-out 37.2 vs 47.8, NL Team B real BBH 89.2 vs 82.1. Every other ft100 /
  NL / v2 / 1.7B / 4B number is a single fine-tune seed.
* Greedy bf16 decoding depends on batch size and GPU: rerunning the same v2 host with another batch size left 87-92% of
  BBH transcripts identical (call-gate silence check); expect 27-task means within about ±1 point. BBH itself was run once
  per arm (protocol), so there are no repeated BBH runs to average.
* The no-call rows (`make_nocall.py`) are greedy outputs of the base; regenerating them may differ in a few rows (the
  shipped file is the one used).

### 6.5 Not reproducible from the repo alone

* The fine-tuned hosts are not shipped (rebuild with the commands above; a rebuilt host will differ in bf16 noise).
* `report_v2.py`'s per-tool composition analysis and `score_nl.py`'s truncated-budget re-score read the generated text,
  which is not shipped (reduced records only); their outputs are in `integrate/TABLES_v1_vs_v2.md` and
  `nl_control/results/scores_nl.json`.
* The 0.6B "frozen + Team C" 27-task row is an older run (`ft100`) with per-question rows for 22 of 27 tasks only.


## 7. Add-one-implant curves

### 7.1 IFEval: add one rule at a time (blog figure "Climbing IFEval, one rule implant at a time")

Commands, expected numbers and noise: section 8.2 (`bash ifeval/transfer/run_all.sh` runs the k = 0..14 curve on each host;
`python ifeval/transfer/plot_curve.py` draws it). The curve values are in `ifeval/transfer/results/ifeval_curves.json`.
The order of rules is fixed in `ifeval/transfer/curve2.py` (it was chosen from Team D's own FRESH-set runs, which are
not in the repo, so the order itself is not re-derived).

### 7.2 BBH: add one tool at a time, all 27 tasks (blog figure "Climbing BIG-Bench Hard, one implant at a time")

```bash
bash bbh_generic/transfer/code/run_transfer.sh vetoes
bash bbh_generic/transfer/code/run_transfer.sh ft
bash bbh_generic/transfer/code/run_transfer.sh checks
bash bbh_generic/transfer/code/run_transfer.sh q17
bash bbh_generic/transfer/code/run_transfer.sh q4
FT_HOST=bbh_generic/integrate/runs/ft_int_v2 bash bbh_generic/transfer/code/run_transfer.sh curve
python bbh_generic/transfer/code/report_transfer.py bbh_generic/transfer/results q17_base q17_ft_tools q4_base q4_ft_tools   # CPU, shipped
python bbh_generic/transfer/code/curve_table.py bbh_generic/transfer/results                                                  # CPU, shipped
```

Expected, curve (27 / 12 / 15 means): 33.61 / 38.10 / 30.01 (no implants, fine-tuned) → +interval 33.31 / 40.60 → +track
39.33 / 59.90 → +order 43.51 / 71.63 → +truth 43.61 / 72.47 → +walk 44.08 / 75.00 → +Team C passive 48.36 / 84.33 / 19.58.
Bigger hosts (27-mean base → FT + 8 tools; 12-task): 1.7B 52.08 → 65.86 (96.37); 4B 71.14 → 70.98 (97.77; the 15 no-tool
tasks drop 61.79 → 49.54 from the call fine-tune). Frozen + Team C passive tools only (no fine-tune): 0.6B 42.95, 1.7B
52.70, 4B 71.63.


## 8. Transfer to bigger Qwen models and other families

Moving an implant re-solves only its two ends (reader for the tokenizer, write directions / veto against the new LM head);
the circuit is unchanged.

### 8.1 Calculator on Qwen3-1.7B / 4B

Section 1 (`results/gsm_bigger/`): GSM-Hard +7.7 (0.6B), +5.6 (1.7B), +1.4 (4B).

### 8.2 IFEval rules on Qwen3-1.7B / 4B (and the add-one-rule curve)

```bash
bash ifeval/transfer/run_all.sh            # re-solves only force_fix per host (from its own answers to OUR dev prompts)
python ifeval/transfer/plot_curve.py       # blog figure "Climbing IFEval, one rule implant at a time"
```
Expected (ALL 541, seeded scorer): 0.6B 57.86 -> 72.09 (+80 / -3), 1.7B 68.76 -> 78.19 (+55 / -4), 4B 80.41 -> 84.84 (+29 / -5);
silent prompts byte-identical on every host. Curve values in `ifeval/transfer/results/ifeval_curves.json`. 1-2 GPU-days total.

### 8.3 BBH tools on Qwen3-1.7B / 4B

Section 7.2 (`bash bbh_generic/transfer/code/run_transfer.sh q17|q4`): 12 tool tasks 1.7B 55.73 -> 96.37, 4B 82.83 -> 97.77;
27-task mean 1.7B 52.08 -> 65.86, 4B 71.14 -> 70.98 (the call fine-tune costs the 15 no-tool tasks 61.79 -> 49.54).

### 8.4 Other families: calculator and BBH arithmetic tools

Blog: SmolLM2 and granite take the calculator as is; Falcon3 never triggers it; chunked-digit tokenizers can't attach.
Code and results: `transfer/xfamily/` (README has the commands).

```bash
M=HuggingFaceTB/SmolLM2-1.7B; P=sm2
python scripts/solve_veto.py --model $M --out transfer/xfamily/results/veto_$P.pt --device cuda
bash transfer/xfamily/code/drvr.sh $PWD $M transfer/xfamily/results/veto_$P.pt $P 24      # 4 GSM arms -> out/x_*.json
bash transfer/xfamily/code/bbhx.sh $M transfer/xfamily/results/veto_$P.pt $P              # BBH base vs +Team C
```
Expected: GSM-Hard SmolLM2-1.7B 13.95 -> 19.86, granite-3.3-2b 23.20 -> 30.40, Falcon3-3B 41.39 -> 41.39 (outputs
identical: never fires); GSM8K 32.75 -> 34.65 / 50.57 -> 50.80 / 68.99 -> 68.99. BBH multistep 18.4 -> 95.6 / 32.8 ->
97.2 / 75.6 -> 99.6; granite boolean regresses 81.6 -> 78.0. LFM2: CANNOT_ATTACH. Runtime 20-40 min per GSM arm (A40/H100).
One run per arm (greedy). The IFEval port to SmolLM2-1.7B-Instruct and the within-Qwen transfer of BBH / IFEval are in
their own sections.

### 8.5 Other families: IFEval rules on SmolLM2-1.7B-Instruct (A40)

```bash
bash ifeval/xfamily/run_sm2i.sh
```
Expected ODD: 48.49 -> 60.54 (v1, constants-only port; headline), 62.54 (v2.1, post-diagnosis); silent 164/164.

### 8.6 Determinism and scorer noise (IFEval, all hosts)

Greedy decoding is deterministic on one GPU type (reference vs reference 541/541; the curve's
filler rows 268/268 and 250/250). Across GPU types bf16 text differs: the same 0.6B implant gives 58.23 -> 72.83 on the A40
and 57.86 -> 72.09 on the A100. The official scorer is not order-independent: keys 1122 / 1129 (`letter_frequency`) draw a
missing kwarg from Python `random`, and `langdetect` can flip key 1129: +-1 prompt (+-0.2-0.3 pp). `ifeval/transfer/score_t.py`
and `reduce_seeded.py` seed per key; re-scoring the A40 records gives 58.04 -> 72.83 (+82 / -2).


## 9. Mistake detector / rethink

Blog §2.1: AUROC 0.85 (GSM8K) / 0.84 (GSM-Hard); GSM8K 52 -> 56, GSM-Hard 27 -> 29 with the detector; calculator + detector
35% GSM-Hard (34% calculator alone) / 57% GSM8K; beats a random trigger at the same rate by about 1.5 points (GSM8K) and 1
(GSM-Hard); rethinking every time is up to 2 points better but spends about a third more tokens. Not quoted in the final
blog: fires on 66-75% of problems.
Commands: `DETAILS.md` "Quickstart" block, lines starting `python scripts/rethink_v2/...` (vLLM generation of ~52k labelled
training states, `policy3.py` fit, `queue.sh jobs_{baselines,hybrid}.txt`, `summarize_v2.py`, `auroc_eval.py`,
`check_v2.py`). Expected (`results/rethink_v2/summary.json`): GSM8K 52.01 / gated 55.65 / calc 53.30 / calc+gated 56.71
/ calc+always 57.54; GSM-Hard 26.61 / 29.04 / 34.34 / 35.18 / 37.15; AUROC `results/rethink_v2/auroc_{test,hard}.json`
0.849 / 0.84. Runtime: state collection ~2 h (vLLM + HF prefill), detector fit minutes, each final arm 1-2.5 h on one
A100. Noise: gated vs always differ by 0.5-2 pp (CIs in `summary.json`); the blog example is GSM8K test #755
(`figures/blog/figs/anim/detector_v2_trace.json`).
Not scripted: the "of the 28 fired-and-fixed GSM8K problems, ~16 plan errors / 5 arithmetic slips / 7 unclassified"
split is a manual reading of those 28 outputs (project notes); the outputs themselves are not released (text).


## 10. Topic blocker

Blog §2.2: catches ~99% of requests it should block, rarely blocks prompts that only mention math; blocks ~4% of normal,
non-math requests (`probe_text_stop` ret_fire 4.2) and fires on 23% of non-math MMLU questions (mmlu_fire 23.0); a small
text classifier blocks 91% and fires on 2% of MMLU (`topic_fence/baselines/`); a system prompt gets ignored, a strict one blocks most normal requests
(85.5%), a keyword list can't separate doing from mentioning; the reply monitor fires on all drift prompts, 17% still get
some math out. Not quoted in the final blog: near-misses ~1%; the LLM judge's block rate on normal requests, 9.5% vs 5.5%
for the plain model (row `ret`).
Commands: `topic_fence/README.md` "Reproduce" (dev generation with Qwen2.5-7B, probes, monitors, `run_test.sh`), then the
compiled-in-weights version `topic_fence/inweights/README.md`. Expected: `topic_fence/results/q17_{math,ml}_table.txt`;
compiled == emulated on 480/480 dev replies, silence 1998/1998 rows, and every test metric identical
(`topic_fence/inweights/results/`). Runtime ~1.5 h per fence on one H100 (generation + Qwen2.5-7B judge). Greedy; the
dev set cannot be regenerated row for row (LLM-written), so the fitted probes are shipped in `topic_fence/fitted/`.


## 11. Latent write

Blog §2.3: (23*7)+58 with no CoT 15% -> 96-99% (any layer <= 12), written wrong value followed about 97%, word problem
3% -> 98%; writing only from that point forward does nothing (13%, same as no implant). Not quoted in the final blog: 70%
for a text tool line.
```bash
bash latent_write/run_all.sh                    # six compiled cells -> latent_write/out/ (+ summary.md, figure)
python latent_write/summarize.py                # tables from the shipped results/ (no model)
PYTHONPATH=latent_write LW_OUT=latent_write/causal/results python latent_write/causal/causal.py muladd mid
PYTHONPATH=latent_write LW_OUT=latent_write/causal/results python latent_write/causal/kvrewrite.py muladd mid 0,4,8,12
bash latent_write/v2/queue1.sh                  # optional: two values, chains, value types (Python reference engine)
```
Expected (`latent_write/results/`, `summary.md`): muladd mid plain 15.0, L0-L12 98.3/99.0/98.0/96.0, corrupted value
followed 91.7-99.0%; word problem "lost" 3.3 -> 98.3 (L0) vs tool line 70.0; causal forward-only = plain
(`latent_write/causal/results/`), KV-rewrite 98.7 / 99 / 98. Runtime ~20-40 min per cell on one GPU (writer fit + 300
items x 11 layers x 4 arms). n = 300; fixed seeds; refits on another GPU can differ in the last bits.


## 12. RL with implants (GSM8K GRPO)

Code: `rl/` (details, all arms and tables: `rl/README.md`). Host: `Qwen/Qwen3-0.6B-Base`, trained with GRPO.

**Environment.** Python 3.12.3, torch 2.8.0 (cu128 on A100/A40, cu129 on H100), triton 3.4.0, transformers 4.56.2.
GPUs used: NVIDIA H100 80GB HBM3 (main 300-update runs, arm R traces, wiring), A100 80GB PCIe (64-update ablation, the
blog's dev numbers, interp traces), A40 48GB (arm 4 s0 to u500, 4c s0, wiring W1/W3r; needs `FLEX_SMALL=1`, set by the
launchers). One GPU per run; peak memory ~30-60 GB at `B_max 512` (use `--B_max 128` on smaller GPUs).
Triton needs `Python.h` (set `C_INCLUDE_PATH` in `rl/gsm_grpo/env_local.sh` if your venv lacks it).

**Data and fixed pieces.**
```
pip install -r requirements.txt triton==3.4.0
python data/prepare_gsm.py
python scripts/rethink_v2/make_scaled_train.py data/gsm8k/train.jsonl data/gsm8k/train_scaled.jsonl
python scripts/solve_veto.py                       # runs/veto_qwen3.pt (~50 min on 16 CPU cores; bit-identical rebuild)
```
Dev = GSM8K-train split B (md5(question) % 10 in {6,7}): 400 GSM8K + 400 scaled problems. Test = GSM8K test, GSM-Hard.

**Commands** (run from `rl/gsm_grpo`; recipe lr 1e-6, G 8, 64 prompts/update, T 1.0, Dr.GRPO, no KL):
```
cd rl/gsm_grpo
bash abl.sh q_abl_a100.txt; bash abl.sh q_abl_a100c.txt     # arms 1 / 2 / R, seeds 0,1, 64 updates (blog numbers)
bash final_pod.sh qf_a100.txt                               # one-shot test evals of those 6 checkpoints + base
python compile_abl.py                                       # table from results (CPU)
# 300-update runs (H100): python train2.py --arm {1,2,4} --seed {0,1} --run_dir runs/long/<name> --updates 300 \
#   --prompts 64 --G 8 --T 1.0 --lr 1e-6 --dev_n 800 --dev_every 25 --ckpt_every 100 --fused 1 \
#   [arm 4: --max_new 700 --S_max 912 --tok_budget 8192]; then final_eval.py on step_000300; python compile_long.py
bash sweep.sh q_h100.txt; bash sweep.sh q_a100.txt           # HP sweep (arm 2)
cd ../interp && bash run_all.sh && bash run_aR_h100.sh && python analyze.py && python aR_analysis.py   # traces -> stats
cd ../wiring && bash launch.sh 0 2 40000 "" runs_h100_gpu1.txt && RUNS=../results/wiring/H100 python compile_results.py
bash launch.sh 0 1 40000 "" q4_a100.txt          # half-broken calculator (CWinOx s2); s0/s1: runs_a40_g1.txt, q5_h100_g0.txt; lr-only: q6_h100_g0.txt
bash launch.sh 0 1 40000 "" q5_a100.txt          # W3d (distilled links) s2; s1 in q3_a100.txt; needs runs/sft3_W3e_lr1e-2/links_final.pt
cd ../library && bash launch/h100_LIB_s0.sh && bash launch/h100_LIBscr_s0.sh   # all implants attached, GSM8K RL
python mixed/mixed_data.py && python mixed/sft_warm.py --out mixed/sft_host   # mixed-task data + call-format SFT host (m4: --noisy, see NOTEBOOK)
bash launch/h100_m1_mixLIB.sh   # m1; m2 scrambled, m3 plain, m4 noisy: launch/*mix*.sh (final runs: --updates 150 --prompts 64 --ckpt_every 150 --test_n 400)
cd ../figures && python make_blog_figs_rl.py && python make_rl45.py && python make_rl6.py   # all RL blog figures (CPU)
```

**Blog claims -> numbers -> files** (ledger `rl/gsm_grpo/RESULTS_ledger.md`, `rl/wiring/NOTEBOOK.md`,
`rl/interp/NOTEBOOK.md`; files in `rl/results/`). "u" = RL updates. Note which claims are 300-update (H100) and which are
64-update (A100 / H100 traces): the blog paragraph opens with 300 updates, but the writing-habit numbers and the broken
calculator numbers come from the 64-update runs.

| blog claim | value | updates, GPU, seeds | file / command |
|---|---|---|---|
| plain RL "from 53% to about 70%" on GSM8K test | base 52.8 -> arm 1 68.2 / 70.1 / 70.9 (mean 69.7) | u300; s0/s1 H100, s2 A100 | s0/s1: `rl/results/gsm_grpo/H100/final/H_{base_calc,a1_s*_u300}.json`, `python compile_long.py`; s2: `rl/wiring/NOTEBOOK.md` "Headline seed 2" (per-question records not shipped) |
| "with the calculator attached it lands in the same place" | arm 2 (on) 68.8 / 70.7 / 71.3 (mean 70.3); paired s0/s1 +0.57 [-1.06, +2.16] | u300 | `H_a2_s*_u300.json`, `compile_long.py`; s2 as above |
| GSM-Hard "44% for plain RL vs 55% with the calculator, averaged over three seeds" | arm 1 42.4 / 43.7 / 45.0 (mean 43.7), arm 2 54.7 / 56.0 / 55.8 (mean 55.5); s0/s1 paired +12.24 [+10.12, +14.29] | u300 | same |
| "switch the calculator off ... about 3.5 points worse than plain RL on GSM-Hard (40% vs 44%)" | arm 2 off 40.2 / 39.3 / 40.6 (mean 40.0) vs arm 1 mean 43.7: -3.7; s0/s1 paired -3.30 [-4.93, -1.78] | u300 | same |
| figure "Test accuracy after 300 RL updates and dev accuracy during RL" | bars = mean of 3 seeds (min-max whiskers); dev curve = mean of 3 seeds | u0-300 | `rl/figures/make_blog_figs_rl.py` (`rl1_calc_after_rl.png`) <- `rl/figures/data/`, values in `rl/figures/blog_figs_rl.json` |
| unit-word style: "17% of equations after 64 updates, from 30%; plain RL 24%" | 29.8 (base) -> 23.5 (plain RL) -> 16.7 (RL + calc) | **u64**, A100 traces, one seed | `rl/interp/stats.json`, `rl/interp/NOTEBOOK.md` Q2 |
| (earlier draft, not in the final blog) "copies its answer 99% of the time" | 99.0 / 99.2 / 99.4 | u64, A100 traces | `rl/interp/stats.json` |
| broken calculator "starts at 7%, climbs back to 37%" | dev 7.5 -> 36.6 / 37.9 (A100); 7.0 -> 36.9 / 36.4 (H100) | **u64**, s0/s1 | `rl/results/gsm_grpo/{A100,H100}/abl/aR_*/steps.jsonl` |
| "broken calculator goes from firing on 75% of equations to 12%", "a unit word after most numbers (87%, up from 29%)" | 74.6 -> 12.1 (s1 11.2); unit words 29 -> 86 / 89% | u64, H100 traces | `rl/interp/stats_aR.json`, NOTEBOOK Q4 |
| (earlier draft figure, not in the final blog) "RL with and without the calculator"; the final blog's "Good calculator: RL writes fewer equations the calculator skips. Broken calculator: RL stops triggering it" is `rl2_writing` (`rl/figures/make_blog_figs_rl.py`, from `stats_aR.json` and NOTEBOOK Q2) | left: dev 44/50, 53/60, 56/65 (off/on); right: the two rows above | u64, A100 traces, one seed | `rl/interp/make_fig.py` (v1) / `make_fig_v2.py` (right panel only); **need the traces** (not released) |
| "more than an OOM faster than its first HuggingFace backend" | v2 engine 19,800 tok/s (H100, 512 rows, implant on) vs v1 HF loop 207 tok/s and HF `.generate` 288 tok/s (A100, 16 rows): ~70-96x across GPUs and batch sizes; single stream ~13x | — | `rl/results/gsm_grpo/H100/clean/rollout_*.json`, `rl/results/gsm_grpo/A100/prof_v1_a100.json` |
| (earlier draft, not in the final blog) in-wire + out-wires "add fewer weights than the implant" (CW-full) | 286,469 wire params vs 2,395,647 implant nonzeros (12%) | — | `rl/results/wiring/H100/L300_CW_s0_kl001evalall0/steps.jsonl` (`params`) |
| "In some of our early runs (4 of 8, one seed each), RL first "blocks" most calculator calls ... then lets them back in by digit size: big numbers first, 1-2 digit numbers ... last (the other runs just call every time)"; earlier draft (not in the final blog): CW-full "first vetoes almost everything", "skip 1-2 digit numbers ... keep it for 5+ digits", "the veto fades" | greedy-dev veto share by largest operand (1-2 / 3-4 / 5-6 / 7+ digits): u25 97/90/44/40, u100 69/24/0/2, u250 18/3/0/0, u300 32/6/1/0; dev u300 67.5 | u0-300, H100, s0 (s1 never vetoes; across 8 correct-calculator in-wire runs 4 show a size-ordered veto, 3 of them transient: "the other runs just call every time") | same `steps.jsonl` (`dev.dev_veto_by_digits`); figure `rl/figures/make_rl6.py` (`rl6_wiring.png`; older version `figures/blog/cw_veto.py`) |
| half-broken calculator: "reduces model accuracy down to 18-36% (vs 62% with no calculator)", "in five different runs ... letting through 76-95% of × and ÷ calls and blocks almost every + and − call" | hand trigger dev 19.0 / 18.8 / 36.0 (CWinOx s0-s2), 18.3 / 17.8 (CWinOlr s0/s1); x÷ let through 89.8 / 76.4 / 88.4 (CWinOx), 95.0 / 80.3 (CWinOlr); +− 5.1 / 6.1 / 5.3, 0.4 / 0.2; learned in-wire dev 65.0 / 64.9 / 67.4 / 66.1 / 66.6 vs no calculator 61.3 / 60.1 / 64.3 / 61.3 / 62.0 | u300 (rollout calls pooled u291-300), s0 A40, s1 H100, s2 A100 | `rl/results/wiring/figdata/CWinO{x,lr}_s*.jsonl`, `fig_competence_map.json`; figure `rl/figures/make_blog_figs_rl.py` (`rl3_competence.png`, 3 CWinOx seeds); commands: `rl/wiring/q4_a100.txt`, `q5_h100_g0.txt`, `q6_h100_g0.txt`, `runs_a40_g1.txt` |
| "attached every implant at once ... The BBH and IFEval implants don't fire on GSM8k ... RL shut the scrambled ones off within 25-50 updates, and the correct set ended ahead on GSM-Hard in all three seeds. RL kept the good calculator in two of the three seeds; in the third it switched the calculator off too, which is most of that 14-point gap on GSM-Hard" | GSM8K / GSM-Hard, LIB vs LIBscr: s0 69.4 / 56.1 vs 69.3 / 51.9; s1 70.4 / 56.2 vs 65.5 / 41.6 (LIBscr calls 0.00 per question at test: calculator off); s2 71.3 / 55.2 vs 66.7 / 51.3; GSM-Hard gap +4.25 / +14.56 / +3.87; scrambled expr_gen / calc_gen at the 5% floor by u25-50 | u300, s0 H100, s1/s2 B200 | s0: `rl/library/NOTEBOOK.md`, `rl/results/library/figdata/LIB*_steps.jsonl`, launch `rl/library/launch/h100_LIB{,scr}_s0.sh`; s1/s2: §23 (`checks/seeds3/libpair.md`); figure `rl/figures/make_rl45.py` (`rl4_toolbox.png`) |
| mixed tasks: "Each implant ends up used on its own kind of question, and the model beats plain RL by 13 points (58% vs 45%, 3 seeds)"; figure "With implants on, RL wins overall and on 3 of 4 task types (three seeds)" | all implants (m1 with `--cw_mask 1`) 57.8 [53.2, 62.1] vs plain RL from the worked-solution SFT host 44.7 [40.3, 49.2]; paired +13.1 [+8.7, +17.6] (gsm +10.3, bbh +10.7, combo +2.7 n.s., ifeval +26.8); implants off 32.7 | u150, 3 seeds, test 400 stratified | §23 (`checks/seeds3/`) |
| (earlier one-seed run; quotes from an earlier draft, not in the final blog) mixed tasks: "68% on held-out questions vs 36% for plain RL, 37% with the implants off", "tracker costs 15 points on BBH, the IFEval implant 35 points on IFEval", "calculator calls inside BBH answers fade from 0.16 to 0.04", scrambled "38%" | m1 68.2 (off 36.8), m3 (plain RL from the tagged SFT host) 36.3, m2 37.8; LOO bbh 87.6 -> 72.3, ifeval 81.5 -> 46.7; m4 calc@bbh 0.16 -> 0.04, calc@ifeval 0.22 -> 0.23 (m2 / m4 seeds 1-2: §23) | u150, H100 (m1, m4), A100 (m2), A40 (m3), test 400 stratified | `rl/results/library/figdata/mix_m{1,2,4}_*`, `rl/library/NOTEBOOK.md` "Mixed-task FINAL"; figure `rl/figures/make_rl45.py` (`rl5_mixed.png`); launch `rl/library/launch/*mix*.sh` with `--updates 150 --prompts 64 --ckpt_every 150 --test_n 400` |
| distilled links: "RL makes them 15-20% stronger and roughly doubles their effect: +13 points on GSM-Hard (three seeds; the animation below plays one of them)"; earlier draft (not in the final blog): "+11 points (+24 on big numbers)", animation "gap widens from about 5 to 11 points" | GSM-Hard test, links on − host only, 3-seed rerun: +14.6 / +13.5 / +11.7 (mean +13.3), GSM8K test +0.0 / +0.4 / −0.4 (§23, `checks/seeds3/`); original runs, W3d links-on minus links-off on dev: u0 +4.9 / +6.0 -> u300 +11.3 / +11.5 (s0/s1), +10.6 at u250 (s2, OOM at u251); scaled half +24.6 / +24.0; link norms +15-20% | u300, H100 (s0), A100 (s1, s2) | `rl/results/wiring/figdata/W3d_s{0,1}.jsonl` (s2 log not shipped), `rl/wiring/NOTEBOOK.md`; animation `rl/wiring/make_anim.py` -> `rl/figures/anim_w3d_links.gif`; commands `rl/wiring/q3_a100.txt`, `q5_a100.txt` |
| "RL couldn't discover an implant from scratch through new random links"; earlier draft (not in the final blog): "wiring the calculator in only through zero-initialized links ... links stay near zero" | W2 (zero-init links next to the hand write): link maps 0.044 / 0.043 / 0.012 at u300, links-off = 72.0 = links-on; links only, no hand write: W3e (embedding init, gate 0.1) 63.9 and W3r (random init) 61.1 at u300, maps stay at init; a links-only arm with exactly zero-init gates (W3z) was **not run** | u300 (u500 runs read at u300), H100 / A40, s0 | `rl/results/wiring/H100/L500_{W2,W3e}_s0/`, `rl/results/wiring/A40/L500_W3r_s0_fused0/`, `results.md` |
| (earlier draft, not in the final blog) broken calculator + in-wire: "vetoes every call within 25 updates", "then learns like plain RL" | CWinR: 100% of greedy-dev calls vetoed from u25; dev 7.0 -> 58.9 (u100) -> 61.1 (u300) vs plain RL (arm 1, H100) 59.9 / 62.8 at u300 and arm R 36.6-37.9 at u64 | u300, H100, s0 | `rl/results/wiring/H100/L300_CWinR_s0_evalall0/steps.jsonl`; `rl/results/gsm_grpo/H100/long/a1_*` |
| (earlier draft, not in the current blog) "think again" implant: "RL turns it down to about 4% of answers" | arm 4 rethink fires on 67.3% of test answers at u0, 4.6 / 4.2% at u300 | u300, H100, s0/s1 | `H_a4_s*_u300.json`, `compile_long.py` (rethink frac) |
| (earlier draft, not in the final blog) "Letting RL also train its detector, or the phrase itself, doesn't change that" | 4cRL fires on 13.9 / 11.4% of dev answers at u300 and changes 0.0 / 0.9%; 4sA (soft phrase) fires on 14.1%, changes 0.5%, fixes 0.2% | u300, A100/H100, dev only | `rl/results/gsm_grpo/*/long/a4c*`, `a4sA*`; ledger section 5 |

All wiring runs: `rl/results/wiring/{H100,A40}/<run>/{config.json,steps.jsonl}`, tables `results.md`
(`RUNS=../results/wiring/H100 python compile_results.py`, CPU). The released `config.json` of the CW*, W3d and W2ws runs
were written before the per-arm defaults were applied: those runs used `--link_lr 1e-2 --link_eps 1e-12` unless the run
name says otherwise (the training log prints "in-wire lr 0.01"); the current `train_wiring.py` rewrites `config.json`
after the defaults. W3d starts from the SFT-distilled links `rl/results/wiring/H100/sft3_W3e_lr1e-2/links_final.pt`
(copy it to `rl/wiring/runs/sft3_W3e_lr1e-2/`; it was made by `rl/wiring/smoke/cw2.sh` from `gen_traces.py` traces).
Wiring commands (from `rl/wiring`): one line per run in `runs_*.txt` / `q2_*.txt` = `<arm> <seed> [args]`, run with
`bash launch.sh <gpu> <max_parallel> <free_MiB> "<env>" <list>` (A40: env `FLEX_SMALL=1` and `--fused 0`); the
launcher adds `--updates 300 --dev_every 25 --dev_extra 64 --dev_n 800 --B_max 128 --tok_budget 4096 --eval_all 1`
(`--ckpt_every` 100 for the released runs). CWinOx = `CWinO --iw_eps 0.05 --cw_lr 1e-3 --link_eps 1e-8`.

**Runtime.** 64-update run ~16 min (A100); 300-update run ~35 min (H100, arm 2), ~53 min (arm 4); final eval minutes;
wiring 300-update run ~2 h (H100, B_max 128, with evals; often 2 runs per GPU), ~2.8 h on the A40 (`--fused 0`).
Whole blog subset (6 x 64-update runs + finals + traces) ~3 A100-hours; 300-update table ~5 H100-hours.

**Seed noise.** 2 seeds per main arm: dev differs by <= 1.4 pp (64 updates), test by <= 0.6 pp (arms 1/2, 64 updates) and
up to 3.5 pp (arm 4, 300 updates). Dev SE ~1.7 pp (n=800). Bootstrap CIs are over questions only (seed variance not
included). bf16 numbers differ by GPU and `B_max` by ~1 pp: compare within a GPU. Interp traces use one seed per arm.
Wiring arms, 4c, 4s, library and mixed arms: one seed unless the table says otherwise (CWinOx: 3 seeds, CWinOlr: 2,
W3d: 3, LIB / LIBscr: 3, mixed (m1 with `--cw_mask 1` vs plain RL): 3; the 3-seed reruns are in §23, `checks/seeds3/`).

**Not reproducible from the repo alone / caveats.**
* No checkpoints are released; every number after u0 needs the training runs (~1 GPU-hour each).
* The throughput ratio is across GPUs and batch sizes (v1/HF on A100 at 16 rows, v2 on H100 at 512 rows); there is no
  same-GPU, same-batch comparison. The v1 engine that `prof_v1.py` profiles is not released.
* Interp traces (generated text) are not released; `rl/interp/make_fig.py` / `make_fig_v2.py` (the earlier-draft figure
  "RL with and without the calculator", not in the final blog) and `analyze.py` need them: regenerate with `rl/interp/run_*.sh` from your own 64-update
  checkpoints. Their reduced outputs are shipped (`stats.json`, `stats_aR.json`, `analysis_out.txt`).
* Partial logs from the first packaging stay under `rl/results/wiring/{H100,A40}/` (CW-full s1, CWinOx s0/s1, CWinC
  s0/s1, W4w s0); the finished CWinOx / CWinOlr / W3d logs used by the blog are in `rl/results/wiring/figdata/`
  (steps only, no `config.json`; the arguments are in the `rl/wiring/q*.txt` run lists).
* Seed 2 of the 300-update headline (arms 1 / 2, A100): dev logs in `rl/figures/data/a{1,2}_s2.out`, test numbers in
  `rl/wiring/NOTEBOOK.md`; per-question test records not shipped.
* Library / mixed: no checkpoints or SFT'd hosts; mixed question files are regenerated by `rl/library/mixed/mixed_data.py`
  (SHA-256 of the originals in `rl/library/mixed/DATA_SHA256.txt`; `mixed_data.py` reproduced train/test byte for byte on CPU at release).
* `rl/engine2/` (megakernel engine) is work in progress; no blog number uses it.


## 13. Figures

`bash figures/blog/make_blog_figs.sh` (CPU) rebuilds the static figures into `figures/blog/figs/`;
`figures/blog/README.md` maps each blog figure title to its script and data. The animations need `rsvg-convert`,
`ffmpeg`, `gifsicle`. The current blog's RL figures (rl1-rl6 and the W3d link animation) are drawn by `rl/figures/`
(`rl/figures/README.md` maps each blog figure to its script and data). Older RL figures in `figures/blog/` read
`rl/results/curves/<run>/steps.jsonl` (env `RL_CURVES`).
With the Avenir Next fonts in `figures/blog/fonts/` (not shipped) a fresh clone redraws every committed static PNG pixel for
pixel (`VERIFY.md` section 11). The CW veto figure reads `rl/results/wiring/H100/L300_CW_s0_kl001evalall0/steps.jsonl`.
The earlier-draft figure "RL with and without the calculator" (not in the final blog) is `rl/interp/make_fig.py` and needs the unreleased traces
(section 12). Files with a `_v2` suffix are redraws for the current blog text (`make_figs_v2.py`, `make_figs3_v2.py`,
`transfer_fig_v2.py`, `rl/interp/make_fig_v2.py`); `make_blog_figs.sh` does not run them.

Later re-renders (CPU, matplotlib only; each script writes PNG + SVG next to itself and a `*_numbers.json` with the
plotted values):
* `rl/figures/v3/`: `make_rerender_v3.py` (rl1-rl4, rl6 in the v3 style; reads `rl/figures/data/`,
  `rl/results/{wiring,library}/figdata/`, `rl/interp/stats_aR.json`), `make_new_v3.py` (rl5 mixed tasks, rl7 calculator
  share per update, rl8 trained-implant exactness; data in `rl/figures/v3/data/`), `make_rl_curves_all.py` (rl9),
  `make_rl_stack_curves.py` (rl10-rl12, RL with the gsm_push stack; runs in progress at release).
  `calc_share_remote.py` computed `data/calc_share_*.jsonl` from rollout dumps (not shipped).
* `figures/blog/review_final/`: bar re-styles (`bars_v2.py`, `ifeval_v2.py`, `bars_restyle.py`), `gsm_compose.py`,
  `latent_write.py`, `topic_blocker.py`, `make_rl4_toolbox.py`, `anim_w3d_links.py`, and the zoom animation sources
  (`zoom_src/`, needs `rsvg-convert` and `ffmpeg`; its GIF / MP4 outputs are not shipped).

## 14. Summary: what cannot be reproduced from the repository alone

| item | why | what is shipped instead |
|---|---|---|
| Fine-tuned hosts (BBH v2 0.6B/1.7B/4B, ft100, NL control, Team A/B, finetuned_phase) and all RL checkpoints | weights not released | training rows or seeded generators (byte-identical, SHA-256), scripts, hyperparameters; rebuilding gives bf16-noise-level differences |
| Compiled IFEval implant tensor `implant14_sd.pt` (1.23 GB) | > 50 MB | `ifeval/full14_compiled/export.py` rebuilds it in ~20 s; sha256 `e85650adb761df2a292c5d1ce4aa836ddb9b2a32f3d2dbfc77b37a849787c200` |
| RL engine speed-up ("more than an OOM faster") | no same-GPU, same-batch measurement exists | measured: ~70-96x at large batch (v2 on H100 vs v1/HF on A100), ~13x single stream; `rl/results/gsm_grpo/H100/clean/`, `rl/gsm_grpo/RESULTS_ledger.md` §1 |
| 3-way GSM fine-tune per-seed files, eyes seed 0 raw files | on a machine that was offline at packaging | numbers in `finetuned_phase/results/reported_numbers.json` (`verified_here: false`) |
| Eyes single-seed 8-example 88% (earlier draft) | recipe not recorded; did not reproduce | 4-seed sweep (78.5 at n=8, 94.7 at n=32) is what the blog uses |
| Detector "28 fixed: 16 reasoning / 5 arithmetic / 7 unclassified" | manual reading of generated text (not released) | the 28 problem ids are recoverable from `results/rethink_v2/` per-question records |
| BBH per-tool composition analysis, NL truncated-budget re-score, RL interp traces, animation score jobs | need generated text / token ids (not released) | their outputs: `bbh_generic/integrate/TABLES_v1_vs_v2.md`, `bbh_generic/nl_control/results/scores_nl.json`, `rl/interp/stats*.json`, `figures/blog/figs/anim/*_trace.json` |
| IFEval add-one-rule order | chosen from Team D FRESH-set runs not in the repo | order hard-coded in `ifeval/transfer/curve2.py` |
| Topic-blocker dev set regeneration row for row | written by an LLM (Qwen2.5-7B, sampled) | dev/test data and fitted probes shipped (`topic_fence/data/`, `topic_fence/fitted/`) |
| 0.6B "frozen + Team C" 27-task BBH row | older run; per-question rows for 22 of 27 tasks | aggregate in `bbh_generic/ft100/scores_ft100.json` |
| Wiring arms CW s1, CWinC s0/s1, W4w s0 (partial); W3d s2 (OOM at u251) | partial runs | partial `steps.jsonl` in `rl/results/wiring/`; W3d s2 numbers in `rl/wiring/NOTEBOOK.md` |
| Seed-2 test records of the 300-update headline (arms 1 / 2) | stayed on the training machine | numbers in `rl/wiring/NOTEBOOK.md`, dev logs `rl/figures/data/a{1,2}_s2.out` |
| Mixed-task question files, SFT'd hosts (`rl/library/mixed/`) | regenerated by script; hosts are checkpoints | `mixed_data.py` + `envs.py`, `sft_warm.py`; SHA-256 of the original data in `DATA_SHA256.txt` |
| Latent-write writer banks (four 0.6B banks, 36 MB) | > 5 MB; refit on every run | `latent_write/run.py` refits them (SHA-256 of the originals in `latent_write/README.md`) |
| Call-gate opener states `bbh_generic/callgate/results/states{2,3,4}.pt` (5 MB) | hidden-state dumps of the unreleased fine-tuned host | `run_callgate.sh states` (GPU); the fitted probes `probe_v{1,2,3}.pt` are shipped |
| gsm_push: the RL+calc host, per-row outputs, hidden-state dumps | checkpoint; dumps | rebuild the host with `rl/gsm_grpo` (§15); run logs + `summary.json` in `gsm_push/results/` |
| `ifeval/heldout/vecs2.pt` (11.6 MB) | > 5 MB | its force_fix rows `ifeval/heldout/vecs2_forcefix.pt` (the only part the circuit reads) |
| Earlier-draft RL figure "RL with and without the calculator" (not in the final blog) | `rl/interp/make_fig*.py` read the generated dev traces (not released) | the plotted numbers: `rl/interp/stats.json`, `stats_aR.json`, `rl/interp/NOTEBOOK.md` |
| Wiring arm W3z (links only, exactly zero-init gates) | never run | W3e / W3r (embedding / random init) |
| gsm_stack: u0 host and RL arm hosts, resume states, eval rows | checkpoints; per-row outputs | `rl/gsm_grpo` + `gsm_stack/engine/run_arm.sh` (§16); tables in `gsm_stack/results/`, dev curves in `gsm_stack/results/curves/` |
| RL checkpoints of the uniform arms (GSM8K P64 / CW / P17 / CW17; BBH F0 / I0 / P0 / F17 / I17 / P17; IFEval P / I at 0.6B and 1.7B) and the 1.7B call-FT host `ft17` | weights not released | arm names, seeds and launch lines in §17-§18; rebuilding gives bf16-noise-level differences (compare within one GPU type) |
| Per-item eval outputs of §17-§18 and §20-§24 with generated tokens / text | generated text not released | the tables and `rows*.json` in `uniform/` and `checks/`; reduced per-item rows without text (`*.items.jsonl`) |
| RL checkpoints of the C35 / C36 arms (§20, §21) and of the 6-seed / 3-seed checks (§22, §23) | weights not released | launch lines per seed in `checks/*/`; steps.jsonl and configs are shipped |
| Avenir fonts for figures | license | matplotlib default font fallback |

Across GPU types, bf16 greedy text differs on a few percent of items (e.g. IFEval 0.6B: 58.23 -> 72.83 on an A40, 57.86 ->
72.09 on an A100; GSM8K ~4-6% of problems flip between stacks). Compare within one GPU type, and expect +-1 point.


## 15. GSM8K push: single-call implant stack on the RL+calc host

`gsm_push/README.md` has the commands; `gsm_push/docs/FINAL.md` the full report.

* Host: the RL+calc checkpoint `a2_u300_s2` (§12, `rl/gsm_grpo/train2.py --arm 2 --seed 2`, 300 updates; not released).
  `export CKPT=<run_dir>/step_000300`.
* Hardware: one H100 80 GB (A40 48 GB at `BMAX=32`). Runtime: S0-S2F about 1 min per config per 1,319-item set;
  S3-serial N = 8 about 12 min (test) to 25 min (GSM-Hard, A100).
* Expected (GSM8K test / GSM-Hard): S0 71.27 / 54.59, S1 71.87 / 56.03, S2F 73.39 / 56.18, S3-serial N = 4 78.92 / 60.42,
  N = 8 80.44 / 62.17. Shipped logs: `gsm_push/results/final/{fwd,fgtw,ser4,ser8,ser4b,ser_a100}/`.
* Noise: bf16 batch-shape numerics move a few rows between batch sizes and GPU types (25 / 600 dev rows differ
  token-wise between the stacked engine with every implant off and the stock engine, 595 / 600 equal on correctness).
* CPU check: `python gsm_push/calc_v2/test_exact.py` (calc v2 tracker, 10k expressions per form, 0 wrong expected;
  tokenizer only).
* RL with the stack attached (`gsm_push/final/run_rl.sh`, `docs/RL_STACK.md`) was still running at release; no number
  from it is claimed.

## 16. GSM8K single-call implant stack (`gsm_stack/`)

`gsm_stack/README.md` has the overview, `gsm_stack/COMPLIANCE.md` what is in weights, `gsm_stack/results/RL_RESULTS.md`
every table with CIs. Every number below is one run (GSM8K test or GSM-Hard, 1,319 items each), T = 1, sampling seed
7, 320 tokens per try, graded on the last number, 95% paired bootstrap with 4,000 resamples
(`gsm_stack/eval/ana_rltest.py`, `results_tools/pboot.py`, seed-pooled: `results_tools/pbootm.py`).

* **Environment.** As §12: Python 3.12, torch 2.8.0 (cu128 / cu129), triton 3.4.0, transformers 4.56.2; NVRTC from the
  `nvidia-cuda-nvrtc-cu12` wheel (pulled in by torch). If your image points `TRITON_PTXAS_PATH` at a newer CUDA, run with
  `WSM_TRITON_BUNDLED=1`.
* **Hardware.** Every `gsm_stack` number was measured on **one NVIDIA B200** (RL arms and test runs). The code has the
  same requirements as the §12 / §15 engines and runs on one H100 80 GB (the gsm_push stack, the predecessor of this
  engine, was run there); no `gsm_stack` number was re-measured on an H100, so compare within one GPU type
  (bf16 batch-shape noise moves a few rows; §15). On a 48 GB GPU lower `BMAX` / `FE_BMAX`.
* **Paths.** `gsm_stack/env.sh`: `WSM_DATA` (default `runs/gsm_stack`), `CKPT_U0`, `ARMDIR`. Rows land in
  `$WSM_WORK/fe/<SET>_<NAME>.jsonl`; each eval line prints a `FE ... acc=` summary.
* **Running part of a list.** `fe.sh` runs every line of a list and skips lines whose rows exist; lines whose host is
  an RL arm need that arm's `step_000300` (`grep -v '^#' LIST | grep NAME > my.list` runs a subset).

### 16.1 Hosts (built, not shipped)

```bash
python data/prepare_gsm.py
python scripts/rethink_v2/make_scaled_train.py data/gsm8k/train.jsonl data/gsm8k/train_scaled.jsonl
python scripts/solve_veto.py --device cuda                # runs/veto_qwen3.pt
# u0 = a2_u300_s2: arm 2 (calculator attached), seed 2, 300 GRPO updates (one GPU; ~35 min on an H100, §12)
cd rl/gsm_grpo && python train2.py --arm 2 --seed 2 --run_dir runs/long/a2_u300_s2 --updates 300 --prompts 64 --G 8 \
    --T 1.0 --lr 1e-6 --dev_n 800 --dev_every 25 --ckpt_every 100 --fused 1 && cd ../..
mkdir -p runs/gsm_stack && cp -r rl/gsm_grpo/runs/long/a2_u300_s2/step_000300 runs/gsm_stack/ckpt_a2s2
# harness check on dev (seed 4242): val_u0 must print 81.33 (the arms' u0 dev vote8); about 1 min on a B200
grep val_u0 gsm_stack/eval/lists/val.list > /tmp/val_u0.list && GPU=0 bash gsm_stack/eval/fe.sh /tmp/val_u0.list
```

RL arms (300 updates each, 64 or 256 prompts x G 8, lr 1e-6; arm table in the header of `gsm_stack/engine/run_arm.sh`):
`GPU=0 bash gsm_stack/engine/run_arm.sh ARM`. Wall time per arm on one B200, from the original runs: Pb (plain RL)
~40 min, Bb ~75 min, C / C_s* ~1.5 h, CW_s* ~2 h, A1 / A1W ~2.3-2.6 h. Every arm reads `$CKPT_U0` as its implant source,
so build u0 first even for the base-host arms. RL is not bit-reproducible across GPUs: expect the seed spreads quoted below.

### 16.2 The ladder on the u0 host (GSM8K test; GSM-Hard)

| rung | command | list line | expected test / GSM-Hard | time (B200) |
|---|---|---|---|---|
| host, no implants | `bash gsm_stack/eval/fe.sh gsm_stack/eval/lists/test3.list` | `lad_H` (hard: `hard2.list lad_H`) | 67.70 / 39.42 | 1-2.5 min per set |
| + calc_s4 | same | `lad_C4` (`hard2.list lad_C4`) | 68.69 / 53.90 | |
| + chain2 | same | `lad_D` (`hard2.list lad_D`) | 69.45 / 54.74 | |
| S2F (sanity DFA + forget, one try, no missing-answer trigger) | same | `u0_B` (hard: `test4.list u0_B`) | **72.25** / 55.19 | |
| S2F with the compiled missing-answer trigger | `fe.sh .../miss3.list` | `S2Fw_u0` | 72.63 / 56.03 | |
| serial N = 8, no position reset | `fe.sh .../test1.list` | `u0_N8` (hard: `test4.list u0_N8`) | 71.87 / 57.54 | ~2 min per set |
| rung 1: + WPR | `fe.sh .../wpr_rung1.list` | `r1_u0_N8` | **80.67** / 62.17 | not recorded (N = 8 class: ~2 min per set) |
| rung 2: + adaptive tries | `bash gsm_stack/eval/jobs/rung_job.sh r2 wpr_rung2.list` (runs `test_aw.py` first) | `r2_u0` | **81.58** / 62.62 | not recorded (more tries than rung 1) |
| rung 3: + Nmax 24 + answer-group voter | `bash gsm_stack/eval/jobs/rung_job.sh r3 wpr_rung3.list` (uses `gsm_stack/gadgets/vf_w.pt`, shipped; `gs_tree` copies it in) | `r3_u0` | **82.71** / 60.42 | 11.5 min (test + hard) |
| rung 2 + Nmax 24, no voter | `rung_job.sh r3 wpr_r2n24.list` | | 81.50 / **63.53** | |
| *reference, NON-COMPLIANT: N = 8 + programmatic position reset* | `fe.sh .../test2.list` | `u0_N8pr1` | *80.97* / *62.09* | |

### 16.3 RL from the base model

| row | train | evaluate | expected test (GSM-Hard) |
|---|---|---|---|
| base host, no implants, no RL | – | `fe.sh .../base.list` (`baseH`) | 28.05 (10.46) |
| plain RL, no implants, 64 prompts/update (arm P64, the blog §5.1 row) | `run_arm.sh P64_s0` ... `P64_s5` | §17 / §22 protocol | 6-seed mean **68.32** (41.60) (§22, `checks/seeds35/RESULTS.md`) |
| plain RL, no implants, 256 prompts/update (arm Pb, earlier; not the blog row), seeds 1, 2 | `run_arm.sh Pb_s1`, `Pb_s2` | `seed_Pbs1.list`, `seed_Pbs2.list` | 70.36, 71.19; pooled 70.77 (45.00) |
| RL + S2F tools, one try (arm Bb), seeds 5, 6, 7 | `run_arm.sh Bb_s5` ... | `seed_Bbs{5,6,7}.list` (compiled trigger) | 72.10, 73.39, 73.39; pooled 72.96 (55.72) |
| RL + serial stack N = 4 (arm C), seeds 5, 7, 6 | `run_arm.sh C`, `C_s2`, `C_s3` | `test2.list C_N4`, `seed_Cs2.list`, `seed_Cs3.list` | 74.60, 74.68, 75.13; pooled 74.80 (52.97) |
| RL + serial stack + WPR, N = 4 (arm CW), seeds 0, 1, 2 | `run_arm.sh CW_s0` ... | `seed_CWs{0,1,2}.list` | 77.18, 77.33, 75.06; pooled 76.52 (55.07) |
| **CW hosts + the rung-2 stack** | (the CW arms) | `rung_job.sh r2 wpr_r2cw.list` | 81.12, 81.43, 78.24; pooled **80.26** (sd 1.76; GSM-Hard 58.76); 6 seeds (blog §5.1 row 4, §22): **80.44** (59.01) |

Pooled deltas, e.g. P → CW + rung 2: `python3 gsm_stack/eval/results_tools/pbootm.py test_Pb_s1_u300.jsonl,test_Pb_s2_u300.jsonl test_r2_CW_s0.jsonl,test_r2_CW_s1.jsonl,test_r2_CW_s2.jsonl`
(in `$WSM_WORK/fe`). Greedy single-attempt references: `greedy_H.list` (base 52.54, P 70.74); 4-shot base 56.48
(`RL_RESULTS.md` §1).

### 16.4 RL on the u0 host

| row | train | evaluate | expected test (GSM-Hard) |
|---|---|---|---|
| A1 (attempt-level credit), N = 8, seeds 3, 4, 6 | `run_arm.sh A1`, `A1_s2`, `A1_s3` | `test1.list A1_N8`, `seed_A1s2_Bs2.list`, `seed_A1s3.list` | 79.76, 80.74, 79.68; pooled 80.06 (63.31) |
| B (S2F, one try), seeds 1, 2, 3 | `run_arm.sh B`, `B_s2`, `B_s3` | `miss3.list B_u300w`, `seed_A1s2_Bs2.list`, `seed_Bs3.list` | 74.22, 74.00, 73.62; pooled 73.94 (57.82) |
| A1W hosts + rung 2 (best GSM-Hard) | `run_arm.sh A1W_s{0,1,2}` | `rung_job.sh r2 wpr_r2rl.list` | 79.96 (**64.19**) |
| implant ablation on the RL hosts | – | `abl_test.list`, `abl_hard.list` | `RL_RESULTS.md` §8 (forget mask off: −8.2 on A1) |

The original A1 arm's seed was not recorded (`run_arm.sh` uses 3); its replicates A1_s2 / A1_s3 are seeds 4 / 6.

### 16.5 CPU checks (no model)

`VERIFY.md` §13 lists what was run from a fresh clone. The adaptive-tries fuzz (`test_aw.py`) also runs at the start of
every rung job.

### 16.6 Not reproducible from the repository alone

* The hosts (u0, every arm) and the resume states: rebuilt by the commands above.
* The eval rows behind the tables and the stack figures (`gsm_stack/figures/make_figs.py` reads them): per-item rows
  are rebuilt by the lists; the shipped renders are in `gsm_stack/figures/`.
* The implant lane's dev / dev2 rollouts and hidden-state tables (`implants/*/tables`, `out/`): rebuilt by each lane's
  `job/` or `aw/` scripts (one GPU job each).

## 17. Uniform rows on Qwen3-0.6B-Base (blog §5.1)

Blog claims backed here:
* The same 4 rows for GSM8K, IFEval and BBH: (1) host, (2) host + RL, (3) host + implants, (4) host + implants + RL with
  the implants attached.
* The decomposition of row 3.
* The matched-compute majority-vote control.
* The eval-seed spread.
* The fresh-set (GSM-Plus) check.
* The BBH per-task check.

Claim ids (C1-C23, C33) label the blog's ledger entries; every number is given inline here. Figure data:
`uniform/rl24/uniform.json` (`python3 uniform/rl24/make_uniform_json.py`, CPU, a few seconds). The blog's GSM8K and
IFEval numbers are the 6-RL-seed means of §22 (`checks/seeds35/`). The tables below are the 3-seed runs (RL seeds
0/1/2) that the commands here produce. Each subsection gives the 6-seed value next to the 3-seed one.

* **Code.** `uniform/` (layout in `uniform/README.md`), built on `rl/` (§12) and the `gsm_stack/` engine and trees (§16).
* **Environment.** The B200 row of the Environment table: Python 3.12, torch 2.8.0+cu128, transformers 4.56.2,
  triton 3.4.0. If your image points `TRITON_PTXAS_PATH` at a newer CUDA, run with `WSM_TRITON_BUNDLED=1`, which uses
  triton's bundled ptxas.
* **Hardware.** Every number in §17 and §18 was measured on **one NVIDIA B200** per job. "Packed" means several processes
  shared that one GPU, each with a memory cap. The original jobs went through a cluster queue with 2-12 h limits and were
  resubmitted until done. Job ids (jNNNNN) are kept only as labels. The launchers here run the same commands on a local
  GPU (`GPU=k` or `CUDA_VISIBLE_DEVICES=k`), and the chain drivers restart and resume the same way. Times are B200 wall
  times from the original runs.
* **Paths.** Every launcher sources `uniform/env.sh`. It sources `gsm_stack/env.sh` (`WSM_DATA`, `WSM_WORK`, `CKPT_U0`,
  `ARMDIR`, `WSM_VETO`, `gs_tree`) and sets `FT2_HOST` (the BBH call-FT host `ft_int_v2`), `RL24_DATA`, `RL28_DATA`,
  `ARMDIR17`, `BBH_DIR` and `BBH_COT`. Build the u0 host first (§16.1): every GSM8K arm reads its implant wiring from
  `$CKPT_U0`.

### 17.1 Protocol

* Rows:
  1. plain host;
  2. host + RL on the task, no implants;
  3. host + the task's implant set, no RL;
  4. host + implants + RL with the implants attached.
* RL (rows 2 and 4, all three tasks): GRPO (Dr.GRPO, no KL), 300 updates, 64 prompts x 8, lr 1e-6, T 1, seeds 0/1/2
  (seeds 3-5: §22), train data only.
* Hosts:
  * GSM8K and IFEval: `Qwen/Qwen3-0.6B-Base`.
  * BBH, all 4 rows: `ft_int_v2` = Base + the §6 call fine-tune (`bbh_generic/teamA/ft.py` on `train_union_v2`, 432 rows,
    lr 2e-5, 6 epochs, tok_budget 8000, seed 0; sha256 5e17c7c4…; recipe and command in §6.1). This is a whole-host
    weight change shared by rows 1-4. Plain Base is reference row 0, and RL from plain Base (P0) is row 2'.
  * The call-FT host is not used for GSM8K. It costs row 3 −14.94 [−17.89, −12.21] on test and −5.84 [−8.26, −3.41] on
    Hard (C14). Command: `FE_TREE=stk_r2q GPU=0 bash uniform/final_eval/fe.sh ft2_29a.list`, then `ft2_29b.list`
    (j23729 / j23730; needs `FT2_HOST`). Host decision: `uniform/rl24/HOST.md`.
* Eval settings:
  * GSM8K: test 1319 + GSM-Hard 1319, zero-shot chat format, T 1, seed 7, once per config.
  * IFEval: 541 prompts, chat template (thinking off), greedy, max_new 2048, seeded lm_eval checkers. The main metric is
    prompt-level strict.
  * BBH: BBH-27 (6,511 items), 3-shot CoT, greedy, max_new 600, `\n\nQ` stop.
* CIs: paired item bootstrap, 95%, with multi-seed rows averaged per item. GSM8K uses 2000 resamples; IFEval and BBH
  use 10,000.
* GSM8K eval harness (`uniform/final_eval/`):
  * `fe.sh LIST` runs the list lines `NAME HOST SET KIND N PR VM SAN RM [ENV]`. LIST is looked up in
    `uniform/final_eval/lists/`, then `gsm_stack/eval/lists/`.
  * `fejob.sh TREE LIST [BMAX SMAX]` runs the `test_aw` fuzz test of the tree first.
  * `fepack.sh TREE LIST...` packs several lists on one GPU (`FE_MEMFRAC` 0.9/n, B_max 120/n).
  * The trees `stk`, `stk_r2`, `stk_r2q`, `stk_r3` and `stk_r4` are overlays in `uniform/final_eval/trees/` on the
    `gsm_stack` trees. `gs_tree` materializes them under `$WSM_WORK/trees/` and adds the rung-3/4 gadgets from
    `gsm_stack/gadgets/`.
  * Output: `$FE_OUT_DIR/<SET>_<NAME>.jsonl` (default `uniform/final_eval/out/fe`).

  Every GSM8K eval below is one job on one GPU:
  ```bash
  GPU=0 bash uniform/final_eval/fejob.sh stk_r2q uni_r2base.list
  ```
  Do not pack SCH lists on the `stk` tree: that tree ignores the pack's B_max, and the packed j23878 ran out of memory.
* Analysis without a GPU: the analysis scripts (`ana_uni.py`, `q32/ana_q32.py`, `q33/*.py`, `q32/mech_2b.py`) read the
  reduced per-item rows in `uniform/final_eval/items/` through `uniform/items.py`. `FE_ROWS=uniform/final_eval/out/fe`
  puts your own rows first.

### 17.2 GSM8K rows

Implants (rows 3, 4) = rung 2 of the serial stack (§16.2):
* the calculator v2, chain and sanity implants;
* weight-space position reset (`WPR=1`);
* adaptive N 8 -> 16 attempts;
* the L18 probe voter.

The row-3 list line (`uni_r2base.list`) is
`r2_base base test S 8 0 1 1 0 WPR=1 SW_NS=16 SW_MAXN=7936 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400`.

RL arms (`gsm_stack/engine/run_arm.sh`; one B200 per arm; output `$ARMDIR/rl/<ARM>/step_000300`):
```bash
GPU=0 bash gsm_stack/engine/run_arm.sh P64_s0    # row 2: CFG=P, Base host, 64 prompts, S_max 1280; also _s1, _s2
GPU=0 bash gsm_stack/engine/run_arm.sh CW_s0     # row 4: CFG=A + WPR=1, Base host, N=4, S_max 2944; also _s1, _s2
```
Runtime: P64 about 15 min per arm (j23664 / j23666 / j23669); CW about 2 h per arm (j23383 / j23386 / j23398).

Evals (one GPU each):
```bash
GPU=0 bash uniform/final_eval/fe.sh base.list                         # row 1 (baseH lines), j23321, 3 min
GPU=0 bash uniform/final_eval/fejob.sh stk uni_p64.list               # row 2, j23692, 3.5 min
GPU=0 bash uniform/final_eval/fejob.sh stk_r2q uni_r2base.list        # row 3, j23626, 14 min
GPU=0 bash uniform/final_eval/r2cwjob.sh                              # row 4: stk_r2 tree, wpr_r2cw.list, B_max 128, S_max 8192; j23509, 28 min
GPU=0 bash uniform/final_eval/fejob.sh stk_r3 uni_r3base.list         # 3' (rung 3), j23627, 17 min
GPU=0 bash uniform/final_eval/fejob.sh stk_r3 uni_r3cw0.list          # 4' (CW + rung 3), j23628-30 (uni_r3cw0/1/2), 12-13 min each
GPU=0 bash uniform/final_eval/fejob.sh stk_r4 uni_r4cw.list           # 4'' (CW + rung 4), j23631, 37 min
python3 uniform/final_eval/ana_uni.py                                 # 2x2 effects (CPU, ~10 s)
```

Expected (3 RL seeds; row CIs from `uniform/rl24/uniform.json`):

| row | config | test [95% CI] (per seed) | Hard [95% CI] (per seed) | tok/q test / Hard |
|---|---|---|---|---|
| 1 | Base | 28.05 [25.47, 30.4] | 10.46 [8.79, 12.13] | 103 / 127 |
| 2 | P64 s0/s1/s2 | 68.49 [66.36, 70.56] (69.07 / 67.78 / 68.61) | 41.19 [38.84, 43.54] (41.24 / 41.70 / 40.64) | 129 / 164 |
| 3 | Base + rung 2 | 57.39 [54.89, 59.97] | 35.94 [33.43, 38.59] | 2650 / 3830 |
| 4 | CW s0/s1/s2 + rung 2 | 80.26 [78.34, 82.06] (81.12 / 81.43 / 78.24) | 58.76 [56.36, 61.21] (56.71 / 61.18 / 58.38) | 1612 / ~2650 |
| 3' | Base + rung 3 | 56.03 | 25.55 | 3627 / 5574 |
| 4' | CW + rung 3 | 80.47 (80.21 / 81.05 / 80.14) | 56.41 (55.57 / 57.32 / 56.33) | 1952 / 3401 |
| 4'' | CW + rung 4 | 80.57 (81.35 / 80.74 / 79.61) | 57.14 (55.72 / 58.61 / 57.09) | 1952 / 3395 |

| effect (`ana_uni.py`) | test | Hard |
|---|---|---|
| RL alone (2 − 1) | +40.43 [+37.88, +43.11] | +30.73 [+28.51, +33.03] |
| implants alone (3 − 1) | +29.34 [+26.54, +32.15] | +25.47 [+22.74, +27.98] |
| both (4 − 1) | +52.21 [+49.63, +54.97] | +48.29 [+45.74, +50.95] |
| RL given implants (4 − 3) | +22.87 [+20.37, +25.35] | +22.82 [+20.70, +25.27] |
| implants given RL (4 − 2) | +11.78 [+10.21, +13.29] | +17.56 [+15.59, +19.56] |
| interaction (4 − 3 − 2 + 1) | −17.56 [−21.05, −14.13] | −7.91 [−11.07, −4.70] |

**At 6 RL seeds** (§22, `checks/seeds35/`; these are the blog's numbers, "80% vs 68%", "12 points on GSM8K (17 on
GSM-Hard)"):

| | test | Hard |
|---|---|---|
| row 2 | 68.32 | 41.60 |
| row 4 | 80.44 | 59.01 |
| 4 − 2 | +12.12 [+10.77, +13.57] | +17.41 [+15.59, +19.27] |

Notes on these rows:

* Write audit of row 3: 1 miss in 180,705 writer steps on test, 0 on Hard.
* Row 4 re-run on the current `stk_r2q` build (j23903, 24 min):
  ```bash
  bash uniform/final_eval/fepack.sh stk_r2q q32rq_s0.list q32rq_s1.list q32rq_s2.list
  ```
  This gives 80.41 (81.27 / 81.05 / 78.92), +0.15 [−0.56, +0.88] vs the reported 80.26.
* With plain majority instead of the probe voter (j23904, 26 min):
  ```bash
  bash uniform/final_eval/fepack.sh stk_r2q q32rv_s0.list q32rv_s1.list q32rv_s2.list
  ```
  This gives 79.13 (78.62 / 80.06 / 78.70), −1.29 [−2.12, −0.45] vs the re-run. Analysis:
  `python3 uniform/final_eval/q32/ana_q32.py r`.
* Row 2' (P, 256 prompts per update, §16.3) is 70.66 / 45.08. It is not the uniform recipe.
* In the commands below, `fepack.sh` and `fejob.sh` stand for `bash uniform/final_eval/fepack.sh` and
  `bash uniform/final_eval/fejob.sh`, and `ana_q32.py X` stands for `python3 uniform/final_eval/q32/ana_q32.py X` (CPU).
  Every job runs on one B200.

#### Row 2b, "RL plain, attach after"

P64 hosts + the untrained rung-2 stack, with row-4 settings.

* Test + Hard: `fepack.sh stk_r2q q34att_s0.list q34att_s1.list q34att_s2.list` (j23999, ≈ 65 min).
* GSM-Plus: `fepack.sh stk_r2q q34att_gp_s0.list q34att_gp_s1.list q34att_gp_s2.list` (j24002, ≈ 60 min).
* Analysis: `ana_q32.py t`.

Results:
* test / Hard / GSM-Plus: 82.11 / 61.31 / 68.29, vs row 4 80.26 / 58.76 / 66.41.
* row 4 − 2b: −1.84 [−2.83, −0.88] / −2.55 [−3.69, −1.36] / −1.87 [−2.84, −0.94].
* Writer audit 0.
* At 6 RL seeds (§22): 2b 82.22 / 61.42 / 68.34, row 4 − 2b −1.78 [−2.50, −1.01] / −2.41 [−3.35, −1.38] /
  −2.25 [−3.04, −1.49].

Eval seeds 11/13:
* `fepack.sh stk_r2q q34att_e11.list q34att_e13.list` (j24008, ≈ 95 min; `ana_q32.py T`). Row 4 − 2b, averaged over
  seeds 7/11/13: −1.97 [−2.80, −1.20] test, −2.83 [−3.76, −1.88] Hard.
* GSM-Plus: `fepack.sh stk_r2q q34att_gp_e11.list q34att_gp_e13.list` (j24014): −2.25 [−2.99, −1.56].

#### Why row 4 trails row 2b: mechanism runs

Rung 2 at N=1 and at fixed N=8, on P64 vs CW hosts:
* `fepack.sh stk_r2q q34m_P64_s{0,1,2}.list`, and the same for `q34m_CW_s*` (j24020–21); `ana_q32.py m`.
  CW − P at N=1: −4.02 [−5.51, −2.55] test, −4.95 [−6.49, −3.44] Hard.
* On GSM-Plus: `fepack.sh stk_r2q q34m_gp_P64_s{0,1,2}.list` and `q34m_gp_CW_s*` (j24027–28). N=1 CW − P −5.19
  [−6.41, −4.00].
* Eval seeds 11/13: `fepack.sh stk_r2q q34mE_P64.list q34mE_CW.list` (j24050, ≈ 1 h 50 min); `ana_q32.py E`.
  CW − P64 at N=1 over seeds 7/11/13: −5.07 / −4.62 / −5.92.

Row 4p (CW hosts, stack removed, 1 sample):
* `fejob.sh stk q34_4p.list` (j24035, ≈ 5 min); `ana_q32.py p`.
* 4p − 2: −5.61 [−7.10, −4.14] / −7.51 [−9.05, −5.86] / −6.35 [−7.60, −5.14] on test / Hard / GSM-Plus.

Host only, at u100 / u200 (P64 / CW, aliases `ARM@100`):
* `fejob.sh stk q34_hu.list` (j24042, ≈ 20 min); `ana_q32.py h`.
* CW − P64 on test: −8.44 / −6.70 / −5.61 at u100 / u200 / u300.
* The same at rung 2 N=1: `fepack.sh stk_r2q q34mu_s0.list q34mu_s1.list q34mu_s2.list` (j24039, ≈ 57 min); `ana_q32.py u`.
  GSM-Plus: `q34mu_gp_s*` (j24057).
* GSM-Plus trajectory (j24057): CW − P64 at rung 2 N=1 is −6.63 / −6.51 / −5.19 at u100 / u200 / u300.

Host only on every base-host arm (Pb, Bb, C, C600, CW600):
* `fejob.sh stk q34_hx.list` (j24045, ≈ 17 min); `ana_q32.py x`.
* test / Hard / GSM-Plus: C − P64 −4.50 / −5.86 / −6.05; Bb − Pb −1.16 / −5.38 / −1.44.

Rung 2 on the Bb and C hosts:
* Bb: `fepack.sh stk_r2q q34r2bb_Bb_s5.list q34r2bb_Bb_s6.list q34r2bb_Bb_s7.list` (j24046, ≈ 55 min); `ana_q32.py x`.
  r2_Bb − r2_Pb −0.40 / −0.30 (test / Hard).
* C: `fepack.sh stk_r2q q34r2c_C.list q34r2c_C_s2.list q34r2c_C_s3.list` (j24047, ≈ 57 min). r2_C − r2_P64 −1.21 / −2.02,
  r2_CW − r2_C −0.63 / −0.53.
* Eval seeds 11/13: `q34r2cE_*.list` (j24062), `q34r2bE_{Bb,Pb}_*.list` (j24067–68). Seed-averaged (`ana_q32.py X`):
  r2_Bb − r2_Pb +0.08 / +0.08, r2_C − r2_P64 −1.47 / −2.54, r2_CW − r2_C −0.50 / −0.29.
* Host-only at the same eval seeds: `q34_hxE.list` (j24071). C − P64 −4.52 / −6.05 / −6.13, Bb − Pb −1.15 / −4.46 /
  −1.18.
* GSM-Plus arms: `q34r2gx_{0,1,2}.list` (j24054). Bb − Pb +0.35, r2_C − r2_P64 −1.59, r2_CW − r2_C −0.29
  (`ana_q32.py x`).

The full row-4 stack (rung 2) across updates:
* test + Hard: `q34r2u100_s*` / `q34r2u200_s*` (j24052/53). GSM-Plus: `q34r2u100gp_s*` / `q34r2u200gp_s*` (j24059/60).
  `ana_q32.py U`.
* CW − P64 at u100 / u200 / u300: test −1.44 / −1.49 / −1.84, Hard −2.78 / −2.68 / −2.55, GSM-Plus −1.71 / −1.98 /
  −1.87.

1.7B rung 2 at N=1 at eval seeds 11/13 (j24048, ≈ 1 h 45 min):
* `E4TAG=ME_ PROCS="a33_mEP17 a33_mECW17" bash uniform/rl28/gsm8k/fe/jobE4.sh`, then
  `python3 uniform/rl28/gsm8k/fe_gp/ana_v17.py` (last block).
* CW17 − P17 over seeds 7/11/13: −2.54 / −3.96 / −3.40 (test / Hard / GSM-Plus).

#### Rungs 3 / 4 vs rung 2 on the CW hosts

Averaged over eval seeds 7/11/13:
* Lists: seeds 11/13 `q34r{3,4}_s*.list`; rung 2 `q34s4_CW_s*.list` (j23965–66 / j23969–70). Analysis: `ana_q32.py s`,
  block "C16".
* rung 2 / 3 / 4: test 80.19 / 80.72 / 80.97, Hard 58.91 / 56.28 / 57.11.
* Rung 4 − rung 2 is +0.78 [+0.16, +1.37] on test and −1.79 [−2.74, −0.88] on Hard, at +21% / +28% tokens.

### 17.3 Where row 3 comes from (ablation)

Base host, GSM8K test, T 1, seed 7, once. Lists `uniform/final_eval/lists/q33_*.list`, all on `stk_r2q`, one B200 per job:
```bash
bash uniform/final_eval/fepack.sh stk_r2q q33_nowpr.list q33_fixN8.list q33_vote0.list    # q33a j23882, 28 min
bash uniform/final_eval/fepack.sh stk_r2q q33_fixN4.list q33_N1.list                      # q33b j23883, 6.5 min
python3 uniform/final_eval/q33/decomp.py; python3 uniform/final_eval/q33/deltas.py        # CPU; = uniform/final_eval/q33/deltas.txt
```
Expected (C10-C13; each step Δ is against the row above it):

| config | acc | tok/q | step Δ [95% CI] |
|---|---|---|---|
| row 1: host only | 28.05 | 103 | |
| rung 2 at N=1 (`q33_N1`) | 32.30 | 144 | +4.25 [+1.29, +7.20] |
| rung 2 at fixed N=4 (`q33_fixN4`) | 43.44 | 748 | +11.14 [+8.34, +14.03] |
| rung 2 at fixed N=8 (`q33_fixN8`) | 51.40 | 1575 | +7.96 [+5.16, +10.46] |
| row 3: rung 2, adaptive N 8 -> 16 | 57.39 | 2649 | +5.99 [+3.56, +8.42] |
| rung 2 with plain majority (`q33_vote0`) | 63.31 | 2915 | +5.91 [+3.64, +8.42] vs row 3 |
| rung 2 without WPR (`q33_nowpr`) | 13.42 | 2231 | −43.97 [−46.78, −41.02] vs row 3 |
| Base serial N=4, no WPR (`test2.list`, j23268) | 28.43 | 691 | fixed N=4 with WPR is +15.01 [+12.13, +18.04] above it |

The four steps add up to row 3 − row 1 = +29.34 [+26.38, +32.07]. Two runs give the same 28.43: Base + S2F in
`base.list` and Base N=4 in `test2.list`. They are different runs with the same count: 375/1319 each, 210 shared correct
items, +0.00 [−2.73, +2.65].

**Eval seeds 11/13 for the three step configs** (j23988, 24 min):
```bash
bash uniform/final_eval/fepack.sh stk_r2q q34a33_e11.list q34a33_e13.list
python3 uniform/final_eval/q32/ana_q32.py s
```
Averaged over seeds 7/11/13:
* N=1 32.52, fixed N=4 43.85, fixed N=8 50.97 (row 1 27.80, row 3 57.32).
* Steps: +4.73 [+2.91, +6.49], +11.32 [+9.58, +13.09], +7.13 [+5.43, +8.79], +6.34 [+4.88, +7.86].

**The same steps on GSM-Hard** (Hard protocol, plain majority; j23997, ≈ 60 min):
```bash
bash uniform/final_eval/fepack.sh stk_r2q q34a33h_e7.list q34a33h_e11.list q34a33h_e13.list
python3 uniform/final_eval/q32/ana_q32.py a
```
* row 1 11.57, N=1 16.27, fixed N=4 22.39, fixed N=8 29.16, row 3 36.52.
* N=1 − row 1: +4.70 [+3.31, +6.12].
* The e7 list also runs `q33_nowpr` on Hard: 14.10, −21.83 vs row 3.

### 17.4 Matched-compute majority-vote control

N separate host-only samples (KIND `SCH`) with a plain majority vote. N is chosen so tok/q matches the compared row:
N = 12 test / 16 Hard against row 4, N = 26 / 30 against row 3. T 1, seed 7, once per host. One B200 per job:
```bash
GPU=0 bash uniform/final_eval/fejob.sh stk q32b_a.list         # j23890, 5.5 min
GPU=0 bash uniform/final_eval/fejob.sh stk q32b_b.list         # j23891, 5.5 min
GPU=0 bash uniform/final_eval/fejob.sh stk q32b_r3m.list       # Base N=26/30, j23898, 4.5 min
python3 uniform/final_eval/q32/ana_q32.py b                    # CPU
```
Expected (3 RL seeds; C9, C9b):

| row | test | tok/q | Hard | tok/q |
|---|---|---|---|---|
| 1-MV Base, majority of 12 / 16 | 54.81 | 1213 | 25.47 | 2011 |
| 1-MV3 Base, majority of 26 / 30 | 59.97 | 2625 | 29.57 | 3777 |
| 2-MV P64 s0-s2, majority of 12 / 16 | 75.69 (76.27 / 75.13 / 75.66) | 1559 | 48.65 (49.51 / 49.13 / 47.31) | 2616 |
| 3 Base + rung 2 | 57.39 | 2649 | 35.94 | 3830 |
| 4 CW s0-s2 + rung 2 | 80.26 | 1612 | 58.76 | 2655 |

| paired effect | test | Hard |
|---|---|---|
| 4 − 2-MV | +4.57 [+3.26, +5.96] | +10.11 [+8.19, +12.08] |
| 2-MV − 2 | +7.20 [+5.99, +8.44] | +7.46 [+6.17, +8.79] |
| 3 − 1-MV3 | −2.58 [−5.00, −0.08] | +6.37 [+4.25, +8.64] |
| `q33_vote0` − 1-MV3 | +3.34 [+1.06, +5.53] | – |
| 4 − 2-MV, eval seeds 7/11/13 (`q34s4_CW_s{0,1,2}.list` j23965-66 on stk_r2q, `q34s4_mv.list` j23967) | +4.14 [+2.99, +5.29] | +10.45 [+8.62, +12.37] |
| 3 − 1-MV3, eval seeds 7/11/13 (`q34mv3_seeds.list` j23962) | −2.65 [−4.22, −0.99] | +7.86 [+6.37, +9.35] |
| 4 − 2, all rows at eval seeds 7/11/13 (`q34s2_P64.list` j23977; `ana_q32.py s`) | +11.62 [+10.32, +12.95] | +17.56 [+15.78, +19.25] |

### 17.5 Eval-seed spread

One B200 per job:
```bash
GPU=0 bash uniform/final_eval/fejob.sh stk q32cd_h.list                                  # row 1 seeds 11/13 (+ GSM-Plus rows 1/2), j23892, 4 min
GPU=0 bash uniform/final_eval/fepack.sh stk_r2q q32c_r2_s11.list q32c_r2_s13.list        # row 3, q32r2s j23879, 44 min
GPU=0 bash uniform/final_eval/fepack.sh stk_r3 q32c_r3_s11.list q32c_r3_s13.list         # row 3', q32r3s j23881, 55 min
python3 uniform/final_eval/q32/ana_q32.py c
```
Expected (C33), eval seeds 7 / 11 / 13:

| row | test | mean ± sd | Hard | mean ± sd |
|---|---|---|---|---|
| 1 Base | 28.05 / 26.76 / 28.58 | 27.80 ± 0.94 | 10.46 / 12.28 / 11.98 | 11.57 ± 0.97 |
| 3 Base + rung 2 | 57.39 / 56.48 / 58.07 | 57.32 ± 0.80 | 35.94 / 36.69 / 36.92 | 36.52 ± 0.52 |
| 3' Base + rung 3 | 56.03 / 54.81 / 56.41 | 55.75 ± 0.83 | 25.55 / 26.54 / 24.87 | 25.65 ± 0.84 |

### 17.6 Fresh set: GSM-Plus testmini

GSM-Plus testmini: the 7 answerable perturbation types, 300 items each (2100 items). T 1, seed 7, once. Nothing was
tuned on this set. Build the data once (CPU, pinned revision):
```bash
python3 uniform/final_eval/data/prepare_gsmplus.py    # -> data/gsmplus/testmini.jsonl
```
Then run the evals, one B200 per job:
```bash
GPU=0 bash uniform/final_eval/fejob.sh stk q32cd_h.list                     # rows 1/2, j23892
GPU=0 bash uniform/final_eval/fepack.sh stk_r2q q32d_r2a.list q32d_r2b.list # rows 3/4, q32gp j23880, 58 min
GPU=0 bash uniform/final_eval/fejob.sh stk q32bd_gp.list                    # majority rows, j23901, 10.5 min
GPU=0 bash uniform/final_eval/fejob.sh stk q34gp_mv16.list                  # 2-MV at matched GSM-Plus tokens (N=16), j23946, 6 min
python3 uniform/final_eval/q32/ana_q32.py d
```
Expected (3 RL seeds; C15):

| row | GSM-Plus | tok/q |
|---|---|---|
| 1 Base | 18.10 | 119 |
| 2 P64 s0-s2 | 54.03 (54.38 / 53.71 / 54.00) | 155 |
| 3 Base + rung 2 | 38.24 | 3282 |
| 4 CW s0-s2 + rung 2 | 66.41 (66.57 / 67.05 / 65.62) | 2446 |
| 2-MV P64 + majority of 12 | 63.60 (64.14 / 63.10 / 63.57) | 1859 |
| 2-MV16 P64 + majority of 16 | 64.00 (65.48 / 62.95 / 63.57) | 2478 |
| 1-MV Base + majority of 12 / of 26 | 38.33 / 44.48 | 1433 / 3120 |

| paired effect | GSM-Plus |
|---|---|
| 2 − 1 | +35.94 [+33.90, +37.87] |
| 3 − 1 | +20.14 [+18.05, +22.29] |
| 4 − 2 | +12.38 [+10.89, +13.89] |
| 4 − 3 | +28.17 [+26.19, +30.24] |
| 4 − 2-MV (2-MV uses 24% fewer tokens) | +2.81 [+1.43, +4.06] |
| 4 − 2-MV16 (matched tokens) | +2.41 [+1.10, +3.75] |
| 4 − 2-MV16, eval seeds 7/11/13 (`q34gs_*.list`, j23955-57; `ana_q32.py s`) | +2.28 [+1.04, +3.44] |
| 3 − 1-MV3 (majority of 26) | −6.24 [−8.33, −4.33] |
| 3 − 1-MV3, eval seeds 7/11/13 (`q34gs_r3_e1{1,3}.list` j23972, `q34gp_mv3.list` j23981) | −5.78 [−7.13, −4.40] |
| 4 − 2 / 4 − 3, all rows at eval seeds 7/11/13 (`q34gs_h.list` j23973; rows 18.65 / 54.18 / 38.81 / 66.26) | +12.08 [+10.75, +13.32] / +27.46 [+25.79, +29.12] |

At 6 RL seeds (§22): row 2 53.67, row 4 66.09, 4 − 2 +12.42 [+11.05, +13.74].

Writer-audit misses in j23880: 1, 3, 0, 1 per ~220k-306k steps.

GSM-Plus with plain majority for rows 3 and 4 (`q34gv_*.list`, jobs j23911, j23912):
```bash
bash uniform/final_eval/fepack.sh stk_r2q q34gv_CW_s0.list q34gv_CW_s1.list
bash uniform/final_eval/fepack.sh stk_r2q q34gv_CW_s2.list q34gv_base.list
python3 uniform/final_eval/q32/ana_q32.py v
```
* Row 4: 64.78 (64.33 / 66.81 / 63.19), −1.63 [−2.46, −0.83] vs the probe voter.
* Row 3: 44.86, +6.62 [+4.76, +8.48] vs the probe voter.
* Row 4 − row 3, both on plain majority: +19.92 [+17.89, +21.83].
* Writer audit: 1 / 0 / 1 / 1 misses per ~228k-348k steps.

The earlier GSM-Plus confirmatory set (`gp_r2.list`, `gp_r3.list`, `gp_stk.list` in `uniform/final_eval/lists/`, jobs
j23585 / j23583 / j23586) used the u0 host, not the uniform rows.

### 17.7 IFEval rows

* Implant (rows 3, 4, 2b): the full14 compiled 14-rule circuit with the frozen Qwen3-0.6B LM-head snapshot readout
  (`sd_inst`, TAG `inst`).
* RL data: Team D `gen_data` seed 200 (18,219 rows) and dev seed 201 (415 rows), filtered against the 541.
* Reward = official prompt-level strict.
* Code: `uniform/rl24/ifeval/` (`rl_chain.sh`, `pack_chain.sh`, `code/eng/`). Complete `code/` once with
  `bash uniform/rl24/ifeval/code/setup.sh` (CPU, ~2 min). It copies the shared release files and the 541 prompts, and
  regenerates the RL train / dev data; their sha256 must match `code/gen/filter.log`.

All commands run on one B200:
```bash
cd uniform/rl24/ifeval
bash code/eng/job_eval.sh u0 Qwen/Qwen3-0.6B-Base inst '' chat               # rows 1 and 3 + idle check, j23721, 2m49s
env MODEL=Qwen/Qwen3-0.6B-Base TAG=inst FMT=chat bash rl_chain.sh P_s0         # row 2 (also P_s1); chain driver, 4 h per start
env MODEL=Qwen/Qwen3-0.6B-Base TAG=inst FMT=chat bash rl_chain.sh I_s0         # row 4 (also I_s1); I_s0 took 1h53m alone
bash pack_chain.sh pkA "P_s2 I_s2 degen P_s0x"                                 # seed 2 of both + row 2b re-eval; j23782, 57 min
python3 analyze.py --out RESULTS_TABLE.md                                      # table + rows.json (CPU; reads res/, AO=out for your runs)
```
Jobs: P_s0 j23683, I_s0 j23722, P_s1 j23745, I_s1 j23744. Speed: P arms 4.5 s/update (median), I arms about 17.6 s/update.

Expected (3 RL seeds; C22, C23):

| row | prompt strict [95% CI] | prompt loose | inst strict | tok/q | per seed |
|---|---|---|---|---|---|
| 1 Base | 12.94 [10.17, 15.90] | 13.12 | 23.98 | 1889 | |
| 2 + RL | 43.56 [39.86, 47.44] | 45.96 | 56.59 | 573 | 43.62 / 46.77 / 40.30 |
| 3 + implant | 34.75 [30.68, 38.82] | 36.78 | 47.48 | 1388 | |
| 4 + implant + RL | 48.37 [44.49, 52.19] | 52.87 | 60.35 | 1364 | 49.17 / 48.24 / 47.69 |
| 2b row-2 hosts + implant at eval only | 52.68 [48.92, 56.44] | 55.51 | 64.51 | 546 | 54.16 / 54.71 / 49.17 |

Paired effects (prompt strict):
* 2 − 1: +30.62 [+26.68, +34.50]
* 3 − 1: +21.81 [+18.30, +25.51]
* 4 − 2: +4.81 [+1.11, +8.56]
* 4 − 3: +13.62 [+10.41, +16.94]
* 2b − 4: +4.31 [+1.17, +7.33]

Idle bit-identity (implant attached but never firing = token-identical output) holds on every RL'd host.

**At 6 RL seeds** (§22, `checks/seeds35/IFEVAL.md`; these are the blog's numbers, "3 on IFEval", "48% vs 46%"):
* row 2 45.53 [41.99, 49.20], row 4 48.49 [44.79, 52.22], row 2b 53.70 [50.09, 57.30].
* 4 − 2: +2.96 [−0.28, +6.35], 5/6 seeds positive. The CI now includes 0.
* 2b − 2: +8.16 [+6.13, +10.38].
* 2b − 4: +5.21 [+2.46, +7.89], 6/6 seeds positive.

Row 2b > row 4 holds at both seed counts. Rows 1 and 3 on the BBH call-FT host instead of Base: task [29],
`python3 uniform/rl24/ifeval/analyze29.py` (`res/ftint/`).

### 17.8 BBH rows and the per-task check

* Implants (rows 3, 4): the 8 tools (`integ.TOOLS_V2`) + the call gate refit on `ft_int_v2`
  (`uniform/rl24/bbh/cg/job_refit.sh all`, j23625 -> `$RL24_DATA/bbh/cg/probe_ft2.pt`, held-out AUROC 0.9999).
* Code: `uniform/rl24/bbh/w/` (`job.sh`, `chain.sh`, `final_eval.sh`, `pack.sh`, `final_ana.sh`).
* Needs `FT2_HOST` (the §6.1 call-FT host), the BBH data (`bash data/download_bbh.sh`; `BBH_DIR`, `BBH_COT`) and the
  0.6B calc veto (`WSM_VETO`).

All commands run on one B200:
```bash
cd uniform/rl24/bbh/w; D=$RL24_DATA/bbh
bash job.sh data                                       # RL train / dev look-alike data (CPU)
bash ../cg/job_refit.sh all                            # gate refit -> $D/cg/probe_ft2.pt (j23625)
bash chain.sh F0_s0 ft2 0 - 0                          # row 2: ft_int_v2, tools off; seeds 0/1/2 (F0_s1 ... 1, F0_s2 ... 2)
bash chain.sh I0_s0 ft2 1 $D/cg/probe_ft2.pt 0         # row 4: ft_int_v2 + 8 tools + gate
bash chain.sh P0_s0 base 0 - 0                         # row 2': plain Base
# chain.sh reruns `bash job.sh rl RUN HOST TOOLS GATE SEED` until out/RUN/DONE (resumes from $D/runs/RUN/resume.pt)
bash job.sh eval base_plain base 0 - - --B_max 128                     # row 0
bash job.sh eval ft2_t0_u0 ft2 0 - - --B_max 128                       # row 1
bash job.sh eval ft2_tools_gate ft2 1 $D/cg/probe_ft2.pt - --B_max 128 # row 3
bash final_eval.sh                                     # u300 evals of every finished arm (rows 2, 2', 4, ctl); packed originals: pack.sh, j23773, j23793
bash final_ana.sh                                      # CPU (j23805) -> out/table_final.md, out/rows.json; E=out for your evals, default ../res
python3 ../../../final_eval/q32/bbh_pertask.py > ../../../final_eval/q32/bbh_pertask.md   # CPU
```
Speed: P0 6.5 s/update, F0 4.3 s/update, I0 12.0 s/update. One BBH-27 eval alone at B 128 takes about 2.5 min. With the
shipped reduced rows (`uniform/rl24/bbh/res/`), `final_ana.sh` rebuilds `table_final.md` and `rows.json` on a CPU.

Expected (3 RL seeds; C17-C21):

| row | config | BBH-27 mean [95% CI] | 12 tool tasks | 15 other | tok/q | per seed |
|---|---|---|---|---|---|---|
| 0 | plain Base | 41.19 [40.08, 42.28] | 43.47 | 39.37 | 222 | |
| 1 | ft_int_v2, tools off | 32.59 [31.52, 33.65] | 36.57 | 29.40 | 135 | |
| 2 | + RL, tools off (F0) | 42.27 [41.41, 43.12] | 56.34 | 31.01 | 144 | 41.14 / 43.15 / 42.52 |
| 2' | plain Base + RL (P0) | 53.63 [52.64, 54.61] | 70.44 | 40.18 | 201 | 53.91 / 54.09 / 52.90 |
| 3 | + 8 tools + gate | 54.99 [54.05, 55.92] | 86.80 | 29.54 | 134 | |
| 4 | + tools + gate, RL with tools on (I0) | 60.24 [59.50, 60.97] | 97.64 | 30.32 | 140 | 60.31 / 60.11 / 60.29 |
| ctl | row-2 weights + tools + gate at eval only | 57.62 [56.85, 58.39] | 91.09 | 30.84 | 139 | 57.08 / 58.37 / 57.41 |

Paired effects (27-task mean):

| effect | value |
|---|---|
| 2 − 1 | +9.68 [+8.64, +10.70] |
| 3 − 1 | +22.40 [+21.45, +23.37] |
| 4 − 2 | +17.97 [+17.19, +18.75] |
| 4 − 3 | +5.25 [+4.38, +6.11] |
| 4 − ctl | +2.62 [+1.99, +3.24] |
| 4 − 2' | +6.61 [+5.59, +7.63] (the blog's "7 on BIG-Bench Hard", "60% vs 54%"); non-tool tasks −9.86 |
| 1 − 0 (the call fine-tune) | −8.61 [−10.01, −7.20]; non-tool −9.97 |

Per-task check (row 4 vs row 1; tasks more than 3 points below; 10,000-resample paired bootstrap per task):
* causal_judgement −7.0 [−14.4, +0.4]
* formal_fallacies −4.8 [−10.3, +0.7]
* ruin_names −3.1 [−8.7, +2.5]

All three are non-tool tasks, and every CI includes 0. All 12 tool tasks gain, by +17.7 to +87.7.

**Noise.** The GSM8K rows are single T 1 runs at eval seed 7. The eval-seed sd is under 1 point (§17.5), and row 4
reproduces on a later build within +0.15 [−0.56, +0.88]. The RL seed spread is the per-seed column. 6 RL seeds: §22.
bf16 batch-shape numerics move a few items between B_max settings and GPU types, as in §15.

### 17.9 What is shipped

Everything is under `uniform/`; see `uniform/README.md` for the full layout. No checkpoints and no generated text are
shipped.

* `uniform/env.sh`, `uniform/items.py`.
* `uniform/final_eval/`:
  * `fe.sh`, `fejob.sh`, `fepack.sh`, `r2cwjob.sh`, `ana_uni.py`;
  * `q32/` (`ana_q32.py`, `bbh_pertask.py` + `.md`, `mech_2b.py`) and `q33/` (`decomp.py`, `deltas.py` + `deltas.txt`);
  * `data/prepare_gsmplus.py`;
  * `lists/` (`base`, `test2` and `wpr_r2cw` are in `gsm_stack/eval/lists/`);
  * the code-tree overlays `trees/{stk,stk_r2,stk_r2q,stk_r3,stk_r4}`;
  * the reduced per-item rows `items/*.items.jsonl.gz` (answer, correctness and token counts per item and attempt; no
    text).
* `gsm_stack/engine/run_arm.sh`: the uniform arms `P64_sK`, `CW_sK`, `A1WL_sK`, `A1WD_sK` and `LRP_/LRC_/LRQ_/LRD_<lr>`.
* `uniform/rl24/`: `HOST.md`, `make_uniform_json.py`, `uniform.json`.
* `uniform/rl24/ifeval/`: `RESULTS.md`, `RESULTS_TABLE.md`, `rows.json`, `analyze.py`, `analyze29.py`, `rl_chain.sh`,
  `pack_chain.sh`, `code/` (+ `setup.sh`), and `res/` (per-prompt scores, no text).
* `uniform/rl24/bbh/`: `RESULTS.md`, `table_final.md`, `rows.json`, `w/*.py`, `w/*.sh`, `cg/job_refit.sh` + `cg/code/`,
  and `res/` (reduced per-item rows).
* `uniform/rl28/{gsm8k,ifeval,bbh}/`: see §18.

**Not reproducible from the repository alone:**
* the RL checkpoints of every arm above;
* `ft_int_v2` (weights; its recipe is in §6);
* the per-item outputs with generated tokens (see §14).

## 18. Qwen3-1.7B-Base, same implants (blog §6)

Blog claims backed here: the §17 rows on Qwen3-1.7B-Base with the same 0.6B implants (blog: "90% vs 81% when attached
after plain RL" = row 2b vs row 2 on GSM8K, §18.1). Only the host-tied readers and writers are refit, and idle
bit-identity is re-checked.

* Numbers: `uniform/rl28/{gsm8k,ifeval,bbh}/RESULTS.md`; claims C24-C28.
* Same hardware (one B200 per job), RL recipe and eval settings as §17.
* Code: `uniform/rl28/`. It holds patched copies of the §17 code (host from env `HOST_MODEL` / `HOST_SIZE` / `MODEL`),
  and the 1.7B refits ship inside the overlays.
* The 1.7B RL checkpoints are 3.4 GB bf16 per arm.

### 18.1 GSM8K

What was refit (`uniform/rl28/gsm8k/RESULTS.md`; circuits unchanged, nothing new climbed):
* Readers (calculator / chain2 tables, tk symbol tables): closed form from the 1.7B embedding at attach time.
* Writers: re-solved on the 1.7B final norm + LM head at attach time, margin target `SW_DMARG=40`.
* End-of-number veto: `results/gsm_bigger/veto_q17.pt` (shipped, §8).
* Reserved stop row: stopcal on 1.7B-Base, s = 27.9 (`trees/stk_r2q_17/stk/stoprow.pt`). The pad-row guard is relaxed via
  `SW_PAD_NONDEG_OK=1`, which `fe/run_hook.sh` sets for 1.7B.
* L18 verifier probe: refit on 11,992 1.7B-Base samples of GSM8K train split C, within-question AUROC 0.86
  (`stk/verifier_probe_L18_q17.pt` in the overlays `trees/arm17` and `trees/stk_r2q_17`).
* Checks:
  * idle bit-identity: 73/73 rows token-identical in full, and 527/527 identical up to the first implant step;
  * calc-v2 stack silence: 600/600 and 595/595;
  * forced-token audit of row 3: 0 sampled != written in 151,900 test + 206,955 Hard writer steps;
  * RL fast-check: 600/600 token-identical.

The refit and setup jobs (row 1 + refits j23758, redone in j23795; row 3 j23818, lists `r3_test` / `r3_hard`) are not
shipped as scripts. Their outputs ship in the overlays. Rows 1 and 3 are rerun with
`PROCS="a r3_test r3_hard" bash uniform/rl28/gsm8k/fe/jobE4.sh`. `uniform/rl28/gsm8k/patch17.py` documents the
`HOST_MODEL` patch of the original copies.

All commands run on one B200:
```bash
cd uniform/rl28/gsm8k
MEMFRAC=0.45 BMAX=192 bash eng/pack_17.sh P17a "P17_s0 P17_s1"                    # row 2 (P64 recipe), j23766, 2 arms on one GPU
MEMFRAC=0.49 BMAX=96 FT_BUDGET=16384 bash eng/pack_17.sh CW17a "CW17_s0 CW17_s1"  # row 4, j23802; stopped at u113/u126 and resumed
                                                                                  #   alone: CW17b "CW17_s0" (j23864), CW17c "CW17_s1" (j23863)
MEMFRAC=0.49 BMAX=96 FT_BUDGET=16384 bash eng/pack_17.sh S2 "CW17_s2 P17_s2"      # j23832; FT_BUDGET 16384 for CW17, 32768 for P17
#   one arm alone: GPU=0 bash eng/run_arm_17.sh P17_s0; output $ARMDIR17/rl/<ARM>/step_000300
bash evchain.sh             # row-2/4 evals: PROCS=... bash fe/jobE4.sh (lists p17_s*, cw17_s{0,1,2}_{test,hard}); then mkrows.py + mkresults.py
bash q32chain.sh            # mk32.py -> fe/lists/q32b_{test,hard}.list (j23937), and row 1 / row 3 at eval seeds 11, 13 (j23906)
python3 mkresults.py        # RESULTS.md (+ mkrows.py -> rows.json); CPU, from the shipped rows or FE_ROWS=fe/out/fe
```
* Row-2 evals: P17_s0/s1 j23831, P17_s2 j23861 (2 min).
* P17 arms run at about 9-12 s/update packed, CW17 at about 55-60 s/update alone.
* The 1.7B row-3 list line is the 0.6B `r2_base` line plus `SW_DMARG=40`.

Expected (3 RL seeds; C27):

| row | config | test (per seed) | Hard (per seed) | tok/q test / Hard | 0.6B test / Hard |
|---|---|---|---|---|---|
| 1 | Qwen3-1.7B-Base | 45.26 | 24.49 | 106 / 138 | 28.05 / 10.46 |
| 2 | P17 s0/s1/s2 | 80.62 (80.36 / 80.21 / 81.27) | 56.63 (56.10 / 56.94 / 56.86) | 118 / 152 | 68.49 / 41.19 |
| 3 | 1.7B-Base + rung 2 | 83.47 | 57.01 | 2232 / 3531 | 57.39 / 35.94 |
| 4 | CW17 s0/s1/s2 + rung 2 | 87.89 (87.79 / 88.25 / 87.64) | 67.55 (67.32 / 67.25 / 68.08) | 1853 / 3090 | 80.26 / 58.76 |

| effect | test | Hard |
|---|---|---|
| RL alone (2 − 1) | +35.36 [+32.78, +38.11] | +32.15 [+29.67, +34.52] |
| implants alone (3 − 1) | +38.21 [+35.41, +41.02] | +32.52 [+29.87, +35.25] |
| both (4 − 1) | +42.63 [+39.95, +45.36] | +43.06 [+40.41, +45.59] |
| RL given implants (4 − 3) | +4.42 [+2.86, +5.99] | +10.54 [+8.62, +12.51] |
| implants given RL (4 − 2) | +7.28 [+5.96, +8.59] | +10.92 [+9.05, +12.71] |
| interaction (4 − 3 − 2 + 1) | −30.93 [−33.91, −27.87] | −21.61 [−24.69, −18.60] |

**Further runs.** Every packed eval below is
`env E4TAG=TAG PROCS="LIST+LIST LIST" bash uniform/rl28/gsm8k/fe/jobE4.sh` on one B200. All lists are in
`uniform/rl28/gsm8k/fe/lists/`. The analyses run on a CPU, from the shipped rows (`uniform/rl28/gsm8k/items/`, through
`uniform/items.py`) or from your own (`FE_ROWS`):
* `python3 uniform/rl28/gsm8k/fe_gp/ana_v17.py` (below: `ana_v17.py`)
* `python3 uniform/rl28/gsm8k/fe_gp/ana_gp17.py` (below: `ana_gp17.py`)

#### Write audit and matched compute

* **Row-4 write audit** (E4a j23915, E4b j23918). 0 sampled != written in 107,550 / 110,174 / 116,795 test and
  150,019 / 138,901 / 162,963 Hard writer steps (CW17 s0 / s1 / s2).
* **Matched-compute control** (`q32chain.sh`, j23937 rl28g_Q32b). N = round(row-4 tok/q / row-2 tok/q), on
  P17 s0-s2 and Base17:
  * N = 16 for test (1853 / 117.8 tok/q) and 20 for Hard (3090 / 152.4).
  * 2-MV: 86.56 (86.43 / 86.13 / 87.11) test, 62.75 (63.08 / 62.24 / 62.93) Hard. 1-MV: 78.62 / 48.14.
  * **4 − 2-MV: +1.34 [+0.18, +2.53] test, +4.80 [+3.08, +6.47] Hard.** 2-MV − 2: +5.94 / +6.12.
* **Row-3 matched compute** (lists `q32b3_{test,hard}`, j23928 / j23929). Base with
  N = round(row-3 tok/q / row-1 tok/q) samples + majority:
  * N = 21 (test) / 26 (Hard): 79.15 / 49.66 (2229 / 3592 tok/q).
  * 3 − 1-MV3: +4.32 [+2.50, +6.22] / +7.35 [+5.23, +9.55] (`ana_v17.py`).

#### Eval-seed spread (seeds 7 / 11 / 13; j23906 rl28g_Q32c)

| row | test | sd | Hard | sd |
|---|---|---|---|---|
| 1 | 45.26 / 47.08 / 46.10 | 0.91 | 24.49 / 24.87 / 24.18 | 0.34 |
| 3 | 83.47 / 83.62 / 83.40 | 0.12 | 57.01 / 55.50 / 54.89 | 1.09 |
| 4 (CW17_s2; j23926 / j23934) | 87.64 / 88.63 / 88.02 | 0.50 | 68.08 / 68.23 / 67.48 | 0.40 |

Row 4 and 2-MV averaged over 3 RL × 3 eval seeds (`ana_v17.py`):

| set | lists (jobs) | row 4 | 2-MV | 4 − 2-MV |
|---|---|---|---|---|
| test | `v17_seed1{1,3}_h{0,1}`, `q32b3_mv_e1{1,3}` (j23943-45) | 87.90 | 86.50 | **+1.40 [+0.38, +2.38]** |
| Hard | `v17_hseed1{1,3}`, `q32b3_hmv_e1{1,3}` (j23952 / j23953) | 67.53 | 62.87 | **+4.67 [+3.10, +6.15]** |
| GSM-Plus | `gp17_cw_s*_e1{1,3}`, `q32b3_gpmv_e1{1,3}` (j23949-51) | 75.50 | 75.22 | **+0.28 [−0.76, +1.30]** |

* Row 3 / Base × 24 on GSM-Plus (`v17_gr3_e1{1,3}`, `q32b3_gpb24_e`; j23959): 65.29 / 66.29, 3 − 1-MV3 −1.00 [−2.37, +0.41].
* 1-MV3 at eval seeds 11/13 (`q32b3_mv3_e`, j23961): row 3 − 1-MV3, averaged over seeds 7/11/13, +4.35 [+3.13, +5.56]
  test, +6.04 [+4.62, +7.58] Hard.
* All four rows at eval seeds 7/11/13 (row 2 seeds: `gp17_h_e` j23974, `p17_seeds` j23976; `ana_v17.py`):

  | | row 1 | row 2 | row 3 | row 4 | 4 − 2 | 4 − 3 |
  |---|---|---|---|---|---|---|
  | test | 46.15 | 80.59 | 83.50 | 87.90 | +7.31 [+6.24, +8.35] | +4.41 |
  | Hard | 24.51 | 56.53 | 55.80 | 67.53 | +11.00 [+9.49, +12.54] | +11.73 |
  | GSM-Plus | 34.73 | 67.53 | 65.29 | 75.50 | +7.97 [+6.92, +9.06] | +10.22 |

  All 4 − 3 CIs are above 0.

#### GSM-Plus testmini on 1.7B

2100 items; lists `gp17_*`, `q32b3_gp_*`; j23919-21, j23935-36; `ana_gp17.py`.
* Rows 1-4: 34.33 / 67.44 (67.90 / 67.62 / 66.81) / 64.14 / 75.51 (75.90 / 74.81 / 75.81).
* 4 − 2 +8.06 [+6.83, +9.33]; 4 − 3 +11.37 [+9.56, +13.17]; 2 − 3 +3.30 [+1.48, +5.02].
* Matched compute: 2-MV (P17 × 19) 75.38, **4 − 2-MV +0.13 [−1.05, +1.24]**; Base × 24 66.00, 3 − 1-MV3 −1.86
  [−3.86, +0.05].
* Write audit 0 on all rung-2 runs.

#### Plain majority instead of the probe voter

Lists `v17_t_*`, `v17_g_*` (VM 0); j23924 / j23925; `ana_v17.py`.

| row | set | plain majority | vs the probe voter |
|---|---|---|---|
| 4 | test | 87.72 (87.49 / 88.48 / 87.19) | −0.18 [−0.76, +0.40] |
| 4 | GSM-Plus | 75.16 | −0.35 [−0.90, +0.24] |
| 3 | test | 82.87 | −0.61 [−2.27, +1.06] |
| 3 | GSM-Plus | 64.86 | +0.71 [−0.81, +2.33] |

#### Rung-2 ablation on 1.7B Base

Lists `a33_*` (test) and `a33h_*` (Hard); j23932 / j23933 / j23940; `ana_v17.py`. Test / Hard:

| config | seed 7 | averaged over eval seeds 7/11/13 |
|---|---|---|
| N=1 | 51.10 / 28.81 | 50.11 / 28.48 |
| fixed N=4 | 72.10 / 39.65 | 72.00 / 39.17 |
| fixed N=8 | 79.08 / 50.57 | 79.43 / 50.01 |
| row 3 | 83.47 / 57.01 | |
| minus WPR | 50.27 / 30.55 (−33.21 [−35.86, −30.63] / −26.46 [−29.19, −23.81] vs row 3) | |

* The seed average uses `a33{,h}_seeds_e1{1,3}` (j23986). N=1 − row 1: +3.97 [+2.20, +5.79] / +3.97 [+2.35, +5.59].
* GSM-Plus seed 7 (`a33_gp`, j23993): N=1 37.43 (+3.10 [+0.81, +5.38] vs row 1), fixed N=4 53.67, fixed N=8 60.24,
  row 3 64.14.
* GSM-Plus averaged over eval seeds 7/11/13 (+ `a33_gp_e11`, `a33_gp_e13`, j23998), row 1 / N=1 / N=4 / N=8 / row 3:
  34.73 / 37.89 / 53.98 / 60.51 / 65.29. N=1 − row 1 +3.16 [+1.84, +4.46], row 3 − row 1 +30.56 [+28.89, +32.22].

#### Row 2b, "RL plain, attach after" (the blog's "90% vs 81%")

P17 hosts + the untrained rung-2 stack, with the same settings as row 4.
* Lists `att17_s{0,1,2}` (test + Hard) and `att17_gp_s{0,1,2}`. Run
  `env E4TAG=Att1_ PROCS="att17_s0 att17_s1" bash uniform/rl28/gsm8k/fe/jobE4.sh`, and likewise `att17_s2 att17_gp_s0`
  and `att17_gp_s1 att17_gp_s2` (j24000 / j24001 / j24003, ≈ 45 min each).
* Analysis: `ana_v17.py`, block "[34] 'attach after RL'" (the bracketed numbers in script output are batch labels, see `uniform/README.md`).

Results:
* test / Hard / GSM-Plus: 89.94 / 70.51 / 79.02 (row 2: 80.62 / 56.63 / 67.44).
* Row 4 − 2b: −2.05 [−3.08, −1.09] / −2.96 [−4.22, −1.67] / −3.51 [−4.49, −2.62].
* 2b − 2-MV: +3.39 / +7.76 / +3.63, at 57–70% of 2-MV's tokens.
* Writer audit 0.

Eval seeds:
* Seeds 11/13: lists `att17_s{0,1,2}_e{11,13}`, run `env E4TAG=Ats0_ PROCS="att17_s0_e11 att17_s0_e13" bash uniform/rl28/gsm8k/fe/jobE4.sh`,
  and likewise s1 / s2 (j24009–11). Averaged over seeds 7/11/13, row 4 − 2b is −2.20 [−3.10, −1.39] test and −3.20
  [−4.30, −2.09] Hard.
* GSM-Plus seeds: lists `att17_gp_s{0,1,2}_e{11,13}` (j24015–17). Row 4 − 2b −3.78 [−4.62, −3.00].

Mechanism:
* Per-attempt split of 4 − 2b, both sizes, CPU only, from the seed-7 rows: `python3 uniform/final_eval/q32/mech_2b.py`.
* On GPU: `env E4TAG=M0_ PROCS="a33_mP17_s0 a33_mCW17_s0" bash uniform/rl28/gsm8k/fe/jobE4.sh`, likewise s1 / s2
  (j24022–24); `ana_v17.py` block "mechanism".
* CW17 − P17 at N=1: −2.25 [−3.41, −0.96] test, −3.51 [−5.00, −2.05] Hard.

Row 4p (CW17 with the stack removed, 1 sample):
* `env E4TAG=P4_ PROCS=p17_4p bash uniform/rl28/gsm8k/fe/jobE4.sh` (j24036, ≈ 7 min); `ana_v17.py` block "row 4p".
* 4p − 2: −5.18 [−6.55, −3.82] / −6.87 [−8.42, −5.38] / −5.52 [−6.75, −4.30] on test / Hard / GSM-Plus.

### 18.2 IFEval

* Implant: the same full14 circuit.
* `force_fix` is re-solved from the 1.7B-Base host's own plain dev answers (1190 rows fixed). The LM-head copy and final
  norm are exported from the host.
* The 1.7B-Base readout was chosen on the IFEval-like DEV set only (gen dev seed 201, 415 prompts): 50.84, against 43.61
  for the Qwen3-1.7B chat readout and 12.29 without the implant.
* Code: `uniform/rl28/ifeval/`. `code/` holds the 1.7B changes; complete it with `bash uniform/rl28/ifeval/code/setup.sh`
  (CPU, after `uniform/rl24/ifeval/code/setup.sh`). `envs/{P,I}_s*.env` default MODEL to `Qwen/Qwen3-1.7B-Base`.

All on one B200:
```bash
cd uniform/rl28/ifeval
bash code/eng/job_setup17.sh          # force_fix solve, the two exports, DEV choice, u0 evals (GPU part j23834, 0h15)
bash pack_chain.sh pkA "P_s0 P_s1"   # j23828, 1h10
bash pack_chain.sh pkB "I_s0 I_s1"   # j23840, 2h33
bash pack_chain.sh pkC "P_s2 I_s2"   # j23842, 1h54
python3 analyze.py --out RESULTS_TABLE.md   # CPU; reads res/ (AO=out for your runs)
```
Packed 2 arms per GPU (ARM_FRAC 0.46, B_max 256), 90-97% SM.

Expected (3 RL seeds; C24-C26):

| row | prompt strict [95% CI] | prompt loose | inst strict [95% CI] | tok/q | per seed | 0.6B |
|---|---|---|---|---|---|---|
| 1 | 17.19 [14.05, 20.52] | 18.48 | 31.29 [28.31, 34.28] | 1467 | | 12.94 |
| 2 | 62.97 [59.33, 66.54] | 65.99 | 73.30 [70.87, 75.75] | 390 | 61.92 / 62.48 / 64.51 | 43.56 |
| 3 | 35.12 [31.05, 39.19] | 37.71 | 48.92 [45.44, 52.39] | 1440 | | 34.75 |
| 4 | 74.68 [71.35, 78.00] | 76.96 | 82.21 [79.91, 84.46] | 595 | 74.86 / 73.57 / 75.60 | 48.37 |
| 2b | 72.34 [69.01, 75.72] | 75.11 | 80.42 [78.12, 82.68] | 434 | 71.90 / 71.53 / 73.57 | 52.68 |
| 4p row-4 host, implant detached | 59.58 [55.82, 63.34] | 61.18 | 70.14 [67.46, 72.76] | 579 | 59.33 / 59.15 / 60.26 | 30.31 |

Paired effects (prompt strict):
* 2 − 1: +45.78 [+41.34, +50.28]
* 3 − 1: +17.93 [+14.60, +21.44]
* 4 − 2: +11.71 [+8.19, +15.16]
* 4 − 3: +39.56 [+35.06, +44.05]
* 2b − 4: −2.34 [−5.18, +0.43]
* 4p − 2: −3.39 [−6.65, −0.18]

Idle bit-identity: 355/355 at u0. At u300: P_s0 370/370, P_s1 372/372, P_s2 370/370, I_s0 366/366, I_s1 366/366,
I_s2 369/369.

### 18.3 BBH

Host for rows 1-4: `ft17` = Qwen3-1.7B-Base + the same call fine-tune recipe as `ft_int_v2`, retrained on 1.7B. It takes
100 s on one B200 (`bash uniform/rl28/bbh/w/job.sh ft` -> `$RL28_DATA/bbh/ft17`). Rows:
* 0: plain 1.7B-Base
* 1: ft17, tools off
* 2: + RL, tools off (F17 x3)
* 2': 1.7B-Base + RL (P17 x3)
* 3: ft17 + 8 tools + gate refit on ft17 (`probe_ft17.pt`, `uniform/rl28/bbh/cg/job_refit.sh`)
* 4: + tools + gate + RL (I17 x3)
* ctl: F17 u300 + tools + gate at eval

Same RL data as 0.6B (`$RL24_DATA/bbh/data`; `bash uniform/rl24/bbh/w/job.sh data`). The 1.7B calc veto is
`results/gsm_bigger/veto_q17.pt`. Code: `uniform/rl28/bbh/w/`, copies of `uniform/rl24/bbh/w` with `HOST_SIZE=1.7B`.
`job.sh` copies the unchanged files from `uniform/rl24/bbh/w` and `gsm_stack/engine` at run time.

All on one B200:
```bash
cd uniform/rl28/bbh
bash setup.sh      # call-FT ft17 + gate refit, rows 0, 1, 3, ctl + host / exactness / idle checks (packed, j23812, 1h05)
bash setup2.sh     # optional: redo of the checks/evals + probe re-derivation (j23836, 52m41s; w/cmp_probe.py)
cd w     # RL arms and u300 evals as packed jobs: bash pack.sh TAG MEMFRAC 'bash job.sh rl RUN HOST TOOLS GATE SEED ...' 'bash job.sh eval ...' ...
         # original packs: P17a j23813, F17a j23847, I17a j23848, S2a j23853, E1 j23875, E2 j23886
bash drive.sh I17b I17_s0 I17_s1           # resume + u300 eval chain after the time budget (j23908), likewise I17c for I17_s2
bash final_ana.sh                          # CPU -> out/table_final.md, out/rows.json (E=out for your evals; default ../res)
python3 pertask.py                         # per-task check for 1.7B (CPU, from ../res)
```
Gate refit on ft17: train/test AUROC 1.0000 / 1.0000. A re-derivation (j23836) gives a numerically identical probe.

Expected (3 RL seeds; C28). BBH-27 test, 6,511 items, greedy, max_new 600; paired item-bootstrap CIs (10,000 resamples):

| row | BBH-27 mean [95% CI] | 12 tool tasks [95% CI] | 15 other | tok/q | per seed | 0.6B |
|---|---|---|---|---|---|---|
| 0 plain 1.7B-Base | 51.74 [50.59, 52.85] | 55.37 [53.73, 56.97] | 48.84 | 206 | – | 41.19 |
| 1 ft17, tools off | 42.26 [41.16, 43.37] | 41.90 [40.40, 43.43] | 42.54 | 167 | – | 32.59 |
| 2 + RL (F17) | 51.58 [50.60, 52.54] | 60.88 [59.66, 62.13] | 44.14 | 152 | 51.21 / 51.84 / 51.68 | 42.27 |
| 3 + 8 tools + gate | 66.33 [65.39, 67.28] | 95.73 [95.03, 96.43] | 42.82 | 161 | – | 54.99 |
| 4 + tools + gate + RL (I17) | 68.29 [67.48, 69.12] | 99.29 [99.02, 99.53] | 43.49 | 158 | 68.29 / 68.25 / 68.34 | 60.24 |
| 2' 1.7B-Base + RL (P17) | 66.95 [66.06, 67.84] | 86.17 [85.26, 87.07] | 51.58 | 195 | 67.20 / 67.33 / 66.33 | 53.63 |
| ctl F17 + tools + gate at eval | 67.08 [66.26, 67.92] | 95.82 [95.22, 96.40] | 44.09 | 152 | 66.72 / 67.41 / 67.11 | 57.62 |

Paired effects (mean27):

| effect | value |
|---|---|
| 2 − 1 | +9.32 [+8.32, +10.32] |
| 3 − 1 | +24.08 [+23.29, +24.86] |
| 4 − 2 | +16.71 [+15.96, +17.48] |
| 4 − 3 | +1.96 [+1.25, +2.65] |
| 4 − ctl | +1.21 [+0.65, +1.79] |
| 1 − 0 | −9.49 [−10.89, −8.04] |
| 2' − 0 | +15.21 [+14.13, +16.32] |
| 4 − 2' | +1.34 [+0.41, +2.28] (tool12 +13.12 [+12.21, +14.03], other15 −8.09 [−9.59, −6.57]) |

Per-task check (`pertask.py`; original analysis job j23917): 2 tasks have 4 − 1 < −3, both non-tool.
* geometric_shapes −6.1: the tools never act there, so this is RL weight drift.
* formal_fallacies −3.7: the CI includes 0.

### What ships for §18

* `uniform/rl28/gsm8k/`:
  * `eng/run_arm_17.sh`, `eng/pack_17.sh`;
  * `fe/` (`fe.sh`, `jobE4.sh`, `run_hook.sh`, `lists/`) and `fe_gp/` (`ana_v17.py`, `ana_gp17.py`);
  * the overlays `trees/arm17` and `trees/stk_r2q_17`, which hold the 1.7B refits;
  * `items/` (reduced rows);
  * `evchain.sh`, `q32chain.sh`, `mk32.py`, `mkrows.py`, `mkresults.py`, `ana17.py`, `patch17.py`;
  * `RESULTS.md`, `NOTES.md`, `rows.json`.
* `uniform/rl28/ifeval/`: `code/`, `envs/`, `pack_chain.sh`, `rl_chain.sh`, `analyze.py`, `res/`, `RESULTS*.md`,
  `NOTES.md`, `rows.json`.
* `uniform/rl28/bbh/`: `setup.sh`, `setup2.sh`, `w/`, `cg/job_refit.sh`, `res/`, `table_final.md`, `rows.json`,
  `RESULTS.md`, `NOTES.md`.

**Not reproducible from the repository alone:**
* the 1.7B RL checkpoints (P17, CW17, F17, I17, IFEval P/I);
* `ft17` (weights; recipe above);
* the per-item outputs with generated text (see §14).

## 19. Settings table

Host, prompt, decoding, training, seeds, eval set and tokens per question for every blog result, read from the run dirs,
configs, launch scripts, logs and result JSONs. Paths are relative to the repository root; "(not in the release)" marks a
source that is not part of this repository. R§n = section n of this file; C n = the claim id used in the sections above
(labels only, the numbers are given inline). "n/r" = not recoverable from local files, with the reason in brackets.
tok/q = mean generated tokens per question; "rerun" = later run of the same harness on another GPU, "proxy" = a
different run of the same configuration. "8-shot GSM prompt" = the 8 GSM8K-train shots of `data/gsm8k/shots_v1.json` (written by `data/prepare_gsm.py`),
annotations stripped, `Question: q\nAnswer: body\n#### fin\n\n` per shot, then `Question: q\nAnswer:`.

| Result (blog §) | Host model | Prompt / format | Decoding | Training | Seeds (train; eval) | Eval set (size) | tok/q | Source |
|---|---|---|---|---|---|---|---|---|
| §1.2 calculator, GSM8K / GSM-Hard | Qwen3-0.6B-Base (frozen); also 1.7B-Base, 4B-Base | 8-shot GSM prompt (`shots: 8` in the result JSONs) | greedy argmax, max_new 320, stop at `<\|endoftext\|>` or a blank line; score = first `#### n` before the blank line; length-sorted left-padded batches, bs 32 (`scripts/run_gsm.py`) | none (veto solved per host) | n/a; one run per arm | GSM8K test 1319, GSM-Hard 1319 | 0.6B, rerun (B200, same harness): test 95.9 plain / 96.1 calc, Hard 125.6 / 126.1 ((not in the release); rerun acc 51.55 / 53.22 / 26.38 / 34.19 vs shipped 52.24 / 53.37 / 27.29 / 35.03). 1.7B / 4B n/r [shipped rows have no token counts or text] | R§1, R§8.1; `results/gsm_full_implant/*.json`, `results/gsm_bigger/*.json` |
| §1.2 operand-size sweep | Qwen3-0.6B-Base (frozen), calc_s4 | 2-shot `Q: What is a op b?\nA: a op b =` | greedy, max_new 42, stop at newline, bs 120 | none | operands `random.Random(0)`; one run | 100 items per cell, 1x1..5x5 digits | n/r [`evidence/item1/*.json` keep only base / impl / n per cell] | R§1; `evidence/code/item1_calc.py --n 100` |
| §1.2 circuit exactness (no model) | none (circuit only) | tool-call symbol framing | n/a | none | n/a | 160,000 pairs (0..199, 4 ops) | n/a | R§1; `calc_sweep_0_199.log` |
| §1.2 frozen multiplication test | Qwen2.5-0.5B (frozen) | 6-shot `Q: ...\nA:` with ` <<a*b=` call shots (`PREFIX`) | greedy argmax, max_new 48, stop at newline, CPU fp32, `torch.manual_seed(0)` | none | n/a; one run | 4 sets x 100 (bare_mult, word; machine / no machine) | bare 32.4, word 27.6 (both arms; `out` texts re-tokenized with the Qwen3 BPE, same as Qwen2.5 for plain text) | R§1; `finetuned_phase/calc_icl/calc_icl.py`, `finetuned_phase/results/calc_icl_100.json` |
| §1.3 call FT, GSM (3 FTs: no calls / model fills / implant fills) | Qwen2.5-0.5B (base) | train: `Question: q\nAnswer:` + ` body\n#### fin`; eval zero-shot `Question: q\nAnswer:` | greedy, max_new 512, bs 256, bf16 (`gen_eval.py`); calc arm with `--calc net` | full FT, `ft.py` defaults: lr 1e-5, 3 epochs, tok_budget 4096, fp32 master; GSM8K train + `data/aug.jsonl` (`finetuned_phase/eyes_and_finetune/make_aug.py`); arms `--fmt plain` / `--fmt calc --mask_output 0` / `--fmt calc` | 5 FT seeds (per-seed files offline, `verified_here: false`); eval once per seed | dev 500 (shuffled train rows), GSM8K test 1319, GSM-Hard 1319, bare arithmetic 400 | n/r [per-seed outputs offline] | R§2; `finetuned_phase/eyes_and_finetune/seedjobs.sh` (`gsm()`), `ft.py`, `gen_eval.py`, `finetuned_phase/results/reported_numbers.json` |
| §1.3 "about 100 examples" (BBH-style) | Qwen3-0.6B-Base, full FT | eval: official BBH 3-shot CoT | greedy, max_new 600, bs 32 (`bbh_generic/teamA/run_eval.py`) | Team A `bbh_generic/teamA/ft.py`: lr 1e-5, tok_budget 16000, 10 epochs, n = 32 / 100 / 300 rows; Team B `ft_B.py`: lr 1e-5, tok_budget 24000, p_zero 0.2, 15 epochs | FT seed 0; n=100 also seed 1; eval once | Team A fresh test 600, held-out (oos2) 180, real BBH 1500; Team B test 300, oos 150, BBH 750 | n/r [generations not shipped; `scores_ft100.json` has scores only] | R§6.3; `bbh_generic/ft100/README.md`, `scripts/make_jobs.py`, `scores_ft100.json` |
| §1.4 eyes implant (MNIST) vs no eyes | Qwen2.5-0.5B (base), FT | `Q: q\nA: Let's think step by step.` + ` digits\nmath So the answer is v.` (`eyes.py`) | greedy, max_new 512, bs 16 (`bbh_gen.py`) | FT lr 1e-5, 200 epochs (n=8) / 50 (n=32), tok_budget 2048, `--init_special calc+eye`; CNN 421,642 params | seeds 0-3 (all 16 result files present); eval once | eyes test 500 (2img 162 / 3img 248 / 4img 90) | n=8: eyes 83.8, none 85.1; n=32: eyes 82.1, none 82.7 (gen_tokens / 500, mean of 4 seeds) | R§3; `evidence/code/eyes_sweep3.sh`, `evidence/eyes_sweep/results/*_test.json` |
| §1.4 eyes vs plain FT | Qwen2.5-0.5B (base), FT | as above | greedy, max_new 512, bs 32 | FT lr 1e-5, 3 epochs, 4,000 or 16,000 rows; eyes arm masks box tokens from the loss | 5 seeds (seed 0 offline; seeds 1-4 in `eyes.txt`); eval once | eyes dev 300, test 500 | n/r [`eyes.txt` has MACRO only] | R§3; `finetuned_phase/eyes_and_finetune/seedjobs.sh` (`eyes()`), `finetuned_phase/results/eyes.txt` |
| §1.4 eyes vs Python tool | Qwen2.5-0.5B (base), FT | as eyes sweep; `text` arm = CNN digit pasted as text + calc implant | greedy, max_new 512, bs 16 | as eyes sweep (n = 8 / 32); CNN rebuilt (seeded) | seeds 0-3; eval once | eyes test 500 | eyes 82.3 / 82.7, plain 82.8 / 82.5, text 62.7 / 62.2, textnc 61.8 / 61.8 (n=8 / n=32) | B200 run (not in the release) |
| §2.1 mistake detector / rethink | Qwen3-0.6B-Base (frozen) | 8-shot GSM prompt | final arms: greedy, max_new 700, bs 32; gated arm `--intervene hybrid --cap 300`; calc arms calc_s3. Training states: vLLM greedy + K 3 samples at T 0.8 (seed 7), stop blank line (`gen_train.py`) | host frozen; detector fit on ~52k labelled states (`policy3.py`, `fit_detector.py`, `torch.manual_seed(0)`) | n/a; eval once (`_s0/_s1` file suffix = batch chunk 0-20 / 21+, not a seed) | GSM8K test 1319, GSM-Hard 1319 | test plain none / alwaysA / hybrid 105.2 / 250.1 / 212.8; calc 104.6 / 249.9 / 214.9; Hard plain 144.2 / 371.6 / 316.0; calc 140.8 / 381.7 / 329.3 | R§9; `scripts/rethink_v2/run_gsm_v2.py`, `jobs_hybrid.txt`, `results/rethink_v2/final_eval/*.json` (`n_tok`), `summary.json` |
| §2.2 topic blocker | Qwen/Qwen3-1.7B (chat, thinking off), frozen; reduced pass on Qwen3-0.6B | chat template, user message; `sys` / `sys_strong` baselines add a system prompt | greedy, max_new 120, bs 32 (`s5_eval.py`); MMLU letter max_new 8 | probes / monitors fit on a Qwen2.5-7B-written dev set (vLLM T 1.0, top-p 0.95; run b unseeded) | n/a; eval once; test items sampled with `random.Random(2..6)` (`build_test.py`) | 640 hand prompts + public items. Math fence: pos 447, other neg 260, near neg 70, hard 70, drift 30, gen neg 60, retention 567, MMLU 400. ML fence: pos 122, other neg 560, rest as math | n/r [only rates shipped; replies not in the repo] | R§10; `topic_fence/README.md`, `topic_fence/results/test_q17_{math,ml}_summary.json` |
| §2.3 latent write | Qwen3-0.6B-Base (also 1.7B-Base), frozen | `What is (a*b)+c?`, 4-shot, answer at once (no CoT) | greedy, max 6 new tokens, batch 50, bf16 host, writers fp32 | writer banks refit per run (closed-form ridge, lambda 1, fp64); host frozen | shots `random.Random(1234)`, item pool `Random(99)`, corrupted / fit values `Random(11)`, word problem seed 21; eval once | 300 items per cell x 11 layers x 4 arms | n/r [no text kept; at most 6 by construction] | R§11; `latent_write/README.md`, `lwv1.py`, `run_all.sh` |
| §3 RL + calculator, 300-update headline (arm 1 plain vs arm 2 calc) | Qwen3-0.6B-Base | 8-shot GSM prompt (train and eval) | train T 1.0; dev and test greedy, max_new 320, stop blank line / eos (`final_eval.py`, `engine.py`) | Dr.GRPO, kl 0, clip 0.2, 300 u, 64 prompts x G 8, lr 1e-6, warmup 5, tok_budget 16384, B_max 512, S_max 544 | s0/s1 (H100), s2 (A100); eval once | GSM8K test 1319, GSM-Hard 1319; dev 800 (400 GSM8K + 400 scaled) | test / Hard: arm 1 s0 129.0 / 159.9, s1 143.2 / 180.0; arm 2 on s0 130.3 / 166.3, s1 134.3 / 170.8; base + calc 95.6 / 124.7. s2 n/r [only point accuracies in `rl/wiring/NOTEBOOK.md` l.168] | R§12; `rl/results/gsm_grpo/H100/long/*/config.json`, `rl/results/gsm_grpo/H100/final/H_a{1,2}_s*_u300.json` (`len`) |
| §3 writing-habit stats (unit words, copies answer) | Qwen3-0.6B-Base, u64 checkpoints | 8-shot GSM prompt | greedy, max_new 320, A100, B_max 64 (`dump_traces.py`) | 64 u, same recipe | one seed per arm: a2 s0, a1 s1 (A100); traces once | dev 800 (400 GSM8K + 400 scaled) | base 111.2 / 110.8 (off / on); arm 1 125.2 / 125.7; arm 2 134.4 / 135.0 | R§12; `rl/interp/run_all.sh`, `rl/interp/stats.json` (`tok`) |
| §3 broken calculator | Qwen3-0.6B-Base | 8-shot GSM prompt | train T 1.0, max_new 320; dev greedy | arm R (`rand: digits`), 64 u, 64 x G 8, lr 1e-6, Dr.GRPO, kl 0, tok_budget 16384, B_max 512 | s0/s1 on A100 and on H100; eval once | dev 800 | dev at u64: A100 139.0 / 125.6, H100 130.0 / 123.4 (s0 / s1); traces (H100, B_max 256) aR0 130.1, aR1 123.7 | R§12; `rl/results/gsm_grpo/{A100,H100}/abl/aR_*/{config.json,steps.jsonl}` (`dev_len`), `rl/interp/run_aR_h100.sh`, `stats_aR.json` |
| §3 learned wiring (CW-full, CWinOx, CWinOlr, CWinR, W2, W3d, W3e, W3r) | Qwen3-0.6B-Base | 8-shot GSM prompt (`train_wiring.py` `prefix_ids`) | train T 1.0; dev greedy, max_new 320 | 300 u, 64 x G 8, lr 1e-6, B_max 128, tok_budget 4096, link_lr 1e-2 (CWinOx: cw_lr 1e-3) | CW-full s0; CWinOx 3 seeds; CWinOlr 2; W3d 3 (s2 OOM at u251); others 1; eval once | dev 800 (dev_every 25) | dev at u300: CW 130.5, CWinOx 149.2 / 147.7 / 153.1, CWinOlr 149.4 / 153.7, CWinR 152.9, W2 143.2, W3d 140.3 / 144.2, W3e 144.9, W3r 157.5 | R§12; `rl/results/wiring/figdata/*.jsonl`, `rl/results/wiring/{H100,A40}/*/steps.jsonl` (`dev_len`) |
| §4.2 IFEval climb (14 rule implants) | Qwen3-0.6B (chat, thinking off) | chat template | greedy HF loop, max_new 2048, bs 24, 4 shards, A100 | none (implants built on own dev / fresh data, gen seeds 0/1/2) | n/a; scorer seeded per key | IFEval 541 (ALL), ODD 299, EVEN 242 | plain 245.7, implant 256.8 (26 / 28 rows hit the cap) | R§5.1; `ifeval/transfer/README.md`, `ifeval/transfer/results/reduced/q06_ifeval_{plain,imp}.json` (`ntok`) |
| §4.2 thinking-mode baseline | Qwen3-0.6B (thinking on) | chat, answer after `</think>` | greedy, bs 16, max_new 2048 (the cap; 50/541 hit it, 26 never close `</think>`), A40 | none | n/a; eval once | IFEval 541 / ODD 299 | 659.0 | R§5.3; `ifeval/baselines/run_baselines.sh`, `README.md`, `results/ifeval_think_plain.ans.json` |
| §4.2 compiled into weights | Qwen3-0.6B | as §4.2 | as §4.2 | none | n/a | 541 | 256.8 (inferred: texts byte-identical to the circuit run; no own record) | R§5.2 |
| §4.3 BBH climb (8 tools + call FT) | `ft_int_v2` = Qwen3-0.6B-Base + call FT | official 3-shot CoT; `<tool_call>` format | greedy, max_new 600 | full FT 432 rows, lr 2e-5, 6 epochs, tok_budget 8000, 156 steps | FT seed 0; BBH run once per arm | BBH-27 (6,511 items; 12 tool / 15 no-tool tasks) | n/r [records keep chars only: base 697, FT no tools 391, FT + tools 690 chars/q] | R§6.1; `bbh_generic/transfer/results/q06_*.json.gz` (`n_chars`) |
| §4.3 call gate | `ft_int_v2` | as above | greedy, max_new 600 | probe on states (`CG_LAM=0.1`) | FT seed 0; run once | BBH-27 6,511; ARC-Easy 400 val (3-shot CoT) | proxy 133.6 (R§17.8 row 3, same host + 8 tools + refit gate, B 128) | R§6.2; `bbh_generic/callgate/README.md`; `uniform/rl24/bbh/rows.json` |
| §4.3 add-one-tool curve | `ft_int_v2` | as above | greedy, max_new 600 | as §4.3 | single FT seed | BBH-27 6,511 per point | n/r [chars only: k1..k5 460 / 574 / 657 / 669 / 704 chars/q] | R§7.2; `bbh_generic/transfer/results/curve_k*.json.gz` |
| §4.4 GSM8K single-call stack | `a2_u300_s2` (0.6B-Base after RL+calc, arm 2 seed 2, 300 u) | 8-shot GSM prompt (`gsm_push/harness/tw_dump_lean.py`); 320 tokens per answer | S0-S2F greedy, gen_max 960, S_max 1280; S3-serial T 1.0, seed 7, STK_MAXN 5416, S_max 5696, BMAX 64 (Hard N4: 32); Hard serial = plain majority, no probe / sanity filter | none on top of the host | host seed 2; eval once | GSM8K test 1319, GSM-Hard 1319 | test: S0 136.7, S1 136.6, S2F 153.0, N4 633.7, N8 1274.2; Hard: 173.7, 173.4, 175.6, 804.9, 1587.8 | R§15; `gsm_push/final/stack{3,4}_hook.py`, `gsm_push/results/final/{fwd,fgtw,ser4,ser4b,ser8,ser_a100/ser8}/*.log` |
| §5.1 GSM8K row 1 (Base) | Qwen3-0.6B-Base | zero-shot chat format | T 1, seed 7, 320 tok, 1 sample | none | n/a; eval seed 7 (+11/13 spread, C33) | test 1319, Hard 1319 | 103 / 127 | R§17.2; C1; j23321 |
| §5.1 GSM8K row 2 (P64 RL) | Qwen3-0.6B-Base | zero-shot chat | T 1, seed 7, 1 sample | Dr.GRPO 300 u, 64 x 8, lr 1e-6, T 1, S_max 1280 | s0/s1/s2; eval seed 7 | test, Hard | 129 / 164 | R§17.2; C2; j23692 |
| §5.1 GSM8K row 3 (Base + rung 2) | Qwen3-0.6B-Base | zero-shot chat | T 1, seed 7; adaptive N 8 -> 16 attempts, L18 probe voter, SW_MAXN 7936 | none | n/a; eval seed 7 (+11/13) | test, Hard | 2650 / 3830 | R§17.2; C3; j23626 |
| §5.1 GSM8K row 4 (CW RL + rung 2) | Qwen3-0.6B-Base | zero-shot chat | T 1, seed 7; rung 2; FE_SMAX 8192 | as row 2 + implants, WPR, N=4 during RL, S_max 2944 | s0/s1/s2; eval seed 7 | test, Hard | 1612 / ~2650 | R§17.2; C4; j23509 |
| §5.1 GSM8K matched-compute majority vote | Base / P64 hosts | zero-shot chat | T 1, seed 7; N separate samples + plain majority (N 12/16 vs row 4; 26/30 vs row 3) | none / P64 | P64 s0-s2; eval seed 7 (+11/13) | test, Hard | 1213-3777 (per row) | R§17.4; C9, C9b; j23890, j23891, j23898 |
| §5.1 GSM-Plus (fresh set) | Qwen3-0.6B-Base, rows 1-4 | GSM8K protocol per row | T 1, seed 7, once | as rows 1-4 | RL s0-s2; eval seed 7 (+11/13) | GSM-Plus testmini, 7 types x 300 = 2100 | 119 / 155 / 3282 / 2446 | R§17.6; C15; j23892, j23880 |
| §5.1 BBH rows 0-4, ctl | plain Base (row 0, 2'); `ft_int_v2` (rows 1-4, ctl) | 3-shot CoT, `\n\nQ` stop | greedy, max_new 600 | RL: Dr.GRPO 300 u, 64 x 8, lr 1e-6, T 1 | RL s0/s1/s2; eval once (greedy) | BBH-27 6,511 | 222 / 135 / 144 / 134 / 140 (rows 0-4) | R§17.8; C17-C21; j23805 |
| §5.1 IFEval rows 1-4, 2b | Qwen3-0.6B-Base | chat template, thinking off | greedy, max_new 2048, seeded checkers | RL 300 u, 64 x 8, lr 1e-6; RL data gen seed 200 (18,219 rows), dev seed 201 (415); reward prompt strict | s0/s1/s2; eval once | IFEval 541 | 1889 / 573 / 1388 / 1364 / 546 | R§17.7; C22, C23; j23721, j23782 |
| §5.2 RL with all implants (LIB, LIBscr) | Qwen3-0.6B-Base; calc_s4 + 8 BBH tools + IFEval circuit, frozen; per-tool in-wire + out-adapters at L14 / final | 8-shot GSM prompt (`rl/library/common.py`) | train T 1.0; dev / test greedy, max_new 320 | Dr.GRPO, lr 1e-6, G 8, 64 prompts, kl 0, 300 u, B_max 128, tok_budget 4096, in-wire floor 0.05, link_lr 1e-3, link_eps 1e-8 | LIB s0 (H100) and s1 (A40: 32 -> 24 prompts/update by the deadline guard, tok_budget 2048, dev every 50); LIBscr s0 only; eval once | GSM8K test 1319, GSM-Hard 1319; dev 800 | test n/r [test records have no length]; dev at u300: LIB s0 159.5, LIBscr s0 140.8, LIB s1 319.7 (at the 320 cap from u200). Rerun (B200, s1): LIB 128, LIBscr 157 (test) | R§12; `rl/library/launch/{h100_LIB_s0,h100_LIBscr_s0,a40_zz3_LIB_s1}.sh`, `rl/library/NOTEBOOK.md`, `rl/results/library/figdata/LIB*_steps.jsonl`; rerun (not in the release) |
| §5.2 mixed-task run (m1-m4) | Qwen3-0.6B (chat, thinking off) + format SFT warm start (`sft_warm.py`: 80 steps, bs 16, lr 1e-5, AdamW betas 0.9 / 0.95, 5-step warmup + cosine, 200 GSM rows + `train_union_v2.jsonl`, loss on the reply only; m4 = noisy-routing SFT) | chat template, thinking off, system prompt "... The answer is X." (`rl/library/mixed/mixed_task.py`); tagged calls (`<calc>a / b</calc>` in GSM rows, BBH tool-call tags from `train_union_v2.jsonl`) | train T 1.0; dev / test greedy, max_new 512, S_max 896 | 150 u, 64 prompts x G 8, lr 1e-6, Dr.GRPO, kl 0, B_max 128, tok_budget 4096; same library and wiring as LIB | one seed each (m1 s1, m4 s1/s2 launched, no results in the repo); later reruns: m1m (cw_mask 1) s0-s2, m3 s1/s2, aNL s0-s2 (not in the release) | test 400 (stratified from 999); dev 400 | m1 135.3, m4 165.9, m2 131.2; m3 n/r [coordinator run, no file] | R§12; `rl/library/launch/h100_m1_mixLIB.sh`, `rl/library/mixed/sft_warm.py`, `rl/library/NOTEBOOK.md`, `rl/results/library/figdata/mix_m*_test.json` (`test_len`) |
| §6 calculator on Qwen3-1.7B / 4B | Qwen3-1.7B-Base, Qwen3-4B-Base | 8-shot GSM prompt | greedy, max_new 320, bs 32 (as §1.2) | none (veto re-solved) | n/a; one run | GSM8K 1319, GSM-Hard 1319 | n/r [shipped rows have no token counts or text] | R§1, R§8.1; `results/gsm_bigger/` |
| §6 IFEval rules on Qwen3-1.7B / 4B | Qwen/Qwen3-1.7B, Qwen/Qwen3-4B (chat, thinking off) | as §4.2 | greedy, max_new 2048, bs 24, A100 | none (force_fix re-solved per host) | n/a; scorer seeded | IFEval 541 | 1.7B 364.4 -> 384.5, 4B 361.2 -> 361.4 (plain -> implant) | R§8.2; `ifeval/transfer/run_all.sh`, `README.md`, `results/reduced/q{17,4}_ifeval_{plain,imp}.json` |
| §6 BBH tools on 1.7B / 4B | 1.7B / 4B v2 call-FT hosts | 3-shot CoT | greedy, max_new 600 | call FT as §4.3 (74 s / 243 s) | FT seed 0; run once | BBH-27 6,511 | n/r [chars only: 1.7B base 656 / FT + tools 529, 4B 627 / 583 chars/q]; proxy 1.7B (R§18.3, `ft17`): 206 base, 161 + tools + gate | R§7.2, R§8.3; `bbh_generic/transfer/results/q{17,4}_*.json.gz` |
| §6 SmolLM2 / granite / Falcon3 (calculator + BBH arithmetic) | SmolLM2-1.7B, granite-3.3-2b-base, Falcon3-3B-Base (LFM2 cannot attach) | GSM: 8-shot GSM prompt; BBH: official 3-shot CoT | greedy; GSM max_new 400, bs 24, stop `\n\nQuestion:` / eos / first `#### n`; BBH max_new 400, bs 32, stop `\n\nQ` | none (veto per host) | n/a; one run per arm | GSM8K 1319, GSM-Hard 1319; BBH 7 tasks (3 tool + 4 no-tool) = 1,687 | GSM test / Hard, plain -> calc: SmolLM2 119.9 -> 118.7 / 148.1 -> 147.5; granite 121.2 -> 121.3 / 156.1 -> 157.1; Falcon3 108.8 / 141.8 (identical). BBH n/r [no lengths kept] | R§8.4; `transfer/xfamily/code/{run_gsm_x.py,drvr.sh,bbhx.sh,run27x.py}`, `transfer/xfamily/results/{gsm,bbh}/*.json` |
| §6 IFEval on SmolLM2-1.7B-Instruct | SmolLM2-1.7B-Instruct | chat template (ChatML) | greedy, max_new 2048, bs 32, A40 | none (constants-only port v1; v2.1 after diagnosis) | n/a; eval once | IFEval ODD 299 | plain 612.9, v1 672.9, v2.1 691.0 | R§8.5; `ifeval/xfamily/run_sm2i.sh`, `ifeval/xfamily/results/sm2i_odd_*.json` |
| §6 1.7B-Base GSM8K rows 1-4 | Qwen3-1.7B-Base | zero-shot chat | T 1, seed 7, once | RL as §5.1 (P17, CW17); FT_BUDGET 16384 / 32768 | s0/s1/s2; eval seed 7 (+11/13) | test 1319, Hard 1319, GSM-Plus 2100 | 106/138, 118/152, 2232/3531, 1853/3090 | R§18.1; C27; j23915, j23918 |
| §6 1.7B matched-compute MV | P17 / 1.7B-Base | zero-shot chat | T 1, seed 7; N 16 / 20 (vs row 4), 21 / 26 (vs row 3) | as rows | P17 s0-s2; eval 7 (+11/13) | test, Hard, GSM-Plus | 2229 / 3592 (row-3 MV) | R§18.1; C27; j23937, j23928-29 |
| §6 1.7B-Base IFEval rows | Qwen3-1.7B-Base | as §5.1 (chat template) | greedy, prompt strict | RL as §5.1; readout chosen on dev seed 201 (415 prompts) | s0/s1/s2; eval once | IFEval 541 | 1467 / 390 / 1440 / 595 | R§18.2; C24-C26; j23828, j23840, j23842 |
| §6 1.7B-Base BBH rows | 1.7B-Base (row 0, 2'); `ft17` = 1.7B-Base + call FT (rows 1-4) | 3-shot CoT | greedy, max_new 600 | call FT recipe as `ft_int_v2` (100 s B200); RL as §5.1 | RL s0/s1/s2; eval once | BBH-27 6,511 | 206 / 167 / 152 / 161 / 158 | R§18.3; C28; j23917 (ana) |

## 20. Text-calculator-tool baseline (claim C35, `checks/tooltext/`)

`checks/tooltext/README.md` describes the arms and the files.

**Question.** Does a plain text calculator tool, trained with the same RL, do about as well as the calculator implant?

All arms start from Qwen3-0.6B-Base with the §12 harness (`rl/gsm_grpo`: 8-shot, max_new 320) and train with RL for
300 updates (64 × 8, T 1, lr 1e-6, Dr.GRPO, clip 0.2, warmup 5, dev 800 every 25 u).

| arm | what |
|---|---|
| 1 | plain RL |
| 2 | RL + calculator implant (one attempt) |
| X | RL + a text `<calc>E</calc>` tool. The host evaluates E exactly with `fractions.Fraction` and forces the result tokens; checked at a host sync every 4 decode steps. Tool tokens count toward the cap and are masked from the loss. |

* **Code.** `checks/tooltext/code/rl/gsm_grpo/` is an overlay on `rl/gsm_grpo/`.
  * New files: `tooltext.py`, `fe_tool.py`, `check_tool.py`.
  * Modified copies: `engine.py`, `train2.py`.
  * `bash checks/tooltext/mk_code.sh` builds the runnable tree `checks/tooltext/code/`: release `rl/gsm_grpo` +
    `micromachines` + data + veto, then the overlay.
  * It needs the inputs below; their SHA-256 are checked against `code_data.sha256`:
    * `python data/prepare_gsm.py`;
    * GSM-Plus testmini from `uniform/final_eval/data/prepare_gsmplus.py` (writes `data/gsmplus/testmini.jsonl`);
    * `runs/veto_qwen3.pt` from `python scripts/solve_veto.py`.
* **Environment.** As §12. `job.sh` sources `rl/gsm_grpo/env_local.sh` and sets `FLEX_SMALL=1` (needed on B200). Run dirs
  go under `WSM_DATA`, default `runs/tooltext`.
* **Hardware and runtime.** One B200 per job. Train ~24 / 33 / 30 min (arms 1 / 2 / X); eval ~2–3 min.
* **Jobs.** `cd checks/tooltext && bash commands.sh` prints every job line:
  * per train seed K = 0–11: `bash job.sh train ARM K aARM_sK && bash job.sh eval ARM aARM_sK NAME`, with
    `--sync_every 4` for X;
  * three Base evals;
  * `bash job.sh check`.

  Run each line on one GPU: `CUDA_VISIBLE_DEVICES=0 bash -c "$(bash commands.sh | grep ' aX_s3 ')"`.
* **Checks** (`job.sh check` → `res/check_tool.json`; Base, greedy, 256 dev items):
  * parity with the stock engine on the plain prompt: 99.2% of rows;
  * tool ERR 1.7%;
  * teacher-forced HF agreement 99.6% (rows with calls) / 99.9% (rows without);
  * sync 4 = sync 16 outputs, 1.75× faster;
  * 97% of Base rows call the tool.
* **Eval.** GSM8K test (1319), GSM-Hard (1319), GSM-Plus testmini (2100); T 1, eval seeds 7/11/13, one attempt,
  B_max 512.
* **Analysis** (CPU, a few minutes; reads the shipped text-free `res/`):

  ```
  cd checks/tooltext && python3 ana_tt.py --write   # -> rows.json, prints res/ana_tt_12seeds.txt
  ```

  To rebuild `res/` from new job outputs: `python3 ../calcimp/mk_res.py --out out --res res --evals '^(ev_|e37a_)' --runs '^a[12X]_s'`.
* **Expected.** 12 train seeds × 3 eval seeds; test / GSM-Hard / GSM-Plus:

  | row | test | Hard | GSM-Plus | tok/q model (total) | calls/q | ms/q batched | ms/q B=1 |
  |---|---|---|---|---|---|---|---|
  | 0 Base | 28.20 | 11.25 | 19.54 | 101 | 0 | 4.6 | 282 |
  | 02 Base + implant | 29.06 | 14.89 | 19.83 | 101 | 1.89 | 7.3 | 342 |
  | 0X Base + tool | 26.18 | 15.77 | 17.32 | 124 (131) | 2.83 | 5.9 | 379 |
  | 1 | 68.37 | 41.54 | 54.32 | 133 | 0 | 4.8 | 361 |
  | 2 | 69.74 | 54.08 | 55.02 | 128 | 3.00 | 7.4 | 422 |
  | X | 69.29 | 55.97 | 54.27 | 141 (149) | 3.09 | 5.9 | 432 |

  Paired differences (train-seed × item bootstrap 95% CI):
  * X − 2: −0.45 [−1.50, +0.54] / **+1.90 [+0.65, +3.19]** / −0.75 [−1.60, +0.13];
  * X − 1: +0.92 [−0.20, +2.03] / +14.44 [+12.82, +16.22] / −0.05 [−1.00, +1.00];
  * 2 − 1: +1.37 [+0.40, +2.35] / +12.54 [+11.01, +14.16] / +0.70 [−0.04, +1.51].
* **Model-token cap re-eval (a later check).** The arm-X u300 checkpoints of train seeds 0–4 were re-evaluated with the cap on
  model tokens only (`--cap_extra 128`, C36 code; last block of `checks/calcimp/tt/commands.sh`).
  * X·5 69.09 / 55.86 / 54.12; Xm·5 69.16 / 57.26 / 54.43.
  * Xm·5 − X·5: +0.07 / **+1.39 [+0.35, +2.47]** / +0.31.
  * Xm·5 − 2·5: −0.99 / +3.17 [+1.76, +4.63] / −0.59.
* **Not reproducible from the release alone.** Checkpoints and generated text are not shipped. `why/why.py` (the
  arithmetic-slip split) needs the regenerated eval outputs.

## 21. Equal-footing text tool vs exact calculator implant at 28 seeds (claim C36, `checks/calcimp/`)

`checks/calcimp/README.md` describes the arms and the files.

**Setup.** As §20, except for the following:
* both the tool and the implant use the same `<calc>EXPR</calc>` syntax;
* the cap counts model tokens only: 320, plus up to 128 tool- or implant-written tokens (`--sync_every 4 --cap_extra 128`);
* each arm runs at its own dev-selected config.

| row | what | config |
|---|---|---|
| T / T1 / T2 | text tool | host lr 4e-6 (u100 sweep pick) / 1e-6 (the §20 lr) / 2e-6 (the 300-u dev rule (b) pick; **headline tool row**) |
| I | implant: a frozen exact int64 core (`core.py`) fires on `</calc>`; a trainable writer and adapters into layers 7/14/21/final carry the value; an in-wire gate at layer 14 admits it per call (floor 0.05) | `--lr 2e-6 --link_lr 1e-2 --iw_eps 0.05 --ci_adapters 1 --ci_std 1e-3` |
| Ie0 | I's checkpoints evaluated with the gate floor off | `--iw_eps_eval 0` |
| I0 | frozen hand writer, always admits | `--lr 2e-6` |
| Itok | only the writer trains | `--lr 2e-6 --link_lr 1e-2` |
| I1 / I01 / Itok1 | the three implant rows at host lr 1e-6 | 8 seeds |

* **Code.** `checks/calcimp/tt/code/rl/gsm_grpo/` is an overlay on `rl/gsm_grpo/`.
  * New files: `tooltext.py`, `core.py`, `armi.py`, `fe_tool.py`, `check_tool.py`, `check_ci.py`.
  * Modified copies: `engine.py`, `train2.py`, `fastqwen.py`.
  * Build with `bash checks/calcimp/tt/mk_code.sh`, which has the same inputs as §20.
  * With `--cap_extra 0` this tree reproduces the §20 arm-X eval byte-identically.
  * The core runs on CPU (`CORE_DEV=cpu`, exact per-run memo); 17–40 ms per call (`res/prof_core_cpu.txt`).
* **Hardware and runtime.**
  * **B200:**
    * train seeds 0–11 of every row;
    * the 8-seed rows (T, I1, I01, Itok1);
    * the u100 sweep at seed 0;
    * the fIr / fIa diagnostics.

    ~40 min per train + eval line, 62.5 GB peak. Packing two lines per GPU gives ×0.92 throughput.
  * **A100-SXM4 40 GB:**
    * train seeds 12–27 of T2 / I (+ Ie0) / I0 / Itok;
    * seeds 12–19 of T1;
    * the seed-1 I-wiring and seed-2 swT sweep replicates.

    These ran with `--tok_budget 4096 --B_max 192` (train) and `--B_max 192` (eval). They took ~1h20 (T1, T2) to ~2h05
    (I, I0) per line. `tt/res/configs.json` has `"gpu"` per run.
* **Jobs.** `cd checks/calcimp/tt && bash commands.sh` prints every line: checks, Base evals, the u100 sweep, the
  finals (B200 block and A100 block), the arm-I collapse diagnostics, and the §20 cap re-evals. Example final line:

  ```
  bash job.sh train I 3 fI_s3 --lr 2e-6 --link_lr 1e-2 --iw_eps 0.05 --ci_adapters 1 --ci_std 1e-3 --sync_every 4 --cap_extra 128 \
    && bash job.sh eval I fI_s3 ev_fI_s3 --sync_every 4 --cap_extra 128 && bash job.sh eval I fI_s3 ev_fIe0_s3 --sync_every 4 --cap_extra 128 --iw_eps_eval 0
  ```
* **Checks.** `bash job.sh check_ci` writes `tt/res/check_ci.json`. It checks u0 bit-identity of I0 / Itok vs X (sampled
  and greedy) and the core on every live call.
  * Core exactness, CPU: `python3 verify_core.py` (needs the §20 arm-X traces) writes `res/verify.txt`,
    `res/verify_core.json` and `res/mismatches.txt`.
  * The core matches the tool on **99.98% of 580,834 calls (119 mismatched calls)**.
* **Analysis** (CPU; reads the shipped text-free `tt/res/`):

  ```
  cd checks/calcimp
  python3 ana38.py --all --write          # -> rows38_all.json       (~7 min)
  python3 ana38.py --all --sens --write   # -> rows38_all_sens.json  (~10 min; adds Ix / Ie0x)
  python3 ana38.py --write                # -> rows38.json, train seeds 0-7 (~5.5 min)
  python3 sel300.py 8                     # dev at u300 per arm and lr (rule (b))  = res/sel300_8seeds.txt
  python3 ana_sweep.py                    # u100 sweep                             = res/ana_sweep_a100.txt
  (python3 links38.py 'fI_s%d' 8; python3 links38.py 'fItok_s%d' 8) > res/links38.txt   # wiring admit / link norms over training
  ```

  To rebuild `tt/res/` from new job outputs: `python3 mk_res.py --out tt/out --res tt/res`.
* **Expected (rows38_all.json).** Test / GSM-Hard / GSM-Plus; 28 train seeds × 3 eval seeds unless noted:

  | row | seeds | test | Hard | GSM-Plus | calls/q | admit |
  |---|---|---|---|---|---|---|
  | 0X / 0I0 (Base) | – | 27.19 / 26.59 | 15.14 / 16.12 | 17.89 / 17.68 | 2.85 / 2.84 | |
  | T (4e-6) | 8 | 65.93 | 56.28 | 51.21 | 2.96 | |
  | T1 (1e-6) | 20 | 69.37 | 57.45 | 54.50 | 3.14 | |
  | **T2** (2e-6) | 28 | 70.04 | 58.90 | 55.01 | 3.20 | |
  | I | 28 | 69.70 | 55.89 | 54.77 | 2.92 | 0.883 |
  | Ie0 | 28 | 69.95 | 57.17 | 55.05 | 2.92 | 0.929 |
  | I0 | 28 | 69.95 | 58.82 | 54.80 | 3.18 | 1.00 |
  | Itok | 28 | 70.04 | 58.96 | 55.18 | 3.16 | 1.00 |
  | I1 / I01 / Itok1 | 8 | 68.90 / 69.23 / 68.85 | 55.49 / 56.95 / 57.15 | 53.98 / 54.77 / 54.45 | | |
  | Ix / Ie0x (no seeds 14, 17; `_sens`) | 26 | 69.80 / 70.07 | 57.04 / 58.41 | 54.83 / 55.14 | | |

  Headline paired differences (train-seed × item bootstrap 95% CI; test / Hard / GSM-Plus):

  | difference | test | Hard | GSM-Plus |
  |---|---|---|---|
  | I0 − T2 | −0.09 [−0.69, +0.55] | −0.08 [−0.71, +0.56] | −0.20 [−0.74, +0.30] |
  | Itok − T2 | −0.00 [−0.59, +0.65] | +0.07 [−0.54, +0.72] | +0.17 [−0.44, +0.76] |
  | I − T2 | −0.33 [−0.98, +0.34] | **−3.00 [−4.81, −1.60]** | −0.24 [−0.82, +0.31] |
  | Ie0 − T2 | −0.08 [−0.73, +0.61] | −1.73 [−3.66, −0.19] | +0.04 [−0.51, +0.57] |
  | Ix − T2 | −0.24 [−0.90, +0.44] | −1.85 [−2.54, −1.22] | −0.17 [−0.78, +0.44] |
  | Ie0x − T2 | +0.03 [−0.63, +0.70] | −0.48 [−1.12, +0.15] | +0.13 [−0.44, +0.70] |
  | I − I0 | −0.25 [−0.89, +0.43] | −2.93 [−4.80, −1.53] | −0.04 [−0.63, +0.55] |
  | Ix − I0 | −0.15 [−0.81, +0.50] | −1.78 [−2.39, −1.17] | +0.03 [−0.55, +0.64] |
  | Itok − I0 | +0.08 [−0.53, +0.72] | +0.14 [−0.45, +0.75] | +0.37 [−0.26, +0.98] |
  | T2 − T1 | +0.67 [−0.04, +1.38] | +1.45 [+0.75, +2.21] | +0.51 [−0.07, +1.09] |
  | I − T (u100 picks, 8 seeds) | +3.78 [+2.45, +5.04] | −0.39 [−2.65, +1.51] | +3.55 [+2.40, +4.69] |

  Arm I collapsed on train seeds 14 and 17: GSM-Hard 39.63 and 42.33. Both ran on A100. B200 reruns did not collapse:
  `fIr_s14/17` used B200 batching and `fIa_s14/17` used A100 batching on a B200. These diagnostics are not pooled.
* **Dev selection.**
  * 300-u rule (b), 8 seeds, dev accuracy at u300: T 72.52 / 73.23 / 69.66 at lr 1e-6 / 2e-6 / 4e-6; I* 72.02 / 73.16
    at 1e-6 / 2e-6; I0 73.05 / 72.81; Itok 72.19 / 73.14. Rule (b) picks T2 = 2e-6.
  * The u100 rule picks T 4e-6 and I 2e-6.
* **Not reproducible from the release alone.**
  * Checkpoints and generated text are not shipped.
  * `why/why38.py`, `why/why37a.py`, `forms.py`, `tokprobe.py` and `verify_core.py` read generated text, so they run
    only after regenerating the evals. Their outputs are shipped in `res/` and `why/why37a.txt`.
  * A rerun on another GPU type (or with A100 batching) changes the RL batch composition, so per-seed numbers differ;
    compare seed-pooled means.

## 22. Blog §5.1 uniform rows at 6 RL seeds (`checks/seeds35/`)

The §17 uniform rows on Qwen3-0.6B-Base were first run at RL seeds 0-2. This section adds RL seeds 3-5 for GSM8K and
IFEval, so those rows are means over 6 RL runs. BBH stays at 3 seeds (§17). Details are in `checks/seeds35/README.md`,
`RESULTS.md` (GSM8K) and `IFEVAL.md` (IFEval, including the NLTK issue).

* **Rows.**
  * Row 2 = plain RL from Base (GSM8K arm P64; IFEval arm P), evaluated alone.
  * Row 2b = the row-2 host with the untrained stack or implant attached at eval only.
  * Row 4 = RL with the stack or implant attached (GSM8K arm CW = rung 2 + WPR; IFEval arm I = the 14-rule implant).
* **Hardware.**
  * GSM8K: every RL arm and eval ran on **one B200** per job.
  * IFEval: seeds 0-2 ran on one B200. Seeds 3-5 ran on **one A100 40 GB** (A100-SXM4-40GB per `nvidia-smi`) with the
    same commands. Batch settings on the A100: B_max 128 / tok_budget 8192, eval B_max 96. On the B200 they were
    B_max 512 | 256 / tok_budget 32768. These settings change batch scheduling only.
* **Shipped.** `res/` holds the per-item correctness and token counts of the seeds 3-5 eval runs, the in-run dev
  records of all 12 arms and the seeds 3-5 IFEval per-item scores. The seeds 0-2 / u0 per-item rows are the §17 runs.
  The analysis reads them from `uniform/final_eval/items/` (via `uniform/items.py`) and `uniform/rl24/ifeval/res/`
  (via `ifana/ifitems.py`). No generated text, no checkpoints.

### 22.1 Recompute every table (CPU, no model)

```bash
cd checks/seeds35
python3 ana35.py --stamp 2026-11-23T10:40Z > AUTO_final.md                          # ~50 s; byte-identical to the shipped file
cd ifana
SEEDS=0,1,2,3,4,5 ROWSJ=rows_s0-5.json python3 analyze6.py --out table_s0-5.md     # ~1 min; = shipped table_s0-5.md
SEEDS=0,1,2 ROWSJ=rows_s012.json python3 analyze6.py --out check012.md             # = the §17 IFEval table
python3 perseed.py > perseed.txt; python3 nltk_check.py > nltk_check.md
```

All four reproduce the shipped files byte for byte (checked on the release copy). `WSM_ROOT` overrides the release root.

### 22.2 GSM8K (eval seed 7, the §17 protocol; mean over RL seeds 0-5; paired item bootstrap, 2000 resamples)

| row | GSM8K test | GSM-Hard | GSM-Plus testmini |
|---|---|---|---|
| 2 RL (P64) | 68.32 | 41.60 | 53.67 |
| 2b RL, then rung 2 attached | 82.22 | 61.42 | 68.34 |
| 4 RL with rung 2 (CW) | 80.44 | 59.01 | 66.09 |
| **4 − 2** | **+12.12 [+10.77, +13.57]** | **+17.41 [+15.59, +19.27]** | **+12.42 [+11.05, +13.74]** |
| 2b − 2 | +13.90 [+12.53, +15.33] | +19.83 [+18.04, +21.67] | +14.67 [+13.41, +15.98] |
| 4 − 2b | −1.78 [−2.50, −1.01] | −2.41 [−3.35, −1.38] | −2.25 [−3.04, −1.49] |

* Averaged over eval seeds 7 / 11 / 13 (18 runs per row): 4 − 2 = +11.97 / +17.58 / +12.26 and 4 − 2b = −1.96 / −2.62 /
  −2.44. All CIs exclude 0. Every 3-seed GSM8K claim of §17 holds at 6 seeds.
* Row 4 test per RL seed: 81.12 / 81.43 / 78.24 / 81.35 / 79.38 / 81.12.
* RL (one B200 each; output `$ARMDIR/rl/<ARM>/step_000300`):
  `for K in 3 4 5; do GPU=0 bash gsm_stack/engine/run_arm.sh P64_s$K; GPU=0 bash gsm_stack/engine/run_arm.sh CW_s$K; done`.
  * P64 takes ~15 min per seed and CW ~1 h 45 min.
  * u300 in-run dev: CW dev_vote4 84.83 / 85.17 / 86.00 (seeds 0-2: 86.50 / 86.67 / 84.17); P64 dev_acc 80.17 / 77.50 /
    83.00.
* Evals: one line of `checks/seeds35/jobs.tsv` per job. Run it from `uniform/final_eval/` with `checks/seeds35/lists/*.list`
  copied there (§17 harness `fe.sh` / `fejob.sh` / `fepack.sh`, builds `stk` and `stk_r2q`). Per RL seed K on one B200:

  | eval | command | time |
  |---|---|---|
  | row 2, 9 runs | `bash fejob.sh stk s35_H_P64_sK.list` | ~5 min |
  | rows 2b / 4, test + Hard at seed 7 | `bash fepack.sh stk_r2q s35_r2_{P64,CW}_sK_t.list s35_r2_{P64,CW}_sK_h.list` | ~17 min |
  | seeds 11 / 13 + GSM-Plus | `bash fepack.sh stk_r2q s35_r2_<ARM>_sK_e11.list s35_r2_<ARM>_sK_e13.list s35_r2_<ARM>_sK_gp.list` | 1-1.5 h |

  `fe.sh` writes `uniform/final_eval/out/fe/<SET>_<NAME>.jsonl` (`FE_OUT_DIR` overrides). Then run
  `python3 checks/seeds35/reduce_fe.py uniform/final_eval/out/fe <run names>` and rerun `ana35.py`. Runs in
  `checks/seeds35/res/fe/` take precedence over the shipped §17 bundles.
* Caveat: row 4 at eval seed 7 for RL seeds 0-2 is the §17 `stk_r2` run (`wpr_r2cw.list`). Every other row-4 / 2b run
  uses `stk_r2q`; the two builds differ by +0.15 [−0.56, +0.88] on that row (§17).

### 22.3 IFEval-541 (prompt-level strict, RL seeds 0-5 pooled; paired item bootstrap, 10,000 resamples)

| row | prompt strict [95% CI] | per seed s0 … s5 |
|---|---|---|
| 1 plain host | 12.94 [10.17, 15.90] | – |
| 2 host + RL (P) | **45.53** [41.99, 49.20] | 43.62 / 46.77 / 40.30 / 44.55 / 46.77 / 51.20 |
| 3 host + implant | 34.75 [30.68, 38.82] | – |
| 4 host + implant + RL (I) | **48.49** [44.79, 52.22] | 49.17 / 48.24 / 47.69 / 48.24 / 49.91 / 47.69 |
| 2b P host + implant at eval | **53.70** [50.09, 57.30] | 54.16 / 54.71 / 49.17 / 53.97 / 53.79 / 56.38 |

* **4 − 2: +2.96 [−0.28, +6.35]** at 6 seeds. At 3 seeds it was +4.81 [+1.11, +8.56]. 5 of 6 seeds are positive; P_s5 is
  the outlier. The 6-seed CI includes 0.
* 2b − 2: +8.16 [+6.13, +10.38] and 2b − 4: +5.21 [+2.46, +7.89], each positive in 6 / 6 seeds.
* RL + eval of one seed on one GPU: `CUDA_VISIBLE_DEVICES=0 bash checks/seeds35/ifeval_seed.sh {P|I} K`, then
  `python3 checks/seeds35/ifana/reduce_ifeval.py checks/seeds35/res_new/<P|I>_sK <P|I>_sK`.
  * It first runs `uniform/rl24/ifeval/code/setup.sh` (CPU, ~2 min, idempotent; needs `datasets`). That completes the
    code dir from its release homes, fetches the IFEval prompts and regenerates the training data (sha256-checked).
  * This is the script seeds 3-5 ran. It drives `uniform/rl24/ifeval/code`, unchanged, with the recipe of
    `uniform/rl24/ifeval/rl_chain.sh`: 300 updates, 64 prompts x G8, lr 1e-6, T 1, max_new 1024, reward = lm_eval
    prompt-strict; then a greedy IFEval-541 eval with the implant on and off.
  * Time on one A100 40 GB: I ~3 h 15 min training + 8 min eval; P ~1 h 30 min + 4 min.
  * Time on one B200 (seeds 0-2): I ~1 h 53 min, P ~62 min, eval ~4 min.
* **NLTK issue (seeds 3-5).**
  * NLTK 3.10.4 silently skipped the `punkt_tab` download into a group-writable dir. For the first ~35 min of training,
    every prompt with a `number_sentences` instruction therefore scored reward 0: 3.8% of the train prompts, about 2.4 of
    64 per update. This was fixed in the running jobs; the reward was correct from about update 60-90 (I) and 100-125 (P)
    of 300.
  * A group in which all 8 samples score 0 has zero advantage and gives no gradient.
  * The u300 evals ran after the fix. They were re-scored offline with a punkt_tab assertion: 0 differing items for
    I_s3-5 and P_s4-5 (`res/ifeval/*/rescore_diff.txt`). P_s3 uses its remote score file (`FALLBACK_*.txt`).
  * u300 dev of seeds 3-5 is within the seeds 0-2 range for I (55.18 / 62.17 / 58.55 vs 54.46 / 57.83 / 54.70) and for
    P (35.66 / 37.35 / 46.99 vs 39.52 / 41.45 / 33.49) (`ifana/nltk_check.md`).
  * `ifeval_seed.sh` now installs punkt into a private dir with `raise_on_error=True` and asserts it is found.

### 22.4 Not reproducible from the repository alone

* RL checkpoints and resume states: rebuilt by the commands above (12 GSM8K arms, 12 IFEval arms).
* Generated text: the per-item eval jsonls, 221 MB for GSM8K, are rebuilt by the lists. `res/` keeps only the per-item
  correctness and token counts the analysis reads.
* Bit-exact reruns: not expected (bf16, batch scheduling, A100 vs B200). Expect seed-level scatter like the per-seed
  columns above.

## 23. Blog three-seed checks (`checks/seeds3/`)

Six one-seed claims get more seeds: (a) in §5.1; (b), (c), (d), (e) in §5.2; (f) in §3.4. For (b)-(d) two new seeds are
added to the original; (e) and (f) are full 3-seed reruns. Details: `checks/seeds3/README.md` and `RESULTS.md`.

* **Code.**
  * All new runs used the release `rl/` tree plus three trainers with a resume / per-item patch, and
    `rl/library/mixed/make_nl_sft.py` (the aNL host's worked-solution SFT rows) (`checks/seeds3/code_overlay/`).
  * `bash checks/seeds3/rebuild_code.sh DIR` rebuilds that tree. It copies the release dirs, moves four stale
    `rl/library/*.py` copies aside, applies the overlay and puts the W3d init links
    (`rl/results/wiring/H100/sft3_W3e_lr1e-2/links_final.pt`) in `rl/wiring/runs/`.
  * `run_seed.sh` does the rebuild on first use.
* **Hardware.**
  * New runs: **one B200** each.
  * Seed-0 originals of (b)-(d): H100 (m4, LIB, LIBscr) or A100 (m2), in `rl/results/library/figdata` (§12).
  * Analysis: CPU, seconds.

### 23.1 Recompute every table (CPU, no model)

```bash
cd checks/seeds3
python3 try8.py > try8.md            # (a)  < 1 s
python3 summarize.py > summarize.md  # (b), (c)  < 1 s
python3 libpair.py > libpair.md      # (d)  ~5 s
python3 mixpair.py > mixpair.md      # (e), (f)  ~10 s, needs numpy
```

These scripts read `res/` plus the release files `gsm_stack/results/curves/A1*.steps.jsonl`,
`rl/results/library/figdata/{mix_m4_mixLIBnoisy,mix_m2_mixLIBscr,LIB_s0,LIBscr_s0,LIB_s1,mix_m1_mixLIB}_*` and
`rl/results/wiring/figdata/W3d_s{0,1}.jsonl`. They print the shipped `.md` files byte for byte. On the original unreduced
outputs they give the same numbers. `mixpair.py` reproduces the original analysis CIs digit for digit; its `--burn 28`
default skips the bootstrap draws that the original script spent on two sections not shipped here (with `--burn 0` the
(f) CIs move by at most 0.1).

### 23.2 (a) §5.1 A1, try-8 accuracy (3 RL seeds; no new run)

The A1 arms and evals are §16.4: `run_arm.sh A1 / A1_s2 / A1_s3` (seeds 3 / 4 / 6), evaluated with `test1.list A1_N8`,
`seed_A1s2_Bs2.list` and `seed_A1s3.list`. One B200 each, ~2.3-2.6 h RL. `res/a1/try8.items.jsonl.gz` holds their per-try
results.

| | u0 | seed 3 | seed 4 | seed 6 | mean (sd) |
|---|---|---|---|---|---|
| GSM8K test, vote N = 8 | 71.87 | 79.76 | 80.74 | 79.68 | **80.06** (0.6) |
| test, try 8 | 22.7 | 65.7 | 60.0 | 60.7 | **62.1** (3.1) |
| dev 600, try 8 (u300) | 29.00 | 70.67 | 65.33 | 67.67 | **67.89** (2.7) |
| GSM-Hard, try 8 | 15.0 | 50.0 | 46.8 | 46.6 | 47.8 (1.9) |

The one-seed 66% (test) / 71% (dev) is the best of the three seeds. The position reset with zero RL gives try 8 = 71.7%
on test (§16).

### 23.3 (b) m4 noisy-SFT clean-up and (c) m2 scramble-all (mixed task, 150 updates, 3 seeds)

```bash
python scripts/solve_veto.py --device cuda                                  # once: runs/veto_qwen3.pt, ~5 min GPU
CUDA_VISIBLE_DEVICES=0 bash checks/seeds3/run_seed.sh m4_s1                 # also m4_s2, m2_s1, m2_s2
python3 checks/seeds3/reduce_run.py runs/seeds3/runs/m4_s1 m4_s1 checks/seeds3/new/m4_s1
cd checks/seeds3 && S3_RES=new python3 summarize.py
```

Each run takes one B200, 2 h 45-50 min: 64 prompts x G8, lr 1e-6, Dr.GRPO, dev 400 at u0 / 25 / 50 / 100 / 150, and
test 400 + per-tool leave-one-out at u150. `run_seed.sh` also builds the SFT host (`sft_warm.py`, `--noisy 0.3` for m4;
a few minutes) and installs the IFEval checker libs and NLTK punkt_tab.

* **(b)** Off-type fired calculator calls per answer, greedy dev, seeds 0 / 1 / 2.
  * On BBH (costly): u0 0.16 / 0.15 / 0.15 → u150 **0.04 / 0.05 / 0.03**.
  * On IFEval (cheap; leave-one-out of calc changes no IFEval score): u0 0.22 → u25 0.01 / 0.00 / 0.00 → u150
    **0.23 / 0.00 / 0.06**. All seeds suppress them early; 1 of 3 drifts back.
  * Test acc: 62.8 / 63.8 / 63.8; tools off 32.2 / 34.5 / 34.2.
* **(c)** In-wires closed (< 5% let through on test 400, of those that fire): **4 / 7, 5 / 9, 5 / 10**.
  * expr_gen and bool_gen are closed in all 3 seeds.
  * The unscrambled calculator is closed in seed 0 only.
  * Test acc: 37.8 / 44.0 / 38.8; dev u150: 35.2 / 45.2 / 40.2.
* The misfire probe (within-group reward cost) was not re-run on the new seeds (seed 0 only: −0.39 BBH vs −0.08 IFEval).

### 23.4 (d) LIB vs LIBscr on GSM8K (300 updates, third pair)

```bash
CUDA_VISIBLE_DEVICES=0 bash checks/seeds3/run_seed.sh LIB_s2      # and LIBscr_s2 on a second GPU
python3 checks/seeds3/reduce_run.py runs/seeds3/runs/LIB_s2 LIB_s2 checks/seeds3/new/rlctl/LIB_s2
```

Each run takes one B200, ~4 h 15 min. The original jobs hit a 4 h limit and resumed for 15 min.

| pair | LIB test / Hard | LIBscr test / Hard | LIB − LIBscr test | LIB − LIBscr Hard |
|---|---|---|---|---|
| s0 (H100) | 69.37 / 56.10 | 69.29 / 51.86 | +0.08 | +4.25 |
| s1 (B200) | 70.36 / 56.18 | 65.50 / 41.62 | +4.85 [+2.20, +7.35] | +14.56 [+11.75, +16.98] |
| s2 (B200) | 71.34 / 55.19 | 66.72 / 51.33 | **+4.62 [+2.27, +6.97]** | **+3.87 [+1.36, +6.22]** |
| mean (3) | 70.36 / 55.83 | 67.17 / 48.27 | +3.18 | +7.56 |

* CIs come from a paired item bootstrap with 2000 resamples. s0 has no per-item test data. s1 + s2 pooled: +4.74 [+2.92, +6.60] /
  +9.21 [+7.43, +11.03].
* The scrambled run kept the calculator in 2 of 3 seeds. LIBscr test calls/q: 2.19 / 0.00 / 2.42. The s1 Hard gap is
  mostly the lost calculator.
* Big numbers (dev u300, scaled half, implants on → all off), over 4 LIB runs (s0, s1 A40 original, s1 B200, s2):
  **64.56 → 37.38**.

### 23.5 (e) §5.2 mixed task: all implants vs plain RL (150 updates, 3 seeds)

```bash
for K in 0 1 2; do CUDA_VISIBLE_DEVICES=$K bash checks/seeds3/run_seed.sh m1m_s$K & done; wait   # ~2 h 50 min each
for K in 0 1 2; do CUDA_VISIBLE_DEVICES=$K bash checks/seeds3/run_seed.sh aNL_s$K & done; wait   # ~21 min each
CUDA_VISIBLE_DEVICES=0 bash checks/seeds3/run_seed.sh m3_s1      # optional third arm: m3_s1, m3_s2, ~21 min each
for R in m1m_s0 m1m_s1 m1m_s2 aNL_s0 aNL_s1 aNL_s2 m3_s1 m3_s2; do
  python3 checks/seeds3/reduce_run.py runs/seeds3/runs/$R $R checks/seeds3/new/rlctl/$R; done
cd checks/seeds3 && S3_RES=new python3 mixpair.py
```

* One B200 per run. Same recipe as (b) / (c): 64 prompts x G8, lr 1e-6, Dr.GRPO, dev 400, test 400 (stratified, same
  items for all runs) at u150.
  * m1m = all implants, `--cw_mask 1` (out-wires act only where an implant emits), from the tagged SFT host.
  * aNL = plain RL from `sft_host_nl`: the same SFT recipe on the same questions, with every tool call rewritten as a
    worked solution. `run_seed.sh` builds its rows with `make_nl_sft.py` (CPU, seconds; SHA-256 checked).
  * m3 = plain RL from the tagged host (seeds 1, 2).
* Test 400 at u150, seed means, 95% item bootstrap (10000 resamples):

  | arm | full | gsm | bbh | combo | ifeval | implants off |
  |---|---|---|---|---|---|---|
  | m1m (s0-2) | **57.8** [53.2, 62.1] | 37.9 | 81.3 | 21.9 | 70.3 | 32.7 [28.5, 36.8] |
  | aNL (s0-2) | **44.7** [40.3, 49.2] | 27.6 | 70.6 | 19.1 | 43.5 | - |
  | m3 (s1-2) | 33.9 [29.6, 38.0] | 26.8 | 46.7 | 12.3 | 37.5 | - |
  | **m1m − aNL (paired)** | **+13.1 [+8.7, +17.6]** | +10.3 [+1.5, +19.1] | +10.7 [+2.9, +18.5] | +2.7 [−6.6, +12.0] | +26.8 [+17.8, +35.5] |

* Seed SD of the full score: m1m 1.1, aNL 0.8. m1m − m3 +23.9 [+19.3, +28.6]; aNL − m3 +10.8 [+6.4, +15.2].
* Leave-one-out (m1m): the IFEval implant +7.6 [+5.3, +10.1], the BBH tracker +4.8 [+2.8, +7.0], the calculator +1.0.
* The one-seed original m1 (`--cw_mask 0`, H100: 68.2, implants off 36.8) was not rerun; the blog's 3-seed claim is m1m.

### 23.6 (f) §3.4 distilled links W3d on GSM-Hard (300 updates, 3 seeds)

```bash
for K in 0 1 2; do CUDA_VISIBLE_DEVICES=$K bash checks/seeds3/run_seed.sh W3d_s$K & done; wait   # ~1 h each incl. test
for K in 0 1 2; do python3 checks/seeds3/reduce_run.py runs/seeds3/runs/W3d_s$K W3d_s$K checks/seeds3/new/rlctl/W3d_s$K; done
```

* One B200 per run, 56-59 min. `train_wiring.py --arm W3d --init_links runs/sft3_W3e_lr1e-2/links_final.pt`, 300 updates,
  dev 800 every 25; then a separate test call (`--arm W3e --eval_all 0 --test_sets test,hard --eval_ckpt .../step_000300`)
  that scores GSM8K test and GSM-Hard with the links on and with the host only. `reduce_run.py` reads it from
  `runs/seeds3/runs/TEST_W3d_sK/`.

  | seed | GSM8K test on / host only | GSM-Hard on / host only | **Hard on − host only** (paired) | dev u300 links effect |
  |---|---|---|---|---|
  | s0 | 71.0 / 71.0 | 55.8 / 41.2 | **+14.6** [+12.4, +16.8] | +10.2 |
  | s1 | 71.0 / 70.7 | 55.4 / 41.9 | **+13.5** [+11.3, +15.8] | +11.9 |
  | s2 | 69.7 / 70.1 | 54.7 / 43.1 | **+11.7** [+9.4, +14.0] | +12.0 |
  | mean | | | **+13.3** (seed SD 1.5) | |

* On GSM8K the links add nothing (+0.0 / +0.4 / −0.4). The original W3d runs (s0 H100, s1 A100) had dev links effects of
  +11.2 / +11.5 at u300 (§12).

### 23.7 Not reproducible from the repository alone

* Checkpoints, rollouts, generated text and trainer logs: rebuilt by `run_seed.sh`. The shipped `steps.jsonl` files
  keep per-update scalars and every dev record. `test_eval.json` keeps the per-item test correctness.
* (a): the per-item fe.sh outputs (rebuilt by the §16.4 lists). Only per-try valid / right ships.
* The misfire probe for seeds 1-2 (not run).
* The broken-calculator 3-seed rerun of the same batch (arm R): not quoted by the final blog, not shipped.
* RL is not bitwise deterministic across GPUs. A rerun should land within the seed spread above, not on the same digits.

## 24. Off-task benchmarks after RL, implants attached (`checks/offtask/`)

`checks/offtask/README.md` has the files and method. `checks/offtask/RESULTS.md` has every table with per-seed values and fire statistics.

**Question.** Does RL with the implants hurt general skills more than plain RL does?

**Rows.** These are the §5.1 GSM8K rows at u300, with RL seeds 0–2 per arm:
- **Qwen3-0.6B-Base:** base, P64_s0..2 (plain RL, row 2) and CW_s0..2 (RL with the stack, row 4).
- **Qwen3-1.7B-Base:** base17, P17_s0..2 and CW17_s0..2.

**Scoring.** Every forward pass is scored twice from the same pass: host only, and host + the rung-2 DualImplant write (calc_s4 + chain2). So rows 3 (base + rung 2), 2b (P + rung 2) and 4h (CW host alone) come from the same jobs.

**Benchmarks.** Full test sets, 30,133 items per host:
- MMLU 5-shot (14,042)
- ARC-Challenge acc_norm (1,172)
- HellaSwag acc_norm (10,042)
- WinoGrande partial scoring (1,267)
- NQ-open, 5-shot greedy generation, EM (3,610)

The lm-eval scoring rules are reimplemented in `ot_hook.py`; lm-eval is not used.

**CIs.** Two-level bootstrap (seeds, then items) with 2,000 resamples and fixed seeds.

**Hardware and runtime.** Original runs: one B200 per job, through a cluster queue.

| run | time per row |
|---|---|
| 0.6B row, alone on the GPU | about 8 min |
| 0.6B, 5 rows packed on one GPU (`ot_pack.sh`) | about 38 min |
| 1.7B, 3–4 rows packed | about 30–33 min |

Total: about 1 h (0.6B) + 1.2 h (1.7B) on one B200. The eval is deterministic (log-likelihood scoring and greedy NQ); the only seeds are the 3 RL seeds per arm.

**Requirements.**
- The u0 host `$CKPT_U0` (§16.1).
- The arms `$ARMDIR/rl/{P64,CW}_s{0,1,2}/step_000300`, built by `GPU=0 bash gsm_stack/engine/run_arm.sh P64_sK` / `CW_sK` (§16.3).
- The 1.7B arms `$ARMDIR17/rl/{P17,CW17}_s{0,1,2}/step_000300` (`uniform/rl28/gsm8k/`; `ARMDIR17` defaults to `$WSM_DATA/rl28g`).
- The 1.7B veto `results/gsm_bigger/veto_q17.pt`.
- Python package `pyarrow`, for the data download only.

**Commands.**

```bash
bash checks/offtask/dl.sh        # 5 benchmarks -> $WSM_WORK/offtask/bench (CPU, ~1 min); sha256sum -c checks/offtask/bench.sha256
export CUDA_VISIBLE_DEVICES=0
bash checks/offtask/ot_job.sh 06 base base
bash checks/offtask/ot_pack.sh 06 P64_s0:P64_s0 P64_s1:P64_s1 P64_s2:P64_s2 CW_s0:CW_s0 CW_s1:CW_s1 CW_s2:CW_s2
bash checks/offtask/ot_pack.sh 17 base17:base P17_s0:P17_s0 P17_s1:P17_s1 P17_s2:P17_s2
bash checks/offtask/ot_pack.sh 17 CW17_s0:CW17_s0 CW17_s1:CW17_s1 CW17_s2:CW17_s2
python3 checks/offtask/mk_items.py $WSM_WORK/offtask/j06/ot checks/offtask/j06/res base P64_s0 P64_s1 P64_s2 CW_s0 CW_s1 CW_s2
python3 checks/offtask/mk_items.py $WSM_WORK/offtask/j17/ot checks/offtask/j17/res base17 P17_s0 P17_s1 P17_s2 CW17_s0 CW17_s1 CW17_s2
cd checks/offtask && python3 ana_ot.py > ana06.md && OT_SIZE=17 python3 ana_ot.py > ana17.md     # CPU, ~7 min each
```

The jobs run `stk/ot_hook.py` inside `dig_san/tw_dump_lean.py` with BMAX=16 and SAN_SMAX=1024. The code trees come from the overlays:
- 0.6B: `checks/offtask/j06/tree` = `uniform/final_eval/trees/stk_r2q` + the hook, run via `gsm_stack/run_hook.sh`;
- 1.7B: `checks/offtask/j17/tree` = `uniform/rl28/gsm8k/trees/stk_r2q_17` + the hook, run via `uniform/rl28/gsm8k/fe/run_hook.sh`.

**CPU check (no GPU).** The shipped per-item results (`j06/res`, `j17/res`; no log-likelihoods, no text) reproduce the tables exactly:

```bash
cd checks/offtask && OT_ROWS=r06.json python3 ana_ot.py | diff - ana06.md && cmp r06.json rows06.json
OT_SIZE=17 OT_ROWS=r17.json python3 ana_ot.py | diff - ana17.md && cmp r17.json rows17.json
```

**Expected.** Difference 4 − 2 (RL with implants − plain RL), in points, with 95% CI:

| host | MMLU | ARC-C | HellaSwag | WinoGrande | NQ-open |
|---|---|---|---|---|---|
| 0.6B | −0.14 [−0.38, +0.10] | +1.14 [+0.26, +2.10] | −0.11 [−0.29, +0.08] | −0.39 [−1.29, +0.50] | −0.09 [−0.46, +0.27] |
| 1.7B | −0.08 [−0.32, +0.14] | −1.65 [−2.99, −0.31] | −0.23 [−0.43, −0.03] | −0.21 [−1.16, +0.74] | +0.18 [−0.21, +0.56] |

Absolute scores:

| row | MMLU | ARC-C | HellaSwag | WinoGrande | NQ-open |
|---|---|---|---|---|---|
| 0.6B base | 52.44 | 38.48 | 53.82 | 58.25 | 6.95 |
| P64 (mean of 3 seeds) | 52.61 | 37.43 | 53.99 | 59.01 | 7.31 |
| CW (mean of 3 seeds) | 52.47 | 38.57 | 53.89 | 58.62 | 7.22 |
| 1.7B base | 62.59 | 44.62 | 66.33 | 64.72 | 12.66 |
| P17 (mean of 3 seeds) | 62.53 | 45.42 | 66.71 | 64.14 | 12.98 |
| CW17 (mean of 3 seeds) | 62.45 | 43.77 | 66.48 | 63.93 | 13.16 |

**The implants are inert off-task.** They never write on a scored position, so rows 3 = 1, 2b = 2 and 4 = 4h exactly, and every NQ generation is byte-identical. They write somewhere in the prompt on 0.04% of MMLU items.

**What a GPU rerun will and will not match.** The tables above are fixed by the shipped per-item results. bf16 numerics on a different GPU type, or a retrained arm, will move per-row scores by a few tenths. The fire statistics (all zero) should not change.

**Not shipped:**
- checkpoints;
- the benchmark data (fetched by `dl.sh`);
- the raw per-item rows with log-likelihoods and NQ text (58 MB per size).

## 25. Sourced blog numbers (§4.4 and related; `checks/sourced/`)

`checks/sourced/README.md` lists every number with its script and inputs. `checks/sourced/RESULTS.md` has the definitions and the blog wording.

All scripts here are CPU-only, standard-library Python, and re-derive the numbers from existing run outputs that ship in the release. Run them from the release root (`WSM_ROOT` overrides it). Each prints markdown; the expected output is the matching `*.out.md`.

| item | command | expected | runtime |
|---|---|---|---|
| (a) §3.3 size-ordered veto | `python3 checks/sourced/veto33.py --seeds 20` | early configs 4/8 flagged (3 transient); C36 fI 0/20, fI1 0/8 (`veto33.out.md`) | 1 s |
| (a′) all 28 fI seeds | `python3 checks/sourced/veto33.py` | fI 0/28, C36 0/36 (`veto33_s28.out.md`) | 1 s |
| (b) §3.4 BBH swap | `python3 checks/sourced/gbc_swap.py` | gbC2_s0 Δ BBH −10.4 (−22/212); the other runs −1.9 to +0.5 | <1 s |
| (c) §4.4 S2F vs H | `python3 checks/sourced/s2f_vs_h.py` | S2Fw +4.93 [+2.96, +6.90]; S2Frw +4.25 [+2.27, +6.22]; H 67.70 (n = 1,319, 10k paired bootstrap, seed 0) | 13 s |
| (d) §5.2 `<calc>` fade | `python3 checks/sourced/calc_fade.py` | u1–5 → u41–50: control 79→33 / 82→17; no-"=" 81→82 / 83→94; 25%-corrupted 83→96 / 84→92 | <1 s |
| (e) §5.1 IFEval 2b − 4 | `python3 checks/sourced/ifeval_2b4.py` | +4.31 [+1.17, +7.33] seed-averaged prompt-strict (541 prompts); 546 vs 1,364 tokens per answer | 12 s |
| (f) §4.2 IFEval | `python3 checks/sourced/ifeval_42.py` | 1/2/3 instructions: 56.4/33.1/10.5%; implant-written tokens ≤ 2.2% of all, ≤ 7.4% on fired prompts | <1 s |
| (g) §4.4 never solved in 32 | `python data/prepare_gsm.py` once, then `python3 checks/sourced/nv32.py dev checks/sourced/res/fe/dev_sc8_s{7,11,13}.items.jsonl checks/sourced/res/fe/dev_sc4_s{7,11,13}.items.jsonl` | 23/600 = 3.83% never solved in 32 (and in 36); 3 WRONG + 2 AMBIG + 1 ILLPOSED of 23 → 22% (26%); sound-gold unsolved 17/600 = 2.8% | 1 s |
| (h) §1.2 Read layer on the full vocabulary | `bash checks/sourced/vocabjob/run.sh` (torch, transformers, downloads Qwen3-0.6B-Base) | 151,936 rows × 26 symbols, 0 mismatches for calc_s3 and calc_s4; margins ≤ −2 / ≥ +6 | 10 s |

Inputs for each item:
- **(a)** `rl/results/wiring/H100/L300_*/steps.jsonl` and `checks/calcimp/tt/res/train_steps.jsonl.gz`.
- **(b), (d)** `rl/figures/v3/data/`.
- **(c), (g)** the reduced fe_hook rows in `checks/sourced/res/fe/`, which come from the §16 u0 host.
- **(e)** `uniform/rl24/ifeval/res/` (gzipped score files).
- **(f)** `ifeval/transfer/results/reduced/`.
- **(g)** also needs `data/gsm8k/{train.jsonl,shots_v1.json}`, which rebuild the dev order and check the gold of every item.

**Verified on the release copy** (outputs identical to the originals):
- (a), (b), (c), (d), (f), (g). In (c) and (g) only the printed file names changed (`.items.jsonl`). (g) was checked with the original `shots_v1.json`.
- (e), on the gzipped score files `uniform/rl24/ifeval/res/*/eval_u300.*.score.json.gz`.

**Not re-run.** (h) was not re-run in the release check (no model download); the shipped output is the original CPU run.

**Not derivable from the repository.**
- CW-in s0 and CWfx s0/s1 in (a) have no step logs. They are quoted from `rl/wiring/NOTEBOOK.md`.
- The (g) gold labels are hand labels (`nv_gold_labels.json`).
- The "half selection / half hard" split in §4.4 comes from the §16 oracle numbers, not from this folder.
