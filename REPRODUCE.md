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
| First implant: a calculator | `micromachines/`, `scripts/run_gsm.py`, `results/gsm*`, `evidence/`, `tests/` | [1](#1-first-implant-the-calculator) |
| Teaching the model how to use an implant | `finetuned_phase/`, `bbh_generic/ft100/` | [2](#2-teaching-the-model-how-to-use-an-implant-fine-tuning) |
| Give Qwen eyes | `finetuned_phase/eyes_and_finetune/`, `evidence/eyes_sweep/` | [3](#3-the-eyes-implant) |
| Scaling laws / how Claude builds an implant; the overfitting trap | `bbh_generic/PROTOCOL.md`, retired `micromachines/circuits/` | [4](#4-a-trap-implants-can-overfit-the-test-look-alike-data) |
| Climbing IFEval | `ifeval/transfer/` (headline run), `ifeval/heldout/`, `ifeval/full14_compiled/`, `ifeval/baselines/` | [5](#5-climbing-ifeval-14-rule-implants) |
| Climbing BIG-Bench Hard | `bbh_generic/` (`integrate/`, `callgate/`, `ft100/`, `nl_control/`) | [6](#6-climbing-big-bench-hard-8-tools-call-fine-tune-call-gate) |
| Add one implant at a time | `ifeval/transfer/`, `bbh_generic/transfer/` | [7](#7-add-one-implant-curves) |
| Do implants transfer across models? | `results/gsm_bigger/`, `ifeval/transfer/`, `bbh_generic/transfer/`, `transfer/xfamily/`, `ifeval/xfamily/` | [8](#8-transfer-to-bigger-qwen-models-and-other-families) |
| Mistake detector | `micromachines/rethink_circuit_v2.py`, `scripts/rethink_v2/`, `results/rethink_v2/` | [9](#9-mistake-detector--rethink) |
| Topic blocker | `topic_fence/`, `topic_fence/inweights/`, `topic_fence/baselines/` | [10](#10-topic-blocker) |
| Writing into intermediate representations | `latent_write/` (+ `v2/`, `causal/`) | [11](#11-latent-write) |
| RL with implants | `rl/` (`gsm_grpo/`, `wiring/`, `library/`, `interp/`, `figures/`; `engine2/` = WIP engine) | [12](#12-rl-with-implants-gsm8k-grpo) |
| Figures | `figures/blog/`, `figures/blog/review_final/`, `rl/figures/`, `rl/figures/v3/` | [13](#13-figures) |
| GSM8K push past the RL+calc host | `gsm_push/` | [15](#15-gsm8k-push-single-call-implant-stack-on-the-rlcalc-host) |


## 1. First implant: the calculator

Blog: "52% -> 53% on GSM8K, 27% -> 35% on GSM-Hard" (frozen Qwen3-0.6B-Base); "+8 / +6 / +1 on GSM-Hard for 0.6B / 1.7B /
4B"; "multiplication 4% -> 100% on a frozen Qwen2.5-0.5B"; "every pair up to 199, four operations, plus negatives:
0 mistakes"; figure "Accuracy as the numbers get longer".

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
(`results/gsm_bigger/reanalysis/gsm_reanalysis.md`). The blog's "160k cases up to 199" is the circuit test sweep; the
shipped `tests/test_calc_circuit.py` runs 0..59 exhaustively plus 12k random pairs.


## 2. Teaching the model how to use an implant (fine-tuning)

Blog: three fine-tunes of Qwen2.5-0.5B on the same GSM8K solutions (no calls / calls filled by the model / calls filled
by the calculator implant): GSM8K ties (~33-35%), GSM-Hard ~13 -> 23%, unseen arithmetic templates ~51 -> 99%; and "about
100 examples ... 75% vs 29%" on BBH-style questions.

Code: `finetuned_phase/eyes_and_finetune/` (snapshot of the earlier phase; `ft.py`, `gen_eval.py`, `seedjobs.sh`; see
`finetuned_phase/README.md` "Reproduce"). The BBH 100-example study is `bbh_generic/ft100/` (section 6.4).

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

Blog: with 32 worked solutions + the eyes implant, 95% vs 4% without eyes; plain fine-tuning needs ~4,000 solutions for 93%
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

Blog claims backed here: 58% -> 72% prompt-level strict on all 541 IFEval prompts (fixed 80 / broke 3), silent prompts
byte-identical (324/324); on the 299 "odd" prompts never used for building 58% -> 71% (+42 / -1); per-rule chart; thinking
mode 59%; readers catch ~40-75% of rephrasings (misses fail safe); all 14 rules compiled into plain weights (~9,300 ReLU
units), byte-identical to the reference on 541/541; add-one-rule curve; transfer to 1.7B / 4B (+9 / +4 points); SmolLM2 port.

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
building) **57.53 -> 71.24 (+42 / -1)**; EVEN 242 58.26 -> 73.14 (+38 / -2). Per-rule chart ("Each rule implant, one by
one") = the ALL per-rule instruction-strict table in `results/transfer_tables.md` (0.6B column): no_comma 31 -> 62 / 66,
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
The blog's IFEval bar figures were first drawn from this run; see section 13 for which run each figure uses.
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

Blog sections: "Climbing BIG-Bench Hard with implants", "Teaching the model how to use an implant" (the ~100-example
claim), "Do implants transfer across models?" (BBH part), and the BBH add-one-implant curve.

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

Expected (blog "Eight implants on BIG-Bench Hard", "All 27 tasks"): 12 tool tasks **43.27 → 84.33** (FT alone 38.10);
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

Expected: Team A at 100 rows, fresh test with tools 76.3 / 73.7 (seeds 0 / 1) vs base 29.0 (blog "75% vs 29%"); call rate
97-100% already at 32 rows. Team B 12k-row host (ft3) without tools: real BBH 95.7, reworded held-out 26.7, with tools
87.3 (blog "96% / 27% / 87%"). NL control, Team A real BBH 39.9 / 56.5 / 55.1 / 72.6 (n 32 / 100 s0 / 100 s1 / 300) vs tool
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

### 7.1 IFEval: add one rule at a time (figure "Add one rule implant at a time")

Commands, expected numbers and noise: section 8.2 (`bash ifeval/transfer/run_all.sh` runs the k = 0..14 curve on each host;
`python ifeval/transfer/plot_curve.py` draws it). The curve values are in `ifeval/transfer/results/ifeval_curves.json`.
The order of rules is fixed in `ifeval/transfer/curve2.py` (it was chosen from Team D's own FRESH-set runs, which are
not in the repo, so the order itself is not re-derived).

### 7.2 BBH: add one tool at a time, all 27 tasks (figure "Add one implant at a time, all 27 BBH tasks")

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
python ifeval/transfer/plot_curve.py       # blog figure "Add one rule implant at a time"
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

Blog: AUROC 0.85 (GSM8K) / 0.84 (GSM-Hard); GSM8K 52 -> 56, GSM-Hard 27 -> 29 with the detector; calculator + detector
35% GSM-Hard / 57% GSM8K; fires on 66-75% of problems.
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

Blog: blocks 98-99% of requests, near-misses ~1% (math), normal requests 10% vs 6%; system prompt ignored, strict prompt
blocks 86% of normal requests, keyword list can't separate doing from mentioning; monitor cuts off drift.
Commands: `topic_fence/README.md` "Reproduce" (dev generation with Qwen2.5-7B, probes, monitors, `run_test.sh`), then the
compiled-in-weights version `topic_fence/inweights/README.md`. Expected: `topic_fence/results/q17_{math,ml}_table.txt`;
compiled == emulated on 480/480 dev replies, silence 1998/1998 rows, and every test metric identical
(`topic_fence/inweights/results/`). Runtime ~1.5 h per fence on one H100 (generation + Qwen2.5-7B judge). Greedy; the
dev set cannot be regenerated row for row (LLM-written), so the fitted probes are shipped in `topic_fence/fitted/`.


## 11. Latent write

Blog: (23*7)+58 with no CoT 15% -> 96-99% (any layer <= 12), written wrong value followed 98%, word problem 3% -> 98% vs
70% for a text tool line; strictly left-to-right version gets nothing.
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
Triton needs `Python.h` (set `C_INCLUDE_PATH` in `rl/gsm_grpo/env_gcr.sh` if your venv lacks it).

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
| "switch the calculator off ... 4 points worse than plain RL on GSM-Hard (40% vs 44%)" | arm 2 off 40.2 / 39.3 / 40.6 (mean 40.0) vs arm 1 mean 43.7: -3.7; s0/s1 paired -3.30 [-4.93, -1.78] | u300 | same |
| figure "Test accuracy after 300 RL updates and dev accuracy during RL" | bars = mean of 3 seeds (min-max whiskers); dev curve = mean of 3 seeds | u0-300 | `rl/figures/make_blog_figs_rl.py` (`rl1_calc_after_rl.png`) <- `rl/figures/data/`, values in `rl/figures/blog_figs_rl.json` |
| "unit-word style from 30% to 17% of equations" | 29.8 (base) -> 23.5 (plain RL) -> 16.7 (RL + calc) | **u64**, A100 traces, one seed | `rl/interp/stats.json`, `rl/interp/NOTEBOOK.md` Q2 |
| "copies its answer 99% of the time" | 99.0 / 99.2 / 99.4 | u64, A100 traces | `rl/interp/stats.json` |
| broken calculator "starts at 7%, climbs back to 37%" | dev 7.5 -> 36.6 / 37.9 (A100); 7.0 -> 36.9 / 36.4 (H100) | **u64**, s0/s1 | `rl/results/gsm_grpo/{A100,H100}/abl/aR_*/steps.jsonl` |
| "broken calculator goes from firing on 75% of equations to 12%", unit word after every number | 74.6 -> 12.1 (s1 11.2); unit words 29 -> 86 / 89% | u64, H100 traces | `rl/interp/stats_aR.json`, NOTEBOOK Q4 |
| figure "RL with and without the calculator" | left: dev 44/50, 53/60, 56/65 (off/on); right: the two rows above | u64, A100 traces, one seed | `rl/interp/make_fig.py` (v1) / `make_fig_v2.py` (right panel only); **need the traces** (not released) |
| "dozens of times faster than our first HuggingFace generation loop" | v2 engine 19,800 tok/s (H100, 512 rows, implant on) vs v1 HF loop 207 tok/s and HF `.generate` 288 tok/s (A100, 16 rows): ~70-96x across GPUs and batch sizes; single stream ~13x | — | `rl/results/gsm_grpo/H100/clean/rollout_*.json`, `rl/results/gsm_grpo/A100/prof_v1_a100.json` |
| in-wire + out-wires "add fewer weights than the implant" (CW-full) | 286,469 wire params vs 2,395,647 implant nonzeros (12%) | — | `rl/results/wiring/H100/L300_CW_s0_kl001evalall0/steps.jsonl` (`params`) |
| CW-full: "first vetoes almost everything", "skip 1-2 digit numbers ... keep it for 5+ digits", "the veto fades" | greedy-dev veto share by largest operand (1-2 / 3-4 / 5-6 / 7+ digits): u25 97/90/44/40, u100 69/24/0/2, u250 18/3/0/0, u300 32/6/1/0; dev u300 67.5 | u0-300, H100, s0 (s1 never vetoes; across 8 correct-calculator in-wire runs 4 show a size-ordered veto, 3 of them transient: "the other runs just call every time") | same `steps.jsonl` (`dev.dev_veto_by_digits`); figure `rl/figures/make_rl6.py` (`rl6_wiring.png`; older version `figures/blog/cw_veto.py`) |
| half-broken calculator: "drags the model accuracy down to 19-36%", "in five different runs ... letting through 76-90% of x and ÷ calls, blocks almost every + and − call" | hand trigger dev 19.0 / 18.8 / 36.0 (CWinOx s0-s2), 18.3 / 17.8 (CWinOlr s0/s1); x÷ let through 89.8 / 76.4 / 88.4 (CWinOx), 95.0 / 80.3 (CWinOlr); +− 5.1 / 6.1 / 5.3, 0.4 / 0.2; learned in-wire dev 65.0 / 64.9 / 67.4 / 66.1 / 66.6 vs no calculator 61.3 / 60.1 / 64.3 / 61.3 / 62.0 | u300 (rollout calls pooled u291-300), s0 A40, s1 H100, s2 A100 | `rl/results/wiring/figdata/CWinO{x,lr}_s*.jsonl`, `fig_competence_map.json`; figure `rl/figures/make_blog_figs_rl.py` (`rl3_competence.png`, 3 CWinOx seeds); commands: `rl/wiring/q4_a100.txt`, `q5_h100_g0.txt`, `q6_h100_g0.txt`, `runs_a40_g1.txt` |
| "attached every implant at once ... BBH implants never fire on a math problem ... scrambled BBH arithmetic implants shut off within 25-50 updates ... same GSM8K score (69%); GSM-Hard 56% vs 52%, one seed" | LIB s0 GSM8K 69.4 / GSM-Hard 56.1; LIBscr s0 69.3 / 51.9; scrambled expr_gen / calc_gen at the 5% floor by u25-50 | u300, H100, s0 | `rl/library/NOTEBOOK.md`, `rl/results/library/figdata/LIB*_steps.jsonl`; figure `rl/figures/make_rl45.py` (`rl4_toolbox.png`); launch `rl/library/launch/h100_LIB{,scr}_s0.sh` |
| mixed tasks: "68% on held-out questions vs 36% for plain RL, 37% with the implants off", "tracker costs 15 points on BBH, the IFEval implant 35 points on IFEval", "calculator calls inside BBH answers fade from 0.16 to 0.04", scrambled "38%", "One seed so far" | m1 68.2 (off 36.8), m3 36.3, m2 37.8; LOO bbh 87.6 -> 72.3, ifeval 81.5 -> 46.7; m4 calc@bbh 0.16 -> 0.04, calc@ifeval 0.22 -> 0.23 | u150, H100 (m1, m4), A100 (m2), A40 (m3), test 400 stratified | `rl/results/library/figdata/mix_m{1,2,4}_*`, `rl/library/NOTEBOOK.md` "Mixed-task FINAL"; figure `rl/figures/make_rl45.py` (`rl5_mixed.png`); launch `rl/library/launch/*mix*.sh` with `--updates 150 --prompts 64 --ckpt_every 150 --test_n 400` |
| distilled links: "RL makes them about 15-20% stronger and roughly doubles the gain they carry, to about +11 points (+24 on big numbers), in all three seeds"; animation "links grow 14-20% stronger ... gap widens from about 5 to 11 points" | W3d links-on minus links-off on dev: u0 +4.9 / +6.0 -> u300 +11.3 / +11.5 (s0/s1), +10.6 at u250 (s2, OOM at u251); scaled half +24.6 / +24.0; link norms +15-20% | u300, H100 (s0), A100 (s1, s2) | `rl/results/wiring/figdata/W3d_s{0,1}.jsonl` (s2 log not shipped), `rl/wiring/NOTEBOOK.md`; animation `rl/wiring/make_anim.py` -> `rl/figures/anim_w3d_links.gif`; commands `rl/wiring/q3_a100.txt`, `q5_a100.txt` |
| "wiring the calculator in only through zero-initialized links ... links stay near zero" | W2 (zero-init links next to the hand write): link maps 0.044 / 0.043 / 0.012 at u300, links-off = 72.0 = links-on; links only, no hand write: W3e (embedding init, gate 0.1) 63.9 and W3r (random init) 61.1 at u300, maps stay at init; a links-only arm with exactly zero-init gates (W3z) was **not run** | u300 (u500 runs read at u300), H100 / A40, s0 | `rl/results/wiring/H100/L500_{W2,W3e}_s0/`, `rl/results/wiring/A40/L500_W3r_s0_fused0/`, `results.md` |
| broken calculator + in-wire: "vetoes every call within 25 updates", "then learns like plain RL" | CWinR: 100% of greedy-dev calls vetoed from u25; dev 7.0 -> 58.9 (u100) -> 61.1 (u300) vs plain RL (arm 1, H100) 59.9 / 62.8 at u300 and arm R 36.6-37.9 at u64 | u300, H100, s0 | `rl/results/wiring/H100/L300_CWinR_s0_evalall0/steps.jsonl`; `rl/results/gsm_grpo/H100/long/a1_*` |
| (earlier draft, not in the current blog) "think again" implant: "RL turns it down to about 4% of answers" | arm 4 rethink fires on 67.3% of test answers at u0, 4.6 / 4.2% at u300 | u300, H100, s0/s1 | `H_a4_s*_u300.json`, `compile_long.py` (rethink frac) |
| "Letting RL also train its detector, or the phrase itself, doesn't change that" | 4cRL fires on 13.9 / 11.4% of dev answers at u300 and changes 0.0 / 0.9%; 4sA (soft phrase) fires on 14.1%, changes 0.5%, fixes 0.2% | u300, A100/H100, dev only | `rl/results/gsm_grpo/*/long/a4c*`, `a4sA*`; ledger section 5 |

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
W3d: 3, LIB: 2).

**Not reproducible from the repo alone / caveats.**
* No checkpoints are released; every number after u0 needs the training runs (~1 GPU-hour each).
* The throughput ratio is across GPUs and batch sizes (v1/HF on A100 at 16 rows, v2 on H100 at 512 rows); there is no
  same-GPU, same-batch comparison. The v1 engine that `prof_v1.py` profiles is not released.
* Interp traces (generated text) are not released; `rl/interp/make_fig.py` / `make_fig_v2.py` (the blog figure "RL with
  and without the calculator") and `analyze.py` need them: regenerate with `rl/interp/run_*.sh` from your own 64-update
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
The blog figure "RL with and without the calculator" is `rl/interp/make_fig.py` and needs the unreleased traces
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
| RL engine speed-up ("dozens of times faster") | no same-GPU, same-batch measurement exists | measured: ~70-96x at large batch (v2 on H100 vs v1/HF on A100), ~13x single stream; `rl/results/gsm_grpo/H100/clean/`, `rl/gsm_grpo/RESULTS_ledger.md` §1 |
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
| RL figure "RL with and without the calculator" | `rl/interp/make_fig*.py` read the generated dev traces (not released) | the plotted numbers: `rl/interp/stats.json`, `stats_aR.json`, `rl/interp/NOTEBOOK.md` |
| Wiring arm W3z (links only, exactly zero-init gates) | never run | W3e / W3r (embedding / random init) |
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
