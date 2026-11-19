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
| Climbing IFEval | `ifeval/heldout/`, `ifeval/full14_compiled/`, `ifeval/baselines/` | [5](#5-climbing-ifeval-14-rule-implants) |
| Climbing BIG-Bench Hard | `bbh_generic/` (`integrate/`, `callgate/`, `ft100/`, `nl_control/`) | [6](#6-climbing-big-bench-hard-8-tools-call-fine-tune-call-gate) |
| Add one implant at a time | `ifeval/transfer/`, `bbh_generic/transfer/` | [7](#7-add-one-implant-curves) |
| Do implants transfer across models? | `results/gsm_bigger/`, `ifeval/transfer/`, `bbh_generic/transfer/`, `transfer/xfamily/`, `ifeval/xfamily/` | [8](#8-transfer-to-bigger-qwen-models-and-other-families) |
| Mistake detector | `micromachines/rethink_circuit_v2.py`, `scripts/rethink_v2/`, `results/rethink_v2/` | [9](#9-mistake-detector--rethink) |
| Topic blocker | `topic_fence/`, `topic_fence/inweights/`, `topic_fence/baselines/` | [10](#10-topic-blocker) |
| Writing into intermediate representations | `latent_write/` (+ `v2/`, `causal/`) | [11](#11-latent-write) |
| RL with implants | `rl/` | [12](#12-rl-with-implants-gsm8k-grpo) |
| Figures | `figures/blog/` | [13](#13-figures) |


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
`scripts/run_bbh.py`, `results/bbh/`, `tests/`) is kept as the overfit example: README "Benchmark-derived knowledge
inside the implants" names the BBH vocabulary inside `track_nat`, `logic_nat`, `time_n`. The fixed loop (build on
look-alike questions, independent judges that paraphrase / swap names / reorder, "name a use outside the benchmark") is
`bbh_generic/PROTOCOL.md` (rules 10-11), with every judge report in `bbh_generic/team*/JUDGE_*.md` and the look-alike
generators in `bbh_generic/team*/` (see the BBH section). Commands for the retired numbers: README "Quickstart"
(`scripts/run_bbh.py`, `results/summarize.py`).


## 5. Climbing IFEval: 14 rule implants

Blog claims backed here: 58% -> 73% prompt-strict on all 541 IFEval prompts (+82 / -3), silent prompts byte-identical
(323/323); per-rule chart; thinking mode 59%; readers catch ~40-75% of rephrasings (misses fail safe); all 14 rules compiled
into plain weights (~9,300 ReLU units), byte-identical to the reference on 541/541; add-one-rule curve; transfer to 1.7B / 4B
(+9 / +4 points); SmolLM2 port.

**Environment.** Python 3.12.3, torch 2.8.0+cu128, transformers 4.56.2, lm_eval 0.4.13 (IFEval checkers), langdetect 1.0.9.
No vLLM. Host `Qwen/Qwen3-0.6B` (chat model, thinking off), bf16, greedy, max_new 2048. GPUs: the headline run on one NVIDIA
A40 46 GB; the compiled implant, transfer and curve on one A100 80GB PCIe. Peak memory < 10 GB (0.6B), ~20 GB (4B).

**Data.**
```bash
python ifeval/data/prepare_ifeval.py          # google/IFEval -> ifeval/data/ifeval.jsonl (541 prompts)
python ifeval/data/prep_transfer.py           # optional: IFBench (needs allenai/IFBench at commit 488918665d53)
```
Our own dev / fresh / ood / mentions prompts are in `ifeval/heldout/data/` (generated by `gen_data.py`, seeds 0/1/2).

### 5.1 Headline: 14 rules, frozen Qwen3-0.6B (A40)

```bash
cd ifeval/heldout && ln -s ../../data/ifeval.jsonl data/ifeval.jsonl
J=(); for i in 0 1 2 3; do J+=("ifeval.jsonl ifeval_plain.s$i.jsonl plain $i/4" "ifeval.jsonl ifeval_imp.s$i.jsonl implant $i/4"); done
./queue.sh 4 "${J[@]}"                                                     # 8 shard jobs, bs 24, one GPU
./final_score.sh                                                            # ODD / EVEN / ALL + IFBench
```
Expected (`ifeval/heldout_results/final_score.out`): ALL 541 58.23 -> **72.83** (+82 / -3); ODD 299 56.19 -> 71.91 (+49 / -2),
inst-strict 67.11 -> 79.91; EVEN 60.74 -> 73.97; silent 323/323 byte-identical; IFBench 295/295 silent identical.
Per-rule (ODD): no_comma 18 -> 35/36, forbidden 8 -> 22/27, paragraphs 7 -> 12/16, repeat 15 -> 18/21, two 6 -> 9/11 (blog
per-rule chart). Text-free per-prompt records: `ifeval/heldout_results/teamd_ifeval_{plain,imp}.json`.
Runtime: ~1.5 h per 541-prompt implant run per GPU process (4 shards in parallel ~25-40 min); plain runs faster.
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
Expected: thinking ON, answer after `</think>`: ALL 59.33 / ODD 59.20 (vs 58.23 plain, 72.83 implants); 50/541 hit the cap.
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
Commands: README "Reproduce" block, lines starting `python scripts/rethink_v2/...` (vLLM generation of ~52k labelled
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
```

**Expected numbers** (ledger `rl/gsm_grpo/RESULTS_ledger.md`; files in `rl/results/`):
| claim | value | file |
|---|---|---|
| plain RL, dev-800, u0 -> u64 | 43.6 -> 54.5 (s0) / 53.9 (s1) | `rl/results/gsm_grpo/A100/abl/a1_*/steps.jsonl` |
| RL + calculator, dev | 48.9 -> 65.2 / 63.8 | `rl/results/gsm_grpo/A100/abl/a2_*/steps.jsonl` |
| broken calculator (arm R), dev | 7.5 -> 36.6 / 37.9 (H100: 7.0 -> 36.9 / 36.4) | `rl/results/gsm_grpo/A100/abl/aR_*`, `rl/results/gsm_grpo/H100/abl/aR_*` |
| arm 2 host, calculator off, dev u64 | 56.2 / 55.5 vs arm 1 54.5 / 53.9 | same steps.jsonl (`host` entries) |
| unit-word equations | 29.8% base -> 23.5% plain RL -> 16.7% RL+calc | `rl/interp/stats.json`, NOTEBOOK Q2 |
| host copies the calculator result | 99.0 / 99.2 / 99.4% | `rl/interp/stats.json` |
| arm R trigger fire share of equations | 74.6% -> 12.1% (s1 11.2%) | `rl/interp/stats_aR.json`, NOTEBOOK Q4 |
| test u64 GSM8K / GSM-Hard, arm 1 vs arm 2 (on) | 65.0, 65.1 / 36.7, 37.9 vs 67.2, 66.6 / 47.6, 47.4 | `rl/results/gsm_grpo/A100/final/*.json`, `rl/results/gsm_grpo/summary/paired.txt` |
| test u300 (H100) GSM-Hard arm 2 - arm 1 | +12.24 pp [+10.12, +14.29]; GSM8K +0.57 [-1.06, +2.16] | `rl/results/gsm_grpo/H100/final/H_*.json`, `compile_long.py` |
| engine throughput (H100, 512 rows) | 19,800 tok/s with implant, 38,800 without; v1 HF loop 207 tok/s, HF .generate 288 tok/s (A100, 16 rows) | `rl/results/gsm_grpo/H100/clean/rollout_*.json`, `rl/results/gsm_grpo/A100/prof_v1_a100.json` |

**Wiring arms** (not in the blog text; the question "let RL figure out the wiring"): `rl/results/wiring/{H100,A40}/results.md`,
dev-800 at u300, mostly seed 0: fixed wiring (arm 2) 70.0; W2 (fixed write + zero-init links) 72.0, links stay ~0; W2w
(trainable warm-started write) 70.1; W3e / W3r (links only, no write) 63.9 / 61.1, i.e. no discovery from random wiring;
CWlean 66.2; CWinR (broken calculator + trainable in-wire) vetoes 100% of calls from u25. Details in `rl/README.md`.

**Runtime.** 64-update run ~16 min (A100); 300-update run ~35 min (H100, arm 2), ~53 min (arm 4); final eval minutes.
Whole blog subset (6 x 64-update runs + finals + traces) ~3 A100-hours; 300-update table ~5 H100-hours.

**Seed noise.** 2 seeds per main arm: dev differs by <= 1.4 pp (64 updates), test by <= 0.6 pp (arms 1/2, 64 updates) and
up to 3.5 pp (arm 4, 300 updates). Dev SE ~1.7 pp (n=800). Bootstrap CIs are over questions only (seed variance not
included). bf16 numbers differ by GPU and `B_max` by ~1 pp: compare within a GPU. Interp traces use one seed per arm.
Wiring arms, 4c, 4s: mostly one seed.

**Not reproducible from the repo alone / caveats.**
* No checkpoints are released; every number after u0 needs the training runs (~1 GPU-hour each).
* "about 25x faster than a HuggingFace generation loop on the same GPU" is not backed by a measurement: the measured ratios
  are ~70-96x at large batch (different GPUs: v1/HF on A100, v2 on H100) and ~13x single-stream (v2 H100 319 tok/s vs v1
  A100 24 tok/s). The v1 engine that `prof_v1.py` profiles is not released.
* Interp traces (generated text) are not released; regenerate with `rl/interp/run_*.sh` from your own checkpoints.
* `L300_CWinO_s0` (wiring) was still running at packaging time.


## 13. Figures

`bash figures/blog/make_blog_figs.sh` (CPU) rebuilds the static figures into `figures/blog/figs/`;
`figures/blog/README.md` maps each blog figure title to its script and data. The animations need `rsvg-convert`,
`ffmpeg`, `gifsicle`. RL figures read `rl/results/curves/<run>/steps.jsonl` (env `RL_CURVES`).

## 14. Summary: what cannot be reproduced from the repository alone

| item | why | what is shipped instead |
|---|---|---|
| Fine-tuned hosts (BBH v2 0.6B/1.7B/4B, ft100, NL control, Team A/B, finetuned_phase) and all RL checkpoints | weights not released | training rows or seeded generators (byte-identical, SHA-256), scripts, hyperparameters; rebuilding gives bf16-noise-level differences |
| Compiled IFEval implant tensor `implant14_sd.pt` (1.23 GB) | > 50 MB | `ifeval/full14_compiled/export.py` rebuilds it in ~20 s; sha256 `e85650adb761df2a292c5d1ce4aa836ddb9b2a32f3d2dbfc77b37a849787c200` |
| RL claim "about 25x faster than a HuggingFace generation loop on the same GPU" | no same-GPU measurement exists | measured: ~70-96x at large batch (v2 on H100 vs v1/HF on A100), ~13x single stream; `rl/results/gsm_grpo/H100/clean/`, `rl/gsm_grpo/RESULTS_ledger.md` §1 |
| 3-way GSM fine-tune per-seed files, eyes seed 0 raw files | on a machine that was offline at packaging | numbers in `finetuned_phase/results/reported_numbers.json` (`verified_here: false`) |
| Eyes single-seed 8-example 88% (earlier draft) | recipe not recorded; did not reproduce | 4-seed sweep (78.5 at n=8, 94.7 at n=32) is what the blog uses |
| Detector "28 fixed: 16 reasoning / 5 arithmetic / 7 unclassified" | manual reading of generated text (not released) | the 28 problem ids are recoverable from `results/rethink_v2/` per-question records |
| BBH per-tool composition analysis, NL truncated-budget re-score, RL interp traces, animation score jobs | need generated text / token ids (not released) | their outputs: `bbh_generic/integrate/TABLES_v1_vs_v2.md`, `bbh_generic/nl_control/results/scores_nl.json`, `rl/interp/stats*.json`, `figures/blog/figs/anim/*_trace.json` |
| IFEval add-one-rule order | chosen from Team D FRESH-set runs not in the repo | order hard-coded in `ifeval/transfer/curve2.py` |
| Topic-blocker dev set regeneration row for row | written by an LLM (Qwen2.5-7B, sampled) | dev/test data and fitted probes shipped (`topic_fence/data/`, `topic_fence/fitted/`) |
| 0.6B "frozen + Team C" 27-task BBH row | older run; per-question rows for 22 of 27 tasks | aggregate in `bbh_generic/ft100/scores_ft100.json` |
| Wiring arm `L300_CWinO_s0` | still running at packaging | — |
| Avenir fonts for figures | license | matplotlib default font fallback |

Across GPU types, bf16 greedy text differs on a few percent of items (e.g. IFEval 0.6B: 58.23 -> 72.83 on an A40, 57.86 ->
72.09 on an A100; GSM8K ~4-6% of problems flip between stacks). Compare within one GPU type, and expect +-1 point.
