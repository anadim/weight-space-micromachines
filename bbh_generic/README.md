# BBH redo: generic tools under a held-out protocol

The nine BBH implants in the top-level README were debugged on BBH test questions, and three of them (`logic_nat`,
`track_nat`, `time_n`) carry BBH vocabulary. This folder is the redo: three agent teams rebuilt the 12 algorithmic BBH
tasks as **generic tools** (functions that are useful outside BBH), developed only on their own generated data, checked by
independent judge agents, and run on real BBH **once** at the end. The rules the agents worked under are in
[`PROTOCOL.md`](PROTOCOL.md) (rules 1-15). The short version:

* no BBH test data is opened during development (only the official 3-shot CoT prompts, which are part of the input);
* each team writes its own generators: dev = seed 0 (30 per task), fresh test = seed 1 (100 per task, extra paraphrases),
  train = seed 2, out-of-style sets = seed 3+ (a different setting than the benchmark, e.g. a meeting calendar instead of
  BBH's temporal puzzles);
* tools are compiled into weights and generic (no closed BBH vocabularies); every tool gets an independent judge who looks
  for test fitting and writes reworded variants the generator would not produce (`JUDGE_*.md`);
* the host may be fine-tuned to write tool calls (only on the team's own seed-2 data), with ~100 rows (rule 15); base,
  fine-tuned-without-tools and fine-tuned-with-tools are always reported side by side.

| folder | what |
|---|---|
| `teamA/` | `track` (state tracker) and `order` (ordering-constraint solver); tracking_shuffled_objects, logical_deduction |
| `teamB/` | `interval` (free intervals), `walk` (2-D position/heading), `truth` (truth propagation); temporal_sequences, navigate, web_of_lies |
| `teamC/` | `expr` (integer expressions), `bool` (boolean formulas), `calc` (decimal a op b); multistep_arithmetic_two, boolean_expressions, object_counting |
| `peek/` | the single real-BBH run (scoring scripts and aggregate scores only) |
| `ft100/` | sample-size control (fine-tune on 32 / 100 / 300 rows) - **results pending** |
| `integrate/` | one host, all eight tools, composed questions - **results pending** |
| `PROTOCOL.md` | the binding protocol given to the agents |

## The tools

All tools share one envelope: the host writes `<tool_call>NAME ARGS</tool_call>`, the tool writes
`<tool_response>RESULT</tool_response>` (empty RESULT = malformed call or undetermined answer; a call with another NAME
leaves the tool silent). Team C's tools also have a passive form that fires when the frozen host itself writes `... =`.
The grammars are in the docstrings of `teamA/callfmt.py`, `teamB/tools_ref.py` and `teamC/callfmt_C.py`.

| tool | f(x), stated without the benchmark | size (stack 1) | judge |
|---|---|---|---|
| `track_gen` | apply swap / set / copy operations to up to 9 one-digit slots; query a slot, the slot holding a value, or the state | 27.0k ReLU units, 2.14M nnz (v1) | PASS with caveats (`teamA/JUDGE_track_gen_1.md`) |
| `order_gen` | orderings of <= 7 objects under before/after/immediately-before/adjacent/at/not-at constraints; answer only if every consistent ordering agrees | 38.7k units, 3.48M nnz (v1) | PASS with conditions (`teamA/JUDGE_order_gen_1.md`) |
| `interval_gen` | free sub-intervals of a window minus busy intervals (am/pm hours or integers 0..99) | 81,886 units, 1.14M nnz | PASS (`teamB/JUDGE_interval_1.md`) |
| `turtle_gen` (`walk`) | dead reckoning from (0, 0) facing north: turns, forward/backward/sideways moves -> position and heading | 14,561 units, 1.61M nnz | PASS with conditions (`teamB/JUDGE_walk_1.md`) |
| `truth_gen` | knights-and-knaves propagation over "K true/false" and "K says J true/false" clauses in any order | 41,559 units, 4.52M nnz | PASS with disclosures (`teamB/JUDGE_truth_1.md`) |
| `expr_gen` / `expr_call` | exact value of an integer expression (+ - * x, unary minus, brackets nested <= 4, values <= 1e12) | 21.7k units, float64 | PASS (`teamC/JUDGE_expr_gen_1.md`) |
| `bool_gen` / `bool_call` | truth value of a True/False formula with not/and/or/brackets | 16.2k units | PASS (`teamC/JUDGE_bool_gen_1.md`) |
| `calc_gen` / `calc_call` | exact a op b (and two-op chains) on signed decimals, + - * / | the repo's `calc_s4` + `calc_neg.py` | PASS (`teamC/JUDGE_calc_gen_{1,2}.md`) |

Sizes are from the team notebooks (`teamA/NOTEBOOK.md` "Tool build log", `teamB/NOTEBOOK.md` section 3, `teamC/NOTEBOOK.md`
L3/L5).

## How they attach (in weights) and what to disclose

Every tool is a compiled integer-weight ReLU circuit built with the repo's `micromachines/hdl.py` and attached with
`micromachines.machine.attach` (exact linear reader on the embedding matrix, a hard-attention hop to the end of the call,
solved write directions through the frozen norm + LM head). At inference the tools are fixed tensors; the only Python in
the generation loop is the decode loop and the hook. The Python references (`callfmt.py`, `tools_ref.py`, `callfmt_C.py`)
are used only to write training completions and to test the circuits.

Disclosures:

* **Teams A and B fine-tune the host** (full fine-tune of Qwen3-0.6B-Base) so that it writes the calls. The tool's reply
  span is in the training loss, so the fine-tuned host without tools attached is a fair control (it must write the replies
  itself). The four envelope tokens share one untrained embedding row in the base checkpoint; both fine-tune scripts give
  them distinct random rows first (seed 1234). **The fine-tuned checkpoints are not released**; the exact rows, scripts and
  hyperparameters to recreate them are below.
* **Team C uses a frozen host** (no fine-tune). `calc_gen` uses the solved "no more digits" veto vector, fitted on GSM8K
  **training** solutions (`scripts/solve_veto.py`). Team C's dev generator mimics BBH's surface form (written from the public
  task description). The end-of-result writes ('.\n' after `expr_gen`, '.' after `bool_gen`) were chosen on DEV failures and
  are similar to fixes the old implants had found on BBH failures (disclosed in `teamC/NOTEBOOK.md` L2/L4 and
  `teamC/JUDGE_bool_gen_1.md`). Known misfire: `The sum of 5 and 3 * 4 =` gets 12.
* **Team A**: the v1 ordering generator's age and price scales were influenced by remembering BBH; v2 replaced them by 12
  generic scales, and an ablation without the age/price scales is reported (it does not lower the scores). Team A read the
  failures of its first out-of-style set (oos1) and of oos3, so those two are not held out; oos2 is.
* **Team B**: ft3's error-recovery rows, extra hiker verbs and the time-range scoring normalization were motivated by reading
  OOS-dev (seed 4) failures; the seed-3 OOS set was only scored.
* `expr_gen` runs its stack in float64 (`compile_net64` in `teamC/teamc_tools.py`), not float32 like the other circuits.

## Results

All numbers below are copied from the team notebooks, the judge files and `peek/scores.txt`; the compact per-question
files are in `teamA/results/`, `teamB/results/`, `teamC/results/` (gold, prediction, correctness, the host's first call and
the tool's reply; no generated text; each file keeps the SHA-256 of the original output and a reconstructed command).
Greedy decoding, official BBH 3-shot CoT prompts unchanged, Qwen3-0.6B-Base as the (base of the) host.

### Real BBH, run once (n = 250 per task; `peek/`)

| | base | host, no tools | host + tools |
|---|---:|---:|---:|
| Team A, 6 tasks (host ft_v3: 14,500 rows x 2 epochs) | 30.0 | 61.4 | **100.0** |
| Team B, 3 tasks (host ft3: 12,000 rows, 1 epoch) | 48.1 | 95.7 | **99.7** |
| Team C, 3 tasks (frozen host, no fine-tune) | 64.7 | (= base) | **92.4** |
| **12 tasks** | **43.2** | **70.8** | **98.0** |

Per task (`peek/scores.txt`): tracking 3/5/7 base 40.8 / 22.0 / 14.8, ft_v3 no tools 70.4 / 66.4 / 58.8, + tools
100 / 100 / 100; deduction 3/5/7 base 57.2 / 28.4 / 16.8, no tools 64.0 / 51.2 / 57.6, + tools 100 / 100 / 100; temporal
24.8 / 100.0 / 100.0; navigate 61.6 / 87.2 / 99.2; web of lies 58.0 / 100.0 / 100.0; multistep arithmetic 45.2 -> 99.2;
boolean expressions 82.8 -> 98.8; object counting 66.0 -> 79.2. For comparison, the retired BBH-specific implants scored
90.1 on the same 12 tasks.

**Read these with the conditions below.** The Team A and Team B hosts behind this table are the large fine-tunes
(12k-14.5k rows), made before rule 15; on the BBH-style tasks they learn much of the task itself (the no-tools column).
Team A's final ~100-row host (ft100a) was not run on real BBH; the sample-size study in `ft100/` runs 32 / 100 / 300-row
hosts on real BBH (pending).

Judges' binding conditions (from the `JUDGE_*.md` files):
* Team A (`order_gen`): report Team A's held-out judge round 3 next to any BBH number: host + tools **5/16**, fine-tuned
  host without tools 4/16, frozen base 3/16 (tool alone exact 16/16; every miss is the host's call). Describe the result as
  ~90% on BBH-like wording and much lower on other wording; fail-safety is the tool's only.
* Team B (`truth`): report web_of_lies with the fine-tuned-without-tools column (the gain there is the fine-tune's) and the
  judge's variants: truth 4/10 (3 through a correct call). `walk`: 6/12 scored, 5/12 through the tool. `interval`: 13/15.

### Team A, final host ft100a (102-row fine-tune), our generated sets (`teamA/NOTEBOOK.md` FINAL)

| set | frozen base | FT100 no tools | FT100 + tools |
|---|---:|---:|---:|
| fresh test (seed 1, n = 100 per task, mean of 6) | 28.5 | 20.8 | **85.2** |
| oos2, held-out out-of-style (n = 30 per task) | 22.8 | 26.1 | **60.0** |
| judge round 3, held out (16 variants) | 3/16 | 4/16 | 5/16 |

Per task, fresh test with tools: track 3/5/7 92 / 91 / 88, order 3/5/7 84 / 84 / 72 (age/price-ablated host:
92 / 97 / 97 / 84 / 90 / 81). The tool alone is exact on every gold call; the misses are the host's encoding (wrong query
type, reversed arguments, a dropped swap).

### Team B, final host ft3 (12,000-row fine-tune) (`teamB/NOTEBOOK.md` section 6)

| set | frozen base | FT no tools | FT + tools |
|---|---:|---:|---:|
| temporal, fresh test (n = 100) | 15.0 | 100.0 | 100.0 |
| navigate, fresh test | 62.0 | 88.0 | 89.0 |
| web_of_lies, fresh test | 55.0 | 100.0 | 100.0 |
| meeting calendar, held-out out-of-style (n = 50, `interval`) | 0.0 | 10.0 | **100.0** |
| delivery robot, held-out out-of-style (`walk`) | 0.0 | 8.0 | **68.0** |
| knights and knaves, held-out out-of-style (`truth`) | 34.0 | 62.0 | **94.0** |

On the BBH-style tasks the fine-tune carries the gain (tools add +0 / +1 / +0); the tools matter where the computation is
not a copyable pattern (+90 / +60 / +32 over the same fine-tuned host).

### Team C, frozen host (`teamC/NOTEBOOK.md` FINAL)

| task / setting | plain | + tools |
|---|---:|---:|
| multistep_arithmetic_two, fresh test (n = 100) | 59.0 | **98.0** |
| boolean_expressions, fresh test | 77.0 | **100.0** |
| object_counting, fresh test | 31.0 | 37.0 |
| inventory totals, out-of-style (n = 60) | 33.3 | **100.0** |
| code conditions, out-of-style | 80.0 | **98.3** |
| shopping word problems, out-of-style (strict / lenient extraction) | 73.3 / 90.0 | 75.0 / 95.0 |

object_counting stays low because the host mislabels item categories; no mechanical tool fixes that.

## Reproduce

From the repo root, once:

```bash
pip install -r requirements.txt
bash data/download_bbh.sh                                           # official prompts + task files (real-BBH run only)
python scripts/solve_veto.py --out bbh_generic/teamC/runs/veto_qwen3.pt   # Team C's calc veto (CPU ~50 min; --device cuda is faster)
```

All model runs need one GPU. Seeds are fixed in the generators (dev 0, test 1, train 2, out-of-style 3/4/5) and in the
fine-tune scripts (`--seed 0`).

**Team A** (`cd bbh_generic/teamA`)

```bash
TOOL_DEVICE=cpu python test_tools.py 300                 # tool alone vs the Python reference (random + malformed calls)
python tool_gold.py data/dev.jsonl data/oos.jsonl        # tool alone on the gold calls
python make_data.py dev test oos                          # regenerates data/{dev,test,oos}.jsonl byte-identically
python gen_oos2.py; python gen_oos3.py                    # data/oos2.jsonl (seed 4), data/oos3.jsonl (seed 5)
# the final host: 102 rows, full fine-tune (AdamW 0.9/0.95, wd 0.01, 5% warm-up + cosine, bf16 autocast)
python ft.py --data data/train100.jsonl --out runs/ft100a --lr 2e-5 --epochs 6 --tok_budget 8000 --seed 0
python ft.py --data data/train100_ablate.jsonl --out runs/ft100a_abl --lr 2e-5 --epochs 6 --tok_budget 8000 --seed 0
mkdir -p res logs && bash jobs_base.sh && bash jobs_final100.sh   # base, FT100 +/- tools on test/oos/oos2/oos3/judge variants
python tables.py                                          # the FINAL tables (python tables.py results: from the shipped files)
# the large host used in the real-BBH run (ft_v3):
xz -dk data/train3.jsonl.xz
python ft.py --data data/train3.jsonl --out runs/ft_v3 --epochs 2 --lr 1e-5 --tok_budget 16000 --seed 0
```

The training files are shipped because the train-split generators are not byte-reproducible: the train-only styles
(`alt`, `code`, `mix`) depend on Python's string-hash seed (`PYTHONHASHSEED`), while dev, test and the out-of-style sets do
not. Shipped: `data/train100.jsonl` and `data/train100_ablate.jsonl` (the final ~100-row hosts) and `data/train3.jsonl.xz`
(14,500 rows; ft_v3 and the source of the `ft100/` subsamples). Not shipped (30-47 MB each, intermediate hosts only;
`python make_data.py train train_ablate train3_ablate train4 train4_ablate` regenerates files of the same kind, not the
same bytes). SHA-256 of the originals:

| file | rows | sha256 | used for |
|---|---:|---|---|
| `data_v1/train.jsonl` | 11,000 | `28030778e855900f310788a9d70aa81b7a21729b33931e50d2eaecde63e065c0` | ft_v1 |
| `data/train.jsonl` | 11,750 | `58961b060e21a0e890727093af6aa5717a89e135ca80008d23649cb20dcb2bbf` | ft_v2 (stopped) |
| `data/train_ablate.jsonl` | 11,750 | `28f188436787e2615afbeec1759d2e5390fa7b2fab214d559d6ed141798cbc3b` | (not trained) |
| `data/train3.jsonl` (shipped as .xz) | 14,500 | `42d9f4f3ce8db88899f370565b1aa79293e614af4be5356cb0c3f8c8c10634fa` | ft_v3 |
| `data/train3_ablate.jsonl` | 14,500 | `cca6a9f083e0c5139586088e0b2ef8070bf2d8cb795b006249e03579af6ca4d8` | ft_v3abl |
| `data/train4.jsonl` | 16,500 | `2281b3ec29a44bd08bd272b745c2f16c71a577532c7773730cd97ab720e8e7fe` | ft_v4 (stopped, never evaluated) |
| `data/train4_ablate.jsonl` | 16,500 | `e1d2836de3b31932121e2f9f4732e270cab7f69bc1d18d4df35d0679abc4b8a7` | ft_v4abl (stopped) |

**Team B** (`cd bbh_generic/teamB`)

```bash
mkdir -p out
python gens.py                                            # writes data/ (dev, test, oos, oosdev, train); byte-identical to the
                                                          # files used (checked with two hash seeds; data/TRAIN_SHA256.txt)
python test_tools.py interval_gen data/temporal_dev.jsonl,data/lab_train.jsonl:200   # also turtle_gen, truth_gen
python ft_B.py --out runs/ft3                             # defaults = ft3: 3x3000 main + 3x1000 aux rows, 1 epoch, lr 1e-5,
                                                          # tok_budget 24000, p_zero 0.2, seed 0 (355 steps)
python run_eval.py out/plain_test.json --sets temporal_test,navigate_test,lies_test --tools none
python run_eval.py out/ft3_all_test.json --model runs/ft3 --sets temporal_test,navigate_test,lies_test --tools all
python run_eval.py out/ft3_none_test.json --model runs/ft3 --sets temporal_test,navigate_test,lies_test --tools none
python run_eval.py out/ft3_all_oos.json --model runs/ft3 --sets meeting_oos,robot_oos,knights_oos --tools all
python run_variants.py judge/variants_truth_1.json judge/results_truth_1.json --model runs/ft3   # judge variants
```

**Team C** (`cd bbh_generic/teamC`)

```bash
mkdir -p out
python gen_tasks.py data; python gen_ood.py data          # data/{dev,test,ood}: byte-identical to the shipped files
python test_tools.py expr 300 0; python test_tools.py bool 300 0; python test_calls.py 300 0
python run_gen.py out/test_plain_multistep_arithmetic_two.json --data data/test --tasks multistep_arithmetic_two --machines none
python run_gen.py out/test_v4_multistep_arithmetic_two.json --data data/test --tasks multistep_arithmetic_two \
       --machines expr_gen,calc_gen,bool_gen --oe dotnl     # frozen config; same for boolean_expressions, object_counting
python run_gen.py out/ood_v4_num_list.json --data data/ood --tasks num_list --machines expr_gen,calc_gen,bool_gen \
       --prompt_file data/ood/num_list.prompt.txt
python rescore.py out/ood_*.json                          # lenient extraction for the out-of-style sets
python make_train_C.py data/train/train_C.jsonl           # call-format rows for the integration step (byte-identical)
```

`data/train/train_C_100.jsonl` (the 100-row subsample used by `integrate/`) is shipped; the snippet that drew it (seed
string `teamC-train-2-sub100`, balanced over the 6 kinds) was not saved.

**Real BBH**: see `peek/README.md`.

## Files

* `teamX/NOTEBOOK.md`: every change and the dev failure that motivated it, in order; `teamX/JUDGE_*.md`: the judges'
  verdicts with their variants and results; `teamX/judge/`: the variant files and the runs on them.
* `teamX/cot/`: the official BBH 3-shot CoT prompts the team used (BIG-Bench Hard, MIT license, pinned commit
  `9ee07bd`); these are the same files `data/download_bbh.sh` fetches.
* `teamX/results/`: compact per-question records of the runs behind the notebook tables.
* `.gitignore` keeps checkpoints, run outputs and run-time copies of BBH test data out of the repo.
