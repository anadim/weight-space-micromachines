# Weight-Space Micromachines: Brain Implants for LLMs

A micromachine (an "implant") is a small network whose weights are written by a coding agent, not learned by gradient
descent. It sits next to a **frozen** language model, reads the model's tokens, and, when its moment comes, adds a write
to the model's last hidden state that makes the model emit the token the implant wants (an exact digit, an answer letter).
At every other position its write is exactly zero, so the model's logits there are bit-identical to the unmodified model.
This repository rebuilds nine such implants from code for the public `Qwen/Qwen3-0.6B-Base` checkpoint (downloaded from
Hugging Face on first use; no custom checkpoint is needed), and contains the harnesses, the exactness tests, and the small
result files behind the numbers below.

## Results

Every number in this section is printed by `python results/summarize.py`, which reads only the files in `results/`.
`VERIFY.md` lists what was re-run from a fresh clone of this repo and how it compared.

### BIG-Bench Hard, 12 algorithmic tasks (official 3-shot CoT prompts, greedy, n = 250 per task)

| task | Qwen3-0.6B-Base (vLLM) | Qwen3-0.6B-Base (HF loop) | **+ implants** (HF loop) | Qwen2.5-7B (vLLM) | Qwen2.5-14B (vLLM) | implant(s) |
|---|---:|---:|---:|---:|---:|---|
| multistep_arithmetic_two | 46.0 | 47.2 | **97.6** | 90.0 | 95.2 | `calc_s4`, `sum_fn`, `expr_n` |
| navigate | 62.4 | 63.2 | **99.6** | 97.6 | 98.8 | `navnat` |
| boolean_expressions | 82.4 | 82.8 | **98.4** | 91.6 | 94.8 | `boolnat` |
| web_of_lies | 56.8 | 60.4 | **97.6** | 99.2 | 100.0 | `liesnat` |
| temporal_sequences | 27.2 | 26.0 | **98.4** | 71.2 | 86.8 | `time_n` |
| tracking_shuffled_objects (3) | 39.2 | 42.4 | **100.0** | 90.0 | 82.4 | `track_nat` |
| tracking_shuffled_objects (5) | 20.0 | 21.6 | **100.0** | 66.8 | 78.0 | `track_nat` |
| tracking_shuffled_objects (7) | 12.0 | 11.6 | **78.8** | 46.0 | 59.2 | `track_nat` |
| logical_deduction (3) | 56.4 | 56.0 | **100.0** | 80.0 | 95.6 | `logic_nat` |
| logical_deduction (5) | 29.6 | 28.8 | **81.6** | 54.0 | 62.0 | `logic_nat` |
| logical_deduction (7) | 14.0 | 11.2 | **49.6** | 38.4 | 42.4 | `logic_nat` |
| object_counting | 66.0 | 65.2 | **79.2** | 87.2 | 91.2 | `sum_fn` |
| **mean** | **42.67** | **43.03** | **90.07** | **76.00** | **82.20** | |

Two base columns, because the baselines ran on two decoding stacks: vLLM (like the 7B/14B columns) and our HF loop
(`scripts/run_bbh.py`, the same loop as the implant runs, implants off). Both are greedy with the same prompts, stop rule,
answer extraction and a 4-token minimum before end-of-text; they agree to within a few points per task.

Provenance:
* vLLM columns: `results/bbh/qwen3_base_vllm.json`, `qwen25_7b_vllm.json`, `qwen25_14b_vllm.json`. The first 7B run gave
  empty outputs on 5 tasks (boolean, multistep, object counting, Dyck, word sorting); for those 5 only,
  `qwen25_7b_vllm_mintok4.json` (a rerun with the 4-token minimum) is used.
* HF-loop base: `results/bbh/qwen3_base_hf.json` (all 27 tasks; `qwen3_base_hf_deduction.json` is an earlier run of
  the 3 deduction tasks with identical results).
* Implant column: one file per task group, `results/bbh/qwen3_implants_{arith,navigate,lies_bool_navv1,temporal_dyck,tracking,deduction}.json`.
  **Each of those runs attached only that task group's implants**, not all nine.

How much is known about attaching all nine at once:
* full size (n = 250), from a separate run whose log is snapshotted in `results/bbh/all_nine_mech.log` (PARTIAL): the
  same scores as the per-group runs on the 6 tasks it had finished (multistep, object counting, web of lies, navigate,
  boolean, temporal). Tracking 3/5/7 and deduction 3/5/7 have **not** been run at n = 250 with all nine attached;
* subset (first 48 questions of 8 implant tasks, including tracking-5 and deduction-3; `VERIFY.md`): generated text
  identical to the per-group runs, 48/48 on every task;
* tasks without an implant: with all nine attached, the scores equal the no-implant HF run on the 4 tasks finished in
  `results/bbh/all_nine_other.log` (PARTIAL), and causal_judgement's first 48 outputs are identical (`VERIFY.md`).

All 27 tasks (no implant on the other 15): with the vLLM base, base 39.42 and with implants 60.49; with the HF-loop base,
40.37 and 61.27. Qwen2.5-7B 67.33, Qwen2.5-14B 74.45.

### GSM8K / GSM-Hard: calculator implant and a "rethink" nudge (full test sets, 1,319 each, greedy, 8-shot)

| Qwen3-0.6B-Base, frozen | as is | + calculator | + rethink | + both |
|---|---:|---:|---:|---:|
| GSM8K | 51.93 | 54.06 | 56.10 | **57.62** |
| GSM-Hard | 26.99 | 34.50 | 29.26 | **37.23** |

Not fully held out: the 8 prompt examples were picked from GSM8K train during a prompt search on the earlier
Qwen2.5-0.5B setup that was scored on a 200-problem train holdout **and on the first 200 GSM-Hard test problems**
(`data/prepare_gsm.py`). Read the GSM-Hard numbers with that in mind. Failures on the GSM8K test set were also read while debugging the calculator.

Provenance: `results/gsm/qwen3_{test,hard}_{plain,calc}_rethink.json` (`first_ok` = before the rethink pass, `ok` = after).
The calculator is `calc_s3` (writes the exact digits after the model's own `a op b =`), an earlier variant than the BBH
calculator `calc_s4`: it has no "dot veto", so when the model writes `.` right after the implant's number, `calc_s3`
reads it as a decimal point and writes a fraction digit (`50` -> `50.0`); `tests/test_calc_tokens.py` with `MM=calc_s3`
shows this in 181 of 200 cases. In the saved GSM runs, equations whose result is the exact value plus extra fraction
digits are rare and about as common without the implant (GSM8K: 5 problems with the calculator, 9 without; GSM-Hard: 4
and 5; `results/verify/gsm_dot_tails.txt`), so the effect on the reported accuracies is at most a few problems out of
1,319. `calc_s4` fixes it. "Rethink" means the **evaluation
code** appends `Hmm, wait. Let me rethink this.` after the first answer and lets the model continue; it is not written by
an implant.

### Reading the model: a probe decides when to rethink (GSM8K test, generations in vLLM)

A logistic probe on layer-28 residual states at the model's `####` token, fitted on 1,500 GSM8K training problems, flags
wrong answers with test AUROC 0.81 (0.83 on solutions written with an oracle calculator). Rethinking only when it fires:

| policy | accuracy | extra tokens / problem |
|---|---:|---:|
| never rethink | 52.08 | 0 |
| probe fires (on 42% of problems) | 55.27 | 45 |
| always rethink | 56.56 | 101 |
| perfect detector (rethink exactly the wrong ones) | 58.30 | 49 |

Provenance: `results/gate/cost_curve.log` (output of `scripts/gate/cost_curve.py` on the saved generations and
detector scores) and `results/gate/detector_summary.json`. The per-problem generations (8 MB) and detector scores are
not included, so `cost_curve.py` cannot be re-run from a fresh clone without first re-running `gen_rethink.py` (vLLM) and
`fit_detector.py`. The gate is applied by evaluation code; the probe is
not packed into an implant yet. This is a vLLM run, so its 52.08 baseline differs slightly from the 51.93 of the HF loop.

## What is written, what is solved, what is trained

* **Written by the agent (no gradient):** every implant's logic: integer-weight ReLU circuits (`micromachines/circuits/`,
  built with the small compiler in `micromachines/hdl.py`). All wires carry small integers, so they are exact in float32.
* **Solved against the frozen model (small fits; no Qwen weight changes):**
  - the *reader* of each implant: a linear map from Qwen's embedding matrix to the implant's 0/1 input flags (closed form,
    with a max-margin Adam fallback for flags that stand for several tokens). It is checked to be exact on every row of
    the embedding matrix (all 151,936) each time an implant is attached;
  - the *write directions*: one residual vector per output token, fitted with Adam (3,000 steps) through Qwen's own final
    norm and LM head so that adding it makes Qwen emit that token;
  - the *veto* vector the calculator uses to end a number (no digit may win; the model's own non-digit choice is kept):
    Adam on hidden states at number endings in 300 GSM8K **training** solutions (`scripts/solve_veto.py`).
* **Fitted on data, used by evaluation code (not an implant):** the layer-28 probe of the rethink gate
  (`scripts/gate/fit_detector.py`, scikit-learn logistic regression on GSM8K training problems).
* **Not trained:** Qwen3-0.6B-Base. No fine-tuning, no new tokens; its weights are bit-identical to the download
  (`scripts/check_silent.py` hashes them before and after attaching).

### Benchmark-derived knowledge inside the implants (read this before citing the BBH numbers)

Three implants are task solvers: they read the BBH question and compute the whole answer when the model writes
`So the answer is (`. They carry information taken from the BBH task files or tied to their exact format:

* **`logic_nat` (logical_deduction)** has a closed vocabulary. Its **50 object names are extracted from the BBH
  logical_deduction task files** (the objects listed in each question's first sentence), and its 64 predicate phrases
  ("is to the left of the", "is the third from the right", "are the cheapest", ...) were written to match BBH's wording
  exactly (`micromachines/circuits/deduction_vocab.py`, generated by `scripts/gen_deduction_vocab.py`). It also assumes
  the BBH convention that the options list the objects in order of first mention. It would not recognize a new object.
* **`track_nat` (tracking_shuffled_objects)** knows the seven BBH names (Alice, Bob, Claire, Dave, Eve, Fred, Gertrude)
  and relies on two BBH conventions: the options list the items in the order of their initial holders, and there are
  as many swaps as people.
* **`time_n` (temporal_sequences)** reads BBH's phrasing ("woke up at", "from X to Y", "closed after", options
  "(A) Xam to Ypm").

The arithmetic, lies, navigate and boolean implants instead act on intermediate steps the model writes itself.

## The nine BBH implants

| implant | what it does | writes when the model has written ... |
|---|---|---|
| `calc_s4` | exact `+ - * /` on signed decimals | `... 12 * 7 =` and a space (the digits, then a "no more digits" veto) |
| `sum_fn` | signed integer sums with brackets | `(-1 + 2 + 45) =` (the value, then `.` and a newline) |
| `expr_n` | <= 4 signed numbers, `+ - *`, precedence | `(3 * -2 + 4) =` (the value, then `.` and a newline) |
| `liesnat` | web of lies: truth along the chain | `... if X says Y lies, then X` (` tells` / ` lies`); `So the answer is` (` Yes` / ` No`) |
| `navnat` | navigate: position and facing after each step | each step's `:` (the new state), `Since` (the final position), the final ` Yes` / ` No` |
| `boolnat` | value of a boolean expression | `A = not not True =` (` True` / ` False`) |
| `time_n` | temporal sequences: the free interval | `So the answer is (` (the letter; computed from the question) |
| `track_nat` | tracking: who holds what after the swaps | `So the answer is (` (the letter; computed from the question) |
| `logic_nat` | deduction: the one ordering consistent with all clues | `So the answer is (` (the letter; computed from the question) |

Sizes are in `results/implant_sizes.json` (`python scripts/build_implants.py`): the nine together have 345,465 ReLU units
and 81.1M nonzero weights; `logic_nat` alone, which enumerates every ordering of up to 7 objects, has 203,503 units and
75.6M nonzeros.

## Quickstart

```bash
pip install -r requirements.txt          # Python 3.12, torch 2.8 (the CPU build is fine for the demo and tests)
python demo.py                           # CPU: frozen Qwen3-0.6B-Base + calculator; prints the generation with
                                         # implant-written tokens in [brackets] and asserts bit-identical logits
```

Reproduce the BBH and GSM numbers (one GPU; ~40 GB of CPU RAM to compile `logic_nat`):

```bash
bash data/download_bbh.sh                # BIG-Bench Hard, pinned commit, MIT license
python data/prepare_gsm.py               # GSM8K + GSM-Hard from Hugging Face, MIT license
python scripts/solve_veto.py             # -> runs/veto_qwen3.pt (CPU)

python scripts/run_bbh.py out/bbh.json --tasks navigate,web_of_lies      # all nine implants attached
python scripts/run_bbh.py out/base.json --tasks navigate --machines none  # same harness, no implant
python scripts/compare_runs.py out/bbh.json                               # question by question vs results/
python scripts/run_gsm.py out/gsm_test.json --split test --mode calc --rethink
python scripts/check_silent.py           # logits bit-identical wherever no implant writes

# exactness tests (Qwen tokenizer + embeddings; CPU or GPU)
python tests/test_calc_tokens.py 200 0
python tests/test_sum_expr.py 300 0
python tests/test_lies_bool_nav.py lies 300 0      # also: bool, nav
python tests/test_temporal.py
python tests/test_tracking.py 200 0
python tests/test_deduction.py 300 0
python tests/test_calc_circuit.py cpu              # circuit only, no model

python results/summarize.py              # the tables above, from results/
pip install -r requirements-extra.txt && python figures/make_figs.py
```

Baselines in vLLM: `scripts/baselines/run_bbh_vllm.py`. Rethink gate: `scripts/gate/gen_rethink.py` (vLLM) ->
`fit_detector.py` -> `cost_curve.py`.

**Time and hardware (measured on one NVIDIA A40 46 GB and a 16-core container, see `VERIFY.md`):**
* attaching implants builds them from code at every start (there is no cache): compiling `logic_nat` takes about 11 min
  on CPU (658 s; ~40 GB RAM), the other eight under 25 s each; with all nine, `attach` took 173-521 s depending on CPU load;
* `solve_veto.py`: about 50 min on CPU (once);
* `demo.py`: about 11 min on CPU (most of it solving the write directions);
* BBH with all nine attached: 3.5 min per 48 questions on the A40 in our check; the full-size runs (250 questions per
  task, GPU shared with other jobs) took 27-65 min per task (`results/bbh/all_nine_mech.log`);
* GSM8K, 64 problems with calculator and rethink: 94 s. The saved full runs (1,319 problems) took 1-2.5 hours each
  with several jobs on one GPU (timings in the original logs, not included).

## Caveats

* **How far can you push it.** While building the implants, the agent read the model's BBH failures (on the test
  questions; there is no separate dev split for these prompts) and fixed what it saw. For example, the navigate implant
  writes the position at `Since (` because the model kept copying `(0, 0)` from the prompt. The BBH numbers show how far
  implants can push a small model on these tasks, not held-out generalization.
* **Benchmark-derived knowledge:** see the section above (`logic_nat`, `track_nat`, `time_n`).
* **Task selection.** The 12 tasks were picked because they have a mechanical core. Nothing was built for the other 15.
* **Remaining errors are not the implants'.** Seven-object deduction (49.6) and tracking (78.8) misses are mostly the
  400-token budget: the model has not reached `So the answer is` when generation stops (no answer extracted on 123/250
  deduction-7 and 52/250 tracking-7 questions). The cap also lowers the 7B/14B baselines on these tasks: no answer
  extracted on 11/250 and 46/250 (7B), 73/250 and 46/250 (14B), mostly long outputs. Object counting stays at 79.2; from
  reading failures, the model mislabels items (a peach as an animal), which no implant addresses.
* **The rethink nudge and the probe gate live in evaluation code**, not in implant weights.
* **Baselines are Qwen2.5-7B/14B**, not Qwen3 models of those sizes, and ran in vLLM.
* **Numerics.** GPU runs use bf16 with left-padded batches, so exact outputs depend on batch composition; reruns match
  token for token with the same batch size and a `--limit` that is a multiple of it. On CPU, Intel MKL is not bitwise
  reproducible across calls unless `MKL_CBWR=COMPATIBLE` is set (`demo.py` sets it).

## Earlier phase (fine-tuned; separate, not re-run)

Before the zero-training work, the project fine-tuned Qwen2.5-0.5B to call implants through special tokens, built an
"eyes" implant with a **trained** MNIST CNN inside, and ran the first frozen-model calculator test
(Qwen2.5-0.5B, bare multiplication 4% -> 100%). That code, its result files and its numbers are in `finetuned_phase/`,
kept apart from the main code. Nothing there was re-run for this repository, and some raw files are on a machine that
was offline while this repo was assembled (marked as such in `finetuned_phase/README.md`).

## Repository layout

```
micromachines/          library: hdl.py (ReLU circuit compiler), machine.py (reader, write directions, attach),
                        specs.py (the implants), circuits/ (one hand-written circuit per implant), calcfmt.py, boolfmt.py
demo.py                 one prompt, one implant, bit-identity asserts
scripts/                run_bbh.py, run_gsm.py, solve_veto.py, build_implants.py, check_silent.py, compare_runs.py,
                        gen_deduction_vocab.py, baselines/run_bbh_vllm.py, gate/{gen_rethink,fit_detector,cost_curve}.py
tests/                  exactness tests (circuit-only and through the Qwen tokenizer/embeddings)
data/                   download scripts (no data is vendored)
results/                result files behind every number + summarize.py (see results/README.md)
figures/                make_figs.py (data figures, from results/), diagram and animation scripts
finetuned_phase/        the earlier fine-tuned phase (snapshot)
VERIFY.md               what was re-run from a fresh clone, where, and the outcomes
```

Figures: `python figures/make_figs.py` writes `figures/*.png`. The blog used Avenir Next; it is optional (put the .ttf
files in `figures/fonts/`), otherwise the figures fall back to DejaVu Sans.

## Related work

* Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos. *Looped Transformers as Programmable Computers.* ICML 2023.
* Weng et al. *Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural Networks* (Neural Comprehension). ICLR 2024.
* Lindner et al. *Tracr: Compiled Transformers as a Laboratory for Interpretability.* NeurIPS 2023.
* Weiss, Goldberg, Yahav. *Thinking Like Transformers* (RASP). ICML 2021.
* Suzgun et al. *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.* 2022 (BBH).
* Cobbe et al. *Training Verifiers to Solve Math Word Problems.* 2021 (GSM8K). Gao et al. *PAL.* 2022 (GSM-Hard).
* Also related: IGC (a calculator integrated into a frozen LLM), Qwenputer (a CPU compiled into Qwen), and Percepta's
  WebAssembly-in-a-transformer work.

## License

MIT (see `LICENSE`). BBH and GSM8K/GSM-Hard are MIT-licensed and are downloaded, not redistributed (pinned BBH commit
and Hugging Face dataset revisions in `data/`). The earlier phase in `finetuned_phase/` downloads MNIST (LeCun, Cortes,
Burges; CC BY-SA 3.0), also not redistributed. The Qwen3 weights
(Apache-2.0) are downloaded from Hugging Face.
