# Weight-Space Micromachines: Brain Implants for LLMs

A micromachine (an "implant") is a small network whose weights are written by a coding agent, not learned by gradient
descent. It sits next to a **frozen** language model, reads the model's tokens, and, when its moment comes, adds a write
to the model's last hidden state that makes the model emit the token the implant wants (an exact digit, an answer letter).
At every other position its write is exactly zero, so the model's logits there are bit-identical to the unmodified model.
This repository rebuilds these implants from code for the public `Qwen/Qwen3-0.6B-Base` checkpoint (the nine BBH
implants, the GSM8K calculator `calc_s3`, and the `rethink` implant; the model is downloaded from Hugging Face on first
use, no custom checkpoint is needed), and contains the harnesses, the exactness tests, and the small result files behind
the numbers below.

**The rule for every result in this repository:** the implant is fully in weights. At inference it is fixed tensors (a
linear reader on the token embeddings, compiled ReLU stacks, hard-attention heads over its own cached values, solved write
directions, and in some implants small fitted parts such as a probe or a detector MLP), attached with forward hooks; the
only Python in the generation loop is the decode loop and attaching the hooks. Every exception is disclosed where the
number is reported, and the list is collected in [Exceptions to the in-weights rule](#exceptions-to-the-in-weights-rule).

## Components

| folder | what | host (frozen unless noted) | status |
|---|---|---|---|
| `micromachines/`, `scripts/`, `results/`, `tests/` | the nine BBH implants, the GSM8K calculator `calc_s3`, the `rethink` implant (state machine `rethink.py`; compiled `rethink_circuit.py`; compiled mistake detector + rethink `rethink_circuit_v2.py`), harnesses, exactness tests | Qwen3-0.6B-Base; calculator also Qwen3-1.7B/4B-Base | done (this README) |
| [`bbh_generic/`](bbh_generic/README.md) | BBH redo: eight **generic** tools (state tracker, ordering solver, intervals, 2-D walk, truth propagation, expression, boolean, calculator) built under a held-out protocol with independent judges; one real-BBH run; sample-size study `ft100/` and one-host integration `integrate/` | Qwen3-0.6B-Base, frozen (Team C) or fine-tuned to write tool calls (Teams A, B; checkpoints not released) | done; `ft100/`, `integrate/` **results pending** |
| [`latent_write/`](latent_write/README.md) | an exact intermediate value (`a*b`) computed by a compiled circuit and written into the residual stream at layer L, never as text | Qwen3-0.6B-Base, 1.7B-Base | done |
| [`ifeval/`](ifeval/README.md) | IFEval rule keepers: 3 rules compiled into weights (`compiled_v2/`); 14 rules developed on our own prompts (`heldout/`) | Qwen3-0.6B (chat) | `compiled_v2` done; `heldout` **results pending** |
| [`math500/`](math500/README.md) | which tools could help on MATH-500: failure reading + oracle text injection. **Analysis only, no implant** | Qwen3-0.6B/1.7B-Base | done |
| [`topic_fence/`](topic_fence/README.md) | probe-gated implant that keeps a chat model off one topic | Qwen3-1.7B (chat) | **results pending** |
| `finetuned_phase/` | the earlier phase: fine-tuned Qwen2.5-0.5B calls implants through special tokens; "eyes" implant with a trained MNIST CNN | Qwen2.5-0.5B, fine-tuned | snapshot, not re-run |

Each folder's README gives what it is, how it attaches (and its disclosures), the exact commands with seeds and
hyperparameters, where the result files are, and its headline numbers.

## What is not released

* **Model checkpoints.** No model weights are in this repository. The base models are public Hugging Face checkpoints
  (`Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`,
  `Qwen/Qwen2.5-0.5B`) and are downloaded on first use.
* **Fine-tuned hosts** (BBH Teams A and B, `bbh_generic/ft100/`, `bbh_generic/integrate/`, `finetuned_phase/`) are not
  released. What is released is the recipe: the exact training rows (or the seeded generator that writes them, with the
  SHA-256 of the original files), the fine-tune scripts and their hyperparameters. See the "Reproduce" sections of
  `bbh_generic/README.md`, `bbh_generic/ft100/README.md`, `bbh_generic/integrate/README.md`, and `finetuned_phase/README.md`.
* **Implant tensors** ("weight additions") are included when small: solved veto vectors (`results/gsm_bigger/veto_*.pt`),
  rethink write directions and the fitted mistake detector (`results/rethink_circuit/`, `results/rethink_v2/`), the
  gated-rethink probe (`results/rethink/`), latent-write writer banks (`latent_write/results/writers/`), IFEval veto
  vectors (`ifeval/*/`), topic-fence probes (`topic_fence/fitted/`). Compiled circuits are rebuilt from code at every
  start (deterministic); the remaining solved pieces are rebuilt by the scripts named in each README.
* **Raw generations.** Result files keep per-question records (gold, prediction, correctness, token counts, fire flags,
  a SHA-256 of the generated text) and the SHA-256 of the full output they were reduced from; no generated text and no
  benchmark question text (`scripts/reduce_outputs.py` and the per-folder reducers).
* **Benchmark data.** BBH, GSM8K, GSM-Hard, IFEval, MATH-500 and MNIST are downloaded by scripts, not vendored (the BBH
  3-shot prompt files that the teams' harnesses read are copied in `bbh_generic/team*/cot/`, MIT license).

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

### GSM8K / GSM-Hard, everything in weights: calculator, compiled rethink, compiled mistake detector (full test sets)

Frozen Qwen3-0.6B-Base, 8-shot, greedy, 1,319 problems per set. Every arm runs in the same generation loop with the same
stop rule, score and budget (`scripts/rethink_v2/run_gsm_v2.py`, max 700 new tokens, batch 32); the harness only attaches
the implants and generates.

| Qwen3-0.6B-Base, frozen | as is | + calculator | + rethink (always) | + both | + rethink gated by the detector | + calculator + gated rethink |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 52.01 | 53.30 | 56.56 | **57.54** | 55.65 | 56.71 |
| GSM-Hard | 26.61 | 34.34 | 29.57 | **37.15** | 29.04 | 35.18 |
| extra tokens / problem, GSM8K (GSM-Hard) | 0 | 0 | 145 (227) | 145 (241) | 108 (172) | 110 (188) |

* **calculator** = `calc_s3` (+ the solved "no more digits" veto), see below.
* **rethink (always)** = the compiled `rethink` circuit (`micromachines/rethink_circuit.py`, through the v2 module with the
  detector output fixed to 1): where the host is about to end its answer (after its first `####`), it writes
  `Hmm, wait. Let me rethink this.` and the host writes a second solution; the last `####` counts.
* **gated** = `micromachines/rethink_circuit_v2.py`: a **fitted mistake detector** (a one-hidden-layer ReLU MLP on the
  host's layer 12-24 states at the token before the first `####`, trained on GSM8K **train** generations plus a
  train-side "scaled numbers" analogue of GSM-Hard, ~52k labelled states; `results/rethink_v2/detector_hybrid.pt`) decides
  whether to rethink before the answer (`####` replaced by the phrase, 300-token budget), after it, or not at all.
  Detector AUROC on the host's own first answers: GSM8K test 0.85, GSM-Hard 0.84 (the earlier layer-28 probe: 0.81 / 0.66).
  Gated saves 22-26% of the extra tokens and stays 0.5-2 points below always-on: always-on breaks only ~1.7% of right
  answers, so there is little for a gate to save.

Provenance: `results/rethink_v2/summary.json` (`python scripts/rethink_v2/summarize_v2.py results/rethink_v2/final_eval`,
with paired bootstrap CIs) over `results/rethink_v2/final_eval/*.json`; AUROCs `results/rethink_v2/auroc_{test,hard}.json`.
Silent check (`scripts/rethink_v2/check_v2.py`, `results/rethink_v2/silent_check_hybrid.json`): bit-identical logits
wherever the implant does not write; host weights hash unchanged. The compiled v1 circuit alone, with its own baselines
at a 320-token budget (`scripts/run_gsm_circuit.py`, `results/rethink_circuit/summary.json`): GSM8K 52.24 -> always 56.56,
+calc 53.37 -> 57.54; GSM-Hard 27.29 -> 29.26, +calc 35.03 -> 37.23; its probe-gated variant 55.34 / 57.09 on GSM8K
(the GSM-Hard gated runs are partial, 192 and 768 problems). Its end detector is a readout of the final hidden state
through a fixed copy of the host's LM-head rows; it agrees with the host's own argmax on 8,565 of 8,565 checked positions.

Calculator on bigger hosts (`calc_s3` circuit unchanged; reader re-checked on the host's vocabulary, write directions and
veto re-solved per host; `scripts/run_gsm.py`, max 320 new tokens):

| host | GSM8K plain -> calc | GSM-Hard plain -> calc |
|---|---:|---:|
| Qwen3-0.6B-Base | 52.24 -> 53.37 | 27.29 -> 35.03 |
| Qwen3-1.7B-Base | 73.09 -> 72.33 | 45.41 -> 51.02 |
| Qwen3-4B-Base | 84.31 -> 84.69 | 58.98 -> 60.35 |

The gain shrinks with host size. On 1.7B the calculator costs 0.8 points on GSM8K (the dot-continuation of `calc_s3`
below and the host's bare-number answer format). Provenance: `results/gsm_full_implant/`, `results/gsm_bigger/`
(`veto_q17.pt`, `veto_q4.pt` are the re-solved veto vectors: `python scripts/solve_veto.py --model Qwen/Qwen3-1.7B-Base
--out results/gsm_bigger/veto_q17.pt --device cuda`, same for 4B).

### GSM8K / GSM-Hard, earlier run: research harness, rethink as evaluation code (superseded; kept for reference)

**Exception to the in-weights rule:** the two rethink columns of this table were produced by evaluation code (the script
cut the first answer, appended the phrase as text and generated again). The in-weights runs above replace them; they
agree within noise.

| Qwen3-0.6B-Base, frozen | as is | + calculator | + rethink (eval code) | + both (eval code) |
|---|---:|---:|---:|---:|
| GSM8K | 51.93 | 54.06 | 56.10 | 57.62 |
| GSM-Hard | 26.99 | 34.50 | 29.26 | 37.23 |

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
1,319. `calc_s4` fixes it.

"Rethink" is the `rethink` implant (`micromachines/rethink.py`, `scripts/run_gsm.py --rethink_implant always`). It
reads the host's last hidden state through the host's own final norm and LM head, and when the host's next token would
end its answer (a token that contains a blank line, such as `\n\n` after `#### 72`, or `<|endoftext|>`, or the 320-token
first-answer budget), it writes `\nHmm, wait. Let me rethink this.\n` instead, one token per position (10 tokens, one
solved write direction each, like the other implants), then goes silent and the model writes a second solution in the
same generation; the last `####` counts. It fires once per answer. The table's numbers were produced by the original
evaluation-code version of the same nudge (`scripts/run_gsm.py --rethink`, kept as the reference): the script cut the
first answer at the blank line, appended the phrase to the text, re-encoded it and generated again.
The implant was checked against that reference on the first 224 problems (length-sorted order) of GSM8K test (with and
without the calculator) and of GSM-Hard, batch 8 (`VERIFY.md` section 10). Up to the end of the phrase it feeds the
model the same text in 671 of the 672 problem runs (a 320-token runaway answer ending in a space, which the script
strips and the implant cannot) and the same tokens in 669 (that one, and two where the script's re-encoding
merges two calculator tokens). Same correctness on 215/224 (GSM8K: 73.66 vs 74.11), 213/224 (with calculator: 73.21 vs
74.55) and 219/224 (GSM-Hard: 45.98 vs 43.75). The
remaining differences come after the phrase and are bf16 noise between re-encoding the text (script) and continuing the
cache (implant): in float32, 31 of 32 GSM-Hard continuations are identical, and the script alone changes as much when
only its batch size changes. The full test sets were not re-run with the implant, so the table's rethink columns are
the script's numbers.

### Reading the model: a probe decides when to rethink (GSM8K test, generations in vLLM; evaluation-code analysis)

**Exception to the in-weights rule:** the policy table below was computed by evaluation code on saved vLLM generations
(it is the analysis that motivated the gate). The in-weights gate is the gated implant described after the table and,
on the full test sets, the compiled mistake detector in the GSM section above.

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
`fit_detector.py`. The table above was computed by evaluation code from those generations. This is a vLLM run, so its
52.08 baseline differs slightly from the 51.93 of the HF loop.

The gate is now also part of the implant: `--rethink_implant gated` evaluates a logistic probe on the host's layer-28
state (Hugging Face `hidden_states[28]`, the final normed state that the LM head reads) at the host's first `####` token,
inside the same hook, and arms the writer only if the score is at least the threshold chosen on the training problems.
The probe's weights, standardization and threshold are **fitted** (scikit-learn, on GSM8K training problems); they are
loaded from `results/rethink/rethink_probe_L28.pt`, made by `scripts/gate/export_probe.py`. The original probe file was not kept,
so the probe was refitted with the same code on a fresh `gen_rethink.py` run; the refit gives train AUROC 0.956
(original 0.955), test AUROC 0.808 on the new vLLM generations (original 0.810), threshold 0.251 (original 0.328). The
fresh vLLM generations themselves differ slightly from the original ones (test: 52.01 never / 56.10 always, vs 52.08 /
56.56), so the refit probe is close to, not identical with, the one behind the table above; with it, the evaluation-code
gate on the new generations gives 55.65% while firing on 66% of test problems. As an implant, on the first 224 GSM8K
test problems (HF loop, batch 8): never 68.75%, always 73.66% at 112 extra tokens per problem, gated 73.21% at 62,
firing on 51%; its gate decisions match the evaluation-code gate on 219/224 (`VERIFY.md` section 10). The full-set
gated run was not done.

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
* **Fitted on data, inside an implant:** the layer-28 probe of the gated `rethink` implant (its weights, input
  standardization and threshold: scikit-learn logistic regression on the host's states for 1,500 GSM8K training
  problems, threshold chosen on those problems; `scripts/gate/export_probe.py`). The rest of the `rethink` implant is
  written: its end-of-answer trigger (the host's own norm + LM head, then a fixed table of end tokens), its token tables,
  its once-per-answer state, and the forced token sequence; its 399 write directions are solved like the others'.
* **Fitted on data, inside the compiled rethink implants:** the end detector of `rethink_circuit.py` is a fixed copy of
  the host's own LM-head rows (not fitted); the mistake detector of `rethink_circuit_v2.py` is a fitted 3-head MLP
  (7 layers x 1,024 inputs -> 256 ReLU -> 3 heads; AdamW, seed 0, weight decay and epoch chosen on a selection split;
  thresholds grid-searched) on host states from GSM8K **train** generations and a train-side scaled-number analogue
  (`scripts/rethink_v2/`, `results/rethink_v2/detector_hybrid.pt`). Everything else in both circuits is written: token
  tables, hard-attention heads (first `####`, gate, once-per-answer), the budget counter and a convolution writer over
  the implant's own cached rows.
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
python scripts/run_gsm.py out/gsm_test.json --split test --mode calc --rethink_implant always   # rethink implant
python scripts/run_gsm.py out/gsm_ref.json --split test --mode calc --rethink        # reference: eval-code rethink
python scripts/compare_rethink.py out/gsm_test.json out/gsm_ref.json                 # problem by problem
python scripts/check_silent_rethink.py   # rethink implant: logits bit-identical wherever it does not write
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

GSM with everything in weights (the first GSM table; one GPU; vLLM only for the detector's TRAIN generations):

```bash
python scripts/solve_veto.py --device cuda                                       # -> runs/veto_qwen3.pt
python scripts/rethink_v2/make_scaled_train.py data/gsm8k/train.jsonl data/gsm8k/train_scaled.jsonl   # seed 1234
python scripts/rethink_v2/make_sel.py                                            # train-side sel / rep splits (md5 rule)
python scripts/rethink_v2/gen_train.py results/rethink_v2/gen_train_gsm.json --src data/gsm8k/train.jsonl --K 3      # vLLM
python scripts/rethink_v2/gen_train.py results/rethink_v2/gen_train_scaled.json --src data/gsm8k/train_scaled.jsonl --K 3
python scripts/rethink_v2/extract_states.py results/rethink_v2/gen_train_gsm.json results/rethink_v2/states_gsm.pt       # ~4 GB
python scripts/rethink_v2/extract_states.py results/rethink_v2/gen_train_scaled.json results/rethink_v2/states_scaled.pt # ~3 GB
python scripts/rethink_v2/policy3.py results/rethink_v2/pol3 300 '[12,14,16,18,20,22,24]' 256   # the fitted detector
cp results/rethink_v2/pol3/detector_hybrid_cap300.pt results/rethink_v2/detector_hybrid.pt      # (shipped)
bash scripts/rethink_v2/queue.sh scripts/rethink_v2/jobs_baselines.txt 4       # none / always arms -> results/rethink_v2/final/
bash scripts/rethink_v2/queue.sh scripts/rethink_v2/jobs_hybrid.txt 4          # detector-gated arms
python scripts/rethink_v2/summarize_v2.py results/rethink_v2/final > out/summary_v2.json
python scripts/rethink_v2/auroc_eval.py results/rethink_v2/detector_hybrid.pt test out/auroc_test.json \
       results/rethink_v2/final/test_plain_none_s0.json results/rethink_v2/final/test_plain_none_s1.json
python scripts/rethink_v2/check_v2.py --mode gated --intervene hybrid --cap 300 --detector results/rethink_v2/detector_hybrid.pt
# compiled v1 circuit and its 320-token baselines; bigger hosts
python scripts/rethink_circuit_build.py                                         # shortlist + write directions (cached)
python scripts/run_gsm_circuit.py out/test_plain_always.json --split test --mode plain --implant always --bs 32
python scripts/run_gsm.py out/test_calc.json --split test --mode calc --bs 32
python scripts/solve_veto.py --model Qwen/Qwen3-4B-Base --out results/gsm_bigger/veto_q4.pt --device cuda
python scripts/run_gsm.py out/q4_hard_calc.json --model Qwen/Qwen3-4B-Base --split hard --mode calc --veto results/gsm_bigger/veto_q4.pt --bs 32
python scripts/reduce_outputs.py out/q4_hard_calc.json out/q4_hard_calc_reduced.json      # the compact form kept in results/
```

The job files run each set as two batch-aligned shards (`--start_batch` / `--max_batches`); a single unsharded run gives
the same batches. The write directions are solved on first use and cached (`results/rethink_v2/write_dirs_all.pt`,
`results/rethink_circuit/write_dirs.pt`, both shipped).

Baselines in vLLM: `scripts/baselines/run_bbh_vllm.py`. Rethink gate: `scripts/gate/gen_rethink.py` (vLLM) ->
`fit_detector.py` -> `cost_curve.py`; the probe for the gated implant: `gen_rethink.py` -> `export_probe.py` ->
`results/rethink/rethink_probe_L28.pt`, then `run_gsm.py --split test --mode plain --rethink_implant gated`.

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
* **The rethink implant is unlike the others in two ways.** It reads the host's hidden state (its own next-token choice,
  and for the gate a fitted linear probe), not only its tokens, and it writes a fixed English phrase rather than a
  computed answer. Its state machine is a few lines of tensor code (`micromachines/rethink.py`), not a compiled ReLU
  circuit; the compiled versions (`rethink_circuit.py`, `rethink_circuit_v2.py`) replace the state machine with fixed
  tensor ops and give the first GSM table. The earlier GSM rethink numbers came from the evaluation-code version
  (`--rethink`); `rethink.py` matches it token for token up to the end of the phrase and within bf16 noise after it
  (`VERIFY.md` section 10). Where it does
  not write, the host's logits are bit-identical to the plain model, also at the `####` positions where the gated
  version evaluates its probe (`scripts/check_silent_rethink.py`: 11,598 positions over 80 problems, 0 mismatches).
* **Baselines are Qwen2.5-7B/14B**, not Qwen3 models of those sizes, and ran in vLLM.
* **Numerics.** GPU runs use bf16 with left-padded batches, so exact outputs depend on batch composition; reruns match
  token for token with the same batch size and a `--limit` that is a multiple of it. On CPU, Intel MKL is not bitwise
  reproducible across calls unless `MKL_CBWR=COMPATIBLE` is set (`demo.py` sets it).

## Exceptions to the in-weights rule

Collected from this README and the component READMEs (each is also disclosed next to its number):

* **GSM, earlier table** (research harness): the rethink columns were evaluation code. Superseded by the in-weights table.
* **Rethink-gate policy table** (vLLM generations): evaluation-code analysis; the in-weights gate is reported separately.
* **`micromachines/rethink.py`** keeps its once-per-answer state in a few lines of tensor code, not a compiled circuit;
  the compiled `rethink_circuit*.py` are the in-weights versions.
* **Reading the host's own next-token choice:** the rethink end detector, IFEval's quote-close, and the v2 read point
  ("the host is about to write `####`") read the host's state through a fixed copy of its LM-head rows. This is a fixed
  linear map, but it is close to a logit processor.
* **Fitted parts inside implants** (in weights, but learned from data, not written): the rethink probe and mistake
  detector, the solved write and veto directions, IFEval veto vectors, the latent-write writer banks, the topic-fence
  probes (all disclosed with their training data in the component READMEs).
* **Hosts that are not frozen:** BBH Teams A and B (and `ft100/`, `integrate/`) fine-tune the host to write tool calls;
  the tools themselves are compiled. Fine-tuned-without-tools is always reported next to with-tools.
* **`math500/`** has no implant: its tool numbers are oracle text injections from Python (upper bounds).
* **`latent_write/`** control arms (corrupted / mean values) override the written value from Python; they are controls,
  not results. Its window looks a few tokens ahead within the prompt (prefill).
* **`ifeval/heldout/`** (pending) is batched tensor code not exported as weight matrices, and forces the host's best
  non-blocked token (a masked hard-max, close to a logit processor).
* **`topic_fence/`** (pending): hooks that emulate in-weight units; its keyword arm is a Python regex baseline.

## Earlier phase (fine-tuned; separate, not re-run)

Before the zero-training work, the project fine-tuned Qwen2.5-0.5B to call implants through special tokens, built an
"eyes" implant with a **trained** MNIST CNN inside, and ran the first frozen-model calculator test
(Qwen2.5-0.5B, bare multiplication 4% -> 100%). That code, its result files and its numbers are in `finetuned_phase/`,
kept apart from the main code. Nothing there was re-run for this repository, and some raw files are on a machine that
was offline while this repo was assembled (marked as such in `finetuned_phase/README.md`).

## Repository layout

```
micromachines/          library: hdl.py (ReLU circuit compiler), machine.py (reader, write directions, attach),
                        specs.py (the implants), circuits/ (one hand-written circuit per implant), rethink.py (the
                        rethink implant), rethink_circuit.py / rethink_circuit_v2.py (compiled rethink; + mistake
                        detector), calcfmt.py, boolfmt.py
demo.py                 one prompt, one implant, bit-identity asserts
scripts/                run_bbh.py, run_gsm.py, solve_veto.py, build_implants.py, check_silent.py, compare_runs.py,
                        check_silent_rethink.py, compare_rethink.py, gen_deduction_vocab.py, baselines/run_bbh_vllm.py,
                        gate/{gen_rethink,fit_detector,cost_curve,export_probe,eval_gate}.py, run_gsm_circuit.py,
                        rethink_circuit_build.py, check_rethink_circuit.py, summarize_rethink_circuit.py,
                        rethink_v2/ (detector data, fit, policy, run, checks, summary), reduce_outputs.py
tests/                  exactness tests (circuit-only and through the Qwen tokenizer/embeddings)
data/                   download scripts (no data is vendored)
results/                result files behind every number + summarize.py (see results/README.md)
figures/                make_figs.py (data figures, from results/), diagram and animation scripts
finetuned_phase/        the earlier fine-tuned phase (snapshot)
bbh_generic/            BBH redo with generic tools (teamA/, teamB/, teamC/, peek/, ft100/, integrate/, PROTOCOL.md)
latent_write/           compiled latent write (exact value written into the residual stream)
ifeval/                 IFEval rule keepers (compiled_v2/, heldout/)
math500/                MATH-500 failure analysis (no implant)
topic_fence/            topic fence (results pending)
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

MIT (see `LICENSE`). Datasets are downloaded, not redistributed, with these exceptions noted:

* BBH (MIT; Suzgun et al.): downloaded at a pinned commit (`data/download_bbh.sh`). The official 3-shot CoT prompt
  files of the 12 tasks are copied in `bbh_generic/team*/cot/` (MIT permits redistribution with the notice; copyright
  the BIG-Bench Hard authors). No BBH test questions are included.
* GSM8K (MIT) and GSM-Hard (MIT): downloaded from Hugging Face at pinned revisions (`data/prepare_gsm.py`).
* IFEval (`google/IFEval`, Apache-2.0), IFBench and Multi-IF: downloaded by `ifeval/data/*.py`. Our generated prompts in
  `ifeval/heldout/data/` reuse a few generic instruction sentences that also appear in IFEval (disclosed there).
* MATH-500 (`HuggingFaceH4/MATH-500`, from the MATH dataset, MIT): downloaded by `math500/gen.py`; only per-problem
  grades are included, no problems.
* MNIST (LeCun, Cortes, Burges; CC BY-SA 3.0), used in `finetuned_phase/`: downloaded.
* Topic fence: Alpaca (CC BY-NC 4.0) rows are not included; AdvBench (MIT) is downloaded; `topic_fence/data/test_hand.jsonl`
  is our own.
* All other data files in this repository were generated by our own seeded generators.

The Qwen3 / Qwen2.5 weights (Apache-2.0) are downloaded from Hugging Face.
