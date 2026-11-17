# MATH-500 failure analysis (analysis only, no implant)

**No implant was built here, and nothing in this folder runs in weights.** This is a measurement of which generic tools could
help small base models on MATH-500, done by reading failures and by *oracle injection*: Python cuts the model's own solution
where a perfect tool would fire, inserts the exact tool output as text, and lets the frozen model continue greedily. The
oracle numbers are upper bounds for a tool, not implant results. Full write-up: `NOTES.md`.

Setup: `Qwen/Qwen3-0.6B-Base` and `Qwen/Qwen3-1.7B-Base`, frozen, bf16, Minerva 4-shot prompt (`gen.py`), greedy, max 1,024
new tokens, stop at `\nProblem:`. Grading (`grade.py`): last `\boxed{}` (nested braces), else the last
`The final answer is $...$`; correct if Hendrycks-style normalized strings match or `math_verify` says they are equal.

## Headline numbers (from `NOTES.md`, `results_table.md`)

* Baselines: 0.6B 151/500 = **30.2%**, 1.7B 203/500 = **40.6%**.
* Hand read of 120 (0.6B) and 40 (1.7B) failures: about 2/3 start with a wrong plan, a hallucinated step or fact, a wrong
  formula or a misread question, which no exact tool fixes.
* Oracle injection, net of the no-insert (NOOP) controls (0.6B / 1.7B): LaTeX exact calculator +13 / +5; equation solver
  +7 / +2; expand/factor/simplify +1 / +2; all four combined 0.6B 30.2 -> 35.2%, 1.7B 40.6 -> 43.0% (raw).
* Loop breaker "try a different approach": 0 fixes on both models. A perfect gold-answer verifier + "redo" only +14 / +4 (raw).
  Unconditional "Hmm, wait. Let me rethink this.": 0.6B +21/-12 (+9 net), 1.7B +11/-18 (-7 net).
* An inserted value must be closed off (inside the model's own `\boxed{}`), otherwise the model keeps extending its own
  expression: 8 calculator fixes instead of 17 (0.6B).

The full per-tool table (fires, fixed, broke, noise-adjusted, ids) is `results_table.md`; `python math500/final_table.py`
rebuilds it from the reduced files in `results/runs/` (checked identical).

## Reproduce (one GPU)

```bash
pip install datasets math-verify latex2sympy2_extended sympy   # plus torch/transformers as in the top-level requirements
cd math500
python gen.py Qwen/Qwen3-0.6B-Base gen_0.6b.json 64       # batch 64 (as run); downloads HuggingFaceH4/MATH-500 (test split, 500 problems)
python grade.py gen_0.6b.json graded_0.6b.json && python autopass.py graded_0.6b.json
python gen.py Qwen/Qwen3-1.7B-Base gen_1.7b.json 32
python grade.py gen_1.7b.json graded_1.7b.json && python autopass.py graded_1.7b.json
python dump.py graded_0.6b.json 120 read_0.6b.txt          # stratified failure dump for reading (seed 1); labels: annotations.py
python dump.py graded_1.7b.json 40 read_1.7b.txt
# oracle injection: kind:rounds[:noop], batch 24
./run_seq.sh 0.6B num:4 num:1:noop solve:4 solve:1:noop sym:4 sym:1:noop all:6 all:1:noop loop:4 loopcommit:4 check:4 rethink:1 rethink:1:noop
./run_seq.sh 1.7B num:4 num:1:noop solve:4 solve:1:noop sym:4 sym:1:noop all:6 all:1:noop loop:4 loopcommit:4 check:2 rethink:1
python final_table.py .                                    # the table, from your own or_*.json
```

Rounds per kind as in `NOTES.md` (check: 4 rounds on 0.6B, 2 on 1.7B; rethink 1; all 6; NOOP controls 1 round). Greedy
re-generation from a cut point is not bit-identical across batch compositions, which is why the NOOP controls exist; expect a
few problems to flip.

## Files

* scripts: `gen.py`, `grade.py`, `autopass.py`, `autopass_lib.py`, `dump.py`, `detect_test.py` (fire-point check),
  `oracle_lib.py` (fire-point detection, sympy/latex2sympy), `oracle_run.py` (cut, inject, continue), `run_seq.sh`,
  `final_table.py`, `summarize.py`; `annotations.py` = hand labels (first error class, all classes, which tool alone fixes it)
  for the 160 read failures, keyed by MATH-500 index.
* `results/graded_{0.6b,1.7b}.json`: per problem subject, level, token count, hit-limit, extraction kind, correct, loop flags.
* `results/runs/or_<model>_<kind>.json`: per problem `orig_correct`, `fired`, `new_correct`, number and type of fires.
* `results/read_*.ids.json`: which failures were read.

Not included: the generations, the injected texts and the read dumps (they contain MATH-500 problems and model output);
superseded runs. Rerun the commands above to get them. MATH-500 (HuggingFaceH4/MATH-500, a subset of the MATH dataset of
Hendrycks et al., MIT license) is downloaded by `gen.py`, not redistributed; result files refer to problems by index only.
