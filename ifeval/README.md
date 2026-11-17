# IFEval rule-keepers

Implants that make a frozen chat model keep formatting instructions it reads in the prompt ("do not use any commas", "answer
in lowercase", "wrap your response in double quotes", ...). They read the prompt's tokens, latch the instruction, and during
the answer add a write to the host's last hidden state (before its final norm) that vetoes or forces tokens. When no
instruction is latched, the write is exactly zero and the answer is byte-identical to the plain model.

**Host: `Qwen/Qwen3-0.6B`** (the post-trained chat model, thinking off, chat template), not the Base model used elsewhere in
this repo. Its weights are not changed. Greedy decoding, max 2,048 new tokens, bf16, one GPU.
Scoring: the official IFEval checkers from `lm_eval` (`lm_eval.tasks.ifeval.utils.process_results`), `langdetect` seed 0.

There are two parts:

| part | rules | status |
|---|---|---|
| `compiled_v2/` | 3 rules: `no_comma`, `lowercase`, `quote` (open + close) | **done**, held-out result below |
| `heldout/` (Team D) | 14 rules, developed on our own generated prompts, judged per tool | **results pending** (final IFEval / IFBench run in progress when this was packaged) |

Older numbers that came from the Python reference rule-keepers (prompt-strict 58 -> 70 with 8 rules, the Qwen3-1.7B port,
Multi-IF transfer) are **not** reported here: those rules ran as Python logic at inference, which this repo does not count as
an implant result. `compiled_v2/gen.py` keeps the Python reference only as the equivalence check for the compiled circuit.

## Data

```bash
python ifeval/data/prepare_ifeval.py     # google/IFEval from Hugging Face (Apache-2.0) -> ifeval/data/ifeval.jsonl (541 prompts)
python ifeval/data/prep_transfer.py      # optional: IFBench (allenai/IFBench_test) and Multi-IF first turns, same format
```

The prepared `ifeval.jsonl` was checked to be record-for-record identical to the copy the runs used (541/541). IFEval is not
redistributed here. Split convention: **even keys = development** (patterns may be written while reading them), **odd keys
(299 prompts) = held-out**, run once.

`ifeval/data/reduce_ifeval_run.py` turns a generation file into the compact result files in `*/results/` (per prompt: the
instruction ids, strict/loose pass per instruction, finish reason, token count, which rules latched, and the SHA-256 of the
response; no text).

## compiled_v2: three rules compiled into weights

What runs per token (`compiled_v2/gen_c.py`, `compiled.py`, `wsm_hdl.py`; see `compiled_v2/NOTES.md`):
* reader: linear readers on the host's token embeddings (13 groups) + 0/1 lookup rows (12 groups) -> 25 flags, checked
  exact on all 151,936 vocabulary rows at start-up;
* a ReLU stack over a 16-token window (written with `wsm_hdl.py`, the same compiler pattern as `micromachines/hdl.py`):
  pattern matches for the three rules, role-header flags, first-answer position;
* hard-attention latch heads (any match so far) and a turn head (in the assistant turn or not);
* gates as ReLU on 0/1 wires; writes = solved veto vectors (comma; upper case, x2) and an emit-`"` direction (rms-normalised
  unembedding row / final-norm gain, gain 1e4).

Written vs solved: the circuit is written; the veto vectors in `compiled_v2/dirs.pt` (4.2 MB) are **solved** against the
frozen host (`solve_dirs.py`, Adam, seed 0) on its hidden states, teacher-forced on its own vLLM baseline answers to the
**even-key IFEval prompts** (`gen_baseline_vllm.py`). The rule patterns were also written while reading even-key prompts.

**Disclosed non-weight pieces:** the greedy decode loop (same as the baseline); a Python switch that turns the hook on for
generation (not a rule decision); and, for closing the quote, a **second readout of the host state through the frozen LM
head** (logits of rmsnorm(h + W0)): if end-of-turn is the host's top token, the write becomes "emit `"`". That readout is a
fixed linear map of the host state, but it is close to a logit processor; read the quote-close result with that in mind.

Reproduce (one GPU; run from `ifeval/compiled_v2`):

```bash
python ../data/prepare_ifeval.py
# optional, to re-solve dirs.pt instead of using the included file:
python gen_baseline_vllm.py Qwen/Qwen3-0.6B ../data/q3_nothink.jsonl nothink     # vLLM, temperature 0, seed 0
python solve_dirs.py dirs.pt                                                       # reads the even keys only
# held-out run: compiled, same-harness baseline, Python reference; same 4 shards and batches (bs 16, max_new 2048)
bash launch.sh               # -> runs/odd_{comp,none,pyref}.jsonl, then score2.py and analyze.py
python summarize.py          # the numbers below, from results/ (no model, no scorer)
```

Results (`compiled_v2/results/odd_{none,comp,pyref}.json`, odd keys, n = 299):

| | prompt strict | prompt loose | instruction strict | instruction loose |
|---|---:|---:|---:|---:|
| baseline (no implant) | 56.52-56.86 | 61.54-61.87 | 67.99-68.21 | 71.30-71.52 |
| + compiled 3-rule implant | **61.54-61.87** | 66.22-66.56 | 72.19-72.41 | 75.50-75.72 |

* Ranges: the `lm_eval` language checker is not deterministic on key 1129 (a silent prompt with identical text in both runs
  scores differently between scorer calls).
* Prompt-strict vs baseline: 18 gained, 3 lost (keys 279, 1021, 1781), net +15.
* Per rule strict passes: no_comma 18 -> 34 of 36; lowercase 18 -> 21 of 23; quotation 21 -> 22 of 23.
* Silent prompts (circuit never fired): 222 of 222 byte-identical to the baseline.
* Compiled vs the Python reference of the same three rules: 299 of 299 responses byte-identical.
* Truncations (2,048-token cap): 18 vs 15 for the baseline over all prompts.
* Dropped before the run (not compiled, not run): repeat, forbidden words, two responses, paragraphs, nth-paragraph word.
  An all-caps veto was rejected earlier (answers loop).

## heldout/ (Team D): 14 rule-keepers, developed without the held-out prompts

Protocol: `bbh_generic/PROTOCOL.md` (rules 1-13), applied to IFEval: develop only on our own generated prompts (`gen_data.py`;
some templates are our rewordings of instruction sentences in even-key IFEval prompts, each tagged `drv:<even key>`), judge
every tool with an independent reviewer who writes reworded variants (`JUDGE_<tool>_<round>.md`, `judge/`), show each tool
outside its benchmark style (`data/ood.jsonl`), and run the odd keys once. Full log: `heldout/NOTES.md`.

Rules: no_comma, lowercase, uppercase, wrap in quotes, end with phrase, postscript, forbidden words, repeat the request, two
responses, N paragraphs (`***` or blank lines), paragraph k starts with X, title `<<...>>`, highlight >= N sections, exactly N
bullets. Judge verdicts (final round): PASS or PASS-conditional for all 14 (table at the end of `heldout/NOTES.md`). The judges
note that recall on their novel phrasings is modest (about 40-75% per tool) and that misses fail safe (byte-identical to plain).

How it attaches (`heldout/circuit.py`, `tables.py`; frozen, md5 in `heldout/frozen_MD5.txt`): `run.py` calls
`circuit.step(token ids)` once per token and a forward pre-hook on the host's final norm adds `circuit.write(h)`. The circuit
is a fixed tensor program: 0/1 vocabulary lookup tables and an n-gram matcher over the last 5 token ids (readers), scope /
recency / latch heads (hard attention over earlier positions), pointer registers for argument spans, counters, and an emitter
that forces a token with a large-gain unembedding direction (1,695 rare tokens use a solved direction instead:
`vecs2_forcefix.pt`, 7.4 MB, the `force_fix` rows of the frozen `vecs2.pt`, solved by `solve_vecs2.py` on the host's plain
answers to our dev prompts). The same code runs for every prompt; no regex and no Python branching on data.

**Disclosures (read before citing any heldout number):**
* the circuit is written as batched tensor code (like `micromachines/rethink.py`), not exported as ReLU weight matrices;
* it uses **three readouts of the host's final state through the frozen LM head**: is the top token end-of-turn, does it
  contain a newline, is it in the currently blocked set. When the top token is blocked (commas, `***`, `\n\n` once N
  paragraphs, end-of-turn while a required part is missing, extra bullets, tokens completing a forbidden word), the emitter
  forces the host's best token outside the set (a masked hard-max over the unembedding rows). This is close to a constrained
  -decoding logit processor, computed from the host's own logits;
* case rules respell each emitted token through fixed upper/lower-case tables (up to 6 tokens);
* parts 5-9 of the development were shaped using detection runs on the judges' own variants, so judge precision numbers are
  partly in-sample.

Reproduce (one GPU; run from `ifeval/heldout`):

```bash
python gen_data.py --seed 0 --per 30 --out data/dev.jsonl                                     # DEV (included)
python gen_data.py --seed 1 --per 60 --out data/fresh.jsonl --extra fresh_templates.json      # FRESH test (included)
python gen_data.py --seed 2 --per 25 --out data/train2.jsonl                                  # seed-2 split (included)
python detect_test.py data/dev.jsonl                                                          # detection only (prompt tokens)
ln -s ../../data/ifeval.jsonl data/ifeval.jsonl
./queue.sh 4 "fresh.jsonl fresh_plain.s0.jsonl plain 0/6" "fresh.jsonl fresh_imp.s0.jsonl implant 0/6" ...   # bs 24
./queue.sh 4 "ifeval.jsonl ifeval_plain.s0.jsonl plain 0/4" "ifeval.jsonl ifeval_imp.s0.jsonl implant 0/4" ...
./final_score.sh                                                                              # IFEval odd / even / all, IFBench
# optional: re-solve the force_fix rows (needs runs/dev_plain.s{0..3}.jsonl from a plain dev run)
python solve_vecs2.py vecs2.pt runs/dev_plain.s0.jsonl runs/dev_plain.s1.jsonl runs/dev_plain.s2.jsonl runs/dev_plain.s3.jsonl
```

`queue.sh` sets `VECS=vecs2_forcefix.pt`; without that file `circuit.py` silently falls back to zero force_fix rows, so keep it.
`tables.py` caches its lookup tables in `tables.pt` (118 MB, rebuilt on first use, not included).

**Results: PENDING.** At packaging time the final held-out IFEval run (odd keys) and IFBench were still running; the fresh
(seed 1) and out-of-style runs had finished but were not yet scored in `NOTES.md`. No heldout number is reported here yet.

## Files not included

* model generations (`runs/`): only reduced, text-free records for the numbers above;
* `ifeval.jsonl`, IFBench, Multi-IF: download with the scripts in `data/`;
* `heldout/data/even_target.json` (the 181 even-key IFEval prompts with a target instruction, a reading aid): derivable from
  `ifeval.jsonl`;
* `tables.pt`, `vocab_strs.json`: caches, rebuilt automatically; the original `vecs2.pt` (only its `force_fix` rows are used).

Licenses: IFEval (google-research, Apache-2.0), IFBench (allenai), Multi-IF (facebook) are downloaded, not redistributed. Our
generated prompts (`heldout/data/`, `heldout/judge/`) contain a few short standard instruction sentences that also occur in
IFEval (Apache-2.0; most are from even keys, one generic sentence, "Wrap your entire response in double quotation marks.",
also appears in an odd-key prompt).

Software: torch 2.8.0, transformers 4.56.2, lm_eval 0.4.9 (compiled_v2 scoring) / 0.4.13 (heldout), langdetect; vLLM only
for `gen_baseline_vllm.py`; IFBench scoring needs the allenai/IFBench repository (commit 488918665d53).
