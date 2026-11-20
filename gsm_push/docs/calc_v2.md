# Calculator v2: exact arithmetic for the forms calc_s4 misses (2026-11-20)

> **Update (forward-only requirement).** Every implant must work in one generation call, be forward-only and be a compiled weight-space circuit. The `verify_rsl` hook below rewinds, so it does **not** qualify.
> The qualifying version is the compiled micromachine **chain2**: §7 has its numbers, and §8 has the failure diagnoses for the "debug failures through implants" request.
> **Current recommendation: KEEP chain2** (calc_s4 + chain2, compiled). §1-6 are kept as the record of the hook study.

**Recommendation: KEEP** (mode `verify_rsl`). On the RL+calc host (a2_u300_s2):

| set | off | v2 | delta | fixes / breaks | rows where v2 writes |
|---|---|---|---|---|---|
| DEV (600) | 79.83 | **80.83** | **+1.00** | 6 / 0 | 2.2% |
| dev2 confirm (300) | 78.00 | **79.00** | **+1.00** | 3 / 0 | 2.0% |
| scaled-number slice of split B (300, "GSM-Hard-like") | 61.33 | **68.33** | **+7.00** | 22 / 1 | 12.3% |

- Rows where v2 does not write are bit-identical to the run with the hook off: 587/587, 294/294 and 263/263.
- The one break is a scaled item with an impossible gold; there v2 wrote the exact value of the expression the host had written.
- Secondary check, base host (u0) + calc: dev 62.17 → 63.33 (+1.17, 7/0), dev2 62.00 → 62.00 (0/0), scaled 33.0 → 40.3 (+7.3, 22/0).
- GSM8K test was NOT run.

**Which implementation.** This is the decode-time token hook in the RL engine, not a compiled circuit. The time budget did not allow a compiled N-operand, parenthesis-aware evaluator.

## 1. Hypothesis and what the DEV traces show

Hypothesis (from `dig/gsm8k_failures.md` §2c): calc_s4 fires only on `a op b (op c) =`. Everything longer is host-written and often wrong.

**DEV.**
- DEV = GSM8K train split B (`common.part`: md5 % 10 in {6, 7}; not the RL pool A).
- Selection: GSM items only, length filter as in the harness, `random.Random(0)` shuffle.
- dev = items [0:600], dev2 = items [600:900].
- Scaled slice = the scaled-number train items of split B (same split rule, by source question), shuffled with seed 0, first 300.
- The exact question lists are in `calc_v2/out/r7/splits.json`.

Greedy traces of the host with the hook off (`analyze.py`, `scan_traces.py`):

| | dev (121 failures) | dev2 (66) | scaled (116) |
|---|---|---|---|
| equations written by calc_s4 | 1,838, 0 wrong | 893, 0 wrong | 903, 2 wrong (see below) |
| host-written results, 4+ operands | 40, 9 wrong | 23, 4 wrong | 18, 13 wrong |
| host-written results, brackets / comma / sign | 32, 3 wrong | 20, 3 wrong | 21, 2 wrong |
| failures that contain a wrong host-written result, by type | chain 9, decimal 6, comma 1, paren 1, sign 1 (any: 10) | chain 4, comma 2, decimal 1 (any: 6) | chain 13, decimal 7, paren 2, sign 1 (any: 20) |
| wrong *restatements*, e.g. `$50 - ($6669528 + $6669531) = $50 - $13338959` | a few | 0 | about 6 |

Three more patterns came out of reading the traces:
1. **The host often restates instead of answering.** Examples: `(212 - 41) / 3 = 171 / 3 = 57`, `5 x (1 - 0.80) = 5 x 0.20 = 1`. Most host-written bracket "errors" counted by the old scanner are such restatements, not wrong results.
2. **calc_s4 has a second blind spot: negative results.** It writes only the `-` and the host writes the digits (`65 - 664045 =⟦ -⟧664080`). This accounts for both "calc wrong" cases on the scaled slice.
3. **Decimals.** calc_s4 prints quotients truncated to 6 decimals (3/7 → 0.428571). The host then multiplies the truncated literal (`2100 x 0.428571 = 899.9991`).

## 2. Implant design

Code: `gsm_push/calc_v2/`
- `calc2.py`: parser, exact evaluator and the per-row token state machine. Model-free.
- `c2_engine.py`: the engine hook.
- `c2_eval.py`: the eval hook. It is exec'd by `dig_fail/tw_dump_lean.py`.
- Runners: `run_c2.sh` (A100), `run_c2_h100.sh`.
- `test_exact.py`: the exactness test. Its output is in `test_exact.log`.

**What f computes.**

f(text of the current line before `=`) = the exact value of the longest arithmetic expression that ends at that `=` and has a clean left boundary.

- **Grammar.**
  - Numbers: `1,234,567` grouping, decimals, `$` prefix.
  - Operators: `+ - * / x × ÷`, with standard precedence.
  - Nested brackets.
  - Unary minus.
  - `x` counts as an operator only between spaces or between digits.
- **Left boundary.**
  - A word before the expression is fine.
  - A digit, `)`, `%` or an operator before it is not. So `3J - 8`, `2x + 3` and `5 hours x 4 + 2` never parse.
- **Arithmetic** uses `Fraction`.
- **Output.** The value is exact when it has at most 6 decimals; otherwise it is rounded half-up to 6 decimals.
- **Exact literals.** The evaluator remembers the exact rational behind every truncated or rounded literal it or calc_s4 printed (`0.428571` → 3/7). A later expression that uses that literal is evaluated with the exact value, so `2100 x 0.428571` → 900.

Nothing in f refers to GSM8K. It is a checker for any "expression = value" text: worksheets, unit conversions, BBH multistep arithmetic, and spreadsheet-like or financial reasoning.

**Where it lives.**
- *Inside the decode CUDA graph:* one extra op per step. The sampler is wrapped so that it also stores `ALT[b]`, the argmax over tokens that do not start with a digit or an operator / `=` / `(`. The host's own token is not touched.
- *Outside the graph, between replays (sync every step):* each row runs `calc2.Row` on the token the host just booked.

**Policy `look`.** v2 never pre-empts the host. After a trigger `=`, it lets the number be written, by the host or by calc_s4, and records a snapshot at the number start: the row's KV slot, its output position and calc_s4's state. When the number ends, there are four cases:
- **Restated expression** (the next token is an operator): stand down. With `_rs`, the text between the two `=` signs is parsed when the next `=` arrives. If it does not equal the first expression's exact value, the row is rewound to the snapshot and the exact value is written (`rewrite_restate`).
- **Value equal to the exact one** (|diff| ≤ 1e-6): nothing happens.
- **Value equal to the strict left-to-right reading of a flat expression** (`170 + 150 + 160 / 3 = 160`, meaning an average; the `l` in `rsl`): treated as the host's grouping, so nothing happens.
- **Any other value**: rewind (KV slot, output position, calc_s4 state restored) and write the exact value's tokens. At the next step, a digit or operator continuation is replaced by `ALT`.

**Silence.** A row that is never rewritten keeps every token the host sampled. With finished-row compaction off (`C2_NOCOMPACT=1`, so every row keeps its batch slot), silent rows are bit-identical to the hook-off run on every set.

With compaction on (the stock engine), batch shapes change when rewritten rows change length. Then a few silent rows differ by bf16 batch-shape noise: 588/590 on dev, 267/283 on scaled. That is the same noise the stock engine shows between sync schedules: off with sync 16 vs off with sync 1 differ on 1 of 16 rows.

**Exactness** (`test_exact.log`, real Qwen3 tokenizer). For each case, the text is fed token by token, then a simulated host writes a wrong number. 10,000 random expressions per form:

| form | wrong | missed |
|---|---|---|
| N-operand chains (4-9 operands, mixed precedence) | 0 | 0 |
| nested brackets (depth ≤ 4) | 0 | 0 |
| comma-grouped numbers | 0 | 0 |
| leading minus | 0 | 0 |
| decimals up to 6 places, incl. non-terminating quotients | 0 | 0 (checked against an independent ast evaluator, to 6 decimals) |
| exact-literal case (`a / b` truncated by calc_s4, then `c x literal`) | 0 | 0 |

It also never acts where it should not:
- correct restatements or correct values: 0 / 10k acted;
- wrong restatements: 0 wrong, 0 missed;
- basic calc_s4 forms, algebra lines and unit-word lines: 0 / 10k acted.

## 3. Iterations (all on the RL host; dev / dev2 / scaled deltas vs off)

| run | variant | dev | dev2 | scaled | notes |
|---|---|---|---|---|---|
| r1 (h100, preempt) | `ext`: write the value as soon as the host starts a number, on 4+ / brackets / comma / sign | +0.17 (8/7) | +1.67 (5/0) | – | **7 breaks.** The host was restating (`= 171 / 3`, `= 60 + 30`, `= 15000 - 2000`); the pre-written value, then the veto, pushed it into nonsense (`= 57 degrees / 3 = 19`). One bug: `(1/6)X - 30` parsed `X` as times. Both fixed. |
| r2-r3 | `ext`, policy `look` (rewind on mismatch, stand down on restatement) | +0.83 (5/0) | +1.0 (3/0) | +5.0 (15/0) | the breaks disappear |
| r4 | `verify` (check every equation, including calc_s4-written ones) + absolute 1e-6 tolerance | +0.83 (5/0) | +1.0 (3/0) | +5.67 (17/0) | catches calc_s4's sign-only negatives |
| r5 | `verify_rs` (also check restatements) | +1.0 (6/0) | +1.0 (3/0) | +7.0 (22/1) | the scaled break has an impossible gold |
| r6/r7 base host | `verify_rs` → `verify_rsl` | +1.17 (7/0) | -0.33 (0/1) → 0 (0/0) | +7.3 (22/0) | the base-host break was `170 + 150 + 160 / 3 = 160` (an average): the left-to-right agree rule removed it, and it changed nothing on the RL host |

`all` (fire also where calc_s4 is silent on basic forms) gave the same as `ext` on dev/dev2 and the same accuracy on scaled. Its gain is subsumed by `verify`.

## 4. Fixes and breaks (final, RL host)

Fixes (dev, host value → v2):
- `$7 + $20 + $20 + $20 = $77` → 67 (#76)
- `90 + 98 + 92 + 94 = 384` → 374 (#193)
- `8 + 16 + 5 + 2 + 8 + 5 = 40` → 44 (#299)
- `$585.80 + $565 + $599.20 + $567 = 2316` → 2317 (#354)
- `30 + 60 + 40 + 35 = 125` → 165 (#359)
- `6 - 3 = 3 x 8` → `= 3` (#78): restatement check. This one is partly lucky: the host had misused `=` to continue the computation.

Fixes (dev2):
- `72 + 58 + 45 + 50 + 65 + 54`
- `3 x $160,000 + $2000`
- `1 / (4 x 45)`, then the exact-literal chain `1 / 0.083333 = 12`

Fixes on the scaled slice, via `rewrite_restate`:
- `$50 - ($6669528 + $6669531) = $50 - $13338959` → -13339009
- `15 + (2275724/100) x 15 = 15 + 3413586` → 341373.6
- `915230/5 x $500 = 915230 x $500` → 91523000

Breaks:
- dev 0, dev2 0.
- Scaled #103: `85 - 6701619 + 340 = ... = 340 - 6701619 = -6701279`. The host reached the scaled gold by a wrong restatement; v2 writes -6701194, which is exact for what was written.

Off-by-construction failures that v2 cannot see:
- the host restating inside brackets: `= (6309348) / 3`;
- expressions with words or `%`: `20% of 45`;
- "x" used for an unknown;
- equations without `=`.

## 5. Cost

- Writes on 2% of GSM rows and 12% of scaled rows.
- Rewound tokens: dev 79, dev2 29, scaled 447 in total. Net output tokens change by -0.05% / -0.03% / -0.1%.
- Sync every step plus Python bookkeeping cost no measurable time at B = 128 on the A100: dev off 21.2 s vs v2 20.9 s per 600 items, greedy.
- The `look` policy needs the torch implant step (`--fused 0`), so calc_s4's state can be snapshotted.

## 6. How to stack

- **Attach.** `eng.__class__ = C2Engine; eng.c2_setup(); eng.set_c2('verify_rsl')`. Run with `--fused 0`, and use `C2_NOCOMPACT=1` if bit-identity of silent rows matters. It needs no change to the host or to calc_s4.
- **Composes with the answer-sanity resample** (`dig/sanity/san_engine.py`). v2 only edits equation results; sanity acts at `####`. Make one engine subclass call both hooks: v2 between replays, sanity in-graph. Expect the gains to add roughly. v2 removes some non-integer answers (`899.9991`, `3.47619`) that sanity would otherwise have resampled, so a small overlap is likely.
- **Next steps if wanted:**
  - compile the evaluator as a circuit;
  - a `%`-operand rule (`x% of y`);
  - catching restatements inside brackets.
- **Expected test effect:**
  - The dev / dev2 gains (+1.0 each, 9 fixes / 0 breaks over 900 items) suggest +0.7 to +1.2 pp on GSM8K test.
  - That is below the failure analysis's 1.5-2 pp upper bound, because many host "errors" were restatements the host later fixed itself.
  - GSM-Hard should gain several points, given +7 on the scaled slice.

## Files

- Code: `gsm_push/calc_v2/*.py`, `*.sh`.
- Runs (jsonl per row with marked text: ⟦calc_s4⟧ «v2» ‹veto›, events, tokens): `calc_v2/out/r1 … r7, r6base, r7base`.
- Pod copy: `A100:<work>/calc_v2/`; the RL checkpoint was also copied to `H100:<work>/calc_v2/ckpt/` (sha256 checked).


## 7. calc v2 as a COMPILED forward-only micromachine: chain2

**What changed.** One generation call, no rewinding and no Python hook. chain2 is a micromachine built with the project's own circuit compiler: `micromachines/hdl.py` → `compile_net`. It sits next to the unchanged calc_s4 and has the same architecture as every machine in `micromachines/machine.py`:

- a linear reader from the host's embedding matrix to symbol flags, exact on every vocabulary row;
- stack 1 at the `=` token, a ReLU net over a 96-token window;
- one hard attention hop;
- stack 2 at every position, a ReLU net;
- a one-hot output times write directions, solved against the frozen host.

Code: `calc_v2/chain2.py` (the circuit) and `calc_v2/implant2.py` (calc_s4 + chain2 as one implant in the RL engine).

Size: 17,063 ReLU units, 784k nonzero weights, 65 input symbols. The reader is exact on the RL host's own embedding rows: 0 mismatching rows over 151,936.

**Compiled grammar.**
- **Numbers:** up to 9 integer digits, comma groups of 3, up to 4 decimals, optional `$`.
- **Operators:** binary `+` and `-`, spaced or unspaced.
- **Sign:** a unary minus on the first number. Negative results are written in full: `-` and then the digits. The host's lead can be `' '`, `' $'` or `' -'`.
- **Output:** the exact value, with trailing fraction zeros dropped, followed by the solved end-of-number veto.

**Fire rule.** This is the rule agreed with the combination agent, restricted to the compiled subset. Every condition below is decidable from the tokens before `=`.
- It fires only where calc_s4 is silent: 4 or more operands, OR a comma-grouped number, OR a leading minus with 3 or more operands.
- No brackets and no `x` / `/`, so a mixed or bracketed expression stays silent. This also removes every precedence-vs-left-to-right disagreement and the average form `170 + 150 + 160 / 3`.
- The span must start at a safe token. Safe tokens are `:`, a newline, `=`, a non-number comma, or one of 47 single-token words such as `' is'`, `' has'`, `' of'`, `' total'` (taken from what precedes chains in traces). So `2S - 4 * 4`, `3J - 8` and `5 hours + ...` never fire.
- calc_s4 misfires on about 5% of decimal chains with 4 or more operands. For that reason the composite masks calc_s4's output wherever chain2 writes, using one fixed AND on the two one-hot outputs.

**Compiled vs not compiled.**

| form | status |
|---|---|
| N-operand `+`/`-` chains, `$`, decimals ≤ 4 places, comma groups, leading minus, negative results | **compiled, exact** |
| `x` / `÷` chains (`a x b x c x d`) | not compiled (0 dev fixes in the hook study; left for later) |
| brackets | deliberately silent (3 of the 7 breaks of the pre-empting hook) |
| completing calc_s4's sign-only negative results (`a - b =⟦ -⟧`), which the stack agent's hook does | not compiled |
| exact-literal substitution (`2100 x 0.428571` → 900) | not compiled |

**Exactness.** `calc_v2/test_chain2.py` (`test_chain2_final.log`). It runs through the real decode path: the same `prime_rows` / `step` calls the engine makes, with calc_s4 attached. Each form uses 10,000 random expressions and both host leads, `' '` and `' $'`.

| form | wrong | missed | unexpected fires |
|---|---|---|---|
| chains of 4-9 operands (up to 9 digits) | 0 | 0 | 0 |
| comma-grouped numbers (2-6 operands) | 0 | 0 | 0 |
| leading minus (3-7 operands) | 0 | 0 | 0 |
| decimals (4-8 operands, up to 4 places) | 0 | 0 | 0 |
| negative results | 0 | 0 | 0 |
| **silence**: 10k texts of calc_s4 forms, 3-operand sums, mixed `x`/`+`, brackets, algebra lines, words inside chains, lists without `=`, decimals in prose | – | – | **0 of 10,000 acted** |

Before the gate, calc_s4 also wrote on 550 decimal-chain steps and disagreed on 309 of them. With the gate, calc_s4 is masked there, and it never writes after chain2 goes quiet (0 cases).

**DEV / dev2 / scaled** (`calc_v2/out/c4`, `c4base`). Same splits as §1, greedy, one `generate` call. Baseline = calc_s4 alone in the same engine.

RL+calc host:

| set | calc_s4 | + chain2 | fixes / breaks | rows where chain2 writes | silent rows bit-identical |
|---|---|---|---|---|---|
| DEV 600 | 79.83 | **80.67 (+0.83)** | 5 / 0 | 6.8% | 559/559 |
| dev2 300 | 78.00 | **79.00 (+1.00)** | 3 / 0 | 8.3% | 275/275 |
| scaled 300 | 61.33 | **65.33 (+4.00)** | 12 / 0 | 6.0% | 282/282 |

Base host (u0) + calc:

| set | calc_s4 | + chain2 | fixes / breaks |
|---|---|---|---|
| DEV | 62.17 | 62.50 (+0.33) | 3 / 1 |
| dev2 | 62.00 | 62.00 | 0 / 0 |
| scaled | 33.00 | 34.67 (+1.67) | 5 / 0 |

- **chain2 writes on many more rows than it fixes.** Most writes agree with what the host would have written.
- **Fixes:**
  - dev:
    - `$4.50 + $3.50 + $8.00 + $0.50 + $5.00 + $3.50 = $27.00` → 25.0
    - `8 + 16 + 5 + 2 + 8 + 5 = 40` → 44
    - `30 + 60 + 40 + 35 = 125` → 165
    - `$585.80 + $565 + $599.20 + $567` → 2317
    - `$7 + $20 + $20 + $20 = $77` → 67
  - dev2: `72 + 58 + 45 + 50 + 65 + 54` → 344, `$480,000 + $2000` → 482000, `35 + 46 + 27 + 61 + 31` → 200.
  - Scaled: 12 long sums with 7-9 digit numbers.
- **Breaks:** none on the RL host. The base-host dev break is a downstream re-reading after a correct sum.
- **Against the rewinding hook** (`verify_rsl`: dev +1.00 / dev2 +1.00 / scaled +7.0): the compiled circuit keeps all of the dev2 gain and most of the dev gain. It loses the restatement repairs and the `x`/`/` and bracket forms.
- **Cost:** about 19% more decode time (dev 600: 25.3 s vs 21.2 s; the sparse stacks run in torch, not fused). Output tokens change by under 0.1%.

**Operator veto (implant 2, tried and DROPPED).**
- **Idea.** Re-solve the end-of-number veto (`calc_v2/solve_veto_op.py`, split-A solutions only) so it also forbids an operator or `=` right after a calculator-written value. The motivating failures are #164 `42 / 6 x $3.00 =⟦ 21⟧ x $3.00 = 63` and #78 `6 - 3 =⟦ 3⟧ x 8`: calc_s4 evaluated the whole chain while the host meant to restate a partial result.
- **How the veto fits.** It holds the host's own non-veto choice on 98.9% of held-out number endings.
- **Results.** calc_s4 + op-veto: dev +0.17 (1/0), dev2 0 (0/0), scaled -0.33 (2/3). With chain2: dev +1.00 (6/0), dev2 +1.00 (3/0), scaled +3.67 (14/3).
- **Decision.** It does not hold up on dev2 or on the scaled slice, so it is not kept.

**Known writer quirk** (shared with calc_s4's dot-veto). After an integer value, a host `.` triggers one more digit veto. So `$X.00` can become `$X. 00` (the stack agent's test #988), or the host writes `.0` anyway (`$25.0`, value unchanged). A fix is to let tokens made only of zeros pass after the value and `.`. It is not done.

## 8. Debugging DEV failures through implants (about 40 arithmetic / quantity failures)

**Method.** I read the first 50 greedy failures of DEV (RL+calc host, calc_s4 only; `out/r7/dev_off.jsonl`) and kept the 44 that involve numbers or quantities. For each, the table gives the first wrong step and the side implant that would have stopped it, or "none" when no small exact or forward-only device can see the error.

Implant families:
- **A** exact function (calculator v2 / chain2);
- **B** answer-sanity trigger (negative or non-integer final answer → re-think; this is the sanity agent's implant);
- **C** unused-quantity ledger (a register that marks question numbers as used; flags at `####`);
- **D** operator veto after a calculator value;
- **E** a probe / steering implant (needs semantics: wrong entity, wrong relation);
- **–** none (plan or world knowledge).

| # | gold → pred | first wrong step | class | implant |
|---|---|---|---|---|
| 1 | 3 → 4 | `150 + 35 + 55` adds whole Tue/Fri times instead of the extra 5 + 25 | double count | E |
| 5 | 3 → 1.5 | "space for 4 more" read as 12 + 4 | relation direction | E (+B: 1.5 non-integer) |
| 6 | 96 → 576 | 6-inch tile side read as 6 sq in | units / world | – |
| 29 | 36 → 100 | "each horseshoe 2 kg" but a horse needs 4 shoes; also the 28 kg for farms never subtracted | world + bookkeeping | C (28 computed, never used) |
| 30 | 53 → 35 | stops at 105/3; the 18 left over is dropped | dropped quantity | C (18 unused) |
| 31 | 216 → 480 | "5 Fridays in a week" | plan | – |
| 36 | 11 → 121 | sum and product treated as separate equations | algebra | – |
| 39 | 30 → -10 | reward count from all 20 customers | plan | B (negative) |
| 53 | 3 → 10 | `7 - 17 = -10` then sign flipped by hand | relation | B (negative intermediate) |
| 59 | 50 → 100 | "50% play it" never applied | dropped quantity | C (50% unused) |
| 64 | 118 → 56.3 | gas cost per week mis-planned (×2, no ×5 days) | plan | B (non-integer) |
| 66 | 4 → 0.0667 | average per minute instead of per hour | unit choice | B |
| **76** | 3 → 3.476 | **`$7 + $20 + $20 + $20 = $77`** | **host chain arithmetic** | **A (fixed by chain2)** |
| 78 | 72 → 240 | `6 - 3 =⟦ 3⟧ x 8 = 24`: calc value continued as a product, then 24 used as a count | calc-induced continuation | D |
| 88 | 2 → 37 | boxes of 8 never used | plan | C |
| 102 | 26 → 28 | computes A = 6 in the second half, then uses 8 | wrong earlier number | E (register: 6 computed, never used) |
| 107 | 2200 → 110 | the 20 birds never multiplied | dropped quantity | C (20 unused) |
| 116 | 1200 → 1100 | 1/4 applied to the remainder, not to the original | plan | – |
| 117 | 100 → 350 | deliveries per station misread | plan | – |
| 120 | 2000 → 1400 | rent shared with Shelby dropped | dropped event | E |
| 121 | 26 → 40 | third week of eating dropped | dropped event | E |
| 129 | 21 → 12 | drainage 3 in/day never used | dropped quantity | C (3 unused) |
| 133 | 5 → 67.5 | net per two weeks used as per week | plan | B (non-integer) |
| 136 | 400 → 270 | couples' rooms hold 2 | world | – |
| 140 | 49 → 343 | multiplies by 7 twice | plan | – |
| 145 | 300 → 412.5 | adds Matt's books to Pete's | wrong entity | E (+B) |
| 159 | 2 → 2.67 | `$20x + $10x` ignores the +1 | algebra | B |
| **164** | 21 → 63 | **`42 / 6 x $3.00 =⟦ 21⟧ x $3.00`**: calc wrote the whole chain, host continued | calc-induced continuation | D |
| 167 | 200 → 250 | counts the 50 sliced slices twice | double count | E |
| 170 | 360 → 1680 | sheep eat 1/2 bag, not 2; months 12-8 not used | plan | – |
| 190 | 4 → 15 | Caleb's +11 never added | dropped event | E |
| 191 | 1990 → 1950 | younger brother born later: 1970 + 20 | relation | E |
| **193** | 96 → 86 | **`90 + 98 + 92 + 94 = 384`** (wrong value), then uses 4 quizzes not 5 | host chain arithmetic + plan | A (value fixed; answer still wrong) |
| 196 | 36 → 48 | uses Julian's boys (48) for Boyd | wrong entity | E |
| 207 | 24 → 27 | bus fare ×2, not ×4 (two people, both ways) | dropped factor | – |
| 211 | 60 → 2880 | 1/2 bottle a day never used | dropped quantity | C (1/2 unused) |
| 216 | 3 → 2.5 | average must exceed, so ceil | rounding | B (non-integer) |
| 219 | 72 → 84 | `24 + 12 / 2` meant (24 + 12)/2 | missing brackets | – (any exact calculator agrees with the host) |
| **235** | 25 → 27 | **`$4.50 + $3.50 + $8.00 + $0.50 + $5.00 + $3.50 = $27.00`** | **host chain arithmetic** | **A (fixed by chain2)** |
| 237 | 80 → 368 | one-fifth written as 0.02 | number transcription | E (fraction-word → value reader) |
| 240 | 28 → 42 | `70 - 28` drops day one (should be 56 - 28) | wrong earlier number | E |
| 288 | 7 → 3 | friend's subtotal invented, `$1 + $1 + $4 + $1.5 + $0.2 = $9.70` (exact 7.7) | bookkeeping + chain arithmetic | A (partly) |
| 291 | 24 → 120 | 5 weeks never used | dropped quantity | C (5 unused) |
| 296 | 90 → 0 | 5% discount never applied | dropped quantity | C (5% unused) |
| **299** | 44 → 40 | **`8 + 16 + 5 + 2 + 8 + 5 = 40`** | **host chain arithmetic** | **A (fixed by chain2)** |

Counts over these 44:

| implant | rows | covered by |
|---|---|---|
| A exact chain arithmetic | 5 | chain2 (4 fixed on DEV, including #354 and #359 beyond this list) |
| B answer sanity | about 10 | the sanity agent's resample; out of my scope |
| C unused-quantity ledger | 9 | not built |
| D operator veto | 2 | built and dropped (§7) |
| E semantic probe / steering | about 13 | not built |
| none | about 10 | – |

**Why C (unused-quantity ledger) is not built tonight.** It flags 9 of the 44 here. But the failure analysis (§4 there) measured "a question number never written" at 18/380 test failures against 4/939 correct answers, and many of these drops are words ("half", "both ways"), not printed numbers. As a forward-only implant it would need a phrase to write on firing, like the sanity trigger, and it is cheapest stacked into the sanity implant as an extra flag class. I suggest that to the stack / sanity agent rather than building a separate machine.

**Why E is not attempted.** Wrong entity, dropped event and relation direction need a semantic detector, e.g. the probe-verifier in `verifier/NOTEBOOK.md`. That is not a small exact device, and it is the selection agent's lane.

**Built and kept:** chain2 (A). **Built and dropped:** operator veto (D).

## 9. Files (compiled version)

- `calc_v2/chain2.py`: the circuit (`build()`, `register(tok)`).
- `calc_v2/implant2.py`: the calc_s4 + chain2 composite with the priority gate.
- `calc_v2/test_chain2.py`, `test_chain2_final.log`: exactness and silence.
- `calc_v2/c3_eval.py`, `run_c3.sh`, `run_c3x.sh`: the dev evals.
- `calc_v2/solve_veto_op.py`, `veto_op_qwen3.pt`: the dropped operator veto.
- `calc_v2/out/c4`, `out/c4base`: per-row outputs. «…» marks chain2-written tokens; the closing mark also takes in the host's token right after the value.
