# latent_write2: chains, word problems, other value types, a larger model

## SUMMARY (2026-11-17)
Frozen Qwen3-0.6B-Base unless noted, no CoT, n=300 per cell, exact teacher-forced scoring; "follow" = output equals
the answer implied by an injected corrupted value. Key figure: `latent_write2_curves.png`.
* **Several values (Q1).** `(a*b)+(c*d)`: 5.3% -> 90.7% (both products written at their own spans, L8); corrupting
  either or both moves the output to exactly the implied sum 85-92%, leakage <= 1%. Writing only one: 0-5%.
  Three values: 3.7% -> 60% (L8; model's own text ceiling for adding three numbers 70%), each of 4 corruption
  patterns followed 60-63%. 1.7B: 15% -> 96%, follow 93-96%. The first summand must be written no later than the
  second (P1 late: 7%, P2 late: 38%). Nested `((a*b)+c)*d`: writing P1 = 5-11% (= text with P1, 7%); writing
  P2 over the outer span = 97% / follow 98%. Chains work only if the last intermediate is written.
* **Word problems (Q2).** No product span: write over BOTH operand sentences with the states of a sentence stating
  the total: 3.7% -> 96.7% (lost, L4; follow 97.7%) and 6% -> 93.7% (rows), vs text tool line 71.7% / 83.0% and
  text rewrite 95% / 93%. The completing (2nd) sentence alone also works when its text rewrite does (rows 96.7%).
  Number-only writes: 0% at L0-L4 but 62-67% at L12-L16 (rows): early writes must replace the relation; after
  ~L12 the relation is bound and the digit slots act as the operand.
* **Other value types (Q3).** Boolean (truth of a 4-literal expression, consumed by `not`): 51% -> 100% /
  follow 100% (L0-L8). Comparison bit `a*b > c` consumed by `not`: 52% -> 100% / 100% up to L12, chance at L14
  (step); text tool with the product: 53.7%. Product feeding a close comparison: 99% at L0 but 64% at L12
  (comparisons read the operand earlier than addition). Weekday offset and literal "if true say X" are limited by
  the model in text (26%, 71%); the latent write reaches but cannot exceed that text ceiling.
* **Larger model (Q4).** Qwen3-1.7B: same write works (mid 19.7% -> 95.7-98.7%, hard 2.7% -> 93-97%, follow 90-99%;
  hard beats text tool line 89%). Window does NOT grow at equal depth (both 28 blocks: slot+suffix dead at L18;
  slot-only closes ~2 blocks earlier in 1.7B). Across Qwen2.5 0.5B (24 blocks) -> 1.5B (28): last working layer
  L14 -> L20 (0.58 -> 0.71 depth).
* Mean-value controls ~0% (numbers) or exactly chance (bits) everywhere: gains come from the value.
* Most interesting: what counts as "the right span" changes with depth (number-only 0% early, 67% mid), and a
  latent 1-bit comparison result beats a text tool line by 46 points because the model can use a bit it cannot
  derive from the product.

Follow-up to <notes>/latent_write (read its NOTEBOOK first). Frozen Qwen3-0.6B-Base (and Qwen3-1.7B-Base for Q4),
4-shot prompts, answer immediately (no CoT), greedy. Pod: A40 pod:<workdir>/latent_write2 (one GPU job at a time, self-capped
at 7.5 GB with `set_per_process_memory_fraction`). Mirror: <notes>/latent_write2.

## Engine (`lw2.py`)
* Prompts carry `[[...]]` markers around spans. A condition = task prompt + an *oracle* prompt: the same problem with
  each written span replaced by text that states the value (also marked).
* Alignment per span: oracle value tokens right-aligned onto the task span tokens; left-over task positions get the
  state of the oracle token just before the value (the E3 ' (' fix); if the oracle span is longer, its leftmost tokens
  are dropped (spans are chosen so that no value token is dropped).
* Writer (only fitted part): per layer and aligned position, ridge map [1, one-hot(value)] -> residual state, fitted on
  2000 oracle prompts with random values (never task answers). "write at L" = replace the residual stream entering
  block L at the aligned task positions. Modes: synth (the writer), donor (real oracle states; upper bound), mean
  (writer with the mean feature vector = no value information; control).
* Scoring: exact teacher-forced check (greedy output = candidate tokens, then a non-alphanumeric token), as in
  latent_write E2+. Each item is scored against [true answer, answer implied by each corrupted value].
  "follow" = output equals the answer implied by the injected corrupted value.
* Speed: shared-prefix KV cache (the 4-shot prefix is computed once; checked: same accuracies as without).

## Q4 larger model: Qwen3-1.7B-Base (`q4_scale.py`, A100 node; n=300 per cell; acc / follow-corrupted)
Same task and writer as latent_write E3/E4: `(a*b)+c`, product written at the span `a*b` (oracle ' (' state fills
the extra slot). Qwen3-1.7B has the SAME depth as 0.6B (28 blocks), width 2048 vs 1024.

| | none | text tool line | text oracle `(P)+c` | text oracle corrupted: follow |
|---|---:|---:|---:|---:|
| mid (a 11-29, b 3-9, c 10-99) | 19.7 | 95.7 | 98.7 | 99.0 |
| hard (a 12-49, b 12-29, c 100-899) | 2.7 | 89.0 | 96.3 | 95.7 |

Slot write (synth) entering block L (mean-value control: 0.0-0.3% for every L <= 20):

| L | 0 | 4 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|
| mid acc | 98.3 | 98.7 | 96.7 | 95.7 | 79.3 | 49.3 | 16.0 | 6.7 | 1.7 | 19.3 (=none) |
| mid follow corrupted | 99.3 | 99.3 | 96.7 | 94.7 | 78.7 | 47.7 | 14.7 | 7.3 | 2.0 | 0.3 |
| mid slot+suffix acc / follow | - | - | - | 97.3/99.3 | 97.3/98.0 | 95.3/95.7 | 86.7/87.3 | 9.0/7.3 | 2.0/1.0 | 20.0/0.3 |
| hard acc | 96.7 | 96.7 | 94.7 | 93.3 | 74.3 | 32.3 | 7.7 | 8.0 | 2.7 | 2.3 |
| hard follow corrupted | 94.0 | 95.7 | 92.0 | 89.7 | 70.3 | 32.7 | 6.3 | 3.3 | 2.3 | 2.0 |
| hard slot+suffix acc / follow | - | - | - | 96.3/94.3 | 95.7/94.3 | 95.0/91.7 | 84.7/79.3 | 8.0/6.3 | 2.0/1.7 | 1.3/1.3 |

Same code on Qwen3-0.6B-Base (mid, `q4_mid_0.6B`, for a like-for-like comparison; none 15.3%, tool line 90.0%, oracle 97.3%):

| L | 0 | 4 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.6B mid slot acc | 97.7 | 96.7 | 97.3 | 97.0 | 94.0 | 64.7 | 51.7 | 32.7 | 12.7 | 14.3 (=none) |
| 0.6B mid slot follow corrupted | 98.7 | 97.3 | 96.3 | 96.0 | 93.7 | 62.7 | 53.3 | 34.0 | 11.0 | 1.3 |
| 0.6B mid slot+suffix acc / follow | - | - | - | 96.0/96.7 | 95.3/96.3 | 96.7/97.3 | 89.0/90.7 | 15.0/19.0 | 6.3/8.0 | - |

* The write transfers unchanged to the larger model: 19.7% -> 95.7-98.7% (mid) and 2.7% -> 93-97% (hard) for L<=10,
  corrupted values followed 90-99%. Hard: latent write at L10 (93.3%) beats the text tool line (89.0%).
* Window vs size at EQUAL depth (both 28 blocks): it does not grow with width. Slot-only, the 1.7B model closes
  ~2 blocks earlier (L12 79%, L14 49%, L16 16%) than 0.6B (L12 94%, L14 65%, L16 52%). With slot+suffix both models
  work to L16 (87% vs 89%) and both fail at L18 (9% vs 15%). So the "last usable layer" is ~L16/28 = 0.57 depth for
  both; what the wider model does earlier is copy the operand into the later positions (the suffix), which the
  suffix refresh repairs.
* Note vs latent_write E6 (0.6B slot+suffix 94% at L18): prompts there ended "A: " (the first answer digit is computed
  at a prompt position that the suffix write overwrites); here they end "A:" (the space is generated, every answer
  digit is a decode position). With "A:" the window is 2 blocks shorter. The answer position format matters.
* Between L18 and L20 a write is worse than no write (1.7B mid: 2-9% vs 19.7% none): the edit breaks positions the
  model is reading without being consumed. From L22 the edit has no effect (= none, follow ~0%).
* Depth comparison within another family (Qwen2.5 base models, same code, mid, n=300):

| model (blocks) | none | tool line | text oracle | slot: last L >= 74% | slot+suffix by L (acc / follow) | first dead L (slot+suffix) |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (24) | 1.3 | 10.0 | 93.3 | L8 (74.3; L10 31.3) | L8 89.0/89.7, L12 89.7/89.7, L14 85.3/88.7, L16 14.7/12.0 | L16 (0.67 depth) |
| Qwen2.5-1.5B (28) | 14.0 | 62.0 | 98.0 | L14 (76.7; L16 57.0, L18 49.0, L20 40.0) | L12 96.0/96.0, L16 94.7/96.0, L18 78.7/83.3, L20 57.7/65.7, L22 7.3/4.3 | L22 (0.79 depth) |
| Qwen3-0.6B (28) | 15.3 | 90.0 | 97.3 | L12 (94.0; L14 64.7) | L14 96.7/97.3, L16 89.0/90.7, L18 15.0/19.0 | L18 (0.64 depth) |
| Qwen3-1.7B (28) | 19.7 | 95.7 | 98.7 | L10 (95.7; L12 79.3) | L14 95.3/95.7, L16 86.7/87.3, L18 9.0/7.3 | L18 (0.64 depth) |

  The window is a property of the model, not a fixed fraction of depth: in Qwen2.5, going 0.5B (24 blocks) ->
  1.5B (28 blocks) moves the last working layer from L14 to L20 (+6 layers for +4 blocks, i.e. it also grows as a
  fraction of depth, 0.58 -> 0.71), and the slot-only window is much longer (L8 -> L14-L20). In Qwen3, 0.6B -> 1.7B
  at the same depth does not move it (slot+suffix dead at L18 in both) and shortens the slot-only window by ~2
  blocks. So "more depth -> later operand read -> later usable write" holds across the Qwen2.5 pair; width alone
  (Qwen3 pair) does not buy a later window. Two points per family; read it as a trend, not a law.
* Text tool line is unreliable across models (Qwen2.5-0.5B: 10%; 1.5B: 62%), the latent write is not
  (0.5B: 1.3% -> 85-92% for L <= 6; 1.5B: 14% -> 96-98% for L <= 8).

## Q1a two independent latent values: `What is (a*b)+(c*d)?` (`q1_multi.py sum2`, 0.6B, n=300)
a, c 11-29; b, d 3-9; both products 3-digit. Each product written at its own span (`a*b`, `c*d`), writer fitted on
`(V1)+(V2)` prompts with random V1, V2. Candidates: P1+P2, P1'+P2, P1+P2', P1'+P2' (P' = random other product).
Baselines: none **5.3%**, text tool line (`Tool: a*b = P1, c*d = P2`) **95.3%**, text oracle `(P1)+(P2)` **93.3%**
(corrupted text: follows 92.3%).

| L (both spans) | 0 | 4 | 8 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|
| both true: acc | 94.0 | 81.3 | 90.7 | 84.7 | 52.0 | 16.0 |
| corrupt P1 only: output = P1'+P2 | 94.3 | 78.3 | 87.0 | 80.3 | 52.0 | 21.3 |
| corrupt P2 only: output = P1+P2' | 93.7 | 84.0 | 92.0 | 82.0 | 52.3 | 18.7 |
| corrupt both: output = P1'+P2' | 94.0 | 77.3 | 85.3 | 79.3 | 54.3 | 20.0 |
| mean-value control: acc | 0 | 0 | 0 | 0 | 0 | 0.3 |
| donor (real oracle states): acc | - | 90.7 | - | 92.3 | - | - |
| both + suffix refresh: acc / follow both | - | - | - | 84.7/79.0 | 56.0/58.0 | 21.3/23.7 |

Only ONE of the two products written (the model must still compute the other and add): 0-5% at every layer (e.g.
only P1 at L12: 2.0%; only P2: 0.0%). Mixed layers (P1 at L_a, P2 at L_b), acc / follow-both:
[4,12] 79.0/78.3, [12,4] 66.0/58.0, [0,16] 38.0/44.0, [16,0] 7.0/6.3.

* Two latent values work together: 5.3% -> 91-94% (L0-L8), each value independently controls the output: corrupting
  either one moves the answer to exactly the implied sum 78-94% of the time, with essentially zero leakage into the
  other candidates (<= 1.3%). Writing only one value gives nothing (the model still faces a two-hop problem).
* The window is narrower than for one value: 52% at L14, 16% at L16 (one value: 65% / 52%, Q4 table).
  The suffix refresh does not extend it here (L16: 21%).
* Order asymmetry: the FIRST operand must be written earlier than the second: P1 late (L16) + P2 early = 7%, but P1
  early + P2 late (L16) = 38%. The model reads the first summand's positions before the second's.
* Writer is not the bottleneck at L4-L12: donor 90.7/92.3 vs synth 81.3/84.7 (a 6-9 point writer gap here, larger
  than for one value; the two-span writer R^2 is 0.72-0.80).

Same on Qwen3-1.7B-Base (A100 node, n=300): none 15.3%, tool line 98.7%, text oracle 96.3% (corrupted: follow 96.3%).

| L (both spans) | 0 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| 1.7B both true: acc | 80.0 | 96.0 | 96.0 | 94.0 | 34.0 | 1.0 |
| corrupt P1 / P2 / both: follow | 76.7/74.3/77.7 | 93.3/95.7/94.0 | 94.7/95.3/95.0 | 90.7/94.7/90.7 | 35.7/32.0/31.0 | ~0 |
| mean-value control | 0 | 0 | 0 | 0 | 0.3 | 1.0 |

Mixed layers (1.7B), acc / follow-both: [4,12] 96.3/96.0, [12,4] 75.3/71.0, [0,16] 44.7/54.7, [16,0] 9.7.
Same order asymmetry as 0.6B: the first summand must be in place earlier than the second.

## Q2 natural word problem, template "lost" (`q2_word.py lost`, 0.6B, n=300; acc / follow-corrupted)
`Each box has {a} pens. There are {b} boxes. {c} pens are lost. How many pens are left?` (a 11-29, b 3-9, P=a*b
3-digit, c 10-99; answer P-c, immediately). No `(a*b)` span exists. Baselines: none **3.7%**, text tool line
(`Tool: a*b = P`, 4-shot with tool lines) **71.7%**.
Each location = (task span <- oracle span); the oracle is a rewritten problem that states P in text, and its
text-oracle accuracy is the ceiling for that location:

| location (task span) | oracle text | text oracle | L0 | L4 | L8 | L12 | L14 | L16 |
|---|---|---|---|---|---|---|---|---|
| both operand sentences `Each box has a pens. There are b boxes.` | `There are P pens.` | 95.0 / 93.3 | 96.0 / 95.7 | **96.7 / 97.7** | 94.3 / 93.0 | 88.0 / 87.3 | 89.3 / 86.7 | 67.7 / 69.0 |
| sentence with b `There are b boxes.` | `Each box has a pens. There are P pens.` | 54.3 / 47.0 | 27.0 / 27.3 | 35.7 / 33.3 | 44.0 / 35.0 | 42.7 / 33.3 | 26.0 / 20.0 | 16.7 / 15.0 |
| sentence with a `Each box has a pens.` | `There are P pens. There are b boxes.` | 15.3 / 19.3 | 15.3 / 19.3 | 16.7 / 19.3 | 11.0 / 12.3 | 9.3 / 9.7 | 35.3 / 35.7 | 21.7 / 21.7 |
| number tokens of a only (` 24`) | (same as above) | 15.3 / 19.3 | 2.7 / 1.7 | 1.7 / 1.0 | 3.3 / 1.3 | 5.0 / 2.3 | 11.7 / 8.7 | 10.7 / 6.3 |

Mean-value control: 0-1% everywhere. Donor (real oracle states) = synth within 1-5 points (both: L4 95.3, L12 89.0).

* **Where to write:** over the whole span that DEFINES the quantity — both operand sentences, overwritten by the
  states of one sentence that states the total. Then 3.7% -> 96.7% (L4), corrupted P' followed 97.7%, vs 71.7% for
  the text tool line at extra token cost and 95.0% for rewriting the problem in text. The latent write matches or
  beats the best text rewrite at L0-L8 and stays at 88-89% at L12-L14.
* Writing only the number (` 24` <- digits of P) fails (<= 12%): the rest of the text still says "per box" and
  "8 boxes", and the model multiplies again / gets confused. Writing one sentence is capped by how well the
  model reads the corresponding half-rewritten text (15%, 54%) and reaches 27-44% of it. The value has to replace
  the *relation*, not just a number.
* The window is later than in the `(a*b)+c` prompt (L14: 89% vs 65%): there are ~10 tokens between the span and
  the question, so the model reads the span later (as in latent_write E4 "sentence context"). Dead from L18
  (both: 3.3 / 2.3 / 3.3% at L18 / L20 / L22; number-only and first-sentence: 3-4% at L18-L20).

## Q3a boolean value (`q3_types.py bool`, 0.6B, A100 node, n=300)
`What is not ((x o y) o (z o w))?` with x..w in {True, False}, o in {and, or}; value = truth of the inner expression,
written at its span (9 tokens) from the oracle `What is not (v)?` (1 token + fill). Writer feature = one-hot(v).
Baselines: none **51.3%** (chance; the model alone does `(..) o (..)` 90% but not `not (...)` of it), text oracle
100%, text tool line 100%. (Writer R^2 = 1.0: the oracle varies only in v, so synth = donor.)

| L | 0 | 4 | 8 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|
| acc | 100 | 100 | 100 | 81.7 | 67.0 | 60.3 | 72.0 | 64.0 |
| corrupted: follow | 100 | 100 | 100 | 79.7 | 67.3 | 67.7 | 57.7 | 52.3 |
| mean value (no info) | 50.3 | 49.7 | 49.7 | 49.7 | 49.7 | 49.7 | 53.0 | 51.3 |

A one-bit latent value written over a 9-token span is consumed perfectly (100% / 100% follow) up to L8; the
"mean" state sits exactly on the decision boundary (50%).

## Q1b nested chain: `What is ((a*b)+c)*d?` (`q1_multi.py chain`, 0.6B, n=300)
a, b 2-9 (P1 = a*b 2-digit), c 10-89 with P2 = P1+c in 20-99, d 3-9. Two intermediate values: P1 (span `a*b`) and
P2 (span `(a*b)+c`, which contains the P1 span). Candidates: true, (P1'+c)*d, P2'*d.
Baselines: none **4.0%**; text tool with both values (`a*b = P1, P1+c = P2`) **97.0%**; text tool with P1 only
**3.0%**; text oracle with P1 in place `((P1)+c)*d` **7.0%**; text oracle with P2 in place `(P2)*d` **99.7%**.

| L | 0 | 4 | 8 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|
| P1 at inner span: acc / follow P1' | 10.0 / 7.3 | 11.0 / 9.0 | 5.0 / 5.0 | 4.7 / 2.7 | 4.7 / 3.0 | 4.7 / 2.0 |
| P2 at outer span: acc / follow P2' | 97.3 / 97.7 | 95.3 / 96.7 | 93.3 / 96.7 | 74.3 / 69.7 | 45.7 / 37.7 | 20.3 / 14.3 |
| P2 at outer span + suffix refresh: acc / follow | - | - | - | 95.3 / 97.0 | 93.3 / 94.7 | 66.0 / 62.3 |
| mean-value control (max over cells) | 1.7 | 1.3 | 3.0 | 1.7 | 2.7 | 2.7 |

* A nested chain is only as good as the LAST intermediate the model receives: P1 alone (latent or text) leaves the
  model with `(P1+c)*d`, still two hops, and it fails exactly as it fails in text (latent 5-11% vs text 7%).
  P2 written over the whole outer span: 4% -> 97%, corrupted P2' followed 97%. So for chains, the tool must
  compute the chain up to the last step the model can do in one hop and write that value; writing earlier
  intermediates does not help a model that cannot compose the remaining steps.
* For P1 and P2 to both be "latent values in play", they must be independent (Q1a): there, 2 (and 3, Q1c) values
  written at separate spans are each consumed and each followed.

## Q2 second template "rows" (`q2_word.py rows`, 0.6B, n=300; acc / follow)
`A farmer has {b} rows of trees. Each row has {a} trees. He plants {c} more trees. How many trees does he have now?`
(operands in the other order; answer P+c). Baselines: none **6.0%**, text tool line **83.0%**.

| location (task span) | oracle text | text oracle | L0 | L4 | L8 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|
| both operand sentences | `A farmer has P trees.` | 93.0 / 93.0 | 92.0 / 93.0 | 93.7 / 94.7 | 92.3 / 94.3 | 85.0 / 85.0 | 89.3 / 84.7 | 77.3 / 70.7 | 16.0 / 24.0 | 5.0 / 5.7 |
| 2nd sentence `Each row has a trees.` | `A farmer has b rows of trees. A farmer has P trees.` | 96.0 / 96.3 | 96.0 / 96.3 | **96.7 / 97.0** | 94.0 / 92.0 | 85.7 / 85.7 | 83.7 / 82.3 | 82.0 / 78.3 | | |
| 1st sentence `A farmer has b rows of trees.` | `A farmer has P trees. Each row has a trees.` | 57.3 / 56.7 | 56.3 / 55.7 | 46.0 / 45.3 | 46.7 / 43.7 | 43.7 / 36.0 | 34.0 / 27.3 | 19.0 / 14.0 | | |
| digits of a only (` 24` in `Each row has 24 trees`) | (same as 2nd sentence) | 96.0 / 96.3 | **0.0 / 0.0** | 0.7 / 1.3 | 8.3 / 8.3 | **62.3 / 60.3** | **66.7 / 61.0** | **64.7 / 60.7** | 32.3 / 34.0 | 15.3 / 12.3 |

Mean-value control <= 1% everywhere (5% at L22 = none). Donor = synth within 0-5 points.

* Replacing the operand sentences (both, or the second one, i.e. the one that completes the product) with the
  states of a sentence stating the total works: 6% -> 93-97% (L0-L8), 82-89% still at L14-L16, corrupted
  value followed 93-97%; vs 83% for the text tool line. The window ends between L16 and L18 (77% -> 16%).
* **Number-only writes switch from useless to working with depth.** Writing only the digits of P into the slot of
  `a` gives 0% at L0-L4 (the text then literally says "each row has 192 trees", and the model multiplies again by
  the rows), but 62-67% at L12-L16 (follow 60-61%; L10 24.7%, L18 32.3%, L20 15.3%: an inverted U): by L12 the relation "per row x rows" has already been
  bound around the original number's positions, and the digit slots now act as the operand the adder reads.
  (In the "lost" template the same trend is weaker: 2-3% at L0-L8 -> 11-12% at L14-L16; first-sentence write
  9% at L12 -> 35% at L14.) So "where to write" depends on the layer: early, write over the whole defining
  relation; late, writing over the operand's digits is enough.

## Q1c three latent values: `What is (a*b)+(c*d)+(e*f)?` (`q1_multi.py sum3`, 0.6B, n=300)
Same ranges; candidates: true, each single corruption, all three corrupted. Baselines: none **3.7%**, text oracle
`(P1)+(P2)+(P3)` **70.0%** (adding three 3-digit numbers is itself hard for the 0.6B model; corrupted text followed
68.7%), text tool line **77.0%**.

| L (all three spans) | 0 | 4 | 8 | 12 | 14 |
|---|---|---|---|---|---|
| all true: acc | 34.0 | 53.0 | 60.0 | 47.3 | 8.7 |
| corrupt P1 / P2 / P3 only: follow | 42.3/36.0/35.3 | 60.3/56.3/53.0 | 61.0/60.0/60.7 | 50.0/50.3/50.0 | 8.3/7.7/10.0 |
| corrupt all three: follow | 36.0 | 56.7 | 62.7 | 46.0 | 10.7 |
| mean-value control | 0 | 0 | 0 | 0 | 0 |
| donor | - | 54.3 | - | 52.3 | - |

Mixed layers [4,8,12] (first span earliest): 47.7 / follow 47.0; reversed [12,8,4]: 25.0 / 26.3.
Only one of three written: 0-2%.

* Three latent values are each consumed: at L8 the output is the sum implied by whatever three values were written
  60-63% of the time, for any of the 4 corruption patterns, with <= 1% leakage to the other candidates. Relative to
  the model's own ceiling for adding three given numbers (70% in text) that is 86%.
* L0 (34%) is below the text oracle because the right-aligned fill duplicates the `)+(` separator token states at
  three places; from L4 on the fill is harmless (donor = synth).
* The window shrinks with the number of values: last good layer L12 for 1-2 values (94 / 85%), L12 at 47% for 3,
  and L14 collapses to 9% (vs 52% for 2 values, 65% for 1).

## Q3b-e other value types (0.6B, n=300 each; acc / follow-corrupted; `q3_types.py`, runs `q3_*`/`q3b_*`)

**Comparison result, consumed by a negation (`cmpN`)**: `What is not (a*b > c)?`, |a*b-c| <= 25. Value = truth of
`a*b > c` (1 bit) written over the 8-token comparison span from the oracle `What is not (v)?`.
Baselines: none 52.3% (chance), text tool line `Tool: a*b = P` **53.7%** (knowing P does not help the 0.6B model:
it cannot do compare-then-negate in one pass), text oracle 100%.

| L | 0 | 4 | 8 | 12 | 14 | 16-20 |
|---|---|---|---|---|---|---|
| cmpN acc / follow | 100 / 100 | 100 / 100 | 100 / 100 | **100 / 100** | 53.0 / 52.0 | 52-53 / 47-48 (= none) |
| mean bit (no info) | 47.3 | 52.7 | 52.7 | 52.7 | 52.7 | 52-53 |

A one-bit comparison result is consumed perfectly through L12 and then nothing at all from L14: a step, not a
slope (the single answer token reads the span between L12 and L14). Latent bit: 100%; text tool with the product:
53.7%.

**Numeric value feeding a comparison (`yn`)**: `Is a*b greater than c? Answer yes or no.` with the product written
at `a*b` (corrupted P' chosen on the other side of c). Baselines: none 50.7%, text tool line 95.7%, text oracle
99.7% (corrupted text: follow 96.7%).

| L | 0 | 4 | 8 | 12 | 14 | 16-20 |
|---|---|---|---|---|---|---|
| yn acc / follow | 98.7 / 96.0 | 80.3 / 77.7 | 68.0 / 68.7 | 64.3 / 60.7 | 49.3 / 51.0 | ~50 (chance) |
| donor acc | - | 91.3 | - | 68.3 | - | - |

The same digit-code write that is consumed at 94% at L12 by an ADDITION is consumed only at 64% at L12 by a
COMPARISON with a close threshold, and degrades from L4. The comparison is sensitive to the exact tens digit and
reads it early; the linear writer is also weaker here (L4: 80 vs donor 91).

**Boolean expression (`bool`)**: see Q3a above: 51% -> 100% / follow 100% for L <= 8, 82% at L12.

**Date offset (`wk`)**: `Today is S. What day of the week will it be N days from today?` (N 10-69), value
k = N mod 7 written at the N span, oracle `... k days from today?`. The 0.6B model cannot add k to a weekday even in
text: text oracle 25.7% (with small-offset shots; 12.7% with task-format shots), none 6.7%, tool line 22.3%. The
latent write reaches exactly that ceiling (L4 25.7%, corrupted followed 29.3% = text-corrupted 29.3%) and is gone by
L12. Uninformative beyond "latent = text" because the downstream step is beyond the model.

**Literal conditional (`cmp`, `cmpE`)**: `Condition: a*b > c. If the condition is true, say w1; otherwise say w2.`
The model follows such conditionals poorly even in text (oracle `Condition: true.` 71.3%; easy-comparison oracle
`If 5 is greater than 6, ...` 52.3%, chance). Latent bit (literal oracle): 66.0% at L0, 60-67% at L4-L14 (follow
60-65%), chance from L16; easy-comparison writer: chance (its writer sees only the truth bit, R^2 0.1-0.2: the easy
comparison's states are dominated by the digits, not the result). Negative, limited by the model's instruction
following, not by the write.
