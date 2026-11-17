# JUDGE_truth_1: truth propagation tool (`truth`, spec `truth_gen`)

Judge: independent reviewer. Phase 1 (code review + variants) and phase 2 (results + final verdict), 2026-11-17.
Files read: PROTOCOL.md, teamB/circuits_B.py (shared helpers + truth part; v1, then v3), micromachines/machine.py,
micromachines/hdl.py (function list), teamB/tools_ref.py (v1, then strict v3), teamB/gens.py, teamB/NOTEBOOK.md,
judge/summary_truth_1.txt and judge/results_truth_1.json (my variant runs). No BBH data opened. No model or circuit run
by me (I ran only the pure-Python reference to cross-check my hand results).
Variants: `judge/variants_truth_1.json` (10 host variants + 20 tool-alone probes).

## FINAL VERDICT: PASS (the tool), with required disclosures

The `truth` tool v3 is generic and exact, and it fails safe. The weak link is the fine-tuned host's translation of
unseen wording into a call. That weakness also appears on BBH-style wording, so it is not "the tool only works on its
benchmark's style". The details and conditions are in sections 4 and 5.

## 1. Findings (reward hacking / generality)

### 1.1 What the tool reads: generic call-grammar tokens only. OK.
* `R_SYMS` contains the digits `0`-`9`, ` true`, ` false`, ` says`, `;`, ` ` (space), the NAME `truth` and the four
  envelope tokens. Each symbol is one token id. `machine.make_reader` checks the reader on every vocabulary row.
* The tool reads no names, places or task words. People are integers 1..12. The host maps names to numbers, as it maps
  a word problem to `48 / 2` for the calculator. I found no closed list of BBH entities and no BBH template.
* Only tokens after the most recent `<tool_call>` count. v3 adds a COVERAGE check: the tokens inside the call must be
  exactly NAME + `;` separators + the tokens of the parsed clauses. Stray tokens therefore give the error response.

### 1.2 Python logic at inference: none. OK.
* `Micromachine.forward/step` is a linear reader, a window, a compiled ReLU stack 1, one hard attention head, a ReLU
  stack 2 and a linear write. The 12 propagation rounds are unrolled in the circuit. There is no data-dependent branch.
* Team statement (phase 2): `run_eval.py` / `run_variants.py` do not parse the call during generation. Python reads the
  finished text only for scoring. I accept this; I did not read those files.

### 1.3 Reference vs circuit (phase-1 defect): FIXED
In v1, `truth_ref` did no validation. It accepted duplicates, 0, 13, 100 and a trailing `;`, and it mis-parsed a comma.
In v3, `truth_ref` is a strict regex grammar with limits and an error on contradiction, and it uses the same update rule
as the circuit. In the results, all 11 bad probes are "" or silent, and the tool equals the strict reference on all 30
probes and on every host call.

### 1.4 Generality of the interface (phase-1 defect): FIXED in v3, with one remaining limit
* v3 treats `K says J c` as an undirected same (c = true) / opposite (c = false) edge. It allows any number of clauses
  per person, lists every person mentioned, and writes the error response on a contradiction. The natural call for
  "a known-honest person accuses someone" now works (`dup_natural`, and the host's own call on `dinner_downward`).
* Soundness, from my reading of the v3 circuit: after 12 rounds every person connected to a fact is known (at most 12
  people, so the path depth is at most 11). The circuit then checks every fact and every edge between known people.
  If no check fails, the values satisfy all the constraints. A same/opposite system has one solution per anchored
  component, so the tool reports that solution or an error. It never reports a wrong value.
* Remaining limits (scope, not defects): only atomic pairwise claims (no "A says B and C both lie"); at most 12 people.
  A contradiction is detected only in a component that has a fact. The liar paradox `1 says 1 false` gives `1 unknown`,
  not an error (see `self_reference`). That is acceptable: no value is claimed.

### 1.5 Generators and training data
* NAMES, PLACES and ACTS are the team's own lists. The canonical web_of_lies sentence forms come from the official
  3-shot prompt, which rule 2 allows. I found no leakage.
* The train data was chain-heavy (3000 strict `lies` chains). v3 changed `rumor` so that the fact can sit on any person.
  The host results on `lies_middle`, `lies_tree` and `dinner_downward`, and the knights OOS sets, show that the host
  does not only number people in chain order and read the last value.
* `ft_B.py` does not mask the `<tool_response>` span, on purpose. The host therefore also learns to write responses
  itself. This is acceptable because the team reports the "fine-tuned host without tools" ablation next to every
  with-tool number, and the tool's write (gain 1e4) overrides the host when the tool is attached.

## 2. Lead's question: does `truth` earn its place? Could Team C's `bool` do the same?

**Facts:**

| set | FT + tool | FT without tool | frozen base |
|---|---|---|---|
| DEV web_of_lies (seed 0, n=30, official prompt) | 100.0 | 100.0 | 66.7 |
| knights OOS-dev (seed 4, n=30, zero-shot, never trained) | 100.0 | 66.7 | - |
| knights held-out (seed 3, n=50, ft2) | 100.0 | 88.0 | - |

1. On BBH-style chains the tool adds nothing over the fine-tune: the DEV gain (66.7 to 100) is the fine-tune's.
   The team MUST report web_of_lies results with this attribution: the fine-tuned host without tools next to
   the host with tools, and no claim that the tool caused the web_of_lies gain.
2. The tool's measured value is outside the benchmark's style: +33.3 pp on knights OOS-dev and +12 pp on held-out
   knights. In those sets the statements come in random order and the question can ask about anyone. That is the
   setting where the host's own propagation fails. This is the protocol's intended use (rule 12): a tool earns its
   place by being useful outside the task, not by the BBH score.

**`bool` vs `truth`:**
* `bool` evaluates an expression that is already linearized. A single chain from a fact to the queried person can be
  written as `not not ... True` (no parentheses, about 12 tokens of `not` for 12 people, so inside bool's limits).
* To write that expression, the host must first do the graph work. It must find the path from a fact to the queried
  person through statements in any order, and decide the direction of each edge. That path-finding is exactly what the
  host gets wrong without the tool (knights 66.7 / 88.0 without the tool).
* `bool` cannot express these unless the host does the propagation itself:
  - trees;
  - several facts;
  - facts on a speaker;
  - the values of several people at once;
  - `unknown` (not determined);
  - contradiction detection.
* So the two tools compute different things. `truth` computes anchored same/opposite constraint propagation, which is
  2-colouring with anchors, a parity union-find. `bool` evaluates the final formula. They overlap only on a single
  chain that the host has already ordered.

**Answer: yes, `truth` earns its place**, on general usefulness and on its out-of-style gain, not on web_of_lies.
If the integration step must cut tools, `truth` is the cheapest one to cut for BBH, because the fine-tune alone covers
web_of_lies. It should still stay as a general tool for the reasons above.

## 3. USEFUL-GENERALLY (rule 13): YES

f(x), stated without the benchmark: given entities, some with a known boolean value, and pairwise "same as / opposite
of" relations in any order and in any number, return every entity's value. The value is `unknown` when the entity is
not connected to a known value. The tool returns an error when the relations contradict the known values.

Three settings outside BBH:
1. Testimony and source chains: witnesses in a trial, sources that vouch for or debunk other sources (fact-checking,
   rumour tracing), knights-and-knaves puzzles with anchored statements.
2. Same/opposite parity and 2-colouring: switches wired in the same or opposite state, "same team / other team"
   assignments, sign of a product of negations, a consistency check of pairwise constraints.
3. Nested negation in text ("it is not true that she denies that ..."), as a chain of same/opposite steps.

v3 serves these settings better than v1 (both directions, many clauses per person, contradiction check).
USEFUL-GENERALLY: **yes**.

## 4. Variants (rule 11) and results

Host = ft3 + all Team B tools (v3). Style `lies` used the official web_of_lies 3-shot prompt; style `other` was
zero-shot. "tool OK" means the tool's response equals the strict reference of the call the HOST wrote, or is the
correct error response.

| id | style | what it probes | target | host call (first) | tool wrote | tool OK | pred | answer OK |
|---|---|---|---|---|---|---|---|---|
| lies_synonyms | lies | unseen verbs / predicates / question form | No | `...; 4 says 3 false` (should be `true`: "not lying") | 1 true, 2 false, 3 false, 4 true | yes | Yes | no (host mistranslated a negation) |
| lies_shuffled | lies | statements out of order | No | `1 says 2 false; 3 says 2 true` (fact dropped) | all unknown | yes | No | yes, but not from the tool (host dropped the fact; the tool correctly said unknown) |
| lies_middle | lies | question about a middle person | Yes | correct | 1 false, 2 false, 3 true, 4 false, 5 false | yes | Yes | yes |
| lies_noise | lies | irrelevant sentences | No | `...; 5 owns 2 cats; ...` | "" (x5, same call retried) | yes (error) | none | no |
| lies_negation | lies | "denies", "it is not true that" | Yes | `2 denies 1 true; 3 says false 2 false` | "" (x9, same call retried) | yes (error) | none | no |
| lies_tree | lies | tree, not chain | No | correct | 1 true, 2 false, 3 true, 4 false | yes | No | yes |
| court_witnesses | other | courtroom testimony | Yes | `2 testified 1 false; ...` | "" (x2) | yes (error) | No | no |
| knights_quotes | other | quotes, forward reference, extra islander | No | `1 says 2 a knave; ...; 5 a fisherman` | "" (x8) | yes (error) | none | no |
| dinner_downward | other | known speaker accuses (both directions) | No | `1 true; 1 says 2 false; 2 says 3 true` (natural call) | 1 true, 2 false, 3 false | yes | No | yes |
| switches_parity | other | lamps same/opposite (stretch) | Yes | `1 on; 2 always same as 1; ...` | "" (x8) | yes (error) | none | no |

Summary:
* **Tool: 10/10 correct or fail-safe.** It was correct on all 5 well-formed host calls. It gave the empty error
  response on all 5 broken calls, and it never wrote a wrong value.
* **Host + tool: 4/10 correct answers.** Only 3 of them came through a correct tool call (`lies_shuffled` was right
  by the host's own guess after an all-unknown response). The host called the tool on 10/10.
* Every miss is the host's translation:
  - it copies story words into the call (`denies`, `testified`, `a knave`, `on`, `always same as`);
  - it includes an irrelevant sentence as a clause;
  - it flipped one negation (`lies_synonyms`).
  The failures hit `lies` style (3 of 6) as much as `other` style (3 of 4). This is a wording-robustness problem of
  the fine-tune, not a tool that works only on BBH style.
* Error recovery did not work: after an empty response the host re-sent the same broken call up to 9 times and gave
  no answer. ft3's recovery rows (generic corruptions) did not cover story-word copying.

Tool-alone probes (20). The tool equals the strict v3 reference on 30/30 calls, counting the 10 variant calls.

| id | phase-1 expect | tool output (v3) | OK? |
|---|---|---|---|
| refs_out_of_order | 1 false, 2 false, 3 true, 4 true | same | yes |
| undefined_ref | 1 true, 2 unknown, 3 unknown | 1 true, 2 unknown, 3 unknown, 7 unknown | yes (v3 lists every mentioned person; my expect was v1) |
| twelve_chain | 12 values | same | yes |
| twelve_reversed | 12 values | same | yes |
| sparse_two_digit_tree | 3 false, 5 true, 10 true, 12 true | same | yes |
| eleven_mixed_order | 11 values | same | yes |
| self_reference | 1 unknown, 2 unknown | same | yes |
| two_cycle | 1 unknown, 2 unknown, 3 true | same | yes |
| single_clause | 5 false | same | yes |
| malformed_comma | "" | "" | yes |
| malformed_trailing_semi | "" | "" | yes |
| malformed_capital | "" | "" | yes |
| malformed_names | silent | silent | yes |
| malformed_double_space | "" | "" | yes |
| dup_defined_twice | "" | "" (now a contradiction: 1 = true, 2 = false, but 1 says 2 true) | yes |
| dup_natural | "" (v1) | 1 true, 2 false, 3 false | yes (v3 both-direction semantics; correct) |
| out_of_range_13 | "" | "" | yes |
| out_of_range_0 | "" | "" | yes |
| three_digit | "" | "" | yes |
| other_name_silent | silent | silent | yes |

20/20 correct under v3 semantics. The two differences from my phase-1 expectations are v1 vs v3 semantics, and I
checked both by hand.

## 5. Verdict, conditions, recommendations

**PASS for the `truth` tool (v3).** The reasons:
* The tool reads only generic grammar tokens and has no BBH vocabulary, template or Python logic.
* It is exact on every valid call and fails safe on every bad call (error response or silence).
* It is used and helps in a different setting than its benchmark: knights OOS +33 / +12 pp, and `dinner_downward`.
* USEFUL-GENERALLY is yes.

Conditions (binding for the use of this verdict):
1. **Attribution.** Report web_of_lies (DEV, FRESH, and the final BBH run) with the "fine-tuned host without tools"
   column. Say plainly that the web_of_lies gain is from the fine-tune; the DEV gain from the tool is 0.
2. **Report the host-translation weakness** with the out-of-style results: 4/10 on these judge variants (3/10 through
   the tool), with the failure kinds listed above. Do not report the knights 100.0 alone.
3. **No fitting to these variants.** These 10 variants are now seen data. If the team changes the generators or the
   fine-tune because of these failures, it must:
   - write the change in the NOTEBOOK as motivated by judge variants;
   - not copy my wording or names into training;
   - get a new judge run with new variants before claiming the fix.
   Generic fixes are fine: a wider synonym pool for "says/true/false", negation forms, distractor sentences, and
   recovery rows where the retry REWRITES the call instead of repeating it. Each must be written from the team's own
   generic word lists.

Recommendations (not required):
* Error recovery: after an empty response, never re-send an identical call; fall back to answering without the tool.
  Today the host loops and gives no answer. That is worse than the fine-tuned host alone.
* Optional: report a contradiction for an unanchored odd cycle (the liar paradox) too. It is not needed for correctness.

## 6. Technical explanations
* Window offsets: offset 0 is `</tool_call>`, and a larger offset is earlier text. A clause is found at its last word
  (` true`/` false` followed by `;` or `</tool_call>`), with 1 or 2 digits per number at fixed offsets before it.
* Validity (v3): `#clauses - #';' == 1`; every inside token belongs to NAME, a `;` or a parsed clause (coverage); no
  foreign token; every K and J in 1..12; no contradiction.
* Propagation (v3): edges are same (ET) or opposite (EF), symmetric. `known_k`, `val_k` start from the facts. Each of
  12 rounds sets `known_k = fact_k OR any edge to a known j`, and `val_k = factTrue_k OR any (same edge AND val_j) OR
  (opposite edge AND known_j AND NOT val_j)`. After the rounds, every fact and every edge between known people is
  checked. Any mismatch gives the error response, so a reported value always satisfies all the constraints.
