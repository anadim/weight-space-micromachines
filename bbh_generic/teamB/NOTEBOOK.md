# Team B notebook: generic INTERVAL, TURTLE (2-D position/heading) and TRUTH-PROPAGATION tools

Tasks: temporal_sequences, navigate, web_of_lies. Host: Qwen3-0.6B-Base. Runs: one H100 (GPU 1), from a copy of this folder.
Code: bbh_generic/teamB/. Binding: ../PROTOCOL.md (rules 1-13).

## 0. Ground rules followed
* No BBH test data opened. Only the three official cot-prompt files were fetched (pinned commit 9ee07bd, `cot/`).
* Old circuits read only for machinery (hdl.py, machine.py, the window/attention/emitter pattern). No old per-question outputs.
* Route chosen: protocol 9(b). Fine-tune the host to WRITE `<tool_call>ARGS</tool_call>`; the tool (compiled ReLU circuit,
  attached with the repo's `machine.attach`) writes `<tool_response>RESULT</tool_response>`. Training data = our own
  generators, seed 2, never BBH/dev/test.

## 1. Is f(x) generally useful? (rule 13, stated before building)
| tool | function, benchmark-free | 3 settings outside the benchmark |
|---|---|---|
| `interval_gen` | free sub-intervals of a window minus a set of busy intervals (union, any order, overlaps allowed); hours with am/pm or plain integers 0..99 | finding a common meeting slot across calendars; free hours of a bookable room/lab; free days of a rental (days as integers) |
| `turtle_gen` | dead reckoning: start (0,0) facing north, apply turns (left/right/around) and moves (forward/backward/strafe left/right N) -> final (x, y) and heading | robot / delivery-route final position; hiking or orienteering directions; game-character / drone command replay |
| `truth_gen` | knights-and-knaves truth propagation: clauses `K true/false`, `K says J true/false` (any order, refs to any clause) -> truth value of every K (true/false/unknown) | knights-and-knaves logic puzzles; chains of testimonies / rumours (who is lying); parity of a chain of "same/opposite" relations (x_K = x_J or not x_J) |
All three: yes, generally useful. Interfaces chosen to cover those settings (plain integers as well as am/pm; strafe moves as
well as turns; references in any order, not only a chain; 'unknown' when undetermined).

## 2. Call grammar (shared envelope, rule 14; full spec in the tools_ref.py docstring)
`<tool_call>NAME ARGS</tool_call><tool_response>RESULT</tool_response>`; empty RESULT = error; other NAME -> silent.
* `interval A-B busy C-D, E-F, ...` -> `X-Y, Z-W` (<= 4 gaps, else error) or `none`. Hours `7am`/`12pm` (12am = 0 at a start,
  24 at an end) or plain integers 0..99. Whole units.
* `walk turn left, forward 7, backward 12, left 3, right 5, turn around` -> `(X, Y) facing north|west|south|east`; <= 24 commands, N 0..99.
* `truth 1 true; 2 says 1 true; 3 says 2 false` -> `1 true, 2 true, 3 false` (k <= 12; 'K says J c' = same/opposite
  constraint, used in both directions; several clauses per person; contradiction -> error).
Generic word lists (and why): am/pm (clock notation); turn/left/right/around/forward/backward (the turtle's command words);
says/true/false (the clause words); `busy` (separates the window from the busy list). No names, places, activities, or BBH
phrasing are read by any tool. Every call must fit in the tool's 128-token window, else the tool is silent.

## 3. Machinery (same as the calculator, scaled to a whole argument list)
Window of the last K tokens (K = 128 for all three). Stack 1 parses the call at `</tool_call>` and computes the whole answer as
J one-hot token slots; one attention hop copies that state from the most recent `</tool_call>`; a shared emitter (stack 2)
writes slot j when `</tool_call>` is j tokens back. The tool fires only if the token after `<tool_call>` is its NAME and the
next token belongs to its grammar. Error (empty RESULT): a token outside the tool's alphabet, items vs separators mismatch,
not every token covered by a parsed item (coverage count), a number out of range, more than 4 gaps / 24 commands, or (truth)
a contradiction.
Sizes: interval stack1 81,886 ReLU units / 1.14M nnz; turtle 14,561 / 1.61M; truth 41,559 / 4.52M; emitters < 1.2k units.

## 4. Generators (gens.py)
Own name list (98 multicultural first names), own places (30) and activities (30). Several phrasings per sentence type;
fresh test (seed 1) additionally draws from EXTRA phrasing pools (X_*) never used in train/dev, plus (temporal) shuffled
sighting order and an irrelevant sentence. Difficulty: temporal 3-6 sightings, hours 5am-11pm; navigate 2-6 moves (+turns),
1-10 steps, 40% 'always face forward' style, ~50% Yes by construction; lies 3-7 people (mode 5).
Out-of-style EVAL sets (rule 12; never trained): meeting (common slot across 2-4 calendars, overlapping meetings),
robot (delivery robot, U-turns, final (east, north) position), knights (knights/knaves quotes, statements in random order,
asks about anyone). Train-only auxiliary styles (so the host learns the tools are general): lab (plain 24-h integers,
several gaps), hiker (final position), rumor (honest/liar, any order).

## 5. Log
### 2026-11-17
* Circuits v1 built. Exactness test (test_tools.py, teacher-forced over the token sequence, writes must equal the reference
  response exactly and be zero elsewhere): 280/280 per tool on first run.
* Fuzz (1500 random calls per tool, wider ranges) + malformed calls: truth/interval OK; turtle wrote an answer for
  `walk: right 74 and forward 38, ...` (one ',' replaced by ' and'): the unreadable command and the missing comma cancel in
  the item-vs-separator count. FIX (v2, all tools): any token inside the call outside the tool's alphabet -> silent. Also
  stops tools answering each other's calls. Re-run: interval 1809/1809 exact, silent 567/567; turtle 1843/1843, 562/562;
  truth 1829/1829, 563/563 (these counts include the reference calls of dev, test, out-of-style and 300 train rows; only
  pass/fail of the tool on the generator's own reference calls was looked at, no host output on test).
* Plain host, official prompts, DEV (n=30 each): temporal 26.7, navigate 53.3, lies 66.7. Out-of-style zero-shot (n=50):
  meeting 0.0, robot 0.0, knights 34.0.
* Embedding rows of `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` are identical (untrained) in the base
  checkpoint -> the fine-tune re-initializes them with distinct random directions (needed for the reader too).
* ft1: full fine-tune, 1 epoch, lr 1e-5, 10,800 rows (3x3000 BBH-like + 3x600 aux), 80% behind the official prompt.
* Plain-host evals above are the frozen base model; OOS uses a zero-shot prompt (no official prompt exists), so 0.0 there
  mostly reflects that the base model has no answer format for those questions.
* Rule 14 (shared envelope) arrived during ft1: switched to `<tool_call>NAME ARGS</tool_call>` (names interval/walk/truth),
  own-name invalid call -> empty response, other names -> silent. Stricter name check after the test showed
  `interval of 3 numbers` / `truthful` got an empty response: NAME must be followed by a token of its grammar.
  ft1 discarded; ft2 = same recipe with the new envelope.
* ft2 results (seed 0 DEV, official prompts): with tools temporal 100.0, navigate 100.0, lies 100.0; fine-tuned host WITHOUT
  tools 100.0 / 86.7 / 100.0. OOS held-out (seed 3, n=50): with tools meeting 22.0, robot 58.0, knights 100.0; without
  tools 2.0 / 10.0 / 88.0. Decided: do not read seed-3 OOS failures; made an OOS-dev split (seed 4, 30 each) for that.
* Judges phase 1 (opus, independent): no reward hacking found in any tool. Defects reported:
  - walk: a command that fails to parse cancels against its missing comma (`walk left 3 turn left, forward 2` -> answer
    instead of error). FIX (v3, all three tools): COVERAGE check, the number of tokens inside the call must equal NAME +
    separators + the tokens of all parsed items; otherwise error.
  - walk: 24 two-digit commands do not fit the 112-token window -> window 128.
  - interval: ' busy' optional and stray time tokens ignored -> exactly one ' busy' required + coverage check.
  - references looser than circuits (no gap cap, no limits) -> tools_ref.py made strict (regex grammar, limits, errors).
  - truth: one clause per person, values only flow from the person spoken about to the speaker (`1 true; 1 says 2 false`
    was an error). Generality fix (rule 13): truth v3 = same/opposite constraints used in BOTH directions, any number of
    clauses per person, every mentioned person reported, contradiction -> error. Train-only `rumor` generator now puts the
    known fact on any person (values flow both ways).
  - NOTEBOOK grammar section was stale (`free`, `walk:`) -> fixed above.
  Re-test v3 (2000 fuzz calls/tool + 13 corruption kinds incl. missing comma, doubled word, stray token, dropped ' busy'):
  interval 1803/1803 exact, malformed 520 -> empty response, 0 wrong; walk 1822/1822, malformed 522 empty + 29 silent
  (NAME followed by a foreign word), 0 wrong; truth 1796/1796, 543 empty, 0 wrong; all silent on other tools' calls.
* ft2 + v3 tools on OOS-dev (seed 4): meeting 33.3, robot 43.3, knights 100.0. Read failures:
  - meeting: tool right, host wrote `11am-12pm` while target is `11am to 12pm` -> scoring normalization: a time range
    `X-Y` == `X to Y` (run_eval.norm; applied to every set and to both runs). Some calls dropped busy intervals (copying).
  - robot: host copies the story's verbs into the call (`reverse 4`, `turn U`) -> tool correctly gives the empty
    response, but the host then invents a position after it.
  Generic fixes (no robot/meeting templates added): (1) ERROR RECOVERY rows (15% of train rows): a first call broken by a
  generic corruption (synonym for a grammar word, dropped separator, ' to ' for '-') gets the empty response, then
  'That call was not valid, so I fix it.' and the correct call; the broken call is masked from the loss. (2) train-only hiker
  style had no backward moves and few verbs -> added backward moves and more verbs (none of the robot template's words),
  30% Yes/No questions; lab bookings 2-8 (longer busy lists). (3) aux styles 1000 rows each.
  DISCLOSURE: these were motivated by reading OOS-dev (seed 4) failures; the seed-3 OOS numbers after this are not
  untouched in the same way as fresh test.
* ft3: same recipe as ft2 + the changes above (12,000 rows, 1,792 with error recovery).
* ft3 results. DEV (seed 0, official prompts, n=30): with tools 100.0 / 100.0 / 100.0 (temporal / navigate / lies);
  fine-tuned host WITHOUT tools 100.0 / 93.3 / 100.0; frozen base 26.7 / 53.3 / 66.7.
  OOS-dev (seed 4, zero-shot, n=30): with tools meeting 93.3, robot 40.0, knights 100.0; without tools 20.0 / 0.0 / 66.7;
  frozen base 0.0 / 0.0 / 33.3.
  Robot OOS-dev misses read: the host still copies the story verb ('reverse 4' for 'reverses 4 blocks'); the tool gives the
  empty response and the host retries the same call. NOT fixed: adding 'reverse' to training would fit the OOS template.
* Judge phase 2 (variants of rule 11, run with ft3 + v3 tools; run_variants.py; results in judge/results_*_1*.json):
  - interval: host+tools 13/15 correct (T-style 6/7, other settings 7/8), called 15/15, tool == strict reference on every
    host call; tool alone 36/37 (P22, call longer than the window: silent, as allowed). Misses: 24-h 'HH:MM' times (outside
    the whole-unit grammar -> empty response, host retries); 'none' returned correctly but ignored by the host.
    JUDGE_interval_1.md: PASS. Non-blocking notes: ' busy' position not enforced (meaning cannot change: the window is
    always the first interval) -- documented, not changed; no 'none' rows in training (host limitation, not changed);
    retries repeat the same call.
  - truth: host+tools 4/10 (3 through a correct call), called 10/10; tool correct or the empty response on every call, tool
    alone 29/30 (+1 silent where the judge expected silence). Every miss: host translation (copies 'denies', 'testified',
    'a knave', 'on', 'always same as'; makes the irrelevant sentence a clause; one flipped negation). JUDGE_truth_1.md:
    PASS with conditions: (1) report web_of_lies with the fine-tuned-without-tools column (on DEV the tool adds 0; the
    gain is the fine-tune's); (2) report 4/10 on the judge's variants; (3) the judge's variants are now seen data, no fix
    may be motivated by them without disclosure + a new judge run. Lead's question (does truth earn its place vs Team C's
    `bool`?): judge says yes -- bool evaluates one formula the host has already ordered; truth does the propagation over
    statements in any order, trees, facts on a speaker, unknown and contradictions (knights OOS: +33.3 / +12 pp).
  - walk: phase-2 run showed 'walk up 3, ...' was SILENT (NAME followed by a non-grammar word) and the host then wrote a
    response itself -> v3.1: any call whose first token is 'walk' is ours (empty response on bad args). Re-test: 1433/1433
    exact, 414/414 malformed -> empty response, 136/136 other-tool calls silent. Variants rerun with v3.1: host+tools 6/12,
    called 12/12, tool == strict reference on every host call; tool alone 26/27 (the 1 was a run_ref name-split artifact,
    fixed: run_ref now matches names the way the tokenizer splits them).
* FREEZE: host ft3 + tools v3 (interval, truth) / v3.1 (walk). Fresh test (seed 1) and held-out OOS (seed 3) run once
  with that configuration; only scores are read.
* Frozen base on FRESH TEST (seed 1, n=100): temporal 15.0, navigate 62.0, lies 55.0.
* JUDGE_walk_1.md: PASS with conditions: report variants as 6/12 scored but 5/12 through the tool (navigate-style 4/6,
  other settings 1/6 through the tool); report navigate with base / FT-no-tools / FT+tools columns; variants are now seen data.
  All three tools PASS. Not done (recommended by judges, each would need a re-judge): retry cap / no repeated identical
  calls, 'none' and start-heading/heading-question training rows, absolute moves and 'turn left 90' in the walk grammar.

## 6. FINAL RESULTS (frozen: host ft3 = Qwen3-0.6B-Base fully fine-tuned on our seed-2 data; tools v3 / walk v3.1)
Official BBH 3-shot CoT prompt, unchanged, for the three BBH-style tasks; zero-shot for out-of-style sets. Greedy.

| set | frozen base | FT host, no tools | FT host + tools |
|---|---:|---:|---:|
| temporal DEV (seed 0, n=30) | 26.7 | 100.0 | 100.0 |
| navigate DEV | 53.3 | 93.3 | 100.0 |
| web_of_lies DEV | 66.7 | 100.0 | 100.0 |
| temporal FRESH TEST (seed 1, n=100) | 15.0 | 100.0 | 100.0 |
| navigate FRESH TEST | 62.0 | 88.0 | 89.0 |
| web_of_lies FRESH TEST | 55.0 | 100.0 | 100.0 |
| meeting OOS held-out (seed 3, n=50; interval) | 0.0 | 10.0 | 100.0 |
| robot OOS held-out (walk) | 0.0 | 8.0 | 68.0 |
| knights OOS held-out (truth) | 34.0 | 62.0 | 94.0 |
| meeting OOS-dev (seed 4, n=30) | 0.0 | 20.0 | 93.3 |
| robot OOS-dev | 0.0 | 0.0 | 40.0 |
| knights OOS-dev | 33.3 | 66.7 | 100.0 |

The host wrote a tool call on 100% of fresh-test and held-out OOS questions. On the BBH-style tasks the fine-tune, not
the tools, carries the gain (tools add +0 / +1 / +0 on fresh test); the tools matter where the computation is not a
copyable pattern (meeting +90, robot +60, knights +32 over the same fine-tuned host). Caveat: the OOS sets' style was never
trained on, but ft3's error-recovery rows, hiker verbs and scoring normalization were motivated by reading OOS-dev (seed 4)
failures; seed-3 OOS was only scored. Judge variants (rule 11): interval 13/15, truth 4/10 (3 via the tool), walk 6/12 (5
via the tool); every miss was the host's call, the tool was right or wrote the empty error response on every call.
