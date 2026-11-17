# JUDGE walk #1 (tool `walk`, spec `turtle_gen`): FINAL

Judge: independent reviewer (protocol rules 10-13).
Phase 1 read: PROTOCOL.md, circuits_B.py (shared helpers + turtle part), micromachines/machine.py (hdl.py names only),
tools_ref.py, gens.py, NOTEBOOK.md.
Phase 2 re-read: circuits_B.py (helpers + turtle, v3.1), tools_ref.py (turtle_ref, run_ref), NOTEBOOK.md log, and the
variant results judge/summary_walk_1b.txt + judge/results_walk_1b.json.
No BBH data, no BBH results, and no old outputs were opened. The judge did not run a model or circuit. The only code run was
tools_ref.turtle_ref, to check the hand-computed expected results.

**Final verdict: PASS (with conditions, section 5).**

## 1. Findings

### 1a. Reward hacking / BBH fitting: none found
* Call alphabet: digits, ' ', ',', `walk`, ' turn', ' left', ' right', ' around', ' forward', ' backward', and the
  envelope tokens. Output alphabet: digits, '(', '(-', ',', ' ', ' -', ')', ' facing', ' north/west/south/east', and the
  response tags. This is a generic turtle (dead-reckoning) vocabulary. The tool reads no names, no "step(s)", no "Take",
  no "Always face forward", and no question wording.
* The tool reads only the tokens between the most recent `<tool_call>` and `</tool_call>`, which the host writes. It never
  reads the question.
* No Python logic at inference. machine.py does: linear reader + clamp -> window -> compiled ReLU stack 1 -> hard
  attention to the most recent `</tool_call>` -> ReLU stack 2 -> write into the residual stream. No step branches on the
  data.
* Semantics are standard turtle geometry: start at (0, 0) facing north (+y). 'left N' and 'right N' are sidesteps
  relative to the heading. This is also how absolute moves can be written (with the heading kept north: up = forward,
  right = right, and so on), so it is not a benchmark quirk.
* The navigate generator copies the public BBH surface form, which rules 1 and 2 allow. Because of this, DEV says little
  about generality. Sections 3-4 cover generality.
* Fix history. v2 (foreign token -> error) came from the team's own fuzz test. v3 (coverage check, window 128, strict
  reference) came from this judge's phase-1 report. v3.1 (every `walk...` call is ours, so `walk up 3` gets the error
  response instead of silence) came from a run of this judge's variants. The NOTEBOOK discloses all of these. Each is a
  generic robustness fix: none adds vocabulary and none is tied to BBH wording.

### 1b. Phase-1 defects: all fixed and verified
1. Missing comma or doubled word gave a wrong non-empty answer. v3 adds a coverage check:
   `#tokens inside = 1 (NAME) + #commas + 2 per turn + 3 per 1-digit move + 4 per 2-digit move`.
   The judge read it in circuits_B.py and the formula is right. Probes p10-p12 now give the empty response.
   Team fuzz on v3.1: 414/414 malformed own calls give the empty response, with 0 wrong answers and 0 silences.
2. The window was shorter than the grammar. K is now 128. p13 (22 two-digit commands) and p14 (23) both answer
   correctly. 24 commands is about 120 tokens, so they also fit.
3. Spec drift is fixed. The NOTEBOOK grammar now reads `walk CMD, ...; <= 24 commands, N 0..99`. turtle_ref is a strict
   regex grammar and returns '' on any violation. run_ref matches the name the same way the circuit does.
* Remaining minor point (no action needed): with `nxt=None`, any call whose first token is `walk` belongs to this tool.
  Only the exact token `walk` matches. A word such as `walking` is a different token, so it stays silent. Other tools'
  calls stay silent: 136/136.

### 1c. Generality of the interface (rule 12)
The interface is a small command language, in the same way as the calculator's expression language. It accepts 90- and
180-degree turns plus integer moves (0..99) in four relative directions, up to 24 commands. It returns the final position
and heading. Other start headings can be written with a leading turn, and absolute moves can be written as sidesteps.
For a 2-D accumulator this is generic enough.

The variants show two natural forms that the host tried and the grammar does not accept: absolute moves
(`up 3`, `down 7`) and degree turns (`turn right 90`). Section 5 recommends them as optional, generic extensions. They are
not needed for PASS.

## 2. Rule 13: is f(x) generally useful?
f(x): given a list of relative motion commands on a grid (turn left/right/around; move forward/backward/left/right by an
integer), starting at the origin facing north, return the final (x, y) position and the final compass heading. This is
dead reckoning, or turtle geometry.

Three settings outside the benchmark:
1. Robot, drone, or delivery-route questions: where is the vehicle now, how far is it from base, which way is it facing.
   Shown: the drone variant was correct through the tool. Robot OOS-dev: 40.0 with tools vs 0.0 without. Held-out robot
   (ft2): 58.0 vs 10.0.
2. Grid games and code-like text: arrow-key sequences, turtle-graphics or LOGO programs ("does the pen return to its
   start").
3. Everyday directions: hiking, orienteering, or street directions, to find where you end up or whether you are back at
   the start.

**USEFUL-GENERALLY: YES.**

## 3. Variants (rule 11) and results
File: `judge/variants_walk_1.json` (27 entries). Run with host ft3 (fine-tuned Qwen3-0.6B-Base, all Team B tools
attached) and tool walk v3.1. Navigate style uses the official navigate 3-shot prompt. "Other" uses zero-shot.
Expected results were computed by hand from the docstring and checked with turtle_ref.

### 3a. Host + tools (12 questions)
| id | what differs from the generator | target | host's call (summary) | tool response | tool == strict ref | answer |
|---|---|---|---|---|---|---|
| nav1-spelled-numbers | number words, "paces ahead", "Make a right turn" | Yes | right | (0, 0) facing west | yes | **OK** |
| nav2-twodigit-noise-ffparaphrase | 2-digit numbers, ff paraphrase, irrelevant sentence | No | right | (2, 0) facing north | yes | **OK** |
| nav3-turns-mixed-with-strafes | turns and sidesteps mixed | Yes | right | (0, 0) facing east | yes | **OK** |
| nav4-then-chain-uturn-words | one "then" sentence, U-turns, number words | Yes | right | (0, 0) facing north | yes | **OK** |
| nav5-long-16-commands | 16 commands | No | wrong: "Take 4 steps backward" -> `forward 4` | (0, 0) facing north | yes | miss (host) |
| nav6-story-question-last-noise | story frame, noise sentence | Yes | wrong: noise put in the call (`dog bark`) | empty, 5 times | yes | miss (host retry loop) |
| oos1-drone | drone, meters | (8, -5) | right | (8, -5) facing south | yes | **OK** |
| oos2-warehouse-robot-starts-east | start heading east | (6, -6) | wrong: no leading turn | (6, 6) facing west | yes | miss (host) |
| oos3-game-absolute-arrows | absolute up/down arrows | (3, -2) | invalid: `up 3 ... down 7` | empty, 11 times | yes | miss (host retry loop) |
| oos4-courier-irrelevant-number | irrelevant number, courier | (14, -4) | wrong: put the "3 parcels" number in, dropped 12 | (7, -6) facing east | yes | miss (host) |
| oos5-sailboat-heading-question | asks for the heading | north | right | (-4, 0) facing north | yes | miss (host reported the position) |
| oos6-turtle-graphics-yesno | "rotates right 90 degrees", Yes/No | Yes | invalid: `turn right 90` | empty, 2 times | yes | "OK", but the host guessed after the error; the tool gave no help |

Summary:
* The host called the tool on 12/12 questions.
* The tool's response equals the strict reference of the host's call on 12/12: on every call it either answered
  correctly or wrote the empty error response.
* The host wrote a semantically correct call on 6/12: nav1-4, oos1, oos5.
* Correct answers: 6/12 as the team scores them. Correct through the tool's answer: 5/12, because oos6's "Yes" is a guess
  the host made after an error response.
* By style:
  * navigate-style variants: 4/6.
  * other settings: 1/6 through the tool (oos1), 2/6 as scored (oos6 by a guess). Semantically correct calls in other
    settings: 2/6 (oos1, oos5).
* Every miss is in the host: translation errors (nav5, oos2, oos4), noise copied into the call (nav6), forms outside the
  grammar (oos3, oos6), or ignoring the heading it was given (oos5). The tool never gave a wrong answer.

### 3b. Tool alone (15 probes + the 12 reference calls)
26/27 matched the strict reference as first reported. The one exception, p15 (`walk: forward 3`), gave the empty
response, which is what the judge expected. Its "mismatch" came from the old run_ref name split, which is now fixed.
So 27/27 match the judge's expected results.
* p1 (N = 99/98/97): (2, 1) facing north.
* p2 (strafes after turns): (-9, 3) facing north.
* p3 (exactly 24 commands): (2, 0) facing south.
* p9 (N = 0): (0, 0) facing south.
* p13 / p14 (22 / 23 two-digit commands): (0, 220) / (0, 230) facing north.
* Empty error response on all of: p4 (25 commands), p5 (number word), p6 (3 digits), p7 (negative), p8 (capitalized),
  p10-p12 (missing comma, doubled word), p15 (colon).

## 4. Judgement
* **The tool itself is exact and fails safe.** On all 39 runs (27 tool-alone + 12 host calls) it either gave the
  reference answer or wrote the empty error response. It never gave a wrong answer. This meets the rule 11 bar (fire
  correctly or fail safe on most variants) by a wide margin.
* **The tool works outside its benchmark style (rule 12).** It produced the right answer on a drone question that is not
  in any training style. It also gives large gains on the robot OOS sets (+40 pp OOS-dev, +48 pp held-out with ft2). It
  is not a tool that "only works on its benchmark's question style".
* **The weak link is the fine-tuned host's translation from text to call, not the tool.** In other settings the host wrote
  a correct call on only 2/6 variants. This must be reported honestly next to the DEV numbers (section 5). It is not a
  reason to FAIL the tool, because every wrong outcome came from the call the host wrote, and the tool handled each such
  call correctly.
* On DEV navigate the tool adds 6.7 pp over the same fine-tuned host without tools (100.0 vs 93.3). Most of the gain over
  the frozen base (53.3) comes from the fine-tune. Report both columns.

## 5. Verdict
**PASS: `walk` (turtle_gen v3.1) may be used.**

Conditions (binding for the report):
1. Report this judge's variant result as 6/12 as scored, and note that it is 5/12 through the tool's answer (oos6 was
   correct by a guess after an error response). Split it: navigate-style 4/6, other settings 1/6 through the tool.
2. Report navigate DEV/FRESH with three columns: frozen base, fine-tuned host without tools, fine-tuned host with tools.
   The fine-tune causes most of the gain.
3. These 27 variants are now seen data. Do not motivate any later fix by them without disclosing it in the NOTEBOOK and
   running a new judge with new variants.

Recommendations (not required; any of them is a substantial change and needs a re-judge):
* Host: an error response leads to retry loops (up to 11 identical calls). Cap the retries in the harness, or train
  "after a second error, stop and answer without the tool". Also never answer as if the tool had answered when it gave
  the empty response; the host did this on oos6.
* Host training data (generic, not from these variants' wording): start headings other than north, questions that ask for
  the heading, irrelevant sentences and numbers in the story, and number words.
* Grammar: absolute moves (`north/east/south/west N` or `up/down N`) and `turn left|right 90|180` are generic extensions
  that the host tried by itself. They would widen the interface toward grid games and turtle graphics.
