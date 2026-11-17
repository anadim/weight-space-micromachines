# JUDGE: `interval` (interval_gen), round 1 — FINAL: **PASS**

Judge: independent subagent, 2026-11-17.
Phase 1 read: PROTOCOL.md; teamB/circuits_B.py (shared helpers + interval part); micromachines/hdl.py; micromachines/machine.py;
teamB/tools_ref.py; teamB/gens.py; teamB/NOTEBOOK.md. Phase 2 re-read: circuits_B.py and tools_ref.py (v3), and the
variant results judge/summary_interval_1.txt with spot checks in judge/results_interval_1.json.
No BBH data, BBH results or old outputs were opened. I ran no models and no circuits. The only code I ran was
`tools_ref.run_ref` (plain Python reference), to cross-check my hand-computed expectations.

## 1. Findings (reward hacking / generality)

### 1a. What the tool reads (rules 3, 12): CLEAN
* Input alphabet (`I_SYMS`): the ten digits, `am`, `pm`, `-`, `,`, ` ` (space), ` busy`, the name `interval`, and the four
  envelope tokens. The reader (`machine.make_reader`) maps each symbol to exactly one host token id and is checked exactly
  over the whole vocabulary. No other token can light up a symbol.
* There is no name, place, activity, option letter or BBH phrase in the circuit. The only words are clock notation (am/pm),
  the tool name and one header word (` busy`). That is a generic list, like the calculator's operators.
* The tool never sees the question or the options. The host picks the option letter from the tool's result.
* Interface: `interval A-B busy C-D, E-F, ...` gives the maximal free sub-intervals of [A, B). Busy intervals can come in any
  order, overlap, and extend past the window. Times are am/pm or plain integers 0..99. The result is `none` when nothing is
  free. The host writes the arguments in text, as with the calculator. This is as generic as the calculator.

### 1b. Python at inference (rule 3): CLEAN
* `interval_stack1` is built only from ReLU gates (`relu`, `AND`, `ge`, `eq`, `sel`, `clip01` in hdl.py). `compile_net` turns it
  into weight matrices. `machine.Micromachine` runs these steps: linear reader, clamp, fixed window, ReLU stack 1,
  hard-attention copy from the most recent `</tool_call>`, ReLU stack 2, then one-hot times the write directions. There is no
  data-dependent branch, no regex, and no Python call to `interval_ref` at inference. `tools_ref.py` only makes training
  completions and test references.
* The fail-safes are in the circuit:
  * a call with another NAME: silent;
  * NAME correct but any problem in ARGS (a foreign token, an unmatched dash, an item/comma mismatch, not exactly one
    ` busy`, a token not covered by the grammar (v3), an hour out of range, more than 4 gaps): `<tool_response></tool_response>`;
  * `<tool_call>` outside the 128-token window: silent.

### 1c. Generators and data (rules 1, 2, 9b): no leakage found, one caveat
* The names (98), places (30) and activities (30) are the team's own lists. I found no BBH entity names in them.
* `temporal_item` copies BBH's surface structure: a wake time, "Y saw X ... from a to b", "the P was closed after t", exactly
  one free gap, and 4 options (the gap plus 3 busy segments). Rule 2 asks for this, and the official 3-shot prompt shows the
  same structure, so this is not leakage. But the host fine-tune has seen mostly one sentence shape. The variants tested
  whether the host still writes the right call when that shape changes. It does (section 3).
* The test split adds unseen paraphrase pools, shuffled order and a noise sentence. There is an out-of-style eval set that is
  never trained on (`meeting_item`) and a train-only style with plain integers (`lab_item`).

### 1d. Phase-1 defects, as of v3
1. NOTEBOOK grammar section was stale (`free A-B busy`): **fixed**. P16 (`free ...`) is silent, as it should be.
2. The reference had no 4-gap cap but the circuit did: **fixed**. The reference now returns '' for more than 4 gaps, and P06
   gives '' from both.
3. Lax parsing (stray time token; optional ` busy`): **fixed** by the v3 coverage check (the token count must equal NAME +
   ` busy` + commas + the tokens of each parsed interval) and by requiring exactly one ` busy`. P13 and P20 now give ''.
   I checked the coverage count by hand: LA1/LA2/LP1/LP2 = 4/5/3/4 tokens (digits + am/pm + '-' + leading space) and
   RA1/RA2/RP1/RP2 = 2/3/1/2. The arithmetic is right.
4. Residual (minor, not blocking): from reading the v3 code, the circuit counts ` busy` but does not check *where* it is.
   `interval 9am-5pm 1pm-2pm busy, 3pm-4pm` passes every count. I predict the circuit writes `9am-1pm, 2pm-3pm, 4pm-5pm`,
   while the strict reference gives ''. The meaning cannot change: the window is always the first interval. So this is
   cosmetic. To close it, require ` busy` directly after the first interval, or document it.
5. Interface limits (documented in the tools_ref docstring; all fail safe; none is BBH-specific):
   * whole units only (no `9:30`);
   * plain integers up to 99;
   * no window across midnight;
   * at least 1 busy interval;
   * at most 4 gaps;
   * the call must fit in 128 tokens (P22: silent).
   The whole-units limit is the one that matters most for real calendars (half-hours), and T6 shows it in practice.
   Recommended for a later version (not required here): minutes, or a range above 99.

### 1e. Host-side observations (not tool defects; for the team)
* **O5:** the tool wrote `none` correctly, and the host ignored it and answered `11am to 1pm`. In the phase-1 gens.py, no
  interval training style ever shows a `none` response: `lab_item` and `meeting_item` both resample until `resp != 'none'`,
  and `temporal_item` always has exactly one gap. Add `none` cases to train-only data.
* **T6:** the host copied `07:00` into the call, got the error response, and then wrote "That call was not valid, so I fix
  it." and repeated the same call 6 times. The error-recovery behavior is shallow: the host recognises the error but does not
  repair the call. Add train-only examples where HH:MM (or other notations) is converted to plain integers, and where the
  retry is actually different.
* **ft3 without tools reaches 100.0 on temporal DEV**, the same as ft3 + tools. So on the BBH-style set the fine-tune alone
  explains the score, and that set is not evidence for the tool. The tool's own value shows out of style: on MEETING
  (seed 4, zero-shot), ft3 + tools scores 93.3 and ft3 alone scores 20.0. Report both numbers side by side so nobody credits
  the temporal score to the tool.

## 2. USEFUL-GENERALLY (rule 13)
f(x): given a window [A, B) and a list of busy intervals (any order, may overlap, may leave the window), return the maximal
free sub-intervals of the window. This is the set difference of an interval and a union of intervals, on an integer grid.

Settings outside the benchmark where it helps:
1. Finding a common meeting slot across several people's calendars (O1; MEETING set 93.3 with the tool vs 20.0 without).
2. Booking a room, court or piece of equipment: which hours are still open (O2, O4, O6).
3. Rental or reservation availability by day number (cabin, hotel room, car) (O3).
It also helps with staff-coverage gaps (O8) and alibi gaps (O7).

**USEFUL-GENERALLY: YES.**

## 3. Variants (rule 11) and results
File: `judge/variants_interval_1.json` (37 entries). I computed the expectations by hand from the tools_ref grammar and
cross-checked them with `run_ref`. Results come from the team's run: host = ft3 (fine-tuned Qwen3-0.6B-Base) with all three
Team B tools attached. T-style variants were shown with the official temporal 3-shot prompt; O-style variants were zero-shot.
Raw outputs are in judge/results_interval_1.json; the summary is judge/summary_interval_1.txt.

### Host + tools (15 variants)
| id | style | what is different | target | host call (verbatim) | tool wrote | tool = ref | host answer | ok |
|---|---|---|---|---|---|---|---|---|
| T1 | temporal | reordered sightings, new verbs, names, places | (B) | interval 6am-4pm busy 1pm-4pm, 6am-8am, 11am-1pm, 8am-10am | 10am-11am | yes | (B) | yes |
| T2 | temporal | close sentence first, irrelevant sentence, passive voice | (A) | interval 11am-9pm busy 11am-2pm, 2pm-5pm, 6pm-9pm | 5pm-6pm | yes | (A) | yes |
| T3 | temporal | no "saw" verbs | (D) | interval 5am-3pm busy 5am-9am, 9am-11am, 1pm-3pm | 11am-1pm | yes | (D) | yes |
| T4 | temporal | one prose paragraph, no "We know that:" | (B) | interval 2pm-11pm busy 2pm-4pm, 4pm-7pm, 8pm-11pm | 7pm-8pm | yes | (B) | yes |
| T5 | temporal | 7 sightings shuffled, title + pronoun, gap first | (A) | interval 6am-10pm busy 2pm-5pm, 9am-10am, 7am-9am, 12pm-2pm, 10am-12pm, 5pm-7pm, 7pm-10pm | 6am-7am | yes | (A) | yes |
| T6 | temporal | 24-hour HH:MM | (B) | interval 07:00-18:00 busy 07:00-10:00, 12:00-15:00, 15:00-18:00 (x6, identical) | '' (error) | yes | (none) | **no** (host; tool failed safe) |
| T7 | temporal | overlapping sightings, gap last | (C) | interval 8am-8pm busy 8am-11am, 10am-1pm, 1pm-5pm, 5pm-7pm | 7pm-8pm | yes | (C) | yes |
| O1 | other | 2-person meeting, overlaps | 12pm to 1pm | interval 9am-5pm busy 9am-11am, 1pm-3pm, 10am-12pm, 2pm-5pm | 12pm-1pm | yes | 12pm to 1pm | yes |
| O2 | other | room booking, 24-h integers, 3 gaps | 8 to 9 and 11 to 12 and 16 to 18 | interval 8-20 busy 9-11, 12-14, 13-16, 18-20 | 8-9, 11-12, 16-18 | yes | 8 to 9 and 11 to 12 and 16 to 18 | yes |
| O3 | other | cabin rental by day numbers | 1 to 3 and 8 to 12 and 25 to 30 | interval 1-30 busy 3-8, 18-25, 12-20 | 1-3, 8-12, 25-30 | yes | 1-3, 8-12, 25-30 | yes (listed alternative) |
| O4 | other | dentist, overlaps, 2 gaps, irrelevant sentence | 12pm to 1pm and 4pm to 5pm | interval 8am-6pm busy 8am-10am, 9am-12pm, 1pm-4pm, 5pm-6pm | 12pm-1pm, 4pm-5pm | yes | 12pm to 1pm and 4pm to 5pm | yes |
| O5 | other | no free time | none | interval 9am-1pm busy 9am-11am, 10am-1pm | none | yes | 11am to 1pm | **no** (host ignored a correct `none`) |
| O6 | other | evening until 12am | 8pm to 9pm | interval 6pm-12am busy 6pm-8pm, 9pm-12am | 8pm-9pm | yes | 8pm to 9pm | yes |
| O7 | other | detective alibi, free response | 1pm to 2pm | interval 7am-7pm busy 10am-1pm, 2pm-7pm, 7am-10am | 1pm-2pm | yes | 1pm-2pm | yes (X-Y = X to Y) |
| O8 | other | nurse coverage (busy = covered shifts) | 6pm to 7pm | interval 6am-10pm busy 6am-2pm, 12pm-6pm, 7pm-10pm | 6pm-7pm | yes | 6pm to 7pm | yes |

Scores:
* **Host called the tool:** 15/15.
* **Host wrote a well-formed call with the right arguments:** 14/15. T6 broke the grammar.
* **Tool response matched the strict reference:** 15/15 of host calls. The tool answered correctly on 14 and failed safe
  (empty response) on T6.
* **Final answer correct:** 13/15. Temporal style 6/7, other settings 7/8.
* **Both misses are host errors.** The tool was right or failed safe in both.

### Tool alone (22 probes)
| id | call | expect | tool wrote | ok |
|---|---|---|---|---|
| P01 | interval 8am-6pm busy 3pm-5pm, 9am-11am, 10am-12pm, 4pm-6pm | 8am-9am, 12pm-3pm | same | yes |
| P02 | interval 12am-12pm busy 12am-6am, 11am-12pm | 6am-11am | same | yes |
| P03 | interval 6pm-12am busy 7pm-9pm, 10pm-11pm | 6pm-7pm, 9pm-10pm, 11pm-12am | same | yes |
| P04 | interval 1-30 busy 3-8, 12-20, 18-25 | 1-3, 8-12, 25-30 | same | yes |
| P05 | interval 9am-1pm busy 9am-11am, 10am-1pm | none | none | yes |
| P06 | interval 1-20 busy 2-3, 4-5, 6-7, 8-9, 10-11 (6 gaps) | '' | '' | yes |
| P07 | interval 0-24 busy 3-5, 8-10, 12-20 (4 gaps) | 0-3, 5-8, 10-12, 20-24 | same | yes |
| P08 | interval 9am to 5pm busy 1pm-2pm | '' | '' | yes |
| P09 | interval 9:30am-5pm busy 1pm-2pm | '' | '' | yes |
| P10 | interval 9am-5pm busy 7am-10am, 4pm-8pm | 10am-4pm | same | yes |
| P11 | interval 9-17 busy 1pm-2pm (mixed notation) | 9-13, 14-17 | same | yes |
| P12 | interval 9am-5pm busy (empty list) | '' | '' | yes |
| P13 | interval 9am-5pm busy 1pm-2pm 3pm, 4pm-5pm (stray token) | '' | '' | yes (v3 fix) |
| P14 | interval 5pm-9am busy 1pm-2pm | none | none | yes |
| P15 | interval 9am-13pm busy 10am-11am | '' | '' | yes |
| P16 | free 9am-5pm busy 1pm-2pm (other NAME) | SILENT | silent | yes |
| P17 | interval 10-40 busy 15-35, 20-25, 20-25 | 10-15, 35-40 | same | yes |
| P18 | interval 90-99 busy 95-97 | 90-95, 97-99 | same | yes |
| P19 | interval 0-100 busy 5-6 | '' | '' | yes |
| P20 | interval 9am-5pm 1pm-2pm (no ` busy`) | '' | '' | yes (v3 fix) |
| P21 | interval 12pm-6pm busy 1pm-3pm | 12pm-1pm, 3pm-6pm | same | yes |
| P22 | 16 busy intervals, call longer than 128 tokens | 2pm-3pm (grammar) | silent | fail-safe (predicted and allowed) |

Tool alone: 21/22 exact (36/37 over all 37 calls, counting the 15 host-variant reference calls). The one non-match, P22, is a
documented fail-safe silence. The team also reports a v3 fuzz run: 1803/1803 valid calls exact, 520/520 malformed calls give
the empty response, and 204/204 calls to other tools are silent.

## 4. Verdict: **PASS**
Reasons:
* No closed vocabulary, no BBH strings or templates in the tool, no Python at inference, no BBH data used.
* The interface is calculator-like and serves settings well outside the benchmark: meeting calendars, room booking, day-number
  rentals, coverage gaps and alibis (USEFUL-GENERALLY: yes).
* Rule 11: on 15 reworded or out-of-setting inputs, the host called the tool every time, and the tool's response matched the
  reference on every call (14 correct answers, 1 correct error). The final answer was right on 13/15, including 7/8 in
  settings that are not temporal_sequences, and both misses are host errors. Rule 12 is met: the tool is not tied to its
  benchmark's question style.
* All three blocking phase-1 defects are fixed in v3, and the probes confirm it.

Non-blocking items to carry forward:
1. ` busy` position is not enforced (1d.4). Cosmetic; enforce it or document it.
2. Add `none` responses to train-only interval data (O5).
3. Add notation-conversion and real-repair retry examples to train-only data (T6). A shallow retry loop that repeats the
   same bad call 6 times wastes tokens and should be capped or taught properly.
4. When reporting, show ft3-without-tools next to ft3+tools. The temporal DEV score (100 vs 100) does not credit the tool;
   the MEETING score (93.3 vs 20.0) does.
5. Possible extension: minute resolution or integers above 99, for real calendars.
