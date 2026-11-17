# JUDGE: track_gen (generic state tracker), judge 1 — FINAL VERDICT

Judge: an independent subagent with no access to BBH data. Date: 2026-11-17.

## What I read

- PROTOCOL.md
- The tool and the call format: track_gen.py, circ_common.py, callfmt.py, register.py
- The repo machinery (micromachines/machine.py), excerpts only
- The generators: gen_track.py, vocab.py, make_data.py, gen_mix.py, gen_oos2.py, gen_oos3.py (the last three only to avoid overlapping with them)
- Tests and scripts: test_tools.py, ft.py, run_eval.py
- NOTEBOOK.md and the official tracking 3-shot prompt
- Team A's result files: judge/results_v3_rounds12.md, judge/results_ft100a.md, judge/final_tables_ft100a.md

I opened no BBH task data and no old outputs.

## VERDICT: PASS, with caveats on the host's call writing (details below)

## Rule 13: f(x) and USEFUL-GENERALLY

**f(x).** The tool takes an array of up to 9 one-digit values and applies up to 20 operations, left to right. The operations are:

- `ij`: swap slots i and j
- `i=v`: set slot i to v
- `i<j`: copy slot j into slot i

It then returns one of three things:

- the final value of slot k (`?k`)
- the unique slot holding value v (`@v`); the reply is empty if there is not exactly one
- the whole final array (no query)

Malformed or out-of-range calls get an empty reply.

**USEFUL-GENERALLY: YES.** Three settings outside the benchmark:

1. Card games, inventories and asset assignment: who holds what after trades, replacements and discards.
2. Program reasoning: variables or array cells through swaps, assignments and copies (a register file).
3. Logistics and rostering: which vehicle is at which dock, which team owns which area, who covers which shift after swaps and reassignments.

Shell games, seat swaps and rearranging words in a phrase also fit.

The limits are real but are not specific to the benchmark: one value per slot, at most 9 slots, at most 10 distinct values, and no one-way transfers or multi-item inventories.

## Reward-hacking / test-fitting review

**1. The circuit reads only call syntax.** Its symbols are:

- the digits
- `: , = < ? @` and the merged `:?` `:@`
- the word `track`, a space, and the four call/response tokens

There are no names, objects or task words, and no closed list of BBH entities. The reader is the repo's generic exact linear reader, which is checked on every vocabulary row. Nothing in the circuit could encode BBH phrasing. **Clean.**

**2. No Python logic at inference.** `callfmt.track_ref` is used only to build data and for tests. The test (`test_tools.py`) runs the compiled circuit token by token against that reference on:

- malformed calls
- random calls
- calls preceded by earlier calls of either tool
- text with no call, where the tool must stay silent

On my gold calls the tool alone was exact 34/34 (rounds 1–3). Its reply on a malformed call is always empty. It writes nothing for a call that names `order`; M3 shows this in practice. **Clean.**

**3. Generators and data.**

- The vocabulary is the team's own (vocab.py). It excludes the names in the prompt and BBH's well-known names, which can only make the test harder.
- The BBH-style generator copies BBH's surface template. Rule 2 permits this.
- The fine-tune uses only seed-2 generated rows.
  - The final host (ft100a) uses 102 of them.
  - It was selected on DEV only.
  - The fresh test was generated at freeze.

**4. Disclosed judge-in-the-loop tuning.** After round 2, gen_mix v4 added round-2-derived content: the wallpaper values and the rotation/copy wordings. Rounds 1–2 therefore count as development evidence. Round 3 is the only held-out judge evidence; Team A confirmed it was neither trained on nor inspected before the final run. The final ft100a host was trained on 102 rows that include a few mix rows, so the tuning effect on rounds 1–2 is small but not zero. I found no fix justified by memory of BBH for the tracker.

**5. The tool-free control is fair.** The fine-tuned host without the tool sits at the frozen base's level (fresh test 25/20/13 vs base 39/22/19). The gain therefore comes from the tool, not from the fine-tune learning the task.

## Rule 11: judge variants

Results below are for the final host ft100a with the official prompt, greedy decoding.

- "reply" = whether the tool's reply equals the gold call's result.
- "eq" = the call was encoded differently but is equivalent, and the answer is right.
- "coinc" = the call was wrong but its reply happened to match the gold result.

**Round 1** (`judge/variants_track_1.jsonl`; tuned-on after v1)

| id | setting | host+tool | tool reply |
|---|---|---|---|
| J1_mugs3 | mugs, days of the week, irrelevant sentence | OK | correct |
| J2_parking5 | parking spots, "who" query (@) | OK | correct |
| J3_cards3 | card game, discard + draw | OK | correct |
| J4_code3 | code variables | OK | correct (`21` used where the gold call copies; same result) |
| J5_books7 | 7 people, book club | X | correct; host mapped value 6 to the wrong letter |
| J6_umbrellas5 | question first | X | WRONG (`?1` instead of `?2`) |
| J7_cups3 | shell game | X | empty (malformed) |
| J8_docks5 | warehouse docks | X | empty (extra op) |
| J9_scorebooks7 | 7 people, 9 swaps | OK | correct |
| J10_jerseys4 | jersey numbers | OK | eq (slots 1234 instead of 8361) |

**Round 2** (`judge/variants_track_2.jsonl`; tuned-on)

| id | setting | host+tool | tool reply |
|---|---|---|---|
| K1_hotel5 | hotel rooms | X | empty |
| K2_relay5 | relay legs | OK | coinc (`3=5`) |
| K3_bikes7 | BBH-like 7-person, reworded | OK | correct |
| K4_bedrooms5 | non-event + "switch back" | OK | correct |
| K5_phones3 | 3-way rotation | OK | eq |
| K6_sims4 | SIM cards in devices | X | WRONG (`?` instead of `@`) |
| K7_chores6 | 6 housemates | X | WRONG (extra op, `?` instead of `@`) |
| K8_wallpapers4 | copy in plain words | OK | correct |

**Rounds 1–2 totals:**

- Final answers with the tool: 23/36. That count covers the 18 track rows and the 18 order rows. Without the tool, the fine-tuned host got 3/36 and the base 7/36.
- Track rows only: correct, equivalent or coincidental reply 12/18; empty 3/18; wrong value 3/18.

**Round 3** (`judge/variants_track_3.jsonl`; HELD OUT)

| id | setting | host+tool | tool reply | FT no tool | base |
|---|---|---|---|---|---|
| M1_zoo5 | zoo, animals as slots | X | WRONG (`?1` instead of `@1`) | X | X |
| M2_teams5 | reorg with copy + set | OK | correct | X | X |
| M3_words5 | word positions in a phrase | X | empty (call addressed to `order`) | OK | X |
| M4_array4 | 0-indexed Python list | X | WRONG (index conversion) | OK | X |
| M5_scarves7 | BBH-like 7-person, passive + pronoun | X | WRONG (one swap dropped) | X | X |
| M6_seats5 | airplane seats 14A..15B, @ query | OK | correct | X | X |
| M7_souvenirs3 | pronoun / role references | OK | correct | X | X |
| M8_cells4 | spreadsheet cells | X | no call | X | OK |

**Round 3 totals:**

- Answers: host+tool 3/8; FT without tool 2/8; base 1/8.
- Tool replies: correct 3; silent (empty or no call) 2; wrong value 3.
- Correct or silent: 5/8.

**Combined over rounds 1–3:** correct or silent on 20/26; wrong value on 6/26.

All 6 wrong values come from the host writing a well-formed but wrong call. The tool never computed a wrong value. The main pattern: 4 of the 6 are the wrong query kind or slot (`?` instead of `@`, or the wrong slot for `?`). The other two are a dropped operation (M5) and a failed 0-index conversion (M4).

## Rule 12: out-of-style evidence

All rows below are ft100a + tool. Each cell shows 3 / 5 / 7 people.

| set | base | FT, no tool | FT + tool | reply correct / empty / wrong |
|---|---|---|---|---|
| fresh test (BBH-like style, new vocabulary, n=100) | 39 / 22 / 19 | 25 / 20 / 13 | 92 / 91 / 88 | 275 / 1 / 24 of 300 |
| oos2 (held out: numbered pots on a stove, office desks; n=30) | 40.0 / 10.0 / 23.3 | 40.0 / 30.0 / 20.0 | 83.3 / 56.7 / 63.3 | 60 / 12 / 18 of 90 |
| oos3 (dev-only: bus bays, band song cards; n=20) | 35 / 20 / 25 | 20 / 25 / 10 | 70 / 65 / 75 | — |
| oos1 (seen: card game; n=30) | 33.3 / 23.3 / 6.7 | — | 60 / 50 / 50 | — |
| judge round 3 (held out, 8 new settings) | 1/8 | — | 3/8 answers right | 3 correct, 2 silent, 3 wrong |

The tool does work outside its benchmark's style. On held-out settings it adds +30 to +47 points over base (oos2) and works in a corporate reorg with copy and set, airplane seats, card games, code variables and parking. It is not a single-style tool.

## Why PASS

- The tool is exactly the generic, calculator-like function rule 8 asks for. It has a general text interface and contains nothing BBH-specific.
- It is compiled into weights, exact, and silent on malformed calls and on calls to other tools.
- Its f(x) is generally useful (above).
- Rule 11 (correct or silent on most variants) is met:
  - 5/8 on the held-out round 3;
  - 20/26 over all rounds.
- Rule 12 (works outside its style) is met on a held-out set (oos2) and on held-out judge settings (M2, M6, M7).
- The control shows the gains are the tool's: the fine-tuned host without the tool stays at base level.

## Caveats, binding for how results are reported

1. **The bottleneck is the host's call writing, not the tool.**
   - Off the BBH-like style, accuracy drops: oos2 57–83%, judge round 3 3/8.
   - The common errors are:
     - writing `?k` when the question asks "who / which X holds Y" (should be `@v`);
     - dropping an operation in long, varied narratives;
     - 0-index conversion;
     - choosing the wrong tool name (`order` for a list-rearrangement story).
   - Reports must describe the system as "BBH-like style: ~90%; other styles: much lower". They must not claim general robustness.
2. **Fail-safety belongs to the tool, not to the host+tool system.**
   - About 20% of held-out out-of-style calls were well-formed but wrong, and produce wrong answers: 18/90 on oos2 and 3/8 on round 3.
   - The fresh test also has 24/300 wrong values.
3. **Rounds 1–2 were tuned on (disclosed).** Only round 3, oos2 and the fresh test count as held-out evidence.
4. **The "tool reply = gold" column has an encoding effect.** It undercounts equivalent encodings (J10, K5) and overcounts coincidences (K2).
5. **Suggested (not required) improvements for a later version, trained on generated data only:**
   - question-kind training ("who has X" → `@`);
   - long narratives with mixed wordings;
   - index-base conversion.

Any such version needs a new judge round with fresh variants.
