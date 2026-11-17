# JUDGE: order_gen (generic ordering-constraint solver), judge 1, rounds 1-3

Judge: independent subagent. Read: PROTOCOL.md, order_gen.py, circ_common.py, callfmt.py, register.py, machine.py (structure),
gen_order.py, vocab.py, make_data.py, gen_mix.py, gen_oos2.py, gen_oos3.py (to avoid overlap), test_tools.py, ft.py,
run_eval.py, NOTEBOOK.md, the official 3-shot logical_deduction prompt, and the team's result files
(judge/results_v3_rounds12.md, judge/results_ft100a.md, judge/final_tables_ft100a.md). No BBH task data was opened.

## VERDICT: PASS (with conditions; see the end)

The tool is generic, exact, and fully compiled. It has no BBH vocabulary and no Python at inference, and it computes a function that is useful in general. The weak part of the system is the host's translation of text into a call (a 102-row fine-tune). On hand-written, held-out, adversarial wording (round 3), host+tool did no better than the frozen base. That is a limit of the host, not reward hacking in the tool, and I report it so it is not hidden behind the BBH number.

## Rule 13: f(x) and USEFUL-GENERALLY
f(x): the input is N <= 7 numbered items and a conjunction of constraints. The constraint kinds are:
- before / after (`<`, `>`)
- immediately before (`+`)
- adjacent (`~`)
- at position k from the start or the end (`=k`, `=-k`)
- not at position k from the start or the end (`!=k`, `!=-k`)

The output is whatever every consistent ordering agrees on: the item at position k (`?k` / `?-k`), the position of item a (`@a`), or the whole order (no query). The reply is empty when no ordering fits, when the answer is not determined, or when the call is malformed.

**USEFUL-GENERALLY: YES.** Three settings outside the benchmark:
1. Race, tournament or exam results reconstructed from partial reports ("A beat B, C was third").
2. Seating, parking, shelf or line-up arrangements with adjacency and exclusion constraints.
3. Ordering tasks in a schedule (recipe steps, talks, assembly stations, deadlines in a week), with precedence and fixed slots. The tool answers "is X determined, and where?"

The "agreed by all solutions" rule makes it useful on underdetermined inputs too.

## Review for reward hacking / test fitting
| item | finding |
|---|---|
| Circuit (order_gen.py, circ_common.py) | **Clean.** It reads only call-syntax tokens (digits, `: , < > + ~ = =- != !=- ? ?- @ :? :@`, the word `order`, a space, the call/response tokens). It has no object names, no relation words and no BBH strings. All 5,912 orderings for N=2..7 are fixed ReLU units. It stays silent on calls that name another tool. |
| Python at inference | **None.** callfmt.py is used only for tests and training data. machine.py does tensor operations only. |
| Tool exactness | **Exact** on every gold call I wrote: 25/25 over rounds 1-3. The team's test_tools.py reports 0 failures on 300 random calls and 20 malformed calls. |
| Generator vocabulary | Names and objects are the team's own. The few-shot names and the well-known BBH names are excluded on purpose. |
| Recalled BBH scales | **Disclosed.** The v1 'bbh' style had age (oldest/newest, "{o}-newest") and price ("{o}-most expensive", plural produce) families that are not in the official prompt; the team admits these were recalled from BBH. v2 replaced them with 12 generic gradable scales. The ablation without age and price (ft100a_abl) scores the same or higher on fresh test: 84/90/81 against 84/84/72. **The gain does not depend on the recalled scales.** |
| Leak of judge wording | **Disclosed.** After my round-2 summary, the team added a wells, deeper/shallower axis (v4 PROPS, gen_oos3). Round 2 is therefore counted as tuned-on, and I weigh round 3, which stayed held out, most. |
| Fine-tune | 102 rows from the team's generators (seed 2), chosen on DEV. Without the tool it scores at the frozen-base level or below, so the gain comes from the tool. The official prompt is unchanged at eval. |

## Rule 11: reworded variants and results (host = ft100a, official prompt, greedy)
Files: judge/variants_order_1.jsonl (round 1), judge/variants_order_2.jsonl (round 2), judge/variants_order_3.jsonl (round 3).
- Round 1: the team saw the v1 results before v3 → partly tuned-on.
- Round 2: tuned-on.
- Round 3: held out; first and only run with ft100a.

"Tool" column values:
- **gold**: the reply equals the reply to my gold call.
- **empty**: fail-safe.
- **WRONG**: the call was mistranslated, so the reply was wrong.
- **no call**: the host did not call the tool.

### Round 3 (held out)
| id | setting / what it tests | target | host+tool | tool | FT no tool | base |
|---|---|---|---|---|---|---|
| judge3_order_1 | spelling-bee elimination ("outlasted", winner = last out) | (A) | (E) X | WRONG: "outlasted" / "lasted longer" written in the wrong direction | X | X |
| judge3_order_2 | homework deadlines; weekday options (`@` query) | (D) | (C) X | WRONG: `?2` instead of `@2`; both "day after" clues reversed | X | X |
| judge3_order_3 | follower counts ("No one has fewer than B") | (C) | (A) X | no call | X | OK |
| judge3_order_4 | reworded BBH-like intro, 7 shirts on a clothesline | (D) | (D) OK | gold (call exactly = gold) | X | OK |
| judge3_order_5 | beach huts, mirror viewpoint | (B) | (E) X | empty (inconsistent call) | OK | X |
| judge3_order_6 | time trial ("did not beat anyone", "nobody between") | (B) | (C) X | empty ("did not beat anyone" written as `!=1`) | X | X |
| judge3_order_7 | book chapters ("in the middle") | (C) | (C) OK | empty (`?middle` is malformed); host then answered right | OK | X |
| judge3_order_8 | 7-station assembly line (upstream/downstream) | (E) | (B) X | WRONG: "directly downstream of" reversed | X | X |

Round 3 summary:
- host+tool 2/8, FT no tool 2/8, base 2/8.
- Tool reply on the host's call: correct 1, empty 3, no call 1, wrong 3. So it **fires correctly or fails safe on 5/8**.

### Round 1 (partly tuned-on)
| id | setting | target | host+tool | tool |
|---|---|---|---|---|
| judge_order_1 | nails by price, a year number in a filler sentence | (B) | OK | gold |
| judge_order_2 | marathon | (D) | OK | gold |
| judge_order_3 | 7 people at a banquet table | (F) | OK | empty |
| judge_order_4 | conference talks (`@`) | (C) | OK | gold |
| judge_order_5 | clues before the setting, pantry shelves | (B) | OK | gold |
| judge_order_6 | exam scores (`!=`) | (D) | X | empty |
| judge_order_7 | 7 cars, west/east | (C) | OK | empty |
| judge_order_8 | siblings by age | (C) | OK | gold |
| judge_order_9 | "between" clue | (D) | OK | gold |
| judge_order_10 | "not next to" (cannot be written as a call) | (B) | X | no call |

### Round 2 (tuned-on)
| id | setting | target | host+tool | tool |
|---|---|---|---|---|
| judge2_order_1 | train stops, numbers in a filler sentence | (D) | OK | gold |
| judge2_order_2 | recipe steps (`@`) | (C) | X | empty (wrote `track`) |
| judge2_order_3 | wells (deeper), 7 | (D) | OK | gold |
| judge2_order_4 | town history | (A) | X | no call |
| judge2_order_5 | full-order query | (A) | X | WRONG |
| judge2_order_6 | DJ playlist, 7 | (C) | OK | gold (adjacency written reversed, but the reply happened to be the same) |
| judge2_order_7 | hot sauces | (D) | X | empty |
| judge2_order_8 | ferries | (C) | OK | gold |

Rounds 1-2: host+tool 13/18 against a frozen base of 4/18. The tool fired correctly or failed safe on 17/18.

**All 26 order variants:**
- Correct or fail-safe on 22/26; wrong value on 4/26.
- Tool alone exact on 25/25 gold calls; judge_order_10 has no gold call.

**Main recurring host error: direction reversal.** The host reverses "right after / directly downstream of / outlasted" clues (round 3 #1, #2, #5, #8; round 2 #6). The call format has only "immediately BEFORE" (`a+b`), so the host must swap the arguments for every "immediately after" clue.

## Rule 12: evidence outside the benchmark's style
Answer accuracy for order3 / order5 / order7. "Tool reply = gold" counts are rows where the reply matched the gold call's reply.

| set | status | frozen base | FT100 + tool | tool reply = gold |
|---|---|---|---|---|
| oos2: birthdays in calendar order, parking spaces from an entrance | held out | 33.3 / 13.3 / 16.7 | 70.0 / 50.0 / 36.7 | 21/30, 12/30, 9/30; most of the rest is empty, not wrong |
| oos3: climbers on a rope, lakes by depth, fruits by sweetness | dev-only | 45 / 35 / 25 | 80 / 65 / 55 | |
| oos1: theater seating, contest ranking | seen | 40.0 / 20.0 / 16.7 | 73.3 / 50.0 / 53.3 | |
| Fresh test (BBH-like style, seed 1, generated at freeze) | held out | 46 / 27 / 18 | 84 / 84 / 72 | |

On fresh test:
- Wrong values: 12 / 7 / 6 %.
- Empty replies: 8 / 12 / 25 %.

So the tool works outside the benchmark's style on generated settings that were never trained on. On my hand-written round 3 it did not beat the base. The difference is the host's text-to-call translation on unfamiliar wording, not the tool.

## Why PASS, and conditions
PASS because:
1. **No reward hacking.** There is no closed vocabulary, no BBH strings and no Python at inference.
2. **The recalled-BBH-scale issue is disclosed** and shown not to matter by the ablation.
3. **The leak of judge wording is disclosed**, and I scored the held-out round on its own.
4. **Rule 11 is met on the strict reading.** The tool is exact on every gold call, and in host+tool it fires correctly or fails safe on most variants: 22/26 overall, 5/8 on the held-out round.
5. **Rule 12 is met.** There are large gains on held-out out-of-style sets.

Conditions and required disclosures:
1. Next to any BBH number, report round 3 honestly: host+tool 2/8 = base 2/8. Also report that 3/8 held-out variants and 6-12% of fresh-test rows got a confidently WRONG tool value from a mistranslated call. The fail-safe design covers malformed or inconsistent calls, not wrong translations.
2. Recommended generic interface fix: add an "immediately after" operator (e.g. `a-b`), or train both phrasings symmetrically, because argument reversal is the main failure mode. Treat this as a new tool version, which needs re-judging.
3. Known limits of what the call can express:
   - no disjunctions ("between" without an orientation, "not next to"), no distance clues, no ties;
   - at most 7 items and at most 20 constraints;
   - an empty constraint list with a `?-k` query is not supported.

   The fail-safe outcome when one of these comes up is to skip the call or get an empty reply; round 1 #10 behaved this way under ft100a.
4. Keep round 3 out of all future training. The team has confirmed this.
