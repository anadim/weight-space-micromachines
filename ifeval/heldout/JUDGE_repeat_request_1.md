# JUDGE: repeat_request, round 1

- **Tool:** repeat_request. It echoes the user's request verbatim at the start of the reply, then answers.
- **Variants:** `judge/repeat_request_variants.jsonl`, keys 707000-707022 (14 positive, 9 negative).
- **Results:** `judge/repeat_request_results_r1.md`.
- **Code reviewed:** `circuit.py` and `tables.py` as of 2026-11-17 13:41, which includes NOTES.md change log parts 4 and 5.
- **What I read:** `circuit.py`, `tables.py`, `gen_data.py`, `NOTES.md`, `run.py`, `judge_run.sh`, `detect_test.py`.
- **What I did not open:** fresh, ood, `fresh_templates.json`, and all IFEval data.

**Important caveat.** The builder changed the circuit after my preliminary review (part 4). The change in part 5 was informed by running detection on these variants. So the negatives below are no longer unseen. The 0/9 false-fire result is partly fitted to these items. Round 2 must use new items.

## 1. Code review (current code)

### Detection (circuit.py, about lines 389-435)

The rule fires at a sentence boundary in a user or system turn when all of the following hold:
- the sentence contains a REPEATV word (repeat / restate / copy / echo / reproduce / recite / reiterate);
- it contains a REQTERM word (request / prompt / question / query / instruction / text / sentence / line / message / task / ask);
- it contains a verbatim or direction word: VERBATIM (verbatim / exactly / exact / unchanged / change / changing / as-is / word for word / as it is / as written), ABOVED (above / previous / preceding / earlier / prior), or BELOWD (below / following / follows / next / underneath / beneath);
- the request guard holds: the sentence is not an information question, and it has no third-person subject or third-person verb;
- there is no negation within 2 tokens before the repeat verb;
- this is the first hit in the turn.

Cue words inside double or single quotes are masked.

### Span selection

- **'above' mode (the default, and it wins if ABOVED is present):** the span runs from the message start to the end of the previous sentence.
- **'below' mode:** the span runs from the token after the next newline or colon to the end of the message.
- All span positions are pointer registers. The copy is a gather from token memory.

### Turn scope (part 5)

- User-turn latches are dropped at each new user turn.
- System-turn latches persist.

### Enforcement (about lines 573-581)

- At the start of the answer, the span is forced token by token by pointer, then `\n\n`.
- If a case rule is active, the forced tokens go through the case transducer.
- The forced token won on every step (forced-token misses 0 on all 23 items).

### Python logic at inference

None found:
- Rules are `torch.where` / `gather` on 0/1 wires, integer pointer registers and fixed tables.
- The Python `for` loops run over fixed key lists. They do not branch on data.
- Regex appears only in the offline table build.
- The hard-coded token ids (151644, 151645, 151668, 198, 271, 77091 = assistant, 872 = user, 8948 = system) are chat-template structure.
- The readouts are the disclosed LM-head readouts.

### Reward hacking / test fitting

None found:
- The word lists are generic English synonym sets. None is an IFEval sentence or template.
- The span logic ("above = everything before this sentence; below = the next line onward") is a generic reading of deictic words.
- The "drv:" dev templates are rewordings of even-key items. The author allowed this, and it is disclosed.
- The guards added in parts 4 and 5 are generic (verbatim-word requirement, 'above' overrides 'below', request guards, single-quote masking, turn scope).
- The guards were motivated by judge reviews and judge items, not by IFEval test data.

### Remaining defects and risks (current code)

1. **CONFIRMED defect, wrong output (not fail-safe).** A repeat instruction in a SYSTEM message latches during the system turn. Its span then points into the system message: it copies the system sentences before the instruction instead of the user's request.
   - Item 707006: the output begins "You are a friendly assistant for a library help desk."
   - Part 5 keeps system latches on purpose, so this happens on every user turn of the conversation.
   - Fix (narrowing): suppress the repeat latch in system turns (fail safe), or re-point the span to each new user message.
2. **Risk, untested.** VERBATIM contains `change` and `changing`. These words exist for "without change", but they also satisfy the gate positively. Examples:
   - "Copy the text and change the tone to formal." would fire in 'above' mode and copy the earlier text verbatim.
   - "Reproduce the table below exactly but in CSV" or "Can you copy the text below into a list?" would also fire. They are requests, but not echo requests.
   - Suggestion: allow `change/changing` only after without/no/not.
3. **Risk, untested.** ABOVED includes `previous / earlier / prior`. In a multi-turn chat, "Repeat the previous message exactly" means the assistant's last reply. The circuit instead copies the earlier sentences of the current user message, or stays silent if there are none.
4. **Fail-safe gaps (recall).** These cases stay silent, so the output is byte-identical to plain:
   - the request and the instruction are in one sentence (707007);
   - 'below' mode with the request on the same line (707008);
   - repeat verbs outside the list, such as "parrot" (707011).
5. **Host limitation, not a circuit fault.** After the forced copy, Qwen3-0.6B often echoes the instruction sentence instead of answering (707001-707005, 707009, 707010, 707013). Plain shows the same behaviour on several items (707005, 707010, 707012).
   - The checker passes these.
   - A real user would get a correct read-back but often no answer. The implant guarantees the prefix, not the answer.

## 2. Rule 13: is f(x) generally useful?

- **f(x):** given a message that asks for its request to be echoed verbatim, emit an exact copy of the referenced span of that message (the text before the instruction, or the text after it for 'below/following') at the start of the reply, then let the model answer.
- **USEFUL-GENERALLY: yes.** Three settings:
  1. Read-back and confirmation protocols: pharmacy or order confirmation, ATC-style read-back, support-ticket acknowledgement.
  2. Email and forum replies that quote the question before answering.
  3. Audit logs, transcripts and compliance records that must hold the exact request next to the response.

## 3. Variants and results

- "fired" means the repeat latch was set, and "span" is the captured span.
- The strict columns score only the instruction(s) listed for the item, using the official checkers.

| key | type | what it tests | fired / span | plain | implant | verdict |
|---|---|---|---|---|---|---|
| 707000 | POS | reiterate + prompt + verbatim | yes / correct | F | T | gain |
| 707001 | POS | casual, lowercase, newline-only boundary, 'echo' | yes / correct | F | T | gain |
| 707002 | POS | "Do not answer until you have repeated ..." (negation must not veto) | yes / correct | F | T | gain |
| 707003 | POS | repeat + lowercase in one sentence | yes / correct (lowercased copy) | F,T | T,T | gain |
| 707004 | POS | multi-turn follow-up (span = current message only) | yes / correct | F | T | gain |
| 707005 | POS | buried in a long casual message, 4-sentence span ending in '?' | yes / correct | T | T | same |
| 707006 | POS | instruction in a SYSTEM message after another system sentence | yes / **WRONG** (system text) | F | F | wrong output |
| 707007 | POS | request + instruction in one sentence | silent | T | T | fail-safe (byte-identical) |
| 707008 | POS | 'following' + request on the same line | silent | F | F | fail-safe (byte-identical) |
| 707009 | POS | 'below' + colon + newline, instruction first | yes / correct | T | T | same |
| 707010 | POS | 'above' explicit but "write your answer below" + trailing line | yes / correct ('above' wins) | T | T | same |
| 707011 | POS | verb not in list ('parrot') | silent | T | T | fail-safe (byte-identical) |
| 707012 | POS | 'above' + 'next' | yes / correct ('above' wins) | T | T | same |
| 707013 | POS | repeat + end phrase in one sentence | yes / correct (+ end phrase) | T,T | T,T | same |
| 707014 | NEG | "What does it mean to restate the question ...?" | silent | - | - | OK, identical |
| 707015 | NEG | cue in double quotes | silent | - | - | OK, identical |
| 707016 | NEG | cue in single quotes | silent | - | - | OK, identical |
| 707017 | NEG | third-person habit ("My son keeps repeating every question ...") | silent | - | - | OK, identical |
| 707018 | NEG | "repeat customers" + message | silent | - | - | OK, identical |
| 707019 | NEG | bash "echo an error message" | silent | - | - | OK, identical |
| 707020 | NEG | "copy text from a PDF" | silent | - | - | OK, identical |
| 707021 | NEG | "Please don't repeat my question back to me" | silent | - | - | OK, identical |
| 707022 | NEG | stale repeat instruction in an earlier history turn | silent | - | - | OK, identical (part-5 turn scope) |

### Trigger metrics (repeat latch)

| measure | result |
|---|---|
| Fired on positives | 11/14 (recall 79%) |
| Fired with the correct span | 10/14 (71%) |
| Fired on negatives | 0/9 (false-fire rate 0%) |
| Trigger precision (fires that were on positives) | 11/11 (100%) |
| Fires with the correct span | 10/11 (91%) |
| Silent items byte-identical to plain | 12/12 (3 positives, 9 negatives) |

### Generation (strict, all instructions of the item, 14 positives)

| measure | result |
|---|---|
| Prompt-strict, plain | 7/14 |
| Prompt-strict, implant | 12/14 |
| Gained | 5 (707000-707004) |
| Lost | 0 |
| Remaining failures | 707006 (wrong system span) and 707008 (silent; plain also fails) |

### Compared with my preliminary predictions (made on the old code)

- On the old code I predicted about 6/9 false fires and 2 wrong-span outputs (707006, 707010). 707012 was predicted to stay silent.
- Parts 4 and 5 removed all predicted false fires and fixed 707010 and 707012.
- 707006 is still wrong.

## 4. Verdict: PASS (conditional)

Reasons:
- **No reward hacking or Python logic.** The detector, the span pointers and the forced copy are generic and fully tensorised.
- **Positives:** correct on 10/14 and fail-safe on 3/14 (silent and byte-identical), so 13/14 are correct or safe. One item gives wrong output (707006, system message).
- **Negatives:** 0/9 false fires.
- **Silent outputs:** byte-identical to plain in every case.
- **Generation:** +5 / -0 on the positives.

Conditions:
1. **Narrow the rule for system messages** before the final eval: no repeat latch in system turns (fail safe), or re-point the span to each new user message. Today it produces a wrong copy, not a silent miss.
2. **Round 2 must use new items.** Part 5 was tuned with detection results on these items, so the 0/9 false-fire rate is not an unseen measurement. Round 2 should probe:
   - `change/changing` used without a negation ("copy the text and change the tone");
   - "copy/reproduce the text below into/as ...";
   - "repeat the previous message" in multi-turn;
   - repeat verbs outside the list (parrot, read back, quote back, echo back variants).
3. **Optional recall work.** Accept "quote back" / "read back" as repeat verbs, and handle 'following' with the request on the same line. Today both stay silent, which is safe.

**Keep the rule; do not drop it.**
