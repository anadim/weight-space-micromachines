# JUDGE: repeat_request, round 3 (final verdict on the FROZEN circuit)

- **Tool:** repeat_request. It echoes the user's request verbatim at the start of the reply, then lets the host answer.
- **Code:** the frozen circuit (NOTES part 9). The local `circuit.py` and `tables.py` md5 match `frozen_MD5.txt` (cd15769c..., 78f6181b...).
- **Results:** `judge/repeat_request_results_r2b.md`. These cover 8 new spot-check items (707200-707207, `judge/repeat_request_variants_r2b.jsonl`) plus my 22 round-2 items (707100-707121), all on the frozen code.
- **Earlier verdicts:** `JUDGE_repeat_request_1.md` (PASS conditional) and `JUDGE_repeat_request_2.md` (FAIL: 2 false fires, minimal narrowing requested).

## 1. Code: what the frozen rule does

### Detection (circuit.py line 464)

The rule fires at a sentence end in a user turn, not in a system turn, when all of these hold:
- the sentence has a REPEATV word, a REQTERM word and a VERBATIM word. A verbatim word is now REQUIRED (verbatim / exactly / exact / unchanged / word for word / as it is / as written / without change...).
- no REFMT word ('into' / 'but') appears anywhere in the sentence. This marks reformat requests.
- the request guards hold: not an information question, not a third-person or reported or topic or code sentence.
- the sentence has a request marker (an instruction verb opening a clause, or you / please / must / ...).
- no negation within 4 tokens before the repeat verb.
- no 'your' / 'yours' within 2 tokens before the request noun.
- this is the first hit in the turn.

### Span and copy (unchanged)

- 'above' (the default) copies from the message start to the end of the previous sentence.
- 'below' copies the next line through the end of the message.
- The copy is forced by pointer at the answer start.
- Forced-token misses: 0 on every item.

### Python logic / reward hacking

None found:
- The rule is still pure `torch.where` / `gather` on 0/1 wires, pointer registers and fixed tables.
- Regex appears only in the offline table build.
- The word lists are generic English.
- No IFEval strings or templates.

### Residual design risks (not measured by a failing item, disclosed)

1. **Request markers do not check the subject.** A modal (must / should / need) counts as a request marker whatever its subject is. So a rule stated about a third party with a REQTERM and a verbatim word fires. Confirmed on 707207.
2. **The 'but' veto covers the whole sentence.** A real echo request such as "Repeat my request above word for word, but do not answer yet" will be silent. This is fail-safe (byte-identical), but it costs recall.
3. **Host quality after the copy.**
   - On 707100 the host still loops to 2048 tokens after the forced copy; plain stopped after 26 tokens.
   - The host often echoes the instruction sentence after the copy instead of answering (for example 707200).
   - The checker passes these. The implant guarantees the read-back, not the answer.

## 2. Rule 13: is f(x) generally useful?

- **f(x):** given a message that asks for its request to be echoed verbatim, emit an exact copy of the referenced span of that message at the start of the reply. The span is the text before the instruction, or the text after it for 'below/following'.
- **USEFUL-GENERALLY: yes.** Three settings:
  1. Read-back and confirmation protocols: pharmacy or order confirmation, ATC-style read-back, support-ticket acknowledgement.
  2. Email and forum replies that quote the question before answering.
  3. Audit logs and transcripts that keep the exact request next to the response.

## 3. Results on the frozen circuit (r2 + r2b, 30 items)

### r2b spot checks (new items)

| key | type | tests | result |
|---|---|---|---|
| 707200 | POS | "repeat my request exactly as written" | fired, correct span; plain T, implant T |
| 707201 | POS | multi-turn follow-up, "Echo this request back verbatim" | fired, correct span; T / T |
| 707202 | POS | "Recite the prompt below without changing a word, then respond:" + newline | fired (below), correct span; T / T |
| 707203 | POS | "Could you restate my question above exactly" (regression probe for the 'your' veto) | fired, correct span; T / T |
| 707204 | NEG | multi-turn "repeat your last answer exactly" | silent, byte-identical |
| 707205 | NEG | "Reproduce the table below exactly as CSV" | silent, byte-identical |
| 707206 | NEG | "Reproduce the text below exactly, but as a numbered list" | silent, byte-identical (REFMT veto) |
| 707207 | NEG | "A witness must repeat the clerk's question word for word ..." (third-party rule) | **FALSE FIRE**: "I'm studying court procedure for a class." prepended; the rest of the answer is normal |

### r2 items on the frozen code

- **Round-2 false fires fixed:** 707115 ("Copy the text below into bullet points") and 707116 ("Repeat your previous message exactly") are now silent and byte-identical.
- **Other negatives:** 707112-707114 and 707117-707121 are silent.
- **Positives:**
  - 8/12 fired with the correct span: 707100-707106 and 707111.
  - 4/12 are fail-safe silent and byte-identical: system message, 'following' + colon on the same line, 'read back', 'everything I wrote'.

### Metrics (30 items, frozen code)

| measure | result |
|---|---|
| Recall: positives fired with the correct span | 12/16 (75%) |
| Correct or fail-safe on positives | 16/16 (no wrong outputs) |
| False fires on negatives | 1/14 (7%; 707207) |
| Trigger precision | 12/13 (92%) |
| Silent items byte-identical to plain | 17/17 |
| Prompt-strict on positives, plain → implant | 9/16 → 15/16 (+6 / -0) |

**Seen-data caveat:** every one of these 30 items was used during development (part 7: r2 detection; part 9: r2b). They are not an unseen measurement. The held-out evaluation remains the real IFEval run.

## 4. Verdict: PASS

Reasons:
- No reward hacking and no Python logic.
- The function is generally useful.
- Every positive is correct or fail-safe: 16/16, with 0 wrong outputs.
- +6 / -0 on strict.
- Every silent output is byte-identical to plain.
- Both round-2 FAIL causes are fixed generically:
  - the verbatim word is now required;
  - the 'your' veto is correct and did not cost the "Could you restate my question" case;
  - the 'into'/'but' reformat veto works.
- One residual false fire remains (707207, a third-party rule stated with 'must'). It is disclosed as known and is a rare sentence type. Its harm is one extra prefix sentence; the answer itself is intact.

Two items for the record and for future versions (neither is needed for this freeze):
1. **Make modal markers subject-aware.** must / should / need would count only with a you-subject or no subject. This would remove 707207-type fires for every sentence-level rule, not only repeat.
2. **Scope the 'but' veto to the clause after the repeat verb's object**, to recover "..., but do not answer yet"-type echo requests.
