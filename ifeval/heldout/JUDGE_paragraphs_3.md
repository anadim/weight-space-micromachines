# JUDGE: paragraphs (round 3: final spot check on the FROZEN circuit)

Code checked: frozen/circuit.py, frozen/tables.py, frozen/vecs2.pt. The md5 sums are in frozen_MD5.txt:
- circuit cd15769c...
- tables 78f6181b...
- vecs2 5f5f5422...

Changes since round 2: NOTES parts 8-9. For paragraphs, whether/if now count as wh-words in the topic test, and a sentence that opens with an auxiliary or modal not followed by 'you' counts as a question.

Items:
- the 6 fresh r2b items (judge/paragraphs_variants_r2b.jsonl, keys 709200-709205);
- the 16 r2 items again (709100-709115).

Results: judge/paragraphs_results_r2b.md.

## Verdict: PASS

The round-2 FAIL had one cause: a false fire on an information question about a count (709110). It is fixed, and the fix holds on fresh items.

### Spot-check results (frozen code)
| key | type | item | result |
|---|---|---|---|
| 709110 | NEG (r2 false fire) | "Could you tell me whether 3 paragraphs is enough for a scholarship essay?" | silent, byte-identical (was 14 writes) |
| 709200 | NEG fresh | "Can you tell me if 4 paragraphs would be too much for a best-man speech?" | silent, identical |
| 709201 | NEG fresh | "I'd love to know whether five paragraphs works for a blog intro, or is that overkill?" | silent, identical |
| 709202 | NEG fresh | "Please explain whether 2 paragraphs are enough for a conference abstract." (no '?') | silent, identical |
| 709203 | POS fresh | "If you can, write 3 paragraphs about honeybees and separate them with ***." | N=3, *** mode, 3 writes; checker plain F, implant **T** |
| 709204 | POS fresh | "Tell me about the Great Wall of China in 4 paragraphs, with *** between them." | N=4, *** mode, 5 writes; checker plain F, implant **T** |
| 709205 | POS fresh | "Describe the basic rules of cricket in 3 paragraphs with *** between them, and keep it simple if you can." | miss, fail-safe (identical to plain) |

The other r2 items on the frozen code:
- All r2 negatives are still silent and byte-identical (709109, 709111-709115).
- The r2 positives are unchanged:
  - 709101, 709103 and 709106 fire and pass;
  - 709102 is latched with N=2, and the host already complied;
  - 709100, 709104, 709105, 709107 and 709108 miss, each byte-identical to plain.

### Totals on the frozen code (r2 + r2b, 22 items)
- **Negatives:** 0 of 10 fire. 3 of them are fresh, never seen before this run.
- **Positives:** 6 of 12 detected with the correct N, and all 6 enforced correctly. The other 6 miss and are byte-identical to plain.
  - So 12 of 12 are correct or fail-safe.
- **Precision:** 6 of 6. **Recall:** 6 of 12.
- **Checker:** number_paragraphs went from 0/9 plain to 4/9 implant, and nth from 0/1 to 1/1. No regressions.
- The builder reports that on all 78 spot-check items, 59 of 59 silent outputs are byte-identical to plain.

### Code (unchanged findings, see JUDGE_paragraphs_1.md and _2.md)
- No reward hacking, no IFEval strings, no Python logic in the data path. All word lists are generic English.
- **USEFUL-GENERALLY: yes.**
  1. CMS or newsletter templates with N blocks.
  2. Slide or card generators that split output on `***` or blank lines.
  3. Fixed-length structured documents: cover letters, 3-paragraph summaries, school essays.

### Residual issues (none blocking; all fail safe or minor)
- **R1. The topic veto is too broad and covers the whole sentence.** explain/describe/tell together with any wh-word or **whether/if** anywhere in the sentence cancels every rule in it.
  - 709205 misses because of a conditional "if you can".
  - "...in 3 paragraphs, explaining how..." also misses (709108).
  - Common phrasings will be lost, so recall on real data will be lower than on dev. This is the main recall fix for later: make the veto clause-scoped, and treat 'if' as a wh-word only right after the verb ("tell me if").
- **R2. The window is counted in tokens.** Upper-case "4 PARAGRAPHS" splits into 4 tokens and misses (709107).
- **R3. Requests with no verb miss**, as do follow-ups like "Now X, but 4 paragraphs with ***" (709100, 709104, 709105).
- **R4. The `***` block in the answer still works one token at a time** (F4). Not seen in r2 or r2b.
- **R5. detect_test.py scores 'para' only in *** mode** (F7), so blank-line false fires do not show in detection numbers. Reported generation runs do show them, through output identity.
- **R6. Quality cost of forced length.** On the 0.6B host, a 5-paragraph requirement can pad and repeat (709103: 126 -> 512 tokens). It still passes the checker.

PASS on: clean code; 0 false fires on 10 negatives (3 fresh); every positive correct or byte-identical to plain; no checker regressions.
