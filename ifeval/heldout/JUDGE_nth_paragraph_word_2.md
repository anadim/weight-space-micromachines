# JUDGE nth_paragraph_word, round 2 (final verdict on the CURRENT code, change log part 7)

**Tool:** paragraph k must start with word X, with N paragraphs separated by blank lines.
**Variants:** judge/nth_paragraph_word_variants_r2.jsonl (keys 710100 to 710116; 9 positives, 8 negatives; new topics, wordings and settings).
**Results:** judge/nth_paragraph_word_results_r2.md.

Caveat: the builder ran detection on these items while writing part 7, so they are partly seen. Round 1 is in JUDGE_nth_paragraph_word_1.md (verdict FAIL, because of a false fire on a first-person description, 710013).

## 1. Code findings (current code)

### No reward hacking, no Python logic at inference
- The nth path is still made of registers, tables and pointer copies:
  - `armw`: arms the capture when a start verb is in the sentence and the current token is "with".
  - `cap0` / `cont`: capture the span of X. SKIPNTH is now "the word term"; an article right after "with" cancels the capture ("with a question" is not a word).
  - `s_k` / `s_klast`: read k from an ordinal, "paragraph k", or last/final.
  - `nth_hit`: fires at the sentence end, through four guards: question / third person (`rqs`), sentence request marker (`reqs`), per-clause negation with the start verb (`n_nth`), and the data, topic, report and code masks.
  - `n_go` / `n2`: copy X into paragraph k.
- The only Python condition is the lazy dict initialisation in `_cue_neg`, which is structural rather than a decision on data.
- **New word lists are generic English, not IFEval wording:**
  - INSTV: instruction verbs.
  - REPORTV: says, claims, insists, ...
  - CODEW, DATAW, ARTW (a, an).
- X is still captured by pointer; there is no closed list of target words.

### What the round-1 FAIL reasons look like now
- **Description guard.** "My / Maria's / Usually my ... paragraph begins with the word X" no longer latches (710109, 710110). The request marker now needs an instruction verb, a modal, "you" or "please".
- **Negation scope.** The check is now per clause, with the start verb.
- **Capture after "with a".** It no longer takes the next noun (710111).

### Remaining issues (all generic)
1. **Pasted, unquoted instruction text still fires (710115).** "Here is the assignment text my professor posted:" followed by a blank line and then "Write 3 paragraphs. Start paragraph 2 with the word Consequently." The data mask stops at the blank line. The implant added two paragraphs, which broke the user's own request ("in one or two sentences").
2. **The forced word is always capitalised (CAP1), whatever the user wrote.** In 710108 the user asked for all lowercase and wrote "finally" in lowercase. The lowercase rule did not latch: "in all lowercase letters, write ..." is a lowercase-tool miss. The host's plain answer was all lowercase, and the forced "Finally" broke it (lowercase T → F). Copying X as the user wrote it would avoid this, and the checker is case-insensitive anyway.
3. **Recall dropped with the part-7 narrowing (fail-safe):**
   - Verbless requirement lines ("Paragraph 2 opens with the word Crucially." in a numbered list; 710100).
   - A system-message count with no verb or modal ("every reply has exactly 2 paragraphs"; 710104). nth latched there, but no count, so it stayed silent.
4. **710101 is an unexplained total miss.** Neither the count nor nth latched for "Write four paragraphs about night trains in Europe with blank lines between them, opening the third paragraph with the word Notably, and don't use any exclamation marks." It is fail-safe (byte-identical), but this is exactly the round-1 risk that the per-clause negation was meant to fix. Please trace it.
5. **Enforcement quality comes mostly from the paragraph-count tool:**
   - 710103 ends with a "**Never**." paragraph.
   - 710108 paragraph 3 re-copies paragraph 1.
   - 710107 is clean.

## 2. Rule 13: is f(x) generally useful?
- **f(x):** given a requested paragraph count N (blank-line separated), an index k (ordinal, "paragraph k", or last/final) and a word X, emit a reply whose paragraph k begins with X.
- **USEFUL-GENERALLY: yes.** Three settings:
  1. Legal and contract drafting ("Whereas", "Accordingly").
  2. Templated business writing (status emails whose delays paragraph opens "Unfortunately", reports with a "Summary" paragraph).
  3. Structured teaching material and form letters with fixed lead words.

## 3. Variants and results (part-7 code)
"Latched" means the circuit latched the nth rule and recorded k, X and N. "nth strict" is the official checker for this instruction, plain → implant. "Identical" means the implant output is byte-identical to plain.

### Positives (9)

| key | what it tests | latched (k, X, N) | nth strict | identical |
|---|---|---|---|---|
| 710100 | numbered requirement list, "Paragraph 2 opens with the word Crucially." | no (verbless) | F → F | yes (fail-safe) |
| 710101 | negation in another clause of the same sentence | **no** (neither count nor nth; unexplained) | F → F | yes (fail-safe) |
| 710102 | curly quotes "Meanwhile," with the count in another sentence (";") | yes (2, Meanwhile, 3) | T → T | yes (forced token = host's) |
| 710103 | target word is a negation ("Never") | no (expected); count only | F → F | no (count tool adds "**Never**.") |
| 710104 | system-message rule, unrelated user question, k=1 | nth yes (1, Greetings); count no | F → F | yes (gated off) |
| 710105 | multi-turn follow-up, "lead the final paragraph" | yes (last, Today, 3) | T → T | yes |
| 710106 | casual question "mind starting para 2 with ...?" | no (expected); count only | F → F | yes |
| 710107 | "You must start the 4th paragraph with the word Unsurprisingly" | yes (4, Unsurprisingly, 5) | **F → T** | no |
| 710108 | nth + all lowercase (lowercase rule not latched) | yes (3, finally, 3) | **F → T** (but lowercase T → F) | no |

**Totals on positives:**
- Latched with correct arguments: 5/9. Latched with a count, so actually enforced: 4/9.
- 2 misses were expected (710103, 710106); 3 were not (710100, 710101, 710104's count).
- For nth, every miss failed safe (nth silent).
- nth checker: 2/9 plain → 4/9 implant. Prompt level: 2/9 → 3/9 (710108 trades nth for lowercase).

### Negatives (8): the nth tool must stay silent

| key | what it tests | nth latched | identical |
|---|---|---|---|
| 710109 | named person's essay ("Maria's essay ... begins with the word Still") | no | yes |
| 710110 | adverb-opened habit ("Usually my reports ... starts with the word Overall") | no | yes |
| 710111 | "open the second paragraph with a question" (+ legitimate count) | no | no (count tool, legitimate) |
| 710112 | negated "don't start the second paragraph with the word Basically" | no | yes |
| 710113 | system rule cancelled by "plain text" | no (dropped) | yes |
| 710114 | "Can you explain why my teacher insists ... must start with the word Imagine?" | no | yes |
| 710115 | pasted, unquoted assignment text (already submitted) + "tell me in one or two sentences" | **yes (2, Consequently, 3)** | **no: FALSE FIRE** (2 extra paragraphs) |
| 710116 | "Some newspapers start the first paragraph ... with the word Beloved" | no | yes |

**Totals on negatives:**
- nth silent on 7/8. The one false fire (710115) is the hard case with no quote marks.
- Byte-identical to plain on 6/8 (710111 differs because of a legitimate count).
- 710112 latched the forbidden-words rule on "Basically", but it wrote nothing.

### Trigger precision and recall (nth, round 2)
- **Recall:** latched with correct arguments 5/9 (56%); enforced 4/9 (44%). Excluding the 2 expected misses: 5/7.
- **Latch precision:** 5/6 (83%).
- **Enforced precision:** 4/5 (80%).
- **False fires on negatives:** 1/8 (12.5%), on the ambiguous pasted-text item.
- **Builder's overall r2 numbers:** 206/206 silent prompts and 172/172 unlatched prompts were byte-identical to plain.

## 4. Verdict: PASS (conditional)
- **No reward hacking, no Python logic, no IFEval strings.** USEFUL-GENERALLY: yes.
- **Round-1 failures fixed generically.** Every description and mention negative from both rounds that does not involve pasted text is now silent, including the four risks I flagged in round 2 (named subject, adverb opener, "with a question", polite "explain why ... must").
- **Positives are correct or fail-safe.** Every nth miss stayed silent for nth. When enforced, the tool is correct (4/4 on the checker).
- **The remaining false fire (710115) is the hardest negative type.** It is an unquoted instruction pasted after a blank line, which a pattern reader cannot tell from a request without the surrounding "already submitted" context. It must be disclosed as a known limitation.

**Conditions (minimal narrowing; do not drop the rule):**
1. Copy X as the user wrote it, instead of always capitalising with CAP1 (710108). This matters most when the user wrote X in lowercase or a lowercase rule is active.
2. Extend the data mask that follows a data-introducing colon or DATAW line ("here is the assignment text ... :") across blank lines up to the next user sentence that carries its own request marker, or to the end of the message. If that is not done, disclose 710115-type pasted instructions as a known false-fire case.
3. Trace 710101, where an imperative "Write four paragraphs ..." plus nth plus a negation in another clause latched nothing. It is fail-safe, but it is the scenario the per-clause negation claims to handle.

**Optional recall work (not required for PASS):**
- Verbless requirement-list lines (710100).
- A system-message count stated as a description ("every reply has exactly N paragraphs"; 710104).
