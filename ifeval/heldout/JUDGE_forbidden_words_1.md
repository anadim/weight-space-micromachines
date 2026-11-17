# JUDGE forbidden_words, round 1

Judge: independent reviewer (subagent). Tool: forbidden_words, which never emits the given words.
Variants: judge/forbidden_words_variants.jsonl (keys 706000-706022). Run results: judge/forbidden_words_results_r1.md.
Code reviewed: circuit.py and tables.py, the CURRENT version after NOTES "Change log, part 4" and "part 5".
I opened no test data: no fresh, ood or IFEval files.

**Verdict: FAIL. Narrow the rule, do not drop it.** The code is clean and enforcement is exact. The problem is detection: it still false-fires on 2 of 8 negatives, and 1 of those 2 changes the output in a harmful way. The brief asks for a false-fire rate near zero, and 25% is not near zero. Both remaining false fires come from generic gaps that can be fixed (see section 5).

Caveat on the evidence: the builder ran detection on these variants before making the part-5 changes (single-quote masking, turn scope). My items are therefore no longer fully held out. The silent results on 706015, 706016 and 706021 are weaker evidence than the numbers alone suggest. Round 2 needs fresh variants.

## 1. Code findings (current code)

Reward hacking and test fitting: none found.
- The cue lists are generic English closed-class words: NEG, WORDTERM, DETW, SEPW, APPEARS, NEGADJ, STOPF, FREQ3, TIMES.
- I found no IFEval sentences, templates or entity lists.
- STOPF is shaped to close an unquoted list at common instruction tails ("in your response", "anywhere", "should"). It is still a generic function-word list.

Python logic at inference: none in this path.
- Region capture, slot writes, commit, turn scope and blocking all use torch.where, gather or integer registers.
- The Python `for` loops run over fixed key lists only, never over data.
- The readout "top token is in the blocked set" is the disclosed LM-head readout.

Disclosure point (not hacking): word identity is a polynomial rolling hash mod 2^31-1.
- It runs in int64 registers with fixed per-token tables (HT, POWT, LEAD, TRAIL, HASAN).
- This is integer register arithmetic, not 0/1 wires. The claim that it "compiles into weights" is therefore weaker than for the rest of the circuit.
- It is disclosed in NOTES. A piece-sequence pointer compare would remove this point.
- Collision risk is negligible.

Enforcement is correct for the checker:
- A token is blocked if (current word prefix + token core), lowercased, equals a forbidden word. This makes the block case-insensitive. It also covers "-word", "word's" and "word,".
- When the host's top token is not blocked, the write is zero, so the output is byte-identical to plain.
- In every positive where the rule fired, the implant passed the checker. Forced-token misses: 0.

Enforcement weaknesses (quality, not checker failures):
- **Word fragments.** The block acts on the piece that would complete the word, and the host cannot backtrack. When a word is split into pieces, a fragment stays in the text. Example, 706020: `say "be" or "s"` came from blocked "beach" and "sand". A word-start lookahead, or a block on the first piece of a forbidden word's tokenizations, would fix this.
- **Phrase split.** A multi-word phrase becomes separate word slots. 'money printer' also blocks 'money' (706012), and the explanation drifted to "a lot of toys".
- **Slot limit.** At most 6 words (S_W). More words are dropped silently.
- **Forced programs bypass the block.** This applies to the end-phrase copy, the repeat copy and the nth first word. Those combinations contradict each other anyway.
- **Broad cancel cue.** "this time" is in CANCELW. A system-message ban followed by a user turn "this time keep it short" drops the system ban. This is a fail-safe miss, but a surprising one.

Detection (current code):
- A list region opens after "the/these/those/following word(s)/keyword(s)/term(s)", or at a quote opened within 6 tokens after a negation.
- The region commits if a negation comes before it (within 8 tokens) or a pattern-B form follows it: NEG + appear/be/used..., or a negative adjective.
- The commit now also requires: the sentence is not an information question, has no third-person subject or verb, and has no FREQ3 or TIMES word. Cue words inside single or double quotes are masked, but the quoted list items are still captured.
- Fixed since my preliminary review:
  - 'free' is no longer a negation word (706015 silent).
  - 'under' no longer drops the sentence (706006 and 706010 now fire).
  - The third-person guard now applies (706016 silent).
  - Single-quoted reported speech is masked (706021 silent).
  - The lowercase+NEG clause now gives lowercase, not caps (706005).
- Still open:
  - 'once' and 'times' in TIMES still drop the sentence, so "never use the word X, not even once" gives no fire.
  - **Compound after 'the word'.** An unquoted item after "the word" is captured even when it forms a compound noun with 'word': 'the word count' commits 'count' (706017).
  - **Rule descriptions with generic 'you'.** A declarative rule description passes every request guard: "In Taboo, you cannot say 'beach' or 'sand' when ... How do teams score?" (706020).

Side observation (not this tool): 706003 latched 'para' with N_par=1 from "One house rule". It is inert because a paragraph count needs N >= 2, but it is a detection false fire for the paragraphs rule.

## 2. Rule 13: is the function useful in general?

f(x): given words taken from a natural-language prohibition (a negation before the list, or "... should not appear / is banned" after it), the tool guarantees the output never spells any of those words as a whole word, in any case or tokenization. When the host's next token would complete a forbidden word, the tool substitutes the host's best other token.

Settings outside the benchmark:
1. Brand, legal or compliance banned-term lists, and competitor names in marketing copy.
2. Anonymisation and spoiler avoidance, e.g. keep a person's name or a plot term out of a summary.
3. Word games (Taboo), children's-content profanity filters, and style rules that ban cliche words ("delve", "tapestry").

**USEFUL-GENERALLY: yes.**

## 3. Variants and results

POS = must fire. NEG = must stay silent. "Fired" means the rule latched with the right words. "Strict" is the official checker result (plain then implant). "Same" means the implant output is byte-identical to plain.

| key | type | what it tests | fired | words captured | strict plain / implant | same | note |
|---|---|---|---|---|---|---|---|
| 706000 | POS | rule in system message | yes | cheap, disposable | T / T | yes | host already complied |
| 706001 | POS | multi-turn follow-up "without the word 'romantic'" | yes | romantic | F / **T** | no | "romantic dinner cruise" became "dinner cruise"; "most romantic" became "most beautiful" |
| 706002 | POS | casual, 'dont', no period | yes | delve, tapestry | T / T | yes | |
| 706003 | POS | buried in long message, "is banned" after | yes | synergy | F / **T** | no | became "synergies" (passes \b check; literal compliance) |
| 706004 | POS | middle of prompt + irrelevant sentence | yes | plot, twist | T / T | yes | |
| 706005 | POS | lowercase + forbid in one sentence | yes | very | TT / TT | yes | case rule now correct (lower, not caps) |
| 706006 | POS | forbid + "under 60 words" in one sentence | yes | plastic | TT / TT | yes | fixed by the narrowed frequency words |
| 706007 | POS | "should never appear" after the list | yes | cozy, warm | F / **T** | no | plain wrote "avoids using the words 'cozy' or 'warm'"; implant used warmly/warmth |
| 706008 | POS | "NO 'awesome', NO 'amazing'!!!" | yes | awesome, amazing | F / **T** | no | "amazing" became "pretty good" |
| 706009 | POS | "steer clear of the terms" | **no** | - | T / T | yes | miss (negation synonym); fail-safe |
| 706010 | POS | "Under no circumstances should you use" | yes | delicious | T / T | yes | |
| 706011 | POS | "Banned words for this answer: rich, gold, money." | **no** | - | F / F | yes | miss (header-style list, no determiner); fail-safe but a lost case |
| 706012 | POS | phrase 'money printer' | yes | money, printer (split) | T / T | no | over-block of 'money' changed the explanation |
| 706013 | POS | "'Literally' must not show up" | **no** | - | T / T | yes | miss (subject-first quote, no 'word'); fail-safe |
| 706014 | POS | bullet list after ':' | **no** | - | T / T | yes | miss (list after ':\n'); fail-safe |
| 706015 | NEG | "Feel free to use the word 'fuzzy'" | no | - | - | yes | correct |
| 706016 | NEG | grandmother "never says the word 'okay'" | no | - | - | yes | correct |
| 706017 | NEG | "Don't worry about the word count" | **YES** | count | - | yes | false fire (compound 'word count'); inert here |
| 706018 | NEG | double-quoted rule + information question | no | - | - | yes | correct |
| 706019 | NEG | "no words to describe" | no | - | - | yes | correct |
| 706020 | NEG | Taboo rules "you cannot say 'beach' or 'sand'" + scoring question | **YES** | beach, sand | - | **no** | false fire, harmful: output says `"be" or "s"` and the explanation degrades |
| 706021 | NEG | single-quoted teacher instruction | no | - | - | yes | correct |
| 706022 | NEG | yes/no question "is 'irregardless' banned?" | no | - | - | yes | correct |

Trigger statistics (detection):
- Positives: 11/15 fired. The 4 misses are 706009, 706011, 706013, 706014. Recall = 73.3%.
- Word capture was exact on 10/11 fires. On 706012 the phrase was split into two words.
- Negatives: 2/8 fired (706017, 706020). False-fire rate = 25%.
- Precision = 11/13 = 84.6%.

Generation:
- Positives: strict 11/15 plain and 14/15 implant (+4 gained, 0 lost).
- No enforcement leaks: every fired positive passes the checker.
- Misses failed safe: all 4 are byte-identical to plain.
- Negatives: 7/8 byte-identical to plain. 1 is harmfully changed (706020).

## 4. Verdict against the brief's criteria

- No reward hacking or Python logic: **met**.
- Correct detection and enforcement on most positives, or a fail-safe miss: **met**. 11/15 fired correctly and enforced with 0 leaks; the 4 misses are silent and byte-identical.
- False-fire rate near zero on negatives: **not met**. 2/8 fired. 706020 visibly damages an answer that only asked a question.

**FAIL. Narrow, do not drop.** The function is generally useful and the enforcement is exact.

## 5. What to change (all generic), then re-judge with fresh round-2 variants

1. **Compound guard after 'word/term'.** An unquoted item right after singular 'word' or 'term' that forms a noun compound ('word count', 'word limit', 'word choice', 'term paper') should not be captured. A generic way: the negation must govern a production verb (use, say, write, include, mention, type, utter, employ) before the list, or the list must use the pattern-B form (should not appear / is banned). "Don't worry about the word count" has neither.
2. **Rule descriptions vs requests.** Do not commit in either case:
   - the negation is inside a conditional or temporal clause (when, if, whenever, unless);
   - the message's request is an information question, and the forbid sentence has no response anchor (your answer, reply, response, text, description; "in it"; "when you write") and no imperative position.

   Check on dev that this does not lose 706002-like casual imperatives.
3. **Optional recall work** (misses are fail-safe today):
   - negation synonyms ('steer clear of', 'leave out', 'keep out', 'stay away from');
   - header form "<banned/forbidden> words: a, b, c";
   - a list after ':' + newline with '-' or '*' items;
   - drop 'once' from the commit-killing TIMES set when a negation precedes it ("not even once").
4. **Optional quality work.**
   - Block a forbidden word at its first piece (lookahead over the word's tokenizations), so no fragments like "be" or "s" are left.
   - Keep a quoted multi-word phrase as one slot sequence, so 'money printer' does not block 'money'.
