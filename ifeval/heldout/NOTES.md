# IFEval held-out redo (Team D): compiled instruction keepers for Qwen3-0.6B (thinking off)

Ran on one GPU (GPU 0 only).
Protocol: ../../bbh_generic/PROTOCOL.md (rules 1-13) applied to IFEval + author updates
(even-key IFEval prompts may be reworded/recombined into dev data; odd keys never opened; judges per tool with rule-11 variants,
rule-12 out-of-style tests, rule-13 usefulness).

## Data (all ours)
- gen_data.py: 60 task requests (our topics) x instruction templates (our wording) + 7 distractor instruction types
  (word/sentence counts, keyword existence/frequency, capital-word frequency, placeholders, letter frequency) that the implants must not break.
  Combinations that the official checkers make unsatisfiable are dropped (e.g. repeat + forbidden words, since the repeated request
  contains the forbidden words).
- DEV = seed 0, 30 prompts per target type as primary (420 prompts). Templates tagged drv:<even key> are OUR rewordings of instruction
  sentences seen in even-key IFEval prompts (allowed by the author for dev); the topic is always ours. Derivation record in each row's meta.tags.
- FRESH = seed 1, 60 per type (840 prompts); its template pool adds >=14 phrasings per type written by an independent agent
  (fresh_templates.json) that I have NOT read. 610/840 fresh prompts contain at least one such phrasing.
- OOD (rule 12) = data/ood.jsonl, 84 items written by the same independent agent, 4 settings (system message, multi-turn follow-up,
  casual, buried in a long message). Not read by me.
- train2 = seed 2 (350), for nothing but extra solver states if needed.
- Scoring: official lm_eval IFEval checkers (instructions_registry via utils.process_results), score.py.

## Circuit (circuit.py, tables.py) -- what runs at inference
See the module docstring. Every prompt goes through the same tensor program; no regex, no Python branching on data.
Readers: fixed 0/1 lookup tables over the vocabulary (single-token words) + an n-gram matcher over the last 5 token ids
(multi-piece words such as 'lower'+'case'). Word lists (tables.W) are generic English lists, see below.

Scopes: clause features (reset after , ; . ! ? newline) and sentence features (reset after . ! ? ; newline).
Clause-level rules are read at the END of the clause (so '...in all capital letters at least 5 times' can be vetoed by 'times').

| rule (function, not IFEval-specific) | detector (generic cues) | argument capture | enforcement |
|---|---|---|---|
| no_comma: never emit ',' | clause: comma-word + negation, no 'more/less/than' | - | veto vector (comma set) |
| lowercase | clause: lowercase-word w/o negation, or capital-word + negation | - | upper-case veto x2 |
| uppercase | clause: capital-word + all/entire/in w/o negation/'times'/'start', or lowercase-word + negation | - | lower-case veto |
| wrap in quotes | clause: quote-word + wrap/in/around + whole/response, not single, not an 'end with' clause | - | force '"' first; at end-of-turn force '"' + EOS |
| end with phrase | clause: end/finish/conclude/last/wrap up w/o start/each/paragraph, then a quoted span or ':' span | span pointers | at end-of-turn: copy span (+ '"' if quoted) + EOS |
| postscript | clause: postscript / P.S. ; P.P.S marker if 'P.P.S' appears | - | at end-of-turn: force '\n\nP.S.' (or P.P.S.), host writes it |
| forbidden words | list region after 'the word(s)/keyword(s)' or a quote near a negation; committed if negation precedes (<=8 tok) or 'should not appear/cannot be/is forbidden' follows; dropped in sentences with times/more/less | rolling hash per word | kill map write on tokens that would complete a forbidden word, gated by readout |
| repeat request | sentence: repeat/restate/copy + request/prompt/text..., not 'do not repeat' | span: message start..end of previous sentence ('above', default) or after the next line break..end of message ('below/following') | copy the span token by token at the start (+ '\n\n') |
| two responses | literal '******' or 'six asterisks' | - | at end-of-turn: '\n\n******\n\n'; then block '***' and end-of-turn until 5 words |
| N paragraphs (*** or blank lines) | number <=3 tokens before paragraph(s) (or section(s) when *** is specified) | N | *** type: insert '***\n\n' after the host's paragraph breaks until N-1 dividers; block host '***'; EOS->'\n\n' while short. Blank-line type: EOS->'\n\n' while short; block '\n\n' once N reached |
| nth paragraph starts with X | sentence: start/begin + paragraph + ordinal/'paragraph k'/'last' + 'with' word | k, span of X | force X (capitalised) at the start of paragraph k |
| title <<>> | literal '<<' or angle/angular + bracket | - | force '<<' at the start; '>>' when the host wants a newline/end |
| highlight >= N | sentence: italic-word, or highlight/bold + section/markdown/asterisk; no 'times' | N (first number after, else before) | at end-of-turn add '\n\n*' + host text + '*' until N |
| exactly N bullets | number <=3 tokens before bullet(s) | N | at end-of-turn add bullets until N; block new bullet lines once N |

Disclosed non-weight pieces: the greedy decode loop (same as plain); the hook on/off switch; three readouts of the host's final state
through the frozen LM head ([top is end-of-turn], [top has a newline], [top is a token that would complete a forbidden word]).

## Change log (DEV-driven)
- v0: first full circuit. Dev detection (prompt tokens only, detect_test.py): all FN for case/comma -> word-list parser bug (fixed).
- v0.1: clause matches latched mid-clause fired 'caps' on 'Use words in all capital letters at least 5 times' (15 FP) -> read clause
  matches at the clause end. 'Please wrap up your answer with "X"' missed (end) -> 'wrap up', 'round off' added to the end-word list.
  Forbidden words: 'no other words after it' captured 'after' -> unquoted lists only after the/these/those/following + word(s);
  'should not appear' committed nothing because 'appear' was in the frequency list -> removed; list region now closes at sentence end
  with the updated quote state; a token like '."\\n' was read as an opening quote -> openers are only (bracket/colon)+quote tokens;
  unclosed quote aborts the region after 12 tokens. Paragraph count: 'two new lines. Paragraph 3' gave N=2 -> number recency is
  sentence-scoped, <=3 tokens, ordinals ('2'+'nd') are not cardinals. Repeat 'below': a later 'before the repeated request' sentence
  re-triggered in 'above' mode -> first hit only; boundary token holding the newline starts the span at once.
  Result: dev detection exact on all 14 types (args too). seed-2 split: 4 nth errors where the required word was 'first'
  (read as the ordinal) -> the captured word is excluded from ordinal reading.

## Rule 13: is each function generally useful? (stated before judging; judges re-assess)
- no_comma (emit no ',' characters): CSV/TSV cell values; legacy SMS/APIs that reject or split on commas; constrained-writing style rules.
- lowercase (all output lowercase): slugs/usernames/tags; brand voice guides with an all-lowercase style; case-insensitive downstream systems.
- uppercase (all output in capitals): radio/telegram/aviation-style messages (NOTAM/METAR); signage and headlines; legacy terminals.
- quote-wrap (whole output inside "..."): producing a string literal / JSON string value; CSV quoting; piping a reply into a shell argument.
- end-with-phrase (output ends with an exact given phrase): mandated sign-offs / disclaimers; protocol terminators ("END OF MESSAGE"); templates.
- postscript (append a P.S. / P.P.S section): emails, letters, newsletters, marketing copy.
- forbidden words (never emit given words, any case/tokenization): brand/legal banned-term lists; profanity or competitor-name filters; word games (taboo).
- repeat-the-request (echo the request verbatim first): read-back protocols (aviation/pharmacy), support-ticket confirmations, audit logs, quoting in email replies.
- two-responses (two alternatives split by a delimiter): A/B copy variants; offering choices; downstream parser that splits on a delimiter.
- N paragraphs with a divider (*** or blank line): CMS templates with a fixed number of blocks; slide/section layouts; markdown docs with rules.
- paragraph k starts with word X: templates that require lead words ("Summary", "Whereas", "Conclusion"); acrostic/structured writing.
- title in <<...>> (title with given delimiters, first line): document generators that parse a title marker; consistent headers for indexing.
- highlight >= N sections (*...*): study notes, key-point emphasis in UI copy, marketing copy.
- exactly N bullets: "give me 3 bullets" executive summaries; slide bullet limits; UI cards that render exactly N items.
All: USEFUL-GENERALLY yes (interfaces are the generic cue words and literal markers, not IFEval sentences).

## Change log, part 2 (generation side)
- solve_vecs.py (v1, v2): solved veto vectors on host states from OUR dev answers (comma, ***, \n\n, \n-start, EOS, bullet-start,
  upper, lower) + kill map + force fixes. Dev run v1 (veto vectors, upper-veto x2, lower-veto for caps):
  P-strict 43.57 -> 80.95 (+162/-5), silent 53/53 byte-identical. Failures that motivated v2:
  * caps 30/43: the fixed lower-case veto makes the host degenerate ("P","est","-","**","**",...; trace of dev key 120) - a fixed
    vector cannot push ~90k lower-case tokens down without wrecking the distribution.
  * paragraphs: the host, primed by its own repetition, writes '***' THROUGH the '***' veto (trace dev 276: 5th divider with no program).
  * forbidden words: kill map leaves garbage ("junioroplevelopleveldata", "art ofMM") and still leaks capitalised variants.
  * two responses 23/34: the second response is a verbatim copy of the first (checker requires different), or the host opens with '******'.
  * end phrase written twice when the host already ended with it: the matcher ignored the closing '!' piece.
- v2 (current): no solved veto/kill vectors at all. One exact mechanism for every "never emit X" rule: the host's own logits are
  read out (DISCLOSED readout through the frozen LM head); if the top token is in the currently blocked set (a 0/1 vocabulary mask
  built from fixed tables x circuit gates: commas; '***' after the divider / in *** mode; '\n\n' once N paragraphs; EOS while a
  required part is empty; bullet starts once N bullets; tokens completing a forbidden word), the emitter forces the host's best
  token outside the set. Case rules = case transducer: the chosen token is respelled through fixed tables (UPSEQ/LOWSEQ: each
  token's upper/lower-case spelling as <= 6 tokens) by the emitter. Anti-repeat: at the start of a forced continuation (paragraph
  after a forced break, second response) the first word of the previous paragraph / first response is blocked.
  End matcher includes the closing piece; highlights appended inline (' *...*') when a blank-line paragraph count is active;
  the first response must have >= 5 words before '******' may appear.
  Forced-token check (diagnostic ffail): 0 failures on the smoke set.

## Change log, part 3 (false-fire guards, after the lead's request; motivated by our own 30 "mention" negatives, data/mentions.jsonl)
- Before: 6/30 mention prompts fired a rule (lowercase x2: "My friend writes ... in lowercase", "Why did e. e. cummings ...";
  caps: "Is it rude to write emails in all capital letters?"; no_comma: "...keyboard has no comma key"; two: "the band Six Asterisks";
  title: "In C++, what does the << operator do?").
- Added generic guards: (a) cue words inside double quotes do not count (quote-parity register; quoted = reported speech);
  (b) a clause/sentence ending in '?' counts only if it is a request ('please', or 'you' + can/could/would/will/mind);
  (c) a clause with a third-person subject (he/she/they/people/someone/...) or a third-person verb form (writes/types/texts/prefers/
  likes/capitalizes/says) does not count for the case/comma/quote/postscript rules; (d) postscript needs an add/include/end/... verb;
  (e) '<<' needs title/heading/name in the sentence (or angle+bracket words); 'six asterisks' needs a separate/divide/between word;
  (f) singular 'comma' followed by a content word is a modifier ('comma key'), plural 'commas' always counts.
- After: mentions 0/30 fire; dev and seed-2 detection still exact (all 14 types, arguments too).
- Dev v2 (readout-gated blocking + case transducer), 300/420 scored so far: P-strict 89.0 (plain 43.6). Remaining failures are
  host degeneration (loops in caps mode), langdetect on all-caps/short text, and checker conflicts (repeat + nth/paragraphs:
  the repeated request itself contains paragraphs / '***').

## Change log, part 4 (after the round-1 judges' CODE REVIEWS, before their variants were run)
The 14 judges' preliminary reviews (code reading) agreed on generic weaknesses; fixed generically:
- Request guards for EVERY rule (not only case/comma): information question, third-person subject/verb, and negation
  ("don't add a title / a P.S. / quotes / N paragraphs" no longer latches; "don't forget to ..." is not a negation).
- Clause boundaries also at 'and'/'but' (so "answer in all caps and do not mention X" no longer turns caps into lowercase).
- Case cues must be about letters: 'capital letters', 'uppercase', 'all caps'/'in caps' (not 'capital city', 'baseball caps',
  'capitalization'); lowercase needs a scope word (all/only/in/entire/response/letters). 'free' is no longer a negation word
  (only for commas: 'comma-free'); 'inverted commas' is not a comma word.
- Quote rule needs quotation MARKS / double quotes or a wrap verb, and a whole-response word (not 'all book titles').
- Counts (bullets/paragraphs/sections): number <= 3 tokens before the item word, not after a determiner/possessive
  ('the 3 paragraphs below'), no unit word in between ('40 words per paragraph'), 1 <= N <= 20, 'bullet' must be followed by
  point/list/item (or be 'bullets/bulleted'); committed at the sentence end through the request guards.
- Two responses: the divider (literal '******' outside quotes, or six asterisks) needs a separate/two/responses word in the
  sentence (read at sentence end; any order).
- Title: '<<' needs title/heading/name in the sentence; the angle+bracket path needs title/you/whole-response words.
- End phrase: needs you/your/whole-response words or the end-word in imperative position (clause opening, after please/also/...).
- Repeat: needs a verbatim/direction word (verbatim, exactly, word for word, above, below, ...); 'above' overrides 'below'.
- nth: start VERB required (start/begin/open/lead), not the ordinal 'first'.
- Forbidden words: commit goes through the request guards; frequency words narrowed to more/than/fewer/less/least/most/times.
- Highlight count > 20 is not a count (default 1).
- Docstring now lists the full readout (host top-token identity + masked hard-max).
- Not changed (design decision, disclosed): latches persist across user turns (instructions accumulate in a conversation, as in
  Multi-IF); a later "cancel" is not understood.
Dev and seed-2 detection still exact after these changes; our 30 mention negatives: 0 fires.

## Change log, part 5 (after running detection on the round-1 judges' variants; these variants are now "seen", so round-2
judges must write new ones)
- Detection on the 311 judge items with the part-4 circuit: most false fires were (a) latches carried from an EARLIER user turn
  into a follow-up ("Thanks! Now ..."), (b) descriptions without any request marker, (c) instructions inside single quotes.
- Turn scope changed to fail-safe: an instruction applies to the reply to the user turn that gives it; system-message
  instructions persist; at each new user turn earlier user-turn latches are dropped; a cancel cue (back to normal / plain text /
  anymore / no longer / this time / fine again ...) also drops system latches. (Multi-turn carry-over is therefore NOT supported:
  "same rules as before" will fail safe.)
- Clause rules (comma/case/quote/postscript) need a request marker in the clause: imperative or instruction verb at the clause
  opening, you/your, please, must/should/need/cannot/have to, a whole-response word, or an opening negation ("No commas.").
- Single-quoted spans are masked like double-quoted ones (reported speech).
- End phrase: an imperative end-word must come with 'with' ("End with: ...", not "Finish this sentence: ...").
- Postscript: verbs that take a P.S. as object (add/include/append/attach/put/insert/end/close/finish/sign/tack/stick), not 'write',
  and not in an explain/describe clause; P.P.S marker selection is unguarded (it only picks the marker).
- Title: '<<' + title/titles/heading/headline ('name'/'header' removed: C++ code); highlight via highlight/bold needs
  markdown/asterisk/'*'/a number in the sentence ('bold opening section', 'star players' no longer fire).
Judge-item detection now (311 items): false fires nc 1, lower 1, caps 1, quote 1, end 0, ps 1, forbid 3, repeat 0, two 0,
paragraphs 0, nth 2, title 1, highlight 1, bullets 0. Dev/seed-2 detection still exact; mentions 0/30.

## Change log, part 6 (after the round-1 VERDICTS: PASS-conditional: bullets, repeat, no_comma, postscript, highlights, quote,
end_phrase, uppercase; FAIL: title, two_responses, nth, forbidden_words, lowercase, paragraphs)
- title FAIL (real bug): '>>' was inserted whenever the HOST wrote '<<' (C++ code), even with no title rule. Now the title is opened
  only by the circuit's own '<<' (end of program 3) and closed by '>>' or a newline; the '>>' swap is gated by the title rule.
- two_responses FAIL: end-of-turn block after the divider counted only Latin-letter words and had no cap (loop to 2048 on a
  numeric answer). Now any word counts, and the block lasts at most 40 tokens after the divider.
- lowercase FAIL: polarity when the negation FOLLOWS the case word ("Capital letters are not allowed"): now silent, unless the
  clause has allowed/used/appear-type words, then the polarity flips; scope words required on both negated forms.
- paragraphs/nth/forbid FAILs (false fires on descriptions): sentence-level rules (counts, title, two, highlight, nth, repeat,
  forbid commit) now need a sentence-level request marker; negation and 'times' are checked per CLAUSE with the rule's cue word
  (not the whole sentence); '*** mode' switch goes through the same guards; 'count/limit/choice/order' end a word list
  ("the word count").
- Request marker made generic: a clause counts as a request if it opens with a content word that is not a subject/determiner/
  wh-/preposition word (imperative opening: 'Write ...', 'Highlight ...', 'Separate ...'), or has you/please/must/should/need/
  cannot/have to/make sure anywhere, or opens with a negation ('No commas.'). Whole-response words alone no longer count.
- Word classes only on word-initial tokens ('book'+'end' is not 'end'); 'all caps' as a phrase (any casing); quote openers may
  carry punctuation (' "-'); single-quote opener needs a space/bracket before it (the possessive in "kids'" no longer opens a
  mask); a '?' inside quotes is not a question; '.P'/'.e' pieces are not boundaries ('.S' still ends a P.S. marker).
- Repeat: not from system messages. Anti-repeat block: only after a break the circuit forced. Bullets: while fewer than N
  bullets, a line the host starts becomes a '*' line (instead of appending filler bullets at the end); empty bullet lines blocked.
- Dev + seed-2 detection exact again, mentions 0/30.

## Change log, part 7 (after round-2 judges' CODE REVIEWS of part 6; their r2 items were then run for detection, so r2 items
became partly seen before the r2 generation run)
- Root cause flagged by all judges: part 6's "imperative opening = any non-subject word" let descriptions with noun/name/adverb
  subjects ("Grandpa wraps ...", "Teachers usually ...") count as requests. Now an imperative opening needs an instruction VERB
  from a generic list (tables.INSTV: write/give/use/keep/put/add/avoid/answer/.../highlight/separate/...); an opening negation
  counts only at a sentence opening; must/should/please/you/... anywhere still count.
- Topic sentences are not requests: explain/describe/tell + why/how/what/when; reported claims (says/said/suggests/...);
  code requests (code/function/program/regex/heredoc/python/bash). Clause rules (comma/case/quote/P.S.) are now committed at the
  sentence end (a later non-request '?' cancels them: "Should I include a P.S. ...?"); a ':' commits them (instruction: question).
- Data masks: text after a colon that introduces data (translate/proofread/convert/... or here/email/text/note/...) up to the
  end of that line / blank line; `backtick` spans; star runs across tokens (' **' + '*,' = '***').
- Smaller: 'never mind' is not a negation; "don't remove/change ... commas" is not a no-comma request; 'stop/quit/cease' are
  negations; partial scope ('just', 'the title', 'names') vetoes whole-response case rules; capB needs 'letters'; polarity flip
  only when allowed/used directly follows the negation; end phrase: request needs must/should/please/an instruction verb, colon
  form only after with/phrase-word/end-word/'is', 'last/final' needs with/is/be; repeat needs a verbatim word (not 'change'),
  negation window 4, 'your previous message' excluded; highlight/bold needs markdown/asterisk/'*' or a section/phrase-type object;
  nth: 'with a/an ...' is not a word; cancel cues narrowed (no 'this time'/'as usual'/'go back').
- Dev + seed-2 detection exact (one seed-2 end miss), mentions 0/30.

## Change log, part 8 (round-2 VERDICTS: PASS quote, uppercase, lowercase, no_comma, end_phrase, title; PASS-conditional bullets,
highlights, postscript, two_responses, forbidden_words, nth; FAIL repeat_request (2 false fires), paragraphs (1 false fire))
- repeat: a verbatim word is REQUIRED (direction words alone no longer suffice: "Copy the text below into bullet points");
  'you/your' within 3 tokens before the request noun vetoes ("Repeat your previous message exactly").
- paragraphs (and all sentence rules): whether/if count as wh-words in the topic test ("Could you tell me whether 3 paragraphs is
  enough"); a sentence that opens with an auxiliary/modal not followed by 'you' ("Should I ...", "Is it ...") is a question.
- postscript: the judge's required fix (make '.S' a non-boundary) broke "... starting with P.P.S Answer in all caps." (two
  instructions merged), so '.S' stays the end of the marker; the question-opening guard handles "Should I include a P.S. ...?".
- two_responses: only a '******' token on its own counts as the divider; the anti-repeat window is anchored on letter words.
- bullets: the forced line-start marker is '-' (a forced '*' was read as italics and closed by the host).
- Dev + seed-2 detection exact (one seed-2 end miss), mentions 0/30.
Host: frozen Qwen3-0.6B, no fine-tuning (PROTOCOL rule 15 not applicable).

## Part 9 (spot checks for the round-2 FAILs; then FREEZE)
- From the repeat/paragraphs/postscript judges' fresh spot-check items (r2b): 'wonder/asked + whether' and reported 'X say ...'
  are topics; the 'your' veto narrowed to your/yours within 2 tokens (so "Could you restate my question" still fires);
  echo/recite/reproduce/reiterate/read added to the instruction verbs; repeat is vetoed when the sentence also has 'into'/'but'
  (reformat requests: "Reproduce the text below exactly, but as a numbered list").
- Known residual false fire: "A witness must repeat the clerk's question word for word ..." (third-party rule with 'must').
- FROZEN: frozen/{circuit.py,tables.py,vecs2.pt} (md5 in frozen_MD5.txt). vecs2.pt is used only for its force_fix rows.

## Judge history (JUDGE_<tool>_<round>.md; all judges opus, independent, each wrote its own variants, rules 10-13 applied)
| tool | round 1 (part-5 code) | round 2 (part-7 code, fresh items) | round 3 (frozen, spot-check) |
|---|---|---|---|
| no_comma | PASS-cond | PASS | - |
| lowercase | FAIL (polarity when negation follows the case word) | PASS | - |
| uppercase | PASS-prov | PASS | - |
| quote | PASS-prov | PASS | - |
| end_phrase | PASS-cond | PASS | - |
| postscript | PASS-cond | PASS-cond (required fix) | PASS |
| forbidden_words | FAIL (false fires) | PASS-cond | - |
| repeat_request | PASS-cond | FAIL (2 false fires) | PASS |
| two_responses | FAIL (EOS block runaway) | PASS-cond | - |
| paragraphs | FAIL (counts without request marker) | FAIL (1 false fire) | PASS |
| nth_paragraph_word | FAIL (false fire) | PASS-cond | - |
| title | FAIL ('>>' inserted without a latch) | PASS | - |
| highlights | PASS-cond | PASS-cond | - |
| bullets | PASS-cond | PASS-cond | - |
Caveats stated by the judges: (1) parts 5-9 were shaped using detection runs on the judges' own items, so their precision numbers
are partly in-sample; (2) the 11 tools that passed in round 2 were judged on the part-7 code; the frozen code adds the part-8/9
changes (all narrowing guards, plus three enforcement tweaks: exact '******' divider, letter-word anti-repeat window, '-' bullet
marker). Recall on judges' novel phrasings is modest (roughly 40-75% per tool); misses fail safe (byte-identical to plain).
