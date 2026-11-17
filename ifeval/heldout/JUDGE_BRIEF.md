# Judge brief (one judge per rule-keeper / tool)

You are an independent, skeptical reviewer of ONE compiled "rule-keeper" (implant) that is attached to a frozen chat LM
(Qwen3-0.6B, thinking off). Binding protocol: bbh_generic/PROTOCOL.md (rules 1-13; read rules 3, 5, 8, 10-13 carefully;
it was written for BBH and applies equally to IFEval here). Context: the implants are evaluated once on real IFEval at the end;
development used only the team's own generated prompts (plus rewordings of even-key IFEval instruction sentences, explicitly allowed
by the author for dev data). Odd-key IFEval prompts were never opened.

Files (ifeval/heldout/):
- circuit.py (the whole compiled circuit; every rule shares the reader/scopes/emitter), tables.py (vocabulary tables + word lists),
- gen_data.py (the dev data generator: our templates; 'drv:<key>' marks our rewordings of even-key IFEval instruction sentences),
- NOTES.md (design table + change log), run.py (harness), detect_test.py.
Do NOT open: data/fresh.jsonl, fresh_templates.json, data/ood.jsonl (held-out test material), any IFEval data (ifeval.jsonl, input_data,
odd or even keys), anything under ifeval/ outside heldout/. Do not run models on the development machine.

Your tool: {TOOL}

Tasks:
1. Code review of this tool's path through circuit.py/tables.py: look for reward hacking / test fitting (closed vocabularies tuned
   to IFEval, IFEval-specific strings or templates, behavior that only works on IFEval's wording), Python logic at inference
   (Python branching on data, regex, Python rule objects; tensor ops / tables / gather / where on 0-1 wires are allowed;
   the three LM-head readouts are disclosed and allowed), leakage from test data, fixes justified only by memory of IFEval.
2. Rule 13: state the function the tool computes without reference to IFEval, name 3 settings outside the benchmark where it helps,
   and record USEFUL-GENERALLY yes/no.
3. Rule 11 + 12 + lead's request: write >= 12 test items for this tool that the dev data would NOT contain
   (>= 8 POSITIVE: phrasings not in IFEval or gen_data.py: synonyms, different sentence order, the instruction in the middle of the prompt,
   two instructions in one sentence, casual wording, system message, multi-turn follow-up, buried in a long message;
   >= 4 NEGATIVE: prompts that MENTION the rule without asking for it, e.g. describing someone else's habit, asking what the rule/marker
   means, quoted text containing the instruction, the cue words used in another sense). Details: paraphrases (synonyms, different negation,
   casual wording, odd punctuation), different positions (start/middle/end of the message), mixed with other instructions, an extra
   irrelevant sentence, AND out-of-style settings (instruction in a system message, a multi-turn follow-up with history, casual chat,
   buried in a long message). Also include >= 2 NEGATIVE items where the tool must stay silent (text that mentions the cue words
   without asking for the constraint). Write them as JSON lines to ifeval/heldout/judge/{TOOL}_variants.jsonl with fields:
   key (int, {KEYBASE}+i), prompt, optional system, optional history (list of {role, content}, ending with an assistant turn),
   instruction_id_list and kwargs in lm_eval IFEval format (negatives: an empty list and empty kwargs), and "note" (what it tests).
   IFEval ids/kwargs: no_comma->punctuation:no_comma {}; lower->change_case:english_lowercase {}; caps->change_case:english_capital {};
   quote->startend:quotation {}; end->startend:end_checker {end_phrase}; ps->detectable_content:postscript {postscript_marker: "P.S." or "P.P.S"};
   forbid->keywords:forbidden_words {forbidden_words: [lowercase]}; repeat->combination:repeat_prompt {prompt_to_repeat: exact substring};
   two->combination:two_responses {}; paras->length_constraints:number_paragraphs {num_paragraphs}; nth->length_constraints:nth_paragraph_first_word
   {num_paragraphs, nth_paragraph, first_word}; title->detectable_format:title {}; high->detectable_format:number_highlighted_sections {num_highlights};
   bullets->detectable_format:number_bullet_lists {num_bullets}.
4. Then STOP and report back (preliminary review + the variants path). The builder will run your items (detection + generation,
   plain vs implant) and send you the results. Then give the final verdict: PASS requires no reward hacking / Python logic,
   correct detection+enforcement on most positive variants or failing safe (silent, output identical to plain), and a false-fire rate
   near zero on the negatives (report trigger precision/recall on your positives and negatives); FAIL otherwise, with concrete
   reasons (and say whether the rule should be narrowed or dropped). Write the final verdict to ifeval/heldout/JUDGE_{TOOL}_1.md (code findings, USEFUL-GENERALLY + 3 settings,
   the variants and their results, verdict).
