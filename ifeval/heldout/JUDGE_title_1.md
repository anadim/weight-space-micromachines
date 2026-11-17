# JUDGE_title_1: rule-keeper "title" (title wrapped in <<...>>)

Judge: independent reviewer, round 1. Protocol: bbh_generic/PROTOCOL.md (rules 3, 5, 8, 10-13), brief ifeval/heldout/JUDGE_BRIEF.md.
Code reviewed: circuit.py / tables.py on the development machine, CURRENT version (after NOTES.md change-log parts 4 and 5).
Variants: judge/title_variants.jsonl (23 items, keys 711000-711022). Results: judge/title_results_r1.md (run on the part-5 circuit).

## VERDICT: FAIL (narrow and fix; do not drop)

The detector is now good: it has generic cues, fails safe on its misses, and gives 1 false latch out of 9 negatives.
The ENFORCEMENT path, however, writes with no instruction latched. In 3 of the 9 negatives it inserted '>>' into C++/Python
code. That breaks the module's own contract ("Silent -> write is exactly 0"). The breakage is also hidden by the
silent-identity metric (see finding 1). The fix is one gate (below), plus a re-judge on fresh variants.

## 1. Code findings (current code)

### Finding 1 (BLOCKING): the '>>' swap is not gated by the title rule
circuit.py:524 and circuit.py:648.

```python
self.t_open = torch.where(f('ANGTOK') * a > 0.5, 1.0, torch.where(f('CLOSETITLE') * a > 0.5, 0.0, self.t_open))   # 524
tc = idle * self.t_open * torch.clamp(e_nl + e_eos, 0, 1)                                                          # 648
```

- How it fails:
  - `t_open` is set by ANY '<<' token that the HOST writes in its answer.
  - `idle = (prog==0) * a * body`, and `body` becomes 1 at the start of every answer, whether or not any rule is latched.
  - So whenever the host writes '<<' (a C++ stream, a bit shift, a shell heredoc `<<EOF`, a Ruby/Haskell operator) and then
    wants a newline or end-of-turn, the circuit forces '>>'. `g['title']` is not in the gate.
- Observed on the negatives (title not latched, nothing latched):
  - 711015: `std::endl>>;` and `line break>>.`
  - 711020: `return number << 1>>  # This is a bit-shift operator...`
  - 711022: `<< std::endl>>;` inside a C++ code block.
- These are real breakages: the code no longer compiles.
- The same bug also hits LATCHED title prompts. When the user asks for a title AND C++ code, every later code line that
  contains '<<' gets a '>>' appended.
- Why the headline numbers miss it: score.py counts a prompt as "silent" only when `fires == 0`. These prompts had
  `fires` = 2-3 and nothing latched, so they are left out of the "192/192 silent byte-identical" claim. Negatives carry no
  instruction ids, so no checker sees the damage either. Please add a second metric: "nothing latched -> byte-identical to plain".
- Fix:
  - `tc = idle * g['title'] * self.t_open * ...`
  - Also set `t_open` only from the circuit's own forced '<<' (the end of program 3, `done['title']`). Or close it for good
    after the first '>>' or the first line.
  - Then a requested title plus code in the body is also safe.

### Finding 2: the third-person guard misses possessive-noun subjects
711016, "My editor always wraps titles in <<double angle brackets>>, which I find odd."

- This still latches. The OTHERS list has only pronouns (he/she/they/people/someone...), and DESCV has no 'wraps'.
- A generic fix: require a request marker for title, as the clause rules now do with `reqm` (REQMARK, WHOLE2, imperative
  opening). "My editor always wraps titles..." carries no request marker.

### Finding 3 (minor, quality): the title can swallow the body
The title is "everything from the forced '<<' to the host's first newline".

- When the host does not write a short title, the whole first paragraph becomes the title. Examples: 711002, and 711005
  (plain does the same).
- The checker passes, but the output is poor. This is a design limit, not hacking.

### Clean items
- Python logic at inference: none. The step/write paths are torch.where on 0/1 wires, fixed tables and pointer
  registers. `.tolist()`/decode appear only in report(), which is diagnostic. The newline/EOS identity readout is part of
  the disclosed LM-head readout. Regex appears only offline, in tables.build.
- Reward hacking / IFEval fitting: none found.
  - Detector: '<<' + title/titles/heading/headline in the same sentence, or angle/angular/chevron + bracket(s) + a
    title/you/whole-response word. A sentence-level request guard applies (not an information question, not
    third-person, not negated, "don't forget" is not a negation).
  - Turn scope is fail-safe: earlier user-turn latches are dropped, system latches persist, and cancel cues drop them.
  - The '<<'/'>>' delimiter is the rule's argument. 'angular' matches IFEval's phrasing, but it is an ordinary English
    word, so I do not count it as fitting.
  - gen_data.py has 5 own templates and 2 disclosed drv rewordings (1322, 3606). That is allowed.
- Leakage note (process, not hacking): part 5 ('name'/'header' removed from TITLEW, plus turn-scope reset) was informed
  by running DETECTION on the judges' variants, including mine (711015, 711020, 711022, 711021). Those changes are generic
  and their motivation is disclosed. Even so, the negative-side detection numbers below are not unseen evidence. Round 2
  needs fresh variants.

## 2. Rule 13: USEFUL-GENERALLY

- f(x): "if the user asks for a title marked with << >>, the output's first line is a title wrapped in << >>". In short:
  emit a delimited title line first.
- Settings outside the benchmark:
  1. Document or report generators that split a title from the body on a fixed marker.
  2. CMS / static-site imports and templated pipelines that parse the title field from model output.
  3. Indexing or filing chat outputs (notes, tickets, emails) by a machine-readable header line.
- USEFUL-GENERALLY: yes, but narrow. The delimiter is fixed to << >>. A general version would take the delimiter as an
  argument ([..], '# ', 'Title:'). Rule 12 out-of-style settings work: system message (711000), multi-turn follow-up
  (711001), casual chat (711002), and buried in a long message (711003) all latched correctly.

## 3. Variants and results (plain vs implant; title-checker strict)

| key | type | what it tests | title latched | output vs plain | strict plain -> implant |
|---|---|---|---|---|---|
| 711000 | POS | system message | yes | differs (para rule also latched from "two short paragraphs"; title repeated at end) | T -> T |
| 711001 | POS | multi-turn follow-up, angle+brackets, no literal << | yes | differs | F -> **T** |
| 711002 | POS | casual, end of message, no final punct | yes | differs (whole note inside << >>) | T -> T |
| 711003 | POS | buried mid long message | yes | identical | T -> T |
| 711004 | POS | lowercase + title in one sentence | yes | identical | T,T -> T,T |
| 711005 | POS | terse, at start, '<<>>' | yes | identical | T -> T |
| 711006 | POS | 'heading' synonym | yes | identical | T -> T |
| 711007 | POS | 'name the recipe' synonym | no (miss) | identical (fail-safe) | T -> T |
| 711008 | POS | 'headline' + 'chevron brackets' | yes | differs | F -> **T** |
| 711009 | POS | 'could you ...?' request | yes | identical | T -> T |
| 711010 | POS | "Don't forget to add a title..." | yes | differs (each tip got its own << >>) | F -> **T** |
| 711011 | POS | 'title' and '<<' in separate sentences | no (miss) | differs, but from the para rule ("two-paragraph"), not title | T -> T |
| 711012 | POS | bare yes/no question as request | no (miss) | identical (fail-safe) | T -> T |
| 711013 | POS | 'label' synonym | no (miss) | identical (fail-safe) | T -> T |
| 711014 | NEG | 'angle brackets' about HTML | no | identical | - |
| 711015 | NEG | C++ '<<' + header/name | no | **differs: '>>' forced into code (finding 1)** | - |
| 711016 | NEG | someone else's habit ("My editor always wraps titles in <<>>") | **yes (false fire)** | differs | - |
| 711017 | NEG | "what does it mean when..." question | no | identical | - |
| 711018 | NEG | explicit "do not add a title in <<>>" | no | identical | - |
| 711019 | NEG | instruction only inside a quote | no | identical | - |
| 711020 | NEG | Python bit-shift + 'name' | no | **differs: '>>' forced into code (finding 1)** | - |
| 711021 | NEG | multi-turn, earlier title cancelled ("No title this time") | no | identical | - |
| 711022 | NEG | system message, C++ tutor | no | **differs: '>>' forced into code (finding 1)** | - |

### Trigger statistics (title latch)
- Positives: 10/14 latched, so recall is 71%. All 4 misses were fail-safe for title: the title machinery wrote nothing,
  and the one difference in 711011 came from the paragraph rule. Title instruction strict: plain 11/14 -> implant 14/14
  (+3 gained, 0 lost).
- Negatives: 1/9 latched (711016), so the false-latch rate is 11%. Precision is 10/11 = 91%.
- Negatives whose output was changed by the title machinery: 4/9 (44%), counting the 1 false latch plus 3 unlatched '>>'
  insertions from finding 1. That is not "near zero", and all 3 insertions break code.
- My code-review predictions held for the unguarded first version (7/9 negatives predicted to fire). Parts 4 and 5 fixed
  6 of the 7 at the detector level.

## 4. Reasons for FAIL and required changes
1. (Blocking) Gate the '>>' swap by the title rule. Let only the circuit's own forced '<<' open it, and close it after the
   first line. As it stands, the implant corrupts any answer whose host writes '<<' (C++, shell heredocs, bit shifts),
   with or without a title instruction.
2. Add an "unlatched -> byte-identical" check to score.py. The current silent metric (`fires == 0`) excludes exactly
   these failures.
3. (Recommended) Require a request marker (as the clause rules do) on both title paths, so that third-person descriptions
   with possessive-noun subjects ("my editor wraps ...") do not latch.
4. Re-judge in round 2 with fresh variants that include code answers which naturally contain '<<' (C++ iostream, bash
   heredoc, bit shifts), with and without a title request.

The detector itself is close to PASS. Narrow and fix the rule. Do not drop it.
