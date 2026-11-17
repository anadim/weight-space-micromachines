# JUDGE calc_gen / calc_call, round 2 (calc_call envelope change)

Scope: only the change made for required fix 1 of round 1 (JUDGE_calc_gen_1.md). calc_gen is unchanged; its round-1 PASS
stands. I read build_calc_call and register in teamc_tools.py, NOTEBOOK L7, and the call sites of the 'OE' symbol in
calc.py, calc_neg.py and teamc_tools.py. I compared results_calc_gen_2.json with results_calc_gen_1.json. I opened no BBH data.

## VERDICT: PASS

## What changed
- The response opener `<tool_response>` (OS) is now written when two things hold:
  - the current token is `</tool_call>`;
  - the name token `calc` sits directly after the opening token, and no `</tool_call>` comes between it and the current one
    (`mine_from`).
- So the call is delimited by the envelope, whatever tokens are inside.
- Every call named `calc` now gets at least `<tool_response></tool_response>`. This is the protocol's rule-14 meaning of
  "empty RESULT = error".
- The computation path is unchanged: S[0] is still valid only when the call content parses right after `calc`.

## Is reusing the input symbol 'OE' for '<tool_call>' sound?
Yes:
- **Not read as an input.** calc.py never reads 'OE' as an input condition. The only place it appears is `anysym`, the
  "token has some symbol" test in the unknown-token check, and it is not reached here. `<tool_call>` sits before the name
  token `calc`, which is the call's start class, and the unknown-token check only looks at tokens after the start.
  build_calc_call's new `mine_from` is the only reader of OE.
- **The written token is unchanged.** As an output, OE writes only the first token in its list, `</tool_response>`
  (machine.py `first`).
- **The reader is checked.** make_reader asserts at attach that the reader is exact over the whole vocabulary. That assert
  needs the envelope tokens to have distinct embedding rows. They do after `prep_call_tokens` or the shared fine-tune, and
  the readers are re-solved from the attached model's own embeddings.
- **Small side effect.** `</tool_response>calc ... </tool_call>` (a call name right after a response, with no
  `<tool_call>`) would also count as a call. That sequence is outside the envelope grammar, and the worst result is an empty
  or correct response. Harmless.
- **Window limit.** A call with more than about 50 tokens between `calc` and `</tool_call>` (the K=54 window) still gets no
  response. That is far beyond the `A op B` grammar. A recommended future step is a long-range key, but this does not block.

## Re-run of the round-1 variants (results_calc_gen_2.json)
- **Tool alone: 46/48, the same as round 1.**
  - The two misses, both silent (safe), are unchanged: a16 (expression at the start of the text) and s9 (`is -2.5 * 4`).
  - **There were 0 wrong values.**
- **Only two outputs changed, both as intended:**
  - c12 `calc $4.50 * 3`: no response -> `<tool_response></tool_response>`
  - c14 `calc (2 + 3) * 1.5`: no response -> `<tool_response></tool_response>`
- **Unchanged calc_call results:**
  - c1-c8 are still exact.
  - c9-c11 still give an empty response.
  - c13 (`expr` call) still gets nothing from calc.
- **Host runs: 5/5.** The only difference in the file is the extraction fix (the `<|endoftext|>` that was left on the
  answer), not tool behavior.
- **Team-reported, not seen by me:** test_calls.py 323/323 ok, including `calc $4.50 * 3`, `calc (2 + 3) * 1.5`,
  `calc x * 2`, `calc 2.5 * 4 apples`, and other tools' calls that calc must not answer.

## USEFUL-GENERALLY: yes (unchanged)
f = exact `a op b` (or two operations) on signed decimals. Settings where it helps:
1. money;
2. unit conversions and rates;
3. exact decimal ratios in science and code.

## Remaining recommendations (non-blocking, from round 1)
- Add a start-of-text start class (a16).
- Make run_variants.py host mode use the deployed veto by default.
- Report the calc_gen out-of-style row next to DEV/FRESH (NOTEBOOK L8 appears to be adding this).
