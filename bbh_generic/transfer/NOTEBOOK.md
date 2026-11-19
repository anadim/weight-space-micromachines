# BBH transfer to bigger hosts (Qwen3-1.7B-Base, Qwen3-4B-Base) — integration v2 setup

Question: does the integration-v2 result (ONE frozen Qwen3-0.6B-Base host fine-tuned on 432 generic rows, with 8 generic
micromachines attached: A/B in call form + Team C passive) transfer to frozen Qwen3-1.7B-Base and Qwen3-4B-Base?

Read first: ../integrate/NOTEBOOK.md (INTEGRATION v2), TABLES_v1_vs_v2.md, ../PROTOCOL.md (rules 1-15).

## Hosts / checkpoints
- Same family + variant as the 0.6B host: Qwen3-0.6B-Base -> Qwen3-1.7B-Base, Qwen3-4B-Base (ALL -Base, no thinking,
  greedy; the 0.6B integration host is Qwen3-0.6B-Base). Shared tokenizer (151,936 rows) => readers/flags identical;
  only d_model changes (1024 -> 2048 -> 2560).

## Arms per host (27 real BBH tasks, official 3-shot CoT, greedy, min_tokens 4, max_new 600, bs 32 = v2 harness run27_integ.py)
(a) base                : untouched -Base, --tools none
(b) frozen + Team C pass: untouched -Base + passive expr_gen/calc_gen/bool_gen (--tools C), host-specific veto, NO fine-tune
(c) FT (432 rows) no tools: fine-tuned host, --tools none
(c) FT (432 rows) + 8 tools: fine-tuned host, --tools v2 (A/B call + Team C passive), host-specific veto

## What is re-solved per host (generic/own data only, NEVER BBH)
- readers (embedding -> symbol flags): closed-form, re-solved from each host's embedding matrix (exact over all 151,936 rows).
- write directions D: Adam over the host's own final norm + lm_head (make_out_dirs, 3000 steps over the ~50 output tokens).
- calc_gen veto vector: solved on GSM8K-train number-endings (rows 2000..2300) per host. REUSED the already-solved,
  validated per-host vetoes from the bigger-host GSM runs (results/gsm_bigger) (veto_q17.pt = Qwen3-1.7B-Base, veto_q4.pt = Qwen3-4B-Base), solved via the
  standard solve_veto.py on GSM8K-train = generic data, matching the protocol. Base 0.6B veto NOT reused for 1.7B/4B.
  (Same methodology as 0.6B v2, which reused its base veto for both frozen-C and FT+tools arms.)
- Everything above re-solved automatically inside integ.attach_all / MM.Attached for each host (base AND fine-tuned);
  for the FT host the 4 envelope tokens were randomized then trained, so readers re-solve from the FT embeddings.

## Fine-tune recipe (UNCHANGED from 0.6B v2; teamA/ft.py)
full FT, lr 2e-5, 6 epochs, tok_budget 8000, AdamW(0.9,0.95) wd 0.01, 5% warmup+cosine, loss on completion incl. tool
replies, 4 envelope tokens re-init to distinct random rows first. Data: data/train_union_v2.jsonl = 432 rows
(A 102, B 100, C-passive 100, no-call 100, multi 30). 180,032 tokens / 49,498 target tokens -> 156 steps.
- 1.7B FT: 74 s on H100 (loss -> ~0.02). host runs/ft_v2_17.
- 4B FT:  243 s on H100 (loss -> ~0.02). host runs/ft_v2_4b.
- lr NOT changed for size (kept 2e-5): tiny data + 6 epochs converges cleanly at both sizes (final loss ~0.01-0.02),
  no instability; scaling lr down was unnecessary.

## Compute
- the H100 machine (2x H100) bbh_transfer/ (copy of ../integrate; the frozen 0.6B v2 work untouched).
  NOTE: another agent's Falcon3-3B GSM job appeared on GPU0 mid-run (not mine; left running) -> extra contention on GPU0.
- the A40 machine (2x A40): GPU0 = 0.6B cumulative add-one-tool curve. (GPU1 reassigned by coordinator to another agent.)
- Patches to run27_integ.py: --veto (host veto for C/v2 arms), --machines (explicit subset for the curve). integ.VETO
  reads $WSM_VETO. No change to the computation or the frozen 0.6B code/results.

## Host checks before BBH (test_integ.py --v2; required: silence=bit-identity + per-tool call checks)
Run per host. (results filled below)

### FT 1.7B host check (test_integ --v2 --skip1, veto_q17) — PASS
- check 3 SILENCE: 60/60 all-zero writes, 60/60 bit-identical logits (host untouched when no trigger).
- check 2 e2e via host: track/order/interval/turtle(walk)/truth 40/40 each; expr/bool/calc passive 40/40.
- check 1b (passive C, machine level): expr 444/444, calc 342/342, bool 378/381 (3 = formula > bool_gen 48-tok window:
  silent, fail-safe = identical to 0.6B), passive silent on all 202 A/B training transcripts.
  (check 1 full cross-fire pool skipped for speed; check 1b + 2 + 3 are the required silence + per-tool call gates.)

## Scoring (per coordinator): strict + lenient + no-answer, every arm
- strict = run27_integ extractor: last "the answer is X" before \n\nQ, official norm (../peek/scripts/score.py).
- no-answer% = fraction of items where strict extraction returned '' (template drift). Bigger hosts (esp. 1.7B base)
  drift off the 3-shot template ("So the result is True." with no "the answer is"), inflating no-answer and depressing strict.
- lenient = strict OR a fallback extractor on the same saved output: last (X) option / last integer (arithmetic, counting) /
  last True|False (boolean) / last Yes|No (navigate, web_of_lies) / "answer/result/therefore ... X" regex. Computed POST-HOC
  from the saved rows (out+gold), no re-run. code/report_transfer.py.
- Early signal (partial, base 1.7B): boolean 58.0 strict / 91.2 lenient (36% unextracted); multistep 72.0/80.4;
  hyperbaton 72.8/79.6 -> lenient materially higher for the 1.7B base. FT arms barely drift (no-answer ~0-7%).

## Learning rate
- Kept lr 2e-5 (the 0.6B v2 recipe) UNCHANGED for 1.7B and 4B. Not scaled for size. Justification from OUR data only:
  both FTs converged cleanly in 156 steps (final train loss ~0.01-0.02, no divergence), so no lr reduction was needed;
  a larger model did not require a smaller step here with this tiny 432-row, 6-epoch set.
