# BBH integration step (protocol rules 14 + 15): one host, all generic tools

Binding protocol: ../PROTOCOL.md. Runs: two GPU machines (an H100 node, GPUs 0/1, and an A100). This folder: code/, data/.
The original run used byte-identical vendored copies of micromachines/ and bbh_generic/; the code here imports the repo's
(code/_paths.py).

## 0. What was read
PROTOCOL.md; NOTEBOOK.md of Teams A, B, C and of the bbh_ft100 agent; the teams' code (tools, callfmt/tools_ref/callfmt_C,
ft.py, ft_B.py, make_train_C.py, run_eval/run_gen harnesses, test scripts). No BBH test data was opened; the 27-task run
reads it only inside the harness and only aggregate scores are printed/read.

## 1. Setup (2026-11-17)
- Code copied (md5 checked identical to the copies the teams used): micromachines/ (repo), teamA (track_gen, order_gen),
  teamB (circuits_B: interval_gen, turtle_gen, truth_gen; v3 / walk v3.1), teamC (teamc_tools: expr_call, bool_call,
  calc_call + passive expr_gen/calc_gen/bool_gen).
- `code/integ.py`: ONE micromachines `Attached` object holding all 8 machines (one hook before the final norm, writes
  added, shared output tokens share one solved write direction); Team C's float64 stack + start-of-text 'ANY' flag applied
  as in teamc_tools.attach_gen.
- DECISION, Team C form: the integrated host uses Team C's CALL form (expr / bool / calc in the shared envelope).
  Reasons: (1) rule 14 names calc/expr/bool as envelope tools and asks for one call format; (2) the composition test must
  measure "did the host call every needed tool", which is only well defined for explicit calls; (3) the integrated host is
  fine-tuned anyway (A and B need it), and Team C's 100 call-format rows are part of the ~300-row union; (4) a passive '='
  trigger would also fire inside the host's free text on every BBH task, mixing two mechanisms. The passive form needed no
  fine-tune and is Team C's reported configuration; it is kept as an optional extra arm only (see results).

(log continues below)

## 2. Fine-tune (rule 15: ~100 rows per team)
- `code/make_union.py` -> data/train_union.jsonl = 302 rows, copied verbatim from each team's own seed-2 data:
  A 102 (teamA/data/train100.jsonl = the ft100a file: track bbh 25 / alt 8 / code 7 / mix 12, order bbh 25 / alt 10 / mix 15),
  B 100 (the bbh_ft100 agent's stratified subsample sub/B_n100_s0 of Team B's ft3 rows: temporal/lab/navigate/hiker/lies/rumor
  16-17 each, 13 error-recovery rows; turned into prompt/completion exactly as ft_B.py: official 3-shot prompt w.p. 0.8 for
  the BBH-like styles, zero-shot otherwise, masked broken call in the prompt), C 100 (teamC/data/train/train_C_100.jsonl:
  arith/bool/count/shop 17, dshop/bcode 16; zero-shot prompts).
- Recipe = Team A's ft100a, script teamA/ft.py UNCHANGED: full fine-tune of Qwen3-0.6B-Base, lr 2e-5, 6 epochs, tok_budget 8000,
  AdamW (0.9, 0.95), wd 0.01, 5% warm-up + cosine, loss on the completion incl. the tool replies, the four envelope tokens
  re-initialised with distinct random rows (seed 1234) first. 302 rows, 139,808 tokens (33,637 target tokens) ->
  **126 optimizer steps, 6 epochs**, 74 s on the A100. Host = runs/ft_int300 (not released; recreate with teamA/ft.py on data/train_union.jsonl).
- Rows the integrated host saw per team = the same rows the single-team 100-row hosts saw (A: ft100a; B: bbh_ft100 B_n100_s0),
  but with a different recipe for B (B_n100_s0 used ft_B.py, lr 1e-5, 15 epochs, 60 steps).

## 3. Speed fix (no change of the computation)
Team C's expr/bool call circuits are ~1000 layers deep, so one decoding step of the 8 attached machines took ~240 ms
(launch-bound; host 35 ms). `integ.GraphStep` replays Attached.step for a fixed batch as one CUDA graph: 40 ms/step at bs 16.
check_graph.py: writes bit-identical to the plain Attached.step over 57 steps x 16 sequences containing calls of all 8 tools
(124 non-zero writes). The harnesses (eval_integ.py, run27_integ.py) use it; test_integ.py uses the plain step.

## 4. Composition set (frozen before any model output on it was produced or read; the 60-item harness smoke test ran 2 of its items, outputs not read)
`code/compose.py` -> data/compose.jsonl, 45 questions, 9 families x 5 (F1 track->expr raffle tickets; F2 interval->expr
free studio hours; F3 walk->expr mowing robot street distance; F4 truth->expr quiz-show points; F5 order->expr cycling
prize coins; F6 track->bool card-passing win rule; F7 interval->bool can the club meet; F8 interval->expr->calc decimal
rental cost; F9 walk->expr->calc drone fuel). Own names/settings/wording; gold = the teams' Python references.
Generator fixes made before any model run (looked only at the generated questions/golds): str.format placeholder bug;
track/order acceptance sets narrowed to the queried slots; walks forced to contain a turn and distance >= 5; F7 balanced
(2 fully booked, 3 with free time).

## 5. Runs (all greedy; harnesses = each team's prompts/extraction/scoring, eval_integ.py; real BBH = run27_integ.py)
- A100: sets FT+tools, sets FT-no-tools, BBH FT-no-tools, BBH FT+tools tasks T1 (first 14 alphabetically).
- H100 GPU1: test_integ (base, then FT host), sets base, BBH base. GPU0: Team A round-3 variants (3 arms), BBH FT+tools
  tasks T2 (last 13), extra arm passive-C.
- Extra arm DECIDED after seeing C_test with the call form (55.0 vs Team C's 98/100/37) and BEFORE any BBH tool-arm number:
  "FT + all 8 call tools + Team C passive expr_gen/calc_gen/bool_gen" on C_test, C_ood, compose only. Not run on real BBH
  (BBH is run once, with the pre-registered primary configuration).

### Composition: first read (FT + tools)
0/45 questions with every needed tool called. The host calls the FIRST ("story") tool almost always and correctly for
interval 15/15, truth 5/5, walk 7/10 (track 2/10, order 1/5 correct), then answers from the reply without any expr / bool /
calc call (0 of 55 needed). Failure modes: the single-tool script of its training rows ("call -> answer"); 2-digit values put
into track (track 53204150:...); a Euclidean distance instead of a street distance; repeated identical calls until the
token limit (walk, interval; Team B's retry behaviour). Answer accuracy 11.1% (5/45, all on the True/False families).
- Real BBH runs split by task halves to fit the time (T1 = first 14 tasks alphabetically, T2 = last 13): FT+tools T1 on
  the A100, T2 on H100 GPU0; base T1 on H100 GPU1, T2 on GPU0 (the first full base run was stopped after 2 tasks,
  its partial output discarded unseen); FT-no-tools as one run on the A100. Same script/settings for every part
  (max_new 600, bs 32, greedy); report.py merges the halves. Only per-task aggregates are read.
- test_integ on the frozen base (prep_call_tokens): check 1 all pass (4,470 call x context cases: A rand 450+450,
  malformed 123, B fuzz 1350 + corrupt 420 + other 9, C random 900 + edge 69, gold 240 per team), e2e 40/40 per tool,
  bit-identity 60/60 texts.
- FT+tools T1 was further split (T1a = first 7 tasks on the A100, T1b = next 7 on H100 GPU0 after T2): the T1 run was
  stopped after 3 tasks (its log renamed *.KILLED.log, no output file was written, nothing read) and restarted as T1a/T1b.
  run27_integ.py now also writes OUT.partial after each task (scores only are read).

# INTEGRATION v2 (lead's request, 2026-11-18): three generic fixes, everything else identical
1. Team C in its PASSIVE form (expr_gen, calc_gen with Team C's solved veto, bool_gen; fire on the host's own `EXPR =`);
   A and B tools unchanged in call form. integ.TOOLS_V2 (8 machines, one Attached). Team C's 100 call-form rows are
   converted to the passive form (make_v2_rows.py: `<tool_call>NAME ARGS</tool_call><tool_response>R</tool_response>` ->
   `ARGS = R` in the passive tool's own output format: expr ' R.\n', bool ' R.', calc ' R').
2. "When NOT to call" rows: 100 general prompts written by me (nocall_prompts.py: everyday why/how, definitions, short
   writing, advice, general reasoning in prose; no option lists, no puzzles, no arithmetic, nothing BBH-like), answered by
   the FROZEN BASE's own greedy output (make_nocall.py, self-distillation): 50 zero-shot, 50 behind a 3-shot prefix of my
   own 9 exemplars (exemplar answers also the base's). Same "Q: ...\nA: Let's think step by step." scaffold as all rows.
3. 30 multi-tool rows of 5 NEW kinds (not the 9 frozen test kinds; new tool pairs and settings): M1 truth->bool (club rule),
   M2 order->bool (bake-off jury rule), M3 walk->bool (garden rover rule), M4 track->calc (market stalls swap decimal price
   tags), M5 order->calc (relay prize P / place). call -> use the result -> passive expression. The 45-question
   composition test stays frozen and unseen (not opened while writing these).
- Union v2 = 432 rows (A 102, B 100, C-passive 100, no-call 100, multi 30). Same recipe (teamA/ft.py, lr 2e-5, 6 epochs,
  tok_budget 8000): 180,032 tokens, 49,498 target tokens -> 156 steps, 91 s. Host runs/ft_int_v2 (not released).
- Arms: frozen base (reused from v1: same model, same harness, same settings), v2 FT no tools, v2 FT + v2 tools.
- test_integ.py --v2: C call names have no owner any more (all silent); new checks: passive tools exact on random
  expressions/formulas/decimal ops in 3 contexts with the other 7 machines silent, passive tools silent on all 202 A/B
  training transcripts, e2e through the host; bit-identity texts have '=' replaced (passive tools fire at '=' by design).
- BBH v2 split in 3 interleaved task thirds (P0 = tasks[0::3], P1, P2) per arm, run27_integ.py, max_new 600, bs 32.
- v2 BBH FT+tools P0 split: its last 4 tasks run as P0b on the second A100 slot; P0 is stopped after its 5th task
  (multistep_arithmetic_two) and its per-task partial file is used (P0 -> P0a).
- test_integ --v2 on the base first crashed after check 1 (import path of calcfmt); fixed and re-run with --skip1 for the
  base (its check-1 log kept as test_integ_v2_base_check1.log); the FT run is a full run. A brief third process on
  H100 GPU 0 (the ~10-min base re-run) overlapped two BBH jobs.
- v2 Step 1 (base and FT v2 host, identical results): A/B own-call + silence check 4,470/4,470 (expr/bool/calc CALLS now
  have no owner: all 8 machines silent on them); passive expr 444/444, bool 378/381 (3 = a formula longer than bool_gen's
  48-token window: silent, fail-safe), calc 342/342 (first operand non-negative: Team C's documented limit), with the
  other 7 machines silent; passive tools silent on all 202 A/B training transcripts; e2e via host: A/B 40/40 each, expr
  40/40, bool 39/40 (same over-long formula), calc 39/40 base / 38/40 FT (calc_gen silent on '5.2 / -85.57' and
  '51 + -10.39' after a newline; the host then wrote its own wrong value); bit-identity 60/60 (texts with '=' removed).
- v2 FT+tools BBH final layout (to finish in time; same script/settings everywhere): P0a (boolean_expressions, disambiguation_qa,
  file, stopped after logical_deduction_seven_objects), P0x (multistep_arithmetic_two), P0b (last 4 of P0), P1a (P1's
  first 7 tasks, stopped after snarks), P1x (tracking_shuffled_objects_five_objects, web_of_lies), P2. If a stopped part
  had already finished a task that is also in an x part, the two runs are identical configurations; report keeps one.
  CORRECTED v2 BBH layout: P0a = boolean_expressions, disambiguation_qa, geometric_shapes (P0 stopped after geometric);
  P0x = multistep_arithmetic_two; P0y = logical_deduction_seven_objects; P0b = penguins_in_a_table,
  salient_translation_error_detection (P0b stopped after 2 tasks, partial used); P0z = temporal_sequences,
  tracking_shuffled_objects_three_objects; P1a = first 7 of P1 (stopped after snarks); P1x = tracking 5, web_of_lies; P2 complete.

## v2 RESULTS (full tables: TABLES_v1_vs_v2.md, code/report_v2.py; transcripts compose_transcripts_v2.txt)
Task means, base / v1 FT no tools / v1 FT+tools / v2 FT no tools / v2 FT+tools:
A test 29.0/25.0/88.8/22.8/83.8; A oos2 20.0/22.8/68.9/28.3/47.8; A judges round 3 3/16, 3/16, 7/16, 3/16, 7/16;
B test 44.0/69.7/95.0/70.3/91.7; B oos 11.3/23.3/80.0/21.3/70.7; C test 55.0/40.7/55.0/35.3/71.0; C ood 62.2/35.6/50.0/35.6/97.2.
Composition: answer 8.9 / 17.8 / 11.1 / 15.6 / 24.4 (lenient v2 26.7); every needed tool used 0 / 0 / 0 / 28.9 / 24.4
(v2: passive use = an arithmetic/boolean expression followed by '='); v2 per tool (needed/used/correct): track 10/8/3,
order 5/5/5, interval 15/15/15, walk 10/10/10, truth 5/5/5, expr 35/10/2, bool 10/2/0, calc 10/9/2.
Real BBH: 27 mean 40.98 / 32.13 / 47.98 / 33.61 / 48.36; 12 tool tasks 43.27 / 38.63 / 82.73 / 38.10 / 84.33;
15 no-tool tasks 39.15 / 26.93 / 20.18 / 30.01 / 19.58; call rate on the 15 no-tool tasks v1 FT 88.4-88.6% -> v2 FT 59.2%.
Reading: fix 1 (passive C) works (multistep 33.2 -> 100, C ood 97.2); fix 2 (no-call rows) lowers spurious calls
(88.6 -> 59.2%) and lifts the no-tool mean WITHOUT tools (26.9 -> 30.0) but not WITH tools (19.6): attached A/B tools
answer the spurious calls (mostly track/interval/order) with empty replies and the host loops on retries; fix 3 moves
composition off 0 (11/45 use every tool, 24.4% answers) but the arithmetic step is mostly done wrong or not via '='.
Cost: A oos2 -21 pts, B oos -9 pts vs v1 (more varied rows, same 6 epochs).
