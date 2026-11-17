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
