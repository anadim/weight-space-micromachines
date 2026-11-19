# RL with a whole tool library attached (TRY_LEDGER item 41) — 2026-11-19

Question: GSM8K RL on Qwen3-0.6B-Base with a LIBRARY of frozen implants attached at once (calculator + 8 BBH tools +
IFEval 14-rule circuit), each behind its own learned in-wire veto + zero-init out-adapters. Does RL make the model use
any tool beyond the calculator? Control: same library with every non-calculator tool's output deranged.

## Setup
- Host Qwen3-0.6B-Base, Dr.GRPO, lr 1e-6, G 8, 64 prompts/update (deadline guard may cut, see below), T 1.0, no KL,
  300 updates, max_new 320, GSM8K-train + scaled pool (split A), dev split B n=800 greedy every 25 (+u64), test at u300
  (GSM8K test 1319 + GSM-Hard 1319, questions > 224 tokens dropped as in rl_wiring).
- Tools (all frozen; tool index order = TNAMES):
  0 `calc` = calc_s4 + veto, CW-lean wiring exactly (hand write as warm final link + cw out-adapters @14/final + in-wire);
  1-8 integ.TOOLS_V2 = track_gen, order_gen, interval_gen, turtle_gen, truth_gen (Team A/B, CALL form, fire only inside
  `<tool_call>` envelopes), expr_gen, calc_gen, bool_gen (Team C PASSIVE form: fire at the host's own '=' / 'is'), with
  Team C's veto on calc_gen; designed writes kept (summed before the final norm, as integ.Attached), gated by their in-wire;
  9 `ifeval` = Team D's frozen circuit (circuit_g.py, graph-safe copy, md5-verified frozen tables/vecs2.pt from
  the frozen IFEval dir), frozen LM-head snapshot of Qwen3-0.6B (the host it was solved against).
- Per tool wiring (CW-lean layout): in-wire veto (b, v·LN(h14[t-1]); p(fire)=0.05+0.9σ in rollouts, exact log-probs in the
  update; lr 1e-3, Adam eps 1e-8) + out-adapters g(h)·B·relu(A e + c) at L14 and final (width 32, B=0 at init).
  IFEval adapters read the normalized write (1024-d); its L14 adapter sees the PREVIOUS position's write (the write needs
  the final state). Trainable wiring 775,518 params = 0.13 % of host = 4.95 % of the implants' nonzeros (15.7 M, IFEval
  circuit tables not counted).
- Frozen snapshots: BBH readers read a frozen copy of the base input embedding (with Team C's 4 envelope-token rows
  re-initialised IN THE SNAPSHOT ONLY: the base checkpoint gives the 4 envelope tokens one shared untrained row, so no
  reader can separate them; the host's own rows are untouched). calc reads the live embedding exactly as CW-lean.
- Engine (`lib_engine.py`): rl_wiring Engine + every tool stepped inside the per-token CUDA graph. calc first on the main
  stream (persistent kernel, never concurrent); the deep BBH stacks (expr_gen 667 layers fp64, bool_gen 850 layers) run
  with dense per-layer matrices (`DenseStack`, torch.equal vs the cuSPARSE reference on the real 8-shot prefix windows;
  random 0/1 windows outside the design range differ) each on its own stream, with the other tools and the IF token step
  on side streams, overlapping host layers 0-13; join at L14. IF circuit state is held for B_max rows and stepped for all
  rows (dead rows ignored). IF write at the first answer token (last prompt position) is not computed (disclosed).
- Scrambled control (`--scramble 1`): each BBH tool's write rows deranged (fixed seed 4242 derangement of its output
  symbols); IFEval write dimensions permuted. Adapters still read the tool's (unpermuted) one-hot e. calc untouched.

## Checks
- Bit-identity: u0 greedy dev with every tool detached (`toolsoff`) == host-only (hash). Smoke (16 dev items): identical.
- Update-pass consistency (smoke): importance ratio 0.9997, clip fraction 0.0015.
- HF check (u1 of every run): |logp sampler − logp HF teacher-forced| on sampled tokens without an in-wire decision,
  HF transformers host + tool features recomputed teacher-forced (Micromachine.emit), adapters via hooks. -> table below.

### LIB s0 checks (n=800 dev, H100)
- u0: dev 50.1 (GSM 62.5 / scaled 37.8); host-only 44.25; calc-only (library detached) 50.4 (CW-lean u0 50.4).
- Bit-identity: tools-off greedy dev hash 6f281a66b9e9 == host-only hash 6f281a66b9e9 (all 800 answers identical).
- HF check u1 (48 rows, 5,636 sampled tokens without an in-wire decision): |dlogp| mean 0.017, p99 0.13, max 0.32;
  on 504 tokens where a tool wrote: mean 0.003, max 0.16. Decode-time vs teacher-forced per-tool write flags: 0
  mismatches over 1,043 write events (decode DenseStack/step path vs Micromachine.emit reference).
- Timing: ~50 s/update (rollout ~35 s, update ~13 s), 64 prompts kept; peak alloc ~35 GB.
- Rollout u1-3: calc 1.6-2.0 triggers/answer, expr_gen ~1.0-1.2, calc_gen ~0.5; every other tool 0 (fire 0.9 = init).
  expr_gen and calc_gen are passive '=' tools: they trigger on the host's own equations, next to calc_s4.

### IFEval-reader silence on GSM (if_silence.py, pod CPU, token-level circuit only)
- Over the exact token stream the engine feeds it (8-shot prefix + 'Question: ...\nAnswer:'): 8-shot prefix 0 latches;
  dev-B prompts 0/800 latch any rule; GSM8K-test prompts 1/1319 (rule 'para', "...cut into 4 equal parts...").
- Assistant-role register never set (no chat template) -> every IFEval gate is 0 -> write exactly 0 on all prompts.
- In-run (prompt + generated answer, logged per eval): latch-any rate on greedy dev 0.000 (u0) / 0.003 (u25); rollouts
  ~0.4 % of rows; IF write events 0 (no answer-mode).

## Speed (H100 host H100, shared with another job)
| config | decode ms/step @128 rows |
|---|---|
| host only | ~5-6 |
| + all tools, no IFEval | ~53 |
| + IFEval | ~63 |

## Runs
| arm | host/GPU | status |
|---|---|---|
| LIB s0 | H100 host GPU1 (H100) | running since 04:26; 64 prompts; ~50 s/update alone (97 s while a filler job shared the GPU 05:00-05:40, paused by the scheduler) |
| LIBscr s0 | H100 host GPU0 (H100) | running since 04:51; 64 prompts; ~51 s/update |
| LIB s1 | A40 host GPU0 (A40) | running since 05:40 (first 9 launches died on an IFEval-dir path bug, fixed); A40 ~2.7x slower -> guard will cut prompts; dev every 50 (+u25) |
| LIBnoIF s0 | A40 host GPU1 (A40) | running since 05:19; guard cut to **40 prompts/update** at u5 (119 s/update at 64); dev every 50 (+u25) |

Deadline guard: at u5 and every 25 updates after, prompts/update is cut (never the number of updates) so that the
remaining updates + dev evals + LOO + test fit before the deadline (13:15 UTC for the H100 arms, 13:45 for A40 host).
Seed-1 and noIF runs on A40 differ from the s0 arms in GPU type (bf16 tiling), prompts/update, dev cadence and
tok_budget 2048 (memory only); compare them to s0 with that in mind.

## Results

### Final, seed 0, H100, 64 prompts, 300 updates (greedy dev-B n=800, SE ~1.7 pp; test n=1319 each, SE ~1.3 pp)
| | LIB s0 | LIBscr s0 | CW-lean s0 (ref, KL 0.01) | arm 2 ref (calc, hand) | CW s0 (ref) |
|---|---|---|---|---|---|
| dev u0 | 50.1 | 16.6 | 50.4 | 49.8 | 50.4 |
| dev u100 / u200 | 66.8 / 67.0 | 57.2 / 65.0 | 58.6 / 65.9 | 64.5 / 68.8 | — |
| dev u300 (GSM / scaled) | **68.6** (73.8 / 63.5) | 67.9 (79.0 / 56.8) | 66.2 (— / 58.2) | 70.0 (77.0 / 63.0) | 67.5 |
| dev u300, library detached (calc_s4 only) | 67.1 | 67.9 (identical: broken tools fully vetoed) | | | |
| dev u300, every tool off | 56.2 | 60.6 | | | |
| dev u300, host only (tools + adapters off) | 59.6 (GSM 80.8 / scaled 38.5) | 60.1 | | | |
| GSM8K test u300: full / calc_s4 only | **69.4** / 67.9 | 69.3 / 69.3 | | ~67 | 66.3 |
| GSM-Hard u300: full / calc_s4 only | **56.1** / 53.4 | 51.9 / 51.9 | | ~47.5 | 51.1 |

Leave-one-out on dev at u300 (LIB s0; full 68.6): − calc_s4 68.9 (+0.25), − expr_gen 67.0 (−1.6), − calc_gen 68.4
(−0.25); the other 7 tools (track, order, interval, turtle, truth, bool, IFEval) had 0 triggers on dev, so removing
them changes no token. Removing all three '=' calculators: 56.2 (−12.4). LIBscr s0: − calc_s4 = all off = 60.6 (−7.3);
removing a scrambled tool changes nothing (they are vetoed at every call).

Per-tool use (greedy dev u300, triggers per answer / % fired): LIB calc 3.06/100 %, expr_gen 2.25/100 %, calc_gen
0.93/100 %; LIBscr calc 2.39/99 %, expr_gen 1.37/**0 %**, calc_gen 0.69/**0 %**. Rollout fire rates (floor 5-95 %): LIB
all three ~95 % throughout; LIBscr expr_gen 88 % -> 9 % at u25 -> 5 % (the floor), calc_gen 90 -> 5-8 % by u50; calc
84-94 %. Triggers of the vetoed tools climb back (0.1 -> 1.0-1.4 per answer) once the veto is learned: the host goes back
to writing equations, and the in-wire, not a text dodge, keeps the broken tools out (same as CWinR).

IFEval circuit: silent on GSM. Readers latch on 0/800 dev prompts and 1/1319 test prompts; during generation ~0.1-0.5 %
of rows latch some rule. The assistant-role register is never set, so the write is exactly 0 everywhere: 0 IF write
events in all rollouts and evals of all runs.

### Reading
- **The non-arithmetic tools are ignored**: the 5 call-form BBH tools, bool_gen and IFEval never trigger on GSM8K text
  (0 triggers in ~300 × 512 rollouts). RL has no reason to act on them, and it does not.
- **RL uses every arithmetic tool it is given, and keeps the correct ones**: expr_gen and calc_gen (passive Team C
  tools that fire at the host's own '=') stay on next to calc_s4 (fire ~95 %, the in-wire ceiling). When they are
  scrambled, RL vetoes them within ~25 updates and keeps calc_s4. The in-wire acts as a per-tool competence filter.
- **The three calculators are largely redundant**: removing any one costs ≤ 1.6 pp on dev (≤ 1 SE); removing all costs
  12.4. calc_s4 alone is not needed when expr_gen and calc_gen are there (LOO +0.25).
- **Small net gain on hard numbers, none on plain GSM8K**: LIB vs scrambled at u300: GSM8K test +0.1, GSM-Hard +4.2,
  dev +0.75. The dev gain is on the scaled half (+6.7) with a loss on the GSM half (−5.2). LIB vs its own calc-only:
  test +1.5, GSM-Hard +2.7, dev +1.5. One seed per arm, unpaired; GSM-Hard +4.2 is ~2.2 SE, the rest is noise.
  Plausible mechanism: expr_gen evaluates whole multi-operator expressions that calc_s4 does not cover; not checked
  token by token.
- Caveat: in LIB the tools HURT the plain-GSM half against the same host with everything off (73.8 vs 80.8), and the
  adapters act on silent tokens (all-tools-off 56.2 < host-only 59.6). The host and adapters co-adapt to the always-on
  calculators.

### Seed 1 and no-IFEval (A40 host A40; fewer prompts; finishing ~13:30-13:50 UTC)
(see table update below)
