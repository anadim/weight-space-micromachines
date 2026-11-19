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
  A40 host:ifeval_heldout/frozen), frozen LM-head snapshot of Qwen3-0.6B (the host it was solved against).
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

### Seed 1 and no-IFEval (A40 host A40, deadline-guarded fewer prompts, dev every 50 + u25)
| | LIB s1 (A40, 32->24 prompts) | LIBnoIF s0 (A40, 40 prompts) |
|---|---|---|
| dev u0 / 100 / 200 / 300 | 49.2 / 64.0 / 66.1 / **71.0** | 49.2 / 66.9 / 67.4 / **68.1** |
| dev u300 GSM / scaled | 77.5 / 64.5 | 72.8 / 63.5 |
| LOO (dev u300): − calc_s4 / − expr_gen / − calc_gen | 68.1 / 69.8 / 71.0 | 67.8 / 66.1 / 68.4 |
| calc_s4 only / all tools off | 69.6 / 57.1 | 66.5 / 56.2 |
| GSM8K test full / calc_s4 only | 70.1 / 69.3 | 67.3 / 67.1 |
| GSM-Hard full / calc_s4 only | 53.2 / 52.3 | 54.1 / 53.4 |
- Same picture as s0: only the three '=' calculators trigger; the 5 call-form tools, bool_gen (and IFEval) never do;
  the library over calc_s4 alone adds +0.2 to +2.7 on test sets (all within ~1-2 SE).
- Seed 1 shows RL STARTING to veto the redundant calc_gen late: greedy-dev fire 100 % -> 93 % (u200) -> 26 % (u250)
  -> 55 % (u300), while calc_s4 and expr_gen stay at 100 %. Removing calc_gen there changes nothing (71.0).
- No-IFEval ≈ full library (IFEval never fires on GSM), as expected.

---------------------------------------------------------------------------------------------------------------------
# Mixed-task RL with the tool library (user follow-up, 2026-11-19 afternoon) — code in mixed/

**Host = Qwen3-0.6B INSTRUCT (chat template, thinking off), NOT the Base host of the GSM runs above.** Chat format is
what sets the IFEval circuit's assistant-role register; the circuit was solved against this host.

Data (`mixed/mixed_data.py`, all generated by us; train 5000 / test 999):
| type | train / test | source | disjointness |
|---|---|---|---|
| gsm | 1500 / 300 | scaled-number GSM8K variants | train sources = split A, test = split B |
| bbh | 1500 / 300 | teams' generators, 12 families (track3/5/7, order3/5/7, temporal, navigate, lies, arithmetic, boolean, counting) = the 8 BBH tools | seed 100 vs 102, test deduped vs train |
| ifeval | 1200 / 250 | Team D gen_data (14 rule types + distractors) | seed 100 vs 102, deduped |
| combo | 800 / 149 | gsm/bbh question + 1 of 7 format constraints (lower, caps, quote, end phrase, title, P.S., highlights) | questions disjoint; constraint templates shared |
Rewards (`mixed/mixed_task.py`): gsm numeric match of the last 'answer is X'; bbh team-harness norms; ifeval lm_eval
checker (prompt-level strict); combo = base correct AND constraint passes.

Prompt: system "...think step by step; your last sentence states the result as: The answer is X." First version had
that line in the USER turn: the IFEval circuit's 'end' rule (end-phrase) latched on 40/40 prompts of every type and
forced the phrase into GSM/BBH answers (tools-on dev GSM 0 / BBH 0 vs host-only 15 / 45). Moved to the system prompt:
prompt latch GSM 0/40, BBH 0/40, IFEval 39/40, combo 37/40 (`mixed/if_latch_probe.py`). Nothing else tuned.

SFT warm-start (`mixed/sft_warm.py`; plain host fine-tune, no implants attached, wires not created; 80 steps × 16, lr
1e-5, loss 1.88 -> 0.26): 432 bbh_integrate train_union_v2 transcripts (Team A/B call form, Team C passive, 100 no-call
negatives) + 200 GSM8K split-A originals with '48/2 = 24' equations; no IFEval rows. Without it the call-form tools
cannot fire (the host never writes our envelope syntax). Noisy-routing variant (`--noisy 0.3`): same rows; 57 GSM + 29
no-call rows get a BBH call+response inserted, 105 BBH rows get a random 'x op y = z' line.

u0 (= after SFT; greedy dev 200; triggers per answer, in-wire fires 100 % at init):
| tool | gsm | bbh | ifeval | combo |
|---|---|---|---|---|
| calc_s4 | 1.95 | 0.12 | – | 0.83 |
| expr_gen | 0.98 | 0.28 | – | 0.63 |
| calc_gen | 0.75 | – | – | 0.27 |
| track / order / interval / turtle / truth / bool | – | 0.22 / 0.17 / 0.07 / 0.12 / 0.12 / 0.10 | – | small |
| ifeval | – | – | 1.52 | 0.80 |
Accuracy tools on vs host only: gsm 30.0 / 20.0, bbh 56.7 / 41.7, ifeval 58.0 / 36.0, combo 13.3 / 3.3.
Checks: tools-off == host-only bit-identical (hash e2b7ade75ce4); HF check |dlogp| mean 0.010, p99 0.14; per-tool flag
mismatch 0/787.

Runs: pilot LIB (A40 host A40, 16 prompts, 50 updates). 300-update arms queued on H100 host (32 prompts × G8, max_new 512):
m1 LIB, m2 LIBscr, m3 plain (tools attached, never on), m4 LIB from the noisy-routing SFT.

Pilot LIB (A40 host A40, 16 prompts, 50 updates, dev 200): dev all 42.5 -> 36.5 (u25) -> 38.0 (u50) (gsm 30/15/21.7,
bbh 56.7/60/45, ifeval 58/50/64, combo 13.3/10/13.3; 200 items, noisy). Routing: the **order tool is the first one RL
vetoes** (greedy fire on bbh 100 % -> 8 % at u50, rollout 89 -> 60 %); every other tool keeps firing on its own type;
calc_gen use on gsm grows 0.75 -> 1.77 triggers/answer. ratio 1.000, HF check mean 0.010, flag mismatch 0/787.

300-update arms (32 prompts, max_new 512): m1 LIB H100 host GPU1 (14:59), m4 LIB-noisy H100 host GPU0 (15:17),
m2 LIBscr A100 host A100 (15:20), m3 plain A40 host A40 GPU0 (15:21).
Noisy-SFT u0 vs clean (greedy triggers/answer): calc_s4 on bbh 0.31 vs 0.10, expr_gen on bbh 0.56 vs 0.26, calc_s4 on
combo 2.27 vs 1.13, turtle on ifeval 0.23 vs 0; BBH calls in GSM answers 0.01-0.02 (the inserted call snippets in
GSM rows did not take).

**BUG found 16:20 UTC, arms restarted ~16:30.** The Base-checkpoint fix that gives the 4 envelope tokens distinct random
rows (Team C prep_call_tokens) was also applied on the INSTRUCT host, whose envelope rows are already distinct and
trained. The tied head then made the tools' solved write directions target rows the real host does not have, so the
call-form tools' `<tool_response>` write never won: the host wrote `</tool_call>2</tool_response>` itself, and outputs
with a call tool on vs detached were identical on 449/449 bbh+combo test items (order probe). Passive '=' tools and
IFEval were unaffected; the GSM overnight runs (Base host, shared rows) were correct. Fix: re-init only when the 4 rows
are identical. Order items, frozen SFT host, tools on: 25.7 -> 37.1 after the fix. m1, m2, m4 killed (u68/u35/u37,
kept as runs/broken_envelope_*) and restarted from u0; the pilot's "order veto" is void (the tool could not write).
m3 plain unaffected (tools off).

**BBH call-tool probe with the envelope fix** (SFT host frozen, greedy, test items; `--probe_tool`):
- order items (70): tools on 41.4 vs order tool detached 18.6 (order3 56.5/13.0, order5 29.2/16.7, order7 39.1/26.1;
  59/70 outputs differ). (The first LOO probe had a bug: generate() was called without detaching the tool.)
- tool replies vs the Python reference on the same call string: track 67/67, truth 37/37, interval 34/34, walk 41/41,
  order 73/74; empty replies (walk 17, order 31, interval 4) are host-written unparseable calls (80-step SFT).
- all bbh+combo (449), tools on, before -> after fix: overall 37.9 -> 53.7; track3 30.4 -> 91.3, temporal 25.6 -> 66.7.

**Restart at 64 prompts × G8 (coordinator, convergence)** at ~16:20 UTC: m1 H100 host GPU1, m4 H100 host GPU0 (alone),
m2 A100 host A100; m3 plain-64 on A40 host GPU1 (the 32-prompt m3 continues on GPU0 as reference; 32-prompt partial runs
kept as runs/p32_*). u1: m1 81 s/update (rollout 55 s), 33/64 kept groups; m4 84 s, 35/64.
u0 dev (400 items) with the fix: m1 gsm 30.8 / bbh 70.0 / ifeval 52.0 / combo 8.3; m4 30.0 / 75.0 / 55.0 / 16.7;
m2 (scrambled) 10.8 / 25.8 / 38.0 / 1.7.
Profile (decode @128 rows, H100 shared): host ~5-6 ms/step, + BBH tools ~53, + IFEval ~63 (host 9 %, BBH tools 75 %,
IFEval 16 %). Only calc_s4 is fused; the 8 BBH tools and IFEval are PyTorch inside the step graph; the IFEval head
(logits over the vocab + masks) runs every token whether or not its register is set.

**Restart for 150 updates (user decision)** ~16:29 UTC: --updates 150 --ckpt_every 150 --test_n 400 (final test on 400
stratified items, LOO only for tools that fired), dev u0/25/50/100/150, 64 × G8. m3-64 plain finished (coordinator: test
36.3 = gsm 31.8 / bbh 46.0 / ifeval 41.3 / combo 14.8; dev 22.5 -> 33.5). m4 seed 1 on A40 host GPU1 (A40) from ~17:36.
Bit-exactness reference for the new engine: runs/engine_ref/engine_reference.json (A40 host; laptop mixed/engine_reference_A40.json).

EARLY READOUT (greedy dev 400; all tools on):
| arm | u0 | u25 | u50 (gsm / bbh / ifeval / combo) |
|---|---|---|---|
| m1 LIB | 44.5 | 49.8 | 53.5 (39.2 / 75.8 / 67.0 / 15.0) |
| m4 LIB-noisy | 47.8 | 51.8 | 56.5 (40.0 / 81.7 / 71.0 / 15.0) |
m4 misfires (greedy fired calls/answer u0 -> u25 -> u50): calc on IFEval 0.22 -> 0.01 -> 0.15, expr on IFEval 0.22 ->
0.01 -> 0.15, calc on BBH 0.16 -> 0.07 -> 0.05; rollouts calc/expr on IFEval 0.11 -> 0.19 -> 0.24 (not fading), calc on
BBH 0.43 -> 0.18 -> 0.08 (fading). In-wires fire ~100 % at greedy on every tool: no veto learned yet; routing changes
come from what the host writes. Figure: fig_mixed_routing.png (EARLY CHECKPOINT).

## Mixed-task FINAL (u150) — m1 LIB (clean SFT) and m4 LIB (noisy-routing SFT), H100, 64 × G8, test = 400 stratified
| arm / mode | all | gsm | bbh | ifeval | combo |
|---|---|---|---|---|---|
| m1 tools on | **68.2** | 50.9 | 87.6 | 81.5 | 36.1 |
| m1 all tools off (same host + wires) | 36.8 | 31.8 | 44.5 | 46.7 | 13.1 |
| m1 host only | 31.8 | 24.5 | 49.6 | 29.3 | 8.2 |
| m4 tools on | **62.8** | 39.1 | 84.7 | 76.1 | 36.1 |
| m4 all tools off | 32.2 | 21.8 | 38.7 | 42.4 | 21.3 |
| m4 host only | 28.8 | 22.7 | 39.4 | 33.7 | 8.2 |
| m3-64 plain RL, tools never on (coordinator) | 36.3 | 31.8 | 46.0 | 41.3 | 14.8 |
Dev (400, tools on) u0/25/50/100/150: m1 44.5 / 49.8 / 53.5 / 64.2 / 65.5; m4 47.8 / 51.8 / 56.5 / 58.2 / 61.0.
LOO on test (m1, full 68.2): − IFEval 59.0 (ifeval 81.5 -> 46.7), − track 62.2 (bbh 87.6 -> 72.3), − expr 64.0, − truth 65.8,
− walk 65.8, − order 66.0, − interval 67.0, − calc_s4 67.0 (gsm −3.6), − calc_gen 68.0, − bool 68.2. Each tool hurts only
its own task type. m4: − track 57.0 (bbh −15.4), − IFEval 53.5, − expr 59.5, − order 60.0.
Routing (greedy dev fired calls/answer u0 -> u150): m1 off-type ≈ 0 (calc on BBH 0.10 -> 0.03). m4 noisy misfires:
calc on IFEval 0.22 -> 0.23 (test 0.34), expr on IFEval 0.22 -> 0.23 (test 0.34), calc on BBH 0.16 -> 0.04 (fades).
In-wires fire ~100 % at greedy at u150 in both arms: no veto learned; routing changes come from the host's text.
Reward split (m4 u150, sampled 64 train prompts × G8 per type): rows with an off-type call vs without — calc on IFEval
0.56 vs 0.67 (32 rows; within-group −0.08, 24 groups), expr on IFEval 0.58 vs 0.67 (−0.06), track on IFEval 0.50 vs
0.67 (8 rows, −0.29), calc on BBH 0.42 vs 0.81 (19 rows; within-group −0.39, 8 groups). The costly misfire (BBH) fades;
the cheap ones (IFEval) persist.
Figure: fig_mixed_routing.png (u0 vs u150). Seeds running: m1 s1 (H100 host GPU1), m4 s1 (A40 host A40), m4 s2 (H100 host GPU0
after a 15-min hold), m2 LIBscr (A100) then m2 s1.
