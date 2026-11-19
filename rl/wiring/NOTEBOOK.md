# RL over the calculator wiring: does trainable wiring / a trainable calculator help, and does RL discover the calculator organically? (2026-11-18)

Question (user): if we change HOW the calculator is wired into Qwen3-0.6B-Base and let RL train the wiring (and the
implant's own weights), does it help beyond the fixed hand-wired end-write? Does the model organically start using the
calculator when the hand write is removed?

## Setup
- Base: Qwen3-0.6B-Base, frozen host trained by RL (Dr.GRPO) exactly as arms 1/2/R in `../rl_implants_v2`.
- Recipe (identical to arm 2): G 8, 64 prompts/update, 64 updates, T 1.0, lr 1e-6 (host), no KL, Dr.GRPO, greedy dev.
- Dev = split B, n=800 (400 GSM8K-train + 400 scaled). Test never read. SE of an 800-item dev number ~1.7 pp.
- Implant = calc_s4 + veto (the `<expr> =` calculator). `e` = its one-hot output over {0-9, '.', '-', OS, OE}.
- **Plan revision (coordinator, 17:00 UTC): 500 updates, dev every 25 (+ u64), GSM8K only.** The 64-update queue was
  stopped. The long arm-2 reference is the RL ablation agent's run on H100 machine GPU 0 (`rl_implants_v2/runs/long/
  a2_lr1e-6_s0`, H100, B_max 512, dev every 50): u0 49.8, u50 61.4, u100 64.5, u150 66.6, u200 68.8, u250 67.6.
- **Compute.** H100 machine **GPU 1** (H100; GPU 0 is never touched): W2 s0/s1, W3z s0/s1, arm 1 s0 (500-update reference
  for W3), W1, W4 — fused one-kernel implant, B_max 128, tok_budget 4096 (memory knobs only), up to 3 concurrent,
  each launch gated on ≥40 GB free (`launch.sh`). A40 machine **GPU 0** (A40; GPU 1 = another job, untouched): W3r and W1
  lr 1e-6 — A40 needs `FLEX_SMALL=1` + `fused=0` (reference implant, exact, ~3x slower). A40 numbers are flagged.
- Earlier, my first smoke runs on H100 machine GPU 0 collided with the ablation agent's jobs (OOM-killed two of them).
  I killed my processes there and moved off that GPU.
- bf16 tiling differs by GPU and B_max (~1 pp at u0), so compare within a GPU type; W2's own u0 = arm 2 in this config.

## Arms (all share arm-2's recipe/seeds)
| arm | wiring | trainable besides host |
|---|---|---|
| 1 | no implant | — |
| 2 | implant frozen, hand-wired end write (before final norm) | — (reference) |
| R | deranged calculator, frozen, hand write | — (reference; from rl_implants_v2) |
| **W2** | arm-2 end write KEPT + zero-init trainable links (read `e`) into host residual at layers 7,14,21 | links |
| **W3z** | NO end write; only trainable links, gate **zero**-init (step 0 == arm 1) | links |
| **W3r** | NO end write; only trainable links, gate one-init + small-random W (links live from step 0) | links |
| **W1** | arm-2 wiring + trainable **writer** Dp (= out_dirs⊗gain, 14×1024) | writer |
| **W4** | W1 + W2 (trainable writer AND links) | writer + links |
| **W3e** | NO end write; links at 7,14,21 with **embedding init**: column for output symbol s = unit host input embedding of s's token × the median residual norm at that layer's input at digit positions (8-shot prefix: 21.2 @7, 45.1 @14, 173.4 @21); gate 0.1 | links |
| **W3e3** | W3e with gate 0.3 | links |
| **W2w** | **warm start**: NO separate end write; a `final` link with W = hand-wired write Dᵀ, gate 1 (== arm 2 at step 0) + zero-init links at 7,14,21 | links (incl. the writer) |

**Why only the writer is trainable in W1/W4.** The implant's reader and ReLU stacks are an exact *hard one-hot*
circuit: their output `e` is piecewise-constant in their own weights, so GRPO gets ~no gradient there (and the dense
stacks are 114M params). The one continuous surface is the write directions Dp = out_dirs⊗gain (what maps `e` to the
residual), which is also the only place calculator *exactness* can break. So "train the calculator" = train Dp here.
Links read the frozen `e` and add `gate ⊙ (W e)` to the residual at a layer's input; gate zero-init ⇒ step 0 identical
to the base arm. Links are active in BOTH rollout and the GRPO update (the model can actually route through them).

**Link learning rates.** Gate lr = link_lr (1e-3, Adam). W lr = link_lr × max(1, rms(W_init)) so every link moves by
the same *relative* step whatever its init scale (random links: ×1; emb links: ×1.0/1.36/5.2; warm final link: ×9258,
since Dᵀ has rms ≈ gain). Without this, Adam's ~lr-sized steps could never reshape the 1e4-scale warm write.

**Step-0 checks (dev-80 smoke, H100):** W2w u0 reproduces arm 2 exactly (50.0 / 60 / 40, writes 17.5, calls 2.775),
writer-exactness 1.0. W3e u0 = 45.0 = host-only (no immediate gain or damage on 80 items; dev-800 u0 in the table).

**Schedule (GPU 1, 2 runs fit at once, ~32 GB each):** W2 s0 (started 17:14) ‖ W3e s0 (17:22) → W2w s0 → W3e3 s0 →
arm 1 s0 → W3z s0 → W2 s1 → W3e s1 → W3z s1 → W1 → W4. A40 machine A40: W3r s0 (running) → W1 lr1e-6.

## Metrics logged per run (`runs/<arm>/steps.jsonl`)
- dev acc at u 0/16/32/48/64, split into GSM / scaled halves; fmt, writes/answer, calls/answer, length.
- host-only (implant+links off) and, for W2/W4, links-off (gates zeroed, write on) at u0 and final.
- link gate L2 and W L2 per layer, every update (W2/W3/W4).
- writer exactness drift (W1/W4): integrity[d] = argmax_vocab host(norm+head)(Dp_row_d) == token(d); `drift_exact`
  = fraction of 2000 random (a op b) results whose every digit still decodes to itself (1.0 at u0).

## General links (for the next implant)
`--link_spec SRC:LAYER,...` (LAYER int or `final`). SRC is a key in the implant feature dict (`SRC_DIMS`; now only
`calc` = calc_s4 one-hot e, 14-d). Several sources on one layer are summed (`LinkSum`). `Link.forward` takes either a
tensor (single implant, what the engine passes now) or a dict `{src: tensor}`. To add the rethink/mistake detector:
register its state dim in `SRC_DIMS`, have the engine step and the train pass build `feat = {'calc': e, 'rethink': s}`,
and pass e.g. `--link_spec calc:7,calc:14,rethink:14,rethink:21`. Logged per link every update: gate L2, W L2 and the
effective map ||diag(gate) W||_F (`link_eff`).

## Code
`train_wiring.py` (arms above), `engine.py` (adds the trainable-writer `Dp` path), `queue_a40.sh` (gated runner),
`make_fig.py` (figure). Shared engine/implant/kernels from rl_implants_v2.

## Results
Live table: `runs/results.md` on H100 machine (`python3 compile_results.py`). External arm-2 reference copied read-only to
`runs/ref_a2long_H100abl.jsonl` (ablation agent's 500-update arm 2, H100): u0 49.8, 50 61.4, 100 64.5, 200 68.8,
300 70.0, 400 70.9, 500 70.4 (scaled 63.8; host-only 55.9 at u500).

### u300 (runs capped at 300 per user; seed 0; dev-800; SE ≈ 1.7 pp)
| run | GPU | dev u0 / 100 / 200 / 300 | GSM / scaled @300 | links-off or host-only @300 | links ||diag(g)W|| @300 |
|---|---|---|---|---|---|
| arm 2 ref (ablation agent, fixed write) | H100 | 49.8 / 64.5 / 68.8 / 70.0 | 77.0 / 63.0 | host-only 58.0 | — |
| W2 (fixed write + zero-init links 7/14/21) | H100 | 50.4 / 65.4 / 70.4 / 72.0 | 79.8 / 64.2 | links-off **72.0**; host-only 56.2 | 0.044 / 0.043 / 0.012 |
| W3e (no write, embedding links, gate 0.1) | H100 | 45.6 / 56.4 / 59.1 / 63.9 | 79.5 / 48.2 | host-only (= links off) 61.6 | 7.7 / 16.4 / 65 (≈ init) |
| W3r (no write, random links, gate 1) | A40 | 44.0 / 54.2 / 58.6 / 61.1 | 78.2 / 44.0 | host-only 60.8 | — |

- W2 = arm 2; RL leaves the extra links at ~0 (links-off identical at every eval).
- W3e/W3r: links add ≤ +2.3 / +0.3 at u300; both stay ~8-11 pp (scaled ~16-20 pp) behind fixed wiring. No organic
  discovery at link lr 1e-3. (A 300-update arm 1 on H100 is not in hand; the ablation agent's seeds are the reference.)

### Gradient-flow check (2026-11-18 ~18:30): no plumbing bug
- 4 GRPO updates, host frozen (link grads unchanged), pre-clip grad norms: W3e gate 1-3e-4, W 0.5-2.7e-5; W2 gate ~2e-6,
  W exactly 0 at u1 (gate 0 kills dL/dW), then ~1e-7. Zeroing the gates inside the UPDATE-pass forward changes lp by
  0.025 mean at firing tokens (max 0.6): the update pass uses the links.
- ||gate|| only sees motion along the init direction; real displacement ||gate - g0|| grows ~0.006-0.01 L2/update (Adam
  random walk). W2 u100 checkpoint: ||gate|| 0.27/0.26/0.09, ||W - W0||/||W0|| = 1.4, but effective map only 0.012.
- Implant feature e ≠ 0 on 8-14 % of training tokens (~15 written tokens per ~115-token answer).

### Supervised sanity test (links-only distillation): links CAN carry the answer
- First attempt invalid: the scaled train set has no worked solutions (only `#### n`), so e never fired on targets.
- Redone on 1,217 correct greedy traces made by the hand-wired calculator (split-A scaled), W3e links only, host frozen,
  no end write, 200 CE steps × 64 (e ≠ 0 on ~20 % of target tokens).
- link lr 1e-2: dev scaled 25.0 → 29.3 / 32.8 / 35.3 / 34.3 (u50/100/150/200), overall 43.5 → 49.0; host-only 23.3 /
  43.0; hand write on the same host ≈ 37 scaled. Effective norms ×4-5 (L21 62 → 342). CE 0.139 → 0.075.
- link lr 1e-3: scaled 25.0 → 27.8; links ×1.3-1.5. → **capacity/placement fine; RL signal × lr 1e-3 too small.**
  Wire lr default moved to 1e-2, Adam eps 1e-12 (coordinator).

### Interface arms (fixed hand write + trainable in-wire), 2026-11-19 ~00:40
- **CW-lean s0** final 66.2 (scaled 58.2); hand-only 63.8, out-only 66.2, in-only 64.2, implant off 55.9. Rollout fire
  rate 1-2/3-4/5-6/7+ digits: u25 0 everywhere → u75 10/14/62/62 % → u150 4/14/75/78 → u200 51/63/87/98 → u225+ 100 %.
  Same arc as CW-full: over-veto → size-selective (big-number calls restored first) → always call (correct, free tool).
- **CW-full s0 u300**: 67.5 (scaled 59.0); hand-only 64.8, out-only 66.8, in-only 65.9, L21-masked 66.4, off 55.3
  (+2.7 over hand at u300; +0.7 at u100, +0.3 at u200 — borderline). Veto profile u300 32/6/1/0 %.
- **CWinP** (deranged when an operand has ≥ 5 digits) final 59.9 (scaled 41.8); hand-wired on same host 41.6. 100 % veto
  from u25 in every bucket; trigger fires 2.4 → 2.8 per answer, all vetoed. Veto-all is OPTIMAL here (the tool is right
  only where the host doesn't need it), so the size split cannot test a competence map → **CWinO** (+/− deranged, ×/÷
  exact; in-graph operator tracker) is the real test.
- **CWinR** (fully deranged) u100: 100 % veto from u25; trigger fires 2.50/2.16/2.35/2.74 per answer (u0/25/50/100), all
  vetoed — the host keeps calculator-friendly equations. Arm R (no in-wire) instead learned the TEXT dodge (unit words,
  calls 2.50 → 0.39). Dev: CWinR 7.0 → 54.2 (u50) → 58.9 (u100) vs arm R 7.0 → 28.6 (u32) → 36.9 (u64). Forcing the
  broken tool back on at u100: 6.8. **The in-wire gives RL a clean off switch.**
- **W1 s0** final (A40): 68.9; original writer on same host 68.0; off 57.6; exactness 1.000.
- W2ws calibrated gain 5.56 (p 0.900 on 119 hard digit positions); W3d starts from distilled links at 49.1 / 35.5.
- Dev now also splits accuracy by the problem's dominant arithmetic (gold-solution annotations; ×/÷ vs +/−).

### More finals (u300) and CW-full s0 trajectory
- **W1 s0** (trainable writer Dp, relative lr 1e-2, eps 1e-12, A40): 49.2 / 65.0 / 68.1 / 68.9 at u0/100/200/300
  (GSM 76.5 / scaled 61.2); same host + ORIGINAL writer 68.0; implant off 57.6; writer exactness 1.000 at every eval.
- **CW-full s0 veto by operand size (greedy dev)** stays size-ORDERED at every checkpoint but its LEVEL fades:
  u25 97/90/44/40 %, u50 95/81/21/21, u100 69/24/0/2, u200 38/8/1/1, u250 18/3/0/0 (1-2 / 3-4 / 5-6 / 7+ digits);
  vetoes per answer 1.89 → 1.04 → 0.27; rollout fire rate GSM/scaled 0.08/0.63 (u46) → 0.60/0.79 (u268).
  Ablations (dev): u100 full 60.5, hand-only 59.8, out-only 59.3, in-only 60.0, L21 masked 62.3, implant off 53.1;
  u200 63.9 / 63.6 / 62.5 / 63.6 / 63.6 / 55.4 → learned wiring ≈ hand wiring on the same host. Only L21 open at u268
  (output 117 firing / 20 silent); KL 0.069. Dev u250 66.4 (scaled 58.8) vs W2 70.8, ref 67.6.

### W2w (lr 1e-2) final, W2ws calibration, CW trigger redesign (2026-11-18 ~21:40)
- **W2w lr 1e-2 u300: 70.1 (GSM 78.8 / scaled 61.5); same host + ORIGINAL write 70.8**; fixed wiring 70.0 (ref) / 72.0
  (W2). The saturated write cannot be reshaped (grads ~1e-13); training it changes nothing.
- **W2ws calibration** (log-gain scalar, directions = out_dirs): measured on digit-write positions where the host alone
  would emit a different digit, p(intended digit) = 0.17 (no write), 0.30 (gain 1), **1.00 (hand gain 1e4)**; gain
  **5.74** gives p = 0.90 (0.994 over all digit writes). Two calibration traps fixed on the way: mean p over ALL firing
  positions is 0.91 with no write (most written tokens are predictable; OS/veto rows are zero/replaced), and traces must
  keep the RAW generated ids (re-tokenising merges ' ' + digit and shifts what the implant reads; `gen_traces.py` fixed;
  the earlier SFT used re-tokenised targets, its dev gains were measured on real decoding and stand).
- **CW-full s0: RL redesigns the trigger.** Greedy-dev calls vetoed by the call's largest operand:
  u25: 1-2 digits 97 %, 3-4 90 %, 5-6 44 %, 7+ 40 %;  u50: 1-2 95 % (1046/1106), 3-4 81 % (239/296), 5-6 21 %, 7+ 21 %.
  Rollout fire rate GSM 0.08 vs scaled 0.63. Dev 50.4 → 49.8 (u25 dip) → 59.6 (u50) = W2w/W2 at u50.
  Out-wires u51: L21 open (g 1.0, output 86 firing / 40 silent vs residual 173), L7 g 0.42 (17 / 0.3), L14 + final closed.

### "RL designs the interface" arms (fixed hand write + trainable in-wire only)
- In-graph operand tracker (digits of the current number; largest number in the current arithmetic span, reset by word
  tokens; snapshot at '='), recorded per call → rollout fire rate per bucket every update, no text parsing.
- CWinR: arm-R deranged implant (dev@0 7.5). CWinP: deranged only when the call's largest operand has ≥ 5 digits
  (same derangement; dev@0 GSM intact, scaled 5.0). CWinC: correct implant, reward − 0.002 per fired call.
- With an in-wire present the write is e@D of the VETOED e (the fused kernel's precomputed write ignored the veto);
  decision codes 1 fire / 2 veto (+4 broken call) are replayed exactly in the update pass.
- Test mode `--eval_ckpt DIR`: GSM8K test + GSM-Hard, wiring as trained / hand-only / implant off (questions > 224
  tokens dropped, n reported). `--interp 1`: per-token in-wire logit vs operand size, position in number, op, '='.

### Re-scoped arms (user: let RL CHANGE the hand-made wiring) — u100, seed 0
- **W2w** (hand write as trainable warm link, lr 1e-2, eps 1e-12, H100): dev 50.4 / 54.5 / 58.6 / 59.9 / 63.6 at
  u0/25/50/75/100 (scaled 37.2 → 52.0) = fixed wiring (W2 65.4, ref 64.5 @100). Same host + ORIGINAL write: 65.0 @100.
  Implant off 55.9. Writer exactness 1.0. **The write is saturated:** pre-clip grad on W 7e-6 (u1) → 4e-11 (u10) →
  3e-13 (u100), gate 1.5e-5 → 5e-10; ||W−W0||/||W0|| 0.017 @25 → 0.020 @100 (stalled). Gain 1e4 makes p(digit) ≈ 1,
  so d logp/d write ≈ 0: RL cannot reshape it. Fix would be a below-saturation warm start (lower gain / trainable gain).
- **W1** (trainable Dp, relative lr 1e-2, A40): 49.3 / 53.3 / 59.0 @0/25/50; reset 54.5 / 59.0; Dp change stalls at 0.02.
- **CW-full** early (u46): in-wire fire rate 0.95 → 0.30 (GSM) / 0.66 (scaled); greedy dev vetoes 1.6 of ~2.4 calls per
  answer at u25 (dev 57.5, no damage). Out-wire gates diverge: L7 1e-4, final 0.01 (closed); L14 0.51, L21 0.94 (open),
  ||B|| 16-20; output norm fire/silent L14 24.7/5.1, L21 81/37 (residual norms 45, 173): L21 also acts as a host adapter
  on silent tokens (the unmasked-design confound).

### Co-designed wiring (CW) — built 2026-11-18 ~20:00
- OUT-wires g(h)·B·ReLU(A e + c) at 7/14/21/final (width 64; A~N(0,1), c=0.1, B=0, w_g=0, b_g=0 ⇒ g=0.5); unmasked as
  designed (they also act on silent tokens; logged: wire-output norm at firing vs silent tokens). On W2w (trainable warm
  final write, no fixed write). IN-wire: per write-burst veto, fire ~ Bernoulli(σ(k·0.5 + b + v·LN(h14[t-1]))),
  k = 6.84, b0 = −0.47 (P(fire|on)=0.95, P(fire|off)=0.02; "off" has nothing to write), greedy at eval; decision
  log-probs added to the GRPO logprob; the update pass replays the recorded decisions.
- Params: CW 286,469 = 0.048 % of host = 11.96 % of implant nonzeros (2,395,647); CWlean (w32, 14+final) 84,931 = 0.014 %
  / 3.6 %; CWout 285,444; CWin 16,385; W2w 15,360; W1 14,336.
- Step-0 bit-identity: greedy dev tokens hash 6c789cdfd44c for W2w, CW, CWlean, CWout, CWin (identical).
- Bugs caught in smoke: (1) KL was computed against logprobs that included the in-wire decision term (KL 0.017 at u1
  with host = ref) → KL now on token logprobs (0.0 at u1). (2) Gate L1 through Adam (eps 1e-12) closed gates at full lr
  while B = 0 (g 0.5 → 0.31 in 2 updates) → L1 now decoupled (SGD shrink of b_g by lr·λ·mean g(1−g), ≈1e-6/update).
- My CW smoke (21 GB with the KL reference) OOM-killed my own W2w s0 (lr 1e-3) at u25; re-queued at the end.

### Interim (2026-11-18 ~18:05 UTC; seed 0 each; SE ≈ 1.7 pp)
- **W2** (end write + zero-init links), u103: 50.4 / 56.6@25 / 60.0@50 / 61.5@64 / 63.1@75 / 65.4@100, ≈ arm 2
  (61.4@50, 64.5@100). Links-off: 61.9@64, 65.5@100 = no effect. Links barely grow: ||diag(g)W|| = 0.012 at u103
  (gate L2 0.2 / 0.2 / 0.07), negligible vs the hand write. **With a working end write, RL ignores the extra links.**
- **W3e** (no end write, embedding-init links, gate 0.1), u67: step 0 = 45.6 (GSM 64.2, scaled 27.0) vs host-only 44.2
  → the embedding hand-over alone gives ≤ +1.4 (noise). 48.2@25, 54.5@50, 54.6@64 (scaled 34.8); links-off@64 53.1.
  ≈ plain RL (earlier H100 arm 1: 54.4@64, scaled 35.0), ~7 pp behind fixed wiring, all on the scaled half.
  Link effective norms do not move (7.6/16.2/62.7 at u1 and at u67; gate L2 3.200 → 3.215): **RL does not tune the
  informed links either** — the GRPO signal on them is ~noise at link_lr 1e-3.

