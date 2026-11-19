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
  stopped. The long arm-2 reference is the RL ablation agent's run on H100 host GPU 0 (`rl_implants_v2/runs/long/
  a2_lr1e-6_s0`, H100, B_max 512, dev every 50): u0 49.8, u50 61.4, u100 64.5, u150 66.6, u200 68.8, u250 67.6.
- **Compute.** H100 host **GPU 1** (H100; GPU 0 is never touched): W2 s0/s1, W3z s0/s1, arm 1 s0 (500-update reference
  for W3), W1, W4 — fused one-kernel implant, B_max 128, tok_budget 4096 (memory knobs only), up to 3 concurrent,
  each launch gated on ≥40 GB free (`launch.sh`). A40 host **GPU 0** (A40; GPU 1 = another job, untouched): W3r and W1
  lr 1e-6 — A40 needs `FLEX_SMALL=1` + `fused=0` (reference implant, exact, ~3x slower). A40 numbers are flagged.
- Earlier, my first smoke runs on H100 host GPU 0 collided with the ablation agent's jobs (OOM-killed two of them).
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
arm 1 s0 → W3z s0 → W2 s1 → W3e s1 → W3z s1 → W1 → W4. A40 host A40: W3r s0 (running) → W1 lr1e-6.

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
Live table: `runs/results.md` on H100 host (`python3 compile_results.py`). External arm-2 reference copied read-only to
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

### CWinOx s2, W3d s1, CWinB s0, animations (2026-11-19 ~14:25)
- **CWinOx s2** (A100): pooled u291-300 × ÷ 88.4 % (6,803 calls) vs + − 5.3 % (3,244); greedy veto u300 100/100/0/1 %;
  dev 67.4 (×/÷ 69.8, +/− 58.9), hand trigger 36.0, off 64.2. Map formed u150→u200. Its host also mostly stopped writing
  subtraction equations (126 '−' calls vs ~2,400 in s0): interface veto + partial text dodge. → competence map 3/3 seeds.
- **W3d s1** (A100): 49.0 → 64.6 → 68.6 → 70.9 (u0/100/200/300; scaled 64.0); links-off 59.4/40.0 → +11.5 (+24.0 scaled).
  On/off gap by checkpoint (both seeds): u0 +4.9/+6.0 → u100 +9.4/+9.4 → u300 +11.2/+11.5 (scaled +11.2 → +24.5/+24.0):
  links are useful from distillation and RL roughly doubles their contribution; link norms +15-20 %.
- **CWinB s0** (correct u1-150, deranged from u151; floor 0.05, in-wire lr 1e-3): in-wire keeps ~95 % fire for ~50
  updates (u151-200) while reward ≈ 0.02-0.19; host halves its calculator-triggering equations by u198; in-wire flips u202→
  u228 (93 → 11 %), at the floor by u250. Dev 65.9 (u150) → 4.1 (u175) → 17.8 (u200) → 54.1 (u250) → 57.0 (u300) = off.
- Animations (make_anim.py; real logged values only, frames every 5 updates): anim_competence_map.gif/.mp4 (CWinOx s0:
  per-op fire pooled over last 10 updates; dev at 25-update checkpoints + off/hand markers where evaluated) and
  anim_w3d_links.gif/.mp4 (W3d s0: per-layer link strength vs u1, logged every update; links on/off dev every 25).
- fig_competence_map.png v4: seed 2 added (panels 2-3).

### Masked wires-only, lr-only ablation, competence-map figure v3 (2026-11-19 ~13:50)
- **WOc-mask** (correct calc, out-wires only at firing tokens, host frozen), u186: 49.1 → 39.8 (u25) → 38.6 (u50) → 41.1
  (u100) → 44.1 (u150) → 39.0 (u175); scaled 36.2 → 18-25. Below the hand wiring (49.1/36.2), ≈ frozen host (43.0/23.2).
  Masked out-wires grew large exactly at firing tokens (output L21 198, L14 136 vs residual 173/45) and overwrite the digits.
- **WOr-mask** (deranged), u131: in-wire vetoes every call from u25 (2.40/answer); dev = frozen host exactly (43.0/62.8/23.2).
- → the unmasked wires-only gain (WOr 59.1, WOc 56.9 at u300, mostly the GSM half) was generic adapter capacity, not
  interface rewiring. Unmasked WOc u300 56.9 (GSM 76.0, scaled 37.8) vs hand 49.1/36.2; its in-wire vetoed ~2.2 correct
  calls/answer.
- **lr-only** (no floor, in-wire lr 1e-3, eps 1e-8) s0: ALSO learns the competence map — greedy veto u300 100/100/0/0 %,
  rollout fire (u291-300 pooled) × ÷ 95.0 % (7,514/7,907) vs + − 0.4 % (26/5,995); dev 66.1 (×/÷ 68.1, +/− 58.9), hand
  trigger 18.2, off 61.2. The floor is optional; the step size decides (lr 1e-2 / eps 1e-12 collapses with or without it).
- CWinOx s2 (A100) at u250 forming the map: veto 100/100/41/38 %, fire u250 6/4/55/61 %, dev 63.1.
- fig_competence_map.png v3: one window everywhere (rollout calls pooled over u−9..u), × ÷ / + − pooled by call counts,
  panel 3 "It needs small steps" with 5 bars (no floor lr1e-2, floor lr1e-2, no floor lr1e-3, floor lr1e-3 s0/s1);
  sidecar fig_competence_map.json gives calls / fired / % for every bar.

### Wires-only RL (host frozen) — confounded by the unmasked adapter design (2026-11-19 ~12:15)
- WOr (deranged calc + CW wires, 286,469 params): dev 6.4 → 51.1 (u50) → 57.1 (u200) → 59.1 (u300; GSM 78.8, scaled 39.5);
  frozen host 43.0 / 23.2; in-wire vetoes the broken calls (1.35-1.88/answer at u50-150). WOc (correct calc): 49.1 → 41.5
  (u50) → 56.2 (u200). WOn (param-matched host adapters): 43.0 → 27.0 (u50) → 47.9 (u250), unstable at lr 1e-2/eps 1e-12.
- Broken ≥ correct and the gain sits on the GSM half → the unmasked out-wires (ReLU(A e + c), c = 0.1 ≠ 0 on every token)
  act as a generic host adapter. NOT interface evidence. Clean test: out-wires masked to firing tokens (--cw_mask 1):
  WOc-mask (h100 GPU1, 12:14) and WOr-mask (GPU0, 12:13), reported at 13:45.
- CWinB (tool breaks mid-training) correct→deranged running on GPU0 (switch u150); deranged→correct stopped to make room.

### CWfx determinism check (2026-11-19 ~11:45): the size-ordered veto is seed-dependent
- CWfx = CW-full + in-wire floor 0.05, in-wire lr 1e-3 / eps 1e-8. s0 (A100): veto appears at u125, persists (u300 64/32/
  8/9 %; rollout fire u295 38/65/77/84 %); dev 66.8 (hand 64.8, out-only 66.4, in-only 60.6, off 56.2). s1 (H100): never
  vetoes (rollout fire 95 % = the floor's ceiling); dev 66.2 (hand 65.2, off 57.0).
- Tally over 8 correct-calculator in-wire runs: size-ordered veto in 4 (CW s0, CW-lean s0, CWinC s1 transient; CWfx s0
  persistent), none in 4 (CW s1, CW-in s0, CWinC s0, CWfx s1). Always ordered by size when present. With a free, always-
  correct tool, always-calling is an equally good optimum → the veto is not a reliable outcome. The competence map (tool
  wrong on some calls) is the robust interface-design result.
- Competence-map figure: fig_competence_map.png (+ .json sidecar: rollout calls pooled over 10 updates, n per bar).

### HEADLINE (2026-11-19 ~10:40): RL learns the calculator's competence map (CWinOx, 2/2 seeds)
- CWinOx: calculator exact on ×/÷, deranged on +/−; fixed hand write; trainable in-wire (1,025 params) + host;
  exploration floor p(fire) ≥ 0.05; in-wire lr 1e-3, Adam eps 1e-8.
- s0 (A40): rollout fire u300 + 5 / − 4 / × 89 / ÷ 80 %; greedy-dev veto u300 100 / 100 / 1 / 2 % (map formed u100→u150:
  u150 veto 100/100/5/3). Dev 65.0 (×/÷-dominant 64.9, +/−-dominant 61.7); same host with the hand trigger 19.0; off 61.2.
- s1 (H100): rollout fire u300 5 / 6 / 60 / 87 %; greedy veto 100 / 100 / 23 / 2 %; dev 64.9; hand trigger 18.8; off 60.1.
- Ablations: no floor + lr 1e-2/eps 1e-12 (CWinO) → veto-all from u10-20; floor + lr 1e-2/eps 1e-12 (floor-only, A100)
  → collapses to the floor for every op (fire 2-8 %), dev 61.5; lr-only (no floor, lr 1e-3) queued.
- CWfx s0 (CW-full + CWinOx in-wire settings, A100): size-ordered veto appears at u125 and persists (u150 70/50/13/6 %,
  u275 65/32/10/9 %; rollout fire u295 38/65/77/84 %); dev 66.8 (scaled 59.0) at u275. s1 running.
- Headline seed 2 (train2.py, A100): arm 1 s2 dev u300 63.0 (scaled 46.5), arm 2 s2 70.1 (60.5; off 56.4); tests: arm 1
  GSM8K 70.9 / GSM-Hard 45.0; arm 2 71.3 / 55.8 (off 70.1 / 40.6).
- Library runs (LIB s0, LIBscr s0) finished on H100 host 09:26 / 10:06; WOr / WOc auto-resumed then.

### Transplant of W3d links, CW s0 tests, library pause (2026-11-19 ~05:25)
- **W3d links are portable** (eval only): base host dev 52.8 vs hand 50.4 vs off 44.3, GSM-Hard 35.8 / 33.6 / 26.7;
  arm-1 u300 host dev 68.8 / 68.1 / 62.9, GSM-Hard 53.5 / 51.9 / 43.6 — slightly better than hand wiring on both hosts.
  CW's learned wiring does not port (its in-wire keeps vetoing on foreign hosts): wiring that carries the answer
  transfers, a learned veto is host-specific.
- **CW s0 tests**: GSM8K test 66.3 / hand 68.3 / off 67.1; GSM-Hard 51.1 / 51.8 / 41.4 (s1: 67.9/67.4/67.3, 51.1/49.6/39.8).
- 05:23:05 UTC: WOc (GPU0) and WOr (GPU1) SIGSTOPped per coordinator (co-scheduling halved LIB speed); auto-resume
  (SIGCONT) when the library process on that GPU exits; launchers start nothing on a GPU with a live library reservation.

### CWinC, W4w, 1.7B dot veto, scheduling incidents (2026-11-19 ~05:20)
- **CWinC** (−0.002 reward per fired call): s0 never vetoes, 70.2 (hand 70.2, off 56.2); s1 transient size-ordered veto
  (u25 70/40/25/24 %, u50 54/29/7/8 %) then 0 % from u100, 69.8 (off 53.9). Cost too small to sharpen; calls rise to ~3.1.
  Size-ordered transient veto: 3 of 6 correct-calculator in-wire runs (CW s0, CW-lean s0, CWinC s1).
- **W4w s0** (A40; warm writer + emb-init links 7/14/21, lr 1e-2): NEGATIVE. dev 49.9 → 52.8 → 55.1 → 60.0 (u0/100/200/300;
  scaled 43.0) vs W2w 70.1; host-only 60.6, middle-links-off 60.4, ORIGINAL hand wiring on the same host 65.4. Links
  grew 2.4-3× (L21 eff 62 → 147) and erased the calculator's contribution (writer exactness still 1.0).
- **Qwen3-1.7B GSM8K test (run_gsm protocol, n=1319)**: plain 73.09 (table) | calc_s3 72.93 | calc_s4 73.24; calls 2.47/q.
  Integer-valued equations written with a decimal tail ("40 + 30 = 70.0", one "52.0.0.0.0.0.0.0"): calc_s3 34 eqs in 18
  questions (1.4 %), calc_s4 12 eqs in 7 questions (all host money formatting "$6000.00"). Per-token detector missed them
  (the '.' is inside the implant's own write burst). Still a wash on GSM8K. GSM-Hard pending.
- Scheduling: library poller v1 bug (renamed before bash read the script) → v2 (rename-then-run, 90 s liveness, per-GPU
  cool-down, reservations RESERVE_g* + effree.py so nothing packs next to a library run's declared MEM_GB); WOr drift-probe
  crash for deranged implants (fixed); CWinOx s2 stopped to yield A40 host to the library; A40 host HOLD until 07:00 UTC.

### Transplant, CW-out/in, CW s1 test (2026-11-19 ~04:40)
- **CW-out s0** (A40→A100 run; out-adapters + trainable writer): u100 65.6 vs hand-only 58.2 (+7.4, scaled +12.3);
  u200 66.8 vs 64.1; u300 68.5 vs 67.0 (+1.5), L21-masked 68.5, off 57.0 → out-wires speed early learning, host catches up.
- **CW-in s0**: in-wire never vetoes (0 % throughout); u300 68.9 vs hand-only 69.0.
- **CW s1 tests**: GSM8K test 67.9 / hand 67.4 / off 67.3; GSM-Hard 51.1 / 49.6 / 39.8.
- **Transplant of CW s0 u300 wiring** (no training): base host dev 47.9 vs hand 50.4 vs off 44.3, GSM-Hard 33.1 / 33.6 /
  26.7; arm-1 u300 host dev 64.9 / 68.1 / 62.9, GSM-Hard 48.6 / 51.9 / 43.6. The learned in-wire still vetoes 0.3-0.55
  calls/answer on foreign hosts → the learned wiring is co-adapted; the HAND wiring is the portable interface.
- Library launch bug (mine): poller renamed scripts before bash opened them (04:22); fixed 04:25 (rename-then-run,
  90 s liveness check, per-GPU cool-down). LIB s0 running on H100 host GPU 1 since 04:25:55.

### CW-full s1 (2026-11-19 ~03:55): the size-selective trigger does NOT replicate
- In-wire never vetoes (0 % in every bucket at every dev checkpoint; rollout fire 99-100 % from u50). Dev 50.4 → 60.6
  (u25) → 61.4 (u100) → 67.2 (u200) → 66.2 (u300, scaled 56.2). u300: hand-only 65.6, out-only 66.2, in-only 63.0,
  L21-masked 64.6, off 56.9 → wiring vs hand +0.6 (s0 +2.7; mean +1.6).
- Size-ordered veto appeared in 2 of 3 correct-calculator runs (CW s0, CW-lean s0), seed-dependent at lr 1e-2 / eps
  1e-12 (set by the in-wire's first large Adam steps). Robust findings: broken tools switched off at the interface;
  W2ws/W3d; saturation.

### W2ws, W3d, CWinO (2026-11-19 ~02:30)
- **W2ws** (soft warm start: write directions + log-gain, calibrated gain 5.56 → p(digit | host alone wrong) 0.90):
  RL RAISES the gain 5.55 → 6.94 (u50) → 8.12 (u100) → 9.56 (u300) (hand gain 1e4); p(sampled token) at firing tokens
  0.94 → 0.995; ||W−W0||/||W0|| 0.11 (u50) → 0.22 (u300); grads stay ~3e-5 (W2w: collapse to 1e-13). Dev 48.8 → 64.1 →
  69.0 → 70.2 (scaled 62.8) at u0/100/200/300; same host + ORIGINAL soft write 62.1 / 64.9 / 65.2 (scaled 53.2) → the
  reshaped write is +5.0 (+9.6 scaled); implant off 58.2. Ends = fixed hand wiring (70.0 / 72.0): RL repairs a weakened
  write by hardening it, but does not find a better one. Exactness 1.0.
- **W3d** (W3e links initialised from the SFT-distilled checkpoint, no end write, links + host trained, lr 1e-2):
  dev 49.1 → 65.4 → 71.1 → 69.9 (u0/100/200/300; scaled 35.5 → 57.2 → 63.8 → 62.8); links-off 58.6 / 38.2 at u300
  (+11.3, +24.6 scaled). Links grow (L21 eff 342 → 411; gate rel change 0.34-0.49). **Parity with hand wiring without
  any hand-made write**; the same links started blind (W3e) end 63.9 / 48.2 and never move. RL keeps and strengthens
  wiring that carries signal, but does not discover it.
- **CWinO s0** (+/− deranged, ×/÷ exact, in-wire lr 1e-2, eps 1e-12): veto-all from u10-20 for every operator (rollout
  fire 0 %); dev ×/÷-dominant 57.2 vs +/−-dominant 52.3 at u150 (plain-RL level). Exploration collapse: no call is
  sampled once p(fire) ≈ 0, so ×/÷ cannot be told apart. → **CWinOx** (floor p = 0.05 + 0.9 σ, in-wire lr 1e-3, eps
  1e-8), seeds 0 (A40 host A40) and 1 (H100 GPU 0).

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


## 2026-11-19 ~15:40 UTC: W3d s2 truncated at u250 (OOM); CWinB s1 done
- W3d s2 (A100 host A100) died at u251 with CUDA OOM in backward. Cause: the library m2_mixLIBscr run next to it grew to ~49-57 GB (declared 42 GB). Not relaunched: the A100 now belongs to the library agent, by directive.
- W3d s2 dev curve (links on / links off (reset) / gap; scaled on/off):
  u25 54.6/48.1/+6.5 · u50 59.1/52.1/+7.0 · u100 62.5/53.5/+9.0 · u150 65.0/57.4/+7.6 · u200 65.9/57.6/+8.3 · u225 68.5/57.1/+11.4 · u250 67.6/57.0/+10.6 (scaled 59.5 vs 38.3, +21).
- Same shape as s0/s1 (u300: 69.9/70.9, gaps +11.3/+11.5). The gap grows under RL in all 3 seeds. s2 reads ~2 pp lower at an earlier stop (u250 vs u300), so treat it as a partial seed.
- CWinB s1 finished; H100 host GPU1 was handed to the library agent.
- CWinB s1 (tool broken at u150), result: the in-wire never vetoed. p_fire stayed 0.95-0.99 through u300. The HOST stopped calling instead: calls/rollout 2.8 (u150) -> 1.2 (u200) -> 0.29 (u210) -> 0.06 (u230+). Dev: 67.0 (u150) -> 6.5 (u175) -> 25.2 (u200) -> 58.1 (u225) -> 60.2 (u300), equal to host-only 60.2. Hand wiring at u300 gives 26.8.
- s0 vs s1: both recover to implant-off level. s0 recovered through the host dodge first, then an in-wire flip (u202-228). s1 recovered through the host dodge only. So which component learns the off switch depends on the seed. The host's text dodge is the fast, reliable route (n=2).
- lr-only s2 (CWinO s2, H100 host GPU0) STOPPED at u156 on coordinator request: GPU0 was at 79.8/81.5 GB next to the library m4 run. Partial record: dev 14.4 (u0) -> 48.9 (u25) -> 55.6 (u100) -> 60.1 (u150). The ×/÷-vs-+/− split had not formed by u150 (in-wire rollout fire stays low and mixed). It is not used in figure panel 3. Seeds 0/1 cover the ablation.
- lr-only s1 (CWinO s1, A40 host, no floor, in-wire lr 1e-3) DONE u300: pooled u291-300 ×÷ let through 80.3% (6222/7745) vs +− 0.18% (11/6233). Dev 66.6 vs hand 17.8 vs host-only 62.0. This replicates lr-only s0 (95.0 vs 0.4). So the step size, not the floor, is what makes the competence map learnable (2/2 seeds without the floor, 3/3 with it). fig_competence_map panel 3 now has 7 bars (lr-only s0 and s1).
