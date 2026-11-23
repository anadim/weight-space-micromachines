# seeds3 RESULTS: one-seed claims of blog §3.4 / §5.1 / §5.2 at 3 seeds

Commands and files: `README.md`. Every table here is printed by a script in this folder from `res/` (CPU, seconds):
(a) `try8.py` → `try8.md`; (b), (c) `summarize.py` → `summarize.md`; (d) `libpair.py` → `libpair.md`; (e), (f) `mixpair.py`
→ `mixpair.md`.

## (a) §5.1 A1 (RL with the whole serial stack + per-attempt credit): try-8 accuracy, 3 seeds (no new run)

Per-try data exists for all three A1 RL seeds: the §16 evals of `run_arm.sh A1 / A1_s2 / A1_s3` (test / GSM-Hard, N = 8 tries,
reduced into `res/a1/try8.items.jsonl.gz`) and the in-run dev curves (`gsm_stack/results/curves/A1*.steps.jsonl`). u0 is the
same untrained host + stack for every seed, so there is one u0 number.

| | u0 | A1 seed 3 | A1 seed 4 | A1 seed 6 | 3-seed mean (sd) |
|---|---|---|---|---|---|
| GSM8K test, vote N=8 | 71.87 | 79.76 | 80.74 | 79.68 | **80.06** (0.6) |
| test, try 1 | 72.9 | 73.8 | 75.2 | 73.4 | 74.1 (0.9) |
| **test, try 8** | **22.7** | 65.7 | 60.0 | 60.7 | **62.1** (3.1) |
| dev (600), vote N=8 | 81.33 | 86.67 | 86.00 | 85.00 | 85.89 (0.8) |
| **dev, try 8** | **29.0** | 70.67 | 65.33 | 67.67 | **67.9** (2.7) |
| GSM-Hard, try 8 | 15.0 | 50.0 | 46.8 | 46.6 | 47.8 (1.9) |

(The original A1 run's seed was not recorded; `run_arm.sh` uses 3. Its replicates A1_s2 / A1_s3 are seeds 4 / 6.)

- Per-try test curves (try 1..8):
  - u0: 72.9 61.0 53.6 48.6 40.6 35.3 28.2 22.7
  - seed 3: 73.8 72.9 72.7 69.9 69.4 67.0 66.0 65.7
  - seed 4: 75.2 74.1 71.6 70.4 67.9 63.6 62.9 60.0
  - seed 6: 73.4 72.9 70.5 68.2 66.8 65.2 63.2 60.7
- The one-seed figures in the blog were 66% (test) and 71% (dev). That seed is the best of the three; the 3-seed means are
  62% and 68%.
- The position reset with zero RL gives try 8 = 71.7% on test (`gsm_stack/results/RL_RESULTS.md` §2), so "the position reset
  gets there with zero RL" still holds.
- Blog wording: "RL with the whole stack attached and per-attempt credit takes test from 72% to 80%, and try 8 from 23% to
  62% (3 seeds; 60-66% per seed)". Figure: "try 8 goes from 29% to 68% on dev (23% to 62% on test; 3 seeds)".

## (b) §5.2 clean-up of a sloppy (noisy-routing) fine-tune, m4: 3 seeds

Seed 0 is the original run (H100; `rl/results/library/figdata/mix_m4_mixLIBnoisy_*`). Seeds 1 and 2 are new (one B200 each,
150 updates, about 2 h 45 min). The table shows off-type FIRED calls per answer (triggers x in-wire pass) on greedy dev:

| seed | calc on BBH (costly) u0 → u25 → u150 | calc on IFEval (cheap) u0 → u25 → u50 → u100 → u150 | test u150 BBH / IFEval | test acc (tools off) |
|---|---|---|---|---|
| s0 | 0.16 → 0.07 → 0.04 | 0.22 → 0.01 → 0.15 → 0.28 → 0.23 | 0.11 / 0.34 | 62.8 (32.2) |
| s1 | 0.15 → 0.04 → 0.05 | 0.22 → 0.00 → 0.01 → 0.00 → 0.00 | 0.07 / 0.01 | 63.8 (34.5) |
| s2 | 0.15 → 0.07 → 0.03 | 0.22 → 0.00 → 0.01 → 0.00 → 0.06 | 0.12 / 0.08 | 63.8 (34.2) |

- Costly misfires (calculator on BBH) drop about 4x in all 3 seeds (0.15 → 0.03-0.05 on dev).
- Cheap misfires (calculator on IFEval) cost nothing: leave-one-out of calc leaves IFEval unchanged in all 3 seeds. All
  3 seeds drop them to about 0 by u25. Seed 0 then drifts back to its starting rate (0.23); seeds 1 and 2 stay low (0.00 /
  0.06).
- So "RL leaves the cheap ones" was a one-seed observation. At 3 seeds, RL does not reliably hold them down: 1 of 3 drifts
  back.
- The misfire probe (within-group reward cost) was not re-run on the new seeds. Seed 0: −0.39 on BBH vs −0.08 on IFEval.
- Blog wording: "RL cleans up the wrong calls that cost reward (calculator calls on BBH: 0.15 → 0.04 per answer, 3 seeds).
  Wrong calls that cost nothing (stray calculator calls on IFEval prompts) are not reliably suppressed: all three seeds drop
  them early, but one drifts back to its starting rate."

## (c) §5.2 scramble-all, m2: in-wires closed, 3 seeds

Seed 0 is the original run (A100; `mix_m2_mixLIBscr_*`). Seeds 1 and 2 are new (one B200 each, about 2 h 45 min). The table
shows the greedy in-wire pass rate on test 400 at u150 (number of triggers in parentheses):

| seed | test all / tools off | calc | track_gen | order_gen | interval_gen | turtle_gen | truth_gen | expr_gen | calc_gen | bool_gen | ifeval | closed / fire |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 37.8 / 37.5 | 0% (299) | - | - | 100% (15) | 0% (13) | - | 0% (275) | 100% (76) | 0% (17) | 100% (600) | 4 / 7 |
| s1 | 44.0 / 39.0 | 100% (402) | 100% (5) | 0% (11) | 0% (1) | - | 100% (1) | 0% (426) | 0% (85) | 0% (11) | 100% (617) | 5 / 9 |
| s2 | 38.8 / 35.5 | 100% (354) | 100% (2) | 0% (8) | 0% (11) | 100% (29) | 78% (18) | 0% (324) | 0% (92) | 0% (26) | 100% (488) | 5 / 10 |

"-" means the tool never triggers. "closed" means < 5% let through.

- Dev (all): s0 20.8 → 35.2, s1 17.0 → 45.2, s2 17.0 → 40.2 (u0 → u150).
- Closed in all 3 seeds: expr_gen, bool_gen.
- The calculator (the one implant not scrambled) is closed in s0 only; it is 100% open in s1 and s2. calc_gen (scrambled)
  shows the reverse: open in s0, closed in s1 and s2. The IFEval implant is open in all 3.
- So "even the calculator's" is 1 of 3 seeds.
- Blog wording: "scramble every implant's output (all but the calculator) and RL closes about half of the in-wires that
  fire (4 of 7, 5 of 9, 5 of 10; 3 seeds), always including the scrambled expression and boolean implants. The unscrambled
  calculator stays open in 2 of 3 seeds."

## (d) §5.2 scrambled control on GSM8K (LIB vs LIBscr): third pair

LIB = the tool library on GSM8K; LIBscr = the same with every implant but the calculator scrambled. 300 updates each.

- Pairs: seed 0 (H100) is the original pair; seed 1 (B200) is an earlier rerun; seed 2 (B200) is new.
- s2 ran as jobs j25297 / j25298, hit the 4 h queue limit and resumed (j25396 / j25397, 15 min); about 4 h 15 min of GPU
  each.

| pair | LIB GSM8K / Hard (calc-only) | LIBscr GSM8K / Hard (calc-only) | LIB − LIBscr | LIBscr: test calls/q; train calls/q u1 → u25 → u300; dev calc fire u300 |
|---|---|---|---|---|
| s0 (H100) | 69.37 / 56.10 (67.85 / 53.37) | 69.29 / 51.86 (69.29 / 51.86) | +0.08 / +4.25 | 2.19; 1.65 → 0.53 → 1.77; 99% |
| s1 (B200) | 70.36 / 56.18 (69.37 / 53.75) | 65.50 / 41.62 (65.50 / 41.55) | +4.85 / +14.56 | 0.00; 1.60 → 0.42 → 0.05; 0% |
| s2 (B200) | 71.34 / 55.19 (70.05 / 53.75) | 66.72 / 51.33 (66.72 / 51.33) | +4.62 / +3.87 | 2.42; 1.40 → 0.33 → 2.49; 100% |
| mean (3) | 70.36 / 55.83 | 67.17 / 48.27 | +3.18 / +7.56 | |

- Paired item bootstrap of LIB − LIBscr (2000 resamples):

  | | test | Hard |
  |---|---|---|
  | s2 | **+4.62 [+2.27, +6.97]** | **+3.87 [+1.36, +6.22]** |
  | s1 | +4.85 [+2.20, +7.35] | +14.56 [+11.75, +16.98] |
  | s1 + s2 pooled | +4.74 [+2.92, +6.60] | +9.21 [+7.43, +11.03] |

- The correct implants beat the scrambled ones on Hard in 3 of 3 seeds, and on GSM8K in 2 of 3.
- The calculator survives the scrambled run in 2 of 3 seeds. In s2 it nearly went too (0.33 train calls/q at u25), then
  recovered.
- With the calculator kept (s0, s2), the Hard gap is about 4. The s1 gap of 14.6 is mostly the lost calculator.
- In s2, LIBscr calc-only (66.7) ends about 3 points below LIB calc-only (70.1) on GSM8K, even though the calculator was
  kept and the scrambled implants are off at eval. Seed 0 had no such gap (69.3 vs 67.9).
- "Big numbers drop" (dev u300, the scaled half, implants on → all off), now over 4 LIB runs:
  - s0: 63.50 → 37.75
  - s1 (A40, original): 64.50 → 36.00
  - s1 (B200): 65.50 → 37.25
  - s2: 64.75 → 38.50
  - mean: 64.56 → 37.38, so "64% → 37%" holds.
- Blog wording: "In a control run where every implant but the calculator was scrambled, RL shut the scrambled ones off and
  the correct set ended ahead on GSM-Hard in all three seeds (56 vs 52, 56 vs 42, 55 vs 51) and on GSM8K in two of them
  (69 vs 69, 70 vs 66, 71 vs 67). RL kept the good calculator in two of the three seeds; in the third it switched the
  calculator off too, which is most of that seed's 14-point Hard gap. Switch the implants off and big numbers drop from 64%
  to 37% (dev, 4 runs)."

## (e) §5.2 mixed task: all implants vs plain RL, 3 seeds

Qwen3-0.6B (chat) on the mixed task (GSM8K / BBH / BBH+arithmetic "combo" / IFEval), 150 updates, test 400 (stratified,
the same items for every run), one B200 per run. All three arms start from an 80-step SFT warm-start host.

- **m1m**: every implant attached (the m1 arm with `--cw_mask 1`: out-wires act only where an implant emits), from the
  tagged host `sft_host` (SFT rows with tool calls). About 2 h 50 min per seed.
- **aNL**: plain RL (`--plain 1`, no implants), from `sft_host_nl`, the same SFT recipe on the same questions with every
  tool call rewritten as a worked solution in plain text. This is the fair plain-RL baseline. About 21 min per seed.
- **m3**: plain RL from the tagged host (seeds 1, 2; the original m3 is seed 0). It is handicapped: the host was taught to
  call tools that are not there.

| arm | seeds | full | gsm | bbh | combo | ifeval | implants off | host only | seed SD |
|---|---|---|---|---|---|---|---|---|---|
| m1m (all implants) | 0, 1, 2 | **57.8** [53.2, 62.1] | 37.9 | 81.3 | 21.9 | 70.3 | 32.7 [28.5, 36.8] | 32.7 [28.5, 37.0] | 1.1 |
| aNL (plain RL, worked-solution host) | 0, 1, 2 | **44.7** [40.3, 49.2] | 27.6 | 70.6 | 19.1 | 43.5 | - | - | 0.8 |
| m3 (plain RL, tagged host) | 1, 2 | 33.9 [29.6, 38.0] | 26.8 | 46.7 | 12.3 | 37.5 | - | - | 0.2 |

Paired differences (same 400 items, seed means, 95% item bootstrap, 10000 resamples):

| | full | gsm | bbh | combo | ifeval |
|---|---|---|---|---|---|
| **m1m − aNL** | **+13.1** [+8.7, +17.6] | +10.3 [+1.5, +19.1] | +10.7 [+2.9, +18.5] | +2.7 [−6.6, +12.0] | +26.8 [+17.8, +35.5] |
| m1m − m3 | +23.9 [+19.3, +28.6] | +11.1 | +34.5 | +9.6 | +32.8 |
| aNL − m3 | +10.8 [+6.4, +15.2] | +0.8 | +23.8 | +6.8 | +6.0 |

- With implants on, RL wins overall and on 3 of the 4 task types; on combo the +2.7 is not significant.
- Each implant matters mainly on its own task type: leave-one-out on test costs most for the IFEval implant (+7.6
  [+5.3, +10.1]) and the BBH object tracker (+4.8 [+2.8, +7.0]); the calculator costs +1.0 (`mixpair.md`).
- Dev (400) at u150: m1m 58.2 / 57.8 / 58.0, aNL 40.5 / 41.0 / 40.0, m3 31.0 / 30.8. All m1m seeds start at 44.0, all aNL
  seeds at 27.0.
- The original one-seed m1 (`--cw_mask 0`, H100) reached 68.2 (implants off 36.8). Its masked variant m1m is the one rerun
  at 3 seeds and the one the final blog quotes; m1 itself was not rerun.
- Blog wording: "the model beats plain RL by 13 points (58% vs 45%, 3 seeds)"; figure: "With implants on, RL wins overall
  and on 3 of 4 task types (three seeds)".

## (f) §3.4 distilled links (W3d): GSM-Hard, 3 seeds

W3d: the calculator wired in only through learned links, initialised from a short distillation of the hand wiring (the
W3e links SFT, `rl/results/wiring/H100/sft3_W3e_lr1e-2/links_final.pt`), then 300 RL updates. Test eval at u300 with the
links on and with the host alone (links off). One B200 per seed, 56-59 min including the test.

| seed | dev u300 links on / off | GSM8K test on / host only | on − host only | GSM-Hard on / host only | **on − host only** |
|---|---|---|---|---|---|
| s0 | 70.6 / 60.4 | 71.0 / 71.0 | +0.0 [−1.5, +1.5] | 55.8 / 41.2 | **+14.6** [+12.4, +16.8] |
| s1 | 71.5 / 59.6 | 71.0 / 70.7 | +0.4 [−1.2, +2.0] | 55.4 / 41.9 | **+13.5** [+11.3, +15.8] |
| s2 | 69.5 / 57.5 | 69.7 / 70.1 | −0.4 [−2.0, +1.3] | 54.7 / 43.1 | **+11.7** [+9.4, +14.0] |
| mean | | | | | **+13.3** (seed SD 1.5) |

- The links matter on big numbers only: on GSM8K the trained host alone does as well (±0.4).
- Dev links effect at u300 (links on − links off, the trainer's `dev_reset_curve`; equal to host only here): +10.2 /
  +11.9 / +12.0, vs +11.2 / +11.5 in the original W3d s0 / s1 (H100 / A100, recomputed from `rl/results/wiring/figdata/W3d_s{0,1}.jsonl`).
- Blog wording: "RL makes them 15-20% stronger and roughly doubles their effect: +13 points on GSM-Hard (three seeds)".
  The "15-20% stronger" (link norms) and the animation come from the original runs (REPRODUCE §12).
