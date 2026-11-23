# Sourced blog claims

All scripts are in `checks/sourced/` (run from the release root, see README.md). CIs are 95% paired item bootstraps with 10k resamples unless stated otherwise.

| item | claim | result |
|---|---|---|
| (a) | §3.3 "about half the runs" size-ordered veto | DONE |
| (b) | §3.4 BBH "never helped, one cost 10 points" | DONE (no paired CI: aggregates only) |
| (c) | §4.4 compliant S2F Δ vs H | DONE |
| (d) | §5.2 `<calc>` fade 80→25 / 88 / 95–98 | DONE |
| (e) | §5.1 IFEval 2b − 4 paired CI | DONE |
| (f) | §4.2 IFEval instructions/prompt; `circ.fires` | DONE |
| (g) | §4.4 never solved in 32 tries; wrong-gold share | DONE on CPU from 36 existing independent dev tries: 3.8% never solved in 32, 5–6 of 23 bad gold |
| (h) | §1.2 Read layer checked on all ~151k vocabulary tokens | DONE (CPU) |

---

## (a) §3.3 size-ordered veto count — DONE

**Blog claim** (blog draft): "In about half the runs, RL first 'vetoes' most implant calls, then lets them back in by size: big numbers first, 1-2 digit numbers (which the model handles fine) last."

**Definition used.** This matches the blog direction (big numbers are let back in first). A run is flagged if, at any logged point, admit(operands with ≥ 5 digits) − admit(operands with 1–2 digits) > 0.3 while admit(1–2) < 0.5.
- C36 data: train-rollout `admit_d1_2` / `admit_d5p` per update. I use only rows that have `calls`; the rows with only `dev` are dev-eval lines and have no per-size admit. I skip updates with fewer than 100 in-wire decisions (`iw_n` < 100). Without that filter, fI_s14 "flags" at u153 on 2 decisions in the whole update, after it had already collapsed to calling nothing.
- Early runs: greedy-dev `dev_veto_by_digits` ({bucket: [n_calls, n_vetoed]}), with admit = 1 − vetoed/n, and the 5-6 and 7+ buckets pooled.

**Result**

| set | flagged | notes |
|---|---|---|
| Early correct-calculator in-wire configs (NOTEBOOK l.152 tally) | **4 / 8** | CW s0 (u25–275, fading: u300 admit 68/94/100 %), CW-lean s0 (u64–175; below threshold from u200, all calls admitted from u225), CWinC s1 (u25–50; below threshold from u64, all calls admitted from u100) are **transient**. CWfx s0 is **persistent** (notebook only: veto u300 64/32/8/9 %). Not flagged: CW s1, CWinC s0 (from logs), CW-in s0 and CWfx s1 (notebook only). |
| C36 arm I, host lr 2e-6 (`fI_s0–19`) | **0 / 20** | max per-update a5p − a12 ≤ +0.12 (fI_s8 at u3–7, uniform dip). The u291–300 admit is 0.943–0.952 in every size bucket. 2 seeds lost the calculator. fI_s14: the in-wire vetoed everything **uniformly** at u4 (admit 0.07/0.08/0.08) and the host then stopped calling. fI_s17: the in-wire kept admitting ~0.95 and the host stopped calling at about u40. |
| C36 arm I1, host lr 1e-6 (`fI1_s0–7`) | **0 / 8** | max a5p − a12 ≤ +0.044 |
| C36 total | **0 / 28** | arms I0 / Itok have no trainable in-wire (admit ≡ 1), so they are not counted |
| pooled | 4 / 36 | |

Early-run trajectories (admit %, 1-2 / 3-4 / 5+ digits, greedy dev):
- CW s0: u25 3/10/60 → u100 31/76/98 → u200 62/92/99 → u300 68/94/100
- CW-lean s0: u25 0/0/0 (veto-all) → u100 8/12/65 → u175 23/32/95 → u225 100/100/100
- CWinC s1: u25 31/61/76 → u50 47/71/92 → u75 91/98/100
- CW s1, CWinC s0: 100/100/100 at every checkpoint

**Inputs**
- C36 per-update train records of runs `fI_s{0..19}`, `fI1_s{0..7}`: `checks/calcimp/tt/res/train_steps.jsonl.gz` (the calcimp release copy, floats rounded to 5 significant digits; the table is unchanged)
- `rl/results/wiring/H100/L300_{CW_s0_kl001evalall0, CW_s1_kl001evalall0, CWlean_s0_kl001evalall0, CWinC_s0_evalall0, CWinC_s1_evalall0}/steps.jsonl`
- CW-in s0, CWfx s0 and CWfx s1 have no step logs in the workspace. Their entries come from `rl/wiring/NOTEBOOK.md` l.148–153 and 199.

**Script:** `veto33.py --seeds 20`. Its full per-seed table is in `veto33.out.md`. The calcimp release copy also has fI seeds 20–27 (run later); `veto33.py` without `--seeds` covers all 28 fI seeds → `veto33_s28.out.md`: fI flagged 0/28 (seeds 20–27: max a5p − a12 ≤ +0.047), so the C36 total becomes 0/36.

**Blog-ready wording:** "In 4 of 8 early single-seed runs (3 of them only for a while), RL first vetoed most calls and then let them back in by size, big numbers first. In a 28-seed replication of the final setup, the gate never learned a size-ordered veto: it admitted ~95% of calls of every size."

---

## (b) §3.4 BBH "never helped, one cost 10 points" — DONE

**Blog claim** (blog draft): "In four mixed-task runs, training the BBH implants never helped on BBH (in one, it cost 10 points)."

**What was computed.** Each run's final `mixed_test` record (u50, mixed GSM8K+BBH test split, BBH n = 212, GSM n = 188) contains a swap eval. The trained-implant effect on the trained host is `full` (trained host + trained implants) − `trained_host_orig_tools` (trained host + original implants).

| run | BBH trained impl. | BBH orig impl. | Δ BBH (items) | ±unpaired 95% | GSM trained impl. | GSM orig impl. | Δ GSM (items) | ±unpaired 95% |
|---|---|---|---|---|---|---|---|---|
| gbC2_s0 | 73.1 | 83.5 | **−10.4** (−22/212) | ±7.8 | 38.3 | 29.8 | +8.5 (+16/188) | ±9.5 |
| gbC3_s0 | 80.2 | 82.1 | −1.9 (−4/212) | ±7.4 | 18.6 | 27.1 | −8.5 (−16/188) | ±8.4 |
| gbC3_s1 | 82.1 | 81.6 | +0.5 (+1/212) | ±7.3 | 40.4 | 26.1 | +14.4 (+27/188) | ±9.4 |
| gbC3_s2 | 76.4 | 78.3 | −1.9 (−4/212) | ±8.0 | 38.3 | 27.1 | +11.2 (+21/188) | ±9.4 |

- The original host with trained implants (`orig_host_trained_tools`) gets BBH 71.2 / 76.4 / 78.8 / 72.6.
- **No paired CI is possible.** No per-item records were saved; the swap eval is logged as aggregates only. The ± column is an independent-binomial half-width, which is conservative: a paired CI would be narrower.
- Only the −10.4 clears even that conservative bound. The other three BBH deltas (−4, −4, +1 items) are noise. On GSM, the same swap favours the trained implants in 3 of 4 runs (+16, +27, +21 items) and goes against them in 1 (−16).

**Inputs:** `rl/figures/v3/data/gbC{2_s0,3_s0,3_s1,3_s2}_steps.jsonl` (line 63, `mixed_test`).

**Script:** `gbc_swap.py`. Output is in `gbc_swap.out.md`.

**Blog-ready wording:** "In four mixed GSM8K+BBH runs, training the BBH implants never clearly helped on BBH: −10, −2, +0.5 and −2 points against the same host with the original implants (212 items; only the −10 is outside noise). The same swap on GSM8K favoured the trained implants in 3 of 4 runs."

---

## (c) §4.4 compliant S2F Δ vs the no-implant host (H) — DONE

**Blog claim** (P3-38): "with the sanity check and a forget-mask the stack gets to 72%: +4.6 points over the model alone (CI 2.6 to 6.4)". The +4.6 came from the non-compliant post-sampling swap-hook run.

**Computed.** Setup: GSM8K test (n = 1319), host a2_u300_s2 (`u0`), T = 1, seed 7, once each. CIs are paired by item `i`, 10k resamples, 95% percentile; p is an exact McNemar test.

| config (file) | acc | Δ vs H 67.70 [95% CI] | fixed / broken | McNemar p |
|---|---|---|---|---|
| S2F, compiled logit-write trigger (`test_S2Fw_u0`, miss3.list, j23342) | 72.63 | **+4.93 [+2.96, +6.90]** | 124 / 59 | 2e-6 |
| S2F, residual-write trigger (`test_S2Frw_u0`, mw_test.list after mwchk, j23562) — RL§11.5's designated compliant row | 71.95 | **+4.25 [+2.27, +6.22]** | 120 / 64 | 4e-5 |
| *sanity check:* old S2F with the swap hook, NOT compliant (`test_u0_B`) | 72.25 | +4.55 [+2.65, +6.52] | 119 / 59 | 8e-6 |

The sanity row reproduces the ledger's +4.55 [2.58, 6.44]. The CI edges differ only by bootstrap RNG.

**Inputs:** `res/fe/test_{lad_H, S2Fw_u0, S2Frw_u0, u0_B}.items.jsonl` (fe_hook test rows reduced to i, gold, ok, last, ntok). H = `lad_H` (test3.list, RL§4 row "no implants").

**Script:** `s2f_vs_h.py` (pure Python, seed 0). Output is in `s2f_vs_h.out.md`.

**Blog-ready wording:** "with the sanity check and a forget-mask, all in the weights, the stack gets to 72%: +4.3 points over the model alone (95% CI 2.3 to 6.2)". With the compiled logit-write trigger instead it is 72.6%, +4.9 [3.0, 6.9].

---

## (d) §5.2 `<calc>` fade 80 → 25 / 88 / 95–98 — DONE

**Blog claim** (blog draft): "Over RL, <calc> calls fade: from about 80% of GSM answers to about 25% … Remove the '=' implants and <calc> stays at 88% … Corrupt the '=' implants so that 25%, 55% or 87% of its answers are wrong, and <calc> goes up to 95-98%."

**Definition.** Per RL update, the share of GSM8K rollout rows that contain `<calc>`. Resumed duplicate rows are removed. Start = mean of the per-update shares over u1–5. End = mean over u41–50. Both are unweighted by row count, as in the figure (`make_new_v3.py` calc_share()/rl7()). The row-pooled end share is also given.
- The runs are the rl_library competing-tools causal study (NOTEBOOK "2026-11-20 <calc>-drop diagnostics"; completeness.md item 10).
- Training is mixed GSM8K + BBH: rollouts carry `type` and only `type == 'gsm'` rows are counted. GSM rows are ~200–360 of the 512 rollouts per update.
- Control = exact '=' implants (expr_gen / calc_gen) attached. "No '='" = those two removed. "Corrupted" = the '=' implant gives a wrong value on 25 / 55 / 87% of its fires.

**Result** (u41–50 covered by every run; all runs reach u50)

| condition | seeds | mean u1–5 | mean u41–50 | per-seed u41–50 |
|---|---|---|---|---|
| exact '=' implant (control) | 2 | 80.8 | **25.0** | 33.0 / 17.1 |
| no '=' implants | 2 | 81.9 | **88.1** | 81.8 / 94.4 |
| '=' implant 25% corrupted | 3 | 82.9 | **94.9** | 96.4 / 92.3 / 96.1 |
| '=' implant 55% corrupted | 3 | 81.4 | **94.7** | 95.0 / 94.7 / 94.4 |
| '=' implant 87% corrupted | 3 | 82.7 | **98.0** | 96.5 / 98.3 / 99.1 |

- Row-pooled end shares agree within 0.6 points (calc_fade.out.md). The numbers match `v3_new_numbers.json` `rl7` exactly.
- No CI: these are rollout shares with 2–3 seeds. The seed spread is the honest error bar. The control and no-'=' seeds are 16 and 13 points apart at the end.

**Inputs:** `rl/figures/v3/data/calc_share_{A40,H100}.jsonl`, made from each run's `rollouts.jsonl` by `rl/figures/v3/calc_share_remote.py`.

**Script:** `calc_fade.py`. Output is in `calc_fade.out.md`.

**Blog-ready wording:** "In mixed GSM8K+BBH RL runs where two '=' implants also fire on their own, <calc> calls fade from about 80% of GSM answers to about 25% (17% and 33%, 2 seeds) by update 50. Remove the '=' implants and <calc> stays at 82–94% (2 seeds). Corrupt the '=' implant so that 25%, 55% or 87% of its answers are wrong, and <calc> rises to 95–98% (3 seeds each)."

---

## (e) §5.1 IFEval: RL then attach (2b) vs RL with the implant (4), paired CI — DONE

**Blog claim** (blog draft): "The model RL'd without the implant and given it afterwards gets 53% (vs 48%) with less than half the tokens". This is IFEval, Qwen3-0.6B-Base (RL§12.3, C23).

**Definition.**
- Row 2b = the row-2 hosts (plain GRPO, seeds 0–2, u300), evaluated with the 14-rule implant attached.
- Row 4 = hosts trained with the implant attached (seeds 0–2, u300), evaluated with it.
- Both use greedy decoding and lm_eval prompt-strict over all 541 prompts.
- Per prompt, the indicator is averaged over the 3 seeds. The delta is bootstrapped over the 541 prompts (10k resamples), the same method as the rl24 IFEval analysis (`uniform/rl24/ifeval/`).

**Result**

| comparison | Δ prompt-strict (pp) [95% CI] |
|---|---|
| **2b − 4, seed-averaged** | **+4.31 [+1.17, +7.33]** |
| 2b − 4, seed 0 vs seed 0 | +4.99 [+0.92, +9.06] |
| 2b − 4, seed 1 vs seed 1 | +6.47 [+2.22, +10.54] |
| 2b − 4, seed 2 vs seed 2 | +1.48 [−2.40, +5.36] |
| check: 2b − 2 | +9.12 [+6.84, +11.52] (ledger +9.12) |
| check: 4 − 2 | +4.81 [+1.11, +8.56] (ledger +4.81) |

- Levels: 2b 52.68 (54.16 / 54.71 / 49.17) vs 4 48.37 (49.17 / 48.24 / 47.69).
- Tokens per question: 2b 546 vs 4 1364 (0.40×).
- Seed-majority indicator: 2b right & 4 wrong on 67 prompts, the reverse on 43.
- All 3 seed pairs favour 2b. The CI covers prompt sampling only; with 3 seeds there is no seed-level CI.
- Not the same on 1.7B: 2b − 4 = −2.34 [−5.18, +0.43] (RL§12.4.1).

**Inputs:** `uniform/rl24/ifeval/res/{P_s0..2}/eval_u300.imp.score.json.gz` (2b), `{I_s0..2}/eval_u300.imp.score.json.gz` (4), `{P_s0..2}/eval_u300.plain.score.json.gz` (2).

**Script:** `ifeval_2b4.py`. Output is in `ifeval_2b4.out.md`.

**Blog-ready wording:** "On IFEval, the model RL'd without the implant and given it afterwards gets 53% vs 48% (+4.3 points, 95% CI 1.2 to 7.3; all 3 seeds favour it), with less than half the tokens (546 vs 1,364 per answer)."

---

## (f) §4.2 IFEval: instructions per prompt, and how much the implant writes — DONE

**Blog claims** (blog draft): "The model still writes almost all the words while the implant adds things like bullet markers …" (P3-5) and "Lots of IFEval problems combine 2 to 3 instructions" (P3-4).

**What `circ.fires` means.** Read in `ifeval/heldout/circuit.py:783` (same in `ifeval/full14_compiled/circ14.py:790`).
- `fires += active` per decode step. `active` = the circuit's emitter has a program loaded (insert a separator / bullet / quote / end phrase, or respell or replace the host's chosen token) and writes a forced-token vector at the output.
- So one fire = one generated token chosen by the implant, not by the model. `ffail` counts fires whose forced token still lost the argmax; it is 0 here.
- **Caveat found:** run_t.py keeps stepping the circuit on rows that have already stopped (fed EOS) until the whole batch ends, so `fires` also counts steps after a row's stop.
  - 17 prompts have fires > ntok (4,668 fires on 453 tokens). All of them have a case rule active: 16 have a change_case/title instruction, and 1 is a reader false positive.
  - So the raw 7,328 / 138,942 = 5.3% overstates the implant's share. Per prompt, Σ min(fires, ntok) is a strict upper bound on implant-written tokens inside the responses.

**Result** (R§5.1 run: Qwen3-0.6B chat, greedy, plain 57.86 → implant 72.09)

| scope | generated tokens | implant-written tokens (upper bound) | implant share | model's own tokens |
|---|---|---|---|---|
| all 541 prompts | 138,942 | ≤ 3,113 | ≤ 2.2% | ≥ 97.8% |
| the 217 prompts where the implant wrote | 42,197 | ≤ 3,113 | ≤ 7.4% | ≥ 92.6% |
| the same, without the 17 fires>ntok prompts | 41,744 | ≤ 2,660 | ≤ 6.4% | ≥ 93.6% |

- On the 324 silent prompts, the output is byte-identical to the plain run (324/324), consistent with R§5.1.
- Instructions per prompt (n = 541): 1 → 305 (56.4%), 2 → 179 (33.1%), 3 → 57 (10.5%). The mean is 1.54. So 236 prompts (43.6%) combine 2–3 instructions; none has more than 3.

**Inputs:** `ifeval/transfer/results/reduced/q06_ifeval_{imp,plain}.json` (`ids`, `ntok`, `circ`, `resp_sha256`).

**Script:** `ifeval_42.py`. Output is in `ifeval_42.out.md`.

**Blog-ready wording:**
- "The model still writes at least 98% of the tokens (at least 93% on the prompts where an implant stepped in), while the implant adds things like bullet markers, quote marks or the repeated request."
- "Almost half of IFEval's prompts (44%) combine 2 or 3 instructions."

---

## (g) §4.4 "never solved in 32 tries; about a quarter have a wrong gold" — DONE (CPU, from existing samples)

**Blog claim** (blog draft): "only 2-3% (on dev) are never solved in 32 tries, and about a quarter of those have a wrong gold answer". The section is the frozen-host §4.4 stack, host u0 = `a2_u300_s2`.

**Definition.**
- Tries are independent single-attempt samples from the u0 B-stack (calc + chain2 + sanity check + forget mask, compiled triggers), T = 1. This is the SC harness (`fe_hook.py` KIND SC): every sample's parsed answer and `att_ok` are logged.
- An item is never solved if 0 of its tries are correct (GSM8K scorer `C.correct`).
- dev = GSM8K train split B, shuffled with Random(0), first 600 items. The order was rebuilt and checked gold-for-gold against the rows.
- **Gold check criterion** (`nv_gold_labels.json`): hand-read each never-solved item (question, gold solution, the model's answers) and label it:
  - WRONG: the gold contradicts the question as written.
  - AMBIG: two defensible readings or units, and the model's majority answer is the other one.
  - ILLPOSED: internally inconsistent, with no defensible alternative from the model.
  - OK: the gold is right and the model is wrong.

**Tries used.** No new GPU run was needed. The existing u0 dev SC runs SC×8 and SC×4 at eval seeds 7 / 11 / 13 (fe_hook KIND SC rows of the dev controls; shipped as `res/fe/dev_sc{8,4}_s{7,11,13}.items.jsonl`) give 36 tries per item.
- **Independence check:** sc4 and sc8 at the same seed are different samples (G changes the RNG layout). On the 174 items whose first 4 sc8_s7 answers vary, sc4_s7 repeats them exactly on 1 item; across different seeds the figure is 4 of 174.
- "First 32 tries" = sc8 s7/s11/s13 + sc4 s7/s11.

**Result (dev 600)**

| tries per item | never solved | % of 600 |
|---|---|---|
| 1 | 118 | 19.7 |
| 4 | 52 | 8.7 |
| 8 | 34 | 5.7 |
| 16 | 25 | 4.2 |
| 24 | 24 | 4.0 |
| **32** | **23** | **3.8** |
| 36 | 23 | 3.8 |

- The curve is flat from about 16 tries. At 36 tries, 5 more items are solved exactly once.
- Gold labels of the 23 items never solved in 32 (or 36) tries:
  - **3 WRONG.** dev 6: "6 square inches" tile, gold 96, the literal answer is 576, which the model gives 36/36. dev 303: gold 90 takes 3/4 of the remaining leaves, while the question says 3/4 of week 2's shed, so 180, which the model gives 36/36. dev 519: the #### line says 4, but its own solution ends x = 10.
  - **2 AMBIG.** dev 564: "three times older" gives 40 or 30; the model says 30 in 34/36. dev 567: 350 cents vs $3.50; the model says 3.5.
  - **1 ILLPOSED** (dev 213: per-day breaks counted once).
  - **17 OK.**
- So **5 of 23 (22%) have a wrong or ambiguous gold, 6 of 23 (26%) counting the ill-posed one.** Genuinely unsolved with a sound gold: 17 / 600 = **2.8%**.
- The full item list with answers is in `nv32_dev36.out.md`.

An SC×32 GPU confirmation (one process, seed 7) was planned but not run; it is not needed for the claim.

**Scripts:** `nv32.py` and `nv_gold_labels.json`. Run (needs `data/gsm8k/shots_v1.json`, see README.md): `python3 checks/sourced/nv32.py dev checks/sourced/res/fe/dev_sc8_s{7,11,13}.items.jsonl checks/sourced/res/fe/dev_sc4_s{7,11,13}.items.jsonl`.

**Blog-ready wording:** "What's left is half selection (a right attempt is there but the vote misses it) and half hard questions: only about 4% of dev questions are never solved in 32 tries, and about a quarter of those have a wrong or ambiguous gold answer (so ~3% are truly out of reach) :)"

(g) sources only the clause after the semicolon. The "half selection / half hard" split belongs to the existing §4.4 oracle numbers and is not re-derived here. Change the blog's "2-3%" to "about 4%", or keep "2-3%" and say it means items with a sound gold (17/600 = 2.8%).

## (h) §1.2 "Claude checked that [the Read layer] works with all of ~151k tokens in Qwen's vocabulary" (audit P1-19) — DONE (CPU)
Job j25423 (CPU only, 9 s): `vocabjob/vocab_reader.py Qwen/Qwen3-0.6B-Base` re-runs `micromachines.make_reader`
(the reader the implants use; it asserts exactness on every embedding row at each attach) and logs what the assert never logged.
Output: `vocab_reader_Qwen3-0.6B-Base.json`, `vocabjob/res/run06.log`.
- **Exhaustive check:** Qwen3-0.6B-Base embedding matrix = 151,936 rows (151,669 tokenizer entries + 267 padding rows).
  calc_s3 and calc_s4: 26 symbols × 151,936 rows = 3,950,336 flag cells each, **0 mismatches**. 35 rows are read as some symbol
  (the 10 digits, + − * / . = and their spaced forms, ' x'/' ×', '$', newline variants); every other row reads all-zero.
- **Margins:** every non-target pre-activation is ≤ −2.0 (clamps to 0) and every target pre-activation is ≥ +6.0 (clamps to 1),
  on every symbol (by construction of the closed-form scale; checked in fp32 on the bf16 embeddings).
- **Regex cross-check on the decoded vocabulary:** Qwen has exactly 10 tokens with an ASCII digit (the single digits 0–9),
  and all 10 are read; no token mixes a digit with other characters (Qwen splits numbers into single digits). 12 tokens hold
  a non-ASCII digit (full-width １ ２ … １０ ２０); they are deliberately not read. 4,767 tokens contain an operator character
  inside a longer string ('==', '//', '->', '**', …); they are not operators and are correctly not read.
- **Verdict:** P1-19 → OK, now sourced. Possible wording: "Claude checked the Read layer on all ~151k tokens in Qwen's vocabulary
  (every row of the embedding table): it flags exactly the right tokens, with no false positives." The blog's "~151k" is right. 1.7B not re-run (its model files were not available on the CPU node);
  DETAILS.md's "re-checked on the host's vocabulary" for bigger hosts rests on the same assert at attach time.
