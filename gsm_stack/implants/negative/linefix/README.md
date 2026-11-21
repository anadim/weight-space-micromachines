# linefix: step-detector line rewind. Gate result: NO-GO (step 1 only)

**Idea.** At each reasoning line end, the step-error detector (logistic on L18+L22, features hdh = [h, h − h_prev_line_end]) reads the host state. If it fires, the forget mask hides the line just written and the host resamples it from the previous line end. At most R rewinds are allowed per attempt.

**Gate.** Build the implant only if the offline per-attempt gain on dev is at least +1.0 point.

**Result.** The raw gain is +0.05 pts on dev and −0.07 on dev2 at the 2% threshold. Even under the most favourable calc correction it is at most +0.25. **I did not build it (no step 2).**

## Method (`gate/gate.py`, GPU jobs j23235 and j23251; analysis in `ana.py`, output in `gate_results.md`)

**Detector refit**
- Refit on trainC as in divprobe task a: hdh, l2=0.1, 47,950 lines, 3,056 D positives.
- OOF AUROC is 0.829, which matches probe.md.
- Thresholds at the train-OOF line FPR: 1% → 2.239, 2% → 1.821, 5% → 1.183.
- Weights are in `gate_full_out/detector.pt`, in input units over [h18, h22, Δh18, Δh22]. The layer inputs are the nl rows from probe/extract.py.

**Scoring and rewind**
- I scored every line end with step ≥ 1 before '####' in all dev and dev2 attempts (`gate_full_out/scores_*.npz`).
- Rewind point: the first flagged line of each attempt.
- I teacher-forced the 8-shot prefix + C.q_ids(q) + seg_tokens[:prev_line_end+1] through HF Qwen3-0.6B-Base + ckpt_a2s2 (bf16).
- From there I sampled K=4 continuations with plain `generate` at T=1 (no top-k or top-p), up to the '#### n' line or a blank line, max 400 tokens.
- Grading used C.correct.

**Controls:** 300 per set, at a random line of attempts that were not flagged at the 5% threshold.

**Coverage**
- Jobs ran in order of increasing prompt length, with a 45-minute cap.
- All first-flag lines at 1% and 2% were covered for 85% of attempts (the missing ones are the longest prompts).
- At 5%, only lines that coincide with the 2% first flag were covered (40%).

**Caveats**
- The calc implant is absent during continuation.
- There is a single rewind (R=1), and the resampled line is not re-checked.
- Positions are contiguous, whereas in-stream they would skip the hidden line.

## Results (per attempt; P = mean over K=4 continuations)

| set | thr | flagged att | flagged wrong: n, P_fix | of which flag at D line / before D / after D | flagged right: n, P_keep | Δ per-att (pts, scaled to full coverage) | Δ if only breaks above control count |
|---|---|---|---|---|---|---|---|
| dev | 1% | 6.1% | 592, 0.144 | 0.174 / 0.258 / 0.094 | 313, 0.756 | **+0.06** | +0.10 |
| dev | 2% | 10.1% | 984, 0.143 | 0.170 / 0.282 / 0.086 | 657, 0.800 | **+0.06** | +0.13 |
| dev | 5%* | 19.3% | 867, 0.146 | 0.167 / 0.285 / 0.090 | 615, 0.807 | +0.11 | +0.25 |
| dev2 | 1% | 6.5% | 300, 0.158 | 0.193 / 0.250 / 0.089 | 202, 0.785 | **+0.05** | +0.11 |
| dev2 | 2% | 10.7% | 518, 0.136 | 0.161 / 0.265 / 0.062 | 362, 0.791 | **−0.07** | +0.04 |
| dev2 | 5%* | 21.8% | 454, 0.145 | 0.162 / 0.269 / 0.069 | 327, 0.796 | −0.02 | +0.19 |

\* At 5%, the covered rows are those where the 2% and 5% first flags coincide, so the scaling assumes those rows are representative.

**Controls.** The control rewinds are at a random line of unflagged right attempts.
- P_keep is 0.955 on dev (n=270) and 0.943 on dev2 (n=274).
- So resampling plus the missing calculator breaks at most about 5%.
- The 20% break rate on *flagged* right attempts is therefore real: the detector fires on fragile spots.

**Plurality-vote simulation.** Flagged attempts' answers are replaced by a random continuation, averaged over 50 draws. The vote changes by +0.08 to +0.16 on dev (base 87.25) and +0.06 to +0.16 on dev2 (base 85.17).

## Why it fails

**1. A rewind rarely fixes the error.** Only 14% of flagged wrong attempts become right.
- In 42% of cases the rewind lands on the same wrong answer (the attractor), and 44% reach a different wrong answer.
- The rewind fixes only 17% when the flag is exactly at the D line and 9% when it is after D.
- The items of these attempts have p_right ≈ 0.32, so **a whole fresh attempt is about 2× better than a line rewind.** The error is planned before the flagged line, consistent with the atlas: 89% of branches come before the D line.

**2. False alarms are costly.** Flagged right attempts keep the right answer only 0.76–0.80 of the time. That is about the item's base rate (0.69–0.79) and far below the control's 0.95. The detector fires where the trajectory is genuinely uncertain, so resampling there converts right attempts to wrong ones at roughly the rate it converts wrong ones to right.

**3. The upper bound is too low.** Even if no flagged right attempt broke at all, the gain would be +0.57 to +0.86 points at 1–2%. That is below the +1.0 gate. At 5% the same bound is +1.7, but the real result at 5%, breaks included, is about 0.

## Implications
- A line-level rewind is not a good use of the step detector.
- Its signal is better spent as a **voter feature**: down-weight attempts by their max D-score (probe.md §6.2a).
- Alternatively, use it to trigger a **whole new attempt**, where the fresh p_right (~0.3) beats the rewind's 0.14. That would only help if the attempt budget allows it, and probe.md §5 shows that value-gated restarts lose.

## Files
- `gate/`: job dir (gate.py, run.sh, copies of probe common/fit/wsmrepo and tables). `gate/out/ctrl.jsonl.gz` holds the controls.
- `gate_full_out/`: the main run (gen.jsonl.gz with 2,521 rewinds × 4 continuations), the detector and line scores.
- `ana.py` → `gate_results.md`.
