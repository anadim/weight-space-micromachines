# Rung 4 candidate all-weights stack: drop-in for the stack lane, from the implant lane lane sidenet (2026-11-21 ~10:00Z)

Rung 4 = rung 3 (avf) + a multi-layer linear verifier register (AW_SN) that feeds the final answer-group writer.
dev600 / dev2, u0 host ckpt_a2s2, seeds 7/11/13/17: **90.46 / 88.33**, at 1401 / 1522 tok/q and 7.7 / 8.2 tries (= rung 3).
Rung 3: 90.29 / 87.33. Rung 4 vs rung 3 fixes/breaks: 19/15 on dev, 16/4 on dev2; pooled 35/19, one-sided sign p = 0.020.
Whether this counts as a rung is the orchestrator's call (see LADDER.md).

| set | rung 3 per seed (7/11/13/17) | rung 4 per seed | mean (Δ) | f/b | tok/q |
|---|---|---|---|---|---|
| dev | 90.50/90.33/89.83/90.50 = 90.29 | 90.67/90.67/90.17/90.33 | 90.46 (+0.17) | 19/15 | 1401.3 (= rung 3) |
| dev2 | 87.67/87.67/87.00/87.00 = 87.33 | 89.00/87.67/88.33/88.33 | 88.33 (+1.00) | 16/4 | 1522.4 (= rung 3) |

Generation is token-identical to rung 3 up to the final `####` writer line: 3600/3600 rows match before that line, with the same tries.
Only the writer's pick changes, so tok/q is unchanged up to the length of the written answer string.

## Exact env (dev; from lad4_job.sh cfg 'sn')
    env SAN_SMAX=8192 BMAX=128 bash aw.sh sn_dev AW_SETS=dev AW_SEEDS=7,11,13,17 \
        SW_NS=24 SW_MAXN=7936 AW_ADAPT=A=0.7/24/4:S=4/0.5:B=3:C=400 WPR=1 AW_VF=1 AW_SN=$PWD/stk/sn_w.pt
- `bash lad4_job.sh sn dev` (or `sn dev2`) sets the triton ptxas path and runs the test_aw fuzz first.
  - `off` = rung 3 (AW_SN unset), dev seed 7.
  - `snq` = rung 4 smoke, dev seed 7.
- AW_SN requires AW_VF=1. With AW_SN unset, the code path is the rung-3 path: dev seed 7 is 600/600 token-identical to rung 3.
- Weights files:
  - stk/sn_w.pt: side-net vector, bias and BLD writer θ.
  - stk/vf_w.pt: rung-3 θ. It is still loaded but unused when AW_SN is set.

## Files (snapshot of gsm_stack/implants/sidenet/aw; rung 3 + the following)
- stk/aw_sn.py: loader (per-layer (1024, 2) projection) and `vf_vote_sn` (the ClusterLin writer with the BLD per-slot feature).
- stk/comp_serial.py: rung-3 comp_serial plus the AW_SN hooks. The diff vs rung 3 is in aw_sn_comp_serial.diff (about 50 lines).
- stk/fastroll.py and stk/aw_hook.py: harvest R_sn into the rows as field `sn` (per slot). The file tag gets the suffix `_sn`.
- test_sn.py: CPU fuzz of `vf_vote_sn` against the offline re-pick code. It needs gsm_stack/implants/sidenet/repick_sn.py and vfrefit/off.
  - Real θ: 20000 trials, 0 mismatches.
- sn_check.py: compares in-stack rows with the offline side-net.
- For TEST / GSM-Hard through eval/fe_hook.py:
  - Do the rung-3 recipe.
  - Also copy aw_sn.py and sn_w.pt, plus the comp_serial / fastroll / aw_hook changes.
  - Add env `AW_SN=<path>/sn_w.pt`.

## What is weights
Everything in rung 3, plus the following:
- **Fixed linear vectors** on the layer-input residual states of layers 14, 18, 22 and 27, as a (1024, 2) [wH, wM] matrix per layer (8192 weights + bias in total).
  - They are the 5-fold mean of linear side-nets trained on trainC rung-3 rows with a pairwise loss (cfg c4:HM:lin:pair).
  - The trainC standardisation is folded into the vectors.
  - Each step projects the current token's state: u += x_L · [wH_L, wM_L].
- **Registers per row:**
  - s_snH: the H projection, captured at the token completing '####' (like the L18 probe register).
  - s_snM / s_sncnt: the running sum and count of M projections at host-read tokens that contain '\n', in the segment before '####'. This uses a fixed token table NL[tok] and the gate flag s_snhs.
  - All of these are reset at each new segment.
  - At the attempt's END: slot register R_sn[slot] = s_snH + s_snM / max(cnt, 1) + b.
- **The final writer** (`vf_vote_sn`): ClusterLin over answer-symbol groups with a fixed 18-dim θ + c0.
  - The per-slot feature is the blend b = max(sn, −12) + max(lg, −12).
  - The group term Σσ(lg) becomes Σσ(b/2).
  - θ was fit on trainC rows using the side-net's out-of-fold trainC scores (same weights).
- No host-side Python makes a per-row decision. The projections, gates, counters, slot commits and argmax writer are all batched tensor ops inside the compiled decode step.

## Exactness
- `off` reproduces rung 3 token-identically: dev seed 7, 600/600.
- In-stack R_sn vs the offline side-net (teacher-forced extraction of the same attempts), over 28084 committed attempts:
  - corr 0.9989, MAE 0.079, max 0.48. This is bf16 noise of the same class as the L18 probe check.
- Picks vs the offline BLD writer on the row's own registers: 0/3600 mismatches.
- Picks vs the offline re-pick prediction: 3/3600 differ. All three are near ties, with a group-score margin of at most 0.09.
