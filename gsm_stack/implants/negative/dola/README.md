# lane dola: DoLa-style contrastive layer decoding as a weight-space implant (todo 'dola')

Idea: implant logits z' = z + alpha*(z - z_l), z = lm_head(norm(h_27out)) (host logits), z_l = lm_head(norm(h_l)) with h_l the
layer-l input residual (as the side-net reads it); optional APC (tokens with p(z) < beta*max p(z) -> -inf). Equivalent single
matmul: z' = E . ((1+alpha) n(h_fin) - alpha n(h_l)), n = final RMSNorm (shared weights). Applies only to host-sampled tokens.
Base = rung 4 (implants/sidenet/aw, `bash lad4_job.sh sn dev|dev2`, rows sidenet/aw/out/sn_{dev,dev2}).

## log (UTC, newest last)
- 12:40Z offline pair prescreen (CPU, no GPU): `python3.11 implants/dola/pre/pairs_off.py` -> pre/pairs_off.{log,json}. Uses the
  branch-probe teacher-forced states at fork positions (probe/branch/data; layers 14/18/22/27 input + 28 = final-norm input) and
  unembedding rows; D_L = z_L[a] - z_L[b] (normalisation cancels), D' = D_28 + alpha (D_28 - D_l). See results below.
- 12:39Z GPU prescreen job submitted: `cd implants/dola/pre/job && <queue-submit> $PWD 2400 job_dola_pre -- bash run.sh
  python3 dola_pre.py --sets dev,dev2` -> **j23532**. Full-vocab z, z_l (l in 14/18/22/24 + dyn = max-JS layer), alpha in
  .1/.25/.5/1, beta in 0/.1; every segment position of all dev/dev2 mixed attempts (32/item) + always-right items' seed-7
  attempts (tables: `python3.11 pre/mk_tables.py`). Outputs per-position lp'(x), argmax changes, fork candidates.
- 12:48Z j23532 DONE (6m52s). dev 1.68M positions (11807 attempts), dev2 0.88M; host lp vs stored sampled lp MAE 0.011 / 0.009.
  Analysis: `python3.11 implants/dola/pre/ana.py > pre/ana.log` (ana.json). NOTE selection: (l, alpha, beta) picked on dev only;
  dev2 + seeds are the confirmation. No trainC atlas with control attempts exists (trainC has forks only; pairs_off covers it).

### Prescreen results (pre/ana.log; pre/pairs_off.log)
Columns: atlas branch pairs (clean; dev 1686, dev2 921): sign flips of lp(r)-lp(w) toward right / toward wrong, full-vocab argmax
wrong->right / right->wrong; soft trie-fork pairs weighted acc (unbiased labels; host 0.502 / 0.499); drift on always-right
attempts (host-sampled tokens): mean dlp of the sampled token, argmax-change rate; seq: SNIS dP(right) per mixed item from
w = exp(sum dlp) (ESS/n in brackets). APC rows: dlp averages include clamped -99 for the 0.4% sampled tokens APC removes.

| cfg | dev flips ->R/->W | dev amax W->R/R->W | dev fork wacc | dev drift dlp / amax chg | dev SNIS dP | dev2 flips | dev2 amax | dev2 fork wacc | dev2 SNIS dP |
|---|---|---|---|---|---|---|---|---|---|
| host | – | – | 0.502 | – | – | – | – | 0.499 | – |
| 14 a.1 | 38/47 | 45/56 | .515 | −.002 / .006 | −.004 (.54) | 22/38 | 24/37 | .510 | −.008 |
| 18 a.25 | 81/131 | 79/118 | .518 | −.019 / .016 | +.007 (.17) | 53/99 | 50/83 | .512 | −.003 |
| 18 a.5 | 112/221 | 105/185 | .515 | −.064 / .027 | +.011 (.08) | 71/145 | 63/118 | .509 | +.004 |
| 18 a.5 b.1 | 103/190 | 102/171 | .498 | (APC) / .022 | +.033 (.08) | 66/132 | 59/111 | .491 | +.004 |
| 22 a.5 | 144/299 | 128/242 | .500 | −.119 / .044 | −.001 (.06) | 94/173 | 72/137 | .514 | +.033 |
| 24 a.5 | 124/257 | 105/207 | .494 | −.116 / .040 | +.012 (.07) | 75/163 | 61/131 | .499 | +.058 |
| 22 a1 | 182/401 | 144/291 | .500 | −.449 / .081 | +.002 (.04) | 107/223 | 76/157 | .506 | +.009 |
| dyn a.5 | 107/182 | 98/167 | .514 | −.042 / .021 | −.005 (.11) | 68/111 | 63/97 | .515 | +.003 |
(all 40 configs in pre/ana.log; trainC/dev/dev2 pair-only sweep incl. alpha 2 and l=27 in pre/pairs_off.log: same picture)

Reading: at the branch token DoLa moves the pair the WRONG way in every config (flips toward wrong exceed flips toward right
~1.5-2x on both sets; the atlas pair labels carry the partner-selection artifact that favours the host's token, so this is biased
against any change); on the unbiased soft fork labels every config is at chance (0.475-0.524; host 0.50). Drift is small for
alpha <= 0.25 (1-2% argmax changes on right attempts), 4-8% at alpha 0.5-1. Sequence-level SNIS dP(right) is within +-0.03 with
ESS 4-17% (noise level). => token-level prescreen gate FAILS (no right-branch preference); sequence level ambiguous/noise.
Because the in-stack build was cheap (patch ready), I run 2 dev-selected configs in-stack anyway as the decisive test:
C1 = 18:0.5:0.1 (best dev SNIS overall), C2 = 24:0.5 (best dev SNIS without APC; tie with 18:0.5).

### In-stack build (aw/ = copy of sidenet/aw; diff vs rung 4: stk/comp_serial.py.rung4, stk/c2_engine.py.rung4)
- stk/aw_dola.py: parse + dola_logits(z, z_l, on, alpha, beta). comp_serial._step_triton (DOLA set): at layer l, the layer-l
  input residual x goes through the host's final RMSNorm kernel (same lnf weights) into buffer rows [B, 2B); after the final norm
  one lm_head matmul over [n(h_fin); n(h_l)] gives z, z_l; logits = z + alpha (z - z_l) (+ APC mask from z) on rows whose step has
  no write (bits | calc-write flag == 0), z on all others; the host sampler (T=1 Gumbel) draws from these. calc v2's ALT argmax
  reads the plain z (c2_engine proxy). DOLA unset: code path untouched. Not covered: the first token after prefill (sampled from
  the prefill logits in fastroll._f_admit; 1 token per item, plain host logits).
- test_dola.py (CPU fuzz vs float64 log-prob reference, 1853 rows, ties, alpha .1-2, beta 0/.01/.1/.5, gate on/off): 0 bad,
  max |p - p_ref| 7.6e-7. Runs at the start of every job (dola_job.sh).
- 12:58Z submitted `cd implants/dola/aw && <queue-submit> $PWD 2400 job_dola_r1 -- bash dola_job.sh off,d18_0.5_0.1,d24_0.5`
  -> **j23536** (off = gate-off dev seed 7; configs seeds 7,11 dev+dev2).
- 13:39Z j23536 started (queued 23m), DONE 13:57Z rc 0 (ran 17m53s). Outputs aw/out/{off_dev,d18_0.5_0.1_{dev,dev2},d24_0.5_{dev,dev2}}.
  GPU fuzz in job: AW FUZZ PASS, DOLA FUZZ OK.

### In-stack results (seeds 7,11; ref = rung-4 rows implants/sidenet/aw/out/sn, same seeds)
Gate-off: `python3 aw/eq_cmp.py implants/sidenet/aw/out/sn_dev aw/out/off_dev` -> dev s7 tok-identical 600/600, acc 90.67->90.67,
tok/q 1347.6->1347.6. Summary: `python3 implants/ladr3/aw/summ.py implants/sidenet/aw/out/sn aw/out/d18_0.5_0.1 aw/out/d24_0.5`

| config | dev s7/s11 = mean (Δ vs rung 4) | dev f/b | dev tok/q | dev2 s7/s11 = mean (Δ) | dev2 f/b | dev2 tok/q | per-attempt acc dev / dev2 |
|---|---|---|---|---|---|---|---|
| rung 4 (ref) | 90.67/90.67 = 90.67 | – | 1377.2 | 89.00/87.67 = 88.33 | – | 1518.8 | 64.33 / 61.37 |
| DOLA=18:0.5:0.1 | 88.50/88.83 = 88.67 (−2.00) | 21/45 | 1269.3 | 86.67/85.33 = 86.00 (−2.33) | 9/23 | 1484.3 | 64.65 / 59.22 |
| DOLA=24:0.5 | 88.17/87.33 = 87.75 (−2.92) | 13/48 | 1665.4 | 87.33/86.67 = 87.00 (−1.33) | 13/21 | 1897.2 | 61.42 / 56.96 |
pooled f/b: 30/68 and 26/69 (sign-test p = 1.000). Per-attempt acc = attempts whose parsed value == gold / all attempts made (adaptive
stopping changes attempt count, so treat as indicative). 4-seed confirmation NOT triggered (needed >= +0.3 on both sets).

### Verdict: NEGATIVE (not a rung candidate)
Teacher-forced prescreen: no right-branch preference at any (l, alpha, beta) (pair flips go toward the wrong token 1.5-2x more often;
unbiased fork labels at chance). In-stack, the two dev-selected configs cost 1.3-2.9 points on both sets with ~2:1 breaks:fixes;
the APC config is the mildest on tok/q, the l=24 contrast makes chains longer (+20-25% tok/q, more retries). Consistent with the
earlier lanes (trunc/M1/M2): the right/wrong branches are the host's top-2 beliefs, and the premature-layer contrast does not
carry information about which one is right; it only perturbs the distribution away from the calibrated host.
What is weights: z' = E((1+a) n(h_fin) - a n(h_l)) = one fixed linear map (host final RMSNorm weights + tied lm_head, the
premature residual read at layer l input) + an APC comparator (z < max z + log beta -> -inf); the per-row on/off is the existing
write-flag register state (sw bits | calc write flag), no host Python per row. Gate DOLA unset = rung 4 code path (verified
token-identical). Build kept in aw/ (patch: stk/comp_serial.py vs .rung4, stk/c2_engine.py vs .rung4, stk/aw_dola.py).
Not tried (low prior given the above): per-attempt alpha via the attempt counter, smaller alpha (<=0.25 drift ~1-2% would be
near-null by construction).
