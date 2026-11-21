# vfeat — cheap in-graph voter features (lane voter-feat)

Host: frozen Qwen3-0.6B RL+calc, stack S3-serial (8 serial attempts, probe-weighted voter writes `#### X`).
Base voter = Σ sigmoid(lg) over same-value kept slots. Base offline sim reproduced exactly: seeds 7,11 dev 88.42 / dev2 85.33;
4 seeds (7,11,13,17) dev 88.50 / dev2 85.75; trainC (seeds 23,29) 87.12.
All fitting on **trainC only** (2 seed streams + 8 random-subset bootstrap streams/item); dev/dev2 only evaluated. No test split.

## Features (feat.py → cache/vf_{set}.pkl, per attempt)
| name | definition (all from the attempt text / sampled logp — in-stream computable) |
|---|---|
| qcov_d | fraction of distinct question digit-numbers present in the attempt body (before `####`) |
| qcov_w | same, question numbers incl. number words (twice, half, dozen, one..twenty, ...); 'half' also covered by 2/0.5/50 |
| nmiss_d / nmiss_w | count of missing question quantities |
| nnum | distinct numbers in the body / 10 |
| z_ntok | attempt length (tokens), z-scored within the stream (8 attempts) |
| fk{30,50,77,90} | per 100 host tokens: sampled tokens with p < t (close-fork proxy; trainC has no top-2 data) |
| tfk3 / tfk5 | true close-fork rate p2/p1 ≥ .3/.5 (probe/branch pos data; dev/dev2 mixed attempts only) |
| ans_last_eq | answer == last `= v` of the body |
| z_* | within-stream z-score of the feature |

fk77 correlates 0.91 with the true close-fork rate tfk3 (mean tfk3 per 100 tok: right 5.2, wrong 6.8 — reproduces branch.md).

## Per-feature AUROC (offline.log; committed attempts; pooled / within-item right-vs-wrong pairs; > 0.5 = higher → right)
| feature | trainC | dev | dev2 |
|---|---|---|---|
| lg (base probe) | .864 / .701 | .824 / .766 | .837 / .684 |
| qcov_d | .525 / .522 | .541 / .534 | .520 / .528 |
| qcov_w | .527 / .527 | .545 / .536 | .523 / .533 |
| nmiss_w | .473 / .473 | .454 / .464 | .476 / .467 |
| nnum | .417 / .571 | .423 / .601 | .393 / .571 |
| z_ntok | .520 / .522 | .525 / .545 | .516 / .529 |
| fk77 (proxy) | .283 / .422 | .285 / .404 | .258 / .400 |
| tfk3 (true) | – | .356 / .422 | .346 / .427 |
| ans_last_eq | .537 / .535 | .549 / .539 | .534 / .521 |

* Fork density is a real but **redundant** signal: within-item .40–.43 (i.e. 0.57–0.60 inverted) but it is mostly length/difficulty
  and already absorbed by lg; every voter that adds it gets worse or no better (below).
* Question-number coverage is weak (≤ .55). By atlas category (bycat.log), on *dropped* items qcov_w within-item AUROC is only
  .55 / .53 (dev / dev2), while nnum is .64 / .65 and z_ntok .59 / .59. In the base-vote gap streams (vote wrong, a right
  attempt exists: dev 122, dev2 97) **qcov is identical for the right cluster and the voted wrong cluster in 99/122 and 82/97
  streams**: the wrong attractor nearly always mentions every question number too — a "dropped" clause is dropped in meaning,
  not literally. Longer (z_ntok, right>wrong 84/122, 69/97) and more-numbers (nnum 78 vs 27, 47 vs 31) attempts are the
  better signal: the right cluster does more of the work.

## Voters (offline, trainC fit; d = Δ vs base, 4 seeds / seeds 7,11; f/b = fixes/breaks over 4 seeds)
Listwise exp slot weights (`voter/learn.py` Listwise; w = exp(θ·f)) — none passes the gate (offline.log):
| voter | dev d4 / d2 | dev2 d4 / d2 |
|---|---|---|
| lg+sane | −0.21 / −0.17 | +0.25 / +0.17 |
| lg+sane+qcov_w | −0.17 / +0.00 | +0.42 / +0.33 |
| lg+sane+z_ntok | +0.12 / +0.42 | +0.33 / +0.33 |
| lg+sane+qcov_w+fk77 | −0.38 / −0.25 | +0.25 / +0.33 |
| lg+sane+nnum+z_ntok+qcov_w | +0.12 / +0.33 | +0.67 / +0.67 |
| 9 cheap feats (qcov, nmiss, fk50/77, z_ntok, z_fk77, ans_last_eq) | +0.12 / +0.33 | +0.58 / +0.67 |
| sig(lg)·exp(−a·fk77), best a | ≤ 0 | ≤ 0 |
| sig(lg)·exp(−a·nmiss_d), a=1 | +0.21 / +0.08 | +0.25 / +0.17 |

Answer-cluster conditional logit (`voter/learn.py` ClusterLin: per answer cluster g = [count/8, log count, Σ sigmoid(lg),
mean f, max f, min f]; score = θ·g; trainC fit, l2 1e-3) — cluster.log:
| voter (ClusterLin) | trainC | dev acc4, d4 / d2, f/b | dev2 acc4, d4 / d2, f/b |
|---|---|---|---|
| lg+sane (family only, no new feature) | 87.12 | 88.79, +0.29 / +0.33, 11/4 | 85.83, +0.08 / +0.50, 6/5 |
| lg+sane+nnum+z_ntok | 88.16 | 89.00, +0.50 / +0.50, 31/19 | 86.08, +0.33 / +0.50, 18/14 |
| **lg+sane+nnum+z_ntok+qcov_d (built)** | 88.26 | **89.21, +0.71 / +0.58, 33/16** | **86.33, +0.58 / +0.67, 20/13** |
| lg+sane+nnum+z_ntok+qcov_w | 88.43 | 89.21, +0.71 / +0.75, 34/17 | 86.33, +0.58 / +0.50, 22/15 |
| + fk77 + nmiss_w | 88.26 | 89.04, +0.54 / +0.58 | 86.25, +0.50 / +0.50 |

Robustness (robust.log, qcov_w variant): 3 bootstrap seeds × l2 {1e-3, 1e-2}: dev d4 +0.50…+0.79, dev2 d4 +0.50…+0.75 in all 6
refits. Paired item-bootstrap 95% CI: dev ≈ [+0.0, +1.4], dev2 ≈ [−0.3, +1.6] — **not individually significant**.

**Gate** (≥ +0.5 on both dev and dev2, mean over seeds): passed (marginally) by ClusterLin + {nnum, z_ntok, qcov}. Caveats:
ClusterLin was tried after the listwise family failed (forking paths, 2 families × ~10 feature sets judged on dev/dev2);
the family alone gives +0.29 / +0.08; qcov contributes only ~+0.2 / +0.25 of the gain. qcov_d (digits) was built instead of the
trainC-preferred qcov_w (88.43 vs 88.26) because number words would need word-token tables in-graph; on dev/dev2 the two tie.

## Implant (job/, SER_VF=1)
* `job/stk/numreg.py`: batched token automaton (classes digit / ',' / '.' / other — Qwen splits digits) that reproduces
  `\d+(?:\.\d+)?` after thousands-comma removal; values M/10^E in float64; per-row distinct set (K=32 body, 16 question).
* `job/stk/serial_stack.py` (all edits under `if self.vf`): question set read from the prompt tokens at wave start; body set
  read on live sampled tokens with reader phase 0 (up to and incl. the `####` token), reset at every new segment (wait /
  repair), flushed at commit; segment-start register → ntok. At commit: R_nn, R_qc, R_nt. Voter: z_ntok over the committed
  slots, ClusterLin cluster scores (same-matrix sums, masked max/min, θ from `job/stk/vf_w.pt`, trainC fit) replace the pick;
  `pick0` = base voter on the same registers (diagnostic). Reads only — nothing is written into the residual stream, so all
  tokens up to the final vote are unchanged; SER_VF unset → no code path changes (voter2 engine, SER_V2 off).
* `job/stk/gen_hook.py`: rows get nn, qc, nt, pick0; tag suffix `_vf`. Launcher: `job/run_vf.sh SEEDS SETS`.
* `test_vfeat.py` (CPU, stubs as voter2): [1] automaton fuzz 8192 random streams vs regex reference: 0 mismatches;
  [2] automaton on all 9546 valid dev2 attempts (real Qwen tokens) vs cache nnum / qcov_d: 0 / 0 mismatches;
  [3] compiled `_act` fuzz (vf on and off) vs numpy ClusterLin (voter/learn.py) and base voter: 0 pick / pick0 / register /
  `#### X` queue mismatches (7360 rows; 470 where vf and base differ); [4] token-level `_read`/`_act` drive over two attempts: PASS.
  `--runs`: in-stream registers vs offline features on rows token-identical to gen/out.

## In-stream runs (gq j23248 seeds 7,11 and j23260 seeds 13,17; `gsub.sh implants/vfeat/job … -- bash run_vf.sh SEEDS dev2,dev`, 13 min each)
Outputs: `job/out/*_vf.jsonl.gz` (+ nn, qc, nt, pick0); analysis `ana.py` → ana.log; `test_vfeat.py --runs` → runs.log.

Register check (`--runs`, rows token-identical to gen/out): dev2 9600/9600 attempts nnum / qcov_d / ntok all exact; dev
7884 attempts, 1 mismatch each (one attempt). Tokens/q unchanged where streams are identical (vf is read-only).

| set, seed | vf (written `#### X`) | pick0 same stream | Δ same-stream (fix/brk) | base gen/out |
|---|---|---|---|---|
| dev2 s7 | 84.67 | 85.33 | −0.67 (2/4) | 85.33 |
| dev2 s11 | 87.33 | 85.33 | +2.00 (7/1) | 85.33 |
| dev2 s13 | 87.33 | 85.67 | +1.67 (7/2) | 85.67 |
| dev2 s17 | 86.00 | 86.67 | −0.67 (4/6) | 86.67 |
| **dev2 mean (4 seeds)** | **86.33** | 85.75 | **+0.58** (20/13), CI [−0.33, +1.58] | 85.75 |
| dev s7 | 88.50 | 87.83 | +0.67 (7/3) | 88.50 |
| dev s11 | 88.67 | 88.67 | +0.00 (6/6) | 88.33 |
| dev s13 | 89.33 | 89.00 | +0.33 (5/3) | 88.50 |
| dev s17 | 89.67 | 88.67 | +1.00 (12/6) | 88.67 |
| **dev mean (4 seeds)** | **89.04** | 88.54 | **+0.50** (30/18), CI [−0.17, +1.17] | 88.50 |

CI = paired item-bootstrap 95% over the 4-seed mean. dev2 streams (and dev s17) are token-identical to the base rollouts, so
those deltas reproduce the offline numbers exactly (dev2 4-seed offline +0.58). dev waves > 0 of seeds 7,11,13 diverge from
gen/out (global sampler counter, as in voter2): fresh paired streams. Per-seed spread (−0.67…+2.00) ≈ the effect size.

## Optional: step-error detector line scores (linefix, dfeat.py → dfeat.log)
Per attempt max D-score (dmax) and count above the 2% threshold 1.821 (dcnt). AUROC (higher → right): dmax pooled .204 / .169,
within-item .307 / .372 (dev / dev2); dcnt .355 / .365, within .409 / .447 — a strong per-attempt error signal, but no trainC
scores exist, so ClusterLin was fit by item-grouped 5-fold CV on dev and fit-on-dev → tested on dev2 (seed streams 7,11,13,17):
| voter (dev-fit) | dev CV d | dev2 d |
|---|---|---|
| lg+sane | +0.46 | −0.75 |
| lg+sane+dmax | +0.50 | −0.58 |
| lg+sane+nnum+z_ntok+qcov_d | +0.88 | −0.17 |
| … + dmax | +0.71 | −0.08 |
| … + dcnt | +0.71 | −0.08 |
The detector adds ≤ +0.17 on dev2 and costs on dev CV: it is redundant with lg at vote time (like fork density). Note also the
dev-fit of the gate model transfers to dev2 at −0.17, whereas the trainC fit gave +0.58 — the cluster-voter gain is fragile
to the fitting set.

## Conclusion
* Question-number coverage (the targeted 'dropped' signal) is nearly useless: wrong attractors cite the same question numbers.
  Close-fork density is real (right 5.2 vs wrong 6.8 per 100 tok) but redundant with lg; all fork-weighted voters lose.
* The only gain comes from changing the voter to a per-answer-cluster logit with nnum + z_ntok (+ qcov): offline +0.71 / +0.58
  (dev / dev2, 4 seeds), passing the gate marginally; **in-stream same-stream, 4 seeds: +0.50 dev / +0.58 dev2** (fixes/breaks
  30/18, 20/13) — confirmed at the gate level, but each 95% CI includes 0.
* Implant is exact (registers match offline, all-off unchanged, zero token cost). Verdict: small positive mean gain
  (~+0.5 pt on both sets over 4 seeds, 6/8 seed-set cells ≥ 0), not significant on its own; reasonable to keep as an optional
  voter (SER_VF=1). Not a fix for the wrong-attractor problem (qcov/fork signals are redundant or absent).
