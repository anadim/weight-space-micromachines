# lane prmstep: step-level (PRM-style) verifier on line-end residuals vs the rung-4 sn+lg blend (todo prm-step)

Question: does a verifier trained on a DIFFERENT supervision signal (step correctness, from GSM8K train gold annotations) add
within-item right/wrong information beyond rung 4 (side-net sn + L18 logit lg, BLD writer)? Prior verifiers (sidenet, sndata,
attnver) saturate at within-item AUROC ~0.75 dev / ~0.67 dev2 and are mutually redundant.
Pre-registered gate for an in-stack build: offline re-pick >= +0.3 on BOTH dev and dev2 vs rung 4 with pooled fixes > breaks
(selection on trainC only).

## design
- Data: rung-3/4 attempt tables (implants/sidenet/tables = job/tables; rung-4 generation is token-identical to rung 3 before the
  final writer line). TRAIN = trainC (GSM8K train split C, 1499 items x 4 seeds, 47.5k attempts). dev/dev2 held out (their line
  labels are only used for a line-level diagnostic). Never test.
- Step label per model-written line (line ends as in sidenet extract_sn.py): result = first number after the LAST '=' on the line;
  good (1) if it matches a gold intermediate value (<<expr=c>> annotations of the GSM8K train solution) or the gold final answer
  (rel tol 1e-3 or equal at 2 dp), else bad (0); lines without '=' + number skipped (-1). job/mk_gold.py -> job/tables/goldvals.json.
- Features per line: residual (layer input) at the line-end token, layers 14/18/22/27 (sidenet c4 layers; = the per-line version
  of sn's M features), teacher-forced in the in-stack context (pre + q_ids + seg_tokens) -> job/extract_pl.py.
- Probe: logistic regression on standardized 4096-d per-line features, trainC labeled lines, 5-fold item CV (idx % 5): trainC OOF,
  dev/dev2 fold-mean; l2 x steps chosen on trainC OOF line AUROC -> job/train_pl.py.
- Attempt score = min or mean of per-line logits over labeled-eligible lines (choice on trainC OOF within-item AUROC); analysis +
  re-pick locally (off/ana.py; re-pick through attnver/repick_av.py BLD writer with delta = w * z(prm), w fit on trainC).

## log (UTC, newest last)
- 12:09Z job/ = copy of attnver/job (common.py, run.sh, wsmrepo, tables incl. sc_g4.npz) + mk_gold.py, extract_pl.py, train_pl.py, pl_job.sh.
  `cd job && <queue-submit> $PWD 2400 job_prm_xt -- bash run.sh bash pl_job.sh all` -> **j23522** (extract trainC,dev,dev2 per-line
  features to $WSM_DATA/implants/prmstep/{set}/{X.npy,lines.npz,meta.npz}, then train_pl.py -> job/out/{sc_prm.npz,train_pl.json}).
- 12:25Z j23522 DONE (12.1 min run, 4 min queue). Extraction (L18 probe at '####' vs register lg: corr 0.9982-0.9983, MAE 0.082 on all
  sets = in-stack context reproduced). Features $WSM_DATA/implants/prmstep/{trainC,dev,dev2}/X.npy (4.0 GB fp16, may be deleted).

  | set | attempts | line ends | labeled lines (have '= number') | good | good frac in right attempts | in wrong attempts |
  |---|---|---|---|---|---|---|
  | trainC | 47525 | 302954 | 178822 | 112072 | 0.795 | 0.378 |
  | dev | 18477 | 119213 | 71826 | 44392 | 0.788 | 0.346 |
  | dev2 | 9804 | 64964 | 39111 | 23845 | 0.786 | 0.364 |

  Per-line probe, line-level AUROC on labeled lines (trainC OOF / dev / dev2, fold-mean): best on trainC = l2 0.1, 100 Adam steps:
  **0.781 / 0.781 / 0.746** (grid l2 1e-4..0.1 x steps 50..800: 0.750-0.781 trainC; weaker l2 overfits with steps). job/out/train_pl.json.
- 12:27Z offline `python3.11 off/ana.py` (off/ana.log, off/ana.json; writes off/sc_inc.npz) and the gold-label ceiling `python3.11 off/oracle.py`
  (off/oracle.log); re-pick `cd ../attnver && python3.11 repick_av.py --sc ../prmstep/off/sc_inc.npz --l2 1e-3` (off/repick_v.log;
  controls: delta 0 reproduces rung-4 picks 0/2400, 0/1200).

## results (committed attempts; within-item AUROC = right vs wrong attempts of the same item, all 4 seeds, pair-weighted)
Baseline b = rung-4 slot score max(sn,-12)+max(lg,-12): trainC 0.6975 / dev 0.7525 / dev2 0.6726 (= attnver numbers).

| attempt score (probe 0.1@100) | within AUROC trainC / dev / dev2 | within-item corr with b (trainC / dev / dev2) | best v on trainC | AUROC of b + v*prm |
|---|---|---|---|---|
| min of line logits | 0.539 / 0.587 / 0.496 | 0.28 / 0.28 / 0.28 | 0 (every v>0 lowers trainC: −0.0004 @0.05 … −0.068 @3) | = b (+0.000) |
| mean of line logits | 0.567 / 0.586 / 0.522 | 0.37 / 0.39 / 0.36 | 0 (−0.0001 @0.05 …) | = b |
| last labeled line | 0.620 / 0.654 / 0.592 | 0.71 / 0.71 / 0.67 | 0 (−0.0002 @0.05 …) | = b |
| (ceiling, NOT deployable) TRUE gold step labels, fraction good | 0.899 / 0.890 / 0.886 | 0.20 / 0.25 / 0.19 | | b+frac 0.905 / 0.908 / 0.885 |
| (ceiling) TRUE labels, min | 0.708 / 0.678 / 0.684 | | | |

Agg choice on trainC OOF: 'last' > 'mean' > 'min' alone, but none gives a positive increment over b on trainC at any v -> the
pre-registered selection gives v = 0 (= rung 4 exactly; re-pick Δ 0.00 / 0.00, 0/0).

Sensitivity (fixed v, NOT trainC-selected), re-pick through the rung-4 BLD writer, Δ vs rung 4 (dev 90.46, dev2 88.33), fixes/breaks summed over seeds 7/11/13/17:

| delta | X0 (rung-4 theta) dev / dev2 / pooled | BLD refit: trainC CV | dev | dev2 | pooled |
|---|---|---|---|---|---|
| last, v 0.3 | −0.08 (0/2) / −0.17 (1/3) / 1/5 | −0.15 (6/15) | −0.17 (0/4) | +0.08 (2/1) | 2/5 |
| last, v 1 | −0.21 (2/7) / −0.17 (4/6) / 6/13 | −0.27 (11/27) | −0.21 (2/7) | +0.00 (3/3) | 5/10 |
| mean, v 0.3 | −0.04 (1/2) / −0.08 (1/2) / 2/4 | −0.07 (8/12) | −0.12 (1/4) | −0.08 (1/2) | 2/6 |
| mean, v 1 | −0.33 (5/13) / −0.50 (2/8) / 7/21 | −0.13 (17/25) | −0.29 (3/10) | −0.50 (2/8) | 5/18 |
| min, v 0.3 | +0.00 (2/2) / −0.33 (2/6) / 4/8 | −0.18 (9/20) | −0.04 (2/3) | −0.08 (2/3) | 4/6 |
| min, v 1 | −0.04 (10/11) / −1.00 (4/16) / 14/27 | −0.33 (15/35) | +0.00 (7/7) | −0.58 (4/11) | 11/18 |

## VERDICT (12:28Z): NEGATIVE — a step-level (PRM-style) probe on line-end residuals adds nothing over rung-4 sn+lg; no in-stack build
- Gate (re-pick >= +0.3 on dev AND dev2, pooled fixes > breaks, trainC-selected): FAILED. trainC selection picks v = 0 (any PRM weight
  lowers trainC within-AUROC); every fixed-v variant is ≤ 0 on dev, ≤ +0.08 on dev2, pooled fixes < breaks (1/5 … 14/27).
- The step supervision itself is very informative (gold step labels: within-item AUROC 0.89 and only 0.2 corr with b), but the
  line-end residual does NOT linearly encode step correctness WITHIN an item: the probe's line AUROC 0.78 is mostly between-item
  (lines of easy items are good), and its attempt scores reach only 0.54-0.65 within item; the best of them ('last line', ~ the
  answer line) is 0.71-correlated with b, i.e. it re-reads the same "settled/confident" signal sn/lg already use.
- Consistent with sidenet/sndata/attnver: different supervision (step labels instead of final-answer labels) on the same residual
  stream reaches the same within-item ceiling; the information that separates right from wrong attempts of one item is not linearly
  present at line ends of this host.
- Code/data kept: job/{mk_gold.py, extract_pl.py, train_pl.py, pl_job.sh}, job/out/{extract_*.json, train_pl.json, sc_prm.npz},
  off/{ana.py, oracle.py, ana.log, ana.json, oracle.log, repick_v.log, sc_inc.npz}. Fully done (all 4 seeds, dev + dev2).
