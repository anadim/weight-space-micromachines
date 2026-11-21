# selcheck: in-stream verification writer for contested votes (todo sel-check)

Goal: close the "some attempt right, vote wrong" gap of the all-weights stack (rung 2: dev 89.58 / dev2 86.92) with a
check segment per contested answer group, scored by a compiled reader and combined by a fixed linear voter (fit on trainC only).

## Progress log (newest last)
- 04:12Z started. Read BRIEF/task/CONTEXT/atlas_aw/voter2/vfeat/adapt/aw/probe/LADDER. Rung-2 dev2 rows exist now
  (aw/out/r2f_dev2: 87.00/87.00/87.00/86.67 = 86.92). Sandbox has no numpy/torch/pip: local analysis is pure python, model
  work and fits run on the B200 node (job dir implants/selcheck/job, env from probe/run.sh).
- 04:25Z step 1 code: `cands.py` (candidate table job/cdata/cands_{set}.jsonl + stream table data/streams_{set}.jsonl.gz; python
  base vote reproduces the rung-2 rows 2400/2400 dev, 1200/1200 dev2). Candidates: trainC 633 mixed items / 1908 cands (right +
  top-3 wrong of 16 S3 attempts), dev 282 contested items / 1132 cands, dev2 145 / 599 (union over the 4 rung-2 seeds, top 6).
  `job/fz.py` (HF host + ckpt_a2s2, no calc implant, 8-shot prefix, attempts invisible): formats v2 (yes/no verdict), v3 (4 verdict
  shots + verdict), dir (no-CoT logp of x), fwd ("The answer is x. Let's verify it step by step." sampled check, K=4, then verdict
  suffix "\nVerdict: the answer x is" correct-vs-incorrect, + agree = check's '#### y' == x), bwd ("Check: if the answer is x, then"
  + same). Hidden states (layers 10,14,18,22,26,28) at each scoring token -> $WSM_DATA/implants/selcheck/hs_{set}.pt.
  `job/fit.py` (trainC-only probes, grouped 5-fold OOF), `ana.py` (within-item AUROC + vote sim fit on trainC).
  NOTE gq never ships `data/` dirs -> job inputs live in job/cdata. CPU smoke j23355 OK (7.7 min, 12 cands).
  GPU jobs (blocked in gsub at 04:25Z): `cd job && <queue-submit> $PWD 2400 job_sc_fzdev -- bash -c "bash run.sh python3 fz.py --sets dev2 --limit 3 --K 1 --tag _smoke && bash run.sh python3 fz.py --sets dev,dev2"`
  and `... 2400 job_sc_fztrc -- bash -c "bash run.sh python3 fz.py --sets trainC"`.
- 05:20Z GPU j23364 (`job_sc_fzdev`, fz.py dev+dev2, K=4, 38 min of the 40 min budget) done -> job/out/fz_{dev,dev2}.jsonl.gz,
  hs on node $WSM_DATA/implants/selcheck/hs_{dev,dev2}.pt. trainC is too big for one 40-min job: fz.py got `--part i/n`;
  trainC is queued as 2 parts via gsub (`job_sc_fztrc{0,1}`: `bash run.sh python3 fz.py --sets trainC --part $i/2 --tag _p$i`,
  fit.py/ana.py concatenate p0,p1). `python3 ana.py` within-item AUROC (all right-vs-wrong pairs of contested items):
  | score | dev | dev2 |  (lg_mean .806/.763, lg_max .897/.871, count .898/.882)
  v2 .490/.532, v3 .504/.465, dir_mean .634/.615, fwd_v@1 .683/.683 @K .754/.726, fwd_agree@1 .688/.683 @K .778/.786,
  fwd_va@K .771/.763, bwd_v@K .687/.588, bwd_agree@K .692/.608.
  BUT on the pairs that matter (gap streams: some slot right, vote wrong; right group vs vote winner, same seed's sample):
  `python3 gap.py` P(right>winner): dev (119 streams) dir_mean .636, v2 .491, fwd_v .455, fwd_agree .459 (51/110 tie), fwd_va .445,
  bwd_v .473, bwd_va .482, lg_max .403; dev2 (94) dir_mean .511, v2 .641, fwd_v .370, fwd_agree .424, fwd_va .370, bwd_v .467,
  bwd_va .500, lg_max .436. i.e. the checks are at chance or WORSE than chance exactly on the contested hard pairs (the
  conditioned re-solve inherits the model's belief in the majority wrong answer); the 0.75 AUROC comes from easy pairs (right vs rare
  wrong) the vote already gets right.
  In-sample UPPER BOUND of the vote sim (beta/tau/K picked on the eval set itself; `python3 preview.py KEYS`), 1 check/candidate:
  v2 +0.04/+0.00, dir_mean +0.04/+0.00, fwd_v +0.00/+0.17, fwd_agree +0.17/+0.17, fwd_va +0.04/+0.25, bwd_v 0/0, bwd_va +0.04/0.
  With all 4 check samples averaged (4 checks per candidate; gap.py): fwd_agree +0.25/+0.58, fwd_va +0.08/+0.50, fwd_v +0.04/+0.25.
  -> far below the +1/+1 gate even in-sample. Probe (d) upper bound: CPU j23381 `bash run.sh python3 xfit.py` (item-grouped
  5-fold OOF probes on dev/dev2 hs separately -> out/fzo_{set}.json).
- 05:45Z probe (d) upper bound CPU j23381 done (`job/xfit.py` -> job/out/fzo_{dev,dev2}.json; probes fit IN-SAMPLE on the eval set
  with item-grouped 5-fold OOF, l2 picked on the set itself = optimistic). OOF within-item AUROC (all pairs) Kmean/1-sample:
  fwd_v L18 dev .833/.760 dev2 .792/.732 (best), fwd_c .749/.742, bwd_v .743/.663, v2 c4 .663/.616, dir c4 .668/.543.
  Gap pairs P(right>winner) (`python3 gap.py o_...`): dev/dev2 o_v2_c4 .582/.587, o_dir_c4 .618/.446, o_fwd_c_L18 .473/.478,
  o_fwd_v_L18 .400/.391, o_fwd_v_c4 .455/.467, o_bwd_v_L18 .455/.457 (lg_max .403/.436).
  In-sample vote-sim bound (`python3 preview.py o_...`, 1 check/candidate): o_fwd_v_L18 +0.04/+0.42, o_fwd_c_L18 +0.08/+0.33,
  o_fwd_v_c4 +0.04/+0.33, o_v2_c4 +0.12/0, o_dir_c4 +0.12/0, o_bwd_v_L18 +0.08/0. 4 checks/candidate averaged: o_fwd_v_L18 +0.25/+0.42,
  o_fwd_v_c4 +0.21/+0.25, o_fwd_c_L18 0/+0.92.
  DECISION: gate (>= +1 pt on BOTH dev and dev2 with a trainC fit) FAILS even for in-sample-tuned bounds (best dev +0.25, best dev2
  +0.92, never both; a trainC fit of beta/tau/K is dominated by the in-sample grid max). Step 2 (in-stream build) NOT started.
  trainC fz jobs cancelled before submission (no GPU spent; they could only lower the numbers). Clean negative.
  Why: on the gap streams the vote winner out-counts the right answer (W never favours it, lg_max .40/.44) and every check format
  is at chance or below on exactly those pairs: a candidate-conditioned re-solve / verdict by the same 0.6B policy reproduces its
  majority belief (the adaptive-N extra attempts are already "re-solves" and the gap persists). The ~0.75-0.83 all-pair AUROC is
  carried by easy pairs (right vs rare wrong) that the vote already resolves.
  Caveat: the offline host is HF Qwen3-0.6B-Base + ckpt_a2s2 without the calc implant (stack checks would have exact arithmetic);
  this can only move the arithmetic-error share of check disagreements, while the gap is reasoning-dominated (attempts with calc
  already vote the wrong majority). Files: cands.py, ana.py, preview.py, gap.py, job/{fz,fit,xfit}.py; node hs at
  $WSM_DATA/implants/selcheck/hs_{dev,dev2}.pt. Resume (if ever): trainC parts via fz.py --part, then fit.py, ana.py --sim.
