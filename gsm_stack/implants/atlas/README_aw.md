# atlas-aw: failure atlas of the ALL-WEIGHTS stack (rung 1 = SerialWD + WPR; rung 2 = + A+B+C+stop4)
Owner: sub-agent of the implant lane, todo `atlas-aw`. Notes -> gsm_stack/results/notes/atlas_aw.md. Code: atlas/*_aw.py. Data: atlas/data_aw/.
## progress log (newest last)
- 03:10Z started. Rung-1 rows in implants/wpr/fe/out/fe/{dev,dev2}_wpr_s{7,11,13,17}.jsonl are fe format: fields
  i, gold, ok, last, ntok, tokens, natt, nseg, win, sym, rok, bad, lg -- NO wf (queue flags), logp, end, val, q.
  atlas/parse.py needs wf (segmentation) and diverge.py needs logp (branch points) -> regenerate rung 1 in gen format.
  aw/stk/comp_serial.py has the WPR gate (line 49 `WPR = os.environ.get('WPR')=='1'`, merged verbatim from lane wpr; aw_hook tags `_wpr`).
- 03:11Z SUBMITTED (via gsub, may block on the GPU cap): `cd gsm_stack/implants/aw/dev_tree && <queue-submit> gsm_stack/implants/aw/dev_tree 2400 job_atl_r1 -- bash aw.sh r1atl AW_SETS=dev,dev2 AW_SEEDS=7,11,13,17 WPR=1`
  -> aw/out/r1atl/{set}_N8_R1_v1_s1_m0_pr0_wpr_T1.0_seed{S}.jsonl.gz (name job_atl_r1, not job_aw_*, to keep the ladder cap free).
- 03:13Z j23311 (job_atl_r1) started. 03:15Z ORCHESTRATOR NOTE: WPR=1 with several seeds in ONE process leaks state (later seeds
  collapse). => in aw/out/r1atl only seed 7 is valid; aw/out/r2_* seeds 11/13/17 invalid (seed 7 ok). The fe-tree rung-1 rows
  (implants/wpr/fe/out/fe/*_wpr_s*.jsonl) were one process per seed -> valid.
- 03:16Z SUBMITTED (gsub, blocks on cap) job_atl_r1s: `cd gsm_stack/implants/aw/dev_tree && <queue-submit> gsm_stack/implants/aw/dev_tree 2700 job_atl_r1s -- bash -c 'for s in 11 13 17; do bash aw.sh r1atl_s$s AW_SETS=dev,dev2 AW_SEEDS=$s WPR=1; done'`
  -> aw/out/r1atl_s{11,13,17}/. Use r1atl seed7 + these for the gen-format rung-1 atlas (data_aw/r1/).
- 03:15Z FINDING (fe rows, atlas/fe_quick_aw.py, fe_pos_aw.py): WPR does NOT fully flatten the per-attempt decay:
  dev k0..6 80.7 81.0 79.2 79.5 77.7 77.5 74.5 (k7 inflated by the vote line in this quick parser); pr1 (programmatic reset) flat ~81.
  Cause: attempts with no '####' grow with stream position (garbage/degenerate text at high offsets): attempts starting at
  stream index 1000-1500: no-answer 22% (dev) / 34% (dev2) for WPR vs 2.5% / 1.1% for pr1. WPR != pr1 at large offsets.
- rung-2 parse (aw/out/r2_*, copied to data_aw/r2/, parse.py): seed 7 only valid: dev 88.67 oracle@8 93.67, dev2 85.67 / 94.33.
- 03:17Z cancelled j23311 / j23314 (multi-seed / multi-set processes leak state with WPR=1). Kept aw/out/r1atl/dev_*seed7 (first
  generate call of its process: acc 88.50 = fe-tree WPR s7 88.50). SUBMITTED j23317 (job_atl_r1p), one process per (set, seed):
  `cd gsm_stack/implants/aw/dev_tree && <queue-submit> gsm_stack/implants/aw/dev_tree 2700 job_atl_r1p -- bash -c 'for x in dev:11 dev:13 dev:17 dev2:7 dev2:11 dev2:13 dev2:17; do bash aw.sh r1atl_${x%:*}_s${x#*:} AW_SETS=${x%:*} AW_SEEDS=${x#*:} WPR=1; done'`
  -> aw/out/r1atl_{set}_s{seed}/. Check each `AW done` acc vs fe-tree WPR per seed (dev 88.50/88.00/88.50/86.83, dev2 86.00/86.33/85.33/86.00).
  NEXT: copy all 8 gz into atlas/data_aw/r1/ and run `python3 atlas/parse.py --gen atlas/data_aw/r1 --out atlas/data_aw/r1 --notes atlas/data_aw/r1/counts.md`.
- 03:31Z j23317 DONE (fixed WPR code: aw/stk/comp_serial.py `_wpr_ar` .view(1,-1) fix by ladder-r2 was in its snapshot; its
  logs have no `_wpr_ar` in "fastroll compaction moves"). Per (set,seed) acc: dev 11/13/17 = 88.83/88.50/87.50; dev2 7/11/13/17 =
  86.33/86.33/85.67/86.00. FINDING: the clean rows have NO per-k decay (dev 80.8..79.0, no-answer 0.7..3.8% vs fe tree 80.7..74.5,
  none 19->231/2400). The "residual WPR decay" seen in the fe tree (and in aw/out/r1atl dev s7 from j23311, which HAS `_wpr_ar` in
  the moves list) is the `_wpr_ar` compaction bug, not a model effect. Rung-1 fe numbers 87.96/85.92 are slightly understated.
- Hand labels: atlas/hand_sample_aw.py on atlas/data_aw/r1s7 (old j23311 dev s7 rows, answered wrong attempts only) ->
  atlas/data_aw/hand_dump_aw.txt; 57 labels in atlas/data_aw/hand_labels_aw.tsv (gid, cat, note). Heuristic exact agreement 26/57.
- fe-tree evidence saved: atlas/data_aw/fe/{pos,perk,perk_none}.txt (fe_pos_aw.py / fe_quick_aw.py / fe_quick2_aw.py).
- SUBMITTED job_atl_r1s7 (fixed code, dev seed 7 alone): `cd gsm_stack/implants/aw/dev_tree && <queue-submit> gsm_stack/implants/aw/dev_tree 1200 job_atl_r1s7 --
  bash aw.sh r1atl_dev_s7 AW_SETS=dev AW_SEEDS=7 WPR=1` -> aw/out/r1atl_dev_s7. Then data_aw/r1 = r1atl_{dev,dev2}_s* only (drop r1atl/).
- 03:40Z job_atl_r1s7 never got a GPU slot (gsub cap); stopped the waiting gsub (nothing submitted). Used instead
  aw/out/r1f_dev seed 7 (j23319 = ladder-r2's aw_job3.sh: fixed code; log has no `_wpr_ar` in moves). LEAKCHECK 2/600 bitwise diffs.
- 03:45Z RUNG-1 ATLAS (fixed): `bash atlas/run_r1_aw.sh` (copies r1atl_{set}_s* + r1f_dev s7 -> data_aw/r1; parse, analysis_aw,
  diverge_aw, lists_aw --hand) then `python3 atlas/ceiling_aw.py atlas/data_aw/r1 > atlas/data_aw/r1/ceiling.md`,
  `python3 atlas/never_aw.py atlas/data_aw/r1 > atlas/data_aw/r1/never.md`. dev 88.38 / dev2 86.08, oracle@8 93.17/93.08,
  gap 4.79/7.00, none-right 6.83/6.92, per-k flat. Lists -> atlas/data_aw/m{12,3,4,5}_*_aw.json (253/108/20/154 items).
- 03:50Z RUNG-2 dev (fixed, aw/out/r2f_dev from j23319): data_aw/r2f; 89.58 (89.67/90.17/89.17/89.33), oracle 94.54, oracle@8 93.75,
  per-k no-answer <=1.6%, within-stream per-k flat. diverge StopIteration on 18 items (skipped). Old aw/out/r2_* (and data_aw/r2)
  are bug-affected: do not use. Rung-2 dev2 = j23340 (job_aw_r2fdev2, queued at 03:55Z) -> when aw/out/r2f_dev2 exists:
  `bash atlas/run_r2_aw.sh` and update notes/atlas_aw.md §2.
- 03:56Z notes/atlas_aw.md written (headline, tables, categories vs S3, hand labels, gap, lists, ranked implants with ceilings).
