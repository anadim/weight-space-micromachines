# [28] BBH on Qwen3-1.7B-Base: running log (UTC), agent rl24-bbh

> Release note: this is the original run log, scrubbed (cluster/queue details removed; job ids kept as labels only; "GPU k"
> = a GPU of the one B200 node). Path map to this release: `$ROOT` -> `$WSM_DATA` (`$ROOT/rl28` = `$RL28_DATA`,
> `$ROOT/rl24` = `$RL24_DATA`), `w/veto_q17.pt` -> `results/gsm_bigger/veto_q17.pt`, `w/out/*` -> reduced rows in `res/`
> (+ `table_final.md`, `rows.json`), `rl24/bbh/...` -> `uniform/rl24/bbh/...`, `final_eval/q32` -> `uniform/final_eval/q32`.
> Not shipped: `peek/peek.sh` (cluster-only), checkpoints, ft17, the probe/states files, logs. RL_RESULTS §12 -> REPRODUCE §18.

Task (23:31Z): the [26]/[29] BBH 4-row protocol of rl24/bbh (0.6B, RESULTS.md there), repeated on Qwen3-1.7B-Base.
- Host: 1.7B-Base + the same call fine-tune recipe (ft_int_v2 recipe = teamA/ft.py on train_union_v2, lr 2e-5, 6 ep,
  tok_budget 8000, seed 0), retrained on 1.7B -> $ROOT/rl28/bbh/ft17 ("ft17"). GPU-h noted below.
- Rows: 0 plain 1.7B-Base; 1 ft17 tools off; 2 ft17 + RL (F17 x3 seeds); 2' 1.7B-Base + RL (P17 x3); 3 ft17 + 8 tools + refit gate;
  4 ft17 + tools + gate + RL (I17 x3). Controls: ft17 + tools (no gate) at u0; F17 u300 + tools + gate at eval.
- Implants unchanged (integ.TOOLS_V2, same compiled circuits). Host-fitted parts, all solved at attach time on the host
  (micromachines Attached: readers from the host embedding, write directions on the host's final norm + LM head), so they
  refit themselves on 1.7B. Calc veto: results/gsm_bigger/veto_q17.pt (1.7B-Base, same solve_veto.py; same choice as
  0.6B, where the Base veto served the FT host). Call gate: refit with the original callgate recipe (cg/job_refit.sh) on ft17 -> probe_ft17.pt.
- RL: the same recipe (Dr.GRPO, 64x8, 300 u, lr 1e-6, T 1, clip 0.2, max_new 600, same train/dev data = rl24/bbh/data, reused).
- Data reused unchanged: $ROOT/rl24/bbh/data/{train,dev}.jsonl (sha in rl24/bbh/NOTES).

## 05:58Z [28] BBH 1.7B FINAL: RESULTS.md, rows.json, table_final.md written
- I17b j23908 done 05:48Z (rc 0, SM 92.7%): I17_s0 resumed u283 -> DONE (dev u300 98.77), I17_s1 resumed u287 -> DONE (98.77);
  evals I17_s0 68.29, I17_s1 68.25. I17c j23914 done 05:47Z (8 min): I17_s2 (DONE in E1, dev u300 98.04) eval 68.34.
  Both drivers exited ("all evals present"). E1 j23875 done 05:39Z rc 0 (SM 95.5%).
- **j23917 final_ana.sh** -> w/out/table_final.md + rows.json; copied to rl28/bbh/table_final.md, rl28/bbh/rows.json (notes = 1.7B).
- [32](e) `python3 w/pertask.py`: 2 tasks with 4 − 1 < −3, both non-tool: geometric_shapes −6.1 [−9.5, −3.1] (0 writes/latches/calls
  in all 3 I17 evals -> RL weight drift of I17, not the tools), formal_fallacies −3.7 [−8.0, +0.5] (chance-level binary, CI ∋ 0).
- No running/queued rl28bbh jobs. All resume states gone (DONE); disk 808 GB free; rl28/bbh on disk = 36 GB
  (9 × u300.pt, ft17, cg/{probe_ft17*.pt, states2-4.pt (gate-refit scratch, small; deletable, left for the parent per [30](c))}).

### Ready-to-paste for RL_RESULTS §12 (BBH, Qwen3-1.7B-Base) — source rl28/bbh/RESULTS.md, table_final.md, rows.json
Host for rows 1–4 = ft17 = Qwen3-1.7B-Base + the §6.1 call-FT (same recipe as ft_int_v2; a whole-host weight change shared by
all rows 1–4). Same 8 compiled tools as 0.6B (readers/writers solved at attach), call gate refit on ft17 (AUROC 1.000; Python ==
compiled gate, 0/61,774 mismatches). RL: GRPO 300 u, 64×8, lr 1e-6, T 1, 3 seeds. BBH-27 test (6,511 items, greedy, max_new 600),
paired item bootstrap CIs.
| row | config | BBH-27 mean [95% CI] | 12 tool tasks [95% CI] | 15 other | tok/q | seeds (mean27) | 0.6B |
|---|---|---|---|---|---|---|---|
| 0 | plain 1.7B-Base | 51.74 [50.59, 52.85] | 55.37 [53.73, 56.97] | 48.84 | 206 | – | 41.19 |
| 1 | host (ft17), tools off | 42.26 [41.16, 43.37] | 41.90 [40.40, 43.43] | 42.54 | 167 | – | 32.59 |
| 2 | host + RL (F17) | 51.58 [50.60, 52.54] | 60.88 [59.66, 62.13] | 44.14 | 152 | 51.21, 51.84, 51.68 | 42.27 |
| 3 | host + 8 tools + gate | 66.33 [65.39, 67.28] | 95.73 [95.03, 96.43] | 42.82 | 161 | – | 54.99 |
| 4 | host + tools + gate + RL (I17) | 68.29 [67.48, 69.12] | 99.29 [99.02, 99.53] | 43.49 | 158 | 68.29, 68.25, 68.34 | 60.24 |
| 2' | plain 1.7B-Base + RL (P17) | 66.95 [66.06, 67.84] | 86.17 [85.26, 87.07] | 51.58 | 195 | 67.20, 67.33, 66.33 | 53.63 |
| ctl | row-2 weights + tools + gate at eval | 67.08 [66.26, 67.92] | 95.82 [95.22, 96.40] | 44.09 | 152 | 66.72, 67.41, 67.11 | 57.62 |
| ctl | row-2 weights + tools, no gate | 66.99 [66.17, 67.82] | 95.92 [95.33, 96.49] | 43.85 | 153 | 66.49, 67.37, 67.12 | 57.37 |
| ctl | host + tools, no gate | 66.04 [65.11, 66.97] | 96.23 [95.57, 96.87] | 41.88 | 164 | – | 52.66 |
Effects (mean27 / tool12 / other15): 2−1 +9.32 [+8.32, +10.32] / +18.98 / +1.60; 3−1 +24.08 [+23.29, +24.86] / +53.83 / +0.27;
4−2 +16.71 [+15.96, +17.48] / +38.41 / −0.64; 4−3 +1.96 [+1.25, +2.65] / +3.56 / +0.68; 4−ctl(+tools+gate) +1.21 [+0.65, +1.79] / +3.47 / −0.60;
1−0 −9.49 [−10.89, −8.04]; 2'−0 +15.21 [+14.13, +16.32] / +30.80 / +2.74; 4−2' +1.34 [+0.41, +2.28] / +13.12 [+12.21, +14.03] / −8.09 [−9.59, −6.57].
Reading: same ordering as 0.6B (4 > 3 > 2 > 1; 4 > 2'). Row 3 is already at 95.7 tool12 on 1.7B, so RL on top adds less (+2.0 vs +5.3
on 0.6B). RL from plain Base (2') is a strong baseline on 1.7B (66.95); row 4 beats it by +1.3 overall (+13.1 tool12, −8.1 other15,
the latter inherited from the call-FT). [32](e): only 2 tasks have 4 − 1 < −3, both non-tool (geometric_shapes −6.1, implants idle
there = RL weight drift; formal_fallacies −3.7, CI includes 0). Jobs: j23812, j23836, j23813, j23847, j23853, j23848, j23875, j23886,
j23908, j23914; analysis j23917.

### §claims line
C28 | BBH, Qwen3-1.7B-Base: host + implants + RL (row 4) beats every other row; 4 = 68.29 [67.48, 69.12] mean27 (99.29 tool12),
4−3 +1.96 [+1.25, +2.65], 4−2 +16.71 [+15.96, +17.48], 4−2' (RL from plain Base) +1.34 [+0.41, +2.28], 4−ctl(attach after RL)
+1.21 [+0.65, +1.79]; 3−1 +24.08 [+23.29, +24.86]; 2−1 +9.32 [+8.32, +10.32]; call-FT 1−0 −9.49 [−10.89, −8.04] |
rl28/bbh/RESULTS.md, rl28/bbh/table_final.md, rl28/bbh/rows.json | j23812, j23836, j23813, j23847, j23853, j23848, j23875,
j23886, j23908, j23914, ana j23917

## 05:40Z (relaunch 3) I17 nearly done
- j23848 I17a ended 05:22Z rc=1 (time budget); driver I17b submitted **j23908 rl28bbh_I17b** (GPU 0): I17_s0/s1 resume -> u300 eval each.
  05:34Z: I17_s1 DONE (dev u300 98.77, max_mem 55.7 GB), eval running; I17_s0 at u299. SM 90%.
- j23875 E1 (GPU 3, SM 95.8%): I17_s2 at u282 (05:34Z, ~15 s/u alone) -> DONE ~05:40Z; eval workers done: F17_s0_u300 _t1 66.49,
  F17_s1_u300 _t1 67.37 (tools no gate; _t1g also in E1 out). When E1 ends, driver I17c (pid 598850) submits rl28bbh_I17c = eval I17_s2_u300.
- Drivers alive: pids 598857 (I17b), 598850 (I17c). disk 701 GB free. Next: wait for evals, final_ana.sh, pertask.py, RESULTS.md, rows.json.

## 04:35Z waiting on I17
- I17a j23848 (I17_s0/s1, ~u215 now) and E1 j23875 (I17_s2 + F17_s0/s1 _t1g/_t1 evals) running; drivers I17b/I17c alive
  (pids 598857/598850), waiting on them. Nothing else is pending for [28] (all non-I17 evals done except the F17_s0/s1 ctl evals in E1).

## 04:15Z E2 done; resume-save fix verified
- **j23886 E2 done** (27 min, rc 0, 97.2% SM): P17_s2_u300 66.33 (tool12 85.9, other15 50.68, 191 tok/q);
  F17_s2_u300 (tools off) 51.68; F17_s2 + tools + gate (_t1g, attach-after-RL ctl) 67.11; + tools no gate (_t1) 67.12.
  (queue message "1 process still running, TERM" = the dmon sampler; all 4 eval json complete, n=6511.)
- E1 eval workers: F17_s0_u300 51.21, F17_s1_u300 51.84 (tools off); _t1g/_t1 of s0/s1 running.
- **save fix verified** on I17_s2 (E1) saves.jsonl: u10 block 10.7 s (first-time pinned alloc), u20/30/40 block 0.37 s,
  background write 17-68 s overlapped with rollouts. E1 SM 97.7%, I17a 98.9%. disk 676 GB free.
- I17a u185 at 04:07Z (45 s/u) -> budget exit ~05:20Z at u~260-280; driver I17b takes over.

## 03:45Z S2a done; E2 evals; chain drivers for I17; [32](e) script ready
- F17a j23847 rc=1 (03:35Z note): NOT a crash of the RL - both F17_s0/s1 reached u300 + DONE (03:16Z); rc 1 = the
  OOM'd eval worker. No resume needed.
- **j23853 S2a done** (1h47m, rc 0, 87.7% SM): P17_s2 dev u300 78.4, F17_s2 dev u300 57.6 (max_mem 53.9 GB at cap 0.38);
  evals P17_s0/s1 u300 = 67.20 / 67.33 in w/out.
- **j23886 rl28bbh_E2** (03:37Z): 4 eval workers (B_max 32, cap 0.22): P17_s2_u300, F17_s2_u300, F17_s2_u300_t1g, F17_s2_u300_t1.
- I17a (j23848) at u~143 (03:35Z), ~45 s/update -> hits max_hours 3.7 at ~05:19Z near u~280 (blocking final save) -> chain.
  E1 (j23875): I17_s2 u8 at 03:35Z (~40 s/u while 2 eval workers share the GPU), SM 88%.
- **Chain drivers** (detached on the submit host; release version runs the packs locally): `w/drive.sh I17b I17_s0 I17_s1` and
  `w/drive.sh I17c I17_s2` (after j23848 / j23875 ended): run rl28bbh_I17b / _I17c = pack (cap 0.46) of
  "job.sh rl RUN (resume); job.sh eval RUN_u300 (ft17 + tools + gate)" per unfinished run, until out/eval_RUN_u300.json exists.
  Logs w/out/drive_I17b.log, drive_I17c.log. job.sh rl is now a no-op when runs/RUN/DONE exists (no accidental restart).
- [32](e): w/pertask.py = final_eval/q32/bbh_pertask.py with the 1.7B configs (rows 0,1,2=F17,3,4=I17); run locally when I17 evals exist.
- When all evals exist: `cd w && bash final_ana.sh` (CPU job -> out/table_final.md + out/rows.json), `python3 pertask.py`, then RESULTS.md.

## 03:25Z F17_s0/s1 done; F17a stall diagnosed + fixed ([31](c), 02:45Z); E1 submitted
- **j23847 F17a** ended 03:16Z (1h42m; rc 1 only because the eval worker OOM'd at start). Dev u0 39.2 -> u300 **60.8 / 58.1**
  (0.6B F0: 38.2 -> 51.5/56.1/57.6). u300.pt in runs/F17_s{0,1}.
- **Stall diagnosis** (w0/w1.log timestamps vs per-update total_s, dmon): mean SM 67.4%. Per update rollout 7.8 s + train 2.4 s
  = 10.3 s, dev 3 min per run in total, but wall 6,064 s vs sum(total_s) 3,063 s: after every 10th update the loop stalled
  **~97 s** (29 stalls, ~48% of the wall time; other updates +0.4 s). Cause: save_resume() (20.6 GB fp32 params + AdamW) did a
  synchronous pageable D2H copy and then JOINED the previous background torch.save, which takes > 10 updates' time on disk when
  2 processes write at once -> the GPU idles (dmon: runs of 0% for ~100 s). The idle eval worker (OOM) made it worse.
- **Fix** (w/rl_bbh.py, I/O only, bitwise-neutral for the update math and RNG): pinned reusable host buffers (non_blocking D2H +
  one sync), background write (.tmp + rename) and **skip** the save if the previous write is still in flight (logged); the
  time-budget exit always does a blocking save of the last update (`_last_saved`). Per-save timings -> out/RUN/saves.jsonl
  (block_s, write_s, skipped). resume_every stays 10. Applies to jobs submitted from now on (E1 onwards); S2a (92% SM, 2 procs)
  and I17a (47 s/update, saves every ~8 min, 95.6% SM) are left running.
- **j23853 S2a** P17 u300 evals done in-pack (worker 3, B_max 32 cap 0.20 worked): **P17_s0 67.20 / P17_s1 67.33** mean27
  (tool12 86.6 / 86.0, other15 51.7 / 52.4). P17_s2 at u~150, F17_s2 at u~200 (03:16Z).
- **j23875 rl28bbh_E1** (03:22Z): I17_s2 (row 4 seed 2; B_max 128 to fill the GPU alone once evals end - slots only, same
  update math; cap 0.52) + 2 eval workers (B_max 32, cap 0.22): F17_s{0,1}_u300 (row 2), _t1g (ctl + tools + gate), _t1 (ctl + tools).
- disk 570 GB free 03:20Z.

## 01:50Z F17a eval worker OOM; S2a replaced (j23853); peek tool
- j23847 F17a: worker 3 (P17 u300 evals, cap 0.18 = 32 GB, B_max 64) OOM'd at start (an eval needs > 32 GB at B_max 64). The 2 RL
  workers are fine: F17_s0 u17 / F17_s1 u14 at 01:39Z, ~16 s/update each, ratio 1.0000 -> done ~03:00Z.
- I17a (j23848): DEV u0 89.95 (= idle check's tools+gate acc), fire 99% latch 1%.
- **j23853 rl28bbh_S2a** (replaces queued j23850, same place): P17_s2 + F17_s2 (cap 0.38 each; P17 needs 63.6 GB) +
  worker 3 = evals P17_s0_u300, P17_s1_u300 at B_max 32, cap 0.20.
- Peek at running jobs: a CPU job running peek/peek.sh on the node (cluster-only; not in the release).
- disk 408 GB free 01:39Z; 6 resume states (20.6 GB each) when all arms run -> ~300 GB free expected (writes skipped < 250).

## 01:45Z j23836 setup2 done (52m41s, rc 0, mean SM 97.4%); rows 0, 1, 3 + ctl; checks; S2a submitted
Outputs copied into w/out (from out/w, out/cg). BBH-27 (6,511 items, greedy, max_new 600):
| row | mean27 | tool12 | other15 | tok/q | 0.6B mean27 |
|---|---|---|---|---|---|
| 0 plain 1.7B-Base | 51.74 | 55.37 | 48.84 | 206 | 41.19 |
| 1 ft17 tools off (call rows 42.6%) | 42.26 | 41.90 | 42.54 | 167 | 32.59 |
| 3 ft17 + 8 tools + gate (fire 47%, latch 7.9%) | 66.33 | 95.73 | 42.82 | 161 | 54.99 |
| ctl ft17 + 8 tools, no gate | 66.04 | 96.23 | 41.88 | 164 | 52.66 |
- Gate refit on ft17: AUROC train/test 1.0000/1.0000 (n 441/522), theta 3.438: test pos-pass 0.996, neg-veto 1.000.
  Re-derivation (j23836) gives a numerically IDENTICAL probe (w max|diff| 0, theta/b/pct equal; cmp_probe.py, j23849;
  the file sha differs only by pickle serialisation). Python gate vs compiled gate on the teams' held-out sets A_oos2/B_oos/C_ood:
  61,774 steps, 0 latch / 0 token mismatches (sets_compiled.json shadow); set acc none 78.3/73.3/100, gate = compiled 76.1/73.3/100.
- Exactness (chk, 128 dev rows; run 4-way shared so speeds there are not representative):
  base17: greedy vs HF 83/128 identical (acc 43.75 vs 41.41; div gaps <= 0.25 = bf16 near-ties), b32 vs all 128/128, TF ratio 0.99990,
  grad cos vs HF32 0.961 (HF bf16 vs HF32 0.966), max_mem 67.6 GB.
  ft17+tools: 124/128 identical (acc 89.06 vs 88.28), +gate 125/128; b32 vs all 128/128; TF ratio 0.999995; write flags TF vs decode
  0 mismatch / 1,515 writes; tool fuzz 0/57,312 steps (trainer and engine); grad cos 0.074 but HF bf16 vs HF32 is 0.105 (this chk batch had
  few non-zero-advantage docs, loss -1.9e-5: gradient is bf16-noise dominated for every implementation); max_mem 62.2 GB.
- idle bit-identity (408 dev): tools and tools+gate: idle rows 6/6 identical, acting rows 402/402 identical up to the first write/latch, 0 violations.
  dev acc: plain ft17 39.2, + tools 90.2, + tools + gate 90.0.
- disk 455 GB free (j23849 01:38Z); rl28/bbh: ft17 6.5 GB, runs 18 GB.
- **j23850 rl28bbh_S2a** (queued 01:39Z): P17_s2 + F17_s2, cap 0.46 each. Running: j23847 F17a (GPU 4), j23848 I17a (GPU 0, 01:37Z).
- Remaining: I17_s2 (+ evals of finished runs as an extra worker), u300 evals F17_s*, I17_s*, P17_s2, ctl F17 u300 + tools + gate; final_ana.sh.

## 01:40Z (relaunch 2) P17_s0/s1 done; F17a + I17a submitted
- **j23813 P17a done** (rc 0, 1h49m, mean SM 98.4%, 2 procs, max_mem 63.6 GB each; ~21 s/update per proc packed, u1 TF ratio 1.0000).
  Dev (408, greedy) u0 46.3 -> u300 **80.4** (s0) / **78.4** (s1) (0.6B P0: 36.3 -> 67.9/67.9/64.7). u300.pt in $ROOT/rl28/bbh/runs/P17_s{0,1}; resumes deleted.
- pack.sh: a worker whose command already has --mem_frac keeps its own caps (lets RL + eval workers share a GPU with different caps).
- **j23847 rl28bbh_F17a** (GPU 4, 01:34Z): F17_s0 + F17_s1 (row 2, ft17 tools off, cap 0.38 each, B_max 64, budget 8192) +
  worker 3 = BBH-27 evals P17_s0_u300, P17_s1_u300 (cap 0.18). (j23846 = the same pack mis-typed, cancelled after 3 s, nothing written.)
- **j23848 rl28bbh_I17a** (queued 01:34Z): I17_s0 + I17_s1 (row 4, ft17 + 8 tools + probe_ft17.pt gate, cap 0.46 each).
- Node disk 458 GB free (01:33Z). Remaining after these: P17_s2, F17_s2, I17_s2 (pack when setup2 ends, sized from its chk max_mem),
  u300 evals of F17_s*, I17_s*, ctl F17 u300 + tools + gate; then w/final_ana.sh -> RESULTS.md + rows.json.

## 23:36Z port of the code (copies in rl28/bbh/w and rl28/bbh/cg; rl24 files untouched)
- fastqwen.py: (D, FF) from env HOST_SIZE ('0.6B' default -> 1024/3072; '1.7B' -> 2048/6144). H/KVH/HD/NL identical.
  job.sh exports HOST_SIZE=1.7B. rl_bbh install() asserts the HF config == fastqwen shapes; gate probe dim == D.
- bbhtrain: auto activation-checkpointing threshold scales with D and respects the per-process cap (BBH_MEM_CAP from --mem_frac).
- rl_bbh: new `idle` mode (greedy dev: implants detached vs attached [vs + gate]; rows with no write/latch must be
  token-identical in full, the others identical up to the first write/latch step); max_mem_GB logged in check / every update / DONE;
  runs_dir $ROOT/rl28/bbh/runs; --ckpt_every 300 (only u300.pt, 3.4 GB bf16); resume.pt deleted at DONE (and no final resume write);
  resume writes are SKIPPED if disk free would drop below 250 GB ([30]).
- job.sh: hosts base17 = Qwen/Qwen3-1.7B-Base, ft17 = $ROOT/rl28/bbh/ft17; modes ft, idle added. VETO = w/veto_q17.pt.
- cg/job_refit.sh: FT_HOST ft17, VETO q17, outputs $ROOT/rl28/bbh/cg/probe_ft17.pt.

## [30] disk declaration (before any write > 20 GB)
- ft17 (fp32 safetensors, as ft.py saves): ~6.9 GB. Gate-refit states (cg/states2-4.pt, 2048-d): a few GB; deleted after the probe is final.
- RL: resume.pt per running arm ~20.4 GB (fp32 params + Adam) + a .tmp copy during a write -> <= ~41 GB per running arm;
  at most 4 arms running (2 jobs x 2 procs) -> transient <= ~120 GB typical, 160 GB worst case (all writes coincide).
  Deleted at DONE. Writes are skipped (logged) if free space would fall below 250 GB.
- Kept: u300.pt bf16 3.4 GB x 9 runs = ~31 GB + ft17 6.9 GB + probe (KB).

## 23:38Z submitted (2 jobs, one B200 node, 4 h timeouts)
- **j23812 rl28bbh_setup** (setup.sh, packed, dmon w/out/pack_setup): call-FT ft17 -> gate refit (states, probe, sets);
  base17 hc -> chk (B 96, budget 8192, cap 0.40: memory measurement for RL packing) -> row 0 eval; after ft17: hc ft17_t0 -> row 1;
  hc ft17_t1 -> chk -> idle bit-identity (408 dev rows) -> ctl eval ft17 + tools; after the probe: hc/chk/idle with gate -> row 3.
  Evals B_max 64, cap 0.28.
- **j23813 rl28bbh_P17a** (pack.sh): P17_s0 + P17_s1 (row 2', Base + RL, needs no FT) as 2 processes, cap 0.48 each,
  B_max 64, trainer budget 8192 (0.6B used B 256 / budget 16384; memory-driven, does not change the update math).
  u1 TF ratio and u0 dev logged by the run itself; the exactness check for the 1.7B engine is chk base17_t0 in j23812.

## 00:44Z j23812 setup finished (1h05m, rc 0, mean SM 95.2%), but its per-step outputs are lost
- Persisted on the node: **ft17** ($ROOT/rl28/bbh/ft17; FT took **100 s** on one B200 = **0.03 GPU-h**; 0.6B took ~90 s) and the
  refit gate **probe_ft17.pt** + states2-4 ($ROOT/rl28/bbh/cg; refit wall 1961 s = 0.54 GPU-h, shared GPU).
- Lost: hc/chk/idle/eval JSONs and the probe log. Cause: submitted from rl28/bbh, where the queue copied back only ./out; the
  workers wrote into w/out and cg/out. (Lesson: setup2.sh copies w/out and cg/out into ./out on EXIT.)
- **j23836 rl28bbh_setup2** (setup2.sh): redo hc/chk/idle/evals (rows 0, 1, 3, ctl ft17 + tools), chk steps serialized by flock;
  re-derive the probe into probe_ft17_rederive.pt (sha must equal probe_ft17.pt; the original stays in use); gate sets none/gate/compiled.
- j23813 P17a (P17_s0, P17_s1) running since 23:38Z on GPU 4, 100% SM, 134 GB used by the 2 processes (gpumon).
