# [28] GSM8K on Qwen3-1.7B-Base: running log (UTC)

> Release note: this is the original run log, scrubbed (cluster/queue details removed; job ids kept as labels only).
> Path map to this release: `fe/stk_r2q_17` -> `trees/stk_r2q_17` (overlay), `eng/` code tree -> `trees/arm17` (overlay),
> `res/<set>_<config>.jsonl` -> the bundles in `items/` (read via `uniform/items.py`), `fe/out_j*/` -> not shipped
> (generations), `$ROOT` -> `$WSM_DATA`. Scripts named here but not shipped: refit17.py, stopcal17, probe/, pack2_17.sh,
> mkq32cdir.py (their outputs, the refit `.pt` files, ship inside the overlays). RL_RESULTS §n -> REPRODUCE §17/§18.

Task: repeat the uniform 4-row protocol of [26] (RL_RESULTS §12.1) on Qwen3-1.7B-Base. GSM8K only. Same implants as on 0.6B:
only readers and writers are refit, circuits unchanged, nothing new climbed. RL recipe from [27] (lr 1e-6).
Everything lives in this folder. Shared files are not edited.

## 20:33Z start / feasibility audit

**Host.** Qwen3-1.7B-Base has hidden 2048, FF 6144, 28 layers, 16 q / 8 kv heads, head_dim 128, tied embeddings and the same tokenizer.
- It was not in the node HF cache (probe/hfchk.sh j23740: only Qwen3-0.6B(-Base) and Qwen2.5-0.5B were present).
- The node has network. Fetched with probe/fetch.sh j23741 into $ROOT/hf (snapshot ea980cb0…, 3.44 GB).
- Config verified, and the tokenizer vocab is identical to 0.6B (151669 entries, same map).

**Where the code assumes 0.6B.** One constants line, `wiring/fastqwen.py`:
`H, KVH, HD, D, FF, NL = 16, 8, 128, 1024, 3072, 28`, plus the literal model name `'Qwen/Qwen3-0.6B-Base'`
(tw_dump_lean.py tokenizer/HF load, fe_hook.py, rl_hook.py, comp_sanity.py, ...).
- Every kernel takes Q.D / Q.FF / Q.H / Q.HD as constexpr. add_rms D=2048 is a power of 2, silu grid FF//1024 = 6.
- Attention kernels depend only on H/KVH/HD, which are equal for both hosts.
- So the KV cache per token is identical (28 × 8 × 128). Only the weights and the residual width double.

**Host-fitted pieces** (what must follow the host):
1. Calculator / chain2 reader tables and tk symbol tables: built closed-form from the host embedding inside `attach()` /
   `make_reader`. They refit themselves automatically.
2. Write directions:
   - implant `make_out_dirs`: Adam on model.norm + lm_head;
   - comp_serial `_solve_dirs`: unit-rms state, margin target SW_DMARG;
   - comp_sanity `_solve_dirs`.
   All are solved at attach time on the host's final norm + LM head, so they refit themselves automatically.
   SW_DMARG=40 is used everywhere on 1.7B: the RL engine copy gets patch_dmarg, the eval tree already has it.
3. End-of-number veto vector. Reuse `results/gsm_bigger/veto_q17.pt`: Qwen3-1.7B-Base, same scripts/solve_veto.py,
   GSM8K train rows 2000–2299; held-out no-digit 100%, own choice kept 99.37%. Selected by HOST_MODEL in wiring/implant.py.
4. Reserved stop row (stk_r2q `stoprow.pt`, rung 2 only): u = min-eigvec of the host's post-lnf state second moment.
   Refit with the same stopcal_hook.py on the 1.7B host, teacher-forced on the same dev600 rung-2 token rows (split B = train).
5. Verifier probe `verifier_probe_L18.pt`: a logistic probe on the layer-18 residual (2048-d on 1.7B) at '####'.
   - Rung 2 uses it in the vote (VM=1) and in AW_ADAPT (A share, S stop). The CW RL arm also uses it in its reward vote (vmode 1).
   - Refit with the ana.py --refit recipe: StandardScaler + LogisticRegression C=0.001, folded.
   - Data: 1.7B-Base samples on GSM8K train split C, T=1, K=8, calc v2 on (stack_hook.py samp).
     This matches the 0.6B probe's data spec (split C, T=1, G=8).
6. dfa_tables.pt depends only on the tokenizer. Reused.
7. ckpt_a2s2 (`--eval_ckpt`): with `--host_ckpt base|DIR` and arm 2, nothing host-shaped is loaded from it (no links/inwire/Dp).

**Plan / layout.**
- `fe/`: copy of final_eval (fe.sh, fejob.sh, run_hook.sh, dig_san, wiring) plus `stk_r2q_17` (copy of stk_r2q). Used for the evals, as on 0.6B.
- `eng/`: copy of engine/rel/v5 (dig_san, stk, wiring), used for RL. `eng/run_arm_17.sh` is the arm script copy:
  ARMDIR $ARMDIR17, its own resume dir, one GPU.
- `patch17.py` (idempotent) patches the copies only. Host chosen by env HOST_MODEL; the default is 0.6B, so behaviour is then unchanged.
  - fastqwen: (D, FF) per host, plus a config assert in load_hf_params.
  - Model literal replaced by `os.environ.get('HOST_MODEL', ...)`.
  - Per-host veto file.
  - Dim asserts on the probe and stoprow loads, so a 0.6B file on the 1.7B host fails loudly.
  - SW_DMARG in eng/stk/comp_serial.
- fe/run_hook.sh copy: exports HOST_MODEL=Qwen/Qwen3-1.7B-Base and copies only dfa_tables.pt from the node.
  The 0.6B probe and stoprow are kept as *_q06.pt and never loaded.
- Checks:
  - (a) idle bit-identity, greedy dev600: host only (H) vs calc_s4+chain2 attached (D). Tokens must match up to the first implant step,
    and rows with no implant step must match fully. Also stack_hook STK_CHECK (stock engine == calc-v2 stack off) and S1 silent rows == S0.
  - (b) forced-token audit on rung 2 (writer_steps / sampled_ne_written, expect 0 with SW_DMARG=40).
  - (c) RL: RL_FASTCHECK greedy old-vs-fast identity and the TF ratio at u1.
- Rows:
  - row 1: KIND H, test/hard;
  - row 3: the r2_base lines + SW_DMARG=40;
  - row 2: P64 config × seeds 0,1(,2);
  - row 4: CW config × seeds 0,1(,2), evaluated with the rung-2 line.
  CIs with the ana_uni.py method.

## 20:42Z jobs A (j23742: row 1 + H-vs-D idle check + stopcal/stoprow) and B (j23743: stack silence check + probe refit) submitted

## 21:25Z task [30] disk rules: footprint of the 1.7B RL arms (declared before any RL submit)
- Resume state: rl_hook keeps ONE file, `$ROOT/rl28g/states/rlstack_<ARM>.pt`, overwritten atomically every 2 u. It holds the fp32 host and the AdamW moments, about 20 GB for 1.7B.
  - During a write, a `.tmp` copy also exists, so the peak is about 41 GB per arm.
  - The file is deleted when the arm finishes. This already meets the "latest ≤2 states" rule.
- Model checkpoints: rl_hook writes step_000100/200/300 (bf16, 3.4 GB each). Only step_000300 is used, so run_arm_17.sh deletes 100/200 when the arm is done.
  This leaves 3.4 GB per arm, about 14 GB for 4 arms.
- Peak: at most 2 arms at once, so at most about 2 × 41 + 14 ≈ 96 GB transient under $ROOT/rl28g. Steady state after all arms: about 14 GB.
- Other writes:
  - inductor cache $ROOT/rl28g/inductor_<ARM> (small);
  - HF snapshot of Qwen3-1.7B-Base, 3.4 GB (already on disk);
  - probe-refit samples (trainC_samp.pt, ~50 MB, job-local out/). Deleted after use.

## 21:36Z task [31] packing applied
- Queued j23742 (A) was replaced by packed **j23758 rl28g_AB** (fe/jobAB.sh). Queued j23743 (B) was cancelled.
- j23758 runs 2 concurrent processes on 1 GPU, each `--mem_frac 0.45`:
  - A: row 1 test+hard KIND H at B_max 128, then the H-vs-D greedy dev600 idle check, then stopcal → stoprow;
  - B: calc-v2 silence check on dev600, then trainC K=8 samples → probe fit.
- `nvidia-smi dmon` logs to out/dmon.txt, and the job prints its mean SM%.
- fe/fe.sh copy: FE_XA env is appended to XARGS, which is how --mem_frac reaches tw_dump_lean.
- Plan for the RL phase:
  - pack P17_s0 + P17_s1 (+s2) as concurrent processes in one job (1.7B P config: KV 56 GB at B 384, so B_max 192 when packed 2-3 per GPU);
  - CW17 seeds 2 per job if memory allows (each ~0.45: B_max 128);
  - evals packed with them, or packed together.

## 21:45Z re-launch (previous instance killed 21:38Z): state + P17 pack submitted
- State found: j23758 rl28g_AB still queued at low priority (behind the [26] normal jobs). No rl28 driver processes were alive. Nothing was resubmitted.
- Row 2 does not depend on AB (no host-fitted file is used by RL_ARM=P), so I started it now as the 2nd allowed job:
  - `eng/pack_17.sh` (new): a packed chain driver. One cluster-queue job runs several arms as concurrent processes on one GPU, each `run_arm_17.sh ARM --inner` with `--mem_frac` (new env MEMFRAC in run_arm_17.sh) and B_max via BMAX.
  - It logs dmon to out/dmon_<NAME>.txt and resubmits until every arm has DONE. Each arm resumes from its own state file. An OOM inside an arm halves its B_max.
  - **j23766 rl28g_P17a** = P17_s0 + P17_s1, mem_frac 0.45 each, B_max 192 (P config otherwise unchanged: S_max 1280, N=4, 64×8, lr 1e-6, 300 u, seeds 0/1).
  - Driver runs detached on the sandbox; chain log eng/pack_P17a.chain.log.
- The CW17 pack waits for the 1.7B verifier probe (AB part B). It will be submitted when AB ends, keeping ≤2 jobs.

## 22:25Z waiting: j23758 (AB) and j23766 (P17a) both still queued (low priority; all 8 GPUs busy with normal-priority [26] jobs). Eval lists for rows 2/3/4 written: fe/lists/{r3_test,r3_hard,p17_s*,cw17_s*}.list (rung-2 line of 0.6B r2_base + SW_DMARG=40; stk_r2q_17 for both rows 3 and 4; note 0.6B row 4 used stk_r2, row 3 stk_r2q).

## 23:10Z j23758 (AB) ran 22:55–22:58Z on one B200:0. Row 1 is done. Idle bit-identity check PASSED. Refits failed (fixed); resubmitted as j23795.
- **Row 1 (plain Qwen3-1.7B-Base, baseH line, T=1 seed 7): test 45.26, Hard 24.49; tok/q 105.5 / 138.0** (res/test_base17H.jsonl, res/hard_base17H.jsonl).
  For comparison, 0.6B: 28.05 / 10.46.
- **Idle bit-identity check** (greedy dev600; host only `idH` vs calc_s4+chain2 attached `idD`; idle_chk.py):
  - 73/73 rows with no implant step are token-identical in full;
  - 527/527 rows with an implant step are identical up to their first implant step;
  - 0 divergences before w1;
  - dev acc H 80.17, D 80.00.
- Failures:
  - (1) stopcal / stkdev / trainC ran without `--host_ckpt base`. run_hook falls back to `--eval_ckpt` a2s2, a 0.6B host, so the 1.7B fastqwen got a shape mismatch, failing loudly as designed.
    On 0.6B the stoprow and probe were fit on that a2s2 host; 1.7B has no a2s2, so the 1.7B refits use the Base host (the only one).
  - (2) The node venv has no sklearn. refit17.py now has StandardScaler / L2 LogisticRegression (lbfgs, C=0.001, intercept unpenalised) / AUROC in torch float64.
    Locally tested on synthetic 4000×2048 data: gradient norm at the optimum 1.5e-7, i.e. the exact convex optimum, so the same solution as sklearn.
- **j23795 rl28g_AB2** (fe/jobAB2.sh), one job:
  - phase 1: stopcal→stoprow ∥ (stack silence check + trainC samples → probe), 2 procs at mem_frac 0.45;
  - phase 2: copy the refit stoprow and probe into stk_r2q_17, run test_aw, then **row 3** test ∥ hard (B_max 64, S_max 8192, mem_frac 0.45 each).
  - The old out/ was moved to fe/out_j23758.
- P17a j23766 has been running on one B200:2 since 22:56Z. No stdout until it ends; arm logs are on the node.

## 23:15Z j23795 (AB2, 23:03–23:09Z, GPU 4): refits done, calc-v2 silence check PASSED, row 3 crashed on a stop-row guard (fixed)
- **Calc-v2 silence check** (stack_hook, greedy dev600, Base host):
  - stock engine == stack off on 600/600 rows;
  - S1 (calc-v2 on): silent rows identical 595/595 (5 rows fire);
  - acc S0 80.33, S1 80.67.
- **Stop-row refit** (stopcal_hook hn_mineig on the 1.7B Base host, dev600 rung-2 token rows):
  - s = 27.9;
  - forced margin min 39.43 (cal) / 39.44 (hold);
  - normal states: top − z_c ≥ 16.48, z_c − lse ≤ −16.7. On 0.6B the normal-state gap was ≥ 12.3.
  - → refit/stoprow.pt (c = 151669, 2048-d).
- **Verifier probe refit** (L18, 2048-d; split C train, T=1, K=8, calc v2 on, Base host):
  - 11,992 rows, 11,457 with an answer, mean_ok 58.6%;
  - in-sample within-question AUROC 0.86 over 1170 mixed questions;
  - → refit/verifier_probe_L18.pt, installed in fe/stk_r2q_17 and eng/stk.
- Aw fuzz (test_aw) PASS.
- Row 3 crash: `_implant_stop` asserts that the padded vocab rows 151669..151935 are degenerate (identical), as they are on 0.6B. On Qwen3-1.7B-Base they are distinct small vectors.
  - The purpose of the guard is "row c is an unused pad the host never emits". stopcal shows this for 1.7B: the original row c has z_c − lse ≤ −19.2 over 669k positions.
  - So the guard is relaxed only under env SW_PAD_NONDEG_OK=1. fe/run_hook.sh sets it when HOST_MODEL is 1.7B. The pad stats are printed at attach time.
  - No circuit change: the stop row and its writer are the same recipe.
- **CW17 pack launched: j23802 rl28g_CW17a** (eng/pack2_17.sh = pack_17 + progress lines every 10 min on stdout).
  - CW17_s0 + CW17_s1 concurrent, mem_frac 0.49 each, B_max 96, FT_BUDGET 16384.
  - FT_BUDGET is the micro-batch token budget: gradient accumulation only, so the recipe is unchanged. It was lowered so that the 1.7B trainer + KV fit twice on one GPU.
  - Started 23:14Z on one B200:6.
- Next, when P17a ends: fe/jobE1.sh = row 3 test ∥ hard (B_max 64) + row 2 evals of P17_s0/s1.

## 23:55Z P17a DONE (j23766, 22:56–23:50Z, GPU 2, mean SM 96.3%): row 2 RL arms P17_s0 and P17_s1 finished u300
- FASTCHECK (greedy dev600, old engine vs fast engine), both arms: 600/600 token-identical, max |Δlogp| 0. First-update TF ratio 0.99995 / 1.0003 (s0 / s1).
- At B_max 192 the first update OOMed under the 0.45 cap. The in-job retry halved B_max to 96:
  - s0 restarted from u0;
  - s1 resumed from u2.
  - The recipe is unchanged. B_max is only the rollout concurrency.
- Peak mem 71–72 GB per process; 9–10 s per update.
- Greedy dev600 accuracy, u0 → u300: s0 80.17 → 90.83, s1 80.17 → 90.00. Reward over the last 10 updates: 0.77 / 0.78.
- Checkpoints: $ROOT/rl28g/rl/P17_s{0,1}/step_000300 (step_000100/200 deleted by the script).
- **j23818 rl28g_E1** submitted (fe/jobE1.sh): row 3 test ∥ hard, then the P17_s0 / P17_s1 test+hard evals.
- CW17a (j23802): about 85 s per update with 2 arms → about 7 h for the pair (ETA ~06:30Z).

## 00:30Z Row 3 and row 2 (2 seeds) evaluated. Third seeds launched as a pack.
- **Row 3 (1.7B-Base + rung 2, j23818): test 83.47, Hard 57.01; tok/q 2232 / 3531.**
  - Forced-token audit: 0 misses in 151,900 (test) and 206,955 (Hard) writer steps, so SW_DMARG=40 holds on 1.7B.
  - The reserved stop token is c = 151669. The 1.7B padded rows have 5 distinct values, max norm 0.375 (real rows median 1.62).
- **Row 2 (P17 u300, j23831; stk_r2q_17 tree, B_max 128, same as row 1):**
  - s0: test 80.36, Hard 56.10;
  - s1: test 80.21, Hard 56.94;
  - tok/q about 117 / 152.
  - The first E1 attempt at P evals failed: the fe copy has no `stk` tree, and the default B_max 384 OOMs under the 0.45 cap. Rerun as E2.
- **Disk notice (task [30]).** Starting the s2 pack means 4 arms at once (CW17_s0/s1 + CW17_s2 + P17_s2).
  - Transient resume states: about 4 × 41 GB ≈ 164 GB under $ROOT/rl28g.
  - the run disk had 443 GB free at 22:30Z.
  - P17_s2 ends in about 1 h, which frees about 41 GB.
- **Pack S2 launched** (eng/pack3_17.sh = pack2 + per-arm FT_BUDGET):
  - CW17_s2 (B_max 96, FT_BUDGET 16384) + P17_s2 (B_max 96, FT_BUDGET 32768, as s0/s1);
  - mem_frac 0.49 each.

## 00:45Z
- j23802 CW17a running: CW17_s0 u50 / CW17_s1 u49 at 00:36Z (~95 s/update) → ETA ~07:15Z.
- j23832 S2 running since ~00:29Z on one B200:2: P17_s2 u52 at 00:40Z (~12 s/update, ETA ~01:30Z); CW17_s2 u2.

## 01:15Z
- P17_s2 u212 at 01:10Z (ETA ~01:35Z); CW17_s2 u23 (~85 s/update while packed). CW17a s0/s1 still running.
- Wrote fe/jobE3.sh (row-4 eval: CW_SEEDS procs ∥, same settings as row 3 [stk_r2q_17, B_max 64, S_max 8192, mem_frac 0.45], optional P_SEEDS after).
  Job-slot cap (2) is held by j23802 + j23832 → P17_s2 eval will be done together with CW17_s2 eval at the end (CW_SEEDS=2 P_SEEDS=2).

## 01:40Z re-launch 2 (previous instance killed ~01:38Z): state, eval automation, RESULTS.md draft
- Found: j23802 rl28g_CW17a running (CW17_s0 u78 / CW17_s1 u82 at 01:26Z, ~100 s/u → ETA ~07:40Z); j23832 rl28g_S2 running:
  **P17_s2 DONE_RUN at u300 (~01:30Z)**, CW17_s2 u34 at 01:30Z, now alone on its GPU. Drivers pack2_17.sh (CW17a) and pack3_17.sh (S2) alive. Nothing resubmitted.
- Job cap 2 is held by the two packs, so the P17_s2 eval waits for the S2 pack to end (it is ~20 s per set; packed with the CW17_s2 evals).
- New fe/jobE4.sh: generic packed eval job, PROCS = procs, each a '+'-chain of lists (cw17_* = rung-2 settings of row 3: stk_r2q_17,
  B_max 64, S_max 8192, mem_frac 0.45; p17_* = KIND H, B_max 128). Split lists fe/lists/cw17_sK_{test,hard}.list.
- New **evchain.sh** (detached driver on the sandbox, log evchain.log; restart-safe: skips evals whose res/ files exist, attaches to a
  queued/running rl28g_E4a/E4b instead of resubmitting):
  - A: when eng/pack_S2.chain.log has RLDONE → rl28g_E4a: PROCS "cw17_s2_test+p17_s2 cw17_s2_hard";
  - B: when eng/pack_CW17a.chain.log has RLDONE → rl28g_E4b: PROCS "cw17_s0_test+cw17_s0_hard cw17_s1_test+cw17_s1_hard";
  - after each: fe/out → fe/out_<jID>, jsonl → res/, mkrows.py (rows.json, seeds auto-detected) + mkresults.py (RESULTS.md), NOTES line.
  - If evchain is not alive after a restart: `cd rl28/gsm8k && bash evchain.sh > evchain.driver.log 2>&1` (detached).
- New mkresults.py (+ RESULTS_tail.md: refits/checks text) writes RESULTS.md: 4-row table 1.7B next to 0.6B, per-seed, tok/q, and the
  ana_uni.py paired bootstrap (2-1, 3-1, 4-1, 4-3, 4-2, interaction); rows pending until res/ has them. Draft RESULTS.md written (rows 1-3).
  Current 1.7B effects: RL alone (2-1) test +35.03 [+32.41, +37.72], Hard +32.03 [+29.30, +34.84]; implants alone (3-1) test +38.21
  [+35.41, +41.02], Hard +32.52 [+29.87, +35.33].

## 01:52Z ETAs (evchain alive, pid 84559)
- CW17_s2 alone on j23832: u48 at 01:40Z, ~43 s/u → u300 ~04:40Z; then rl28g_E4a (~25 min) → rows 2 (3 seeds) + row-4 s2 ~05:10Z.
- CW17a (j23802): CW17_s0/s1 ~100 s/u → u300 ~07:40Z; then rl28g_E4b (~40 min) → final table ~08:20Z.
- Remaining manual step after E4b: check RESULTS.md / rows.json (auto-regenerated), add the row-4 audit numbers + a reading paragraph.

## 02:20–02:50Z (the coordinator 02:22Z: GPU 5 idle, temporary 3rd slot) P17_s2 evaluated; CW17a split into 2 single-arm jobs
- **j23861 rl28g_E4p** (normal prio, GPU 5, 2 min): P17_s2 u300, KIND H, B_max 128: **test 81.27, Hard 56.86** (tok/q 119 / 154). Collected
  into res/ (fe/out_j23861); fe/done_p17_s2 marker makes jobE4.sh skip p17_s2 in E4a. **Row 2 (3 seeds): test 80.62 (80.36/80.21/81.27),
  Hard 56.63 (56.10/56.94/56.86)**; RL alone (2-1) test +35.36 [+32.78, +38.11], Hard +32.15 [+29.54, +34.77]. rows.json + RESULTS.md regenerated.
- **Critical-path decision: split CW17a.** Packed 2/GPU, CW17_s0/s1 ran ~100 s/u (u108/u118 at 02:16Z → ETA ~07:35Z); the lone CW17_s2 runs
  ~47–57 s/u at 97–100% GPU util (nvidia-smi samples). Projected gain ≈ 2 h > 1.5 h. Resume is exact: rl_hook state (every 2 u, async,
  atomic os.replace) holds fp32 host + AdamW moments + py/torch/cuda RNG + the pending batch; a TERM mid-write leaves only a stale .tmp.
  Note: with packed saves taking ~250–350 s some 2-u saves are skipped, so ≤ ~4 u were redone.
  - 02:23Z: killed evchain + the pack2_17 driver (and its queue wait), cancelled j23802 (GPU 6, ran 3h09m).
  - 02:24Z: **j23863 rl28g_CW17c** = CW17_s1 alone (GPU 5), **j23864 rl28g_CW17b** = CW17_s0 alone (GPU 6); pack3_17.sh, same settings
    (MEMFRAC 0.49, B_max 96, FT_BUDGET 16384, seeds/hyperparameters unchanged). Both resumed (s1 u126 at 02:35Z, s0 u113 at 02:36Z).
  - Rate alone: ~55–60 s/u → **ETA CW17_s1 ~05:20Z, CW17_s0 ~05:35Z** (was ~07:35Z). S2 (CW17_s2) ETA ~04:45Z.
  - Job count now 3 (S2, CW17b, CW17c) using the temporary 3rd slot; it drops back as the arms end.
- evchain.sh relaunched 02:24Z (pid 277467): stage A unchanged (S2 → E4a: cw17_s2 test ∥ hard; p17_s2 skipped), stage B now waits for
  packs CW17b and CW17c → E4b. Final table ETA ~06:15Z.

## 03:30Z task [32](b)/(c) for 1.7B prepared and chained (runs after the row-4 table; does not touch evchain)
- Revised RL ETAs (rates incl. dev evals): CW17_s2 u149 at 03:21Z (~58 s/u) → ~05:50Z; CW17_s0 u153 at 03:16Z (~60 s/u) → ~05:45Z;
  CW17_s1 u160 at 03:15Z (~70 s/u) → ~06:00Z. Row-4 evals (E4a/E4b) then → final table ~06:45Z.
- fe/stk_r2q_17/fe_hook.py: KIND=SCH ported from final_eval/stk/fe_hook.py (KB = 'H' for SCH: host-only engine, G = N samples per item,
  plain majority, ties → earliest sample). Other kinds unchanged (diff = KIND→KB in the 4 setup lines, SC→SCK in the 3 sampling lines).
- fe/jobE4.sh: new list kinds q32c_r3_* (rung-2 settings of row 3), q32c_h* (KIND H, B_max 128 as row 1), q32b_* (SCH, B_max 256,
  S_max 1280 as 0.6B SCH); E4TAG env tags out/ files so two jobs can copy back into fe/out without clashes.
- Lists: fe/lists/q32c_r3_s{11,13}_{test,hard}.list (row-3 line + FE_SEED), q32c_h.list (base17H at FE_SEED 11/13);
  mk32.py writes fe/lists/q32b_{test,hard}.list with N_set = round(row-4 tok/q / row-2 tok/q) (3-seed means) for P17_s0-2 + Base17.
- **q32chain.sh** (detached, pid 539834, log q32chain.log): waits for evchain end + all 6 row-4 res files → mk32.py → 2 parallel low-prio
  jobs rl28g_Q32c (PROCS "q32c_r3_s11_test+q32c_r3_s11_hard q32c_r3_s13_test+q32c_r3_s13_hard") and rl28g_Q32b ("q32b_test q32b_hard+q32c_h")
  → fe/out_q32_<ids>, res/, mkrows.py + mkresults.py, NOTES line; one retry of anything missing. Restart-safe (skips existing res/,
  attaches to queued/running Q32b/Q32c).
- mkresults.py: new sections "[32](b) matched-compute" (rows 2-MV / 1-MV, CIs for 4 − 2-MV, 2-MV − 2, 1-MV − 1, 3 − 1-MV) and
  "[32](c) eval-seed spread" (rows 1/3: seeds 7/11/13, mean, sd). mkrows.py: extra rows.json rows 2-MV, 1-MV, 1-seeds, 3-seeds.
  Both verified on synthetic copies of res/ (scratch removed).
- Still manual after the table: row-4 write-audit counts + a reading paragraph in RESULTS.md (RESULTS_tail.md).

## 05:10Z parent 05:12Z: [32](c) submitted early (independent of row 4): j23906 rl28g_Q32c (low prio, timeout 7200), submitted from fe_q32c/ (copy of fe/ via mkq32cdir.py, own out/ so no clash with evchain's fe/out). PROCS "q32c_h+q32c_r3_s11_test+q32c_r3_s11_hard q32c_r3_s13_test+q32c_r3_s13_hard" (2 procs @ mem_frac 0.45). Detached driver q32c_now.sh (log q32c_now.log; id in q32c_now.id, restart-safe): wait, fetch, out -> fe_q32c/out_<id>, jsonl -> res/, tables regen, NOTES line. q32chain then sees res/ files and submits only Q32b (q32c_h dropped from it too since res/test_base17H_s13 exists). evchain untouched.

## 05:54Z [q32c_now auto] [32](c) job j23906 done (rc=0), jsonl -> res/ (8/8 files): FE done test base17H_s11 n=1319 acc=47.08 tok/q=106.4 sec=16;FE done hard base17H_s11 n=1319 acc=24.87 tok/q=139.9 sec=24;FE done test base17H_s13 n=1319 acc=46.10 tok/q=104.9 sec=32;FE done hard base17H_s13 n=1319 acc=24.18 tok/q=138.9 sec=33;FE done test r2_17_s11 n=1319 acc=83.62 tok/q=2233.4 sec=950;FE audit test r2_17_s11 writer_steps=152084 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669;FE done hard r2_17_s11 n=1319 acc=55.50 tok/q=3543.8 sec=1314;FE audit hard r2_17_s11 writer_steps=208308 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669;FE done test r2_17_s13 n=1319 acc=83.40 tok/q=2180.8 sec=880;FE audit test r2_17_s13 writer_steps=148458 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669;FE done hard r2_17_s13 n=1319 acc=54.89 tok/q=3509.6 sec=1439;FE audit hard r2_17_s13 writer_steps=206193 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669;

## 06:03Z [evchain auto] E4a (j23915) done:  FE done test r2_CW17_s2 n=1319 acc=87.64 tok/q=1739.9 sec=823; FE audit test r2_CW17_s2 writer_steps=116795 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669; FE done hard r2_CW17_s2 n=1319 acc=68.08 tok/q=2870.6 sec=1018; FE audit hard r2_CW17_s2 writer_steps=162963 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669;. res/ + rows.json + RESULTS.md regenerated.

## 06:31Z [evchain auto] E4b (j23918) done:  FE done test r2_CW17_s0 n=1319 acc=87.79 tok/q=2033.0 sec=930; FE audit test r2_CW17_s0 writer_steps=107550 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669; FE done hard r2_CW17_s0 n=1319 acc=67.32 tok/q=3534.7 sec=1454; FE audit hard r2_CW17_s0 writer_steps=150019 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669; FE done test r2_CW17_s1 n=1319 acc=88.25 tok/q=1787.1 sec=872; FE audit test r2_CW17_s1 writer_steps=110174 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669; FE done hard r2_CW17_s1 n=1319 acc=67.25 tok/q=2864.8 sec=1354; FE audit hard r2_CW17_s1 writer_steps=138901 sampled_ne_written=0 rows_with_mis=0 stop_tok=151669;. res/ + rows.json + RESULTS.md regenerated.

## 06:52Z [q32chain auto] [32](b)/(c) jobs done (8/8 SCH files): test scH16_P17_s0 n=1319 acc=86.43 tok/q=1896.2 sec=156;test scH16_P17_s1 n=1319 acc=86.13 tok/q=1839.9 sec=161;test scH16_P17_s2 n=1319 acc=87.11 tok/q=1894.0 sec=199;test scH16_base17 n=1319 acc=78.62 tok/q=1697.9 sec=219;hard scH20_P17_s0 n=1319 acc=63.08 tok/q=3075.2 sec=261;hard scH20_P17_s1 n=1319 acc=62.24 tok/q=3006.2 sec=249;hard scH20_P17_s2 n=1319 acc=62.93 tok/q=3082.3 sec=253;hard scH20_base17 n=1319 acc=48.14 tok/q=2764.0 sec=133;. rows.json + RESULTS.md regenerated.
