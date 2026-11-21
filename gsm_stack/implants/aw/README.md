# lane aw-base: all-weights (SerialWD, no RoPE reset) rung-0 baseline
## progress log (newest last)
- 02:45Z started. Reading stack-lane HOWTO_RUN_STACK.md, eval/ (the stack lane's copy of rel/v4.2 + stk/fe_hook.py = arms' dev-eval path).
- 02:52Z FOUND: the stack lane's gsm_stack/eval = frozen rel/v4.2 engine + stk/fe_hook.py (builds SerialWD exactly like the arms' dev eval:
  DualImplant calc_s4+chain2, comp_serial.SerialWD sw_setup(N, acap=320, R=1, posreset=False, vmode=1 probe L18, san=1, repmiss=0), fastroll,
  T=1). Host u0 = --eval_ckpt $ROOT/gsm_sel/ckpt_a2s2 (same host as our gen/). Its validation dev_val_u0 (seed 4242) = 81.33 (ref 81.8), 51 s/600 items.
  Same dev/dev2 split as our gen_hook (load_train('B') shuffle Random(0) [:600]/[600:900]).
- Copied eval/{stk,wiring,dig_san,run_hook.sh} to gsm_stack/implants/aw/dev_tree/ (fe_hook.py is the stack lane's; ours = aw/stk/aw_hook.py, gen/out row format).
- 02:58Z SUBMITTED j23275 (job_aw0): `cd gsm_stack/implants/aw/dev_tree && <queue-submit> gsm_stack/implants/aw/dev_tree 3000 job_aw0 -- bash -c "bash aw.sh val AW_SETS=dev AW_SEEDS=4242; bash aw.sh base AW_SETS=dev,dev2 AW_SEEDS=7,11,13,17"`
  val = harness check (must give ~81.33 = the stack lane dev_val_u0, seed 4242). base = rung 0, rows in aw/out/base/*.jsonl.gz (gen/out format,
  name {set}_N8_R1_v1_s1_m0_pr0_T1.0_seed{S}.jsonl.gz). Check: `gq log j23275 --tail 40`.
  NEXT: fill LADDER.md rung 0 from the AW lines; audit host-side decisions (below).

## How to run (rung-0 path)
- Code: gsm_stack/implants/aw/dev_tree/ = copy of gsm_stack/eval (frozen rel/v4.2 engine) + aw.sh (launcher, caches under $WSM_DATA/implants)
  + stk/aw_hook.py (our hook). Host u0 = `--eval_ckpt $ROOT/gsm_sel/ckpt_a2s2` (= a2_u300_s2, the same host as gen/).
- `cd gsm_stack/implants/aw/dev_tree && <queue-submit> gsm_stack/implants/aw/dev_tree 3000 job_NAME -- bash aw.sh RUNNAME AW_SETS=dev,dev2 AW_SEEDS=7,11,13,17 [AW_N=8 AW_R=1 AW_VM=1 AW_SAN=1 AW_RM=0 AW_PR=0]`
  -> aw/out/RUNNAME/{set}_N8_R1_v1_s1_m0_pr0_T1.0_seed{S}.jsonl.gz (gen/out fields; c2ev empty: calc here is DualImplant calc_s4+chain2,
  not calc v2) + `AW done` lines in the gq log. ~1 min per 600 items at N=8 (fastroll, B_max 192, S_max 5696).
- Stack = SerialWD(sw_setup N=8, acap=320, R=1, posreset=False, vmode=1 (sigmoid L18-probe weighted vote), san=1, repmiss=0),
  DualImplant(calc_s4 + chain2), weight-space forget (forget_ws), fastroll+TK1+COMPACT+OVL+ATT3, T=1. Identical to the RL arms' dev eval.

## Audit: host-side decisions in this path (cf. gsm_stack STATUS 02:43 [7]; RL_RESULTS.md not written yet)
All per-step stack logic is comp_serial.SerialW._sw: fixed tables (DFA TT[token,state], sym[token], stop/ENL/SNL token sets),
register updates via torch.where, ReLU voter (_vote), writer pointer table SH, residual writes gain*DIR[token]. The probe logit is a
fixed linear map at L18. Counters already in registers: cap (s_c vs acap), repairs (s_rep vs k_R), attempts (s_a vs k_N).
Forget = forget_ws extra q/k dims keyed on segment counter s_nf (attn3 kernel applies the same mask). No .item()/python branch on
row content inside the step (_step_triton / _sw / _book are tensor-only; graph-captured).
Remaining host-side items (disclosed, not content decisions unless noted):
1. Generator loop + T=1 sampling kernel + token budget (maxnew = 5416) stop. Disclosed in comp_serial docstring.
2. Stop rule `done = implant wrote eos (s_p == r_eos) | budget` (a read of the implant's register: OK).
3. FLAGGED (fixed in our copy, = stack-lane engine (rel/v5)): fastroll.py:120 / c2_engine.py:93 ended a row at token 0 if the FIRST sampled token
   was eos or contained "\n\n" (host stop rule). Never fired on dev (stack-lane: 0/600). Our j23275 ran v4.2 code (pre-fix); later runs use the fix.
4. fastroll admission / compaction / harvest (`done.tolist()`, slot moves): scheduling only, no effect on a row's tokens except the
   RNG stream position (sampler draws rand(seed, batch_pos)).
5. TK exactness guard flag (fastroll:235) is only printed.
6. posreset (AW_PR=1 / RL_POSRESET) = programmatic position-id offset: NON-COMPLIANT, off here (reference only).
7. Python-side: output decoding, regex grading (C.correct on last '#### n') — evaluation, not inference.

## RESULT rung 0 (j23275, done 02:51Z, 6.6 min)
- harness check dev seed 4242: 81.33 (= the stack lane dev_val_u0 exactly).
- dev  seeds 7/11/13/17: 81.83, 81.50, 80.83, 80.00 = **81.04**, tok/q 1079, oracle@8 90.50
- dev2 seeds 7/11/13/17: 79.67, 78.67, 79.33, 78.67 = **79.08**, tok/q 1085, oracle@8 89.17
- (non-compliant S3+reset ref: 88.50 / 85.75.) Gap vote vs oracle ~9.5 / ~10 pt. Written to LADDER.md rung 0.
- atlas/parse.py works on these rows: `python3 atlas/parse.py --gen aw/out/base --out aw/atlas_data --notes aw/atlas_counts.md`
  (register cross-checks 99.8-100%). KEY FINDING: per-attempt accuracy decays with attempt index (no reset => question is
  ever farther in RoPE distance): dev k=0..7: 80.7 69.1 61.8 53.3 46.5 40.2 33.5 29.5 (mean 51.8; oracle@32 96.5, oracle@8 90.5).
  => weight-space position reset is THE lever; a cheap interim is weighting slots by index / N=4 (vote8 vs vote4 ~equal in stack-lane tables).

## Porting A+B+C (implants/adapt, STK_SER ...:A=0.7/16/4:S=4/0.5:B=3:C=400) onto comp_serial.SerialW._sw (all in registers)
- A adaptive N (compiled comparator): in _sw replace `do_wait = end & ~do_rep & (s_a+1 < k_N)` by
  n=s_a+1; reuse _vote's score: share = score[win] / sum(keep*w); ext = (n>=8)&(n<Nmax)&((n%blk!=0)|(relu(tau*sum - top)>0));
  stop4 = (n==4) & all4 sane & equal & min sigmoid(lg)>=th; do_wait = end&~do_rep&((n<8)&~stop4 | ext). All torch.where/ReLU on
  registers, no python. Needs NS 8->16/24 (module const), forget_ws labels c_maxf=NS*2 (E power of 2), sw_maxn/SW_NMAX, SAN_SMAX ~8192,
  budget guard (glen+acap+56<=maxnew) as a register compare.
- B repair limit 3 for valid non-integer: rlim = where(valid & fl==fnz & ~neg, max(k_R,3), k_R); do_rep uses s_rep < rlim (s_rep is
  already a register counter; FLAG table gives fnz/neg).
- C cap 400: acap is a python int in `capped = rd & (s_c+2 >= self.acap)`; make it a tensor k_acap (setup constant), keep the item
  filter at 320 (same item set), recompute sw_maxn.
- Exactness: fuzz _sw vs adapt/test_adapt.py references; all-off must be bitwise equal to rung 0 (j23275 files).

## Task 2 (02:57Z): port A+B+C+stop4 into aw/stk/comp_serial.py (env AW_ADAPT spec, all-off bitwise = j23275), fuzz vs
implants/adapt/test_adapt.py refs, then 4 seeds dev+dev2 `A=0.7/16/4:S=4/0.5:B=3:C=400`. Keep diff small (sibling wpr lane patches
comp_serial too). Status: reading adapt code.
- 03:01Z PORT DONE (code) + SUBMITTED j23303 (job_awabc2: fuzz + all-off NS=8 bitwise check vs out/val + A+B+C+S on dev2 x4 seeds)
  and j23306 (job_awabc1: fuzz + dev x4 seeds). Cmd: `cd gsm_stack/implants/aw/dev_tree && <queue-submit> gsm_stack/implants/aw/dev_tree 3600 job_X -- bash aw_job.sh SETS [1]`.
  Outputs: aw/out/off8, aw/out/abcs_dev2, aw/out/abcs_dev; gq log lines `AW FUZZ PASS`, `OFF8 BITWISE n/600`, `AW done ...`.
  NEXT: when done, mean of 4 seeds -> LADDER.md rung 1 (A+B+C+S) if fuzz PASS + off bitwise 600/600; compare vs rung 0 (81.04/79.08).
### Patch design (merge with wpr lane)
- New file aw/stk/aw_adapt.py (pure torch: vote_share = comp_serial._vote over slots < n + winner share; ext_more; stop_early with
  symbol-equality unanimity; setup(spec) knobs; decide() replaces (do_rep, do_wait, do_fin, has, widx) when eng.aw_on).
- comp_serial.py: 3 hunks only (full diff: aw/aw_comp_serial.diff): (1) `NS = int(os.environ.get('SW_NS', 8))` + `import aw_adapt as AWA`;
  (2) sw_setup: `sw_maxn` env override SW_MAXN, `self.aw_on = False`; (3) in _sw at the voter line:
  `if self.aw_on: do_rep, do_wait, do_fin, has, widx = AWA.decide(self, B, end, valid, fl, rbad) else: <old _vote>`.
  None touch positions/attention, so they merge with WPR's patch (pos / q-k dims) textually.
- C (cap 400) = aw_adapt.setup sets eng.acap (python constant in the `capped` compare, graph-captured); item filter stays at 320.
- Run config: SW_NS=16 (forget E -> 64 automatically), SW_MAXN=7936, SAN_SMAX=8192, BMAX=128 (as adapt lane: STK_MAXN 7936, SAN_SMAX 8192, B 128).
- Also fastroll.py/c2_engine.py: first-token host-stop fix (= stack-lane engine (rel/v5)).
- 03:08Z NEW ORDER: merge WPR (lane wpr, rung 1 87.96/85.92) then A+B+C+S on top = rung 2. Cancelled queued j23306 (A+B+C w/o WPR, dev).
  j23303 (fuzz PASS 0 mismatches / 20k vote + 20k decide; all-off NS=8 bitwise vs out/val; A+B+C w/o WPR on dev2) left running as ablation.
  MERGED WPR verbatim into aw/stk/comp_serial.py (module WPR gate + _wpr_rot + SerialW._wpr_tab, sw_setup table build, off register +
  q/K R(-off) after qk_rope_kernel). `diff comp_serial.py.orig stk/comp_serial.py` = exactly WPR's lines + my 3 AW hunks (aw/aw_comp_serial.diff);
  aw/comp_serial.aw_only.py = AW-only version. Hook tag gets `_wpr` when WPR=1.
- 03:09Z SUBMITTED j23307 (job_awr2dev): `bash aw_job2.sh dev` = fuzz + WPR=1 SW_NS=16 SW_MAXN=7936 SAN_SMAX=8192 BMAX=128
  AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400, seeds 7,11,13,17 -> aw/out/r2_dev. NEXT: submit `aw_job2.sh dev2` (job_awr2dev2) when a slot frees
  (gsub blocks at 3 implant-lane GPU jobs), then LADDER rung 2.

## ladder-r2: WPR multi-run leak (sub-agent ladder-r2, 2026-11-21)
- 03:25Z ROOT CAUSE (also found independently by ladder-next and m5-div): `SerialW._wpr_ar = torch.arange(B_max+1)` has first
  dim B_max+1 -> fastroll `_f_rowregs()` moves it on compaction (`t[D]=t[S]`; aw/out/r2_dev/log.txt "fastroll compaction moves:
  [..., '_wpr_ar']"). After the first compaction `rar` is not arange: `Kc[rar,:,slot] = _wpr_rot(...)` rotates the wrong row's
  K and leaves the moved row's new K unrotated (q at R(-off), K at absolute phase). Never reset (unequal rows = "table" for
  fastroll) -> persists into the next generate() from step 0, and each compaction permutes more -> progressive collapse
  across seeds. Also hits the tail rows (live at the first compaction) of the FIRST run, incl. rung-1 fe rows (FAST_COMPACT=1).
  Not causes (checked): s_fr reset at admission (in _f_regs0), s_qlen written at admission, no compounding of the in-place
  rotation (qk_rope_kernel rewrites Kc[r,:,slot[r]] for every row each layer/step right before it), prefix cache untouched.
- FIX in aw/stk/comp_serial.py (WPR-only lines; WPR-off path untouched): `self._wpr_ar = torch.arange(B_max+1).view(1,-1)`,
  `rar = self._wpr_ar[0, :B]`. ladder-next: apply the same 2 lines in implants/ladnext/aw (you said you did).
  Fixed WPR-only patch for the stack lane: implants/wpr/wpr_fixed.diff; HANDOFF §4 updated.
- 03:16Z SUBMITTED j23319 (job_aw_r2fdev): `cd gsm_stack/implants/aw/dev_tree && <queue-submit> gsm_stack/implants/aw/dev_tree 2400 job_aw_r2fdev -- bash aw_job3.sh dev 1`
  aw_job3.sh SET [1] = test_aw fuzz; [OFF8 bitwise vs out/val -> out/off8f]; rung 2 4 seeds one process -> out/r2f_SET;
  rung 2 seed 11 alone fresh process -> out/r2f1_SET + `LEAKCHECK` (bitwise multi vs fresh); rung 1 (WPR only) 4 seeds -> out/r1f_SET.
  dev2 twin: `... 2400 job_aw_r2fdev2 -- bash aw_job3.sh dev2` (blocked in gsub at submit time).
  Old-code confirmation tree: implants/aw/confirm_old (aw copy with the fix reverted), `bash conf.sh` = rung-2 dev2 s11 alone
  (out/c_r2s11) + WPR-only dev2 s7,11 one process (out/c_w711).
- 03:50Z j23319 (fixed, dev) DONE: AW FUZZ PASS; OFF8 BITWISE 600/600 (acc 81.33). rung 2 dev 4 seeds ONE process:
  89.67/90.17/89.17/89.33 = 89.58 (tok 1196, natt 6.70, rep/q 0.76; old leaky 78.71, rep/q 3.64). rung 1 same tree:
  88.67/89.50/88.33/88.67 = 88.79 (tok 1229). rung 2 vs rung 1 fixes/breaks s7 +14/-8 s11 +10/-6 s13 +14/-9 s17 +11/-7 = +49/-30.
  LEAKCHECK seed11 multi-vs-fresh 2/600 bitwise (fresh 88.50 vs 90.17): NOT a WPR leak — the engine's graph warm-up
  (`engine._graph`: 2 eager steps per newly captured (B,S) bucket) advances `eng.seed` (not in `_mut()`), so the FIRST run in a
  process samples a shifted RNG stream at every first-time bucket capture; later runs reuse cached graphs. Harness-only, also
  present without WPR (fresh-vs-fresh is bitwise: OFF8 600/600). Added AW_PRECAP=1 (aw_hook.py, default off => unchanged):
  1-item warm generate + capture all buckets before the seed loop, so first and later runs sample identically; leak test
  aw_leak.sh / aw_leakjob.sh (rung-2 dev2 seed 11 after seed 7 vs fresh, WPR=1 and WPR=0) and the same on the OLD code in
  confirm_old/conf.sh (c). First seed in old code: r1 fe s7 vs r1 fixed s7 token-identical 503/600 (97 rows = those live at
  the first compaction differ; acc +3/-2) => within-seed admission path OK, only the post-compaction tail was hit.
  Analysis: `python3 implants/aw/ana_r2.py`. Queued: j23340 (dev2 aw_job3), j23346 (job_aw_conf, old code), leak job (gsub-blocked).
- 04:35Z ladder-r2 DONE. Jobs: j23319 (fixed, dev: fuzz + OFF8 + r2 + r2 s11 fresh + r1), j23340 (fixed, dev2: same w/o OFF8),
  j23346 (OLD code confirm, implants/aw/confirm_old: `bash conf.sh`), j23351 (fixed leak test: `bash aw_leakjob.sh`).
  * Old-code confirmation (dev2): rung-2 seed 11 ALONE fresh = 86.00 (rep/q 1.34) vs 76.33 as 2nd run in j23308;
    WPR alone (no adapt) s7,s11 in one process = 86.00 -> 76.67 (rep/q 1.04 -> 3.32) => leak is WPR's, not aw_adapt's.
    Leak test (AW_PRECAP=1, s11 after s7 vs fresh): OLD 45/300 bitwise (75.67 vs 86.67).
  * FIXED leak test: WPR=1 300/300 bitwise (87.00 = 87.00), WPR=0 control 300/300 (76.67). Also j23340's multi-run s11
    (no precap) is 300/300 bitwise equal to the precap fresh s11 => after the fix the only first-vs-later-run difference is the
    harness RNG shift from first-time graph capture (eng.seed not in _mut; affects WPR-off identically; samples are equally valid).
  * Checks: AW FUZZ PASS (both jobs); OFF8 BITWISE 600/600 (WPR-off path untouched).
  * RESULTS fixed aw tree, 4 seeds 7/11/13/17 in ONE process, u0 host, T=1:
    rung 2 (WPR + A+B+C+stop4): dev **89.58** (89.67/90.17/89.17/89.33) tok/q 1196 natt 6.70 rep/q 0.76
                                dev2 **86.92** (87.00/87.00/87.00/86.67) tok/q 1299 natt 7.05 rep/q 0.83
    rung 1 (WPR only, same tree): dev **88.79** (88.67/89.50/88.33/88.67) tok/q 1229 natt 8 rep/q 0.49
                                dev2 **85.92** (86.33/86.33/85.67/85.33) tok/q 1252 natt 8 rep/q 0.46
    rung 2 vs rung 1 fixes/breaks: dev s7 +14/-8 s11 +10/-6 s13 +14/-9 s17 +11/-7 = +49/-30 (+0.79);
                                   dev2 s7 +8/-6 s11 +6/-4 s13 +9/-5 s17 +10/-6 = +33/-21 (+1.00).
    rung 1 vs rung 0 (base): dev +224/-38, dev2 +109/-27. Old (leaky) r2 rows: dev 78.71 / dev2 76.33.
    rung 1 fixed vs fe rung 1 (old code, tail bug): dev 88.79 vs 87.96 (+54/-34), dev2 85.92 vs 85.92 (+35/-35).
  * Rows: aw/out/r2f_{dev,dev2}, r1f_{dev,dev2}, r2f1_* (fresh s11), lkw{0,1}_{m,f}; old: confirm_old/out/{c_r2s11,c_w711,lkold_*}.
  * HANDOFF (implants/wpr/HANDOFF.md §4) updated with root cause, 2-line fix, wpr_fixed.diff and these verification numbers.
