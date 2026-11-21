# lane wpr: weight-space RoPE position reset
## progress log (newest last)
- 02:44Z started. Reading BRIEF/task, stack-lane HOWTO_RUN_STACK.md, STATUS (reset), comp_serial SerialWD, gen/stk reset impl.
- 02:48Z FINDINGS: programmatic reset lives in comp_serial.SerialW._step_triton (rel/v4.2 line ~330):
  `off = where(s_fr>0, s_fr - s_qlen, 0); pos = Lp + slot - off` -> RoPE for q AND the new K-cache row use pos.
  Earlier attempts are masked by forget_ws, so the ONLY effect of the reset is the relative distance between the current
  segment's tokens and prefix/question keys (in-segment relative distances are unchanged; off is constant in a segment).
- DESIGN (ii, exact): rotary re-phasing writer. RoPE is a rotation, R(p-o) = R(-o) R(p). Keep the host's RoPE at the
  absolute slot position, then apply a fixed block-diagonal rotation R(-o) (cos/sin table row selected by an offset
  REGISTER s_off) to q and to the new K row of every token. s_off is set by the same newseg trigger that drives forget_ws
  (s_off := s_fr - s_qlen on newseg, else held) -> a counter register, fixed table, fixed linear map; no host decision.
  Exact up to fp rounding (q.k vs programmatic reset to be fuzzed). Env gate WPR=1; WPR=0 -> code path untouched.
  Diagnosis (step 1) is lower priority because (ii) is exact for every head (no need to pick heads).
- 02:58Z IMPLEMENTED: fe/ = copy of gsm_stack/eval (rel/v4.2 engine + fe_hook/fe.sh, same as aw lane). Patch in
  fe/stk/comp_serial.py, gate WPR=1 (WPR unset -> code path identical to base; off=0 rows bit-identical by construction):
  after HK.qk_rope_kernel per layer, q <- R(-off) q and K row <- R(-off) K, fp32 table _wpr_c/_wpr_s (built in sw_setup).
  off = where(s_fr>0, s_fr - s_qlen, 0) from registers (s_fr set by the newseg trigger, same as forget_ws).
  Fuzz: fe/stk/wpr_fuzz.py (fuzz.sh). Job A list fe/wpr_a.list (greedy 64-item equivalence + dev600 s7 N=8 wpr/pr1/pr0).
  Run: <queue-submit> gsm_stack/implants/wpr/fe 3600 job_wpr_a -- bash joba.sh ; results fe/out/fe/dev_*.jsonl
- 03:00Z SUBMITTED j23279 (job_wpr_a). Check: gq log j23279 --tail 50. Next: if wpr ~= pr1 on s7, submit 4 seeds (7,11,13,17) dev+dev2 for wpr (pr0 = aw lane rung 0).
- 02:49Z FUZZ (j23279): max|logit diff|/mean|logit| over 200x64 rows: question keys wpr-vs-reset 2.4e-2, wpr-vs-exact 2.2e-2,
  reset-vs-exact(bf16 RoPE own error) 2.5e-2; in-segment keys 3.3e-2 / 2.9e-2 / 2.4e-2 -> wpr is as close to exact as the
  programmatic reset itself (differences = bf16 rounding). off=0 rows bit-identical OK.
  greedy 64 dev items: pr1 82.81, wpr 81.25 (pr0 pending).
- 02:52Z greedy64: pr0 75.00, wpr 81.25, pr1 82.81. Submitting job B (wpr 4 seeds rest, fe/wpr_b.list) and C (pr1 ref, fe/wpr_c.list) via jobl.sh.
- 02:53Z RESULT dev600 N=8 T1 seed 7 (u0): WPR 88.50 (1243 tok/q) vs programmatic reset 88.33 (1221 tok/q); pr0 pending. job B (wpr 3 more dev seeds + 4 dev2 seeds) waiting in gsub (GPU cap 3).
- 02:56Z dev600 s7: pr0 81.83 (1075 tok/q) | WPR 88.50 | pr1 88.33. WPR vs pr0: +51 fixes / -11 breaks; WPR vs pr1: 14/13.
  greedy64: answers agree wpr/pr1 62/64 (pr0/pr1 51/64); token-identical 21/64; median common prefix 660 tok (pr0/pr1 143)
  -> WPR reproduces the programmatic reset up to bf16 rounding butterflies. job B = j23292 (wpr: dev s11,13,17 + dev2 s7..17).
  Next: job C (pr1 reference same seeds, fe/wpr_c.list) ; then LADDER rung 1 once 4 seeds done (pr0 rung 0 from aw lane j23275).
- 03:01Z job B partial: WPR dev s11 88.00, s13 88.50 (s7 88.50).
- 03:04Z DONE 4 seeds (7,11,13,17), N=8 T1 u0:  WPR dev 87.96 (88.50/88.00/88.50/86.83) dev2 85.92 (86.00/86.33/85.33/86.00), tok 1241/1261
  programmatic reset (pr1) same seeds: dev 88.46 (88.33/88.67/88.00/88.83) dev2 86.58 (84.67/87.67/86.00/88.00), tok 1232/1248
  rung 0 (aw, no reset): 81.04 / 79.08.  WPR recovers ~93% of the reset gain (+6.9/+6.8 of +7.4/+7.5); gap -0.5/-0.7 within
  seed noise (per-seed paired diffs -2.0..+1.3). Added LADDER.md rung 1.
  WHAT IS WEIGHTS: off register (from s_fr trigger register and s_qlen admission register), fixed rotary table, fixed
  per-row rotation map on q/K. Host Python: none decides (the KV slot index is the generator's write pointer, as for rung 0).
  Possible follow-ups: in-kernel fp32 composition (R(-off) before the bf16 cast) to cut the bf16 butterfly; integrate with the
  aw lane's comp_serial (patch = WPR block in _step_triton + _wpr_tab/_wpr_rot, ~25 lines) so A+B+C stacks on top.
