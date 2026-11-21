# WPR hand-off to the stack lane (task [7]) — weight-space position reset (rotary re-phasing writer)

Status 2026-11-21 04:10Z: **FIXED (use wpr_fixed.diff).** The multi-run leak (§4) was a 2-line indexing bug in WPR, now fixed and verified on dev (4 runs in one process). Safe to attach to RL with the fixed patch.

## 1. What it is (all weights, no host decision)
- Offset register `off = where(s_fr > 0, s_fr - s_qlen, 0)`: `s_fr` is set by the stack's own new-segment trigger (same register forget_ws uses), `s_qlen` is the admission register.
- A fixed fp32 rotary table `_wpr_c/_wpr_s` (rows 0..Lp+S_max+8, built once from Qwen3 THETA/HD).
- In every layer, after the host RoPE at the absolute slot position: `q <- R(-off) q` and the new K-cache row `K[slot] <- R(-off) K[slot]` (a fixed per-row linear map). Since R(p-o) = R(-o) R(p), each attempt's tokens see the question keys at the same relative distance as attempt 0 (the effect of the programmatic reset), while earlier attempts stay hidden by forget_ws.
- Host Python makes no per-row decision (the KV slot index is the generator's write pointer, as in the base stack).

## 2. Files and flags
- Patch (~25 lines) against `gsm_stack/eval/trees/v42/stk/comp_serial.py` (= rel/v4.2 comp_serial): `diff gsm_stack/eval/trees/v42/stk/comp_serial.py gsm_stack/implants/wpr/fe/stk/comp_serial.py`. Hunks:
  1. module: `WPR = os.environ.get('WPR','0') == '1'` and `_wpr_rot(x, c, s)`;
  2. `SerialW._wpr_tab(off)` (table build + row gather);
  3. `sw_setup`: `if WPR: self._wpr_c = None; self._wpr_tab(...)`;
  4. `_step_triton`: compute `off`, table rows, and after `HK.qk_rope_kernel` apply `_wpr_rot` to q and the new K row.
- Env: `WPR=1` (off by default; with WPR unset the code path is identical to base). Leave `posreset`/`RL_POSRESET` OFF (WPR is skipped when the programmatic reset is on).
- Merged copy (with the aw adaptive-N hunks): `gsm_stack/implants/aw/dev_tree/stk/comp_serial.py`.

## 3. Exactness evidence
- Fuzz (`implants/wpr/wpr_fuzz.py`, job j23279): max |logit diff| / mean |logit| over 200x64 rows — question keys WPR vs programmatic reset 2.4e-2, WPR vs exact fp64 2.2e-2, programmatic reset vs exact 2.5e-2 (bf16 RoPE's own error). off=0 rows bit-identical.
- Greedy 64 dev items: no reset 75.00, WPR 81.25, programmatic reset 82.81; answers WPR vs reset agree 62/64.
- dev600/dev2, u0 host, N=8, T=1, seeds 7/11/13/17, ONE PROCESS PER SEED (fe tree, jobs j23279/j23292/j23293):
  WPR dev 87.96, dev2 85.92 | programmatic reset 88.46 / 86.58 | no reset 81.04 / 79.08. dev seed 7: WPR vs reset 14 fixes / 13 breaks (equivalent up to bf16 butterflies).

## 4. Multi-run state leak: ROOT CAUSE + FIX (ladder-r2, 2026-11-21 03:30Z)
- Symptom: aw tree, WPR=1, 4 seeds sequentially in ONE process (j23307/j23308): first run fine (dev 88.67), later runs 73-78
  with repairs/q 1.1 -> 5.0. WPR-off stable in the same layout.
- ROOT CAUSE: the WPR batch-row index `self._wpr_ar = torch.arange(B_max + 1)` has first dim B_max+1, so fastroll's
  compaction (`_f_rowregs()` = every tensor with first dim B_max/B_max+1 except out_*, w_Kx, w_ar) treats it as a per-row
  register and MOVES it (`t[D] = t[S]`) when the queue drains and live rows are packed into a smaller bucket. log.txt:
  `fastroll compaction moves: [..., '_wpr_ar']`. After that `rar = _wpr_ar[:B]` is no longer arange, so
  `Kc[rar, :, slot] = _wpr_rot(Kc[rar, :, slot])` rotates a WRONG batch row's K and leaves the moved row's new K unrotated
  (q rotated by R(-off), K at absolute phase => wrong relative positions for that row from then on). The permuted index is
  a "table" for fastroll (rows unequal, never reset) and survives into the next generate(): the next run is corrupted from
  step 0 for every row whose slot index got permuted, and each further compaction permutes more => progressive collapse.
  It also hit the TAIL rows of the first run (rows still live after the first compaction), incl. the rung-1 fe rows
  (FAST_COMPACT=1, one process per seed). Anything else that moves/gathers rows by first-dim shape would do the same.
- Checked and NOT a cause: `s_fr` is in fastroll's admission reset set (all -1 after the wave-start reset => reset at every
  admission); `s_qlen` is written at admission; the in-place K rotation does not compound on idle/done/padded rows or rows
  whose slot does not advance, because `HK.qk_rope_kernel` rewrites `Kc[r, :, slot[r]]` unconditionally for every row
  0..B-1 each layer each step right before the rotation (rotation applied exactly once per fresh write); parked rows'
  offsets are unused garbage but in table range. The prefix cache (Kp/Vp) is a separate tensor never touched by WPR.
- FIX (2 lines; WPR-off path untouched): make the index not look like a per-row register:
  ```
  -            self._wpr_ar = torch.arange(self.B_max + 1, device=self.dev)
  +            self._wpr_ar = torch.arange(self.B_max + 1, device=self.dev).view(1, -1)
  -            rar = self._wpr_ar[:B]
  +            rar = self._wpr_ar[0, :B]
  ```
  Full fixed WPR-only patch against `gsm_stack/eval/trees/v42/stk/comp_serial.py`: `gsm_stack/implants/wpr/wpr_fixed.diff`
  (fixed file: `gsm_stack/implants/wpr/comp_serial.wpr_fixed.py`). Merged copy with aw hunks: `gsm_stack/implants/aw/dev_tree/stk/comp_serial.py`.
- Multi-run verification (j23319, fixed code, dev600, 4 seeds 7/11/13/17 in ONE process): WPR-only 88.67/89.50/88.33/88.67 = **88.79**
  (higher than the old buggy-tail rung 1, 87.96, and than the programmatic reset, 88.46); WPR + adaptive-N/stop4/B/C 89.67/90.17/89.17/89.33 = 89.58;
  repairs/q stable (0.67-0.82; leaky run 1.1 -> 5.0). WPR-off path bitwise 600/600 vs base.
- Residual multi-run vs fresh-process difference (2/600 bitwise; accuracy within noise) is NOT WPR: the engine's graph warm-up
  (2 eager steps per newly captured (B,S) bucket) advances eng.seed, so the first run in a process samples a shifted RNG stream.
  Harness-only, also present without WPR. For RL this only shifts RNG; no effect on the model or implant.
- Bitwise leak test with all graphs pre-captured (AW_PRECAP=1): dev2 seed 11 run after seed 7 vs fresh process = 300/300 bitwise
  (WPR on and off); old code 45/300. dev2 fixed-tree 4 seeds: WPR-only 85.92, WPR+adapt 86.92.
- 04:35Z full verification (ladder-r2; jobs j23340 dev2, j23351 leak test, j23346 old-code control):
  * dev2, 4 seeds in ONE process: WPR-only 86.33/86.33/85.67/85.33 = **85.92** (rep/q ~0.46); WPR + A+B+C+stop4
    87.00/87.00/87.00/86.67 = **86.92** (rep/q ~0.83). Old code, WPR only, s7 then s11 in one process: 86.00 -> 76.67.
  * Leak test (rung-2 config, dev2, all graph buckets pre-captured with aw_hook AW_PRECAP=1 so the RNG stream is identical;
    seed 11 run after seed 7 vs seed 11 alone in a fresh process): FIXED **300/300 rows bitwise identical** (87.00 = 87.00);
    WPR-off control 300/300; OLD code 45/300 (75.67 vs 86.67). Also the fixed multi-run s11 without precap (j23340) is 300/300
    bitwise equal to the precap fresh s11 => after the fix there is no state carried between runs.
  * RL note: to make rollouts bitwise reproducible run-to-run, capture all buckets before the first rollout (AW_PRECAP in
    gsm_stack/implants/aw/dev_tree/stk/aw_hook.py); otherwise harmless.
