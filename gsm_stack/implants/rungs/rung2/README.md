# Rung 2 all-weights stack: drop-in for the stack lane (stack-lane task [17](2)), from the implant lane, 2026-11-21 07:15Z

Rung 2 = WPR (weight-space position reset) + adaptive tries A/S (compiled comparator) + B (repair counter) + C (cap 400).
dev600 / dev2, u0 host ckpt_a2s2, 4 seeds (7, 11, 13, 17): **89.58 / 86.92**, at 6.7 / 7.05 tries and 1196 / 1299 tok/q.
Rung 1f is WPR only: 88.79 / 85.92.

## Files (snapshot of gsm_stack/implants/aw/dev_tree; *.pt and __pycache__ excluded)
- stk/comp_serial.py: rel/v4.2 comp_serial + the fixed WPR hunks + the aw_adapt hooks (11 diff lines vs eval/trees/v42/stk beyond WPR).
- stk/aw_adapt.py: A/S/B/C as torch.where/ReLU on the registers. CPU fuzz: `STK_DIR=$PWD/stk python3 test_aw.py`.
- stk/fastroll.py: one change vs eval. In the serial all-weights stack, only the implant eos or the budget stops a row
  (the engine (rel/v5) fix).
- stk/aw_hook.py: the dev-eval hook (AW_* env). It writes rows in the implant-lane gen/out format. dev/dev2 only.
- aw.sh: the launcher, with the same env as eval/fe.sh. *.pt are copied or linked from $WSM_DATA/gsm_final/code
  and gsm_sel/code; caches go under $WSM_DATA/implants (change to your own dirs).
- aw_job3.sh: the exact job that produced the ladder numbers (fuzz, then rung 2 on 4 seeds, a leak check, then rung 1f).

## Exact rung-2 env (dev command, from aw_job3.sh)
    env SAN_SMAX=8192 BMAX=128 bash aw.sh r2f_dev AW_SETS=dev AW_SEEDS=7,11,13,17 \
        SW_NS=16 SW_MAXN=7936 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400 WPR=1
Rung 1f: the same command with only `WPR=1` (no SW_NS/SW_MAXN/AW_ADAPT; default SAN_SMAX/BMAX are fine).

## For TEST / GSM-Hard via your eval/fe_hook.py (aw_hook only knows dev/dev2)
1. Copy stk/comp_serial.py, stk/aw_adapt.py and stk/fastroll.py over a copy of eval/trees/v42/stk.
2. In fe_hook.py, right after `eng.sw_setup(...)` (line ~52), add:
       import aw_adapt as AWA; AWA.setup(eng, E.get('AW_ADAPT', ''))
   With an empty spec, setup is a no-op and the stack is the rung-0 path (bitwise 600/600).
3. Run FE_KIND=S, FE_N=8, FE_PR=0 with env `WPR=1 SW_NS=16 SW_MAXN=7936 SAN_SMAX=8192 AW_ADAPT=A=0.7/16/4:S=4/0.5:B=3:C=400`
   and --B_max 128. The register needs NS ≥ Nmax=16, so SW_NS=16. SW_MAXN/SAN_SMAX give room for 16 tries.
4. Run one process per seed if you can. Multi-seed processes are bitwise-checked (leak fix), but graph warm-up shifts
   the RNG of the first run (harness-only, also without WPR).

## What is weights (no host-side decision)
Offset register + fixed rotary table (WPR). Comparator `relu(tau*total - winner)` on the voter's score registers
decides extend vs final-writer. Stop-at-4 is a unanimity/probe compare. Counters s_rep (≤3 repairs for non-integer
answers) and s_c (cap 400) are register compares. The host only runs the decode loop, the T=1 sampler and the token budget.
