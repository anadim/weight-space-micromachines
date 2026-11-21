# Rung 3 all-weights stack: drop-in for the stack lane, from the implant lane (2026-11-21 ~08:35Z)

Rung 3 = rung 2 + Nmax 24 + the answer-group voter as registers (AW_VF=1).
dev600 / dev2, u0 host ckpt_a2s2, seeds 7/11/13/17: **90.29 / 87.33**, at 1401 / 1522 tok/q and 7.7 / 8.2 tries.
Rung 2 for comparison: 89.58 / 86.92. Rung 3 vs rung 2 fixes/breaks: 38/21 on dev, 27/22 on dev2; pooled 65/43, one-sided sign p = 0.021.

## Exact env (dev; from lad3_job.sh cfg 'avf')
    env SAN_SMAX=8192 BMAX=128 bash aw.sh avf_dev AW_SETS=dev AW_SEEDS=7,11,13,17 \
        SW_NS=24 SW_MAXN=7936 AW_ADAPT=A=0.7/24/4:S=4/0.5:B=3:C=400 WPR=1 AW_VF=1
- `lad3_job.sh` sets the triton ptxas path and runs the test_aw fuzz first.
- The VF fuzz is `python3 test_vf.py`.
- Weights file: stk/vf_w.pt (ClusterLin θ, trained on trainC by the implant lane's lane vfeat).

## Files (snapshot of gsm_stack/implants/ladr3/aw)
- stk/comp_serial.py: rung-2 comp_serial plus the AW_VF hooks.
- stk/aw_vf.py: voter.
- stk/numreg.py: number-set register automaton.
- stk/fastroll.py: includes VF admit/harvest. It also has the **w_WK/w_WQ compaction fix**:
  - Without it, any run with SW_NS 32–63 at BMAX=128 is corrupted (forget-mask weights get permuted).
  - Runs with SW_NS ≤ 24 are unaffected.
  - The same 1-line fix is applied to ../rung2/stk/fastroll.py.
- For TEST / GSM-Hard through eval/fe_hook.py:
  - Follow ../rung2/README.md.
  - Copy comp_serial.py, aw_adapt.py, aw_vf.py, numreg.py, fastroll.py and vf_w.pt.
  - Add env `SW_NS=24 AW_ADAPT=A=0.7/24/4:S=4/0.5:B=3:C=400 AW_VF=1`.
  - Check how aw_hook.py reads the VF pick (field names) if your row writer needs it.

## What is weights
Rung 2 (WPR offset register + fixed rotary table; compiled A/S comparator; B/C counters) plus:
- a number-set register automaton: fixed token class/digit tables, a DFA gate up to '####', hashed number-set registers;
- the question number set, read at admission;
- an attempt-length counter;
- the final writer: ClusterLin over symbol-equality groups with fixed θ.

A/S/B/C still read the base voter. No host-side decision is made.
