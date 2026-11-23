# rl28/ifeval: task [28] IFEval GRPO on Qwen3-1.7B-Base (memory file of rl24-ifeval)

> Release note: the original run log, scrubbed (cluster/queue details removed; job ids kept as labels only). Path map:
> `out/<arm>/*.score.json` -> `res/<arm>/*.score.json.gz`, `out/setup/` -> `res/setup/` (summaries, dev choice, check),
> `$ROOT/rl28/ifeval` -> `$RL28_DATA/ifeval`, code/ = `code/` here after `bash code/setup.sh`. Not shipped: generations,
> steps.jsonl training logs, dmon dumps, timesplit.py (reads steps.jsonl), launch_rest.sh (pack order, see pack_chain.sh).

Start: 2026-11-22T00:06Z. The [26] 0.6B work is finished (see ../../rl24/ifeval/RESULTS.md, rows.json).

## Brief (the coordinator, 23:5xZ)
- Rows: the same 1/2/3/4 plus 2b on Qwen3-1.7B-Base, with the same 14-rule implant. Only the host readers/writers are refit. Check idle identity.
- RL recipe as on 0.6B: lr 1e-6, 300 u, 64 prompts × G 8, T 1, chat format, max_new 1024. 3 seeds for rows 2 (P) and 4 (I).
- Engineering:
  - Baseline is the new rl_if.py (async resume save, resume_every 10).
  - Pack processes per job and pack the evals. Target ≥85% SM, with dmon per job.
- The table puts 0.6B next to 1.7B.

## Port: what is host-tied in the implant
The implant is f14/export.py make(model, vecs). Its host-tied parts:
- voc.E: a copy of the LM head.
- voc.g / eps: the final norm.
- force.F: the force_table over E/g, plus the force_fix rows.

force_fix is the only part that has to be solved. It is solved from the host's own plain answers to heldout/data/dev.jsonl (repo transfer recipe: run_t.py --mode plain, 2 shards, bs 24; then solve_forcefix.py). The 0.6B rows are 1024-d. For 1.7B they are 2048-d and have to be re-solved, giving vecs_q17.pt. The repo's vecs_q17.pt is not on disk, so it is rebuilt here with the same recipe; a matching sha is not expected.

The 0.6B choice was "snapshot": Qwen3-0.6B chat readout with vecs2. On dev it scored 41.7, against 34.2 for the Base refit.

For 1.7B I make both exports and pick the one with the higher dev prompt-strict, by the same rule as on 0.6B (decided in the setup job):
- sd_snap17 = mk_sd(Qwen/Qwen3-1.7B, vecs_q17)
- sd_base17 = mk_sd(Qwen/Qwen3-1.7B-Base, vecs_q17)

Code changes (code/ is a copy of rl24/ifeval/code):
- stk/fastqwen.py: host from env HOST_MODEL; shapes 0.6B (1024, 3072) and 1.7B (2048, 6144); config assert in load_hf_params.
- hostchk/mk_sd.py: VECS argument; asserts on the hidden size of E, g and F.
- eng/hostdev.py and check.py: --mem_frac (and --B_max for hostdev).
- eng/rl_if.py: disk guard. If /work free space after a save would fall below 250 GB, the resume save is skipped and logged.

The kernels (hostkern, ifengine, fasttrain, lmhead.cu, gexec) are shape-generic.

## Memory and packing plan
- 1.7B per-token KV is the same as 0.6B (28 layers, 8 kv heads × 128), so KV is 36 GB at B_max 256.
- Non-KV memory is estimated at ~42 GB (0.6B measured ~20 GB). That gives ~78 GB per arm at B_max 256. The GPU has ~179 GB, so the cap of 0.46 is ~82 GB.
- Packed jobs run 2 arms per GPU (pack_chain.sh: ARM_FRAC 0.46, BMAX 256, workers 6). On OOM the job retries in place with B_max halved.
- The setup smoke measures mem_gb first.

## Disk declaration ([30])
- Per running arm: resume state ~20 GB plus a .tmp during the write, so ≤41 GB transient. It is deleted at DONE.
- Kept per arm: step_000300 bf16, 3.4 GB. 6 arms ≈ 21 GB in total.
- sd exports ~2 × (E 2048 × 151936 bf16 + F) ≈ 2-3 GB. Setup HF answers < 10 MB.

## Jobs
- setup (1 GPU, packed lanes): code/eng/job_setup17.sh.
  - Fetch Qwen3-1.7B and check tokenizer/template identity.
  - force_fix on 1.7B, then the two exports.
  - Meanwhile, P smoke.
  - Then, concurrently: hostdev (none/snap17/base17) → CHOSEN, and the u0 541 evals (rows 1 and 3 plus idle identity) for both exports.
  - Then check.py exactness alongside the I smoke.
- pkA "P_s0 P_s1": P arms do not need the export until the u300 eval.
- After setup: pkB "I_s0 I_s1". After pkA: pkC "P_s2 I_s2".

## Log
- 2026-11-22T00:22:27Z: setup **j23827** (GPU 7) and pkA **j23828** (P_s0 + P_s1, GPU 3; detached driver shell rl28if_pkA, out/pkA/chain.log) submitted at 00:06Z.
  - Qwen3-1.7B fetched. Its vocab and chat template (thinking off) are identical to Qwen3-0.6B, and the same holds for 1.7B-Base.
  - force_fix re-solved on Qwen3-1.7B from 420 own dev answers (d 2048): 3134 force fails, **1190 fixed**. That is the same row count as the repo's vecs_q17. diag: 149700/151644 tokens win; sha ae7cb299...251a.
  - sd_snap17 and sd_base17 written (2.16 GB each; E (151936, 2048), F (151937, 2048)).
  - P smoke: 3 u OK at B_max 256 / cap 0.46.
  - pkA: about 20 s/u per arm, 2 arms packed; GPU at 100% in the progress samples; 160 GB used.
- 2026-11-22T00:42Z: **j23827 cancelled at 00:40Z: it was hung, not CPU-bound.**
  - Stage 1 (force_fix, exports, P smoke) finished at 00:17Z. The next line was a bare `wait`, which also waits on the job's own `nvidia-smi dmon` child, so it never returned.
  - GPU 7 sat idle for 23 min (0% SM, 0 MB).
  - dmon for j23827: 65.5% SM until 00:17:30, then 20.3% over the whole job.
  - Fixed job_setup17.sh to wait on explicit PIDs only. pack_chain.sh already does this.
  - Resubmitted as **j23834** (rl28if_setup2, 2 h limit). It skips the finished force_fix, exports and P smoke, so it runs GPU steps only: dev choice, u0 evals, check, I smoke. The CPU work left is about a minute (cached fetch check, pylib/tables), so no separate --cpu job is needed. Future setup-type CPU work goes to --cpu jobs, as the coordinator asked.
- P smoke (1.7B, B_max 256, cap 0.46):
  - roll 16-18 s, train 3.4 s (u1 5.6 s), update 21-22 s.
  - **mem_gb 82.8**, right at the cap. Packed pkA uses ~160 GB per GPU.
  - u1 TF ratio 0.99994.
- Disk: /work went 418 → 364 GB.
  - Mine: 2 sd exports (4.3 GB), plus pkA's 2 arms × (resume ~20 GB + .tmp ~20 GB while a save is written).
  - Kept per arm: only step_000300 and the resume state; the resume state is deleted at DONE.
  - Scratch (setup smoke dirs) is deleted in-job.
- 2026-11-22T01:00Z: **setup j23834 done (00:41-00:56Z, 84.1% SM over the job, 92.9% in gpumon).**
  - Dev choice on gen/dev 415, greedy 1024, prompt-strict:
    - none: 12.29
    - snap17: 43.61 (fired 85%)
    - **base17: 50.84** (fired 61%)
  - So **CHOSEN = base17**: on 1.7B the readout refit to Base wins. On 0.6B the snapshot won (41.7 vs 34.2).
  - u0 541 prompt-strict:
    - row 1 (plain): **17.19**, tok/q 1467
    - row 3 with base17: **35.12**, tok/q 1440, fired 186
    - snap17 would give 34.94, fired 257
  - Idle identity at u0: base17 **355/355**, snap17 284/284.
  - I smoke: 3 u OK; u2 reward 0.31, fired 73%; roll ~25 s.
  - check.py failed with rc=1 after the fuzz stages. Those passed: fuzz_padded 0 state/mem/widx mismatches; unpadded vs padded 0 widx mismatches out of 22539 live writes. The failure came while the fast engine was being built: the HF reference was still resident under the 0.45 cap (OOM). check.py now frees the reference first.
  - Rerunning the check alone as **j23838** (job_check17.sh, uncapped). pkB "I_s0 I_s1" goes in when j23838 ends (≤2 jobs).
  - out/u0/ (release: res/u0/) = the base17 u0 files, used by analyze.py.
- 2026-11-22T01:10Z: **check j23838 DONE (00:59-01:07Z, 76.3% SM).**
  - 1.7B fast engine vs HF + reference Executor, base17, 64 dev prompts, greedy 1024: **47/64 token-identical**. All 17 divergences are near-ties (|gap| ≤ 0.25 logit; 16 on the host gap). 0.6B had 82/128 identical with 45/46 near-ties.
  - fused vocab vs Executor: 0 / 205056 mismatches. Fuzz: 0 mismatches.
  - RL shape (512 rows): implant on 12.3-12.7k tok/s, implant off 25.9-27.3k tok/s. Fast greedy 5.1k tok/s.
  - So the port is exact up to bf16 near-ties.
- pkB **j23840** = I_s0 + I_s1 (driver launch_rest.sh, detached shell rl28if_launch, submitted 01:07Z). pkC "P_s2 I_s2" is started automatically by launch_rest.sh at pkA ALLDONE.
- pkA at 01:07Z: P_s0 u225 (dev 41.7, tok 149); P_s1 u255 (reward 0.41, len 100). About 3.3 u/min per arm; GPU 100%, 162 GB.
- 2026-11-22T01:22Z: **pkA j23828 ALLDONE at 01:17Z** (00:07-01:17Z; 2 P arms on 1 GPU; dmon **97.2% SM** over 422 samples).
  - Per update: wall 13.2 / 12.8 s; roll 10.4 / 10.1, train 2.2 / 2.1, reward 0.18, dev 0.55, save block 0.07, other -0.2.
  - Async resume write takes 98 s for 1.7B (20 GB), and 7 due saves were skipped because the previous write was still running. That means ≤20 u could be lost on a crash; the stall does not come back.
  - mem_gb 83.1 at B_max 256; u1 ratio 0.99994 / 1.0003.
  - u300 541 (base17 implant for 2b), prompt-strict:

    | arm | plain (row 2) | implant attached (row 2b) | tok/q plain / imp | idle identity |
    |---|---|---|---|---|
    | P_s0 | **61.92** | **71.90** | 375 / 411 | 370/370 |
    | P_s1 | **62.48** | **71.53** | 439 / 486 | 372/372 |

  - pkC **j23842** (P_s2 + I_s2) auto-submitted at 01:18Z. Running: pkB j23840 and pkC j23842.
  - pkB at u10: 35.9 s/u per arm, mem_gb 85.4, 165 GB on the GPU, 100%.
- 2026-11-22T01:40Z: **relaunch of rl24-ifeval (restart note 2).** Drivers still alive (launch_rest.sh pid 4144057, pack_chain pkB pid 4152074, pkC pid 21509); nothing resubmitted.
  - pkB j23840 (GPU 7): I_s0 u35, I_s1 u35 at 01:32Z; ~35 s/u per arm (roll ~26, train 7-11 s); 100% GPU, 166 GB. ETA u300 ~04:10Z, then u300 eval → ~04:30Z (timeout 05:07Z).
  - pkC j23842 (GPU 3): P_s2 u35 at 01:28Z (12.4 s/u, ETA u300 ~02:25Z + eval), I_s2 u10 (34 s/u packed; faster once alone). ETA ~04:00Z.
  - analyze.py works on partial data (rows 1, 2 (2 seeds), 3, 2b (2 seeds)). When all 6 DONE: `python3 analyze.py --out RESULTS_TABLE.md` (writes rows.json), then write RESULTS.md.
- 2026-11-22T02:55Z: pkC j23842: P_s2 finished u300 (+ its evals) by ~02:33Z (results come back with the job's out/ at job end); I_s2 alone at u225, ~12 s/u (GPU 48-100% samples, alone). pkB j23840: I_s0/I_s1 u200, ~23 s/u (len ~190). ETA pkC end ~03:15Z, pkB end ~03:40Z (+ evals).
- 2026-11-22T03:20Z: **pkC j23842 ALLDONE at 03:12Z** (01:18-03:12Z, dmon 89.9% SM over 677 samples; I_s2 ran alone 02:27-03:12).
  - P_s2 u300: plain **64.51**, + implant (2b) **73.57**, idle 370/370; 12.88 s/u wall (roll 10.2, train 2.0).
  - I_s2 u300: with implant (row 4) **75.60**, implant detached (new secondary row 4p) **60.26**, idle 369/369; 21.9 s/u wall (roll 15.8, train 5.1; 34-38 s/u while packed early, 12 s/u alone late).
  - analyze.py now also has row 4p (I hosts evaluated without the implant; 0.6B 4p computed read-only from rl24/ifeval/out (release: uniform/rl24/ifeval/res): 30.31, seeds 31.79/30.31/28.84) and rows.json set ifeval541_4p.
  - Partial: on 1.7B row 4 (1 seed) > 2b by 3.3 pp [0.1, 6.4] — the opposite of 0.6B. Waiting for pkB j23840 (I_s0, I_s1 at u250 03:12Z; ETA ~03:40Z + eval).
- 2026-11-22T03:50Z: **pkB j23840 ALLDONE at 03:40Z** (01:07-03:40Z, dmon **97.4% SM** over 912 samples). All 6 arms DONE; all drivers ended; no jobs of mine running.
  - I_s0 u300 74.86 (implant) / 59.33 (detached), idle 366/366; I_s1 73.57 / 59.15, idle 366/366; 29.4 / 29.3 s/u wall.
  - **FINAL table in RESULTS.md (+ RESULTS_TABLE.md, rows.json; sets ifeval541, ifeval541_inst, ifeval541_2b, ifeval541_4p).**
    1.7B prompt-strict: row 1 17.19 | 2 62.97 [59.33, 66.54] | 3 35.12 | 4 74.68 [71.35, 78.00] | 2b 72.34 [69.01, 75.72] | 4p 59.58.
    4-2 +11.71 [+8.19, +15.16]; 2b-4 -2.34 [-5.18, +0.43] (0.6B had 2b > 4 by +4.3); 4p-2 -3.39 (0.6B: -13.3).
  - Remaining: nothing. Disk: resume states removed at DONE by rl_if.py; kept step_000300 per arm (6 x 3.4 GB) + sd exports (4.3 GB).
