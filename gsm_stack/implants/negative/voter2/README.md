# voter2: verifier-v2 voter for the S3-serial stack (impl-voter2)

## Design
- **Verifier v2 (in-stream, read-only)**: `v2 = w_hash · [h14,h18,h22,h27]('####') + w_mnl · mean_{line ends of attempt}[h14,h18,h22,h27] + b`
  (= probe `hash_mnl_c4`). The hidden states are read at the layer inputs (`x` after the ln1 fused add) and are never modified.
  The line-end register is a running sum and count of the dot with `w_mnl`, which by linearity equals the dot with the mean. It is reset at each
  attempt start (wait or repair), and only counts model-written tokens containing `'\n'` before the `####`.
  The value is committed to register `R_v2[slot]` with the other per-attempt registers.
- **Voter**: each kept slot (sane, else valid) gets weight `w = sigmoid(1.5·z)`, where
  `z = 0.3·(clip(lg,-10)-1.287)/1.528 + 0.7·(clip(v2,-12)-2.006)/1.634`. The two terms are the L18 register logit and v2, standardised with trainC stats.
  Weights are summed per distinct answer, and the argmax is taken with ties going to the earliest slot.
  The base voter (sigmoid(lg)) is kept as `pick0` for logging.
- **Gating**: `SER_V2=1` plus `SER_V2_W=<pt>`. With the gate off the code path is unchanged (see Exactness below).
  Weight file: `job/stk/voter2_w.pt` (`wh, wm, b, th, b0, kind='sig', src`).
- Code: `job/stk/serial_stack.py` (setup, `_read` line-end flag, `_reset_reader`, `_act` commit and voter, layer-loop reads), `job/stk/gen_hook.py`
  (`_v2` output tag, `v2`/`pick0` fields), `job/run_v2.sh`.

## Weights provenance (trainC only)
- Probe: `probe/out/pred_trainC/probes.pt` → `hash/hash_mnl_c4`, a full fit on trainC (split C train, disjoint from dev/dev2), with l2=1 and b=-3.2435.
  Held-out AUROC: dev 0.854, dev2 0.876 (trainC OOF 0.856). For comparison the register lg scores dev 0.824 and dev2 0.837.
- Blend constants follow the notes/probe.md §7 spec (trainC stats). The voter's scale and shape (sigmoid ×1.5) were chosen by trainC vote accuracy
  (`grid.py`, best of 10 one-parameter families). dev and dev2 were never used for selection.

## Offline estimate (trainC-chosen; vote acc %, mean over 4 seed streams; `offline.py`, `grid.py`)
| voter | trainC | dev | dev2 |
|---|---|---|---|
| stack sigmoid(lg) (base) | 87.12 | 88.50 | 85.75 |
| count | 86.12 | 87.58 | 85.58 |
| sigmoid(v2-1) | 87.32 | 88.38 | 86.25 |
| exp(1·blend) | 87.26 | 88.38 | 86.25 |
| **sigmoid(1.5·blend) (built)** | **87.39** | 88.50 | 86.08 |
| listwise exp(th·[v2,lg,sane]) (learn.py-style) | — | 88.25 | 85.50 |
| listwise +ntok+nsteps | — | 88.67 | 86.08 |

**The offline gain is below +0.3 on both sets** (dev about 0, dev2 +0.3). The trainC listwise fit even puts negative weight on v2,
because the L18 register lg looks optimistic on trainC (AUROC 0.86 there vs 0.82 on dev). Better AUROC (0.83 to 0.86) does not move
selection much, because in the gap cases the vote winner is usually a wrong majority with a single right attempt. The best trainC variant was built anyway.

## Tests
- `test_voter2.py` (CPU fuzz, stubs triton/fastqwen/engine and drives the real `SerialEngine._act`): random registers (values with
  collisions, valid/bad/sane flags, lg/v2 incl. -30 sentinels). Compiled pick vs python reference: 4096 rows (2048 exp-weight + 2048 sigmoid-weight), 0 mismatches for the v2 pick, the base pick (pick0) and the queued `#### X` tokens. **PASS**
- Smoke (GPU, 64 dev items, seed 7): SER_V2 off vs on, tokens before the final queue identical 64/64; gate off vs gen/out base fully identical 64/64.
- In-stream v2 vs offline probe on teacher-forced states (`test_voter2.py --runs`, rows whose tokens equal the base run): corr 0.9996–0.9997 for all 4 set×seed, MAE 0.035, sign agreement 0.996–0.999 (n = 1011/1020 dev, 2390/2388 dev2). The residual is bf16 noise from batched vs teacher-forced forward passes.

## Results (full runs, N=8, T=1, `bash run_v2.sh 7,11 dev,dev2`)
Same-stream = base voter (`pick0`, sigmoid(lg), computed in the same run) vs v2 pick on identical tokens. This is the paired comparison.
"Base run" = gen/out files.

| set | seed | prefix identical (all) | wave 0 | waves >0 | pick0 == base pick (aligned) | base run acc | v2 run acc | fixes / breaks vs base run | same-stream pick0 acc | same-stream v2 acc | same-stream fixes / breaks | tok/q base | tok/q v2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | 7 | 129/600 | 128/128 | 1/472 | 129/129 | 88.50 | 88.00 | 11 / 14 | 87.83 | 88.00 | 6 / 5 | 1218.0 | 1218.2 |
| dev | 11 | 129/600 | 128/128 | 1/472 | 129/129 | 88.33 | 88.67 | 13 / 11 | 88.67 | 88.67 | 3 / 3 | 1235.6 | 1225.6 |
| dev2 | 7 | 300/300 | 128/128 | 172/172 | 300/300 | 85.33 | 85.00 | 2 / 3 | 85.33 | 85.00 | 2 / 3 | 1232.5 | 1232.5 |
| dev2 | 11 | 300/300 | 128/128 | 172/172 | 300/300 | 85.33 | 86.00 | 2 / 0 | 85.33 | 86.00 | 2 / 0 | 1254.5 | 1254.5 |

Mean same-stream gain: dev +0.08, dev2 +0.17. Over 1800 item-runs that is 13 fixes vs 11 breaks, which is noise and in line with the offline estimate (< +0.3).
Job j23192 (12.5 min on one B200). Seeds 13 and 17 were not run (time budget).

### Exactness
- Every row whose wave is RNG-aligned with the base run is bit-identical before the final `#### X` line: 1458/1458, i.e. dev2 600/600 and dev wave 0 256/256, plus 2 later dev rows.
  For those rows, pick0 equals the base run's pick (1458/1458). Gate off vs the base gen/out file was also identical (smoke, 64/64).
- dev waves 1–4 (idx ≥ 128) diverge. This is a harness property, not a voter read. The sampler seed is one global per-decode-step counter (`self.seed += 1`, set once per seed run),
  and a wave runs until its longest row ends. In dev wave 0 the v2 vote changed the longest row's final line by one token
  (max len 4367→4366 for seed 7, 4719→4718 for seed 11), so every later wave samples with seeds shifted by one step.
  In dev2 no wave-length changed, so all 300 rows match. The comparison on dev waves ≥1 against gen/out is therefore between two independent
  samples of the same stack, and the paired same-stream columns are the clean voter comparison.
  Fully run-identical tokens would need a per-wave or per-row seed in the base harness, which would change the base stack and was not done.

## Files
`job/` (run copy; `run_v2.sh`, `stk/voter2_w.pt`, `out/*_v2.jsonl.gz`), `test_voter2.py` (fuzz + `--runs`), `analyze.py` (table above),
`offline.py` / `grid.py` (offline design selection; `offline_trainC.json`).

## Verdict
Fully implemented and verified. The voter is exact, the in-stream v2 logit matches the offline probe, and tokens are unchanged. The accuracy gain is negligible:
dev +0.1 and dev2 +0.2 same-stream, offline < +0.3. A probe at AUROC 0.85–0.88 is not enough to beat sigmoid(lg) weighting when the
gap cases are wrong majorities. The real probe falls well short of voter/cache/ceiling.txt (simulated AUROC 0.85 → ~91), probably because its errors are correlated within an item (not verified).
