# checks/seeds35: blog §5.1 uniform rows at 6 RL seeds

The blog §5.1 uniform rows (REPRODUCE §17) were first run at RL seeds 0-2. This folder extends the GSM8K and IFEval
rows to RL seeds 0-5 on Qwen3-0.6B-Base. BBH was not extended and stays at 3 seeds.

| file | what |
|---|---|
| `RESULTS.md` | GSM8K: rows 2 / 2b / 4, RL seeds 0-5; protocol, RL runs, runtimes |
| `IFEVAL.md` | IFEval: rows 1-4 + 2b, RL seeds 0-5; how seeds 3-5 ran, and the NLTK punkt_tab issue |
| `AUTO_final.md` | full GSM8K tables written by `ana35.py` (per seed, eval seeds 7/11/13, dev, writer audit) |
| `ana35.py` | GSM8K analysis (paired item bootstrap; same code as the §17 analysis) |
| `reduce_fe.py` | adds new `fe.sh` per-item outputs to `res/fe/` |
| `jobs.tsv`, `lists/` | the 6 GSM8K RL jobs and 15 eval jobs, and the 33 eval lists they ran |
| `ifeval_seed.sh` | one IFEval RL seed + its IFEval-541 eval on one GPU (the script seeds 3-5 ran) |
| `ifana/` | IFEval analysis: `analyze6.py` (table), `perseed.py` (per-seed facts), `nltk_check.py`, `reduce_ifeval.py` (new run dir → `res/` layout), `ifitems.py` (per-item loader), outputs `table_s0-5.md`, `rows_s0-5.json`, `check012.md`, `rows_s012.json`, `perseed.txt`, `nltk_check.md` |
| `res/fe/{test,hard,gsmplus}.items.jsonl.gz` | per-item GSM8K eval results of the 81 RL seeds 3-5 runs (rows 2 / 2b / 4 x eval seeds 7/11/13 x 3 sets): correct, generated tokens and the last answer token, no text. The 81 seeds 0-2 runs are the §17 runs; `ana35.py` reads them from `uniform/final_eval/items/` (via `uniform/items.py`) |
| `res/dev.json`, `res/fe_audit.txt` | in-run dev records of the 12 RL arms; the writer-audit line of each rung-2 eval |
| `res/ifeval/{P,I}_s0..5/` | per-item IFEval scores of seeds 3-5 (instruction ids, strict / loose per instruction, finish, tokens; no text; u0 and seeds 0-2 are read from `uniform/rl24/ifeval/res/` by `ifana/ifitems.py`), eval summary, RL steps (config, dev, per-update scalars), run facts |

## Final numbers (mean over RL seeds 0-5)

| | GSM8K test | GSM-Hard | GSM-Plus | IFEval-541 (prompt strict) |
|---|---|---|---|---|
| row 2 (RL) | 68.32 | 41.60 | 53.67 | 45.53 |
| row 2b (RL, then attach) | 82.22 | 61.42 | 68.34 | 53.70 |
| row 4 (RL with the attachment) | 80.44 | 59.01 | 66.09 | 48.49 |
| 4 − 2 | +12.12 [+10.77, +13.57] | +17.41 [+15.59, +19.27] | +12.42 [+11.05, +13.74] | +2.96 [−0.28, +6.35] |
| 2b − 2 | +13.90 [+12.53, +15.33] | +19.83 [+18.04, +21.67] | +14.67 [+13.41, +15.98] | +8.16 [+6.13, +10.38] |
| 4 − 2b | −1.78 [−2.50, −1.01] | −2.41 [−3.35, −1.38] | −2.25 [−3.04, −1.49] | −5.21 [−7.89, −2.46] |

GSM8K at eval seed 7 (the §17 protocol; seeds 7/11/13 in AUTO_final.md), paired item bootstrap with 2000 resamples.
IFEval with 10,000 resamples. On GSM8K every 3-seed claim holds at 6 seeds. On IFEval, 2b > 2 and 2b > 4 hold (6/6
seeds each), but 4 − 2 shrinks from +4.81 [+1.11, +8.56] at 3 seeds to +2.96 with a CI that includes 0 (5 of 6 seeds
positive).

## Recompute the tables (CPU, from `res/`)

```bash
cd checks/seeds35
python3 ana35.py --stamp 2026-11-23T10:40Z > AUTO_final.md                         # about 50 s; byte-identical to the shipped file
cd ifana
SEEDS=0,1,2,3,4,5 ROWSJ=rows_s0-5.json python3 analyze6.py --out table_s0-5.md    # about 1 min; = table_s0-5.md
SEEDS=0,1,2 ROWSJ=rows_s012.json python3 analyze6.py --out check012.md            # = the §17 IFEval table
python3 perseed.py > perseed.txt; python3 nltk_check.py > nltk_check.md
```

These read seeds 3-5 from `res/` and the seeds 0-2 / u0 per-item rows from the release's `uniform/final_eval/items/` and
`uniform/rl24/ifeval/res/` (no GPU, no setup; `WSM_ROOT` overrides the release root). All four outputs above were
regenerated from the shipped files and match the shipped outputs. The only difference is
in `perseed.txt`: the GPU label of seeds 0-2 now reads `B200` (an internal host tag was dropped from it).

## Rerun (one GPU per job)

Paths are relative to the release root. GSM8K uses `gsm_stack/` (setup in REPRODUCE §2) and the §17 GSM8K eval
harness in `uniform/final_eval/`. IFEval uses `uniform/rl24/ifeval/code`; `ifeval_seed.sh` first runs its `setup.sh`
(CPU, about 2 min, idempotent; needs `datasets` to fetch the IFEval prompts), which completes the code dir from its
release homes and regenerates the training data.

**GSM8K RL** (one B200 each; `ARMDIR` as in REPRODUCE §2; output `$ARMDIR/rl/<ARM>/step_000300`):

```bash
for K in 3 4 5; do GPU=0 bash gsm_stack/engine/run_arm.sh P64_s$K; GPU=0 bash gsm_stack/engine/run_arm.sh CW_s$K; done
```

- P64 takes about 15 min per seed and CW about 1 h 45 min.
- Seeds 0-2 are the same commands with K = 0, 1, 2 (REPRODUCE §17).

**GSM8K evals**: run each line of `jobs.tsv` from `uniform/final_eval/` after copying `checks/seeds35/lists/*.list`
there, with the same `ARMDIR`. For example, for seed 3:

```bash
bash fejob.sh stk s35_H_P64_s3.list                                                   # row 2, 9 runs, about 5 min
bash fepack.sh stk_r2q s35_r2_P64_s3_t.list s35_r2_P64_s3_h.list                       # row 2b, seed 7, about 17 min
bash fepack.sh stk_r2q s35_r2_CW_s3_t.list s35_r2_CW_s3_h.list                         # row 4, seed 7, about 17 min
bash fepack.sh stk_r2q s35_r2_CW_s3_e11.list s35_r2_CW_s3_e13.list s35_r2_CW_s3_gp.list # row 4, seeds 11/13 + GSM-Plus, 1-1.5 h
```

- The seeds 0-2 lists are the §17 lists in `uniform/final_eval/`.
- `fe.sh` writes `uniform/final_eval/out/fe/<SET>_<NAME>.jsonl` (`FE_OUT_DIR` overrides). To put new eval outputs into
  `res/fe/`: `python3 checks/seeds35/reduce_fe.py uniform/final_eval/out/fe P64_s3 r2_CW_s3 ...` (run names as in the
  lists), then rerun `ana35.py`. Runs in `res/fe/` take precedence over the `uniform/final_eval/items/` bundles, so
  reducing a rerun of seeds 0-2 into `res/fe/` replaces the shipped §17 rows in the analysis.

**IFEval RL + eval** (one GPU each; the scratch in `res_new/<run>/.work` holds a 7 GB resume state):

```bash
CUDA_VISIBLE_DEVICES=0 bash checks/seeds35/ifeval_seed.sh P 3     # also P 4, P 5, I 3, I 4, I 5
python3 checks/seeds35/ifana/reduce_ifeval.py checks/seeds35/res_new/P_s3 P_s3   # replaces res/ifeval/P_s3
```

- Runtime on one A100 40 GB: I about 3 h 15 min training + 8 min eval; P about 1 h 30 min + 4 min.
- Seeds 0-2 ran on B200 through `uniform/rl24/ifeval/rl_chain.sh {P,I}_sK` (P about 62 min, I about 1 h 53 min).
- Bit-exact reruns are not expected (bf16, batch scheduling). Expect seed-level scatter like the per-seed columns in
  IFEVAL.md.

## Where the runs ran

- GSM8K: all RL and eval jobs ran on single B200 GPUs.
- IFEval seeds 0-2: single B200 GPUs.
- IFEval seeds 3-5: single A100-SXM4-40GB GPUs (as recorded by `nvidia-smi` in each run). They used `ifeval_seed.sh`
  with B_max 128 / tok_budget 8192 and eval B_max 96, versus B_max 512|256 / tok_budget 32768 on B200. This changes
  batch scheduling only.

## Labels you will see

- Job labels such as `j25308` are bookkeeping only.
- AUTO_final.md and the list comments name earlier eval lists by their §17 names: `uni_p64`, `q34s2_P64`, `q32cd_h`,
  `q34gs_h`, `q34att_*`, `wpr_r2cw`, `q34s4_CW_s*`, `q34gs_CW_s*`, job `j23509`.
- `§12.1` in AUTO_final.md is the internal section number of the §17 protocol. `ana_q32.py` is the §17 analysis
  script that `ana35.py` copies. `stk`, `stk_r2`, `stk_r2q` are the eval stack builds in `uniform/final_eval/trees/`.

## Not shipped

- RL checkpoints and resume states (GSM8K and IFEval).
- Generated text: the per-item eval jsonls hold token ids and text, 221 MB for GSM8K alone. Only the per-item
  correctness / token counts used by the analysis ship.
- Run logs, job / watcher scripts, and the A100 run bundles.
- The offline IFEval rescore helper (`rescore.sh`). It re-scored the A100 evals with the same `score_reduce.py`; its
  per-run diffs ship as `res/ifeval/*/rescore_diff.txt`.
- Scratch probes of the IFEval setup.
