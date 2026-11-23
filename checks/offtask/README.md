# checks/offtask: off-task benchmarks after RL, implants attached

**Question.** Does RL with the implants hurt general skills more than plain RL does?

**Rows.** The rows are the blog §5.1 GSM8K rows: the same u300 checkpoints, RL seeds 0–2.
- **Qwen3-0.6B-Base** (`j06/`):
  - base;
  - P64 s0–2 (plain RL, row 2);
  - CW s0–2 (RL with the stack, row 4).
- **Qwen3-1.7B-Base** (`j17/`):
  - base17;
  - P17 s0–2;
  - CW17 s0–2.

**Benchmarks.** Five benchmarks, full test sets, 30,133 items per host:

| task | setup |
|---|---|
| MMLU | 5-shot (dev shots per subject), log-lik of " A".." D", 14,042 test items; item-weighted accuracy |
| ARC-Challenge | test, 1,172 items; acc_norm (log-lik per char) |
| HellaSwag | val, 10,042 items; acc_norm (lm-eval preprocessing) |
| WinoGrande | val, 1,267 items; partial scoring (lm-eval) |
| NQ-open | val, 3,610 items; 5-shot, greedy, max 24 tokens, stop at newline, EM with lm-eval normalisation. The implant write is live at every decode step |

lm-eval is not used. `ot_hook.py` reimplements these scoring rules.

## Final numbers

These are from `RESULTS.md`, which has the full tables, per-seed values and fire statistics. Brackets are 95% bootstrap intervals.

| 4 − 2 (RL with implants − plain RL) | MMLU | ARC-C | HellaSwag | WinoGrande | NQ-open |
|---|---|---|---|---|---|
| 0.6B | −0.14 [−0.38, +0.10] | **+1.14 [+0.26, +2.10]** | −0.11 [−0.29, +0.08] | −0.39 [−1.29, +0.50] | −0.09 [−0.46, +0.27] |
| 1.7B | −0.08 [−0.32, +0.14] | **−1.65 [−2.99, −0.31]** | **−0.23 [−0.43, −0.03]** | −0.21 [−1.16, +0.74] | +0.18 [−0.21, +0.56] |

- **RL vs base.** Every RL row is within 1.1 pt (0.6B) or 0.9 pt (1.7B) of its base model on every benchmark.
- **The implants never write on a scored position.** This holds for every item, every benchmark and both sizes. As a result:
  - rows 3 = 1, 2b = 2 and 4 = 4h exactly;
  - no multiple-choice decision changes;
  - every NQ generation is byte-identical.
- **Writes elsewhere in the prompt.** An implant writes somewhere in the prompt on 0.04% of MMLU items (arithmetic in the question).

## Files

| file | what |
|---|---|
| `ot_hook.py` | The eval hook. It is exec'd inside `dig_san/tw_dump_lean.py` the same way the final_eval `fe_hook.py` is. It scores every forward pass twice from one pass: host only (`_h`), and host + the rung-2 DualImplant write (calc_s4 + chain2; `_a`). It also runs and counts the serial layer's attempt-1 END trigger. |
| `j06/tree/`, `j17/tree/` | Code-tree overlays: TREE_BASE plus `stk/ot_hook.py`, a symlink to `../../../ot_hook.py`. The bases are `uniform/final_eval/trees/stk_r2q` (0.6B) and `uniform/rl28/gsm8k/trees/stk_r2q_17` (1.7B, with the 1.7B host plumbing). After the release path changes, the materialized trees are code-identical to the trees the original jobs ran. |
| `ot_job.sh SIZE ROW HOST [TASKS] [LIM]` | Runs one row on one GPU. It goes through `gsm_stack/run_hook.sh` (0.6B) or `uniform/rl28/gsm8k/fe/run_hook.sh` (1.7B), with BMAX=16 and SAN_SMAX=1024. |
| `ot_pack.sh SIZE [SMOKE=..] ROW:HOST ...` | Runs several rows concurrently on one GPU, with starts staggered by 40 s. |
| `dl.sh [DIR]` | Downloads the benchmark data from the HF Hub (parquet → jsonl). Needs `pip install pyarrow`; the original runs used 26.0.1. `bench.sha256` holds the checksums of the files the runs used. |
| `mk_items.py SRC DST [ROW..]` | Reduces the hook's per-item output to `DST/<ROW>.items.jsonl.gz`, dropping log-likelihoods and generated text. |
| `j06/res/`, `j17/res/` | The reduced per-item results of the 14 rows: {base, P64_s0..2, CW_s0..2} and {base17, P17_s0..2, CW17_s0..2}. Each file is about 140 KB. |
| `ana_ot.py` | Rebuilds the RESULTS tables. It writes the markdown to stdout and also writes `rows06.json` / `rows17.json`. |
| `ana06.md`, `ana17.md` | The output of `ana_ot.py`. |
| `rows06.json`, `rows17.json` | The row summaries written by `ana_ot.py`. |
| `RESULTS.md` | The write-up. |

## Rerun

**Tables from the shipped per-item results** (CPU, about 7 min per size; the bootstrap is pure Python):
```
cd checks/offtask
OT_ROWS=rows06.check.json python3 ana_ot.py > ana06.check.md;  diff ana06.check.md ana06.md; cmp rows06.check.json rows06.json
OT_SIZE=17 OT_ROWS=rows17.check.json python3 ana_ot.py > ana17.check.md;  diff ana17.check.md ana17.md
```
The bootstrap seeds are fixed, so the output is identical to `ana06.md` / `ana17.md` and `rows06.json` / `rows17.json`. This was checked on the release copy.

**Full rerun** (GPU). It needs:
- the u0 host `$CKPT_U0` (REPRODUCE §16.1);
- the RL arms: `$ARMDIR/rl/{P64,CW}_s{0,1,2}/step_000300` for 0.6B and `$ARMDIR17/rl/{P17,CW17}_s{0,1,2}/step_000300` for 1.7B (0.6B: `GPU=0 bash gsm_stack/engine/run_arm.sh P64_sK` / `CW_sK`, REPRODUCE §16.1; 1.7B: the rl28 arms P17_sK / CW17_sK shipped under `uniform/rl28/gsm8k/`);
- for 1.7B, `results/gsm_bigger/veto_q17.pt`.

```
bash checks/offtask/dl.sh                      # -> $WSM_WORK/offtask/bench (CPU, ~1 min)
export CUDA_VISIBLE_DEVICES=0
bash checks/offtask/ot_job.sh 06 base base     # ~8 min alone on a B200
bash checks/offtask/ot_pack.sh 06 P64_s0:P64_s0 P64_s1:P64_s1 P64_s2:P64_s2 CW_s0:CW_s0 CW_s1:CW_s1 CW_s2:CW_s2
bash checks/offtask/ot_pack.sh 17 base17:base P17_s0:P17_s0 P17_s1:P17_s1 P17_s2:P17_s2
bash checks/offtask/ot_pack.sh 17 CW17_s0:CW17_s0 CW17_s1:CW17_s1 CW17_s2:CW17_s2
python3 checks/offtask/mk_items.py $WSM_WORK/offtask/j06/ot checks/offtask/j06/res base P64_s0 P64_s1 P64_s2 CW_s0 CW_s1 CW_s2
python3 checks/offtask/mk_items.py $WSM_WORK/offtask/j17/ot checks/offtask/j17/res base17 P17_s0 P17_s1 P17_s2 CW17_s0 CW17_s1 CW17_s2
(cd checks/offtask && python3 ana_ot.py > ana06.md && OT_SIZE=17 python3 ana_ot.py > ana17.md)
```

**Notes on `mk_items.py`.** Pass the row list: smoke rows (`SMOKE=` in `ot_pack.sh`) write to the same directory. A rerun's per-item output can also be passed to `ana_ot.py` directly as `DIR`.

**Runtime.** Original runs: one job per GPU via a cluster queue, all on B200.

| run | time per row |
|---|---|
| 0.6B row, alone on the GPU | about 8 min (task time 423 s; NQ generation is the longest task) |
| 0.6B, 5 rows packed on one GPU | about 38 min |
| 1.7B, 3–4 rows packed on one GPU | about 30–33 min (task time about 1,800–2,000 s) |

The total is about 1 h for 0.6B and 1.2 h for 1.7B, on one B200.

**Seeds.** 3 RL seeds per arm. The eval itself is deterministic: log-likelihood scoring plus greedy NQ.

## Not shipped

- **Checkpoints.** Neither the RL arms nor the u0 host are shipped; they come from the RL reproduction.
- **Benchmark data.** It is fetched by `dl.sh` and is 20 MB.
- **Raw per-item rows.** These held the log-likelihoods of every option and the NQ generated text, 58 MB per size. They are replaced by the reduced `res/` files.
- **Vendored pyarrow.** It was vendored into a local folder; install it with `pip install pyarrow` instead.
- **Environment probe and logs.** The environment probe script and the job logs are not shipped.
- **Copies of the code trees.** These are now referenced as the `uniform/` overlays.
