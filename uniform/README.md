# uniform/: the uniform 4-row comparison (blog §5.1 on Qwen3-0.6B-Base, blog §6 on Qwen3-1.7B-Base)

The same four rows for GSM8K, IFEval and BIG-Bench Hard: (1) host, (2) host + RL, (3) host + implants, (4) host +
implants + RL with the implants attached. There are also the controls, ablations and fresh-set checks around them.
Commands, expected numbers, seeds, runtimes and GPUs are in **REPRODUCE.md §17** (Qwen3-0.6B-Base), **§18**
(Qwen3-1.7B-Base) and **§19** (settings table). This folder holds the launchers, eval lists, code overlays, analysis
scripts and the reduced per-item results that every table there is computed from. It ships no checkpoints and no
generated text.

## Layout

| path | what | REPRODUCE |
|---|---|---|
| `env.sh` | Sources `gsm_stack/env.sh` (`WSM_ROOT`, `WSM_DATA`, `WSM_WORK`, `CKPT_U0`, `ARMDIR`, `WSM_VETO`, `gs_tree`) and `rl/gsm_grpo/env_local.sh`, then sets `FT2_HOST`, `ARMDIR17`, `RL24_DATA`, `RL28_DATA`, `BBH_DIR`, `BBH_COT` (defaults in its header) | all |
| `items.py` | Reader and packer for the reduced GSM8K / GSM-Hard / GSM-Plus per-item rows (`*.items.jsonl.gz`). `$FE_ROWS` puts your own full rows first | §17, §18 |
| `final_eval/` | GSM8K eval harness for the 0.6B rows: `fe.sh` (list runner), `fejob.sh` (fuzz + run), `fepack.sh` (several lists on one GPU), `r2cwjob.sh` (row 4), `ana_uni.py` (2x2 effects), `q32/` (`ana_q32.py`: majority vote, eval seeds, GSM-Plus, [34] fills; `bbh_pertask.py`; `mech_2b.py`), `q33/` (row-3 decomposition), `data/prepare_gsmplus.py`, `lists/`, `items/` (reduced rows) | §17.1-§17.6 |
| `final_eval/trees/{stk,stk_r2,stk_r2q,stk_r3,stk_r4}` | Code trees as overlays on the `gsm_stack` trees (`TREE_BASE` names the base; `gsm_stack/mk_tree.py` builds them) | §17.2-§17.6 |
| `gsm_stack/engine/run_arm.sh` (outside this folder) | The uniform GSM8K arms `P64_sK` (row 2) and `CW_sK` (row 4), plus `A1WL_sK`, `A1WD_sK`, `LRP_/LRC_/LRQ_/LRD_<lr>` | §17.2 |
| `rl24/` | 0.6B: `HOST.md` (host decision), `make_uniform_json.py` -> `uniform.json` (figure data, 3 panels) | §17 |
| `rl24/ifeval/` | 0.6B IFEval rows: `code/` (engine, RL, eval; complete it with `code/setup.sh`), `rl_chain.sh`, `pack_chain.sh`, `analyze.py` / `analyze29.py`, `res/` (per-prompt scores, no text), `RESULTS*.md`, `rows.json` | §17.7 |
| `rl24/bbh/` | 0.6B BBH rows: `w/` (`job.sh`, `chain.sh`, `final_eval.sh`, `pack.sh`, `final_ana.sh`, trainer / engine / data sources), `cg/job_refit.sh` + `cg/code/` (call-gate refit), `res/` (reduced per-item rows), `table_final.md`, `rows.json`, `RESULTS.md` | §17.8 |
| `rl28/gsm8k/` | 1.7B GSM8K: `eng/` (`run_arm_17.sh`, `pack_17.sh`), `fe/` (`fe.sh`, `jobE4.sh`, `run_hook.sh`, `lists/`), `fe_gp/` (`ana_gp17.py`, `ana_v17.py`), `trees/{arm17,stk_r2q_17}` (overlays with the 1.7B refits), `items/`, `evchain.sh`, `q32chain.sh`, `mk32.py`, `mkrows.py`, `mkresults.py`, `ana17.py`, `patch17.py`, `RESULTS*.md`, `NOTES.md`, `rows.json` | §18.1 |
| `rl28/ifeval/` | 1.7B IFEval: `code/` (the 1.7B changes; `code/setup.sh` completes it from `rl24/ifeval/code`), `envs/`, `pack_chain.sh`, `rl_chain.sh`, `analyze.py`, `res/`, `RESULTS*.md`, `NOTES.md`, `rows.json` | §18.2 |
| `rl28/bbh/` | 1.7B BBH: `setup.sh` / `setup2.sh` (call-FT `ft17`, gate refit, rows 0 / 1 / 3, checks), `w/` (`job.sh`, `pack.sh`, `drive.sh`, `rl_bbh.py`, `bbhtrain.py`, `fastqwen.py`, `analyze.py`, `final_ana.sh`, `pertask.py`, `cmp_probe.py`), `cg/job_refit.sh`, `res/`, `table_final.md`, `rows.json`, `RESULTS.md`, `NOTES.md` | §18.3 |

## Running

Every launcher sources `uniform/env.sh` itself and uses one GPU (`GPU=k` or `CUDA_VISIBLE_DEVICES=k`). Set `WSM_DATA`
(runs, checkpoints, caches) before you start, and build the u0 host first (REPRODUCE §16.1). Every GSM8K arm and stack
reads its implant wiring from `$CKPT_U0`, including the base-host arms and the 1.7B arms.

**A GSM8K list (0.6B).** `bash uniform/final_eval/fe.sh LIST` runs each line `NAME HOST SET KIND N PR VM SAN RM [ENV...]`
of `LIST` in order. `LIST` is a path or a name looked up in `uniform/final_eval/lists/`, then `gsm_stack/eval/lists/`.
`FE_TREE` picks the code tree: `stk` (default), `stk_r2`, `stk_r2q`, `stk_r3` or `stk_r4` (the overlays here), or a
`gsm_stack` tree name. The overlay is materialized under `$WSM_WORK/trees/` by `gs_tree`. Rows land in
`$FE_OUT_DIR/<SET>_<NAME>.jsonl` (default `uniform/final_eval/out/fe`). The usual form also runs the
adaptive-tries fuzz test of the tree first and sets B_max 128 / S_max 8192, as for every rung-2/3/4 number:
```bash
source uniform/env.sh                                    # optional; every script sources it
python3 uniform/final_eval/data/prepare_gsmplus.py       # once, for the GSM-Plus lists
GPU=0 bash gsm_stack/engine/run_arm.sh P64_s0            # an RL arm (row 2); CW_s0 for row 4 -> $ARMDIR/rl/<ARM>/step_000300
GPU=0 bash uniform/final_eval/fejob.sh stk_r2q uni_r2base.list            # row 3
GPU=0 bash uniform/final_eval/fepack.sh stk_r2q q32rq_s0.list q32rq_s1.list q32rq_s2.list   # several lists packed on one GPU
FE_ROWS=uniform/final_eval/out/fe python3 uniform/final_eval/ana_uni.py   # effects; without FE_ROWS: the shipped rows
```
HOST `ARM` means `$ARMDIR/rl/ARM/step_000300` and `ARM@U` means step U. `ft_int_v2` means `$FT2_HOST`.

**Analyses without a GPU.** The analysis scripts read the shipped reduced rows when no own rows exist, so they reproduce
the published tables on a CPU. For example: `python3 uniform/final_eval/ana_uni.py`, `python3 uniform/final_eval/q32/ana_q32.py b`,
`python3 uniform/rl28/gsm8k/fe_gp/ana_v17.py`, `python3 uniform/rl24/ifeval/analyze.py`,
`bash uniform/rl24/bbh/w/final_ana.sh` and `python3 uniform/rl28/bbh/w/pertask.py`. Use `E=out` / `AO=out` / `FE_ROWS=...`
to point them at your own runs instead.

**IFEval / BBH / 1.7B GSM8K.** See REPRODUCE §17.7, §17.8 and §18. In short:
* `bash uniform/rl24/ifeval/code/setup.sh` once (CPU), then `rl_chain.sh` / `pack_chain.sh`.
* BBH 0.6B: `cd uniform/rl24/bbh/w && bash job.sh data && bash chain.sh RUN ...`.
* BBH 1.7B: `bash uniform/rl28/bbh/setup.sh`, then `w/pack.sh` / `w/drive.sh`.
* GSM8K 1.7B: `GPU=0 bash uniform/rl28/gsm8k/eng/run_arm_17.sh P17_s0`, then `bash uniform/rl28/gsm8k/evchain.sh`.

The original runs went through a cluster queue (one B200 per job, 2-12 h limits, resubmitted until done). The shipped
launchers run the same commands locally, and each chain driver restarts and resumes the same way.

Job ids (`j239xx`) in comments, notes and analysis output are labels of the original runs. The bracketed numbers in some
script comments and analysis headers are labels of the follow-up check batches:
* `[32](b)`: matched-compute majority vote (REPRODUCE §17.4);
* `[32](c)`: eval-seed spread (REPRODUCE §17.5);
* `[32](d)`: GSM-Plus (REPRODUCE §17.6);
* `[32](e)`: BBH per-task check (REPRODUCE §17.8);
* `[33]`: the row-3 ablation (REPRODUCE §17.3);
* `[34]`: the 1.7B fills, such as row 2b (REPRODUCE §18.1).

## Not shipped

* Every RL checkpoint and resume state, at both sizes. GSM8K: P64 / CW / P17 / CW17. IFEval: P / I. BBH: F0 / I0 / P0 / F17
  / I17 / P17. They are rebuilt by the arm commands. The call-FT hosts `ft_int_v2` (recipe REPRODUCE §6) and `ft17`
  (`rl28/bbh/setup.sh`) are not shipped either, nor the BBH gate-refit states and probes (`probe_ft2.pt`,
  `probe_ft17.pt`, rebuilt by `cg/job_refit.sh`). The 1.7B calc veto is `results/gsm_bigger/veto_q17.pt`.
* Generated text. Only reduced per-item rows ship (`items/`, `res/`):
  * GSM8K: answer, correctness and token counts.
  * BBH: task, gold, ok and ntok.
  * IFEval: per-prompt scores.
* Run logs, `out/` directories, dmon dumps and the original cluster-queue / peek helpers.
* Duplicates. Files identical to copies elsewhere in the release are copied in at run time:
  * the engine files from `gsm_stack/engine`;
  * the `rl24/bbh/w` sources reused by `rl28/bbh/w/job.sh`;
  * the call-gate code from `bbh_generic/callgate/code` and `rl24/bbh/cg/code`.
* `rl28/bbh/w/chain.sh` and `final_eval.sh`: identical to the `rl24` versions and not used for 1.7B, which used `pack.sh` /
  `drive.sh` instead.
* The 1.7B GSM8K helper scripts listed in `rl28/gsm8k/NOTES.md`: the refit scripts, the earlier pack variants and the one-shot
  queue drivers. Their outputs, the refit `.pt` files, ship inside the overlays.
* `rl28/ifeval/launch_rest.sh`: only the pack order, which is listed in `pack_chain.sh`.

Four small `.pt` files are part of the overlays and must stay tracked:
* `final_eval/trees/stk_r2q/stk/stoprow.pt`
* `rl28/gsm8k/trees/arm17/stk/verifier_probe_L18_q17.pt`
* `rl28/gsm8k/trees/stk_r2q_17/stk/stoprow.pt`
* `rl28/gsm8k/trees/stk_r2q_17/stk/verifier_probe_L18_q17.pt`
