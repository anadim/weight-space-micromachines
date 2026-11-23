# Release notes: 2026-11-23 update (final blog numbers)

This update adds everything the final blog's numbers need on top of the `release-prep` branch of 2026-11-21. Nothing that
was in the branch was removed or reverted. Changed files are listed below.

## Added

| path | what | REPRODUCE | files | size |
|---|---|---|---|---|
| `uniform/` | Uniform rows of blog §5.1 (Qwen3-0.6B-Base: GSM8K in `final_eval/`, BBH and IFEval in `rl24/`) and blog §6 (Qwen3-1.7B-Base, `rl28/`). Contents: launchers, eval lists, code trees as `mk_tree.py` overlays, reduced per-item rows (`*.items.jsonl.gz`, `res/`) and result tables | §17, §18, §19 | 582 | 11.10 MB |
| `checks/tooltext/` | C35: text calculator tool baseline, 12 train seeds | §20 | 20 | 0.43 MB |
| `checks/calcimp/` | C36: equal-footing text tool vs exact calculator implant, 28 train seeds | §21 | 53 | 4.98 MB |
| `checks/seeds35/` | Blog §5.1 GSM8K and IFEval rows at 6 RL seeds | §22 | 110 | 1.98 MB |
| `checks/seeds3/` | Six one-seed claims (§3.4, §5.1, §5.2) re-run at 3 seeds | §23 | 78 | 2.98 MB |
| `checks/offtask/` | MMLU / ARC-Challenge / HellaSwag / WinoGrande / NQ after RL, implants attached (0.6B and 1.7B) | §24 | 31 (2 symlinks) | 2.03 MB |
| `checks/sourced/` | CPU re-derivations of individual blog numbers (§1.2, §3.3, §3.4, §4.2, §4.4, §5.1, §5.2) | §25 | 32 | 0.92 MB |
| `ifeval/full14_compiled/check_shared.py` (+ edits, below) | Shared-weights export of the compiled IFEval implant (`export.py --shared 1`: 348.2M → 38.7M numbers, 1.23 → 0.30 GB); `check_shared.py` compares it with the full export. Documented in `ifeval/full14_compiled/README.md` | – | 1 | 0.01 MB |
| `tests/calc_sweep_0_199.py`, `results/verify/{calc_sweep_0_199,test_calc_circuit_cpu_200}.log` | Exhaustive calculator sweep over operands 0..199 and its CPU logs | §1 | 3 | <0.01 MB |
| `RELEASE_NOTES.md` | this file | – | 1 | – |

Every folder has a README with its layout and what is not shipped. Every table in REPRODUCE §17–§25 is recomputed on a CPU
from the shipped per-item rows (VERIFY §14).

## Changed

* `README.md`:
  * the first line names and links the blog post (https://papail.io/papers/wsm/);
  * a "Results at a glance" table with the final blog numbers;
  * headline-table numbers updated to the final blog (e.g. plain RL 68.3 at 6 seeds, mixed task 58 vs 45 at 3 seeds,
    topic blocker ~4% of normal requests);
  * a note that the all-in-weights stack of blog §4.4 / §5 is `gsm_stack/` (blog appendix A.3 names `gsm_push/`);
  * layout rows for `uniform/` and `checks/`.
* `REPRODUCE.md`:
  * §1–§16 numbers matched to the final blog;
  * environment rows for the B200 and A100 40 GB machines;
  * index rows for §17–§25;
  * the calculator-sweep paragraph in §1;
  * new sections §17–§25.
  * The `gsm_stack` rung-3/4 gadgets (`gsm_stack/gadgets/vf_w.pt`, `sn_w.pt`) were described as "not shipped". They
    were already in the branch, so the not-shipped rows were removed and the rung-3 command points to them.
* `VERIFY.md`: new §14 (this update's fresh-clone check).
* `.gitignore`: re-includes six small `.pt` gadgets that the code trees load (`*.pt` is ignored by default):
  * `gsm_stack/gadgets/{vf_w,sn_w}.pt`, which were in the branch but ignored;
  * `uniform/final_eval/trees/stk_r2q/stk/stoprow.pt`;
  * `uniform/rl28/gsm8k/trees/{arm17,stk_r2q_17}/stk/verifier_probe_L18_q17.pt`;
  * `uniform/rl28/gsm8k/trees/stk_r2q_17/stk/stoprow.pt`.

  Each file is 1.5–35 KB.
* `gsm_stack/engine/run_arm.sh`: added the uniform and later arms (`P64_sK`, `A1WL_sK`, `A1WD_sK`, `LRP_/LRC_/LRQ_/LRD_<lr>`).
  The existing arms are unchanged.
* `ifeval/full14_compiled/{README.md,execu.py,export.py,run_c.py}`: the `--shared 1` export path. The default export is unchanged.

## Scrubbed

All added material comes from internal working trees and was scrubbed the same way as the 2026-11-21 release:

* Host, node and cluster names, user and agent names, and e-mail addresses were removed.
* Home, cluster and scratch paths were replaced by repo-relative paths or the env vars `WSM_ROOT`, `WSM_DATA`,
  `WSM_WORK`, `CKPT_U0`, `ARMDIR`, `ARMDIR17` and `HF_HOME`.
* Queue-submit wrappers were replaced by direct one-GPU commands. Job ids are kept as labels only.
* Launchers source `gsm_stack/env.sh`, `uniform/env.sh` or `rl/gsm_grpo/env_local.sh`, never a private venv.
* References to internal task / status items were removed or reworded. Some shipped script comments and analysis
  outputs still contain short task labels such as `[32]`–`[34]`; `uniform/README.md` explains them.
* Output dirs named `out/` (git-ignored) were renamed to `res/`, and the scripts and docs were updated to match.
* Per-item rows keep ids, gold answers, last numbers, correctness, token counts and per-instruction booleans. They keep
  no generated text, token ids or log-likelihoods. Score files with responses keep only a SHA-256 of each response.
* Not shipped: checkpoints, RL resume states, training stdout logs, generated text, `__pycache__`, `.pt` files except
  the six gadgets above, and any file over 5 MB.

## Size

The update adds 911 files (24.5 MB, including this file) to the 2026-11-21 tree, which had 3,016 files and 158.6 MB.
On disk the tree now has 3,927 files and 183.2 MB; a clone has 3,783 tracked files and 175 MB (pre-existing git-ignored local
files are not in it). The largest new file is 2.9 MB, and five new files are over 1 MB, all `.jsonl.gz` per-item or
per-update rows.

## Fresh-clone check (VERIFY §14)

Run on a fresh clone, CPU only. Every CPU analysis in REPRODUCE §17–§25 whose inputs ship was rerun, and its output
matches the shipped tables byte for byte. The only exception is a wall-clock stamp in `uniform/rl28/gsm8k/RESULTS.md`.
The other checks:
* `py_compile` passes on 1,078 `.py` files and `bash -n` on 253 `.sh` files;
* the static import scan of the new folders resolves;
* the rung-3 / rung-4 register fuzz tests pass with the now-tracked gadgets;
* the leak grep is clean, including inside the `.gz` files.

The check found three stale CIs in REPRODUCE (§17.6 one, §18.1 two). Each was a copy error in the doc; the point
estimates were right and the shipped results files were right. It also found a missing `links38.py` command (§21) and a
stale GSM-Plus data path in two `mk_code.sh` defaults and their docs. All were fixed. Details are in VERIFY §14.

## Notes for review

* `.gitignore` now re-includes six small `.pt` gadgets (listed above). Without it, `gsm_stack/gadgets/{vf_w,sn_w}.pt`
  were in the working tree but not in git, although `gsm_stack/env.sh` copies them into every rung-3/4 tree.
* The blog's IFEval "+3" (implants given RL, 4 − 2) is +2.96 [−0.28, +6.35] at 6 RL seeds. The CI includes 0. At 3 seeds
  it was +4.81. REPRODUCE §17.7 and §22 give both.
* Task labels such as `[32]`–`[34]` are left in some script comments and output headers. They are explained in
  `uniform/README.md`.
* The final blog's appendix names `gsm_push/` as the all-in-weights stack. The README says the stack of blog §4.4 / §5
  is `gsm_stack/`.
