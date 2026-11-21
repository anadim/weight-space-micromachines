# gsm_stack/implants: the implant lane's work on the serial GSM8K stack

Every implant here was built on top of the all-weights serial stack (`gsm_stack/engine`, `eval/trees/v42`), tested on
dev (600 items) and dev2 (300 items) from GSM8K **train** split B, with 4 seeds (7, 11, 13, 17) unless stated, and
kept only if it gave a gain on both dev sets with everything in weights. No implant here was tuned on test.
The full write-up is `gsm_stack/results/REPORT.md` (failure atlas, probes, all implants) and the dev ladder is
`gsm_stack/results/LADDER.md`.

Folders that hold a code tree (`stk/`, `wiring/`, `dig_san/`) store it as an **overlay**: only the files that differ
from the base named in `TREE_BASE`. Build a runnable tree with `python3 gsm_stack/mk_tree.py OUT <folder>`. The lane
READMEs are progress logs written while the work ran; they quote cluster job-queue commands as
`<queue-submit> DIR TIMEOUT NAME -- CMD` (run `CMD` inside `DIR` on one GPU) and job ids `j23xxx` (labels only).
Per-row outputs (`out/`, `tables/`, `*.jsonl.gz`, hidden-state `*.npz`) are not shipped.
Lane scripts locate each other through `$WSM_ROOT/gsm_stack/implants/...` (default: the current directory), so run
them from the repo root or set `WSM_ROOT`.

Shared analysis code of the lane (no data): `voter/` (offline voters: `learn.py` with the ClusterLin group scorer that
rung 3's `vf_w.pt` is fit with, `feats.py`, `vote.py`), `atlas/` (attempt parser `parse.py`, first-divergent-step
finder `diverge.py`, step tables, the never-solved labelling), `probe/` (hidden-state extraction and probe fits behind
REPORT.md §3), `vfrefit/off/refit.py` (offline re-pick with refit writer weights, used by `sidenet/repick_sn.py`).

## Rungs (adopted, all weights)

| rung | folder | what it adds | dev / dev2 (4 seeds) | test (u0 host, once) |
|---|---|---|---|---|
| 0 | `aw/` (`aw/dev_tree`: the rung-0 harness) | the all-weights serial stack without a position reset: calc_s4 + chain2, compiled sanity DFA + repair, weight-space forget, L18-probe ReLU voter, N = 8 | 81.04 / 79.08 | 71.87 |
| 1 | `wpr/` | **weight-space position reset (WPR)**: an offset register (`s_fr − s_qlen`) and a fixed fp32 rotary table rotate each attempt's q and new K by R(−off) after the host RoPE in every layer, so attempt k sees the question at the attempt-0 distance. Earlier attempts stay hidden by the forget mask. `wpr_fixed.diff` is the patch on `comp_serial.py`; `HANDOFF.md` the hand-off note | 88.79 / 85.92 | **80.67** |
| 2 | `rungs/rung2` (+ `aw/`) | **adaptive tries** as a compiled comparator on the answer / probe registers: extend by 4 up to 16 if the winner's probe-weight share < 0.7; stop at 4 if unanimous with min probe ≥ 0.5; ≤ 3 repairs for non-integer answers; attempt cap 400 (`stk/aw_adapt.py`; CPU fuzz `test_aw.py`, `test_adapt_refs.py`) | 89.58 / 86.92 | **81.58** |
| 3 | `rungs/rung3` (+ `vfeat/`) | Nmax 24 + **answer-group voter as registers** (`stk/aw_vf.py`, `stk/numreg.py`): a per-stream number-set register automaton and a fixed linear group score (ClusterLin, fit on split C) replace the final pick (`AW_VF=1`, weights `vf_w.pt`; CPU fuzz `test_vf.py`) | 90.29 / 87.33 | **82.71** (GSM-Hard 60.42, −2.2 vs rung 2: the voter is distribution-specific) |
| 4 | `rungs/rung4` (+ `sidenet/`) | **side-net verifier register**: one fixed linear vector on the layer-14/18/22/27 residuals (8,192 weights + bias), scored at the `####` token and averaged over line ends, blended into the rung-3 group writer (`stk/aw_sn.py`, `AW_SN=sn_w.pt`; CPU fuzz `test_sn.py`) | 90.46 / 88.33 (8 seeds: 90.34 / 88.17) | not run on test |

`vfeat/` is the offline voter-feature study behind rung 3 (feature sets, ClusterLin fit `fitw.py`, robustness);
`sidenet/` the side-net training (`job/train_sn.py`, `mk_snw.py` writes `sn_w.pt`, `repick_sn.py` offline re-pick).

## Negative results (`negative/`, not adopted; each folder has its README with numbers)

| folder | idea | verdict |
|---|---|---|
| `m5div` | bias the first steps of later attempts away from earlier attempts' openings (diversity) | the implant makes first lines distinct (4.0 → 5.3-7.9 per 8 tries) but answers stay the same (1.80 → 1.83-1.88 distinct); vote within ±0.5 of base, oracle@8 ≤ +0.9 |
| `selcheck` | in-stream verification writer for contested votes (yes / no verdict, forward / backward checks) | the host cannot verify its own answers better than the vote: clean negative, no in-stack build |
| `dola` | DoLa-style contrastive layer decoding as a weight-space logit write | no right-branch preference in the teacher-forced prescreen; in the stack the dev-selected configs cost 1.3-2.9 points |
| `seqver` | sequence side-net verifier (attention pooling over all body-token states) vs rung 4 | fails the pre-registered +0.3 gate; no in-stack build |
| `prmstep` | step-level (PRM-style) probe on line-end residuals | adds nothing over rung 4's score; no in-stack build |
| `reread` | force each new attempt to re-copy / re-read the question | per-attempt accuracy falls 0.7-4.6 points |
| `trunc` | min-p / temperature floor on branch tokens | null: wrong and right branch tokens are both in the host's top-2 |
| `linefix` | step-detector line rewind (resample from the flagged line) | gate failed (+0.05 / −0.07 vs a +1.0 gate); not built |
| `voter2` | verifier-v2 register in the voter (blended sigmoid score) | null: the pooled AUROC gain is cross-item calibration, it does not change within-item choice |

## Fitted gadgets that are not in git

`vf_w.pt` (rung 3) and `sn_w.pt` (rung 4) are small fitted weight files. They were not synced from the GPU sandbox
for this release, so rung 3 / rung 4 cannot be rerun from the repository until they are added to `$WSM_DATA/gadgets/`
(the launchers copy them into the run tree). Rebuild routes: `vfeat/fitw.py` (needs the split-C rollout cache that
`vfeat/job/gen.sh` writes) and `sidenet/job/train_sn.py` + `sidenet/mk_snw.py` (needs split-C hidden states from
`sidenet/job/extract_sn.py`), each one GPU job of rollouts / state extraction plus a CPU fit. The fit settings are in
`sidenet/vfsn_BLD_0.001.json` and the lane READMEs.
