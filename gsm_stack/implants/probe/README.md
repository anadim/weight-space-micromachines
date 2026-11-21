# probe/ — hidden-state extraction (lanes 2/3)

Node data root: `$WSM_DATA/implants/hs/{set}/` (set = `dev`, `dev2`; smoke runs carry a tag suffix, e.g. `dev_smoke`).

## What is extracted (`extract.py`, job dir = this folder)
For every attempt in `tables/attempts_{set}.jsonl.gz` (copy of `atlas/data/attempts_{set}.jsonl.gz`) the host
(HF `Qwen/Qwen3-0.6B-Base` + `gsm_sel/ckpt_a2s2/model.safetensors`, bf16, sdpa) is teacher-forced on the attempt's
visible context `pre (8-shot prefix, Lp=1066) + C.q_ids(q) + seg_tokens`, positions `0..len-1` (= the stack's forget mask
+ position register). Stored, fp16:
- layers 10..27 = HF `hidden_states[L]` (layer-L input); "layer 28" = final-norm INPUT (layer-27 output, pre-norm).
- kinds: `1 nl` every token of `seg_tokens` whose text contains `\n` (step ends; `j` = newline ordinal; for k>0 / repaired
  attempts j=0 is the '\n' ending the written phrase "Wait, let me solve it again.\n"), `2 hash` the token completing
  `####` (`hash_off`; the stack's L18 verifier read), `3 last` last token of `seg_tokens`; `0 qend` last question token
  (one row per item, in `qend.npy`).

Files per set: `info.json` (layers, shards, counts), `shard_XXX.npy` (nL=19, n, 1024) fp16 layer-major (mmap one layer),
`pos_meta.npz` (att, kind, j, off = offset in seg_tokens, pos = absolute position, tok, shard, row), `qend.npy`
(19, n_items, 1024) + `qend_meta.npz` (idx, qlen, pos), `attempts.npz` (idx, seed, k, ok, lg, bad, nrep, hash_off,
seg_len, gid; row = `att`), `attempts_light.jsonl.gz` (atlas table without seg_tokens, same order), `items_{set}.jsonl.gz`.

## Loader (`fit.py`, run on the node: `bash run.sh python3 ...`)
```python
import fit                                   # HS_ROOT env overrides $WSM_DATA/implants/hs
X, meta = fit.load('dev', layers=[14, 18, 22], kind='hash')   # X (N, 3, 1024) float16; kind: hash|nl|last|all|qend
meta['ok'], meta['idx'], meta['seed'], meta['k'], meta['lg'], meta['off'], meta['j'], meta['cls'] ...
oof_logits, auc = fit.grouped_cv_probe(X[:, 1], meta['ok'], groups=meta['idx'], folds=5, l2=1e-2)
predict, w, b = fit.fit_logreg(Xtr, ytr, l2=1e-2)            # torch LBFGS, standardized, L2; w/b in input units
fit.auroc(y, score)
```
`load(..., rows=mask_or_index)` reads a subset of the kind's rows. qend meta has `p_right` (fraction of the item's
attempts that are right) and `cls`/`n_right` when the items table is present.
CLI: `python3 fit.py --set dev --kind hash --layers 14,18,22,27` (grouped-CV AUROC of `ok` per layer + concat, and of the
register logit lg).

## Jobs
`bash run.sh python3 extract.py --sets dev,dev2 [--limit_items 20 --tag _smoke]` (GPU). `--inspect` = load check (CPU).
Validation (written to `out/extract_{set}{tag}.json`): hook capture == `output_hidden_states` (exact), and the L18 hash state
. verifier probe vs the register logit `lg` (corr / MAE). Notes: `notes/hs_extract.md`.

## Other sets (e.g. trainC)
Any set name works: copy `atlas/data/attempts_{set}.jsonl.gz` (+ `items_{set}.jsonl.gz` if any) into `probe/tables/` and
`bash run.sh bash job_full.sh trainC` (or `bash run.sh python3 extract.py --sets trainC`). Note: `data/` inside a gq job
dir is shared across jobs (not snapshotted), hence `tables/`.
