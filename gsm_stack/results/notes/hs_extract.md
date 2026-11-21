# hs-extract: hidden states of every dev/dev2 attempt (2026-11-20)

Code: `probe/extract.py` (extraction), `probe/fit.py` (loader `load(set, layers, kind)` + question-grouped torch L2
logistic probe), `probe/README.md` (API, file layout), `probe/job_full.sh`, `probe/run.sh` (node env). Job j23144.

**Node paths:** `$WSM_DATA/implants/hs/dev` (6.6 GB) and `$WSM_DATA/implants/hs/dev2` (3.4 GB).
- Context per attempt: `pre (Lp=1066) + C.q_ids(q) + seg_tokens`, positions 0..len-1. Model: HF Qwen3-0.6B-Base +
  ckpt_a2s2 weights (311 keys, bf16, no missing/extra; lm_head tied), bf16, sdpa, plain right-padded batching.
- Layers 10..27 = `hidden_states[L]` (layer input); 28 = final-norm input (pre-norm). fp16, (19, n, 1024) layer-major shards.
- Positions: dev 19191 attempts / 600 items: 140298 nl (newline tokens), 19050 hash, 19191 last; qend 600.
  dev2 9600 / 300: 72212 nl, 9557 hash, 9600 last; qend 300. (141 dev / 43 dev2 attempts have no '####'.)

**Validation.**
- Hook capture == HF `output_hidden_states` exactly (max abs diff 0.0); qend identical across all attempts of an item.
- L18 state at hash_off . verifier probe vs register `lg`: dev corr 0.9982, MAE 0.077, p99 0.25, max 0.41 (n=19050);
  dev2 corr 0.9983, MAE 0.076, max 0.45 (n=9557). Same with repaired attempts (corr 0.999). fp16 storage adds nothing.
  The residual difference is bf16 kernel noise: the engine itself computes in bf16 with its own Triton kernels. An fp32 forward
  (smoke) gave corr 0.9990 but MAE 0.117, so we keep bf16. A CPU bf16 run gave MAE 0.05.
- Loader sanity (in-set 5-fold question-grouped CV, l2=1e-2, target = attempt ok): hash L14/18/22/27 AUROC dev
  .757/.785/.805/.801 (concat .805), dev2 .734/.807/.796/.792 (.802); the register lg itself: dev .817, dev2 .834.
  qend (item p_right>=.5): dev .67-.69, dev2 .63-.73. (Untuned demo only; the probe design belongs to the predictability lane.)

**Timing (B200):** 155k tok/s forward, ~103 attempts/s; dev 186 s total (24.3M tokens), dev2 93 s. The whole job takes ~6 min
including the fit demos. Prefix-KV reuse was not needed.
trainC: `cp atlas/data/attempts_trainC.jsonl.gz probe/tables/ && bash run.sh bash job_full.sh trainC` (see README).
