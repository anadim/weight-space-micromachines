# seeds35 IFEval: §5.1 IFEval rows 2 / 2b / 4 at RL seeds 0-5 (Qwen3-0.6B-Base)

The uniform IFEval rows of blog §5.1 (REPRODUCE §17; code `uniform/rl24/ifeval/`) were first run at RL seeds 0-2. This
check adds RL seeds 3, 4 and 5 of both IFEval RL arms:

- P = host + RL, no implant. Its plain eval is row 2; the same host evaluated with the implant attached is row 2b.
- I = host + RL with the 14-rule implant attached (row 4).

Rows 1 and 3 (no RL) are the seeds 0-2 `u0` evals, unchanged.

## Final (IFEval-541 prompt-level strict, seeds 0-5 pooled; paired item bootstrap, 10,000 resamples)

`cd ifana && SEEDS=0,1,2,3,4,5 ROWSJ=rows_s0-5.json python3 analyze6.py --out table_s0-5.md` (CPU, about 1 min):

| row | prompt strict % [95% CI] | inst strict % | tok/q | per seed (s0 … s5) |
|---|---|---|---|---|
| 1 plain host | 12.94 [10.17, 15.90] | 23.98 | 1889 | 12.94 |
| 2 host + RL (P) | **45.53** [41.99, 49.20] | 58.31 | 517 | 43.62 / 46.77 / 40.30 / 44.55 / 46.77 / 51.20 |
| 3 host + implant | 34.75 [30.68, 38.82] | 47.48 | 1388 | 34.75 |
| 4 host + implant + RL (I) | **48.49** [44.79, 52.22] | 60.91 | 1334 | 49.17 / 48.24 / 47.69 / 48.24 / 49.91 / 47.69 |
| 2b P host + implant at eval | **53.70** [50.09, 57.30] | 65.45 | 504 | 54.16 / 54.71 / 49.17 / 53.97 / 53.79 / 56.38 |

- **4 − 2: +2.96 [−0.28, +6.35]** at 6 seeds (3 seeds: +4.81 [+1.11, +8.56]). Per seed: +5.5 / +1.5 / +7.4 / +3.7 /
  +3.1 / −3.5, so 5 of 6 are positive; P_s5 (row 2 = 51.20) is the outlier. The 6-seed CI includes 0.
- 2b − 2: +8.16 [+6.13, +10.38], positive in 6/6 seeds (3 seeds: +9.12 [+6.84, +11.52]).
- 2b − 4: +5.21 [+2.46, +7.89], positive in 6/6 seeds.
- `ifana/check012.md` = the same script with SEEDS=0,1,2; it equals the §17 IFEval table (row 4 − 2 +4.81).
- Per-seed table with run facts (GPU, idle identity, dev curve): `python3 ifana/perseed.py` (output `ifana/perseed.txt`).

## How seeds 3-5 were run

- Same recipe as seeds 0-2 (`uniform/rl24/ifeval/rl_chain.sh {P,I}_sN` with `MODEL=Qwen/Qwen3-0.6B-Base TAG=inst
  FMT=chat`): `rl_if.py` GRPO / Dr.GRPO, 300 updates, 64 prompts x G8, lr 1e-6, T 1, max_new 1024, chat format with
  thinking off, reward = lm_eval prompt-strict; then `eval541.py` on step_000300 (greedy, max_new 2048, implant on and
  off, idle identity) and `score_reduce.py` (seeded lm_eval scorer). The code (`uniform/rl24/ifeval/code`) is unchanged.
- Seeds 3-5 ran on single A100-SXM4-40GB GPUs with `ifeval_seed.sh` (one self-contained script per arm and seed; it builds
  the implant export `sd_inst.pt` itself with `hostchk/mk_sd.py` from Qwen/Qwen3-0.6B and pins transformers 4.56.2;
  torch 2.8.0+cu126, triton 3.4.0).
  - Memory settings: B_max 128, tok_budget 8192 (auto-resume with 96 / 4096 after a crash; not needed), eval B_max 96.
    Seeds 0-2 on B200 used B_max 512 | 256, tok_budget 32768. This changes batch scheduling only.
  - dev every 100 updates instead of 25 (dev is monitoring only; the RL RNG streams do not depend on it).
  - Wall time (start → train start → train end → end, UTC, from `res/ifeval/*/run_facts.json`): I about 3 h 13-3 h 24
    training + 8 min eval; P about 1 h 25-1 h 34 training + 4 min eval. Peak memory 33.6 GB (I) / 32.2 GB (P).
  - Seeds 0-2 on one B200: P about 62 min, I about 1 h 53 min, eval about 4 min.

## NLTK punkt_tab issue in the seeds 3-5 training (and why the results stand)

- On the A100 hosts, NLTK 3.10.4 silently refused to download `punkt` / `punkt_tab` into a group-writable data dir (the
  download ran with `quiet=True`). Without punkt_tab, `count_sentences` raises and `ifreward.score` returns 0 for the
  whole prompt. So every prompt with a `length_constraints:number_sentences` instruction got RL reward 0: 3.8% of the
  train prompts (about 2.4 of the 64 per update), 5.1% of dev, and 46 of the 541 IFEval items.
- It was fixed in the running jobs about 35 min after they started (punkt / punkt_tab installed; the data is looked up on
  every call). The reward was correct from about update 60-90 (I runs) and update 100-125 (P runs) of 300.
- Effect on training: the advantage is R minus the group mean, and only samples with a non-zero advantage are trained on.
  A number_sentences prompt whose 8 samples all scored 0 therefore gave no gradient at all (it was dropped, not pushed in
  the wrong direction). Prompt sampling and rollout seeds are unaffected.
- The u300 evals ran after the fix. They were re-scored offline from the per-item eval outputs with a scorer that passes
  a punkt_tab / number_sentences assertion: for I_s3-5 and P_s4-5, local = remote on every item (0 differing items, 0
  response-hash mismatches; `res/ifeval/*/rescore_diff.txt`). P_s3's per-item eval outputs were not copied back, so its
  remote score json (scored after the fix) is used (`res/ifeval/P_s3/FALLBACK_*.txt`).
- No measurable deficit on the affected instruction (`python3 ifana/nltk_check.py`, output `ifana/nltk_check.md`): the P
  runs pass the number_sentences instruction on 29 / 29 / 29 of 46 items (seeds 3-5) vs 29 / 28 / 24 (seeds 0-2) plain,
  and 31 / 30 / 29 vs 30 / 29 / 26 with the implant attached. The I runs: 25 / 28 / 23 vs 26 / 22 / 26.
- Visible trace: u0 dev prompt-strict of the I runs is 40.00 on all three new seeds vs 41.69 for seeds 0-2 (7 of 415 dev
  items; 21 dev items are number_sentences). The P runs give 7.23 vs 6.99 at u0 (+1 item: A100 bf16 near-ties move
  single items both ways). At u300 the in-run dev of the new seeds is in or above the seeds 0-2 range: I 55.18 / 62.17 /
  58.55 vs 54.46 / 57.83 / 54.70; P 35.66 / 37.35 / 46.99 vs 39.52 / 41.45 / 33.49.
- `ifeval_seed.sh` (as shipped) makes the NLTK data dir private, downloads with `raise_on_error=True` and asserts
  `count_sentences('A b. C d.') == 2` and a number_sentences reward of 1 before training, so a rerun cannot hit this.
