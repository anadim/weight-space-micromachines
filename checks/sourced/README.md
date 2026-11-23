# checks/sourced: CPU re-derivations of blog numbers

Each blog number here comes from a small CPU script that reads already-existing run outputs. No new training was run. Details, definitions and the blog wording for each number are in `RESULTS.md`. Each `*.out.md` file is the output of the matching script; this README covers the release-copy output.

## Final numbers and inputs

| item | blog claim | number | script → output | inputs | in the release? |
|---|---|---|---|---|---|
| (a) | §3.3 size-ordered veto, "about half the runs" | early single-seed in-wire configs: 4/8 flagged (3 of them transient). C36 replication: fI 0/20 and fI1 0/8 | `veto33.py --seeds 20` → `veto33.out.md` | `rl/results/wiring/H100/L300_*/steps.jsonl` and `checks/calcimp/tt/res/train_steps.jsonl.gz` | yes |
| (a′) | (same, all 28 fI seeds) | fI 0/28; C36 total 0/36 | `veto33.py` → `veto33_s28.out.md` | same | yes |
| (b) | §3.4 BBH: "never helped, one cost 10 points" | gbC2_s0 Δ BBH −10.4 (−22/212); the other 3 runs −1.9 … +0.5 | `gbc_swap.py` → `gbc_swap.out.md` | `rl/figures/v3/data/gbC*_steps.jsonl` | yes |
| (c) | §4.4 compliant S2F vs H | S2Fw +4.93 [+2.96, +6.90]; S2Frw +4.25 [+2.27, +6.22]; H = 67.70 (1,319 test items) | `s2f_vs_h.py` → `s2f_vs_h.out.md` | `res/fe/test_{lad_H,S2Fw_u0,S2Frw_u0,u0_B}.items.jsonl` | yes (shipped here) |
| (d) | §5.2 `<calc>` fade 80→25 / 88 / 95–98 | per-run u1–5 vs u41–50 shares | `calc_fade.py` → `calc_fade.out.md` | `rl/figures/v3/data/calc_share_{A40,H100}.jsonl` | yes |
| (e) | §5.1 IFEval 2b − 4 | +4.31 [+1.17, +7.33] (seed-averaged, 541 prompts); tokens per question: 546 (2b) vs 1,364 (4) | `ifeval_2b4.py` → `ifeval_2b4.out.md` | `uniform/rl24/ifeval/res/{P,I}_s{0,1,2}/eval_u300.{imp,plain}.score.json.gz` | yes (shipped by `uniform/`) |
| (f) | §4.2 instructions per prompt; implant share | 1 / 2 / 3 instructions: 56.4 / 33.1 / 10.5%; implant-written tokens ≤ 2.2% (≤ 7.4% on fired prompts) | `ifeval_42.py` → `ifeval_42.out.md` | `ifeval/transfer/results/reduced/q06_ifeval_{imp,plain}.json` | yes |
| (g) | §4.4 never solved in 32 tries; wrong gold | 23/600 = 3.8% never solved in 32; 5/23 wrong or ambiguous gold (6/23 counting the ill-posed one) | `nv32.py` → `nv32_dev36.out.md` | `res/fe/dev_sc{8,4}_s{7,11,13}.items.jsonl`, `nv_gold_labels.json` (hand labels), and `data/gsm8k/{train.jsonl,shots_v1.json}` | yes, after `python data/prepare_gsm.py` |
| (h) | §1.2 Read layer checked on all ~151k vocabulary tokens | 151,936 rows × 26 symbols, 0 mismatches (calc_s3 and calc_s4); margins ≤ −2 / ≥ +6 | `vocabjob/vocab_reader.py` → `vocab_reader_Qwen3-0.6B-Base.json` and `vocabjob/res/run06.log` (a rerun writes both to `vocabjob/res/`) | `micromachines/` and Qwen/Qwen3-0.6B-Base (HF Hub) | yes, plus a model download |

Notes on the inputs:
- **(c), (g):** `res/fe/*.items.jsonl` are reduced `fe_hook.py` rows (`uniform/final_eval`) with fields i, gold, ok, last and ntok. The dev SC rows also have `answers` and `att_ok`. No generated text is included.
- **(a):** the calcimp copy rounds floats to 5 significant digits. The 20-seed table is unchanged by this rounding; it is byte-identical to the original.
- **(a), entries without step logs:** CW-in s0, CWfx s0 and CWfx s1 have no step logs. They come from `rl/wiring/NOTEBOOK.md`, which is cited in RESULTS.md, and are not re-derived.

## Rerun

Every command below runs from the release root, on CPU, with Python 3 standard library only. Script runtimes are under 1 s unless noted.

```
python3 checks/sourced/veto33.py --seeds 20 > checks/sourced/veto33.out.md        # ~1 s
python3 checks/sourced/veto33.py           > checks/sourced/veto33_s28.out.md
python3 checks/sourced/gbc_swap.py         > checks/sourced/gbc_swap.out.md
python3 checks/sourced/s2f_vs_h.py         > checks/sourced/s2f_vs_h.out.md          # ~13 s (10k bootstrap)
python3 checks/sourced/calc_fade.py        > checks/sourced/calc_fade.out.md
python3 checks/sourced/ifeval_2b4.py       > checks/sourced/ifeval_2b4.out.md        # ~12 s (10k bootstrap)
python3 checks/sourced/ifeval_42.py        > checks/sourced/ifeval_42.out.md
python data/prepare_gsm.py                 # once: writes data/gsm8k/{train,test}.jsonl + shots_v1.json (git-ignored; needs `datasets`, HF Hub)
python3 checks/sourced/nv32.py dev checks/sourced/res/fe/dev_sc8_s{7,11,13}.items.jsonl checks/sourced/res/fe/dev_sc4_s{7,11,13}.items.jsonl > checks/sourced/nv32_dev36.out.md
bash checks/sourced/vocabjob/run.sh        # ~10 s CPU; torch + transformers; downloads Qwen3-0.6B-Base (HF_HOME as you like)
```

The scripts find the release root from their own location. Set `WSM_ROOT` to point them at another copy.

**Verified on the release copy.** These outputs are identical to the original outputs:
- veto33 (20 seeds), gbc_swap, calc_fade, ifeval_42, s2f_vs_h and nv32.
  - s2f_vs_h and nv32 differ from the originals only in the file names they print (`.items.jsonl`); the shipped `.out.md` files are the release-copy outputs.
  - nv32 was checked with the original `shots_v1.json`.
- ifeval_2b4, run on the gzipped score files that `uniform/` ships.

**Not re-run here.**
- `vocab_reader.py`: no model download was possible in the release check. Its shipped output is the original run.

## Not shipped

- **Full fe_hook rows.** They contain generated text and are replaced by the reduced `res/fe/` files.
- **The SC×32 GPU confirmation for (g).** It was never run.
- **A 1.7B vocabulary check.** It was not run.
- **The original `vocabjob/micromachines` copy.** It is identical to the release `micromachines/`, which is used instead.
