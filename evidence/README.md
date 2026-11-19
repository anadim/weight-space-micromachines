# Extra evidence runs (calculator by operand size, silence at scale, eyes seed sweep)

All runs: frozen Qwen3-0.6B-Base (items 1, 2) or fine-tuned Qwen2.5-0.5B (eyes), one A100 80 GB, torch 2.8,
transformers 4.56.2. Log: `NOTEBOOK.md`.

| item | blog claim | result | files |
|---|---|---|---|
| 1 calculator vs operand size (`calc_s4`, 2-shot, n = 100 per cell, 1-9 digit operands) | "Accuracy as the numbers get longer" figure | implant 100% in every multiplication cell; base mult 100 / 69 / 8 / 1 / 0 for 1x1..5x5, mean 24.4%; 9-digit add 80 -> 100, sub 61 -> 100 | `item1/*.json`, `figures/item1_calculator.png` |
| 2 silence at scale (**retired nine BBH implants** all attached; 2,000 passages, 274k tokens) | (supporting; not the current BBH tools) | logits bit-identical wherever no implant writes, 2000/2000 passages; 0 fires on 227,891 wikitext + code tokens; 1,154 fires on GSM8K solutions, all at `a op b =` | `item2/silent_scale.json` |
| 3 eyes, 8 and 32 training questions, 4 seeds | "With 32 worked solutions plus the implant, 95% vs 4%" | n=8: 78.5 (76.2-81.4) vs 4.0; n=32: 94.7 (89.1-97.2) vs 3.7 | `eyes_sweep/results/` |

The "4% -> 100%" multiplication number in the blog text is the earlier frozen **Qwen2.5-0.5B** test
(`finetuned_phase/calc_icl/`, `finetuned_phase/results/calc_icl_100.json`); the figure is item 1 on Qwen3-0.6B-Base.

## Reproduce

```bash
pip install -r requirements.txt && bash data/download_bbh.sh && python data/prepare_gsm.py
python scripts/solve_veto.py --device cuda          # runs/veto_qwen3.pt (only needed for item 4)
# item 1 (~12 min, one GPU)
PYTHONPATH=. python evidence/code/item1_calc.py --n 100 --bs 120 --out evidence/item1
python evidence/code/fig_item1.py                    # evidence/figures/item1_calculator.png
# item 2 (~50 min; downloads wikitext-103 test)
PYTHONPATH=. python evidence/code/item2_silent.py --out evidence/item2
python evidence/code/classify_fires.py evidence/item2/fires.json
# item 4 (fire visualisation, writes BBH question text to evidence/item4/, not committed)
PYTHONPATH=. python evidence/code/item4_fire_viz.py --out evidence/item4
# item 3 (eyes; ~5 h on one A100 for 16 fine-tunes + evals)
cd finetuned_phase/eyes_and_finetune
python eyes.py cnn && python eyes.py data 16000      # MNIST CNN (HF ylecun/mnist) + question sets in data/
cp ../../data/gsm8k/train.jsonl data/train.jsonl     # ft.py opens it even with --gsm 0
bash ../../evidence/code/eyes_sweep3.sh              # appends to evidence/eyes_sweep/results/summary.txt
cd ../.. && python evidence/code/agg_eyes.py evidence/eyes_sweep/results/summary.txt
```
Seed noise (eyes): at n = 32 the implant arm ranges 89.1-97.2 over 4 seeds (std 3.8); the `:N` data spec takes the first
N questions, so seeds change only the init/order. The original single-seed 88 at n = 8 did not reproduce (recipe not
recorded); the 4-seed means replace it. `eyes_sweep/results/*_test.json` are the per-run summaries (macro and per-task
accuracy); per-row generations are not included. Item 2's raw `fires.json` (contains GSM8K text) is not included;
`classify_fires.py` rebuilds the breakdown from a fresh run.
