# Cross-family transfer: calculator and BBH Team C tools on non-Qwen hosts

Question: do the implants built on Qwen3-0.6B attach to small base models from other families? Only the two ends are
rebuilt per host: the token reader (closed form from the new embedding table, asserted exact on the whole vocabulary) and
the write directions / end-of-number veto (solved on the host's own norm + LM head; the veto on GSM8K **train** rows
2000-2299). The circuits are unchanged. Full log, tokenizer analysis and incidents: `NOTEBOOK.md`.
(The IFEval port to SmolLM2-1.7B-Instruct described in the notebook is packaged with the IFEval code, not here.)

## Results

GSM (8-shot plain CoT, greedy, n = 1,319 each, `calc_s4` + per-host veto, stop at the first `#### <number>`;
"200+" = GSM-Hard items 200-1318, the part not used to pick the shots):

| host | tokenizer | GSM8K plain -> calc | GSM-Hard plain -> calc | GSM-Hard 200+ |
|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-1.7B | single digits, space glued to operator (like Qwen) | 32.75 -> 34.65 | 13.95 -> 19.86 | 13.67 -> 18.50 |
| ibm-granite/granite-3.3-2b-base | same | 50.57 -> 50.80 | 23.20 -> 30.40 | 22.34 -> 29.31 |
| tiiuae/Falcon3-3B-Base | single digits, **space is its own token** | 68.99 -> 68.99 | 41.39 -> 41.39 | calculator never fires (outputs byte-identical) |
| LiquidAI/LFM2-1.2B | multi-digit chunks | cannot attach (no single-token digits / `*-`) | | |

SmolLM3-3B-Base, OLMo-2-0425-1B, Phi-4-mini, MiniCPM4-0.5B chunk digits like LFM2 (checked with `code/tokprobe.py`, not run).

BBH Team C tools (frozen host, passive `expr`/`calc`/`bool`, official 3-shot CoT, n = 250; `oe='.'` end write):

| host | multistep_arithmetic_two | boolean_expressions | object_counting | 4 no-tool tasks |
|---|---|---|---|---|
| SmolLM2-1.7B | 18.4 -> 95.6 | 76.8 -> 95.2 | 50.0 -> 73.6 | identical |
| granite-3.3-2b | 32.8 -> 97.2 | 81.6 -> 78.0 (regression) | 70.0 -> 80.4 | identical |
| Falcon3-3B | 75.6 -> 99.6 | 90.0 -> 98.4 | 82.4 -> 85.6 | identical |
| Falcon3-3B, `oe=veto` variant | 75.6 -> 55.6 | 90.0 -> 98.4 | 82.4 -> 85.6 | identical |

Files: `results/gsm/x_{sm2,gr,fal}_{test,hard}_{plain,calc}.json`, `results/bbh/{sm2,gr,fal}_{none,C,Cveto}.json`
(per-question gold / prediction / correctness, generated text replaced by its SHA-256 with `code/reduce_out.py`; the
SHA-256 of the unreduced file is in `source_sha256`), `results/veto_{smollm2,granite,falcon}.pt` (solved vetoes),
`results/silent_*.json` (silence check: logits bit-identical where the calculator does not write).

## Reproduce (one GPU with ~24 GB; A40 46 GB and H100 80 GB were used)

```bash
bash data/download_bbh.sh && python data/prepare_gsm.py
M=HuggingFaceTB/SmolLM2-1.7B; P=sm2                       # or ibm-granite/granite-3.3-2b-base gr, tiiuae/Falcon3-3B-Base fal
python scripts/solve_veto.py --model $M --out transfer/xfamily/results/veto_$P.pt --device cuda
python transfer/xfamily/code/run_gsm_x.py out/x_${P}_hard_calc.json --model $M --split hard --mode calc \
    --veto transfer/xfamily/results/veto_$P.pt --bs 32       # plus --mode plain, --split test
# (or all four arms with retries:  bash transfer/xfamily/code/drvr.sh $PWD $M transfer/xfamily/results/veto_$P.pt $P 24)
python transfer/xfamily/code/check_silent_calc.py --model $M --veto transfer/xfamily/results/veto_$P.pt --out out/silent_$P.json
python transfer/xfamily/code/audit_eq.py out/x_${P}_hard_*.json      # exactness of 'a op b = c' in the outputs
bash transfer/xfamily/code/bbhx.sh $M transfer/xfamily/results/veto_$P.pt $P   # BBH base vs +Team C, 7 tasks
python transfer/xfamily/code/tokprobe.py LiquidAI/LFM2-1.2B          # tokenizer check (no GPU)
```
Runtime: ~20-40 min per GSM arm, ~30 min per BBH arm. Greedy throughout; the batch size (24 vs 32, used after an OOM)
only changes bf16 padding numerics. One run per arm; no seed noise beyond bf16 batching (a few problems per set flip).
`patch_pod.py` is only needed for the discarded first-harness `calc_s3` runs (see "HARNESS CHANGE" in the notebook).
