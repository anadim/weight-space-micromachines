# Topic blocker compiled fully into weights

The emulated blocker in `topic_fence/` (`fc.py`) kept its gate latch, decline-phrase counter and monitor run-counter as
Python state in the hook. `blocker_circuit.BlockerCircuit` replaces every one of those decisions by fixed tensor ops
(token-table reader for the end of the prompt, a ReLU threshold unit on the fitted probe, hard-attention heads that latch the
gate and count positions, a width-3 conv window for the monitor's "3 consecutive hits"), in the same pattern as
`micromachines/rethink_circuit_v2.py`. The fitted pieces are the same as before (probe, monitor probe, solved write
directions in `../fitted/`). Outside the tensors: only the greedy decode loop and the hook on/off switch.
Design, the float32 threshold bug and every check: `NOTEBOOK.md`.

Results (Qwen3-1.7B chat, thinking off, frozen):

| check | result | file |
|---|---|---|
| compiled vs emulated, dev, full greedy replies (2 fences x 2 conditions x 120 prompts) | 480/480 identical | `results/equiv2.log` |
| silence: logits of rows that did not fire vs hook off (640 test prompts, 40 steps) | 1998/1998 rows identical, max diff 0.0 | `results/testc_gen.log` |
| test replies (1540 per condition per fence) and MMLU (400) | byte-identical to the emulated run; every judged metric equal | `results/test_q17c_*_summary.json`, `compare_old_new.py` |

So the topic-blocker numbers in `topic_fence/README.md` (math: blocks 98.7%, near-miss 1.4%, normal requests 9.5% vs 6%
plain; ML: 98.4% / 8.6% / 5.5%) hold for the in-weights implant. `results/test_q17c_gen_*_reduced.json` keep per-row
fired flags / fire steps and a SHA-256 of each reply (no text).

## Reproduce (one GPU, ~24 GB; run from `topic_fence/` after the steps in `topic_fence/README.md` "Reproduce")

```bash
cd topic_fence && mkdir -p res && cp fitted/* res/          # fitted probes / monitors
python build_test.py                                         # data/test.jsonl + data/mmlu_ret.jsonl
export PYTHONPATH=inweights:.:..                             # blocker_circuit, fc, micromachines
python inweights/test_equiv.py math both 60 && python inweights/test_equiv.py ml both 60   # needs data/dev_all.jsonl
FENCES=math OUT=res/test_q17c_gen_math.json python inweights/s5_compiled.py Qwen/Qwen3-1.7B q17c plain,probe_text_stop,monitor
FENCES=ml   OUT=res/test_q17c_gen_ml.json   python inweights/s5_compiled.py Qwen/Qwen3-1.7B q17c probe_text_stop,monitor
python s6_judge.py q17c res/test_q17c_gen_math.json && python s6_judge.py q17c res/test_q17c_gen_ml.json
cp results/test_q17_*_summary.json res/ && python inweights/compare_old_new.py   # old (emulated) vs new summaries
```
`s6_small.py` / `judge_retry.sh` run the same judge computing only last-position logits (used when GPU memory was shared;
verdicts identical). Runtime on one H100: ~1 h generation + ~30 min judging. Greedy; no seed noise.
