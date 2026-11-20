# Call gate: stop the fine-tuned host from calling tools where none fits

What it is: the integration-v2 host (`../integrate/`, Qwen3-0.6B-Base fine-tuned on 432 rows, 8 tools attached) writes
`<tool_call>` on 59% of the questions of the 15 BBH tasks that have no tool; the tools answer those calls with an empty
reply and the host loops to the token cap. The call gate is one more implant: a **linear probe on the host's post-norm
last hidden state at the step where it is about to emit `<tool_call>`** (a single token, id 151657). If the probe says
"no tool fits" (`w·h + b < 0`), the `<tool_call>` token is blocked and the block is latched for the rest of the answer,
so the host answers itself. A second, training-free variant (**breaker**) latches the block after the first empty tool
reply.

The probe (`results/probe_v3.pt`: one 1024-vector + bias, threshold folded into the bias) was trained **only on our own
questions**: Team A/B tool questions (positives) and our own no-tool questions, including programmatic puzzle-format
distractors (`code/cg_neg_prompts.py`, `code/cg_gen_puz.py`), shown zero-shot and behind tool-task prefixes. Its
configuration was frozen before it touched BBH; BBH was run once per variant. `NOTEBOOK.md` records every decision,
including the two amendments made after looks at our own held-out sets (disclosed there).

Not in weights (as for IFEval): the greedy decode loop, the hook on/off switch, the block mask and the per-row latch flag.

## Results

Real BBH, 27 tasks, one look per arm (Qwen3-0.6B-Base v2 host):

| arm | 27 mean | 12 tool tasks | 15 no-tool tasks | call % on the 15 | no-answer % on the 15 |
|---|---:|---:|---:|---:|---:|
| v2 FT + tools | 48.36 | 84.33 | 19.58 | 59.2 | 35.7 |
| **+ call gate (probe v3)** | **54.58** | 85.63 | **29.74** | **3.8** | 7.3 |
| + breaker (no training) | 53.30 | 85.00 | 27.94 | 59.5 | 11.9 |

For reference, the 15 no-tool tasks score 39.15 on the base model and 30.01 on the fine-tuned host without tools: the gate
recovers the fine-tuned host's own level; the rest of the gap to the base is the cost of the fine-tune itself.

Own held-out sets (strict accuracy; none / gate / breaker): Team A oos2 47.8 / 52.2 / 55.0; Team B oos 71.3 / **61.3** /
71.3 (the gate blocks real zero-shot meeting-scheduling calls: meeting 78 → **48**); Team C ood 97.2 / 98.9 / 97.2.
ARC-Easy (400 val, 3-shot CoT, never used for training): 68.25 / 70.0 / 70.5. Probe held-out AUROC on own data 0.9999.

Silence: wherever the gate or breaker never blocked, the text is identical to the no-gate run in the same harness (own
sets 396/396 and 416/416, ARC 352/352 and 369/369); against the published v2 transcripts (a different run, other batch
sizes) 87-92% of unblocked BBH outputs are identical, the usual bf16 batch noise.

## Files

* `code/` — `cg_gate_gen.py` (gated greedy decode: modes none / gate / breaker), `cg_run27_gate.py` (BBH 27),
  `cg_sets_gate.py` (own sets), `cg_arc.py` (ARC-Easy), `cg_collect_states{,2,3,4}.py` (opener-step states),
  `cg_neg_prompts.py`, `cg_gen_puz.py` (own negatives), `cg_train_probe.py`, `cg_select.py`, `cg_loso.py`, `cg_logo.py`
  (probe fit and own-data model selection), `cg_report.py`, `cg_silence.py` (tables, from saved results);
  `run_callgate.sh` (every step); `_cgpaths.py` (paths: `WSM_REPO`, `FT_HOST`, `BBH_DIR`, `BBH_COT`, `CG_RESULTS`, `VETO`).
* `results/` — `probe_v{1,2,3}.pt` (v3 is the one used on BBH), reduced per-question records of every run (`*.json.gz`: gold, prediction, correctness, blocked,
  tool calls, SHA-256 of the text; no text), run logs, `cg_report.txt`, `silence.txt`. The own-data opener states the
  probes were fit on (`states{2,3,4}.pt`, hidden-state dumps of the fine-tuned host, 5 MB) are not shipped; `run_callgate.sh
  states` regenerates them on a GPU.

## Reproduce

Tables from the shipped records (CPU, seconds):

```bash
cd bbh_generic/callgate/code
python cg_report.py ../../integrate/results "v2 + GATE=../results/bbh_gate_even.json,../results/bbh_gate_odd.json" \
                    "v2 + BREAKER=../results/bbh_breaker.json,../results/bbh_breaker_b.json"
python cg_silence.py ../../integrate/results ../results
# probe refit: needs results/states{2,3,4}.pt from `run_callgate.sh states` (GPU, below; not shipped)
CG_LAM=0.1 python cg_train_probe.py probe_check.pt 0.5 ../results/states{2,3,4}.pt   # = results/probe_v3.pt (to 5e-6)
```

From scratch (GPU; needs the v2 host built per `../integrate/README.md`, not released):

```bash
export FT_HOST=bbh_generic/integrate/runs/ft_int_v2 VETO=runs/veto_qwen3.pt
bash bbh_generic/callgate/code/run_callgate.sh states   # greedy decodes of ~2,900 own questions
bash bbh_generic/callgate/code/run_callgate.sh probe
bash bbh_generic/callgate/code/run_callgate.sh bbh      # the original split the gate run into even/odd task halves
bash bbh_generic/callgate/code/run_callgate.sh sets
bash bbh_generic/callgate/code/run_callgate.sh arc
bash bbh_generic/callgate/code/run_callgate.sh report
```

Expected: the numbers above within bf16 batch noise (about +-1 point on the 27-mean; the original gate run used batch 32
for the even tasks and 16 for the odd tasks, the breaker batch 16).
