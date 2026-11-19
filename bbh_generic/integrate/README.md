# Integration: one host, all eight generic tools (protocol rules 14 and 15)

What it is: the eight tools of Teams A, B and C (`track`, `order`, `interval`, `walk`, `truth`, `expr`, `bool`, `calc`)
attached to ONE host at once, and tested (1) on each team's own fresh-test and out-of-style sets, (2) on a new composition
set of 45 questions that need two or three tools in sequence (`data/compose.jsonl`, 9 families x 5, frozen before any
model output on it was produced), and (3) on all 27 real BBH tasks.

## How it attaches

`code/integ.py` puts all eight compiled machines into ONE `micromachines.machine.Attached` object: one hook before the
host's final norm; the machines' writes are added; output tokens shared by several machines share one solved write
direction. Team C's float64 stack and start-of-text flag are applied as in `teamC/teamc_tools.attach_gen`. Team C's tools
are used in their CALL form (`expr_call`, `bool_call`, `calc_call`), so every tool is called explicitly; an extra arm also
attaches Team C's passive `=`-triggered tools (`--tools all+passiveC`; that arm uses the calc veto fitted on GSM8K train).
`code/integ.GraphStep` replays one decoding step of the attached machines as a CUDA graph for speed;
`code/check_graph.py` checks that its writes are bit-identical to the plain step.

Host: a full fine-tune of Qwen3-0.6B-Base on `data/train_union.jsonl` (302 rows, copied verbatim from each team's own
seed-2 data: Team A `teamA/data/train100.jsonl` 102 rows, Team B `ft100/sub/B_n100_s0` 100 rows, Team C
`teamC/data/train/train_C_100.jsonl` 100 rows). Recipe = Team A's ft100a, `teamA/ft.py` unchanged: lr 2e-5, 6 epochs,
tok_budget 8000, seed 0 (126 optimizer steps). **The fine-tuned checkpoint is not released.** This 302-row union is
the **v1** host. The **v2** host (the one in the blog and in `../callgate/`, `../transfer/`) is a full fine-tune with the
same recipe on `data/train_union_v2.jsonl` (**432 rows**, 156 steps): the A (102) and B (100) rows unchanged, Team C's 100
rows converted to the passive form (`data/c_passive_v2.jsonl`, `code/make_v2_rows.py`), 100 self-distilled no-tool-call
rows (`data/nocall_v2.jsonl`: prompts in `code/nocall_prompts.py`, completions = the frozen base's own greedy answers,
`code/make_nocall.py`) and 30 multi-tool rows of 5 new kinds (`data/multi_v2.jsonl`, `code/make_v2_rows.py`), shuffled by
`code/make_union_v2.py`. In v2, Team C's tools are attached in their passive form (`--tools v2` = `integ.TOOLS_V2`).

## Results

All 27 real BBH tasks, one run per arm (official prompts and scoring), base / fine-tuned host **without** tools / with
tools. `mean27 / 12-tool-task / 15-no-tool-task`:

| arm | mean 27 | 12 tool tasks | 15 no-tool tasks |
|---|---:|---:|---:|
| base (frozen, no tools) | 40.98 | 43.27 | 39.15 |
| **v2** host, no tools | 33.61 | 38.10 | 30.01 |
| **v2** host, + all 8 tools | **48.36** | **84.33** | 19.58 |
| v1 host, no tools | 32.13 | 38.63 | 26.93 |
| v1 host, + all 8 tools | 47.98 | 82.73 | 20.18 |

Reading it: attaching all eight tools to one host lifts the 12 tool tasks to **84.33** (v2); `multistep_arithmetic_two`
goes 33.2 (v1) → 100 (v2) once Team C's passive tools fire. The **15 no-tool tasks regress** (39.15 → 19.58) because the
fine-tuned host still calls a tool on many of them; v2's no-call rows cut that over-calling from **88.6%** (v1) to
**59.2%** of no-tool questions, and a retry loop on empty replies drives the remaining no-tool drop. The fine-tuned host
without any tools is weaker than base on the no-tool tasks (30.01), confirming the fine-tune teaches calling, not those
tasks.

Held-out out-of-style sets (strict answer, base → v2 FT + tools; v1 in parentheses): Team A oos2 20.0 → **47.8** (v1
68.9), Team B oos 11.3 → **70.7** (v1 80.0), Team C ood 62.2 → **97.2** (v1 50.0). Team A judge round 3 (held out, 16
variants): base 3/16, v1 7/16, **v2 7/16**.

Composition set (45 questions needing 2-3 tools in sequence, frozen before any output), answer strict:

| arm | strict | every needed tool used |
|---|---:|---:|
| base | 8.9 | 0.0 |
| v1 FT + tools | 11.1 | 0.0 |
| **v2 FT + tools** | **24.4** | 24.4 |

v2 uses every needed tool on 11/45 questions (v1: 0/45); the gap to the per-tool accuracy is mostly the host emitting the
wrong call, not the tools (each tool's "used/correct" counts are in `TABLES` in the study ledger).

Exactness (shipped v1 code, `test_integ.py`, `check_graph.py`): on the frozen base every tool answers only its own calls
with no cross-firing, logits are bit-identical where no tool writes, and the CUDA-graph decode step is bit-identical to
the plain step.

Provenance: all numbers above are in `results/` (reduced per-question records, no generated text: `../reduce_bbh.py`)
and are printed by `python code/report_v2.py results` (except the per-tool composition analysis, which needs the full
outputs; its full-output tables are `TABLES_v1_vs_v2.md`, v1 only: `TABLES.md`). `results/test_integ_*.json` are the
exactness checks. `NOTEBOOK.md` records the setup and the per-arm run splits (the 27-task runs were split into task
shards across GPUs; `report_v2.py` merges `b27_<arm>_[TP]*`). The merged v2 / base rows are also in
`../transfer/results/q06_*.json.gz`.

## Reproduce

```bash
cd bbh_generic/integrate/code
python compose.py ../data/compose.jsonl                         # byte-identical to the shipped file
python make_union.py ../../ft100/sub/B_n100_s0/data ../data/train_union.jsonl   # byte-identical to the shipped file
(cd ../../teamA && python ft.py --data ../integrate/data/train_union.jsonl --out ../integrate/runs/ft_int300 \
                                --lr 2e-5 --epochs 6 --tok_budget 8000 --seed 0)
mkdir -p ../out
python test_integ.py ../out/test_integ_base.json                                  # all tools on the frozen base: own calls, no cross-firing, bit-identity
python test_integ.py ../out/test_integ_ft.json --model ../runs/ft_int300
python check_graph.py ../runs/ft_int300                                            # CUDA-graph step == plain step
python eval_integ.py ../out/sets_base_notools.json --tools none
python eval_integ.py ../out/sets_ft_tools.json --model ../runs/ft_int300 --tools all
python eval_integ.py ../out/sets_ft_notools.json --model ../runs/ft_int300 --tools none
python eval_integ.py ../out/var3_ft_tools.json --model ../runs/ft_int300 --tools all --sets A_var3 --bs 16   # Team A judge round 3
python eval_integ.py ../out/sets_ft_tools_passiveC.json --model ../runs/ft_int300 --tools all+passiveC --sets C_test,C_ood,compose
python run27_integ.py ../out/b27_ft_tools.json --model ../runs/ft_int300 --tools all --max_new 600       # real BBH, 27 tasks
python run27_integ.py ../out/b27_ft_notools.json --model ../runs/ft_int300 --tools none --max_new 600
python run27_integ.py ../out/b27_base_notools.json --tools none --max_new 600
python report.py ../out                                                           # aggregate tables only
```

The original runs split the 27-task runs into task halves (`--tasks`) across GPUs; `report.py` reads `b27_<arm>.json` or merges
`b27_<arm>_T*.json`. Real BBH needs `bash data/download_bbh.sh` (repo root); the calc veto (`--tools all+passiveC`) needs
`python scripts/solve_veto.py --out runs/veto_qwen3.pt`. Paths: `code/_paths.py` (env `WSM_REPO`, `BBH_DIR`, `BBH_COT`,
`VETO`). The original run used byte-identical vendored copies of `micromachines/` and `bbh_generic/`; this code imports the
repo's own.

### v2 host (432 rows; the blog's BBH numbers)

```bash
cd bbh_generic/integrate/code
python make_v2_rows.py ../data/multi_v2.jsonl ../data/c_passive_v2.jsonl       # byte-identical to the shipped files (CPU, seeded)
python make_nocall.py ../data/nocall_v2.jsonl                                   # GPU; greedy self-distillation of the frozen base
                                                                                #   (bf16 greedy: may differ in a few rows; the shipped file is the one used)
python make_union_v2.py ../data ../data/train_union_v2.jsonl                    # byte-identical to the shipped file
(cd ../../teamA && python ft.py --data ../integrate/data/train_union_v2.jsonl --out ../integrate/runs/ft_int_v2 \
                                --lr 2e-5 --epochs 6 --tok_budget 8000 --seed 0)   # 156 steps, ~90 s on one H100
python ../../../scripts/solve_veto.py --out ../../../runs/veto_qwen3.pt         # Team C's calc veto (0.6B base), or set $VETO
python test_integ.py ../out/test_integ_v2_ft.json --v2 --model ../runs/ft_int_v2
python eval_integ.py ../out/sets_v2ft_tools.json   --model ../runs/ft_int_v2 --tools v2
python eval_integ.py ../out/sets_v2ft_notools.json --model ../runs/ft_int_v2 --tools none
python run27_integ.py ../out/b27_v2ft_tools.json   --model ../runs/ft_int_v2 --tools v2   --max_new 600 --bs 32   # ~1.5 h on one H100
python run27_integ.py ../out/b27_v2ft_notools.json --model ../runs/ft_int_v2 --tools none --max_new 600 --bs 32
python report_v2.py ../out
```

Expected (one run each, greedy): real BBH 27 / 12 / 15 means **48.36 / 84.33 / 19.58** with tools, 33.61 / 38.10 / 30.01
without, base 40.98 / 43.27 / 39.15. Greedy bf16 decoding with a different batch size or GPU changes a few percent of the
outputs (`../callgate/NOTEBOOK.md` measured 87-92% identical transcripts across runs), so expect each mean within about
+-1 point. `run27_integ.py` resumes from `OUT.partial` task by task; `--veto` and `--tools subset --machines ...` are used
by `../transfer/`.
