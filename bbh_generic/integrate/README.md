# Integration: one host, all eight generic tools (protocol rules 14 and 15)

> **RESULTS PENDING.** This experiment was still running when the repository was prepared (real-BBH runs of the
> integrated host had not finished). The code and the data are final; no result files are included yet. The notebook
> (`NOTEBOOK.md`) records the setup and the first reads (for example, the composition set's first read with tools:
> 0/45 questions with every needed tool called); treat them as interim.

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
tok_budget 8000, seed 0 (126 optimizer steps). **The fine-tuned checkpoint is not released.**

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
