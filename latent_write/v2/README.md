# Latent write v2: two values, chains, word problems, value types, a bigger host (Python reference)

Follow-up experiments to `latent_write/` (the compiled, in-weights implant). **Not compiled**: here the span parser and the
value are computed by the Python reference engine `lw2.py` (same writer recipe: a least-squares digit -> residual map fit
on plain `(V)+c` prompts with random V, no task data). Use these numbers as "what a latent write can carry", not as an
in-weights result. Log of every run: `NOTEBOOK.md`.

Headline numbers (n = 300 per cell, no chain of thought, frozen hosts):

| cell | plain | latent write (best layer) | text tool line |
|---|---:|---:|---:|
| two values `(a*b)+(c*d)`, Qwen3-0.6B-Base | 5.3 | 90.7 (L8) | 95.3 |
| two values, Qwen3-1.7B-Base | 15 | 96 | |
| three values | 3.7 | 60 | 70 (text ceiling) |
| word problem "lost", 0.6B | 3.7 | 96.7 | 71.7 |
| word problem "rows", 0.6B | 6 | 93.7 | 83 |
| bool / not, 0.6B | 51 | 100 | |

The blog's word-problem numbers (3% -> 98%, text tool 70%) are the **compiled** `lost` cell in `latent_write/results/`
(`latent_write/README.md`), which agree with these reference runs.

## Reproduce (one GPU with >= 5 GB free)

```bash
cd latent_write/v2
mkdir -p logs
bash queue1.sh        # sum2, lost, value types, chain, 1.7B mid/hard, rows, sum3, 0.6B mid  -> results/*.json, logs/
bash gcr_queue2.sh    # value types again + Qwen2.5-0.5B / 1.5B
# one cell:  LW_DEV=cuda:0 python q1_multi.py sum2 0,4,8,12,14,16
python plot.py        # latent_write2_curves.png from results/
```
`runq.sh` waits for a GPU with > 5 GB free and retries on OOM. `runlogs_a40/`, `runlogs_a100/` are the logs of the original
runs (two machines: one A40 46 GB pod and one A100 80 GB). Seeds are fixed inside `lw2.py` / each `q*.py`; one run per cell.
