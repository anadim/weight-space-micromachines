# Latent write, compiled into weights: an exact value written into the middle of the network

The task is `What is (a*b)+c?`, answered at once with no chain of thought, by frozen Qwen3-0.6B-Base (and 1.7B-Base).
The model can do `a*b` alone and `P+c` alone, but not both at once: 15.0% correct (2-digit times 1-digit), 0.0% on the
hard range. The implant reads `a` and `b` from the prompt tokens, computes `P = a*b` exactly in a compiled ReLU circuit,
and **replaces** the residual stream at the token positions of `a*b` (the span ` (` a `*` b `)+`) entering block L with
the state those positions would have if the prompt had said `(P)+c`. The model then answers `P+c`. It is a tool call
whose result never appears as text.

The inference path is fixed tensors only: no Python parsing, masking, position edits or data-dependent control flow.
An earlier, unreleased version of this experiment did the same write from Python (token parse + in-place edit); that
Python write is kept here only as the reference the compiled implant is checked against (`RefWriter` in `lwv1.py`).

## How it attaches (fully in weights)

Per forward call of the frozen host (`implant.py`):

1. **reader**: symbol flags = `clamp(E_tok Wr^T + br, 0, 1)` on the host's token embeddings, 14 symbols for the
   arithmetic tasks (10 digits, ` (`/`(`, `*`, `)+`, `)-`) or the digits plus 9 word symbols for the word problem. The
   reader is solved in closed form from Qwen's embedding matrix and checked exact (0/1) on every vocabulary row when
   the implant is built (same construction as `micromachines.machine.make_reader`).
2. **window**: the flags at offsets -K..+K around every position (a fixed unfold; a tensor cache keeps the last K rows
   across decode steps).
3. **circuit** (written, not trained; `micromachines.hdl`): for each slot of the span, a role flag (AND of the span's
   token flags), the gated digit products that give `P` exactly, digit extraction by threshold units, and the outputs
   `F = [g, AND(g, onehot(digit of P))]`. 270 ReLU units / 3,726 nonzeros (mid), 427 / 5,215 (hard), 690 / 6,298
   (word problem).
4. **write** entering block L: `h' = (h - gate(h, G)) + F @ Wbank_L`, with the big-M ReLU pair
   `gate(h, G) = relu(h - M(1-G)) - relu(-h - M(1-G))`, `M = 1e6`, `G = sum g`. Slot positions get exactly
   `mu_i + A_i f(P)`; every other position gets `h` back bit for bit (checked: max |h| seen 12,096, far below M).

The host's weights are untouched. Attachment = two hooks (a forward hook on the embedding, a forward pre-hook on
block L), plus a `reset()` of the implant's tensor cache at the start of each batch (like starting a new KV cache).
Generation is Hugging Face `generate` (greedy, 6 new tokens).

**Fitted part (the "weight addition")**: `Wbank_L`, one `[mu_i ; A_i]` block per slot and layer, fitted by closed-form
ridge regression (lambda = 1, float64) from digit one-hots of random values `V` to the host's hidden states on plain
`(V)+c` prompts (3,000 prompts for arithmetic, 2,000 for the word problem). No task answers, no gradient training, no
change to Qwen. One writer bank per task layout and few-shot prefix.

### Disclosures

* **The window looks 6 tokens ahead** (the slot at the first digit of `a` must see `*`, `b` and `)+`). It is a fixed
  op but non-causal within the prefilled prompt: a span is written only if it is fully inside the prefill chunk (as
  in the Python version, which also wrote at prefill). Decode tokens see zeros ahead, and the pattern never completes
  in the answer.
* The implant writes **every** matching span, including the 4 few-shot examples (the old Python write touched only the
  query). The results report both (`old Python write (query only)` rows).
* **Controls only, not the implant**: the corrupted-value (`P'`) and mean-value arms override the value features at
  the query span from Python. A fully fixed-tensor no-value control (`meanall`: the writer with the mean code at every
  span) is also run, and gives 0-1% at L <= 20.
* The writer is fitted per task layout (digit counts of a, b, c) and few-shot prefix.

## Reproduce

From the repository root, one GPU (about 7-11 min per 0.6B cell and 15 min per 1.7B cell on a shared H100):

```bash
pip install -r requirements.txt
bash latent_write/run_all.sh            # all six cells -> latent_write/out/ (+ summary.md and the figure)
# or one cell:  LW_MODEL=Qwen/Qwen3-0.6B-Base python latent_write/run.py muladd mid 0,2,4,8,10,12,14,16,18,20,22
python latent_write/summarize.py        # tables from the shipped results/ (no model) -> same as summary.md
python latent_write/plot.py             # latent_inweights_Qwen3-0.6B-Base.png from results/
```

`run.py TASK RANGE LAYERS` with TASK in `muladd`, `mulsub`, `lost` (word problem) and RANGE in `mid`, `hard`. Each run
fits the writers, builds the implant, and runs: the circuit check (compiled spans/slots/values vs the Python token
parse on every eval prompt), the silent check (L0 and L8), plain / text-oracle / text-tool-line baselines, and for
every L the compiled write (true, corrupted, mean, meanall), the Python reference, and the residual check. It saves
`TASK_RANGE_MODEL.json` and `writers_TASK_RANGE_MODEL.pt`. Env: `LW_MODEL`, `LW_DEV` (default `cuda:0`), `LW_CAP_GB`
(GPU memory cap, default 12), `LW_OUT` (default `latent_write/out`).

Seeds and settings (all in `lwv1.py`): few-shot items `random.Random(1234)` (4 shots), item pool `random.Random(99)`
(20,000 items, the most common digit layout kept, first 300 = test), corrupted values and writer-fit values
`random.Random(11)`; word problem seed 21 (300 test items, 2,000 fit prompts). Ranges: mid a 11-29, b 3-9, c 10-99;
hard a 12-49, b 12-29, c 100-899. Greedy, max 6 new tokens, batch 50, bf16 host, writers float32.

## Results

`results/*.json` (one per cell; compact, no generated text), tables in `summary.md` (`python summarize.py`), figure
`latent_inweights_Qwen3-0.6B-Base.png`. n = 300 per cell. Accuracy of the compiled write, % (L = block whose input
residual is replaced; Qwen3-0.6B-Base has 28 blocks):

| cell (Qwen3-0.6B-Base) | plain | text oracle | text tool line | L0 | L4 | L8 | L12 | L14 | L16 | L18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (a*b)+c mid | 15.0 | 98.0 | 96.3 | 98.3 | 99.0 | 98.0 | 96.0 | 67.0 | 54.3 | 18.0 |
| (a*b)+c hard | 0.0 | 96.0 | 61.0 | 95.3 | 94.0 | 94.7 | 95.7 | 38.7 | 44.0 | 18.3 |
| (a*b)-c mid | 2.7 | 99.0 | 95.3 | 98.3 | 98.3 | 99.7 | 99.3 | 83.3 | 67.0 | 34.0 |
| word problem ("pens lost") | 3.3 | 94.3 | 70.0 | 98.3 | 98.3 | 96.0 | 91.3 | 88.7 | 82.7 | 3.7 |

| cell (Qwen3-1.7B-Base) | plain | text oracle | text tool line | L0 | L4 | L8 | L12 | L14 | L16 | L18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (a*b)+c mid | 26.3 | 99.0 | 98.0 | 97.7 | 97.3 | 97.3 | 87.0 | 72.3 | 47.7 | 5.3 |
| (a*b)+c hard | 2.0 | 96.7 | 90.3 | 95.3 | 96.0 | 94.0 | 77.0 | 42.3 | 14.3 | 4.0 |

* The later layers use the written value: with a corrupted `P'` written instead, the output is `P'+c` in 91.7-99.0% of
  cases at L <= 12 (0.6B, all four cells; `follow` rows in `summary.md`). Mean-value controls: 0-2% at L <= 20.
* "text oracle" = the prompt says `(P)+c` in text; "text tool line" = the few-shot prompt has a `Tool: a*b = P` line.
* Exactness: compiled spans, slot positions, value features and `P` equal the Python token parse on every eval prompt
  (0 mismatches). Entering block L, non-slot positions are bit-identical to the plain model in every cell and layer;
  slot positions differ from the Python reference in 0.0007-0.0031% of elements by about one bf16 ulp (summation
  order). Greedy outputs equal the Python reference write
  (true and corrupted arms) on 300/300 at most layers and 273/300 at worst (1.7B, L18); the flips go both ways and
  accuracies agree within 0.4 points.
* Silent: 268-318 prompts per cell without the pattern (near misses, other layouts, free text with parentheses and
  `*`): 0 fired positions, logits bit-identical to the plain model, generations identical.

`results/writers/` holds the four fitted writer banks used for the 0.6B numbers (float32, 36 MB in all; SHA-256 below).
`run.py` refits them on every run (closed-form ridge; same seeds), so it does not need these files. The two 1.7B banks
(9 and 15 MB) are not included; `run_all.sh` regenerates them. Refitting on another GPU can differ in the last bits
(bf16 host forward), so a refit may flip an item or two, not the tables.

```
1b3767942da7bacaf0c705db119c94fcc9d8a0d7112e0139227a84a7154fb90f  writers_lost_mid_Qwen3-0.6B-Base.pt
3b7cb82106c7b80ac0d3fd76a4144273ce7367024d92b7ed81389f2a03971164  writers_muladd_hard_Qwen3-0.6B-Base.pt
49d7db5b701c22301aefbb83e486a80203b0f510d22ec68436a478c8726a069d  writers_muladd_mid_Qwen3-0.6B-Base.pt
3e46216231c01f8e988efc99fdc18baf7e3a3e3387e26e5dec620dfa73e155cc  writers_mulsub_mid_Qwen3-0.6B-Base.pt
```

## What is not here

* The earlier Python-side latent-write experiments (two-value, nested, bits, BBH multistep with key masking and RoPE
  renumbering) are not released: their writes were Python edits, not fixed tensors. The BBH variant compiled into
  weights as one added vector at `)` was a negative result (4-8%, the same as a mean-value control) and is also not
  included. Only the replacement write in this folder is presented as a result.
* `NOTEBOOK.md` is the working log of this experiment (lightly edited to remove machine names).

## Files

`implant.py` (reader, window, circuit, gated write), `lwv1.py` (task setup, writer fit, Python reference write),
`run.py` (all checks and arms for one cell), `run_all.sh`, `summarize.py`, `plot.py`, `silent_text.txt` (free-text
prompts for the silent check), `results/`, `summary.md`, `NOTEBOOK.md`.

## Follow-ups

* [`v2/`](v2/README.md): two / three values, chains, more word-problem templates, value types, Qwen3-1.7B and Qwen2.5
  hosts (Python reference engine, not compiled).
* [`causal/`](causal/README.md): a strictly causal write (forward-only) gives no gain; rewriting the operand positions'
  past state (KV edit) recovers the non-causal accuracy.
