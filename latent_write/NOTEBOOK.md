# latent_inweights_v1: the `(a*b)+c` latent write, fully compiled into fixed tensors (2026-11-17)

Goal: rebuild the earlier latent-write experiments E3/E4 (Python-side write; research code not in this repo) (exact product written into the `(a*b)` span at block L) so that
the inference path is only fixed tensors in an implant beside the frozen host (Qwen3-0.6B-Base). No Python parsing,
masking, position edits or per-token control flow on the inference path; a tensor cache across decode steps is allowed.

Ran on one shared GPU (H100 80 GB).

## Files
* `implant.py` - the compiled implant: reader + window + compiled ReLU circuit (hdl.Net) + big-M gated replacement write.
* `lwv1.py`    - task setup (byte-for-byte the E4 seeds/prompts), writer fit (ridge, the only fitted tensors),
                 Python REFERENCE write (token parse + replace = the old method), helpers.
* `run.py`     - circuit check, residual check, silent check, eval arms. `hdl.py` = copy of micromachines/hdl.py (identical; the packaged `implant.py` imports `micromachines.hdl`).

## Implant (inference path)
1. reader: flags = clamp(E_tok Wr^T + br, 0, 1) on the host's token embeddings, 14 symbols (digits, ` (`/`(`, `*`,
   `)+`, `)-`); exact 0/1 on every vocab row (asserted at build, same construction as make_reader).
2. window: flags at offsets -6..+6 (fixed unfold; tensor cache of the last 6 rows across decode steps; offsets beyond
   the current chunk read 0).
3. circuit (written, not trained): per layout class c=(la,lb,op) and slot i, role = AND of the n+2 token flags
   (` (` a-digits `*` b-digits `)+`); a = gated linear read of a's digits; P = sum_q 10^q sum_v v*relu(a - 100(1-[b_q=v]))
   (exact); digits of P by threshold units; g[c,i] = AND(role, len(P) == n-1); outputs F = [g, AND(g, onehot digit)].
4. write entering block L: h' = (h - gate(h,G)) + F @ Wbank_L, gate = relu(h - M(1-G)) - relu(-h - M(1-G)),
   M = 1e6, G = sum g. Wbank rows = [mu_i ; A_i] per slot (ridge on plain `(V)+c` prompts with random V).
   Slot positions get exactly mu + A f(P); every other position gets h back bit-for-bit.

## Disclosures (what is not a plain causal fixed-tensor op)
* The window looks 6 tokens AHEAD (a slot at `a1` needs to see `*`, b and `)+`). Fixed op, but non-causal within the
  prefilled prompt; a span is written only if it is fully inside the prefill chunk (same as the Python version,
  which also wrote at prefill). Decode tokens see zeros ahead; the pattern never completes in the answer.
* The implant writes EVERY matching span, including the 4 few-shot examples (the old Python write only touched the
  query). Reported both ways: `ref_q` = old query-only Python write.
* Controls only (not the implant): corrupted P' and mean-value arms override the value features at the query span
  from Python (as before). A fully fixed-tensor no-value control (`meanall`: writer with the mean code at every span)
  is also run.
* The host attachment is two hooks (embedding forward hook, block-L pre-hook); a `reset()` of the implant's tensor
  cache at the start of each batch (like starting a new KV cache).
* Writer is fitted per task layout and prefix (as in E4), by closed-form ridge on plain oracle prompts.

## Log
* 17:2x  smoke (mid, L8): compiled 98.0% / follow 97.3 / mean 0; outputs identical to Python ref 300/300. First try.
* v0 runs (results_v0/, arithmetic-only implant code) then refactor to a generic pattern spec (`Spec`, `arith_spec`,
  word template); everything below re-run with the final code (results/). v0 and final agree.
* Bugs on the way (all crashes, no silent errors): writer tensors on CPU; BIGM dropped in the refactor; a shadowed
  variable in the residual check; 1.7B out of memory on the shared GPU (moved to another GPU).

## RESULTS, Qwen3-0.6B-Base (n=300 per cell, greedy generate, max 6 new tokens; acc / follow-corrupted / mean control, %)
Same items, shots, corrupted values and writer-fit values as the earlier E4 script (`e4_general.py`, not in this repo; seeds 1234/99/11) and, for the
word problem, the earlier word-problem script (`q2_word.py`, not in this repo) loc `both` (seed 21). Window: L = layer whose INPUT residual is rewritten.

| cell | plain | text oracle | tool line | L0 | L2 | L4 | L8 | L10 | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a*b)+c mid  acc | 15.0 | 98.0 | 96.3 | 98.3 | 98.7 | 99.0 | 98.0 | 97.7 | 96.0 | 67.0 | 54.3 | 18.0 | 4.0 |
|   follow P' | | 97.7 | | 97.3 | 97.7 | 97.7 | 97.3 | 97.3 | 96.7 | 68.3 | 55.7 | 20.7 | 4.3 |
|   mean ctrl | | | | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (a*b)+c hard acc | 0.0 | 96.0 | 61.0 | 95.3 | 95.0 | 94.0 | 94.7 | 95.0 | 95.7 | 38.7 | 44.0 | 18.3 | 0.0 |
|   follow P' | | 95.3 | | 94.7 | 95.3 | 95.3 | 92.7 | 94.0 | 93.0 | 39.3 | 40.7 | 18.7 | 0.3 |
|   mean ctrl | | | | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| (a*b)-c mid  acc | 2.7 | 99.0 | 95.3 | 98.3 | 97.7 | 98.3 | 99.7 | 99.7 | 99.3 | 83.3 | 67.0 | 34.0 | 14.3 |
|   follow P' | | 99.0 | | 98.0 | 93.3 | 99.0 | 98.7 | 98.0 | 96.3 | 76.3 | 52.7 | 24.7 | 7.7 |
|   mean ctrl | | | | 0.3 | 0 | 0 | 0.7 | 0 | 0 | 0.3 | 0 | 0.3 | 2.0 |
| word "lost" acc | 3.3 | 94.3 | 70.0 | 98.3 | 98.3 | 98.3 | 96.0 | 94.7 | 91.3 | 88.7 | 82.7 | 3.7 | 2.3 |
|   follow P' | | 93.7 | | 97.7 | 98.3 | 97.7 | 96.7 | 95.7 | 91.7 | 86.3 | 82.7 | 3.3 | 1.3 |
|   mean ctrl | | | | 0 | 0 | 0 | 0 | 0.3 | 0.3 | 0.3 | 0.3 | 0.7 | 0.7 |

L22 (arith): mid 13.7 (= plain, follow 1.0), hard 0.0, mulsub 3.0. Fully fixed-tensor no-value control (mean writer
at every span, `meanall`): 0-1% everywhere L<=20. Full per-arm tables (incl. Python reference, query-only old write,
output-identity counts, residual diffs) in `summary.md` (generated by summarize.py).

Old Python numbers for comparison (earlier E3/E4 runs, not in this repo, query-only write, tf-scored): mid 96-98 (L<=12), hard 95.3 (L12),
mulsub 97.0 (L12); earlier word-problem run lost/both 96.0-96.7 (L0-4), 88-89 (L12-14), 67.7 (L16). Reproduced.

## Machine check: compiled implant == Python reference write
* Circuit vs Python token parse, every eval prompt (task + text-oracle prompts): fired spans / slot positions /
  value features F / P all identical (0 mismatches): mid 600 spans (query + one 3-digit shot), hard 300, mulsub 900,
  lost 1500 (query + 4 shots), 21000 slot positions.
* Residual entering block L (compiled vs Python replacement, same writer tensors), every L, all 300 prompts:
  non-slot positions bit-identical to the plain model (all cells, all L); slot positions differ in 0.0007-0.003% of
  elements by 1 bf16 ulp (the compiled path sums mu + A f(P) as one GEMM over F, the reference as mu + f@A; different
  fp32 summation order before the bf16 cast). max |h| seen = 6432 << big-M 1e6.
* Greedy outputs identical to the Python reference write: true and corrupted arms 300/300 for every L <= 18 in mid and
  mulsub and L <= 16 in lost (lost L18-20: 298-300); hard 293-300/300 (the ulp differences flip a handful of items, both directions: acc equal within
  0.4 pt at every L). Mean-control arm: 207-300/300 identical (garbage outputs are sensitive to the ulp; both 0%).

## Silent check (implant attached, prompts without the pattern)
268-318 prompts per cell at L0 and L8: no-pattern shots + `a*b`, `(a+b)*c`, `(a*b)*c`, `(d*ddd)+d` (other layout),
`(P)+c` text oracle, near-miss word templates (3-digit a, 2-digit b, "pencils"), free text with parentheses, `*`,
`(2*x)+3`, `(64*128)`. Fired positions 0; logits bit-identical to the plain model in every batch; generations identical.
At decode steps the implant never fires (pattern needs `)+` ahead); on task prompts every non-slot position is
bit-identical at block L and all layers < L are untouched by construction.

## Findings
* The latent write survives compilation intact: the frozen 0.6B goes 15.0 -> 96-99% (mid), 0.0 -> 94-96% (hard),
  2.7 -> 98-99.7% ((a*b)-c), 3.3 -> 91-98% (word problem) with the write entering any block L <= 12, and follows a
  corrupted value 93-98%. Mean-value controls 0%. The window closes at L14-18 exactly as in the Python version.
* Why this works when the earlier compiled BBH latent write (an added vector at `)`, a negative result, not in this repo) failed: this is a REPLACEMENT (h -> mu + A f(P), via the big-M gate) at
  every token of a span whose length matches the value, not one ADDED vector at one position. The gate makes the
  replacement a fixed-tensor op of (h, token stream).
* Writing every matching span (shots included) is slightly better than the old query-only write at late layers
  (mid L16 54 vs 44, mulsub L14 83 vs 76, lost L16 83 vs 66): the shots then also look like `(P)+c`.
* Word problem: the same circuit machinery (a 14-token template matcher, 9 word symbols) compiles the lw2 "both"
  write; 98.3% at L0-L4 vs text tool line 70.0% and text oracle 94.3%.

## Added when packaging for release (no new runs)
* The Qwen3-1.7B-Base runs (`muladd mid`, `muladd hard`, L0-L18) finished after the log above; their tables are in
  `summary.md` (`python summarize.py`). mid: plain 26.3 -> 97.3-98.0% for L <= 8 (follow 97.7-98.3), 87.0 at L12;
  hard: plain 2.0 -> 94.0-96.0% for L <= 8 (follow 92.3-96.7), 77.0 at L12, 42.3 at L14; mean controls 0.
* Writer files: `results/writers/` keeps the four Qwen3-0.6B-Base writer banks; the 1.7B ones are regenerated by
  `run.py` (see README).
