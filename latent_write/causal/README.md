# Latent write: is it causal? (negative result)

The released latent-write implant (`latent_write/`) detects the `(a*b)` span with a window that **looks up to 6 tokens
ahead** inside the prefilled prompt and overwrites the operand positions. This folder tests a strictly causal version.

* `causal.py` — forward-only: detect the span at its closing token, write the value there (and optionally a few later
  positions). Result: **no gain** (muladd mid: plain 13.3 -> 13.3/0/3.7 by layer; `results/causal_*.json`).
* `kvrewrite.py` — detect at the closing token, then rewrite the operand positions' layer-L state (in a token-by-token
  decode this is an edit of their cached K/V, i.e. of the past). Result: recovers the non-causal accuracy
  (98.7 / 99 / 98 at L0/4/8, `results/kvrewrite_muladd_mid.json`).

Conclusion (as in the blog): the latent write works only when the write can edit earlier positions.

## Reproduce (one GPU; imports `latent_write/lwv1.py` and `latent_write/implant.py`)
```bash
PYTHONPATH=latent_write LW_OUT=latent_write/causal/results python latent_write/causal/causal.py muladd mid 0,4,8,12,14,16 1
PYTHONPATH=latent_write LW_OUT=latent_write/causal/results python latent_write/causal/causal.py lost mid
PYTHONPATH=latent_write LW_OUT=latent_write/causal/results python latent_write/causal/kvrewrite.py muladd mid 0,4,8,12
```
Details and tables: `NOTEBOOK.md`. One run per cell, n = 300, seeds as in `lwv1.py`.
