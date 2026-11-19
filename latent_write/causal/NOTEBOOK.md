# latent_causal: a CAUSAL latent-write variant (2026-11-17, overnight Job 2)

Host: Qwen3-0.6B-Base, frozen. Builds on `<repo>/latent_write/` (the non-causal implant).
Pod: A40 pod A40 (GPU1/GPU0, LW_DEV), code at `<workdir>/wsm/latent_causal` (= local `<notes>/latent_causal`).
Same 300-item sets / seeds / writer-fit values as the non-causal runs (lwv1.setup, seeds 1234/99/11/21).

## Why the original is non-causal (the audit's point 5)
The shipped implant's window reads offsets -6..+6: to fire at an operand digit of ` (a*b)+` it must see the closing
`)+` that is UP TO 6 TOKENS AHEAD. It then rewrites the residual at those earlier operand positions. That only works
because the whole prompt is prefilled in one pass; fed token by token, the write would never fire (the closing token
has not arrived when the operand digit is processed). It also writes every matching span (shots included) and only the
one layout class whose product length fits the slots.

## What "causal" means here, and the two mechanisms tested
A causal online process can use only tokens at or before the current one. The span is fully known at its CLOSING token
`)+`/`)-` (every operand is behind it), so the natural causal trigger is the closing token. Two ways to then inject the
value P = a*b, both still fixed-tensor ops in the Implant (reader + compiled ReLU circuit + big-M gated write; window
now backward-only, KF=0):

1. **causal_close (forward-only):** write the value at the closing token position itself (nwrite=1). Fully causal,
   edits nothing in the past, token-by-token realizable. `causal.py`.
2. **causal_KVrewrite (edits the past):** detect at the closing token, write the value into the OPERAND positions'
   block-L residual (by shifting the closing-token detection backward). In a prefill this is byte-identical to the
   non-causal span write; in a true decode the operand residuals are already consumed, so this is realizable ONLY as
   an edit of those positions' CACHED K/V — a past-state edit, NOT a forward write. Stated honestly. `kvrewrite.py`.

Controls (fully in-weights): `comp_meanall` = the writer with the mean code at every fired position (no value);
for the causal arm, `causal_meanall` = mean code at the closing token.

## RESULTS (acc %, n=300, greedy, 6 new tokens; write enters block L = layer whose INPUT residual is rewritten)

### muladd mid  (plain 13.3, text-oracle 98.0, tool-line 96.0)
| L | non-causal span | nc mean | causal_close | causal mean |
|---|---|---|---|---|
| 0 | 98.7 | 0.0 | 13.3 | 13.3 |
| 2 | 99.0 | 0.0 | 12.3 | 14.7 |
| 4 | 99.0 | 0.0 | 11.0 | 11.7 |
| 8 | 98.0 | 0.0 | 1.3 | 3.3 |
| 12 | 96.3 | 0.0 | 0.3 | 0.3 |
| 14 | 66.7 | 0.0 | 2.0 | 2.3 |
| 16 | 54.0 | 0.0 | 5.3 | 8.0 |
| 18 | 16.7 | 0.0 | 15.3 | 14.3 |

### muladd hard  (plain 0.0, text-oracle 95.7, tool-line 62.3)
| L | non-causal span | nc mean | causal_close | causal mean |
|---|---|---|---|---|
| 0 | 95.3 | 0.0 | 0.0 | 0.0 |
| 2 | 95.0 | 0.0 | 0.0 | 0.0 |
| 4 | 94.7 | 0.0 | 0.0 | 0.0 |
| 8 | 94.0 | 0.0 | 0.0 | 0.0 |
| 12 | 94.3 | 0.0 | 0.0 | 0.0 |
| 14 | 38.7 | 0.0 | 0.0 | 0.0 |
| 16 | 42.0 | 0.0 | 0.0 | 0.0 |
| 18 | 18.7 | 0.0 | 0.0 | 0.0 |

### word "lost"  (plain 3.7, text-oracle 94.3, tool-line 70.0)  -- causal_close here = the NUMBER-ONLY write (value at one position)
| L | non-causal full-clause | nc mean | causal_close = number-only | causal mean |
|---|---|---|---|---|
| 0 | 98.3 | 0.0 | 3.7 | 3.7 |
| 2 | 98.7 | 0.7 | 3.0 | 3.3 |
| 4 | 98.3 | 0.0 | 3.7 | 3.7 |
| 8 | 96.0 | 0.3 | 3.7 | 3.3 |
| 12 | 91.0 | 0.7 | 2.0 | 1.7 |
| 14 | 89.0 | 0.7 | 1.3 | 1.7 |
| 16 | 83.7 | 0.3 | 1.7 | 1.3 |
| 18 | 4.0 | 0.7 | 3.3 | 3.3 |

### causal_KVrewrite (muladd mid) -- confirms the past-state edit recovers the non-causal accuracy
| L | causal_KVrewrite | non-causal span (same items) |
|---|---|---|
| 0 | 98.7 | 98.7 |
| 4 | 99.0 | 99.0 |
| 8 | 98.0 | 98.0 |
| 12 | 96.3 | 96.3 |

(byte-identical to the non-causal span write, as expected: in a prefill the backward write edits the same operand
positions. The point is that a causal TRIGGER suffices; the residual edit lands on past positions, i.e. it is a
cached-K/V edit in true decode, not a forward write.)

## FINDINGS
* **Causality costs essentially all of the accuracy for a forward-only write.** The causal_close write (value at the
  closing token, the only place a token-by-token circuit can write after seeing the whole span) does not inject the
  value: muladd mid stays at plain (13.3% at L0) and collapses to 0-5% by mid-depth; muladd hard is 0.0% at every
  layer (plain 0); lost is 1-4% (= plain) at every layer. The non-causal span write is 94-99% over the same layers.
* **Why:** the host reads its operands from the DIGIT positions, not the closing token. Writing the true value at the
  closing token is no better than writing a garbage mean there (causal_close == causal_mean at every layer). The
  non-causal write's whole power is that it OVERWRITES the operand digit positions — which a causal write cannot reach
  without look-ahead. Contrast the controls: non-causal mean = 0% (garbage AT the operands destroys the computation),
  non-causal value = 94-99% (the model does read those positions).
* **A causal mechanism recovers the accuracy only by editing the PAST** (the operand positions' cached K/V). The
  causal_KVrewrite arm reproduces the non-causal numbers because, in a prefill, editing the operand positions' block-L
  residual is what the non-causal write already does; detecting at the closing token removes the look-ahead from
  DETECTION, but the WRITE still lands on past positions. This is more invasive than a forward write and is realizable
  in true decode only as a cached-K/V edit, not a forward residual write.
* **Number-only write (lost), in weights, by layer: 1-4% at every layer** (<= the audit's <=12% from the old Python
  version; confirmed in weights). Writing just the value at one position never works for the word problem; only
  replacing the whole defining clause (98%+ at L<=12) does. This matches the audit's "you must replace the relation."

## Disclosures
Inference path is fixed tensors (reader + ReLU circuit + big-M gated write) + the plain greedy decode loop + a tensor
cache reset per batch (like a KV cache). causal_close/causal_mean are fully causal (backward-only window). The
KVrewrite arm edits past positions (honest caveat above). Writer banks are the only fitted tensors (ridge on oracle
prompts). `causal.py` (main sweep), `kvrewrite.py` (past-state confirmation).
