# reread: forced re-read of the question at every new attempt (impl-reread)

## Design
In the S3-serial stack, each attempt k>=1 starts with a written queue: `['\n'] "Wait, let me solve it again.\n"`. The weight-space
forget mask hides the earlier attempts, so attempt k sees the 8-shot prompt, the question, its queue and its own tokens.
The question therefore sits far back in context. The implant makes the writer (the compiled queue in `SerialEngine._act`) emit
a **per-row** queue containing an exact copy of the row's question token ids. It is a deterministic copy with no model call,
filled into rowid-indexed buffers at wave start (`rr_fill`) and selected inside the CUDA graph by `torch.where(enl, rr_q[0][r], rr_q[1][r])`.

| variant | gate | attempt k>=1 queue (k=0 unchanged) |
|---|---|---|
| V1 `full` | `SER_RR=full` | `"Wait, let me re-read the question:"` + the question ids copied from `Question: q\nAnswer:` (the last copied token carries the `\n`, e.g. `?\n`) |
| V2 `ask` | `SER_RR=ask` | same phrase + only the final sentence (the tokens after the last sentence end, skipping decimals and Mr./Mrs.) |
| V1b `qa` | `SER_RR=qa` | the base phrase `"Wait, let me solve it again.\n"` + the row's whole `Question: q\nAnswer:` ids, so the attempt starts after `Answer:` exactly as attempt 0 does |
| V3 `phrase` (control) | `SER_WAIT=...` | the fixed phrase `"Wait, let me read the question carefully and solve it again.\n"`, no copy |

Mechanics:
- Queue buffer `QL` goes from 48 to 320 when `SER_RR` is set. The longest queue is 181 tokens.
- The token budget was raised to `STK_MAXN=7936` and `SAN_SMAX=8192` (S_max), in `run.sh`.
- Forget mask: the segment label increments at `do_wait`, before the first queue token. Queue tokens therefore carry the new attempt's label, so they are visible to that attempt and hidden from later ones. Earlier attempts stay hidden.
- Positions: the position register (`posfix`) restarts the new segment at the question end, and the queue tokens and the attempt continue from there.
- Answer cap: `astart = glen + per-row queue length`, so the 320-token attempt cap counts only generated tokens, as before.
- Register reader: queue tokens are not read (`live` needs `qp >= ql`), so the numbers in the copied question never enter the answer register.

Code:
- `job2/` is the canonical copy used for all full runs: `stk/serial_stack.py` (the `RR`, `rr_ids`, `rr_fill` and `_act` hunks), `run.sh` and `test_reread.py`. `job/` holds the first copy (full/ask/phrase only), used for the smoke run.
- `analyze.py` builds the tables below.
- With no gate set, the code path is the original writer: `QL=48` and the constant `WAIT`.

## Tests and exactness
- `test_reread.py`, a CPU test that lifts `rr_ids`/`rr_fill` with `ast` and runs the real Qwen tokenizer on all 900 dev and dev2 questions, in both newline forms. **PASS** (it also runs at the start of every job).
  - Each queue is the phrase plus a contiguous slice of the row's ids.
  - The decoded text matches exactly: `phrase + ' ' + q + '\n'` for full, `WAIT + 'Question: q\nAnswer:'` for qa, and the question tail for ask.
  - The rowid-indexed buffers are correct for G=1 and G=2.
- **Exactness of attempt 0.** Rows of wave 0 (items 0–127, RNG-aligned with the base run) must have tokens bit-identical to the base `gen/out` run through the end of attempt 0, including S2F repairs, and the same `end[0]`.
  - Result: **256/256 per set and variant**, i.e. 128 rows × 2 seeds, for full, qa and phrase on dev and dev2.
  - The smoke run also gave 128/128 for full and 32/32 for ask and phrase.
  - Later waves are not RNG-aligned, because the sampler seed counter is per step and the attempts are longer.

## Results
Runs: N=8, R=1, T=1, seeds 7 and 11, dev600 and dev2. Job j23213 took 44 min on one B200. The smoke run was j23206.
- Attempt ok = register valid and value == gold.
- Fixes and breaks are item-level against the same-seed base run, summed over both seeds.
- tok/q counts the whole stream. gen tok/q excludes written queue tokens.

| variant | set | att acc k=0 | **att acc k>=1** | Δ k>=1 | vote | oracle@8 | fixes/breaks | tok/q | gen tok/q |
|---|---|---|---|---|---|---|---|---|---|
| base | dev | 82.25 | 80.79 | | 88.42 | 93.33 | | 1227 | 1153 |
| base | dev2 | 78.50 | 79.33 | | 85.33 | 93.33 | | 1243 | 1169 |
| V1 full | dev | 81.92 | 76.31 | **−4.5** | 88.50 | 94.92 | 36/35 | 1684 | 1191 |
| V1 full | dev2 | 79.00 | 74.71 | **−4.6** | 86.00 | 94.00 | 25/21 | 1725 | 1218 |
| V1b qa | dev | 81.25 | 80.14 | −0.7 | 88.92 | 94.00 | 29/23 | 1669 | 1157 |
| V1b qa | dev2 | 79.00 | 78.05 | −1.3 | 84.83 | 93.83 | 21/24 | 1689 | 1163 |
| V3 phrase | dev | 81.67 | 79.90 | −0.9 | 88.17 | 94.50 | 30/33 | 1395 | 1286 |
| V3 phrase | dev2 | 79.50 | 77.95 | −1.4 | 86.17 | 93.50 | 22/17 | 1445 | 1335 |
| V2 ask (smoke, dev 0–31, seed 7) | dev32 | 68.75 | 70.98 | −1.3 (n=224 att.) | 84.38 (base 81.25) | 90.6 | | 1567 | 1369 |

k=0 differs from base only through RNG in waves after the first, so it is a noise yardstick (±0.5).

Per-attempt accuracy (k>=1) by atlas item category, dev and dev2 pooled, both seeds, base → variant on the same items:

| category | full | qa | phrase |
|---|---|---|---|
| relation (n=3122) | 64.0 → 61.5 | 64.0 → 63.6 | 64.0 → 63.2 |
| dropped (1344) | 68.2 → 65.1 | 68.2 → 67.3 | 68.2 → 67.2 |
| operand (714) | 66.5 → 68.2 | 66.5 → 69.3 | 66.5 → 70.9 |
| target (588) | 77.9 → 72.3 | 77.9 → 77.4 | 77.9 → 78.2 |
| early_stop (98) | 54.1 → 41.8 | 54.1 → 55.1 | 54.1 → 45.9 |
| not mixed (6272) | 94.9 → 88.5 | 94.9 → 93.6 | 94.9 → 93.1 |

Other observations:
- **Validity and repairs.** Attempts with k>=1 that end without a valid answer rise only slightly: 0.8 → 1.2 % (full), 0.9 % (qa) and 1.1 % (phrase). Sanity repairs per question double under full, from 0.52 to 1.09. The accuracy drop is mostly **valid but wrong** answers.
- **Style shift (full).** After `"Wait, let me re-read the question: …"` the host switches to a chatty monologue register. Openers such as "Okay, so…", "Okay, let's…" and "Let's break…" replace the GSM-style "We know that…" and "First, we…". Accuracy falls even on always-right items (94.9 → 88.5) and on target items (77.9 → 72.3), the very category a re-read was meant to fix.
- **Diversity.** Oracle@8 rises a little (+0.2 to +1.6), but the vote does not gain: dev2 full +0.7, dev qa +0.5, everything else within ±0.5. Fixes and breaks are balanced, 36/35 and 25/21.

## Conclusion
**Negative.** A forced re-read does not raise per-attempt accuracy on either set:
- Copying the question after a "re-read" cue (V1) costs about 4.5 points of per-attempt accuracy.
- Copying it in the prompt's own `Question:/Answer:` format (V1b) costs about 1 point, the same as the fixed-phrase control (V3).
- The ask-only copy (V2) was smoke-tested only (−1.3 on 224 attempts) and was not run in full, because V1 and V1b already show that the question copy itself brings no gain.

Per-attempt accuracy hits no category in a way that tracks the copy:
- Operand improves by 2–4 points for all three variants, including the no-copy control, so this is the phrase or noise, not the re-read.
- Relation, dropped and target fall.
- The question being far back in context is therefore not what causes the misreadings. Attention already reaches the question (the forget mask leaves it visible), and any change to the attempt phrase pushes the host off its RL-trained answer distribution.

Vote accuracy is unchanged within noise, while tokens grow by +37 % (full, qa) or +14 % (phrase). Next step: do not pursue text-level re-read writers. Levers for the semantic misreadings would have to act at the divergence point (hidden-state probes), not through context recency.
