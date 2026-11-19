# IFEval implant-count curve + transfer to Qwen3-1.7B / Qwen3-4B (overnight 2026-11-17/18)

Pod: the A100 pod A100 80GB, GPU 0, work dir `ifeval/heldout (work dir)` (venv the venv: torch 2.8.0, transformers 4.56.2, lm_eval IFEval checkers).
Laptop mirror of code/results: `(laptop mirror)`. No model jobs on the laptop. No commits.

## Inputs (frozen, unchanged)
- `frozen/circuit.py` md5 cd15769ce3f26bc5b5480e0f8864e4be, `frozen/tables.py` md5 78f6181bfa8d5db939c2f4a242ca96a6 (= frozen_MD5.txt).
- `frozen/vecs2_forcefix.pt` md5 04b8a1e6b54f5c11821c14f42a26e86c: the force_fix rows of the frozen vecs2.pt (md5 5f5f5422...), the only part
  circuit.py reads (release copy, ifeval/heldout; the laptop mirror has no frozen/ dir and the A40 pod is off limits).
- Working copies `circuit.py`, `tables.py`, `score.py` = byte copies of frozen/. `common.py`: one line changed, MODEL = env WSM_MODEL (default 0.6B).
- Host settings as Team D: Qwen3-{0.6B,1.7B,4B} (the chat/instruct checkpoints, NOT -Base), bf16, chat template with enable_thinking=False,
  greedy HF decode loop, bs 24, max_new 2048, batches = prompts sorted by tokenized length (same batches plain vs implant).

## Coordinator update (mid-run)
Every IFEval number is also reported on ALL 541 prompts, next to ODD-299 (never opened) and EVEN-242. Nothing built differently.

## Job 1 design: implant-count curve (0.6B)
Order (decided without the ODD half): each rule's instruction-strict gain (implants - plain, pass counts of that rule's
instruction id) on Team D's FRESH set (seed 1; Team D runs `fresh_plain`/`fresh_imp` with the frozen code, 720 of 840 prompts:
fresh_imp shard 5 never finished, OOM). Ties broken by the same gain on DEV (`dev_plain` vs `dev_imp3`). Script `rule_gain.py`.

| rank | rule | FRESH n | plain | imp | gain | DEV gain (tie-break) |
|---|---|---|---|---|---|---|
| 1 | nc (no comma) | 84 | 21 | 71 | +50 | +27 |
| 2 | para (N paragraphs) | 66 | 15 | 57 | +42 | +31 |
| 3 | repeat | 69 | 21 | 63 | +42 | +28 |
| 4 | forbid | 83 | 27 | 68 | +41 | +41 |
| 5 | nth (paragraph k starts with X) | 80 | 24 | 60 | +36 | +25 |
| 6 | caps | 74 | 22 | 56 | +34 | +23 |
| 7 | two responses | 69 | 33 | 62 | +29 | +13 |
| 8 | lower | 86 | 54 | 76 | +22 | +12 |
| 9 | end phrase | 77 | 63 | 73 | +10 | +10 |
| 10 | high (highlights) | 84 | 71 | 81 | +10 | +9 |
| 11 | quote | 70 | 49 | 58 | +9 | +5 |
| 12 | bul (bullets) | 79 | 63 | 72 | +9 | +4 |
| 13 | title | 101 | 100 | 101 | +1 | 0 |
| 14 | ps (postscript) | 79 | 75 | 75 | 0 | 0 |

Disabling a rule (`run_t.py`, env RULES): its latch wire(s) are multiplied by 0 where the enforcer reads them (a wrapper around
`Circuit.gates`; circuit.py itself is unchanged). Detectors are untouched. ps carries its P.P.S latch, para its '***'-mode latch.
Frozen coupling kept as is: nth enforcement is gated by the paragraph gate (g['nth'] *= g['pn']), and the paragraph gate is off
while two-responses is on; caps is off while lowercase is on. (para comes before nth in the order, so nth is never "dead".)

Exact curve with per-row masks (`curve2.py`, `run_t.py` env ROWCFG). Run(k) = implants with rules ORDER[:k]. A prompt's output
depends only on S(p) n ORDER[:k] (S = rules latched by the untouched detectors), and rows of a batch are independent at fixed batch
shapes. So for prompt p: k < its first latched rule -> output = the plain run; k >= its last latched rule -> output = the full
14-rule run; in between (only prompts with >= 2 latched rules) the prompt is regenerated in its ORIGINAL batch (same 24 rows, same
padding) with its own row masked to ORDER[:k]. One rerun of a batch serves one config per row; rows with nothing left to do get a
filler config ('all' or no rule) whose output must equal the full / plain run byte for byte (determinism check).
(A first version, `curve.py`, re-ran whole batches per k; it was replaced before use: ~150 batch runs vs 26.)

## Job 2 design: what is host-tied, and how it is re-solved
circuit.py reads from the host only: `model.lm_head.weight` (force directions E_t/g and the disclosed readout), `model.model.norm`
(gain g, eps), and the `force_fix` rows (solved d-dim directions for tokens whose E_t/g x 1e4 write does not win). Tables
(tables.py) depend only on the tokenizer (identical for Qwen3 0.6B/1.7B/4B, vocab 151,936), GAIN = 1e4 is a constant.
So the only thing to re-solve per host is force_fix (d = 1024 -> 2048 / 2560).
- `solve_forcefix.py` = the force_fix part of Team D's `solve_vecs2.py`, verbatim (veto vectors / kill map skipped: v2 never reads
  them). Data: the host's OWN plain greedy answers to OUR dev prompts (data/dev.jsonl, seed 0, 420 prompts; first 250 answers,
  <= 600 tokens each, teacher-forced). No IFEval prompt is used.
- Reproduction check on 0.6B (Team D's dev_plain answers): 3,665 tokens fail the plain write, 1,695 fixed = exactly the 1,695
  token ids of the frozen vecs2_forcefix.pt; per-row cosine to the frozen rows: median 1.0, min 0.99999 (not bit-identical: GPU/RNG).
  The frozen 0.6B file is used for all 0.6B runs.
- Silence on Team D's 30 "mention" negatives (`data/mentions.jsonl`), 0.6B on the A100 pod: 0/30 fire, 0/30 latched, 30/30 byte-identical.
- 4B force_fix (`vecs_q4.pt`, solved on Qwen3-4B's own plain dev answers `runs/q4_dev_plain.s{0,1}.jsonl`): 3,567 tokens fail the plain
  write, 1,629 fixed (the rest are byte-fallback/special pieces 124-187 etc. that also fail on 0.6B); with the rows in place the
  forced token wins on a random dev state for 149,706 / 151,644 vocab tokens (0.6B: 149,674).

## Throughput note
The circuit is ~1000 tiny kernels per decode step; with 7 processes time-slicing GPU 0 an implant batch took ~700 s (151 s alone).
At 04:08 all jobs were restarted as clients of a private CUDA MPS daemon (user process, own pipe dir a private MPS pipe dir; same kernels, so
outputs are unchanged; run.py resumes finished batches from its .partial pickles).

## Scorer note (found while merging the curves)
The official lm_eval IFEval scorer is not order-independent: for keys 1122 and 1129 (keywords:letter_frequency) a kwarg is missing in
the dataset and `build_description` fills it with Python `random`, so the verdict depends on how many prompts were scored before
(`detprobe.py`: 3 shuffled scoring orders; only these two keys vary, on every host). The 4B curve's k=0 (byte-identical to the plain
run) first scored 1 prompt differently from the plain run's own score. Fix: `score_t.py` calls `random.seed(key)` before each
prompt. ALL numbers below use the seeded scorer (re-scored everything; the FRESH order is unchanged under it). Effect vs unseeded:
0.6B plain ODD 57.19 -> 57.53 (key 1129 passes), 0.6B implants unchanged; 4B plain/implants ALL +0.18-0.37.
Also: key 1129 was listed as "broken" by the 0.6B implants under the unseeded scorer; that was this artifact.

## Result: 0.6B reproduction on the A100 pod (frozen circuit + frozen force_fix)
Team D (the A40 pod) ODD 56.19 -> 71.91 (+49/-2), EVEN 60.74 -> 73.97. Here (the A100 pod, seeded scorer) ODD 57.53 -> 71.24 (+42/-1),
EVEN 58.26 -> 73.14 (+38/-2), ALL 57.86 -> 72.09. The plain host already differs between the two GPUs (bf16 greedy decoding is not
bit-reproducible across GPU types), so the baselines differ; all comparisons below are within the A100 pod runs only.
Silent prompts (circuit wrote nothing) byte-identical to plain: 324/324 (ODD 182/182).

## Result: Job 1, implant-count curve (cumulative; order from FRESH, see table above)
Plan: latched-set sizes over 541 prompts {0: 193, 1: 263, 2: 76, 3: 8, 4: 1} (identical on all hosts); 95 (prompt, k) configs to
regenerate, packed into 26 reruns of original batches. fixed/broken = prompt-strict vs plain (k=0). k=14 = the full implant run.
Merge checks, each host: filler rows with all 14 rules = full run byte-identical 268/268; filler rows with no rule = plain run
byte-identical 250/250; latch sets of all 613 rows in the reruns = the full run's.

### Qwen3-0.6B
| k | rule added | ALL P-strict | ALL I-strict | ODD P-strict | ODD I-strict | EVEN P-strict | EVEN I-strict | ALL fixed/broken | ODD fixed/broken | EVEN fixed/broken |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | (plain) | 57.86 | 67.63 | 57.53 | 67.77 | 58.26 | 67.45 | +0/-0 | +0/-0 | +0/-0 |
| 1 | nc | 61.92 | 71.34 | 61.20 | 71.08 | 62.81 | 71.65 | +22/-0 | +11/-0 | +11/-0 |
| 2 | para | 63.40 | 73.02 | 62.88 | 72.85 | 64.05 | 73.23 | +30/-0 | +16/-0 | +14/-0 |
| 3 | repeat | 64.70 | 73.98 | 63.88 | 73.73 | 65.70 | 74.28 | +37/-0 | +19/-0 | +18/-0 |
| 4 | forbid | 67.47 | 76.14 | 66.89 | 75.94 | 68.18 | 76.38 | +53/-1 | +29/-1 | +24/-0 |
| 5 | nth | 67.84 | 76.38 | 67.22 | 76.16 | 68.60 | 76.64 | +57/-3 | +31/-2 | +26/-1 |
| 6 | caps | 67.84 | 76.38 | 67.56 | 76.38 | 68.18 | 76.38 | +58/-4 | +32/-2 | +26/-2 |
| 7 | two | 69.32 | 77.34 | 68.90 | 77.26 | 69.83 | 77.43 | +67/-5 | +36/-2 | +31/-3 |
| 8 | lower | 70.24 | 77.82 | 69.57 | 77.70 | 71.07 | 77.95 | +71/-4 | +38/-2 | +33/-2 |
| 9 | end | 70.61 | 78.06 | 69.90 | 77.92 | 71.49 | 78.22 | +73/-4 | +39/-2 | +34/-2 |
| 10 | high | 70.61 | 78.18 | 69.90 | 78.15 | 71.49 | 78.22 | +73/-4 | +39/-2 | +34/-2 |
| 11 | quote | 71.16 | 78.66 | 70.57 | 78.59 | 71.90 | 78.74 | +75/-3 | +40/-1 | +35/-2 |
| 12 | bul | 72.09 | 79.50 | 71.24 | 79.03 | 73.14 | 80.05 | +80/-3 | +42/-1 | +38/-2 |
| 13 | title | 72.09 | 79.62 | 71.24 | 79.25 | 73.14 | 80.05 | +80/-3 | +42/-1 | +38/-2 |
| 14 | ps | 72.09 | 79.62 | 71.24 | 79.25 | 73.14 | 80.05 | +80/-3 | +42/-1 | +38/-2 |

### Qwen3-4B (same order)
| k | rule added | ALL P-strict | ALL I-strict | ODD P-strict | ODD I-strict | EVEN P-strict | EVEN I-strict | ALL fixed/broken | ODD fixed/broken | EVEN fixed/broken |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | (plain) | 80.41 | 86.21 | 80.94 | 87.20 | 79.75 | 85.04 | +0/-0 | +0/-0 | +0/-0 |
| 1 | nc | 80.78 | 86.45 | 81.27 | 87.42 | 80.17 | 85.30 | +2/-0 | +1/-0 | +1/-0 |
| 2 | para | 80.78 | 86.45 | 81.61 | 87.64 | 79.75 | 85.04 | +3/-1 | +2/-0 | +1/-1 |
| 3 | repeat | 81.89 | 87.17 | 82.94 | 88.74 | 80.58 | 85.30 | +11/-3 | +7/-1 | +4/-2 |
| 4 | forbid | 82.99 | 88.13 | 83.61 | 89.18 | 82.23 | 86.88 | +18/-4 | +10/-2 | +8/-2 |
| 5 | nth | 82.81 | 88.01 | 83.61 | 89.18 | 81.82 | 86.61 | +19/-6 | +11/-3 | +8/-3 |
| 6 | caps | 82.99 | 88.13 | 83.61 | 89.18 | 82.23 | 86.88 | +20/-6 | +11/-3 | +9/-3 |
| 7 | two | 82.99 | 88.13 | 83.61 | 89.18 | 82.23 | 86.88 | +20/-6 | +11/-3 | +9/-3 |
| 8 | lower | 83.36 | 88.37 | 83.95 | 89.40 | 82.64 | 87.14 | +21/-5 | +12/-3 | +9/-2 |
| 9 | end | 83.55 | 88.49 | 84.28 | 89.62 | 82.64 | 87.14 | +21/-4 | +12/-2 | +9/-2 |
| 10 | high | 83.92 | 88.85 | 84.28 | 89.62 | 83.47 | 87.93 | +23/-4 | +12/-2 | +11/-2 |
| 11 | quote | 84.29 | 89.09 | 84.95 | 90.07 | 83.47 | 87.93 | +24/-3 | +13/-1 | +11/-2 |
| 12 | bul | 84.66 | 89.45 | 85.95 | 90.73 | 83.06 | 87.93 | +28/-5 | +16/-1 | +12/-4 |
| 13 | title | 84.66 | 89.45 | 85.95 | 90.73 | 83.06 | 87.93 | +28/-5 | +16/-1 | +12/-4 |
| 14 | ps | 84.84 | 89.57 | 86.29 | 90.95 | 83.06 | 87.93 | +29/-5 | +17/-1 | +12/-4 |

### Qwen3-1.7B (same order; done because there was time)
| k | rule added | ALL P-strict | ALL I-strict | ODD P-strict | ODD I-strict | EVEN P-strict | EVEN I-strict | ALL fixed/broken | ODD fixed/broken | EVEN fixed/broken |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | (plain) | 68.76 | 76.74 | 71.91 | 79.47 | 64.88 | 73.49 | +0/-0 | +0/-0 | +0/-0 |
| 1 | nc | 71.53 | 79.50 | 74.92 | 82.56 | 67.36 | 75.85 | +15/-0 | +9/-0 | +6/-0 |
| 2 | para | 71.90 | 79.86 | 74.92 | 82.56 | 68.18 | 76.64 | +17/-0 | +9/-0 | +8/-0 |
| 3 | repeat | 71.90 | 79.74 | 74.58 | 82.34 | 68.60 | 76.64 | +18/-1 | +9/-1 | +9/-0 |
| 4 | forbid | 74.68 | 81.65 | 77.26 | 84.11 | 71.49 | 78.74 | +34/-2 | +18/-2 | +16/-0 |
| 5 | nth | 74.49 | 81.65 | 76.92 | 83.89 | 71.49 | 79.00 | +34/-3 | +18/-3 | +16/-0 |
| 6 | caps | 74.31 | 81.53 | 76.92 | 83.89 | 71.07 | 78.74 | +34/-4 | +18/-3 | +16/-1 |
| 7 | two | 74.86 | 82.01 | 77.26 | 84.11 | 71.90 | 79.53 | +38/-5 | +19/-3 | +19/-2 |
| 8 | lower | 75.23 | 82.25 | 77.59 | 84.33 | 72.31 | 79.79 | +40/-5 | +20/-3 | +20/-2 |
| 9 | end | 76.16 | 82.85 | 78.60 | 84.99 | 73.14 | 80.31 | +45/-5 | +23/-3 | +22/-2 |
| 10 | high | 76.52 | 83.21 | 79.26 | 85.65 | 73.14 | 80.31 | +47/-5 | +25/-3 | +22/-2 |
| 11 | quote | 77.08 | 83.57 | 79.60 | 85.87 | 73.97 | 80.84 | +49/-4 | +25/-2 | +24/-2 |
| 12 | bul | 78.19 | 84.65 | 80.94 | 86.75 | 74.79 | 82.15 | +55/-4 | +29/-2 | +26/-2 |
| 13 | title | 78.19 | 84.65 | 80.94 | 86.75 | 74.79 | 82.15 | +55/-4 | +29/-2 | +26/-2 |
| 14 | ps | 78.19 | 84.65 | 80.94 | 86.75 | 74.79 | 82.15 | +55/-4 | +29/-2 | +26/-2 |

Reading: on 0.6B the first four rules (nc, para, repeat, forbid) carry 9.4 of the 13.7 ODD pp (k=4: 66.89 vs 57.53 plain); caps adds nothing net on ALL
(+1 fixed, +1 broken; forbid is the first rule to break a prompt), high/title/ps add 0 prompt-strict (they are near ceiling in plain). On 4B nc is nearly worthless (+1 ODD)
because 4B already avoids commas; repeat and forbid carry most of its gain; ps adds +1 at k=14. Broken prompts appear at nth (k=5)
on all hosts and at bul (k=12) on 4B.

## Result: Job 2, transfer of the 14 frozen rules to Qwen3-1.7B and Qwen3-4B
Same circuit.py/tables.py (md5 unchanged), same decode loop / hook / readout, same bs 24 batches, max_new 2048, thinking off.
Re-solved per host: ONLY force_fix (`vecs_q17.pt`, `vecs_q4.pt`), from that host's own plain answers to our dev prompts.
- 1.7B force_fix: 3,134 tokens fail the plain write, 1,190 fixed; forced token wins on a random dev state for 149,700 / 151,644 tokens.
- 4B force_fix: 3,567 fail, 1,629 fixed; 149,706 / 151,644.  (0.6B frozen: 1,695 rows; 149,674.)
- Forced-token diagnostic during the IFEval runs (circuit ffail = forced token did not win): 0 on 0.6B, 0 on 1.7B, 0 on 4B.
- Latch sets (detector output) identical on all three hosts for all 541 prompts (detectors read prompt tokens only).
- Prompts on which the circuit wrote anything: 0.6B 217, 1.7B 228, 4B 188 of 541 (it acts only when the host is about to break a rule).

| host | split | n | plain P-strict | +impl P-strict | dP | plain I-strict | +impl I-strict | dI | fixed/broken | silent identical | trunc plain/impl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.6B | ALL | 541 | 57.86 | 72.09 | +14.23 | 67.63 | 79.62 | +11.99 | +80/-3 | 324/324 | 26/28 |
| 0.6B | ODD | 299 | 57.53 | 71.24 | +13.71 | 67.77 | 79.25 | +11.48 | +42/-1 | 182/182 | 16/16 |
| 0.6B | EVEN | 242 | 58.26 | 73.14 | +14.88 | 67.45 | 80.05 | +12.60 | +38/-2 | 142/142 | 10/12 |
| 1.7B | ALL | 541 | 68.76 | 78.19 | +9.43 | 76.74 | 84.65 | +7.91 | +55/-4 | 313/313 | 28/33 |
| 1.7B | ODD | 299 | 71.91 | 80.94 | +9.03 | 79.47 | 86.75 | +7.28 | +29/-2 | 176/176 | 15/19 |
| 1.7B | EVEN | 242 | 64.88 | 74.79 | +9.92 | 73.49 | 82.15 | +8.66 | +26/-2 | 137/137 | 13/14 |
| 4B | ALL | 541 | 80.41 | 84.84 | +4.44 | 86.21 | 89.57 | +3.36 | +29/-5 | 353/353 | 11/12 |
| 4B | ODD | 299 | 80.94 | 86.29 | +5.35 | 87.20 | 90.95 | +3.75 | +17/-1 | 201/201 | 10/10 |
| 4B | EVEN | 242 | 79.75 | 83.06 | +3.31 | 85.04 | 87.93 | +2.89 | +12/-4 | 152/152 | 1/2 |

Per-rule instruction-strict passes, ALL (plain -> implants, of n)

| rule | n | 0.6B | 1.7B | 4B |
|---|---|---|---|---|
| nc | 66 | 31 -> 62 (+31) | 40 -> 61 (+21) | 64 -> 63 (-1) |
| lower | 39 | 28 -> 31 (+3) | 35 -> 37 (+2) | 35 -> 36 (+1) |
| caps | 25 | 15 -> 16 (+1) | 18 -> 18 (+0) | 20 -> 21 (+1) |
| quote | 41 | 33 -> 36 (+3) | 39 -> 40 (+1) | 39 -> 40 (+1) |
| end | 26 | 24 -> 26 (+2) | 20 -> 24 (+4) | 26 -> 26 (+0) |
| ps | 26 | 26 -> 26 (+0) | 25 -> 25 (+0) | 24 -> 25 (+1) |
| forbid | 49 | 23 -> 42 (+19) | 28 -> 45 (+17) | 35 -> 44 (+9) |
| repeat | 41 | 26 -> 34 (+8) | 35 -> 36 (+1) | 30 -> 38 (+8) |
| two | 24 | 11 -> 19 (+8) | 17 -> 19 (+2) | 23 -> 23 (+0) |
| para | 27 | 13 -> 21 (+8) | 17 -> 21 (+4) | 21 -> 22 (+1) |
| nth | 12 | 4 -> 11 (+7) | 9 -> 10 (+1) | 9 -> 10 (+1) |
| title | 37 | 35 -> 36 (+1) | 36 -> 36 (+0) | 37 -> 37 (+0) |
| high | 48 | 45 -> 46 (+1) | 46 -> 48 (+2) | 45 -> 48 (+3) |
| bul | 31 | 22 -> 29 (+7) | 23 -> 30 (+7) | 26 -> 30 (+4) |
| other | 342 | 228 -> 229 (+1) | 252 -> 256 (+4) | 285 -> 284 (-1) |

Per-rule instruction-strict passes, ODD (plain -> implants, of n)

| rule | n | 0.6B | 1.7B | 4B |
|---|---|---|---|---|
| nc | 36 | 19 -> 35 (+16) | 21 -> 34 (+13) | 35 -> 35 (+0) |
| lower | 23 | 18 -> 19 (+1) | 22 -> 23 (+1) | 19 -> 20 (+1) |
| caps | 12 | 5 -> 6 (+1) | 8 -> 8 (+0) | 11 -> 11 (+0) |
| quote | 23 | 21 -> 22 (+1) | 23 -> 23 (+0) | 22 -> 23 (+1) |
| end | 13 | 12 -> 13 (+1) | 10 -> 12 (+2) | 13 -> 13 (+0) |
| ps | 14 | 14 -> 14 (+0) | 14 -> 14 (+0) | 12 -> 13 (+1) |
| forbid | 27 | 10 -> 21 (+11) | 15 -> 24 (+9) | 21 -> 24 (+3) |
| repeat | 21 | 15 -> 18 (+3) | 19 -> 18 (-1) | 15 -> 20 (+5) |
| two | 11 | 5 -> 9 (+4) | 8 -> 9 (+1) | 10 -> 10 (+0) |
| para | 16 | 7 -> 12 (+5) | 12 -> 12 (+0) | 13 -> 14 (+1) |
| nth | 6 | 2 -> 5 (+3) | 5 -> 5 (+0) | 4 -> 5 (+1) |
| title | 19 | 17 -> 18 (+1) | 18 -> 18 (+0) | 19 -> 19 (+0) |
| high | 28 | 26 -> 27 (+1) | 26 -> 28 (+2) | 28 -> 28 (+0) |
| bul | 14 | 11 -> 13 (+2) | 10 -> 13 (+3) | 11 -> 14 (+3) |
| other | 190 | 125 -> 127 (+2) | 149 -> 152 (+3) | 162 -> 163 (+1) |

### Broken prompts (prompt-strict pass in plain, fail with implants; seeded scorer)
EVEN keys may be read; for ODD keys only the key, latched rules and failing checker id are listed (prompts not opened).
- 0.6B (3): 3407 ODD (caps; capital_word_frequency), 1314 EVEN (caps false fire on "Use some words in all caps ... at most 10 times":
  whole answer capitalised -> capital_word_frequency fails), 1332 EVEN (two_responses; table answer, the divider/second response breaks it).
- 1.7B (4): 1893 ODD (end+forbid; end_checker; implant run hit the 2048 cap), 2305 ODD (repeat+title; repeat_prompt), 1314 EVEN, 1332 EVEN.
- 4B (5): 2305 ODD (repeat+title; repeat_prompt), 1908 EVEN (bul; placeholders; hit the 2048 cap), 1944 EVEN (nc+repeat: the repeat
  detector captured the wrong span, the instruction sentence "Before you answer the following request, repeat it ..." instead of the
  request after it; the copy contains a comma and the repeat copy is not filtered by the comma block; the host stopped right after the
  copy; this one prompt is the 4B no_comma 64 -> 63), 2078 EVEN ("exactly one bullet" + a quoted "* Bullet point" example: forced '-'
  line plus the host's own '*' line = 2 bullets), 2284 EVEN (bul; number_words).

### Silence checks per host
- IFEval prompts on which the circuit wrote nothing (fires = 0), byte-identical to that host's plain run: 0.6B 324/324, 1.7B 313/313,
  4B 353/353 (ODD: 182/182, 176/176, 201/201).
- Team D's 30 "mention" negatives (instructions mentioned, not requested): 0.6B, 1.7B, 4B each 0 fire, 0 latch, 30/30 byte-identical.
- Curve reruns: rows with every rule masked off = plain, byte for byte, 250/250 on each of 0.6B, 1.7B, 4B.

## Disclosed non-weight pieces (unchanged from Team D)
1. the greedy HF decode loop (run.py; identical for plain and implant);
2. the forward pre-hook on the host's final norm that adds circuit.write(h), and its on/off switch (on only during generation);
3. the readouts of the host's final state through the frozen LM head (top token is end-of-turn / has a newline / is in the blocked
   set; masked hard-max = the host's best token outside the blocked set);
plus, as before, the circuit is batched tensor code (not exported ReLU matrices). Added by this experiment, not by the rules:
4. the curve's enable mask (latch wires x 0/1, a wrapper on Circuit.gates in run_t.py; per row via ROWCFG);
5. WSM_MODEL host selection in common.py; BATCHES (rerun chosen batch ids) in run_t.py.

## Problems hit
- GPU 0 over-subscription: 12 processes OOM'd and restarted batches; moved to a small memory-reserving scheduler (`sched.py`,
  reserve 9 / 12.5 / 19.5 GB per 0.6B / 1.7B / 4B job, cap 80 GB) and a private MPS daemon. No result depends on this.
- `pkill -f` patterns matched my own ssh command line three times (killed the remote shell, not jobs); fixed by running kill
  commands from script files.
- The circuit is CPU/launch-bound (~1000 small kernels per decode step); implant batches took 3-15 min each under sharing.
- vecs2.pt is not on the laptop mirror (the A40 pod only); used the release copy vecs2_forcefix.pt (same force_fix tensors, documented).
- Team D's fresh_imp shard 5 never finished (OOM on the A40 pod), so the order uses 720 of 840 FRESH prompts.

## Files
Pod `ifeval/heldout (work dir)`: runs/q{06,17,4}_ifeval_{plain,imp}.s{0..3}.jsonl (generations), runs/c{06,17,4}_k{0..14}.jsonl (curve
points), runs/*_score.{json,txt}, vecs_q17.pt, vecs_q4.pt, runs/q{17,4}_dev_plain.s{0,1}.jsonl (solver data).
Laptop `(laptop mirror)`: code/ (run_t.py, curve2.py, score_t.py, solve_forcefix.py, rule_gain.py, sched.py,
fmt_*.py, broken.py, detprobe.py, common.py), results/ (all score JSON/TXT, plans, tables).
