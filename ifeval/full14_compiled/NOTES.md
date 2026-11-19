# IFEval implants: "compile to weights" + bit-identity audit (2026-11-18)

Goal given: make the 14 IFEval rule implants for frozen Qwen3-0.6B fully non-programmatic
(compiled into explicit weight matrices) and prove the compiled version is BIT-IDENTICAL to the
current tensor-code version (`circuit.py`) on all 541 IFEval prompts, so the published
58.2->72.8 / 67.6->80.5 scores carry over unchanged.

Pod: the A100 pod (work under (pod work dir)/ only). GPU0 was ~60/80GB, 100% util (live RL ablation) the whole
session -> all checks here were run on CPU so as not to starve those jobs. No laptop compute.

## Executive finding (honest)
A bit-identical compile of the FULL 14-rule `circuit.py` into the pure
{matmul, ReLU, argmax-attention, gather} weight executor is NOT achievable, for two separate
reasons, one representational and one fundamental:

1. There is no existing faithful compile of `circuit.py`. The only weight-space compile in the
   repo (`ifeval_compiled/compiled.py`, "Phase 2") covers 4 of the 14 rules
   (no_comma, lower, caps, quote) and is a *separate, simpler reimplementation* (a K=16-token
   window reader + two ReLU stacks + latch/turn hard-attention heads), NOT a compile of
   `circuit.py`'s deep context-sensitive detection. Its own numbers are ~56.9->61.9 prompt-strict
   (SUMMARY "IFEval COMPILED"), i.e. DIFFERENT from `circuit.py`'s 57.9->72.1. So even on those 4
   rules it is not bit-identical to `circuit.py`, and the 14-rule published scores do NOT transfer
   to it.

2. One rule cannot be compiled exactly into a bounded linear/ReLU net at all. The forbidden-word
   rule (`forbid`) identifies words by a rolling polynomial hash
   `h <- (h*POWT + HT) mod (2^31-1)`, base HB=131 (tables.py lines 234-241; circuit.py HP=2147483647,
   lines 428-435, 632-633, 718-721). With h and POWT each up to ~2^31, the intermediate product
   reaches ~2^62. A ReLU/linear network is exact only on integers inside the float mantissa:
   2^24 (float32), 2^53 (float64). 2^62 exceeds even float64, so the mod reduction
   `a - HP*floor(a/HP)` cannot be evaluated exactly, and the exact collision behaviour of `forbid`
   is not reproducible in a weight matrix net. `circuit.py` sidesteps this by running the hash in
   **int64 tensors** (integer mul+mod, exact to 2^63) — which is precisely the op that is outside
   {matmul, ReLU, argmax-attn, gather}. This is also why the prior Phase-2 work explicitly DROPPED
   forbid, repeat, two, paras, nth.

## What IS true and verified
### circuit.py is already non-programmatic in the control-flow sense
Static audit of the entire inference path (`step`, `write`, `gates`, `_answer_step`, `_triggers`,
`_cue_neg`, `classes`, `prog_token`, `case_seq`, `emit_token`, `force`, `logits`): ZERO
data-dependent Python control flow, no regex, no string ops, no `.item()`. Every Python `if`/`for`
iterates a compile-time-constant rule/program list (rule names, program kinds 1..15) or is
one-time graph setup (`if fx`, `if name not in R`). The only `.decode()`/`.tolist()` live in
`report()`, a diagnostic logger NOT on the write path. So `circuit.py` is one fixed tensor graph
G applied identically to every prompt and every token; "programmatic" here is a *representation*
(torch expressions) not *data-dependent branching*.

Op inventory of G (counts over circuit.py): where/select 207, maximum(OR) 97, clamp 50, plus
elementwise +,-,*,==,>=,<=,>,<; gather/table-lookup; matmul (pattern matcher `@PC`, readout `@E.T`);
argmax over vocab (readout top + masked-argmax alt); rmsnorm (rsqrt,mean); remainder x2 (parity
mod-2, and the mod-2^31-1 hash); sort x2 (emitter prefix/suffix hole compaction). Mapping to the
target set: gathers/copy-heads -> gather; pattern matcher + readout -> matmul; top/alt token ->
argmax-attention (the task's "best pick that isn't X"); boolean scoping/gates (AND=mul/relu,
OR=max, NOT=1-x, select=gated sum, compare=step-threshold) and the small counters (paras/bullets/
highlights/nth all <=20) -> linear+ReLU (the HDL below proves this constructively); parity-mod-2 ->
XOR recurrent cell -> ReLU. The ONLY pieces that do not map are the **mod-2^31-1 hash** (reason 2)
and `torch.sort` (emitter hole-compaction; a fixed small permutation, replaceable but not in the
minimal set).

### The real weight-space compile (4 rules) and its executor op set
`ifeval_compiled/` (wsm_hdl.py + compiled.py) is a genuine compiler: boolean/integer "wires" as
affine forms (`Lin`) -> ReLU units with CSE (`Net`) -> dense ReLU layers (`compile_net`) stored as
sparse CSR weight buffers (`ReLUNet`). It even compiles multi-digit integer arithmetic to ReLU
(carry-lookahead `resolve/normalize/cmp_ge`). Architecture per token:
  reader: token embedding --(linear map rw,rb; exact 0/1, checked on whole vocab; 0/1 lookup rows
          for non-separable groups)--> symbol flags
  stack1: ReLU net over a K=16 flag window -> rule pattern matches + role-header + first-answer-pos
  heads : hard/argmax attention (latch = most-recent match; turn = role at most-recent header)
  stack2: ReLU net -> fire flags
  write : fire @ V  (V = solved veto directions + the '"' emit direction; silent => exactly 0)
Generic executor op set: {linear (reader), sparse-matmul + ReLU (stacks), argmax/hard-attention
(heads), gather + fixed-row linear (write)}, plus the disclosed LM-head readout for quote-close /
end-of-turn. Identical for every rule.

VERIFICATION (CPU, the A100 pod, Qwen3-0.6B embeddings; code/fast_verify.py):
- READER exact on all 151,936 vocab rows: TRUE (the linear reader rw,rb + 0/1 lookup-row fallback
  reproduces the ground-truth symbol flags on every token). 13 of 25 symbol groups are exactly
  linearly separable (closed-form max-margin direction); 12 use a fixed 0/1 lookup row
  (NEG, BE, FILL, CASE, CAPITAL, ALLW, WORDS, WRAP, QUOT, IMSTART, ENDTHINK, QCLOSE) — each such
  row is itself a weight (a one-hot/embedding row), consistent with the disclosed reader design.
- STACK1 (pattern/role/first-pos ReLU net): 33 ReLU units, 148 nonzero weights (1,359 dense).
  STACK2 (fire gates): 6 ReLU units, 19 nonzero weights (69 dense). Both stored as sparse CSR
  weight buffers in ReLUNet; executor is sparse-matmul + ReLU only.
- EXECUTOR self-consistency (incremental step() vs vectorized run_seq(), 40 real IFEval prompts):
  max|fire|=0, max|write|=0 (the two executor forms are exactly identical).
- EXPORT single state_dict (reader + both stacks): <<EXPORT>>  (+ the write matrix V = 5 x 1024
  solved directions added at load). Saved: (pod work dir)/compiled_rulekeeper_statedict.pt.

## Disclosed non-weight pieces (unchanged from the running system)
decode loop; final-norm pre-hook + its on/off switch; LM-head readout (top token identity + masked
argmax for "best token outside the blocked set"); force/veto directions solved on own dev states.

## Bottom line vs the asked deliverable
- 541/541 bit-identical compiled-vs-circuit.py: NOT achieved and not achievable in the target op
  set (reasons 1 & 2). No such compiled artifact exists to run.
- A real weight-matrix compile exists for 4 rules with a clean {linear, ReLU, argmax-attn, gather}
  executor, verified exact reader + self-consistent executor (below); but it is a reimplementation,
  so its scores (~56.9->61.9) are its own, not circuit.py's 14-rule 57.9->72.1.
- circuit.py is already free of data-dependent control flow / regex / string ops (audited); the
  residual gap to "pure weight matrices" is the int64 mod-2^31-1 hash and torch.sort.

## Files
- code/: compiled.py (4-rule RuleKeeperCircuit + reader + stacks + heads), wsm_hdl.py (Lin/Net/
  compile_net/ReLUNet: the boolean-&-integer-wire -> dense-ReLU-matrix compiler), rules.py
  (Phase-1 reference), detect.py, gen_c.py (compiled-only generation), check_compiled.py,
  fast_verify.py + fast_verify.log (the CPU verification run above). Copied from pod
  (pod work dir)/ (Nov 17 prior work) for the record.
- Exported weights (pod): (pod work dir)/compiled_rulekeeper_statedict.pt (15.3 MB).
- Source of truth for the full 14-rule system: ifeval/heldout/
  circuit.py (md5 cd15769ce3f26bc5b5480e0f8864e4be), tables.py (78f6181...).

---------------------------------------------------------------------------------------------------------------------------
## 2026-11-18 (evening): FULL 14-rule compile of circuit.py into plain weight tensors: 541/541 byte-identical (code/full14/)

This SUPERSEDES the "Executive finding" above. Reason 2 there is wrong: the mod-(2^31-1) hash CAN be computed exactly with
ReLU and matmul. You never form h*POWT. You split h into its 31 bits with a ReLU staircase. You multiply each bit by a fixed
residue (2^i * POWT mod p), which is a row of a fixed table. The sum is below 31p < 2^36, and one more ReLU staircase reduces
it mod p. Every number stays far below 2^53, so float64 holds it exactly.
Reason 1 is also fixed: code/full14 compiles circuit.py itself, line by line (not a reimplementation).

### Construction
- `sym.py`: symbolic integer wires. Each value is an affine form over wires (wsm_hdl.Lin) plus a proven interval.
  Non-linear ops are lowered to ReLU units that are exact on integers inside the interval:
    max(a,b) = a + relu(b-a)
    where(c,x,y) = y + relu(d - L - (U-L)(1-c)) + L*c   (d = x-y in [L,U])
    [x>=k] = relu(x-k+1) - relu(x-k)
    AND = relu(a+b-1)
  Other node kinds: gather (a row of a fixed table, selected by an integer wire); attn (hard attention over the token memory);
  memwrite (append to that memory); vocab (see below). CSE comes from wsm_hdl's Net idea. The compiler prunes dead nodes,
  levels the graph and writes sparse layers.
  STATIC EXACTNESS CERTIFICATE (checked at compile time): for every affine form the executor evaluates,
  sum|coef|*max|wire| + |c| < 2^52. Worst case: 7.6e11 (STEP), 6.7e12 (WRITE). So float64 matmul + ReLU is exact in any
  summation order.
  The vocab-plane values are below 2^41, so they are exact as well.
  Bug found on the way: a ReLU elided on a non-negative wire let affine forms grow to coefficients near 2^61, and partial sums
  lost bits. Fix: materialise the hash bits/remainders on fresh wires. The certificate now guards against this.
- `circ14.py`: transliteration of circuit.py (md5 cd15769c...) with circuit.py line numbers in comments. It produces 2 netlists:
  STEP (414 registers + token id -> 414 registers) and WRITE (registers + host state -> registers + write-row index).
  The 414 registers are all of circuit.py's state, under its attribute names.
  (a) hash, in all 4 places:
      - forbidden slots: mux by slot index, then bits31 + PCOEF[36 powers x 31 bits] gather + modp
      - hcur: the same construction
      - blocked mask: hcur*131^l mod p for the 36 distinct lengths l (register plane), then a fixed (V x 36) one-hot
        matmul + HT[v], one ReLU mod step, and a 3-ReLU equality against the 6 slot targets
      - anti-repeat: HT[v] vs pf, 3-ReLU equality, AND with (LEAD[v] or hcur==0)
      The construction gives the SAME hash values as the int64 code, collisions included. So the decisions are identical by
      construction; no new hash was introduced.
  (b) torch.sort hole compaction is replaced by prefix counts + one-hot selection (`compact()`).
  (c) readout: a copy of the LM-head rows (bf16, all 151,936, because alt is an argmax over the whole vocabulary), the
      final-norm gain and eps are stored in the implant. The force rows are a fixed table F (V+1 x 1024). Row V is zero, so
      W = F[where(active, tokf, V)] is a pure gather.
  Init detail found by lockstep: circuit.reset leaves pre/suf at 0, not -1.
  s_commit can reach 2 (line 446); its register range is 0..2.
- `execu.py`: export to a state_dict of plain tensors (op order encoded in key names) and the generic Executor.

### Executor op set (nothing else touches implant state)
1. matmul: y = X[:, cols] @ W^T + b with fixed weights. This also evaluates every gather/attention index.
2. ReLU.
3. argmax / hard attention:
   - copy heads: score -|key - q| over the token memory, then argmax. The memory is the token ids, appended once per step
     (the KV cache).
   - vocab: top = argmax(logits); blocked = blk[top] (read at the argmax); alt = argmax(logits - 1e30*blk).
4. gather of fixed rows: tables FEAT (token features), CODE2/ONEHOT (pattern matcher), PCOEF (hash residues), F (force rows).
The vocab-plane blocked mask uses only (1), (2) and fixed per-token rows.
NOT in the 4 ops: the readout's rmsnorm, x*rsqrt(mean(x^2)+eps)*g. It is computed exactly as in circuit.py, with the stored
g and eps. Squaring and rsqrt are not ReLU-exact, and argmax is NOT scale-invariant here: the bf16 cast happens after the
scaling, and bf16 ties are common. So dropping the norm cannot be exact. This is the same disclosure as the GSM rethink
compile (which calls the host's norm).

### Exported weights
ifeval/heldout (work dir)implant14_sd.pt`, md5 b71f2342612616269b9147c006f028fc. 1.23 GB, 1,827 tensors,
348.2M numbers:

| part | numbers | size |
|---|---|---|
| force table F (fp32) | 155.6M | 622 MB |
| LM-head copy E (bf16) | 155.6M | 311 MB |
| FEAT token table (fp64, 168 cols) | 25.5M | 204 MB |
| POW1H vocab one-hot | 5.5M | 44 MB |
| ONEHOT | 4.2M | 33 MB |
| TB, HTB, LEAD, CODE2 | ~1.5M | ~12 MB |
| ReLU logic (STEP 144 ops + WRITE 157 ops) | 0.33M stored, 77,887 nonzero weights, 9,304 ReLU units | 2.6 MB |

Without the two vocab x d tables (F, E), the implant is about 37M numbers. Its logic is 9.3k ReLU units.
Rebuild: export.py (needs tables.pt, vecs2_forcefix.pt, the host weights). Building takes about 20 s.

### Verification (all on the A100 pod; GPU 0 A100 shared with the RL job; my peak memory < 10 GB)
1. Noise floor, reference vs reference: circuit.py rerun (same run_t.py command, bs 24, max_new 2048) vs the earlier
   q06_ifeval_imp run: 541/541 byte-identical, circ reports 541/541 identical. Same-GPU greedy decoding is deterministic.
2. Compiled vs circuit.py, all 541 prompts, greedy, generation DRIVEN by the compiled implant (run_c.py):
   - vs the original q06_ifeval_imp run: 541/541 byte-identical
   - vs the ref2 rerun: 541/541 byte-identical
   - circ reports (latches, N_*, captured phrases, forbid hashes, fires, ffail): 541/541 identical
   So the published 0.6B numbers (57.86 -> 72.09 prompt-strict ALL) carry over unchanged.
   Shadow mode: circuit.py ran alongside on identical inputs. On every step()/write() it compared all 414 registers, the full
   151,936-entry blocked mask, the emitted token and the bf16 write row. Over 26,242 steps and 24,891 writes: 0 blocked-mask
   mismatches, 0 decision mismatches, 0 write mismatches, and 0 register mismatches after the saturation note below.
3. Silence: 324 prompts on which the implant writes nothing; 324/324 are byte-identical to the plain model (q06_ifeval_plain).
4. Hash stress (CPU, fp32 host weights):
   - hash_state_test: 400 trials x 24 rows of random states. Slot hashes are hashes of real/random multi-piece words;
     hcur is a word prefix, random, or 0; anti-repeat registers are random. 0 mismatches in the full blocked mask over
     9,600 rows (329,652 blocked tokens; 1,464 times the host's top token was blocked).
   - lockstep on random sequences (closed loop), 0 mismatches of any kind:
     - stress_forbid: 240 seqs, 7,000 writes, 123,900 hash-live row-writes, 6.8M blocked tokens
     - stress_mixed (all 14 rules): 240 seqs
     - stress_random (random token ids): 240 seqs
   - CPU lockstep on IFEval batches 0-9 (240 prompts; teacher answers, synthetic host states, closed loop): 0 blocked-mask
     and 0 decision mismatches.

### What is still not exactly circuit.py (precise)
- Digit accumulator: compiled acc = min(value, 10^6). circuit.py keeps a float32 value. Every use of a number compares it with
  a constant <= 21 or tests it for > 0, so all decisions agree. The one exception is a pathological case: a run of >= 39-40
  digits overflows circuit.py's float32 acc to inf and then NaN. The NaN is permanent for the conversation and disables
  circuit.py's number-based latches (counts for bullets, paragraphs, highlights, nth) in LATER user turns. The compiled
  version does not reproduce this float32 overflow; I treat it as a circuit.py bug, not intended behaviour.
  - Seen once, in the CPU synthetic lockstep (batch 7, step 2058: ref acc = NaN, compiled 0).
  - Never in the 541 real generations.
  - It cannot affect a single-turn prompt: in the answer phase these registers only feed user-turn-gated latches.
  - Values above 10^6 occurred in 1 real batch (4,020 row-steps, answer side). There the register compare uses
    min(ref, 10^6) and counts the saturations.
- The readout's rmsnorm (see above).
- The decode loop, the final-norm pre-hook with its on/off switch, and run_c's diagnostics (report decoding, ffail) are
  harness, as before.

### Cost
Executor on GPU, B=24: STEP 17 ms + WRITE 18 ms per token (first version had per-op GPU syncs: 340 ms). Full 541 with shadow:
6,214 s (plus 3,000 s before a resume). The circuit.py rerun took 5,167 s.

### Files
- Laptop: code/full14/ (sym.py, circ14.py, execu.py, export.py, run_c.py, lockstep.py, lockstep_util.py, hash_state_test.py,
  stress_gen.py, compare.py, launch scripts); results_full14/runs (comp_imp.jsonl, ref2_imp.jsonl, shadow log),
  results_full14/logs (stress, lockstep, hash tests).
- Pod: ifeval/heldout/ (same, plus implant14_sd.pt and stress_*.jsonl).
- Incidents:
  - `pkill -f` / `pgrep -f "run_c.py"` matched my own ssh command line twice (exit 255). Kills now run from script files
    with anchored patterns.
  - Mismatch logging made one shadow batch crawl; I resumed from the .partial file. The batches already done are unaffected.
