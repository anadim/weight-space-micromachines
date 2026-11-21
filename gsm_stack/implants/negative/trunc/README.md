# Branch-token floor (min-p / temperature) on the S3-serial stack

## What it is
The atlas showed that wrong attempts branch early from their closest right attempt, often on a low-probability sampled token
(p_w median 0.31; p_w < 0.1 in 24% of branch points but only 1% of all sampled tokens). The floor is a fixed forward-only
logit map applied in the same decode step, with no extra calls: on the tempered fp32 logits z = logit / T, it drops every
token with `z_i - max z < log m`, i.e. `p_i < m * p_max`, before the sampler draws. It acts only on what the host samples. Queue
tokens from the serial writer and calc v2 forced or ALT tokens overwrite the sample as before. They still use the original
logits for their logp and argmax. The other variant lowers T for the whole host sampler.

Code (`job/` is a copy of `gen/`; diff is in `stk/c2_engine.py` and `stk/gen_hook.py` only):
- `c2_engine.py`: `trunc_keep`, `C2Engine.set_trunc(on, m, x)` and `_trunc`. They plug in through the sampler proxy
  `_SampProxy`, which runs inside the decode CUDA graph, and through the first-token sampler in `_wave`. With `x > 0` the
  floor covers only the first x tokens of each attempt or repair segment, using the serial engine's `astart`. That
  refinement is implemented but was not run.
- `gen_hook.py`: STK_SER takes an optional 7th field, `mpM` or `mpM@X`. The output tag gets `_mpM[xX]`, for example
  `dev_N8_R1_v1_s1_m0_T1.0_mp0.1_seed7.jsonl.gz`. Without the field the original code path runs and adds no op. Temperature
  uses the existing mode field (`T0.7`). The temperature only feeds the sampler and the logp bookkeeping.
- Run: `job/run.sh "SER1|SER2" [exact]`, which starts one process per SER group. Analysis: `python3.11 ana.py [--exact]`.
  Its output is saved in `results.md`. Outputs are in `job/out/`.

## Exactness
- `test_trunc.py` is a CPU fuzz test. It pulls the real functions out of `c2_engine.py` with ast and runs 300 random cases:
  bf16 logits, T in {0.5, 0.7, 1, 1.3}, m in {0, 0.01, …, 1}, with ties, sharp peaks and -inf entries. It checks:
  - the kept set exactly matches a numpy fp32 reference;
  - the kept set matches the fp64 rule p ≥ m·p_max away from a 1e-5 boundary band;
  - the renormalized probabilities match p·keep/Σ (atol 1e-6);
  - Gumbel sampling never returns a dropped token, and the argmax is always kept;
  - m=0 is the identity;
  - X gating is correct.

  Result: **OK**, both on the sandbox and in the node venv.
- I ran the stack on the first wave (dev items 0..127, seed 7) and compared it with the baseline file
  `gen/out/dev_N8_R1_v1_s1_m0_T1.0_seed7.jsonl.gz`:
  - **Original path (no field): 128/128 rows identical** in tokens, wf, logp, val, lg and ok.
  - **Floor path forced on with m=0 (`mp0`):** tokens, wf, val, lg and ok are identical in 128/128 rows. 38 of 163k logp values
    differ by 1e-4, the last stored digit. The cause is that the sample kernel reads fp32 instead of bf16 logits. The sampled
    tokens are unaffected.
- Cost: about 5% decode time when the floor is on (59 s vs 56 s per wave).

## Results (N=8 S3-serial, dev600 + dev2, mean of seeds 7 and 11; fixes/breaks are item-level vote changes vs the same-seed baseline, summed over the 2 seeds)

| config | set | vote | per-att | per-att (mixed items) | oracle@8 | distinct ans | tok/q | fixes | breaks | net |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline T1.0 | dev | 88.42 | 80.97 | 67.38 | 93.33 | 1.79 | 1227 | 0 | 0 | +0 |
| baseline T1.0 | dev2 | 85.33 | 79.23 | 63.43 | 93.33 | 1.86 | 1243 | 0 | 0 | +0 |
| T0.7 | dev | 88.50 | 81.76 | 69.29 | 92.58 | 1.69 | 1219 | 21 | 20 | +1 |
| T0.7 | dev2 | 85.00 | 79.35 | 63.80 | 92.00 | 1.79 | 1227 | 20 | 22 | -2 |
| min-p 0.05 | dev | 87.83 | 81.09 | 67.91 | 93.08 | 1.75 | 1219 | 20 | 27 | -7 |
| min-p 0.05 | dev2 | 85.50 | 79.40 | 63.92 | 93.50 | 1.84 | 1242 | 15 | 14 | +1 |
| min-p 0.1 | dev | 88.08 | 81.17 | 67.93 | 93.83 | 1.73 | 1221 | 16 | 20 | -4 |
| min-p 0.1 | dev2 | 86.50 | 79.85 | 65.10 | 93.50 | 1.80 | 1231 | 20 | 13 | +7 |
| min-p 0.2 | dev | 87.67 | 81.47 | 68.62 | 92.83 | 1.67 | 1216 | 18 | 27 | -9 |
| min-p 0.2 | dev2 | 85.83 | 79.92 | 64.85 | 93.00 | 1.73 | 1228 | 20 | 17 | +3 |

Net over dev+dev2 (2 seeds): T0.7 −1, min-p 0.05 −6, min-p 0.1 +3, min-p 0.2 −6. Each config has about 35–45 changed
item-seeds, so the noise on the net count is about ±6. The mixed-item column uses the atlas classes, which were derived from
the baseline seeds, so the baseline value carries a small selection bias.

## Conclusion
**The floor does not help.** Per-attempt accuracy rises by only +0.1 to +0.8 pp (best: T0.7 on dev, min-p 0.2 on dev2).
Diversity falls at the same time: distinct answers per item drop by 0.05–0.15 and oracle@8 often drops. Vote accuracy
therefore stays within noise for every setting. No setting won, so I did not run the "first-X-tokens" refinement (`mpM@X`).
The likely reason: most wrong branches are not tail tokens. p_w has a median of 0.31 and p_w+p_r ≈ 0.95, so the wrong and
right continuations are the two top candidates. A p_max-relative floor removes only the p_w < m·p_max tail. With
p_max ≥ ~0.5, that is roughly 15–25% of branch points at m=0.1 to 0.2 (the atlas has p_w<0.1 in 23%). Even those branches are often re-routed into another wrong path.
**Recommendation: keep T=1 with no floor (baseline).** min-p 0.1 is the only setting that is non-negative overall (+3, inside
noise) and could be kept as a harmless option. To fix the branch points, the choice between the top two candidates has to
be steered, for example with a hidden-state probe or steering at the branch, not by truncating the tail.
