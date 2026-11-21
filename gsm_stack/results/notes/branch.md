# Branch-point probe: can h pick the right token at the wrong attempt's branch point? (todo branch-probe)

**Answer: no.** At the token where a wrong attempt leaves its closest right attempt, neither the host's own probability
nor any probe on h (layers 14/18/22/27/28) beats chance once the selection artifact is removed. Most token forks are
**not decisions**: the outcomes of the two children barely differ beyond binomial noise. A top-2 logit-bias steering
implant conditioned on h is **not viable**, and I did not build one.

Code: `probe/branch/`. The scripts are:
- `build.py`: tries, forks, pairs, job inputs.
- `job/bx.py`: GPU teacher-forced extraction, job j23224 (3 min).
- `ujob/unemb.py`: CPU unembedding rows, job j23230.
- `fit.py`: probes.
- `decisive.py` and `nullbranch.py`: null checks.
- `sim.py`: offline steering simulation.

Data is in `probe/branch/data/` (~480 MB fp16 h; also on the node at `$WSM_DATA/implants/branch/`). Logs are `fit_*.log` and `fit_*.json`, and the simulation output is `sim_had_27.json`.

## Data
- **Branch pairs** come from the atlas definition: the wrong attempt vs the right attempt with the longest common prefix, in the same context group. Pairs are deduplicated by (fork, tok_w, tok_r). Forks where either token was written or calc-replaced are dropped.
- **Trie forks** are all nodes of the per-(item, group) token trie with at least 2 children. They carry soft labels: child value v = n_right/n.

| set | wrong atts with partner (all / clean) | unique clean branch pairs | clean trie forks | forks with differing child values | control forks (right+wrong atts, ≤2/att, p2/p1≥0.3) |
|---|---|---|---|---|---|
| trainC (train) | 3275 / 3260 | 2294 | 7300 | 3588 | – |
| dev | 2729 / 2719 | 1686 | 7017 | 2845 | 18703 |
| dev2 | 1492 / 1489 | 921 | 3727 | 1566 | 9650 |

**Sanity checks.**
- Σ(rmsnorm(h28)·g⊙(u_a−u_b)) vs the host Δlogp: corr 0.9995. Positions and unembedding are aligned.
- Teacher-forced vs sampled logp: MAE 0.009 over all positions; 0.08–0.10 at fork tokens (bf16 batch vs engine at uncertain positions).
- {tok_r, tok_w} = the host's top-2 in 67–69% of pairs.

## 1. Pairwise accuracy at branch points (train trainC, test held-out dev/dev2)
- **acc** = choosing tok_r, over deduplicated pairs.
- **accW** = the same, weighted per wrong attempt.
- **@q** = accuracy on the q most-confident fraction of pairs (the abstain operating point).

Model families:
- **had_L:** logistic regression on [Δlp, rmsnorm(h_L) ⊙ g⊙Δu], i.e. a diagonal W (a re-weighting of the host's own readout).
- **lr_r8:** low-rank bilinear h^T A B^T Δu with rank 8, on layers 18/22/28.
- **trust_L:** logistic regression on h, predicting whether the higher-probability token is right.

| model | dev acc / accW | dev @10% / @25% / @50% | dev2 acc / accW | dev2 @10% / @25% / @50% |
|---|---|---|---|---|
| **host argmax (baseline)** | **0.698 / 0.581** | 0.875 / 0.824 / 0.796 | **0.710 / 0.586** | 0.848 / 0.826 / 0.774 |
| had_27 | 0.700 / 0.584 | 0.887 / 0.827 / 0.798 | 0.706 / 0.582 | 0.848 / 0.830 / 0.772 |
| had_14+18+22+27+28 | 0.693 / 0.576 | 0.881 / 0.834 / 0.790 | 0.706 / 0.581 | 0.870 / 0.848 / 0.776 |
| trust_18 | 0.679 / 0.573 | 0.857 / 0.817 / 0.765 | 0.689 / 0.582 | 0.902 / 0.826 / 0.787 |
| lr_18+22+28_r8 | 0.635 / 0.538 | 0.833 / 0.762 / 0.709 | 0.670 / 0.564 | 0.859 / 0.822 / 0.750 |
| trainC+dev → dev2: had_28 / trust_18 | – | – | 0.712 / 0.597, 0.712 / 0.598 | trust_18: 0.913 / 0.861 / 0.774 |

Other training variants:
- **Soft trie-fork labels:** worse on dev / dev2. had_22 scores 0.650 / 0.667, lr scores 0.63–0.66, and had_18/27/28 and trust score 0.50–0.56.
- **Soft fork pairs (|Δv|-weighted) as the test set:** every model, including the host, scores 0.49–0.54.

Breakdowns (dev host / had_27; dev2 host / had_27):
- **By |Δlogp|:**

  | \|Δlogp\| | dev | dev2 |
  |---|---|---|
  | <0.5 | 0.53 / 0.54 | 0.61 / 0.59 |
  | 0.5–1.5 | 0.62 / 0.62 | 0.67 / 0.67 |
  | 1.5–3 | 0.74 / 0.74 | 0.70 / 0.70 |
  | ≥3 | 0.85 / 0.85 | 0.85 / 0.85 |

- **By category (dev):**

  | category | n | host / had_27 |
  |---|---|---|
  | relation | 766 | 0.68 / 0.68 |
  | dropped | 333 | 0.74 / 0.75 |
  | operand | 192 | 0.69 / 0.69 |
  | target | 164 | 0.73 / 0.73 |
  | plan | 77 | 0.68 / 0.69 |
  | unit | 53 | 0.61 / 0.64 |

- **Critical branches inside the D-line:**

  | set | n | host / had_27 |
  |---|---|---|
  | dev | 185 | 0.62 / 0.63 |
  | dev2 | 97 | 0.67 / 0.68 |

The probe gains nothing anywhere: at most +1 point, against a target of +8. The "abstain" curve is entirely the host's own |Δlogp|.

## 2. Why: the branch label is (almost) noise
**(a) The 58% "higher-prob token is right" is a selection artifact.** I recomputed the atlas branch points with outcome labels shuffled among the attempts of each item (n_right kept). The host's per-wrong-attempt accuracy is as high or higher with the shuffled labels:

| set | real labels | shuffled labels (3 perms) |
|---|---|---|
| trainC | 0.595 | 0.608 ± 0.005 |
| dev | 0.579 | 0.633 ± 0.012 |
| dev2 | 0.576 | 0.603 ± 0.004 |

The longest-common-prefix partner selection favours the popular token by itself. With real labels the higher-probability token is, if anything, slightly *less* often the right one. This is consistent with high-probability "attractor" paths.

**(b) Fork children barely differ in outcome.** The test uses trie forks whose top-2 children each have n ≥ 3. It compares the observed mean |v_a − v_b| with a permutation null that shuffles outcomes among the attempts through the fork:

| set | forks | obs | null | excess | more-sampled child better |
|---|---|---|---|---|---|
| trainC | 596 | 0.270 | 0.233 | +0.037 | 0.478 |
| dev | 604 | 0.249 | 0.190 | +0.059 | 0.497 |
| dev2 | 364 | 0.230 | 0.192 | +0.038 | 0.460 |
| dev, n≥5 | 293 | 0.210 | 0.151 | +0.059 | 0.488 |
| trainC, n≥5 | 156 | 0.195 | 0.190 | +0.005 | 0.530 |

The chi-square heterogeneity X/df is 1.25–1.96, against 1 under the null. Per-fork permutation tests give a p<0.05 rate of only 0.02 (trainC), 0.08–0.09 (dev) and 0.05 (dev2), against a nominal 0.05 under the null. A minority of forks are real decisions, but on average which token is taken shifts P(right) by only about 0.05. Higher host probability does not mean a better child.

The atlas fits this picture: 89% of branch points sit *before* the D-line, and the median branch is at 10% of the attempt, on word tokens such as " define" vs " denote". The first token difference is mostly sampling noise. The outcome is decided later, at the line level, where the step detector (notes/probe.md §3a, about 0.77 within-item AUROC after the error) already sees it.

## 3. Offline steering simulation (sim.py, had_27; dev2, dev similar)
Implant setup: at sampled tokens whose host top-2 satisfy p2/p1 ≥ ρ (not written, not calc), either:
- **calib:** set the pair log-odds to the probe score; or
- **hard +2:** add +2 to the probe-preferred token.

| ρ | branch points gated (per wrong att) | ΔP(take tok_r) per wrong att | gated forks per right attempt | ΔP(leave right token) per gated fork |
|---|---|---|---|---|
| 0.3, calib | 35% | −0.004 | 7.1 | +0.027 (0.19 expected deviations per right attempt) |
| 0.3, hard +2 | 35% | +0.008 | 7.1 | −0.064 (this is just sharpening toward top-1) |
| 0.5, calib | 24% | −0.000 | 4.3 | +0.009 |

- Only 24–35% of branch points are even "close top-2" forks.
- The probe moves P(right branch) by under 1 point.
- Because forks are not decisions (§2b), even a perfect token-level choice would change the outcome only about 5% of the time.
- The hard variant is just a lower temperature, already shown to be useless (implants/trunc).

**Side fact (a voter feature, not a steering one).** Wrong attempts have more close forks:

| p2/p1 | right attempts (per 100 tokens) | wrong attempts (per 100 tokens) |
|---|---|---|
| ≥ 0.3 | 4.7–4.8 | 6.3–6.5 |
| ≥ 0.5 | 2.8–2.9 | 3.7–3.9 |

Per attempt at p2/p1 ≥ 0.3: right 6.9–7.1, wrong 9.9–10.9. This is host uncertainty along the attempt, usable as an attempt-level voter feature.

## 4. Implications
- **Drop token-level steering at branch points**, whether a logit bias between the top-2 or a min-p / temperature change. The branch token carries about 0 learnable signal: probe gain is +0.00 ± 0.01, and the label is near-null by the permutation tests.
- **Where to steer instead: the line/step level.**
  - The step-error detector fires after the error with about 0.77 within-item AUROC.
  - A local rewind-and-resample of the flagged line is the single-call analogue.
  - Its cost is one line, not an attempt.
- **Voting.** Add attempt-level host uncertainty (the close-fork rate) as a cheap voter feature next to verifier v2.
- **If someone revisits this:** restrict to the minority of decisive forks. That means forks with large |v_a − v_b| and n ≥ 5, about 5–10% of forks, mostly inside D-lines. It needs far more samples per item, e.g. 64+ attempts on trainC, to obtain labels.

## Repro (from `probe/branch/`)
1. `python3.11 build.py`: writes tables/ and job/tables/.
2. `<queue-submit> probe/branch/job 1800 job_branchx -- bash run.sh python3 bx.py`: GPU job, about 3 minutes of compute. Move its outputs to data/.
3. Run `ujob/run.sh` as a CPU gq job to write data/unemb.npz. This works around an unemb-save crash at the end of bx.py; lm_head is tied to embed_tokens.
4. Fit the probes:
   - `python3.11 fit.py --tag trainC`
   - `python3.11 fit.py --train trainC,dev --test dev2 --tag trainCdev --trainon pairs`
5. Run the null checks: `python3.11 decisive.py` and `python3.11 nullbranch.py`.
6. Run the steering simulation: `python3.11 sim.py --name had_27`.
