# Topic blocker baseline: TF-IDF text classifier vs the state probe

A text-only gate for the topic blocker: TF-IDF (word + char n-grams) + logistic regression, trained on the **same 9,275 dev
rows** (`data/dev_all.jsonl`) as the hidden-state probe. Threshold chosen by leave-one-style-out out-of-fold scores: the
highest true-positive rate with false fires <= 1% on negatives and <= 5% on near-misses (the probe's rule). CPU only.

```bash
cd topic_fence
./run_dev.sh            # if data/dev_all.jsonl is not built yet (see ../README.md)
python build_test.py    # data/test.jsonl + data/mmlu_ret.jsonl
python baselines/tfidf_gate.py      # -> results/tfidf_gate_{math,ml}.json (copies of the outputs are in baselines/results/)
```

Gate-level fire rates (`results/tfidf_gate_math.json`, `results/tfidf_gate_ml.json`; probe numbers from `../results/`):

| bucket | math TF-IDF | math probe (Qwen3-1.7B, L16) | ML TF-IDF | ML probe (L10) |
|---|---:|---:|---:|---:|
| requests to block (hand test) | 90.5 | 98.7 | 98.3 | 98.4 |
| same, other languages (12) | 75.0 | ~98 | 100 | ~98 |
| near-miss ("mentions the topic") | 2.9 | 1.4 | 1.4 | 5.7 |
| Alpaca (normal requests) | 6.5 | 4.2 | 0.7 | 0.0 |
| MMLU non-STEM multiple choice (400) | **2.0** | **23.0** | 0.0 | 0.0 |

Reading: the text classifier matches the probe in distribution and beats it on the MMLU format trap; the probe's edge is
unseen phrasings and languages for math. For ML (distinctive keywords) the probe buys nothing over TF-IDF. Deterministic
(no sampling); rerunning reproduces the numbers exactly given the same `dev_all.jsonl`.
