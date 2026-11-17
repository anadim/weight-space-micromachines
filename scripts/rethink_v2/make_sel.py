"""Write the 'sel' (selection) split of GSM8K TRAIN and of the scaled TRAIN analogue as jsonl (same split rule as
fit_detector.py: md5 of the source train question, 70 fit / 15 sel / 15 rep), for HF-harness checks on train only."""
import json, hashlib
sp = lambda q: (lambda h: 'fit' if h < 70 else ('sel' if h < 85 else 'rep'))(int(hashlib.md5(q.encode()).hexdigest(), 16) % 100)
train = [json.loads(l) for l in open('data/gsm8k/train.jsonl')]
scaled = [json.loads(l) for l in open('data/gsm8k/train_scaled.jsonl')]
for name, rows, q in [('train', train, lambda r: r['question']), ('train_scaled', scaled, lambda r: train[r['src']]['question'])]:
    for part in ['sel', 'rep']:
        out = [r for r in rows if sp(q(r)) == part]
        with open(f'data/gsm8k/{name}_{part}.jsonl', 'w') as f:
            for r in out: f.write(json.dumps(r) + '\n')
        print(name, part, len(out))
