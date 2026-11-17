"""Accuracy vs. extra tokens for the rethink: never / always / only when the detector fires (thresholds from train
quantiles, applied unchanged). Extra tokens = tokens the rethink pass generates (phrase + new solution) for that problem.
python scripts/gate/cost_curve.py GEN.json DET.json   (from gen_rethink.py and fit_detector.py)"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import json, sys, numpy as np
from transformers import AutoTokenizer
G = json.load(open(sys.argv[1])); D = json.load(open(sys.argv[2])); tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
for src, rt in (('base', 'rethink'), ('ocalc', 'ocalc_rt')):
    R = D[src]; names = [k for k in R if isinstance(R[k], dict) and 'test_scores' in R[k]]
    for split in ('test', 'hard'):
        g = G[split]; n = len(g['gold']); b, r = g[src + '_ok'], g[rt + '_ok']
        base_tok = np.mean([len(tok(t)['input_ids']) for t in g[src]])
        extra = [len(tok(t2[len(t1):])['input_ids']) if t2.startswith(t1) else len(tok(t2)['input_ids']) for t1, t2 in zip(g[src], g[rt])]
        print(f'\n== {src} -> {rt} | {split} | first solution {base_tok:.0f} tok/problem on average')
        print(f'   never   : acc {100 * sum(b) / n:5.2f}% | extra tok/problem    0')
        print(f'   always  : acc {100 * sum(r) / n:5.2f}% | extra tok/problem {np.mean(extra):4.0f}')
        orc = [r[i] if not b[i] else b[i] for i in range(n)]
        print(f'   oracle  : acc {100 * sum(orc) / n:5.2f}% | extra tok/problem {np.mean([extra[i] if not b[i] else 0 for i in range(n)]):4.0f}  (rethink only where wrong)')
        for name in names:
            tr = np.array(R[name]['train_scores']); sc = R[name][split + '_scores']; on_s = dict(zip(sc['keep'], sc['s']))
            for q in (0.3, 0.5, 0.7, 0.9):
                thr = np.quantile(tr, 1 - q)                                  # fires on about q of TRAIN problems
                on = [i in on_s and on_s[i] >= thr for i in range(n)]
                acc = sum(r[i] if on[i] else b[i] for i in range(n)) / n
                print(f'   {name:15s} fire~{int(q * 100):2d}% (actual {100 * np.mean(on):3.0f}%): acc {100 * acc:5.2f}% | extra tok/problem {np.mean([extra[i] if on[i] else 0 for i in range(n)]):4.0f}')
