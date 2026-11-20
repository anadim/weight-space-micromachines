"""Phantom-limb analysis from run3 rollout dumps: per update window, for one task family (prefix), the share of rows that
call the removed tool (<tool_call>NAME), that contain a <tool_response> (necessarily host-written: the tool is gone), and reward.
Usage: python phantom_analyze.py run_dir task_prefix call_word"""
import sys, json, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
rd, pre, word = sys.argv[1], sys.argv[2], sys.argv[3]
W = collections.defaultdict(lambda: [0, 0, 0, 0.0])
for l in open(f'{rd}/rollouts.jsonl'):
    d = json.loads(l)
    if not (d.get('task') or '').startswith(pre): continue
    t = tok.decode(d['toks']); w = W[(d['u'] - 1) // 5]
    w[0] += 1; w[1] += f'<tool_call>{word}' in t; w[2] += '<tool_response>' in t; w[3] += d['r']
print(f'== {rd} tasks {pre}* tool word {word}')
print('updates   rows  call%  resp%  reward')
for k in sorted(W):
    n, c, r, rw = W[k]; print(f'u{5*k+1:>3}-{5*k+5:<3} {n:5d} {100*c/n:6.1f} {100*r/n:6.1f} {rw/n:7.3f}')
