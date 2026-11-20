"""Run on a pod via `ssh pod python3 - run_dir ... < calc_share_remote.py` (read-only, CPU).
Per RL update: share of GSM rollout rows containing <calc>, and share with a plain 'a op b = c' equation (same rules
as rl_library/causal_analyze.py, resumed duplicates removed). Prints one JSON line per run."""
import sys, json, re, os, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
def style(t):
    if '<calc>' in t: return 'calc'
    if re.search(r'\d\s*[-+*/x]\s*\d[^\n]{0,20}=\s*-?\d', t): return 'eq'
    return 'other'
for rd in sys.argv[1:]:
    seen = {}
    for l in open(os.path.join(rd, 'rollouts.jsonl')):
        r = json.loads(l)
        if r.get('type') != 'gsm': continue
        seen[(r['u'], r['p'], tuple(r['toks'][:8]), r['n'], r['r'])] = r
    cnt = collections.defaultdict(lambda: [0, 0, 0])
    for r in seen.values():
        s = style(tok.decode(r['toks'])); c = cnt[r['u']]; c[0] += 1; c[1] += s == 'calc'; c[2] += s == 'eq'
    print(json.dumps({'run': os.path.basename(rd.rstrip('/')), 'u': {u: v for u, v in sorted(cnt.items())}}), flush=True)
