"""Error rate of '=' arithmetic in rollouts (u <= U): A op B = R checked against the true value. Usage: eq_error.py run_dir [U]"""
import sys, json, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
U = int(sys.argv[2]) if len(sys.argv) > 2 else 2
RE = re.compile(r'(?<![\d.])(\d+(?:\.\d+)?)\s*([-+*/x×])\s*(\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)(?![\d.])')
n = bad = 0
for l in open(f'{sys.argv[1]}/rollouts.jsonl'):
    d = json.loads(l)
    if d['u'] > U or d['type'] != 'gsm': continue
    t = tok.decode(d['toks']); t = re.sub(r'<calc>.*?</calc>\s*\S*', '', t, flags=re.S)
    for A, op, B, R in RE.findall(t):
        A, B, R = float(A), float(B), float(R); op = {'x': '*', '×': '*'}.get(op, op)
        try: v = {'+': A + B, '-': A - B, '*': A * B, '/': A / B if B else None}[op]
        except Exception: continue
        if v is None: continue
        n += 1; bad += abs(v - R) > 1e-6 * max(1, abs(v)) and abs(v - R) > 0.011
print(f'{sys.argv[1]} u<={U}: eq expressions {n}, wrong {bad} ({100 * bad / max(1, n):.1f}%)')
