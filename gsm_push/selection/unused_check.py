# offline check of an "unused question number" trigger: does the greedy solution (before its '####') mention every
# number stated in the question? Precision = P(answer wrong | trigger).
import sys, re, torch
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
NUM = re.compile(r'\d[\d,]*(?:\.\d+)?')
def nums(s): return [x.replace(',', '').rstrip('.') for x in NUM.findall(s)]
def norm(x):
    try: v = float(x); return str(int(v)) if v == int(v) else repr(v)
    except: return x
for fn in sys.argv[1:]:
    D = torch.load(fn); n = len(D['q']); fire = []; 
    for j in range(n):
        txt = tok.decode(D['tokens'][j]).replace('<|endoftext|>', '').split('\n\n')[0].split('####')[0]
        qn = {norm(x) for x in nums(D['q'][j])}
        sn = {norm(x) for x in nums(txt)}
        # also accept percent as fraction (20 -> 0.2) and x/100
        sn |= {norm(str(float(x) * 100)) for x in sn if x.replace('.', '', 1).isdigit()}
        un = sorted(qn - sn)
        fire.append(un)
    ok = D['ok']; F = [j for j in range(n) if fire[j]]
    w = sum(1 for j in F if not ok[j])
    print(fn, 'n', n, 'acc', sum(ok) / n, 'fires', len(F), f'({100*len(F)/n:.1f}%)', 'precision(wrong)', round(w / max(1, len(F)), 3),
          'wrong caught', w, 'of', sum(1 for x in ok if not x))
    print('  examples', [(j, fire[j], ok[j]) for j in F[:12]])
