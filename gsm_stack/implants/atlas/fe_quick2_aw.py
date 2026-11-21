import sys, os, json, glob, collections
sys.path.insert(0, 'atlas')
from fe_quick_aw import attempts
from parse import Tok, correct
tk = Tok()
for pre in sys.argv[2:]:
    n = collections.Counter(); none = collections.Counter(); natt = collections.Counter(); ln = collections.defaultdict(list)
    for fn in sorted(glob.glob(f'{sys.argv[1]}/{pre}_s*.jsonl')):
        for l in open(fn):
            r = json.loads(l); A = attempts(r['tokens'], tk); natt[len(A)] += 1
            for k, (a, nr) in enumerate(A): n[k] += 1; none[k] += a is None
            ln['ntok'].append(r['ntok'])
    print(pre, 'natt hist', sorted(natt.items()), 'n per k', [n[k] for k in sorted(n)], 'None per k', [none[k] for k in sorted(n)],
          'ntok max', max(ln['ntok']), 'mean', sum(ln['ntok']) / len(ln['ntok']))
