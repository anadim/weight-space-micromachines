import sys, json, random
from analyze import T, A, N, kinds
from fmt_probe import feats
def mark(k, i):
    r = T[k][i]; out = []; inw = False
    for t, w in zip(r['toks'], r['wflag']):
        if w and not inw: out.append('⟦'); inw = True
        if not w and inw: out.append('⟧'); inw = False
        out.append(t)
    s = ''.join(out); return s[:s.find('\n\n')] if '\n\n' in s else s
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
mode = sys.argv[1]
if mode == 'arith':   # a1_off wrong w/ arithmetic error, a2_on right
    c = [i for i in range(N) if T['a2_on'][i]['ok'] and not T['a1_off'][i]['ok'] and A['a1_off'][i]['nwrong'] and len(T['a2_on'][i]['text']) < 260]
elif mode == 'fmt':   # base_on has an uncalled unit equation, a2_on right and all called
    c = [i for i in range(N) if any(not q['called'] and feats(q)['units'] for q in A['base_on'][i]['eqs']) and not T['base_on'][i]['ok']
         and T['a2_on'][i]['ok'] and all(q['called'] for q in A['a2_on'][i]['eqs']) and len(T['a2_on'][i]['text']) < 300]
elif mode == 'host':  # a2_off right, a1_off wrong, no arithmetic error in a1
    c = [i for i in range(N) if T['a2_off'][i]['ok'] and not T['a1_off'][i]['ok'] and not A['a1_off'][i]['nwrong'] and len(T['a2_off'][i]['text']) < 400]
elif mode == 'R':
    c = [i for i in range(N) if T['base_on'][i]['ok'] and len(T['baseR_on'][i]['text']) < 220]
elif mode == 'a1':    # base_off no answer, a1_off right
    c = [i for i in range(N) if not T['base_off'][i]['fmt'] and T['a1_off'][i]['ok']]
print(mode, 'candidates', len(c))
ks = sys.argv[3].split(',')
for i in random.sample(c, min(3, len(c))):
    print(f'=== #{i} {kinds[i]} gold {T[ks[0]][i]["gold"]}  Q: {T[ks[0]][i]["q"][:300]}')
    for k in ks: print(f'--- {k} ok={T[k][i]["ok"]}\n{mark(k, i)[:700]}')
