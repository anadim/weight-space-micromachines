"""Markdown tables for plain vs implants on 0.6B / 1.7B / 4B from score_t.py JSONs (runs/{q06,q17,q4}_score.json)."""
import json, os
H = [('0.6B', 'q06'), ('1.7B', 'q17'), ('4B', 'q4')]
J = {h: json.load(open(f'runs/{t}_score.json')) for h, t in H if os.path.exists(f'runs/{t}_score.json')}
def pr(h, t):
    r = J[h]; return r[f'runs/{t}_ifeval_plain'], r[f'runs/{t}_ifeval_imp']
print('| host | split | n | plain P-strict | +impl P-strict | dP | plain I-strict | +impl I-strict | dI | fixed/broken | silent identical | trunc plain/impl |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for h, t in H:
    if h not in J: continue
    p, q = pr(h, t)
    for s in ['all', 'odd', 'even']:
        a, b = p[s], q[s]
        print(f"| {h} | {s.upper()} | {a['n']} | {a['ps']:.2f} | {b['ps']:.2f} | {b['ps']-a['ps']:+.2f} | {a['is_']:.2f} | {b['is_']:.2f} | {b['is_']-a['is_']:+.2f} | +{b['fixed']}/-{b['broken']} | {b['silent_same']}/{b['silent']} | {a['trunc']}/{b['trunc']} |")
rules = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul', 'other']
for s in ['all', 'odd']:
    print(f'\nPer-rule instruction-strict passes, {s.upper()} (plain -> implants, of n)\n')
    print('| rule | n | ' + ' | '.join(h for h, _ in H if h in J) + ' |'); print('|---|---|' + '---|' * len(J))
    for r in rules:
        cells = []; n = None
        for h, t in H:
            if h not in J: continue
            p, q = pr(h, t); a = p[s]['inst'].get(r, [0, 0]); b = q[s]['inst'].get(r, [0, 0]); n = a[0]
            cells.append(f'{a[1]} -> {b[1]} ({b[1]-a[1]:+d})')
        print(f'| {r} | {n} | ' + ' | '.join(cells) + ' |')
