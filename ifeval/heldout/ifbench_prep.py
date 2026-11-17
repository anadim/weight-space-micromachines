import json
d = {json.loads(l)['key']: json.loads(l)['prompt'] for l in open('data/ifbench.jsonl')}
P = {json.loads(l)['key']: json.loads(l) for l in open('runs/ifbench_plain_all.jsonl')}
I = {json.loads(l)['key']: json.loads(l) for l in open('runs/ifbench_imp_all.jsonl')}
for n, R in [('plain', P), ('imp', I)]:
    with open(f'runs/ifbench_{n}_p.jsonl', 'w') as f:
        for k, r in R.items(): f.write(json.dumps({'prompt': d[k], 'response': r['response']}) + '\n')
K = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul']
sil = [k for k in I if I[k]['circ']['fires'] == 0]; unl = [k for k in I if not any(I[k]['circ'].get(x) for x in K)]
print('IFBench n', len(I), 'silent', len(sil), 'identical', sum(P[k]['response'] == I[k]['response'] for k in sil),
      '| unlatched', len(unl), 'identical', sum(P[k]['response'] == I[k]['response'] for k in unl),
      '| latched rules:', {x: sum(1 for k in I if I[k]['circ'].get(x)) for x in K if any(I[k]['circ'].get(x) for k in I)})
