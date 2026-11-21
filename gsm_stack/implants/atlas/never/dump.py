import gzip, json, collections, sys
tr = {}
for l in open('gsm_stack/repo/data/gsm8k/train.jsonl'):
    d = json.loads(l); tr[d['question'].strip()] = d['answer']
out = open('atlas/never/dump.txt', 'w')
sel = []
for s in ['dev', 'dev2']:
    items = {d['idx']: d for d in map(json.loads, gzip.open(f'atlas/data/items_{s}.jsonl.gz'))}
    att = collections.defaultdict(list)
    for l in gzip.open(f'atlas/data/attempts_{s}.jsonl.gz'):
        a = json.loads(l)
        if items[a['idx']]['n_right'] <= 4: att[a['idx']].append(a)
    for i in sorted(att, key=lambda i: (items[i]['n_right'], i)):
        it = items[i]; A = att[i]
        c = collections.Counter(a['ans'] for a in A)
        sel.append((s, i, it['n_right']))
        out.write(f"\n{'='*100}\n[{s} {i}] n_right={it['n_right']}/{len(A)} gold={it['gold']} answers={c.most_common(6)}\n")
        out.write('Q: ' + it['q'] + '\n')
        g = tr.get(it['q'].strip())
        out.write('GOLD SOL: ' + (g if g else '??? not found') + '\n')
        right = [a for a in A if a['ok']]
        top = c.most_common(1)[0][0]
        wrong = [a for a in A if not a['ok'] and a['ans'] == top][:2]
        other = [a for a in A if not a['ok'] and a['ans'] != top][:1]
        for tag, L in [('RIGHT', right[:1]), ('WRONG-top', wrong), ('WRONG-other', other)]:
            for a in L:
                t = a['text'].replace('\n', ' | ')
                out.write(f"--{tag} ans={a['ans']}: {t[:900]}\n")
print(len(sel)); json.dump(sel, open('atlas/never/sel.json', 'w'))
