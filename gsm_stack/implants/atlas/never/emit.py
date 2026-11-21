import gzip, json, collections, sys
sys.path.insert(0, 'atlas/never'); from labels import L
T = json.load(open('atlas/never/triggers.json'))['per_item']
GOLD_WRONG = {('dev',6),('dev',303),('dev',488),('dev',519),('dev2',115),('dev2',255)}
out = []; rows = []
for s in ['dev', 'dev2']:
    items = {d['idx']: d for d in map(json.loads, gzip.open(f'atlas/data/items_{s}.jsonl.gz'))}
    wr = collections.defaultdict(collections.Counter)
    for l in gzip.open(f'atlas/data/attempts_{s}.jsonl.gz'):
        a = json.loads(l)
        if (s, a['idx']) in L and not a['ok']: wr[a['idx']][a['ans'] if a['ans'] is not None else 'None'] += 1
    for (ss, i), (cat, note, sec) in sorted(L.items(), key=lambda x: (x[0][0], items[x[0][1]]['n_right'] if x[0][0]==s else 0, x[0][1])):
        if ss != s: continue
        it = items[i]; top = wr[i].most_common(3)
        sec = list(sec) + (['gold_wrong'] if (s, i) in GOLD_WRONG else [])
        out.append(dict(set=s, idx=i, n_right=it['n_right'], n_att=it.get('n_att', 32), gold=it['gold'], top_wrong=top,
                        category=cat, secondary=sec, note=note, trig=T.get(f'{s}:{i}')))
        tw = ', '.join(f'{a} ({c})' for a, c in top[:2])
        rows.append(f"| {s}:{i} | {it['n_right']} | {it['gold']} | {tw} | {cat} | {note.replace('|','/')}{' ['+','.join(sec)+']' if sec else ''} |")
json.dump(out, open('atlas/data/never_labels.json', 'w'), indent=1)
open('atlas/never/rows.md', 'w').write('\n'.join(rows) + '\n')
print(len(out))
