"""Stratified sample of wrong attempts of mixed items for hand labelling (atlas-aw).

usage: python3 atlas/hand_sample_aw.py DATA_DIR N_PER_SET > atlas/data_aw/hand_dump_aw.txt
One wrong attempt per sampled item: the attractor-answer attempt (most common wrong answer), stratified over n_right bands.
"""
import sys, json, gzip, random, collections

d, nper = sys.argv[1], int(sys.argv[2])
rng = random.Random(20261121)


def rd(fn):
    with gzip.open(fn, 'rt') as f: return [json.loads(l) for l in f]


import os
for s, n in (('dev', nper), ('dev2', nper // 2)):
    if not os.path.exists(f'{d}/attempts_{s}.jsonl.gz'): continue
    A = rd(f'{d}/attempts_{s}.jsonl.gz'); D = rd(f'{d}/diverge_{s}.jsonl.gz')
    dv = {r['gid']: r for r in D if r['rec'] == 'att'}; items = [r for r in D if r['rec'] == 'item']
    by = collections.defaultdict(list)
    for a in A: by[a['idx']].append(a)
    bands = collections.defaultdict(list)
    for it in items: bands[min(3, 4 * it['n_right'] // it['n_att'])].append(it)
    pick = []
    for b in sorted(bands):
        L = bands[b][:]; rng.shuffle(L); pick += L[:n // 4]
    for it in pick:
        L = by[it['idx']]; wrong = [a for a in L if not a['ok'] and a['ans'] is not None and a['ans'] == it['attractor']] or [a for a in L if not a['ok'] and a['ans'] is not None]
        if not wrong: continue
        w = rng.choice(wrong); r = dv.get(w['gid'], {})
        rights = [a for a in L if a['ok']]
        rp = next((a for a in rights if a['gid'] == r.get('partner')), None) or rng.choice(rights)
        print(f"\n##### {w['gid']}  n_right={it['n_right']}/{it['n_att']} gold={it['gold']} wrong_ans={w['ans']} "
              f"heur={r.get('cat')}/{r.get('sub')} [{r.get('detail')}]")
        print('Q:', L[0]['q'])
        print('--- WRONG:\n' + w['text'].strip()[:1500])
        print('--- RIGHT (' + rp['gid'] + '):\n' + rp['text'].strip()[:1200])
