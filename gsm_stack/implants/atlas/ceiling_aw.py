"""Vote-error points by dominant category of the stream's wrong attempts (atlas-aw ceilings).
usage: python3 atlas/ceiling_aw.py DATA_DIR  -> per set: vote-wrong (item x seed) cases split by gap/none-right and category."""
import sys, json, gzip, collections
d = sys.argv[1]
for s in ('dev', 'dev2'):
    try: A = [json.loads(l) for l in gzip.open(f'{d}/attempts_{s}.jsonl.gz', 'rt')]; D = [json.loads(l) for l in gzip.open(f'{d}/diverge_{s}.jsonl.gz', 'rt')]
    except FileNotFoundError: continue
    cat = {r['gid']: r.get('cat') for r in D if r['rec'] == 'att'}
    st = collections.defaultdict(list)
    for a in A: st[(a['idx'], a['seed'])].append(a)
    nI = len({a['idx'] for a in A}); seeds = sorted({a['seed'] for a in A}); nS = nI * len(seeds)
    C = {'gap': collections.Counter(), 'none': collections.Counter()}
    VO = {(r['idx'], int(sd)): v for r in (json.loads(l) for l in gzip.open(f'{d}/items_{s}.jsonl.gz', 'rt')) for sd, v in r['vote_ok'].items()}
    for (i, sd), L in st.items():
        if VO[(i, sd)]: continue
        kind = 'gap' if any(a['ok'] for a in L) else 'none'
        cc = collections.Counter(cat.get(a['gid'], 'never_item') for a in L if not a['ok'])
        C[kind][cc.most_common(1)[0][0] if cc else '?'] += 1
    print(f'\n### {s}: {nI} items x {len(seeds)} seeds; points = cases / {nS} * 100')
    print('| dominant cat of stream wrong attempts | gap (some right) pts | none-right pts | total pts |\n|---|---|---|---|')
    keys = sorted(set(C['gap']) | set(C['none']), key=lambda k: -(C['gap'][k] + C['none'][k]))
    for k in keys:
        g, n = C['gap'][k], C['none'][k]
        print(f'| {k} | {100*g/nS:.2f} | {100*n/nS:.2f} | {100*(g+n)/nS:.2f} |')
    G, N = sum(C['gap'].values()), sum(C['none'].values())
    print(f'| all | {100*G/nS:.2f} | {100*N/nS:.2f} | {100*(G+N)/nS:.2f} |')
