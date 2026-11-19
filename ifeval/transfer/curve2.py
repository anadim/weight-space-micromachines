"""Implant-count curve, exact, with per-row rule masks (run_t.py ROWCFG).
Rules are added in ORDER (from FRESH gains). Run(k) = implants with rules ORDER[:k]. A prompt's output depends only on
S(p) n ORDER[:k] (S = rules latched by the untouched detectors) and rows of a batch are independent at fixed batch shapes, so:
  k < first latched rule of p          -> output = plain run (k=0)
  k >= last latched rule of p          -> output = full 14-rule run
  in between (|S| >= 2 only)           -> regenerated, in p's ORIGINAL batch (same rows, same padding), with p's row masked to ORDER[:k].
A rerun of batch b serves one config per row; rows with nothing left to do get filler configs ('all' or none) whose outputs must
equal the full / plain runs byte for byte (determinism check).
python curve2.py plan TAG FULL_PREFIX "ENV" VECS  -> jobs/curve_TAG.txt + runs/TAG_plan2.json
python curve2.py merge TAG PLAIN_PREFIX FULL_PREFIX -> runs/TAG_k{0..14}.jsonl"""
import json, sys, glob
ORDER = ['nc', 'para', 'repeat', 'forbid', 'nth', 'caps', 'two', 'lower', 'end', 'high', 'quote', 'bul', 'title', 'ps']
POS = {r: i + 1 for i, r in enumerate(ORDER)}
def load(p):
    R = {}
    for f in sorted(glob.glob(p + '.s*.jsonl')) or [p]:
        for l in open(f): r = json.loads(l); R[r['key']] = r
    return R
def latched(c):
    s = {r for r in ORDER if c.get(r)}
    if c.get('pps'): s.add('ps')
    if c.get('star3'): s.add('para')
    return s
batches = json.load(open('runs/batches_ifeval_bs24.json'))
cmd, tag = sys.argv[1], sys.argv[2]
if cmd == 'plan':
    F = load(sys.argv[3]); menv, vecs = sys.argv[4], sys.argv[5]
    S = {k: sorted(POS[r] for r in latched(r_['circ'])) for k, r_ in F.items()}
    print('fires without latch:', [k for k, r in F.items() if r['circ']['fires'] > 0 and not S[k]])
    need = {k: s[:-1] for k, s in S.items()}                     # k values to regenerate
    plan = {'S': S, 'runs': []}; L = []
    for bi, b in enumerate(batches):
        R = max(len(need[k]) for k in b)
        for j in range(R):
            cfg = {}; role = {}
            for n, k in enumerate(b):
                if j < len(need[k]): cfg[str(k)] = ORDER[:need[k][j]]; role[str(k)] = need[k][j]
                else:
                    f = 'all' if (j + n) % 2 == 0 else []
                    cfg[str(k)] = f; role[str(k)] = 'fill_all' if f == 'all' else 'fill_none'
            cf = f'runs/{tag}_cfg_b{bi}_r{j}.json'; json.dump(cfg, open(cf, 'w'))
            out = f'runs/{tag}_b{bi}_r{j}.jsonl'
            plan['runs'].append({'b': bi, 'j': j, 'out': out, 'role': role})
            L.append(f"2|{out}|{menv} VECS={vecs} ROWCFG={cf} BATCHES={bi}|python ../transfer/run_t.py data/ifeval.jsonl {out} --mode implant --shard 0/1 --bs 24|")
    json.dump(plan, open(f'runs/{tag}_plan2.json', 'w'))
    open(f'jobs/curve_{tag}.txt', 'w').write('\n'.join(L) + '\n')
    hist = {}
    for s in S.values(): hist[len(s)] = hist.get(len(s), 0) + 1
    print('latched-set sizes', dict(sorted(hist.items())), '| prompt configs to regenerate', sum(map(len, need.values())), '| batch reruns', len(L))
elif cmd == 'merge':
    P, F = load(sys.argv[3]), load(sys.argv[4]); plan = json.load(open(f'runs/{tag}_plan2.json')); S = {int(k): v for k, v in plan['S'].items()}
    reg = {}; chk = {'fill_all': [0, 0], 'fill_none': [0, 0], 'latch_same': [0, 0]}; missing = 0
    for r in plan['runs']:
        try: rows = {json.loads(l)['key']: json.loads(l) for l in open(r['out'])}
        except FileNotFoundError: missing += 1; continue
        for k, role in r['role'].items():
            k = int(k); x = rows[k]
            chk['latch_same'][0] += 1; chk['latch_same'][1] += sorted(POS[q] for q in latched(x['circ'])) == S[k]
            if role == 'fill_all': chk['fill_all'][0] += 1; chk['fill_all'][1] += x['response'] == F[k]['response']
            elif role == 'fill_none': chk['fill_none'][0] += 1; chk['fill_none'][1] += x['response'] == P[k]['response']
            else: reg[(k, role)] = x
    print('missing reruns', missing, '| checks (n, ok):', chk)
    for kk in range(15):
        out = {}
        for k in P:
            s = S[k]
            if not s or kk < s[0]: out[k] = P[k]
            elif kk >= s[-1]: out[k] = F[k]
            else: out[k] = reg.get((k, max(v for v in s[:-1] if v <= kk)))
        bad = [k for k, v in out.items() if v is None]
        if bad: print(f'k={kk}: {len(bad)} rows missing'); continue
        with open(f'runs/{tag}_k{kk}.jsonl', 'w') as f:
            for k in sorted(out): f.write(json.dumps(out[k]) + '\n')
    print('merged')
