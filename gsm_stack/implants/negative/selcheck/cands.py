"""Candidate (item, answer value) table for the selcheck feasibility job + stream table for the vote simulation.
Pure python (sandbox has no numpy).
  trainC: atlas/data/attempts_trainC.jsonl.gz (S3 stack, seeds 23/29 x 8 attempts) -> streams of 8 attempts.
  dev/dev2: rung-2 rows aw/out/r2f_{set}/*.jsonl.gz (all-weights stack, 4 seeds, adaptive N up to 16) registers.
Outputs: job/cdata/cands_{set}.jsonl (idx, q, gold, cands [{x, ok}]) and data/streams_{set}.jsonl.gz
(set, idx, seed, gold, ok(vote), slots [[x, ok_val, bad, lg]] for used valid slots)."""
import gzip, json, glob, os, re, collections, sys
H = os.path.dirname(os.path.abspath(__file__)); W = os.path.abspath(f'{H}/../..')


def correct(pred, gold):
    try: return pred is not None and abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    except ValueError: return False


def fmt(v):
    v = float(v)
    if abs(v - round(v)) < 1e-9: return str(int(round(v)))
    s = ('%.6f' % v).rstrip('0').rstrip('.')
    return s


def streams_trainC():
    rows = [json.loads(l) for l in gzip.open(f'{W}/atlas/data/attempts_trainC.jsonl.gz')]
    by = collections.defaultdict(list)
    for r in rows: by[(r['idx'], r['seed'])].append(r)
    out = []
    for (idx, seed), rs in sorted(by.items()):
        rs.sort(key=lambda r: r['k'])
        sl = []
        for r in rs:
            if not r.get('reg_ok'): continue
            sl.append([fmt(r['reg_val']), bool(r['reg_bad']), float(r['lg'])])
        r0 = rs[0]
        out.append(dict(set='trainC', idx=idx, seed=seed, q=r0['q'], gold=r0['gold'], natt=len(rs), slots=sl))
    return out


def streams_aw(S, run='r2f'):
    out = []
    for f in sorted(glob.glob(f'{W}/aw/out/{run}_{S}/*.jsonl.gz')):
        seed = int(re.search(r'seed(\d+)', f).group(1))
        for l in gzip.open(f):
            r = json.loads(l)
            sl = [[fmt(r['val'][i]), bool(r['bad'][i]), float(r['lg'][i])] for i in range(r['natt']) if r['rok'][i]]
            out.append(dict(set=S, idx=r['idx'], seed=seed, q=r['q'], gold=r['gold'], natt=r['natt'], slots=sl,
                            vote_ok=int(r['ok']), last=r.get('last')))
    return out


def base_vote(sl, san=True):
    """comp_serial._vote: keep sane slots if any sane, weight sigmoid(lg), sum per value, ties -> earliest slot."""
    import math
    idx = list(range(len(sl)))
    if san:
        s = [i for i in idx if not sl[i][1]]
        if s: idx = s
    if not idx: return None
    sc = collections.OrderedDict()
    for i in idx:
        sc.setdefault(sl[i][0], 0.0); sc[sl[i][0]] += 1 / (1 + math.exp(-sl[i][2]))
    best = max(sc.items(), key=lambda kv: round(kv[1], 6))  # first max in slot order
    return best[0]


def main():
    sets = sys.argv[1:] or ['trainC', 'dev', 'dev2']
    os.makedirs(f'{H}/job/cdata', exist_ok=True); os.makedirs(f'{H}/data', exist_ok=True)
    for S in sets:
        st = streams_trainC() if S == 'trainC' else streams_aw(S)
        agree = sum((base_vote(s['slots']) is not None and correct(base_vote(s['slots']), s['gold'])) == bool(s.get('vote_ok', 0))
                    for s in st) if S != 'trainC' else -1
        items = collections.OrderedDict()
        for s in st:
            it = items.setdefault(s['idx'], dict(idx=s['idx'], q=s['q'], gold=s['gold'], cnt=collections.Counter()))
            for x, b, lg in s['slots']: it['cnt'][x] += 1
        cands = []
        nmix = 0
        for it in items.values():
            xs = [x for x, _ in it['cnt'].most_common()]
            oks = [correct(x, it['gold']) for x in xs]
            if len(xs) < 2: continue          # never contested
            if S == 'trainC':                  # training: items with a right and a wrong candidate; right + top-3 wrong
                if not any(oks) or all(oks): continue
                keep = [x for x, o in zip(xs, oks) if o] + [x for x, o in zip(xs, oks) if not o][:3]
            else:
                keep = xs[:6]
            nmix += any(oks) and not all(oks)
            cands.append(dict(set=S, idx=it['idx'], q=it['q'], gold=it['gold'], cands=[dict(x=x, ok=int(correct(x, it['gold'])), n=it['cnt'][x]) for x in keep]))
        with open(f'{H}/job/cdata/cands_{S}.jsonl', 'w') as f:
            for c in cands: f.write(json.dumps(c) + '\n')
        with gzip.open(f'{H}/data/streams_{S}.jsonl.gz', 'wt') as f:
            for s in st: f.write(json.dumps(s) + '\n')
        print(S, 'streams', len(st), 'vote-agree', agree, 'items', len(items), 'contested items', len(cands), 'mixed', nmix,
              'cands', sum(len(c['cands']) for c in cands))


if __name__ == '__main__':
    main()
