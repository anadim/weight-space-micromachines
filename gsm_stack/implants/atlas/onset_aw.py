"""Degeneration onset in attempts with no answer: first window (16 tokens) with mean sampled logp < THR.
usage: python3 atlas/onset_aw.py DATA_DIR SEEDS [THR]"""
import sys, json, gzip, collections, statistics as st
d, seeds = sys.argv[1], {int(x) for x in sys.argv[2].split(',')}; THR = float(sys.argv[3]) if len(sys.argv) > 3 else -2.5
for s in ('dev', 'dev2'):
    A = [x for x in map(json.loads, gzip.open(f'{d}/attempts_{s}.jsonl.gz', 'rt')) if x['seed'] in seeds]
    res = collections.defaultdict(list)
    for x in A:
        lp = x['seg_logp']; n = len(lp); on = None
        for i in range(x['phrase_len'], n - 16):
            if sum(lp[i:i + 16]) / 16 < THR: on = i; break
        res['na' if x['ans'] is None else ('ok' if x['ok'] else 'wrong')].append((on, x['seg_start'], n))
    for g, L in res.items():
        hit = [t for t in L if t[0] is not None]
        print(s, g, 'n', len(L), 'degenerate', f'{100*len(hit)/len(L):.1f}%',
              'onset in-seg median', st.median([t[0] for t in hit]) if hit else '-',
              'onset abs median', st.median([t[0] + t[1] for t in hit]) if hit else '-',
              'abs onset quartiles', st.quantiles([t[0] + t[1] for t in hit], n=4) if len(hit) > 3 else '-')
