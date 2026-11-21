"""Gap-a streams (stack vote wrong, a right attempt exists): relation between the right minority and the winning wrong
answer: right went further (wrong answer is an intermediate of a right attempt), wrong went further (right answer is an
intermediate of a winning attempt), both, neither.  Also: lg of right vs winning cluster."""
import os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vote as V
for s in sys.argv[1:] or ['dev', 'dev2', 'trainC']:
    by, items = V.load_set(s)
    C = collections.Counter(); lgr = []; lgw = []; n = 0; ex = []
    for idx, sd, S in V.seed_streams(by):
        i = V.v_probe(S)
        if i is None or S[i]['vok'] or not any(a['vok'] for a in S): continue
        S = [V._keys(a) for a in S]; n += 1
        R = [a for a in S if a['vok']]; W = [a for a in S if a['_v'] == S[i]['_v']]
        rf = any(S[i]['_v'] in a['_I'] for a in R); wf = any(R[0]['_v'] in a['_I'] for a in W)
        C[('right went further' if rf and not wf else 'wrong went further' if wf and not rf else 'both' if rf else 'neither')] += 1
        lgr.append(max(a['rlg'] for a in R)); lgw.append(max(a['rlg'] for a in W))
        if len(ex) < 4 and rf and not wf: ex.append((idx, sd, S[i]['ans'], R[0]['ans'], items[idx]['q'][:200]))
    print(s, n, dict(C), f'max-lg right>winning in {100*np.mean(np.array(lgr) > np.array(lgw)):.1f}%')
    for e in ex: print('   ', e)
