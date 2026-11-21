"""Step 1: the stack's voter re-run offline from the register columns must equal each seed stream's `ok` and `pick`."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vote import load_set, seed_streams, v_probe, v_count

for s in sys.argv[1:] or ['dev', 'dev2']:
    by, items = load_set(s)
    st = seed_streams(by)
    n = agree_ok = agree_pick = nokeep = 0; acc = {}; bad = []
    for idx, sd, S in st:
        i = v_probe(S)
        it = items[idx]; ok_true = it['vote_ok'][str(sd)]
        voted = [a['k'] for a in S if a['voted']]
        if i is None:
            nokeep += 1; ok = ok_true; pk = -1
        else:
            ok = S[i]['vok']; pk = S[i]['k']
        n += 1; agree_ok += ok == ok_true; agree_pick += (pk == (voted[0] if voted else -1))
        if ok != ok_true or pk != (voted[0] if voted else -1): bad.append((idx, sd, pk, voted, ok, ok_true))
        acc.setdefault(sd, []).append(ok)
    print(f'{s}: streams {n}, ok agreement {agree_ok}/{n}, pick agreement {agree_pick}/{n}, no kept slot {nokeep}; '
          f'vote acc per seed ' + ', '.join(f'{k}:{100*sum(v)/len(v):.2f}' for k, v in sorted(acc.items())) +
          f'; mean {100*sum(sum(v) for v in acc.values())/n:.2f}', bad[:5])
