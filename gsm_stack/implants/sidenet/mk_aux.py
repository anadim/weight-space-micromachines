"""aux_{set}.npz for train_sn.py: per attempt (table order) idx, k, sid (stream id), grp (stream-local answer group by float value, -1 = no answer)."""
import gzip, json, numpy as np
for S in ('trainC', 'dev', 'dev2'):
    rows = [json.loads(l) for l in gzip.open(f'tables/attempts_{S}.jsonl.gz', 'rt')]
    sids = {}; G = {}; idx, k, sid, grp = [], [], [], []
    for r in rows:
        key = (r['seed'], r['idx']); s = sids.setdefault(key, len(sids))
        g = -1
        if r['ans'] is not None:
            try: v = round(float(r['ans']), 6)
            except ValueError: v = r['ans']
            gm = G.setdefault(s, {}); g = gm.setdefault(v, len(gm))
        idx.append(r['idx']); k.append(r['k']); sid.append(s); grp.append(g)
    np.savez(f'job/tables/aux_{S}.npz', idx=np.array(idx, np.int32), k=np.array(k, np.int16), sid=np.array(sid, np.int32), grp=np.array(grp, np.int16))
    print(S, len(rows), len(sids), max(grp))
