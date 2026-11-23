# [34] 4 − 2b mechanism from per-attempt answers (att column of the item bundles) of the seed-7 runs, 0.6B and 1.7B.
# usage: python3 uniform/final_eval/q32/mech_2b.py   (CPU)
import json
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import items   # uniform/items.py: reduced per-item rows (bundles) or your own full rows via $FE_ROWS
def val(t):
    try: return float(t)
    except ValueError: return None
def stats(cfgs):
    n=acc=f1=pa=na=orc=same=0
    for st, nm in cfgs:
        for r in items.load(st, nm):
            k=int(r['natt']); A=items.att_of(r)
            try: g=float(r['gold'])
            except ValueError: g=None
            V=[val(A[i]) if A[i] is not None else None for i in range(k)]
            R=[v is not None and g is not None and abs(v-g)<=1e-6*max(1,abs(g)) for v in V]
            n+=1; acc+=int(r['ok']); na+=k; pa+=sum(R)/max(k,1); orc+=any(R); f1+=bool(R and R[0])
            same+=len({v for v in V if v is not None})<=1
    return n,100*acc/n,100*f1/n,100*pa/n,na/n,100*orc/n,100*same/n
print('set host | final acc | attempt-1 acc | mean per-attempt acc | attempts/q | oracle (any attempt right) | all attempts same answer %')
for st,p in (('test',''),('hard',''),('gsmplus','gp_')):
    for lab,h in (('0.6B P64 (2b)','P64'),('0.6B CW (4)','CW'),('1.7B P17 (2b)','P17'),('1.7B CW17 (4)','CW17')):
        fs=[(st, f'{p}r2_{h}_s{s}') for s in range(3) if items.have(st, f'{p}r2_{h}_s{s}')]
        if len(fs)<3: print(st,lab,'missing',len(fs)); continue
        print(f'{st:8s} {lab:14s} | %.2f | %.2f | %.2f | %.2f | %.2f | %.1f'%stats(fs)[1:])
