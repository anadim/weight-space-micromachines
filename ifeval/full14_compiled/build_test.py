import torch, time, sys
sys.setrecursionlimit(100000)
from circ14 import build
t0=time.time()
T = torch.load('tables.pt')
progs, spec, tb = build(T)
for w,p in progs.items():
    G=p['G']; ops=p['ops']
    from collections import Counter
    c=Counter(k for k,_ in ops)
    nrelu=sum(len(o) for k,o in ops if k=='relu')
    print(w, 'ops', dict(c), 'relu units', nrelu, 'live wires', len(p['live']), 'depth levels', len(set(G.depth[x] for x in p['live'])))
    print('  assumed', p['assumed']); print('  worst |partial sum| bound', p['worst_magnitude'], 2**52)
print('regs', len(spec), 'time', time.time()-t0)
