import torch, time, sys
sys.setrecursionlimit(100000)
from execu import Executor
sd = torch.load('implant14_sd.pt')
ex = Executor(sd, 'cuda')
ex.reset(24)
ids = torch.randint(0, 150000, (24,), device='cuda')
h = torch.randn(24, 1024, device='cuda', dtype=torch.bfloat16)
for _ in range(3): ex.step(ids); ex.write(h)
torch.cuda.synchronize()
from collections import defaultdict
for which in ['step', 'write']:
    tt = defaultdict(float); X = ex._X(which, ids.double() if which == 'step' else None)
    orig = ex.progs[which]
    for p, kind in orig:
        ex.progs['tmp'] = [(p, kind)]
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5): pass
        ex.progs[which] = [(p, kind)]
        # time one op
        sdk = ex.sd
        t0 = time.time()
        try:
            for p2, k2 in [(p, kind)]:
                pass
        except Exception: pass
    ex.progs[which] = orig
torch.cuda.synchronize(); t0 = time.time()
for _ in range(20): ex.step(ids)
torch.cuda.synchronize(); print('step ms', (time.time() - t0) / 20 * 1000)
t0 = time.time()
for _ in range(20): ex.write(h)
torch.cuda.synchronize(); print('write ms', (time.time() - t0) / 20 * 1000)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as pr:
    ex.step(ids); ex.write(h); torch.cuda.synchronize()
print(pr.key_averages().table(sort_by='cuda_time_total', row_limit=12))
