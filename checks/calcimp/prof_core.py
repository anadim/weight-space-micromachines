"""[38] exact-core latency, CPU and GPU (res/prof_core_cpu.txt / prof_core_gpu.txt). python3 prof_core.py"""
import time, torch, random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tt/code/rl/gsm_grpo'))
from core import ExactCore
ex = ['2*24', '42*75', '6+4+2', '578619645984209.75*0.2', '(3+4)*12.5/7', '1000-250*3', '12/5', '0.15*80']
for dev in (['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']):
    core = ExactCore(dev)
    for n in (1, 8, 32, 128):
        E = [random.choice(ex) for _ in range(n)]
        for w in range(3): cp, ln = core.encode(E); out = core(cp, ln)
        if dev == 'cuda': torch.cuda.synchronize()
        t = time.time(); R = 20
        for _ in range(R):
            cp, ln = core.encode(E); out = core(cp, ln); _ = out[0].tolist()
        if dev == 'cuda': torch.cuda.synchronize()
        print(f'{dev} n={n}: {1000*(time.time()-t)/R:.1f} ms/call', flush=True)
