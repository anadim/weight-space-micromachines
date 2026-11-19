"""kernel-level profile of engine2 (lazy IF) decode steps at B=256 (graph replays) + per-step wall time."""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_env
OUT = os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/prof_new'; os.makedirs(OUT, exist_ok=True)
G = ref_env.load(OUT, ['--B_max', '256', '--fused', '0'])
import torch, eng2
old = G['eng']; old.Kc = old.Vc = None; old.graphs = {}; torch.cuda.empty_cache()
new = eng2.LibEngine2Lazy(old.W, G['pre'], G['tok'], G['imp'], G['TOOLS'], G['IFT'], G['E0'], links=G['links'], B_max=256, S_max=G['a'].S_max)
new.chat_stop = old.chat_stop; new.set_mwire(G['INW'])
rs = random.Random(0); batch = rs.sample(G['pool'], 32); qids = [G['QID'](r) for r in batch]
new.generate(qids, G=8, T=1.0, max_new=64)
B = 256
for ser in (False, True):
    new.tk_serial = ser; new.graphs = {}; new.pool = None
    for C in (16,):
        new.ifC = C
        g = new._graph(B, new.S_max)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10): new.slot[:B] = 384; new.glen[:B] = 0; new.done[:B] = False; g.replay()
        torch.cuda.synchronize(); print('STEP serial', ser, 'C', C, (time.time() - t0) / 10 * 1000, 'ms', flush=True)
new.tk_serial = False; new.graphs = {}; new.pool = None
for C in ():
    new.ifC = C
    g = new._graph(B, new.S_max)
    new.slot[:B] = 384; new.done[:B] = False; new.glen[:B] = 0
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): new.slot[:B] = 384; new.glen[:B] = 0; new.done[:B] = False; g.replay()
    torch.cuda.synchronize(); print('STEP C', C, (time.time() - t0) / 10 * 1000, 'ms', flush=True)
# isolate: host only, TK only (same replay conditions)
new.imp_on = False; g = new._graph(B, new.S_max)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10): new.slot[:B] = 384; new.glen[:B] = 0; new.done[:B] = False; g.replay()
torch.cuda.synchronize(); print('STEP host-only', (time.time() - t0) / 10 * 1000, 'ms', flush=True); new.imp_on = True
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10): new.TK.step(new.tokb, B)
torch.cuda.synchronize(); print('TK alone', (time.time() - t0) / 10 * 1000, 'ms', flush=True)
from torch.profiler import profile, ProfilerActivity
new.ifC = 64; g = new._graph(B, new.S_max)
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(3): new.slot[:B] = 384; new.glen[:B] = 0; new.done[:B] = False; g.replay()
    torch.cuda.synchronize()
tab = pr.key_averages().table(sort_by='cuda_time_total', row_limit=40, max_name_column_width=80)
open(f'{OUT}/kernels.txt', 'w').write(tab); print(tab[:7000])
ev = pr.key_averages(); print('n kernels per step', sum(e.count for e in ev) / 3, 'sum ms per step', sum(e.device_time_total for e in ev) / 3 / 1000)
