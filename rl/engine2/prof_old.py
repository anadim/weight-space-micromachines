"""Stage-1 profile of the OLD engine on this GPU: per-component decode-step cost (CUDA-graph replay), kernel-level
breakdown of one full step, prefill, a real rollout (64 prompts x G8 = 512 rows like m1), and the Dr.GRPO train step."""
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--B_max', type=int, default=256); ap.add_argument('--out', default=os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/prof_old')
ap.add_argument('--fused', default='1'); ap.add_argument('--prompts', type=int, default=64)
ap.add_argument('--skip_roll', type=int, default=0); ap.add_argument('--hostonly', type=int, default=0)
A = ap.parse_args()
import ref_env
G = ref_env.load(A.out, ['--B_max', str(A.B_max), '--prompts', str(A.prompts), '--tok_budget', '4096', '--fused', A.fused])
import torch
eng, TOOLS, IFT, imp, Q = G['eng'], G['TOOLS'], G['IFT'], G['imp'], G['Q']
dev = 'cuda'; R = {'B_max': A.B_max, 'gpu': torch.cuda.get_device_name(), 'fused_calc': A.fused}
print('tools', [(t.name, t.n_out, t.K, t.Wn, t.NS, t.dS, getattr(getattr(t.M.stack1, 'net', t.M.stack1), 'L', None),
                 getattr(t.M.stack2, 'L', None)) for t in TOOLS], flush=True)
R['tools'] = {t.name: dict(n_out=t.n_out, K=t.K, W=t.Wn, NS=t.NS, dS=t.dS,
                           L1=int(getattr(getattr(t.M.stack1, 'net', t.M.stack1), 'L', -1)), L2=int(getattr(t.M.stack2, 'L', -1)),
                           f64=hasattr(t.M.stack1, 'net'), dense=t.fast1 is not None) for t in TOOLS}
R['calc'] = dict(n_out=imp.n_out, K=imp.K, W=imp.W, NS=imp.NS, dS=imp.dS)


def gtime(fn, n=30):
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s_)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): g.replay()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1000


rs = random.Random(0)
batch = rs.sample(G['pool'], A.prompts)
qids = [G['QID'](r) for r in batch]
R['q_len_mean'] = sum(map(len, qids)) / len(qids); R['Lp'] = int(eng.Lp)

# ---- 1. warm-up rollout (captures graphs, initialises IF static state), then a timed rollout
if not A.skip_roll:
    torch.manual_seed(0)
    o = eng.generate(qids[:max(1, A.B_max // 8)], G=8, T=1.0, max_new=512)
    print('warm', o['stats'], flush=True)
    torch.manual_seed(1)
    torch.cuda.synchronize(); t0 = time.time()
    out = eng.generate(qids, G=8, T=1.0, max_new=512)
    R['rollout'] = dict(o_stats=out['stats'], wall=time.time() - t0, rows=len(out['tokens']),
                        len_mean=sum(map(len, out['tokens'])) / len(out['tokens']), len_max=max(map(len, out['tokens'])))
    print('ROLLOUT', R['rollout'], flush=True)
    eng.imp_on = False
    if A.hostonly: torch.manual_seed(1); o2 = eng.generate(qids[:max(1, A.B_max // 8)], G=8, T=1.0, max_new=512)
    if A.hostonly:
        torch.manual_seed(1); t0 = time.time(); o2 = eng.generate(qids, G=8, T=1.0, max_new=512)
        R['rollout_hostonly'] = dict(o_stats=o2['stats'], len_mean=sum(map(len, o2['tokens'])) / len(o2['tokens']))
        print('ROLLOUT hostonly', R['rollout_hostonly'], flush=True)
    eng.imp_on = True
else:
    o = eng.generate(qids[:2], G=8, T=1.0, max_new=8)

# ---- 2. prefill alone
torch.cuda.empty_cache()
NP_ = 16
rows = [torch.arange(p * 8, (p + 1) * 8, device=dev) for p in range(NP_)]
for rep in range(2):
    torch.cuda.synchronize(); t0 = time.time(); eng._prefill(qids[:NP_], rows); torch.cuda.synchronize()
R['prefill_s_16prompts_x8'] = time.time() - t0
print('prefill', R['prefill_s_16prompts_x8'], flush=True)

# ---- 3. per-component graph timings
comp = {}
BM = eng.B_max
for B in (128, 256):
    if B > BM: continue
    emb0 = eng.E0[torch.randint(0, 150000, (B,), device=dev)]; embc = eng.W['E'][torch.randint(0, 150000, (B,), device=dev)]
    c = {'calc': gtime(lambda: imp.step(embc, B))}
    for t_ in TOOLS: c[t_.name] = gtime(lambda t_=t_: t_.step(emb0, B))
    c['bbh_sum'] = sum(c[t_.name] for t_ in TOOLS)
    comp[B] = c
    print('COMP B', B, {k: round(v, 3) for k, v in c.items()}, flush=True)
if IFT is not None:
    comp['if_step_Bmax'] = gtime(lambda: IFT.if_step(eng.tokb[:BM]))
    comp['if_write_Bmax'] = gtime(lambda: IFT.if_write(eng.xb[:BM]))
    print('IF', comp['if_step_Bmax'], comp['if_write_Bmax'], flush=True)
R['components_ms'] = comp

# ---- 4. whole decode step (graph replay), host-only vs full, at fixed context lengths
steps = {}
for on in (False, True):
    eng.imp_on = on
    for B in (128, 256):
        if B > BM: continue
        g = eng._graph(B, eng.S_max)
        for ctx in (128, 384, 640):
            eng.slot[:B] = ctx; eng.done[:B] = False; eng.glen[:B] = 0
            g.replay(); torch.cuda.synchronize(); t0 = time.time()
            for _ in range(10):
                eng.slot[:B] = ctx; eng.glen[:B] = 0; eng.done[:B] = False; g.replay()
            torch.cuda.synchronize(); ms = (time.time() - t0) / 10 * 1000
            steps[f"{'full' if on else 'host'}_B{B}_ctx{ctx}"] = ms
            print('STEP', on, B, ctx, round(ms, 3), flush=True)
R['step_ms'] = steps
eng.imp_on = True

# ---- 5. kernel-level breakdown of the full step at B=128, ctx 384
from torch.profiler import profile, ProfilerActivity
g = eng._graph(128, eng.S_max)
eng.slot[:128] = 384; eng.done[:128] = False
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    for _ in range(3):
        eng.slot[:128] = 384; eng.glen[:128] = 0; eng.done[:128] = False; g.replay()
    torch.cuda.synchronize()
tab = pr.key_averages().table(sort_by='cuda_time_total', row_limit=60, max_name_column_width=90)
open(f'{A.out}/kernels_full_B128.txt', 'w').write(tab)
ev = [(e.key, e.device_time_total / 3 / 1000, e.count / 3) for e in pr.key_averages()]
R['kernels_full_B128_ms_per_step_top'] = sorted(ev, key=lambda x: -x[1])[:40]
R['kernels_full_B128_n_per_step'] = sum(x[2] for x in ev)
R['kernels_full_B128_sum_ms'] = sum(x[1] for x in ev)
print(tab[:6000], flush=True)

# ---- 6. train step on the timed rollout (same code path as the main loop)
if not A.skip_roll:
    Rw = [G['SCORE'](t, batch[j // 8])[0] for j, t in enumerate(out['tokens'])]
    Rt = torch.tensor(Rw).view(len(batch), 8); mean = Rt.mean(1, keepdim=True); keep = Rt.std(1) > 0
    advt = Rt - mean
    idx = [p * 8 + g_ for p in range(len(batch)) if keep[p] for g_ in range(8)]
    if not idx: idx = list(range(64)); advt = torch.randn_like(Rt)
    R['reward'] = float(Rt.mean()); R['kept_groups'] = int(keep.sum())
    for rep in range(2):
        torch.cuda.synchronize(); t0 = time.time()
        st = G['train_step']([qids[j // 8] for j in idx], [out['tokens'][j] for j in idx], [out['logp'][j] for j in idx],
                             [float(advt.view(-1)[j]) for j in idx], len(Rw), batch_dec=[out['dec'][j] for j in idx],
                             batch_ifw=[out['ifw'][j] for j in idx])
        torch.cuda.synchronize(); t1 = time.time(); G['sync_weights'](); torch.cuda.synchronize()
        R[f'train_{rep}'] = dict(train_s=t1 - t0, sync_s=time.time() - t1, n_micro=st['n_micro'], ntok=st['ntok'],
                                 loss=st['loss'], grad_norm=st['grad_norm'], rows=len(idx))
        print('TRAIN', R[f'train_{rep}'], flush=True)
R['mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
json.dump(R, open(f'{A.out}/prof.json', 'w'), indent=1, default=str)
print('DONE', flush=True)
