"""train_step2 time breakdown + variants that must stay BITWISE equal (same params after the step):
 base (old code path, per-layer activation checkpointing) | nockpt (packed layers without checkpointing)."""
import os, sys, json, time, random, copy, collections, inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_env, train2
T = collections.defaultdict(float); _last = [0.0]
def _tick(name=None):
    import torch; torch.cuda.synchronize(); now = time.time()
    if name: T[name] += now - _last[0]
    _last[0] = now
def patch(src):
    src = train2.make_patch(src)
    a = src.index('def train_step2('); b = src.index('@torch.no_grad()\ndef hf_check')
    f = src[a:b]
    for x, y in [("    t0 = time.time()\n", "    t0 = time.time(); _tick()\n"),
                 ("    for mb in micro:\n", "    _tick('setup_prefix')\n    for mb in micro:\n"),
                 ("        hcap = None\n", "        _tick('features')\n        hcap = None\n"),
                 ("        lp = lp_of_hidden(Wt, hn[sel], tgt)\n", "        _tick('fwd_packed')\n        lp = lp_of_hidden(Wt, hn[sel], tgt)\n        _tick('lp_of_hidden')\n"),
                 ("        loss = per.sum() / norm + l1\n        loss.backward()\n", "        _tick('loss_misc')\n        loss = per.sum() / norm + l1\n        loss.backward()\n        _tick('backward')\n"),
                 ("    st['frac_e'] = n_e / max(1, n_sel)\n", "    _tick('tail+prefix_bwd')\n    st['frac_e'] = n_e / max(1, n_sel)\n"),
                 ("    st['grad_norm'] = float(gn); st['train_s'] = time.time() - t0\n", "    _tick('clip_opt')\n    st['grad_norm'] = float(gn); st['train_s'] = time.time() - t0\n"),
                 ("        Wt = Q.fuse(P32, torch.bfloat16)\n", "        _tick('tail+prefix_bwd')\n        Wt = Q.fuse(P32, torch.bfloat16)\n        _tick('fuse')\n")]:
        assert x in f, x; f = f.replace(x, y)
    return src[:a] + f + src[b:]
OUT = os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/prof_train2'; os.makedirs(OUT, exist_ok=True)
G = ref_env.load(OUT, ['--B_max', '128', '--fused', '0', '--tok_budget', '4096'], patch=patch, inject={'_tick': _tick})
import torch, eng2
Q = G['Q']
old = G['eng']; old.Kc = old.Vc = None; old.graphs = {}; torch.cuda.empty_cache()
eng = eng2.LibEngine2Lazy(old.W, G['pre'], G['tok'], G['imp'], G['TOOLS'], G['IFT'], G['E0'], links=G['links'], B_max=128, S_max=G['a'].S_max)
eng.chat_stop = old.chat_stop; eng.set_mwire(G['INW']); G['eng'] = eng
rs = random.Random(0); batch = rs.sample(G['pool'], 64); qids = [G['QID'](r) for r in batch]
torch.manual_seed(1); out = eng.generate(qids, G=8, T=1.0, max_new=512)
Rw = [G['SCORE'](t, batch[j // 8])[0] for j, t in enumerate(out['tokens'])]
Rt = torch.tensor(Rw).view(64, 8); keep = Rt.std(1) > 0; advt = Rt - Rt.mean(1, keepdim=True)
idx = [p * 8 + g for p in range(64) if keep[p] for g in range(8)]
args = ([qids[j // 8] for j in idx], [out['tokens'][j] for j in idx], [out['logp'][j] for j in idx], [float(advt.view(-1)[j]) for j in idx], len(Rw))
kw = dict(batch_dec=[out['dec'][j] for j in idx], batch_ifw=[out['ifw'][j] for j in idx], batch_E=[out['E'][j] for j in idx], batch_E_q=[out['Eq'][j] for j in idx])
eng.Kc = eng.Vc = None; eng.graphs = {}; eng.pool = None; torch.cuda.empty_cache()          # training-only from here
opt = G['opt']; params = [p for g in opt.param_groups for p in g['params']]; cws = [l for _, l in G['_all_cw']()]
S0 = ([p.detach().clone() for p in params], copy.deepcopy(opt.state_dict()))
def load():
    with torch.no_grad():
        for p, v in zip(params, S0[0]): p.copy_(v)
    opt.load_state_dict(S0[1])
    for l in cws: l._l1g = 0.0; l._l1n = 0
    for p in params: p.grad = None
# no-checkpoint packed forward (identical ops; activations kept instead of recomputed)
src = inspect.getsource(Q.fwd_packed).replace("if torch.is_grad_enabled():", "if torch.is_grad_enabled() and USE_CKPT[0]:")
USE_CKPT = [True]; ns = dict(Q.__dict__); ns['USE_CKPT'] = USE_CKPT; exec(src, ns); Q.fwd_packed = ns['fwd_packed']
res = {}
for nm, ck in (('nockpt', False),):
    load(); USE_CKPT[0] = ck; T.clear(); torch.cuda.reset_peak_memory_stats()
    st = G['train_step2'](*args, **kw)
    res[nm] = dict(train_s=st['train_s'], loss=st['loss'], gn=st['grad_norm'], parts={k: round(v, 3) for k, v in T.items()},
                   peak_gb=torch.cuda.max_memory_allocated() / 1e9, params=[p.detach().clone() for p in params])
    print(nm, {k: v for k, v in res[nm].items() if k != 'params'}, flush=True)
load(); USE_CKPT[0] = False
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as pr:
    st = G['train_step2'](*args, **kw); torch.cuda.synchronize()
tab = pr.key_averages().table(sort_by='cuda_time_total', row_limit=30, max_name_column_width=90)
open(f'{OUT}/kernels.txt', 'w').write(tab); print(tab, flush=True)
eq = lambda a, b: all(torch.equal(x, y) for x, y in zip(res[a]['params'], res[b]['params']))
print('BITWISE done', flush=True)
