"""Where the old train_step's 41-45 s go (A40): timers injected into a patched copy of train_step (source text patch;
the file on disk is untouched)."""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_env, collections
T = collections.defaultdict(float); _last = [0.0]
def _tick(name=None):
    import torch; torch.cuda.synchronize(); now = time.time()
    if name: T[name] += now - _last[0]
    _last[0] = now
def patch(src):
    a = src.index('def train_step('); b = src.index('@torch.no_grad()\ndef hf_check')
    f = src[a:b]
    reps = [("    t0 = time.time()\n", "    t0 = time.time(); _tick()\n"),
            ("    for mb in micro:\n", "    _tick('setup_prefix')\n    for mb in micro:\n"),
            ("        hcap = None\n", "        _tick('features_nograd')\n        hcap = None\n"),
            ("        lp = lp_of_hidden(Wt, hn[sel], tgt)\n", "        _tick('fwd_packed')\n        lp = lp_of_hidden(Wt, hn[sel], tgt)\n        _tick('lp_of_hidden')\n"),
            ("        loss = per.sum() / norm + l1\n        loss.backward()\n", "        _tick('loss_misc')\n        loss = per.sum() / norm + l1\n        loss.backward()\n        _tick('backward')\n"),
            ("    st['frac_e'] = n_e / max(1, n_sel)\n", "    _tick('micro_tail+prefix_bwd')\n    st['frac_e'] = n_e / max(1, n_sel)\n"),
            ("    st['grad_norm'] = float(gn); st['train_s'] = time.time() - t0\n", "    _tick('clip_opt')\n    st['grad_norm'] = float(gn); st['train_s'] = time.time() - t0\n")]
    for x, y in reps:
        assert x in f, x
        f = f.replace(x, y)
    f = f.replace("        Wt = Q.fuse(P32, torch.bfloat16)\n", "        _tick('micro_tail+prefix_bwd')\n        Wt = Q.fuse(P32, torch.bfloat16)\n        _tick('fuse')\n")
    return src[:a] + f + src[b:]
G = ref_env.load(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/prof_train', ['--B_max', '128', '--fused', '0', '--tok_budget', '4096'], patch=patch, inject={'_tick': _tick})
import torch
eng = G['eng']; rs = random.Random(0); batch = rs.sample(G['pool'], 64); qids = [G['QID'](r) for r in batch]
torch.manual_seed(1); out = eng.generate(qids, G=8, T=1.0, max_new=512)
Rw = [G['SCORE'](t, batch[j // 8])[0] for j, t in enumerate(out['tokens'])]
Rt = torch.tensor(Rw).view(64, 8); keep = Rt.std(1) > 0; advt = Rt - Rt.mean(1, keepdim=True)
idx = [p * 8 + g for p in range(64) if keep[p] for g in range(8)]
torch.save(dict(qids=qids, out={k: out[k] for k in ('tokens', 'logp', 'dec', 'ifw', 'wt')}, R=Rw, idx=idx), os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/prof_train/batch.pt')
for rep in range(2):
    T.clear()
    st = G['train_step']([qids[j // 8] for j in idx], [out['tokens'][j] for j in idx], [out['logp'][j] for j in idx],
                         [float(advt.view(-1)[j]) for j in idx], len(Rw), batch_dec=[out['dec'][j] for j in idx], batch_ifw=[out['ifw'][j] for j in idx])
    print('TRAIN', rep, round(st['train_s'], 2), {k: round(v, 2) for k, v in T.items()}, 'n_micro', st['n_micro'], 'ntok', st['ntok'], flush=True)
