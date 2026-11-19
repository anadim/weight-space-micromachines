"""engine2 end-to-end checks.
--mode ref   : B_max 32 (as the reference dump): greedy 128 tokens on the 32 reference prompts; tokens, logp, per-step
               per-tool fire flags, in-wire codes, IF write positions + vectors must be bitwise equal to the old engine.
--mode roll  : rollout timing, 64 prompts x G8, max_new 512 (m1 shape), new engine (and optionally the old one)."""
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='ref'); ap.add_argument('--B_max', type=int, default=32)
ap.add_argument('--old', type=int, default=0); ap.add_argument('--lazy_if', type=int, default=0); ap.add_argument('--refill', type=int, default=0)
ap.add_argument('--out', default=os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/test_eng2'); ap.add_argument('--prompts', type=int, default=64); ap.add_argument('--diag', type=int, default=0); ap.add_argument('--ln_memo', type=int, default=0)
A = ap.parse_args()
import ref_env
os.makedirs(A.out, exist_ok=True)
G = ref_env.load(A.out, ['--B_max', str(A.B_max), '--fused', os.environ.get('E2_CALC_FUSED', '0')])
import torch
import eng2
old = G['eng']
R = {}
ref = json.load(open(os.environ.get('E2_REF', os.environ.get('E2_LIB', 'SET_E2_LIB') + '/runs/engine_ref/engine_reference.json')))
PRE_OLD = None
if A.mode == 'ref' and A.old:          # run the old engine BEFORE engine2 patches the shared tool objects
    PRE_OLD = old.generate(ref['q_ids'], G=1, greedy=True, max_new=ref['max_new'])
if A.mode != 'ref':                    # free the old engine's KV cache (shared weights stay)
    old.Kc = old.Vc = None; old.graphs = {}; old.pool = None; torch.cuda.empty_cache()
cls = eng2.LibEngine2
if A.lazy_if: cls = eng2.LibEngine2Lazy
if A.refill: cls = eng2.LibEngine2Refill
if A.ln_memo: eng2.install_ln_memo(G)
new = cls(old.W, G['pre'], G['tok'], G['imp'], G['TOOLS'], G['IFT'], G['E0'], links=G['links'], B_max=G['a'].B_max, S_max=G['a'].S_max)
new.chat_stop = old.chat_stop; new.imp_on = old.imp_on
new.set_mwire(G['INW'])
print('engine2 built', flush=True)


def cmp(o, ref, tag):
    n = len(ref['tokens']); res = dict(rows=n)
    res['tokens_equal_rows'] = sum(a == b for a, b in zip(o['tokens'], ref['tokens']))
    res['logp_equal_rows'] = sum(a == b for a, b in zip(o['logp'], ref['logp']))
    res['fire_equal_rows'] = sum(w.tolist() == f for w, f in zip(o['wt'], ref['tool_fire_flags']))
    res['dec_equal_rows'] = sum(d.tolist() == f for d, f in zip(o['dec'], ref['inwire_dec']))
    res['ifw_pos_equal_rows'] = sum(p_.tolist() == f for (p_, _), f in zip(o['ifw'], ref['ifeval_write_pos']))
    res['ifw_vec_equal_rows'] = sum(v_.tolist() == f for (_, v_), f in zip(o['ifw'], ref['ifeval_write_vec']))
    bad = [i for i in range(n) if o['tokens'][i] != ref['tokens'][i]]
    if bad:
        i = bad[0]; k = next(k for k, (a, b) in enumerate(zip(o['tokens'][i], ref['tokens'][i])) if a != b) if any(
            a != b for a, b in zip(o['tokens'][i], ref['tokens'][i])) else min(len(o['tokens'][i]), len(ref['tokens'][i]))
        res['first_token_diff'] = dict(row=i, pos=k)
    print(tag, json.dumps(res), flush=True)
    return res


if A.mode == 'ref':
    if PRE_OLD is not None: R['old_vs_ref'] = cmp(PRE_OLD, ref, 'OLD')
    for rep in range(2):
        torch.cuda.synchronize(); t0 = time.time()
        o = new.generate(ref['q_ids'], G=1, greedy=True, max_new=ref['max_new'])
        if A.refill: print('stats', o['stats'], flush=True)
        R[f'new_vs_ref_{rep}'] = cmp(o, ref, f'NEW{rep}'); R[f'new_s_{rep}'] = time.time() - t0
    R['tk_flag'] = int(new.TK.flag.item())
else:
    rs = random.Random(0); batch = rs.sample(G['pool'], A.prompts); qids = [G['QID'](r) for r in batch]
    modes = [('new', {})] + ([('no_tk', dict(diag_skip_tk=True)), ('no_if', dict(diag_skip_if=True)), ('no_tk_if', dict(diag_skip_tk=True, diag_skip_if=True))] if A.diag else [])
    for nm, fl in modes:
        eng = new
        for k_ in ('diag_skip_tk', 'diag_skip_if'): setattr(eng, k_, fl.get(k_, False))
        eng.graphs = {}; eng.pool = None; torch.cuda.empty_cache()
        torch.manual_seed(0); eng.generate(qids[:max(1, A.B_max // 8)], G=8, T=1.0, max_new=512)       # graphs
        torch.manual_seed(1); torch.cuda.synchronize(); t0 = time.time()
        o = eng.generate(qids, G=8, T=1.0, max_new=512)
        R[nm] = dict(stats=o['stats'], wall=time.time() - t0, len_mean=sum(map(len, o['tokens'])) / len(o['tokens']),
                     reward=sum(G['SCORE'](t, batch[j // 8])[0] for j, t in enumerate(o['tokens'])) / len(o['tokens']),
                     fires=[int(sum(int((d[:, k] == 1).sum()) for d in o['dec'])) for k in range(len(G['TNAMES']))])
        print(nm, json.dumps(R[nm]), flush=True)
    R['tk_flag'] = int(new.TK.flag.item())
if hasattr(new, 'ifC_hist'):
    import collections
    R['ifC_hist'] = dict(collections.Counter(new.ifC_hist)); R['n_rollback'] = new.n_rollback
    print('IFC', R['ifC_hist'], 'rollbacks', new.n_rollback, flush=True)
json.dump(R, open(f'{A.out}/result_{A.mode}.json', 'w'), indent=1, default=str)
print('DONE', R.get('tk_flag'), flush=True)
