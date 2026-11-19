"""Stage 3 check: (1) recorded tool outputs (engine2 rollout) == the old trainer's teacher-forced features (Micromachine
emit / CalcImplant.full) at every position of every sequence; (2) one Dr.GRPO update on the same batch from the same
state: old train_step vs train_step2 -> loss, grad norm, every stat, and every parameter after the step."""
import os, sys, json, time, random, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--B_max', type=int, default=128); ap.add_argument('--prompts', type=int, default=64)
ap.add_argument('--fcheck_rows', type=int, default=64); ap.add_argument('--fast', type=int, default=1); ap.add_argument('--fused', type=int, default=0); ap.add_argument('--refill', type=int, default=0); ap.add_argument('--feat_only', type=int, default=0)
A = ap.parse_args()
import ref_env, train2
OUT = os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/test_train2'; os.makedirs(OUT, exist_ok=True)
G = ref_env.load(OUT, ['--B_max', str(A.B_max), '--fused', '0', '--tok_budget', '4096'], patch=(train2.make_patch_fused if A.fused else (train2.make_patch_fast if A.fast else train2.make_patch)))
import torch, eng2
for a_ in ('make_patch',): pass
old = G['eng']; old.Kc = old.Vc = None; old.graphs = {}; old.pool = None; torch.cuda.empty_cache()
new = (eng2.LibEngine2Refill if A.refill else eng2.LibEngine2Lazy)(old.W, G['pre'], G['tok'], G['imp'], G['TOOLS'], G['IFT'], G['E0'], links=G['links'], B_max=A.B_max, S_max=G['a'].S_max)
new.chat_stop = old.chat_stop; new.set_mwire(G['INW'])
R = {}
rs = random.Random(0); batch = rs.sample(G['pool'], A.prompts); qids = [G['QID'](r) for r in batch]
torch.manual_seed(1); t0 = time.time(); out = new.generate(qids, G=8, T=1.0, max_new=512); R['roll_s'] = time.time() - t0
print('rollout', out['stats'], flush=True)
R['stats'] = out['stats']
Rw = [G['SCORE'](t, batch[j // 8])[0] for j, t in enumerate(out['tokens'])]
Rt = torch.tensor(Rw).view(len(batch), 8); keep = Rt.std(1) > 0; advt = Rt - Rt.mean(1, keepdim=True)
idx = [p * 8 + g for p in range(len(batch)) if keep[p] for g in range(8)]
R['kept_groups'] = int(keep.sum())

# (1) features: recorded vs teacher-forced (old trainer's computation), rows of the batch incl. the last position
dev = 'cuda'; imp, TOOLS = G['imp'], G['TOOLS']
mism = 0; npos = 0; nnz = 0
with torch.no_grad():
    for j in idx[:A.fcheck_rows]:
        q, r = qids[j // 8], out['tokens'][j]
        seq = q + r
        full = torch.cat([new.tool_prefix, torch.tensor(seq, device=dev)])[None]
        Kt = imp.K + imp.W
        ec = imp.full(new.W['E'][full[:, new.Lt - Kt:]])[:, Kt:][0]
        tf = new.tool_feats(full, new.Lt)
        ref = torch.cat([ec] + [tf[t.name][0] for t in TOOLS], -1)          # (len(seq), nout)
        rec = torch.cat([out['Eq'][j], out['E'][j][1:len(r) + 1]]).to(dev).float()
        mism += int((ref != rec).any(-1).sum()); npos += len(seq); nnz += int((ref != 0).any(-1).sum())
R['feature_check'] = dict(rows=min(len(idx), A.fcheck_rows), positions=npos, mismatched_positions=mism, nonzero_positions=nnz)
print('FEATURES', R['feature_check'], flush=True)

# (2) one update, old vs new, same start state
if A.feat_only:
    json.dump(R, open(f'{OUT}/result_feat.json', 'w'), indent=1, default=str); print('DONE', flush=True); sys.exit(0)
new.Kc = new.Vc = None; new.graphs = {}; new.pool = None; torch.cuda.empty_cache()
opt = G['opt']
params = [p for g in opt.param_groups for p in g['params']]
cws = [l for _, l in G['_all_cw']()]
def snap(): return [p.detach().clone() for p in params], copy.deepcopy(opt.state_dict())
def load(s):
    with torch.no_grad():
        for p, v in zip(params, s[0]): p.copy_(v)
    opt.load_state_dict(s[1])
    for l in cws: l._l1g = 0.0; l._l1n = 0
args = ([qids[j // 8] for j in idx], [out['tokens'][j] for j in idx], [out['logp'][j] for j in idx],
        [float(advt.view(-1)[j]) for j in idx], len(Rw))
kw = dict(batch_dec=[out['dec'][j] for j in idx], batch_ifw=[out['ifw'][j] for j in idx])
S0 = snap()
res = {}
for nm in ('old', 'new', 'new2'):
    load(S0)
    G['eng'] = old if nm == 'old' else new
    for p in params: p.grad = None
    torch.cuda.synchronize(); t0 = time.time()
    if A.fast and nm != 'old': train2.install_nockpt(G['Q'])
    if nm == 'old': st = G['train_step'](*args, **kw)
    else: st = G['train_step2'](*args, **kw, batch_E=[out['E'][j] for j in idx], batch_E_q=[out['Eq'][j] for j in idx])
    torch.cuda.synchronize(); dt = time.time() - t0
    res[nm] = dict(st=st, s=dt, params=[p.detach().clone() for p in params])
    print(nm, round(dt, 2), {k: st[k] for k in ('loss', 'grad_norm', 'ntok', 'n_micro', 'clipfrac', 'ratio', 'frac_e') if k in st}, flush=True)
def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))
R['update'] = dict(old_s=res['old']['s'], new_s=res['new']['s'], new2_s=res['new2']['s'],
                   loss=[res[k]['st']['loss'] for k in res], grad_norm=[res[k]['st']['grad_norm'] for k in res],
                   params_equal_old_new=same(res['old']['params'], res['new']['params']),
                   params_equal_new_new2=same(res['new']['params'], res['new2']['params']),
                   max_abs_param_diff=max(float((x - y).abs().max()) for x, y in zip(res['old']['params'], res['new']['params'])),
                   stats_equal={k: res['old']['st'][k] == res['new']['st'][k] for k in res['old']['st'] if k not in ('train_s',)})
p0 = S0[0]
du_old = torch.cat([(x - y).flatten() for x, y in zip(res['old']['params'], p0)])
du_new = torch.cat([(x - y).flatten() for x, y in zip(res['new']['params'], p0)])
R['update']['update_cos'] = float(torch.nn.functional.cosine_similarity(du_old, du_new, 0))
R['update']['update_rel_err'] = float((du_old - du_new).norm() / du_old.norm())
R['update']['loss_rel_err'] = abs(res['new']['st']['loss'] - res['old']['st']['loss']) / abs(res['old']['st']['loss'])
R['update']['grad_norm_rel_err'] = abs(res['new']['st']['grad_norm'] - res['old']['st']['grad_norm']) / res['old']['st']['grad_norm']
print('UPDATE', json.dumps(R['update'], default=str), flush=True)
json.dump(R, open(f'{OUT}/result.json', 'w'), indent=1, default=str)
print('DONE', flush=True)
