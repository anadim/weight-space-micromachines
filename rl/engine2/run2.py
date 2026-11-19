"""engine2 RL loop (m1 recipe: LIB arm, mixed task, Dr.GRPO, G8, lr 1e-6, no KL): rollout with LibEngine2Lazy,
update with train_step2, weights synced with the old sync_weights (engine2.refresh also refreshes calc's symbol table).
Same per-update record fields as train_library's main loop (subset) + timings. No dev/test evals (smoke runs)."""
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--run_dir', required=True); ap.add_argument('--updates', type=int, default=3)
ap.add_argument('--B_max', type=int, default=256); ap.add_argument('--tok_budget', default='4096'); ap.add_argument('--seed', default='0')
ap.add_argument('--warmup', default='5'); ap.add_argument('--vmm', type=int, default=1); ap.add_argument('--fast', type=int, default=1); ap.add_argument('--fusedlp', type=int, default=1); ap.add_argument('--dev_at', default=''); ap.add_argument('--ln_memo', type=int, default=0); ap.add_argument('--refill', type=int, default=0); ap.add_argument('--save_wiring', type=int, default=1); ap.add_argument('--save_host_at', default='')
A = ap.parse_args()
os.makedirs(A.run_dir, exist_ok=True)
import ref_env, train2
G = ref_env.load(A.run_dir, ['--B_max', str(A.B_max), '--fused', os.environ.get('E2_CALC_FUSED', '0'), '--tok_budget', A.tok_budget, '--seed', A.seed,
                             '--warmup', A.warmup, '--updates', str(A.updates)], patch=(train2.make_patch_fused if A.fusedlp else (train2.make_patch_fast if A.fast else train2.make_patch)))
if A.fast: train2.install_nockpt(G['Q'])
import torch, eng2
a = G['a']; old = G['eng']; old.Kc = old.Vc = None; old.graphs = {}; old.pool = None; torch.cuda.empty_cache()
eng = (eng2.LibEngine2Refill if A.refill else eng2.LibEngine2Lazy)(old.W, G['pre'], G['tok'], G['imp'], G['TOOLS'], G['IFT'], G['E0'], links=G['links'], B_max=a.B_max, S_max=a.S_max)
eng.chat_stop = old.chat_stop; eng.imp_on = not a.plain; eng.set_mwire(G['INW']); G['eng'] = eng
import vmm
if A.ln_memo: eng2.install_ln_memo(G)
eng.prefill_chunk = 16                              # A40: dev evals prefill up to 256 prompts per wave
if A.refill: pass
KV = vmm.kv_arena(eng) if A.vmm else None          # KV physical memory released during training (fixed addresses)
opt, base_lrs, pool, TYPES, TNAMES = G['opt'], G['base_lrs'], G['pool'], G['TYPES'], G['TNAMES']
# no reseed: python random continues from train_library's own state, so the per-update prompt batches are the old main loop's
log = lambda o: open(f'{A.run_dir}/steps.jsonl', 'a').write(json.dumps(o, default=str) + '\n')
DEV_AT = {int(x) for x in A.dev_at.split(',') if x}
def dev_eval(u):
    t0 = time.time(); r = G['evaluate'](G['devset']); r.update(u=u, dev_s=time.time() - t0); log({'dev': r})
    print('DEV', u, json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)}), flush=True)
t_start = time.time()
if 0 in DEV_AT: dev_eval(0)
for u in range(1, A.updates + 1):
    for g_, b_ in zip(opt.param_groups, base_lrs): g_['lr'] = b_ * min(1.0, u / a.warmup)
    t_upd0 = time.time()
    batch = random.sample(pool, a.prompts); qids = [G['QID'](r) for r in batch]
    t0 = time.time(); out = eng.generate(qids, G=a.G, T=a.T, max_new=a.max_new); t_roll = time.time() - t0
    R = []; fmt = 0
    for j, t in enumerate(out['tokens']):
        s = G['SCORE'](t, batch[j // a.G]); R.append(s[0]); fmt += s[1]
    Rt = torch.tensor(R).view(a.prompts, a.G); mean = Rt.mean(1, keepdim=True); keep = Rt.std(1) > 0
    advt = Rt - mean
    idx = [p * a.G + g for p in range(a.prompts) if keep[p] for g in range(a.G)]
    kinds = [batch[j // a.G]['kind'] for j in range(len(R))]
    rec = dict(u=u, t=time.time() - t_start, reward=float(Rt.mean()), fmt=fmt / len(R), kept_groups=int(keep.sum()),
               len=sum(map(len, out['tokens'])) / len(R), roll_s=t_roll, roll_tok_s=out['stats']['tok_s'],
               gen_tokens=out['stats']['tokens'], ifC_rollbacks=eng.n_rollback, tk_flag=int(eng.TK.flag.item()))
    if rec['tk_flag']:                                 # exactness guard fired: keep the streams for an offline replay
        torch.save(dict(q=[qids[j // a.G] for j in range(len(R))], r=out['tokens'], flag=rec['tk_flag']), f'{A.run_dir}/guard_u{u}.pt')
        eng.TK.flag.zero_()
    if KV is not None: KV.unmap(); torch.cuda.empty_cache()
    if idx:
        t1 = time.time()
        st = G['train_step2']([qids[j // a.G] for j in idx], [out['tokens'][j] for j in idx], [out['logp'][j] for j in idx],
                              [float(advt.view(-1)[j]) for j in idx], len(R), batch_dec=[out['dec'][j] for j in idx],
                              batch_ifw=[out['ifw'][j] for j in idx], batch_E=[out['E'][j] for j in idx], batch_E_q=[out['Eq'][j] for j in idx])
        t2 = time.time(); G['sync_weights'](); torch.cuda.synchronize(); rec['sync_s'] = time.time() - t2
        rec.update({k: st[k] for k in ('loss', 'grad_norm', 'ntok', 'n_micro', 'clipfrac', 'ratio', 'train_s') if k in st})
    if KV is not None: torch.cuda.empty_cache(); KV.map()
    for t_ in TYPES:
        jj = [j for j in range(len(R)) if kinds[j] == t_]
        if jj: rec[f'r_{t_}'] = sum(R[j] for j in jj) / len(jj)
    rec.update(G['tool_stats'](out, 'roll'))
    cw = G['tool_stats'](out, 'roll')['roll_t_calc']
    rec['calc_wtok_per_burst'] = cw['wtok'] * rec['len'] / max(1e-9, cw['bursts'] * cw['fire'])
    if str(u) in A.save_host_at.split(','):          # full trainable state after update u (host fp32 + wiring)
        torch.save(dict(P32={k: v.detach().cpu() for k, v in G['P32'].items()},
                        links={str(k): l.state_dict() for k, l in G['links'].items()}, inw=G['INW'].state_dict()), f'{A.run_dir}/state_u{u}.pt')
        torch.save(dict(q=[qids[j // a.G] for j in range(len(R))], out={k: out[k] for k in ('tokens', 'logp', 'dec', 'ifw', 'E', 'Eq')},
                        R=R, batch_ids=[r.get('id', r.get('user')) for r in batch]), f'{A.run_dir}/rollout_u{u}.pt')
    if A.save_wiring:                                  # trainable wiring after every update (small; for debugging)
        torch.save(dict(links={str(k): l.state_dict() for k, l in G['links'].items()}, inw=G['INW'].state_dict()), f'{A.run_dir}/wiring_u{u}.pt')
    rec['upd_s'] = time.time() - t_upd0; rec['mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
    log(rec)
    if u in DEV_AT: dev_eval(u)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if not k.startswith('roll_t_') and not k.startswith('roll_if')}), flush=True)
print('DONE', time.time() - t_start, flush=True)
