"""Bitwise test of the tools megakernel against the OLD per-tool step functions (BBHTool.step, CalcImplant.step), stepping
both from zero state over full token streams (tool prefix + chat prefix + question + response): (a) the 32-row greedy
engine reference, (b) sampled 512-token rows (BBH + combo prompts, old engine, T=1) so that every tool fires.
Also checks, on (a), that nonzero outputs == the reference's per-step fire flags (in-wires at init fire at greedy)."""
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--n_sampled', type=int, default=256); ap.add_argument('--B_max', type=int, default=256)
ap.add_argument('--bench', type=int, default=1); ap.add_argument('--sampled', type=int, default=1); ap.add_argument('--cls', default='ToolsKernel3'); ap.add_argument('--streams', default=''); ap.add_argument('--ref', type=int, default=1)
A = ap.parse_args()
import ref_env
OUT = os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/test_tools'
G = ref_env.load(OUT, ['--B_max', str(A.B_max), '--fused', '0'])
import torch
import tk
eng, TOOLS, imp = G['eng'], G['TOOLS'], G['imp']
dev = 'cuda'
machines = [('calc', imp.M, eng.W['E'])] + [(t.name, t.M, eng.E0) for t in TOOLS]
t0 = time.time(); TK = getattr(tk, A.cls)(machines, A.B_max); print('TK built', round(time.time() - t0, 1), 's regs', TK.k.regs, 'local', TK.k.local, TK.info, flush=True)
R = {'regs': TK.k.regs, 'local': TK.k.local, 'info': TK.info}

ref = json.load(open(os.environ.get('E2_REF', os.environ.get('E2_LIB', 'SET_E2_LIB') + '/runs/engine_ref/engine_reference.json')))
assert ref['prefix_ids'] == eng.prefix_ids.tolist()
sp = f'{OUT}/streams_sampled.pt'
if A.streams:
    samp = torch.load(A.streams)
elif os.path.exists(sp):
    samp = torch.load(sp)
else:
    rs = random.Random(5); items = [r for r in G['pool'] if r['type'] in ('bbh', 'combo')]; rs.shuffle(items)
    items = items[:A.n_sampled // 4]
    qids = [G['QID'](r) for r in items]
    torch.manual_seed(5); o = eng.generate(qids, G=4, T=1.0, max_new=512)
    samp = dict(q=[qids[j // 4] for j in range(len(o['tokens']))], r=o['tokens'], tasks=[items[j // 4].get('task', items[j // 4].get('sub', '')) for j in range(len(o['tokens']))])
    torch.save(samp, sp)
print('sampled rows', len(samp['r']), 'mean len', sum(map(len, samp['r'])) / len(samp['r']), flush=True)


@torch.no_grad()
def run(qs, rs_, tag, ref_flags=None):
    pre = eng.tool_prefix.tolist()
    streams = [pre + q + r for q, r in zip(qs, rs_)]
    B = len(streams); T = max(map(len, streams)); lens = torch.tensor([len(s) for s in streams], device=dev)
    ids = torch.full((B, T), eng.eos, dtype=torch.long, device=dev)
    for i, s in enumerate(streams): ids[i, :len(s)] = torch.tensor(s, device=dev)
    # zero state, old and new
    for t_ in [imp] + TOOLS:
        t_.sym_h.zero_(); t_.S_h.zero_(); t_.key_h.zero_()
    TK.reset_rows(torch.arange(TK.Bp, device=dev)); TK.flag.zero_()
    mism = {n: 0 for n in TK.names}; first = {}; nz = {n: 0 for n in TK.names}; nzr = {n: 0 for n in TK.names}
    flag_mis = 0; flag_n = 0
    t_old = t_new = 0.0
    for p in range(T):
        tok = ids[:, p]; live = p < lens
        torch.cuda.synchronize(); a = time.time()
        olds = [imp.step(eng.W['E'][tok], B).float()] + [t_.step(eng.E0[tok], B).float() for t_ in TOOLS]
        torch.cuda.synchronize(); b = time.time()
        new = TK.step(tok, B); vs = TK.views(B)
        torch.cuda.synchronize(); c = time.time(); t_old += b - a; t_new += c - b
        for n, eo, en in zip(TK.names, olds, vs):
            d = (eo != en).any(-1) & live
            if d.any():
                mism[n] += int(d.sum())
                if n not in first: first[n] = (p, int(d.nonzero()[0, 0]))
            z = (eo.abs().sum(-1) > 0) & live
            nz[n] += int(z.sum()); nzr[n] += int((z & (p >= len(pre))).sum())
        if ref_flags is not None:                       # response step k (token k+1 produced) <-> stream position
            for i in range(B):
                k = p - (len(pre) + len(qs[i]) - 1)
                if 0 <= k < len(ref_flags[i]):
                    got = [int(vs[j][i].abs().sum() > 0) for j in range(len(TK.names))]
                    want = ref_flags[i][k][:len(TK.names)]
                    flag_n += 1; flag_mis += int(got != want)
    res = dict(rows=B, positions=T, mismatched_row_steps=mism, first_mismatch=first, nonzero_steps=nz, nonzero_after_prefix=nzr,
               guard_flag=int(TK.flag.item()), old_s=t_old, new_s=t_new, ms_per_step_old=1000 * t_old / T, ms_per_step_new=1000 * t_new / T)
    if ref_flags is not None: res.update(ref_flag_steps=flag_n, ref_flag_mismatch=flag_mis)
    print(tag, json.dumps(res), flush=True)
    return res


# reference: response k = 0 token is produced by prefill from the question's last position; flags index k <-> step whose
# input is the token at stream position len(pre)+len(q)-1+k  (k = 0 is the prefill's last prompt position)
R['ref'] = run(ref['q_ids'], [t[:-1] for t in ref['tokens']], 'REF', ref_flags=ref['tool_fire_flags']) if A.ref else None
if A.sampled:
    n_ = len(samp['r']); R['sampled'] = []
    for c0 in range(0, n_, A.B_max):
        R['sampled'].append(run(samp['q'][c0:c0 + A.B_max], samp['r'][c0:c0 + A.B_max], f'SAMPLED[{c0}:{c0 + A.B_max}]'))
if A.bench:
    for B in (128, 256):
        tok = torch.randint(0, 150000, (B,), device=dev)
        for _ in range(3): TK.step(tok, B)
        torch.cuda.synchronize(); a = time.time()
        for _ in range(20): TK.step(tok, B)
        torch.cuda.synchronize(); R[f'bench_B{B}_ms'] = (time.time() - a) / 20 * 1000
        print('BENCH', B, R[f'bench_B{B}_ms'], flush=True)
json.dump(R, open(f'{OUT}/result.json', 'w'), indent=1, default=str)
print('DONE', flush=True)
