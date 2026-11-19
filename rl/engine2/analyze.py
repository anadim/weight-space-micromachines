"""Bound / exactness analysis of every tool stack + activity statistics on the reference token streams."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_env
G = ref_env.load(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/analyze', ['--B_max', '32', '--fused', '0'])
import torch, numpy as np
import tk_compile as TC
eng, TOOLS, imp = G['eng'], G['TOOLS'], G['imp']
ref = json.load(open(os.environ.get('E2_REF', os.environ.get('E2_LIB', 'SET_E2_LIB') + '/runs/engine_ref/engine_reference.json')))
machines = [('calc', imp.M, eng.W['E'])] + [(t.name, t.M, eng.E0) for t in TOOLS]
res = {}
streams = [eng.tool_prefix.tolist() + q + t for q, t in zip(ref['q_ids'], ref['tokens'])]
for name, M, E in machines:
    t0 = time.time()
    tab, ok = TC.symtab(M, E)
    L1, nu1 = TC.extract(M.stack1); L2, nu2 = TC.extract(M.stack2)
    n_win = M.K * M.NS
    b1 = TC.bounds(L1, n_win, np.zeros(n_win), np.ones(n_win))
    dS = L1[-1]['shape'][0]
    clo, chi = b1['out_lo'], b1['out_hi']
    f64 = hasattr(M.stack1, 'net')
    b2 = TC.bounds(L2, n_win + dS, np.concatenate([np.zeros(n_win), clo]), np.concatenate([np.ones(n_win), chi]))
    # activity on the reference streams: nonzero window bits (excluding always-on symbols), nonzero stack-1 wires
    always = (tab.float().mean(0) == 1).cpu()
    nzb, nzw, triv = [], [], 0; nwin = 0
    with torch.no_grad():
        for s in streams[:8]:
            ids = torch.tensor(s, device='cuda')
            sym = tab[ids].float()[None]                                  # (1, T, NS)
            win = M.windows(sym)[0, 256:]                                 # windows at prefix+question+answer positions
            mask = (~always).repeat(M.K).to(win.device)
            nb = (win[:, mask] > 0).sum(-1)
            nzb.append(nb.float().mean().item()); triv += int((nb == 0).sum()); nwin += win.shape[0]
            net = getattr(M.stack1, 'net', M.stack1)
            _, wires = net(win[::7].to(net.b0.dtype), return_wires=True)
            nzw.append(float((wires[:, n_win:] != 0).float().mean()))
    r = dict(symtab_01=ok, n_always=int(always.sum()), f64=f64, L1=len(L1) - 1, L2=len(L2) - 1, dS=dS,
             s1_max_psum=b1['max_psum'], s1_out_abs=float(np.maximum(np.abs(clo), np.abs(chi)).max()),
             s1_dead=b1['n_dead'], s1_lin=b1['n_lin'], s1_units=nu1,
             s2_max_psum=b2['max_psum'], s2_dead=b2['n_dead'], s2_units=nu2,
             s1_exact_ref=b1['max_psum'] < (TC.F64_EXACT if f64 else TC.F32_EXACT), s2_exact_ref=b2['max_psum'] < TC.F32_EXACT,
             s1_int32=b1['max_psum'] <= TC.I32, s2_int32=b2['max_psum'] <= TC.I32,
             win_nz_bits_mean=float(np.mean(nzb)), win_trivial_frac=triv / nwin, s1_wire_density=float(np.mean(nzw)),
             t=time.time() - t0)
    res[name] = r; print(name, json.dumps(r), flush=True)
json.dump(res, open(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/analyze.json', 'w'), indent=1)
print('DONE')
