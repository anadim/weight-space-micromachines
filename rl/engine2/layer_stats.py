"""Per-layer work/activity of every stack on real windows (sampled streams): units, nnz, nnz reading window bits,
fraction of nonzero unit outputs, and the 'active work' = sum over nonzero inputs of their fan-out (push cost)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_env
G = ref_env.load(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/layer_stats', ['--B_max', '32', '--fused', '0'])
import torch, numpy as np
import tk_compile as TC
eng, TOOLS, imp = G['eng'], G['TOOLS'], G['imp']
samp = torch.load(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/test_tools/streams_sampled.pt')
pre = eng.tool_prefix.tolist()
streams = [pre + q + r for q, r in zip(samp['q'][::8], samp['r'][::8])]
machines = [('calc', imp.M, eng.W['E'])] + [(t.name, t.M, eng.E0) for t in TOOLS]
out = {}
with torch.no_grad():
    for name, M, E in machines:
        tab, _ = TC.symtab(M, E)
        wins, Ss = [], []
        net1 = getattr(M.stack1, 'net', M.stack1)
        for s in streams:
            ids = torch.tensor(s, device='cuda'); sym = tab[ids].float()[None]
            win = M.windows(sym)[0, 256::3]
            wins.append(win)
        win = torch.cat(wins)[:3000]
        o1, w1 = net1(win.to(net1.b0.dtype), return_wires=True)
        x2 = torch.cat([win, o1.float()], -1)
        o2, w2 = M.stack2(x2, return_wires=True)
        rec = {}
        for nm, net, wires, n_in in (('s1', net1, w1, win.shape[1]), ('s2', M.stack2, w2, x2.shape[1])):
            L, _ = TC.extract(net)
            nzw = (wires != 0).float().mean(0).cpu().numpy()          # per wire: fraction of windows where nonzero
            layers = []; p = n_in
            for li, Ly in enumerate(L):
                srcw = Ly['idx'][Ly['cols']]
                act = float(nzw[srcw].sum())                           # expected nonzero (entry) inputs per window
                rowact = None
                if li < len(L) - 1:
                    nu = Ly['shape'][0]; rowact = float(nzw[p:p + nu].mean()); p += nu
                layers.append(dict(nu=Ly['shape'][0], nnz=len(Ly['vals']), nnz_win=int((srcw < win.shape[1]).sum() if nm == 's1' else (srcw < win.shape[1]).sum()),
                                   active_entries=act, out_density=rowact))
            rec[nm] = dict(nnz=sum(l['nnz'] for l in layers), active=sum(l['active_entries'] for l in layers),
                           nnz_win=sum(l['nnz_win'] for l in layers), win_density=float(nzw[:win.shape[1]].mean()),
                           top=sorted(layers, key=lambda l: -l['nnz'])[:4])
        out[name] = rec
        print(name, json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'top'} for k, v in rec.items()}), flush=True)
json.dump(out, open(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/layer_stats.json', 'w'), indent=1)
print('DONE')
