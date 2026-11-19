"""Tool outputs on guard-flagged streams: are they all 0/1 (int8 recording safe)?"""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FILES = sys.argv[1:]
import ref_env
G = ref_env.load(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/chk', ['--B_max', '256', '--fused', '0'])
import tk
eng, TOOLS, imp = G['eng'], G['TOOLS'], G['imp']
TK = tk.ToolsKernel3([('calc', imp.M, eng.W['E'])] + [(t.name, t.M, eng.E0) for t in TOOLS], 256)
for f in FILES:
    d = torch.load(f); pre = eng.tool_prefix.tolist()
    for c0 in range(0, len(d['r']), 256):
        st = [pre + q + r for q, r in zip(d['q'][c0:c0 + 256], d['r'][c0:c0 + 256])]
        B = len(st); T = max(map(len, st))
        ids = torch.full((B, T), eng.eos, dtype=torch.long, device='cuda')
        for i, s in enumerate(st): ids[i, :len(s)] = torch.tensor(s, device='cuda')
        TK.reset_rows(torch.arange(TK.Bp, device='cuda')); TK.flag.zero_()
        bad = 0; mx = 0.0; ex = []
        for p in range(T):
            E = TK.step(ids[:, p], B)[:B]
            nb = ((E != 0) & (E != 1))
            if nb.any():
                bad += int(nb.sum()); mx = max(mx, float(E.abs().max()))
                if len(ex) < 5:
                    r_, c_ = nb.nonzero()[0].tolist(); ex.append((p, r_, c_, float(E[r_, c_])))
        print(f, c0, 'non01 values', bad, 'max', mx, 'examples', ex, 'flag', int(TK.flag.item()), flush=True)
