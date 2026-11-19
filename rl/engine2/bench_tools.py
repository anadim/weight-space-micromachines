"""Per-tool timing of the tools kernel on real token streams (sampled rows), B=256 and 32; all tools together too."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--cls', default='ToolsKernel2'); ap.add_argument('--only', default='')
A = ap.parse_args()
import ref_env
G = ref_env.load(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/bench_tools', ['--B_max', '256', '--fused', '0', '--S_max', '640'])
import torch, importlib
import tk
eng, TOOLS, imp = G['eng'], G['TOOLS'], G['imp']
samp = torch.load(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/test_tools/streams_sampled.pt')
pre = eng.tool_prefix.tolist()
streams = [pre + q + r for q, r in zip(samp['q'], samp['r'])]
T = 600
ids = torch.full((256, T), eng.eos, dtype=torch.long, device='cuda')
for i, s in enumerate(streams[:256]): ids[i, :min(T, len(s))] = torch.tensor(s[:T], device='cuda')
machines = [('calc', imp.M, eng.W['E'])] + [(t.name, t.M, eng.E0) for t in TOOLS]
sets = [[m] for m in machines] + [machines]
if A.only: sets = [[m for m in machines if m[0] in A.only.split(',')]]
res = {}
for ms in sets:
    K = getattr(tk, A.cls)(ms, 256)
    name = '+'.join(m[0] for m in ms) if len(ms) < 3 else 'ALL'
    for B in (256, 32):
        K.reset_rows(torch.arange(K.Bp, device='cuda'))
        for p in range(300): K.step(ids[:B, p], B)                    # warm into the answer region
        torch.cuda.synchronize(); a = time.time()
        for p in range(300, 400): K.step(ids[:B, p], B)
        torch.cuda.synchronize(); res[f'{name}_B{B}'] = (time.time() - a) / 100 * 1000
    print(name, {k: round(v, 3) for k, v in res.items() if k.startswith(name + '_')}, 'flag', int(K.flag.item()), flush=True)
    del K; torch.cuda.empty_cache()
json.dump(res, open(os.environ.get('E2_LIB', os.environ.get('E2_LIB', 'SET_E2_LIB') + '') + f'/engine2/runs/bench_tools_{A.cls}.json', 'w'), indent=1)
