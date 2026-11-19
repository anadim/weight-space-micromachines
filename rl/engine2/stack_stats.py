"""CPU-only: structure of every tool's compiled ReLU stacks (layers, widths, nnz, integrality, fan-in), for the design."""
import os, sys, json, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.environ.get('E2_LIB', 'SET_E2_LIB') + '/bbh_integrate/code'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import integ
from micromachines.specs import SPECS
from micromachines.hdl import ReLUNet
out = {}
for n in list(integ.TOOLS_V2) + ['calc_s4']:
    sp = SPECS[n]; c1, c2 = sp['build']()
    rec = dict(K=sp['K'], W=sp['W'], NS=len(sp['syms']), f64=bool(sp.get('f64')))
    for nm, c in (('s1', c1), ('s2', c2)):
        net = ReLUNet(c)
        Ls = c['layers'] + [c['out']]
        widths = [W.shape[0] for _, W, _ in Ls]; nnz = [int((W != 0).sum()) for _, W, _ in Ls]
        allw = np.concatenate([W[W != 0] for _, W, _ in Ls]); allb = np.concatenate([b for _, _, b in Ls])
        fan = np.concatenate([(W != 0).sum(1) for _, W, _ in Ls])
        rec[nm] = dict(L=len(c['layers']), n_in=c['n_in'], n_units=c['n_units'], nnz=sum(nnz), max_w=max(widths),
                       mean_w=float(np.mean(widths)), int_w=bool(np.all(allw == np.round(allw))), int_b=bool(np.all(allb == np.round(allb))),
                       max_absw=float(np.abs(allw).max()), max_fan=int(fan.max()), mean_fan=float(fan.mean()),
                       n_w1=int(sum(1 for w in widths if w <= 2)), w_hist=np.percentile(widths, [10, 50, 90]).tolist())
    out[n] = rec; print(n, json.dumps(rec), flush=True)
json.dump(out, open(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/stack_stats.json', 'w'), indent=1)
