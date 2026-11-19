"""CPU: how much of the deep stacks is collapsible by sign structure (units with all inputs >= 0, weights >= 0, bias >= 0
are linear: ReLU is the identity) and how far pushes reach (layer distance)."""
import os, sys, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.environ.get('E2_LIB', 'SET_E2_LIB') + '/bbh_integrate/code'); sys.path.insert(0, os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2')
import integ, torch
from micromachines.specs import SPECS
from micromachines.hdl import ReLUNet
import tk_compile as TC
for n in ('expr_gen', 'bool_gen', 'calc_gen', 'track_gen'):
    sp = SPECS[n]; c1, _ = sp['build'](); net = ReLUNet(c1)
    L, nU = TC.extract(net); n_in = c1['n_in']
    lay_of = []; p = 0
    for li, Ly in enumerate(L): lay_of += [li] * Ly['shape'][0]
    lay_of = np.array(lay_of)
    lin = 0; relay = 0; dists = []
    for li, Ly in enumerate(L[:-1]):
        nu = Ly['shape'][0]
        wneg = np.zeros(nu, bool); np.logical_or.at(wneg, Ly['rows'], Ly['vals'] < 0)
        lin += int(((~wneg) & (Ly['b'] >= 0)).sum())
        cnt = np.bincount(Ly['rows'], minlength=nu)
        one = np.zeros(nu); np.add.at(one, Ly['rows'], (Ly['vals'] == 1).astype(float))
        relay += int(((cnt == 1) & (one == 1) & (Ly['b'] == 0)).sum())
        src = Ly['idx'][Ly['cols']]; us = src >= n_in
        dists.append(li - lay_of[src[us] - n_in])
    d = np.concatenate(dists)
    print(n, 'layers', len(L) - 1, 'units', nU, 'linear(sign)', lin, 'relays', relay,
          'push dist pct50/90/99/max', np.percentile(d, [50, 90, 99]).tolist(), int(d.max()), flush=True)
