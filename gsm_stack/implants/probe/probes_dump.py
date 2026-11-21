"""Describe out/pred_{tag}/probes.pt (keys, shapes, biases) and export the '####' verifiers to verifier_v2.npz (node, CPU)."""
import sys, json, numpy as np, torch
tag = sys.argv[1] if len(sys.argv) > 1 else 'trainC'
P = torch.load(f'out/pred_{tag}/probes.pt', weights_only=False); ex = {}
for grp, d in P.items():
    for k, v in d.items():
        w = v['w']; print(grp, k, 'w', tuple(w.shape), str(w.dtype), 'b %.4f' % v['b'], 'l2', v.get('l2'), 'layers', v.get('layers'),
                          '|w| %.4g' % float(w.float().norm()))
        if grp == 'hash' and k in ('L18', 'L22', 'c4', 'call', 'hash_mnl_c4', 'mnl_c4', 'stack_lg', 'w_c4'):
            ex[f'{k}__w'] = w.float().numpy(); ex[f'{k}__b'] = np.float32(v['b'])
np.savez(f'out/pred_{tag}/verifier_v2.npz', **ex); print('saved', sorted(ex))
