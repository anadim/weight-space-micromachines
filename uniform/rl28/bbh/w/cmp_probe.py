# compare the probe in use (probe_ft17.pt, setup.sh, job label j23812) with its re-derivation (probe_ft17_rederive.pt,
# setup2.sh, j23836). CPU. usage: python3 cmp_probe.py   (reads $RL28_DATA/bbh/cg/; source uniform/env.sh first)
import torch, sys, os
D = os.path.join(os.environ.get('RL28_DATA', 'runs/rl28'), 'bbh', 'cg', '')
a = torch.load(D + 'probe_ft17.pt', map_location='cpu', weights_only=False)
b = torch.load(D + 'probe_ft17_rederive.pt', map_location='cpu', weights_only=False)
def walk(x, y, p=''):
    if isinstance(x, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y: print(p + str(k), 'missing in one'); continue
            walk(x[k], y[k], p + str(k) + '.')
    elif torch.is_tensor(x):
        x, y = x.double(), y.double()
        d = (x - y).abs().max().item() if x.shape == y.shape else float('nan')
        cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item() if x.numel() > 1 and x.shape == y.shape else None
        print(f'{p} shape {tuple(x.shape)} max|diff| {d:.3e} max|a| {x.abs().max().item():.3e} cos {cos}')
    else:
        print(p, 'equal' if x == y else f'DIFF {x} vs {y}')
walk(a, b)
