"""Bitwise state check of the lazy IFEval write. Teacher-forced over real streams (reference rows + IFEval-type prompts
answered by the old engine): at every response step, from the same state, (A) the full write on all rows vs (B) the
subset write on the minimal capacity C = #rows that can write (+ optionally random extra rows). The write vector and
EVERY state tensor of the circuit must be bitwise equal."""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_env
OUT = os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/test_if_shift'; os.makedirs(OUT, exist_ok=True)
G = ref_env.load(OUT, ['--B_max', '64', '--fused', '0'])
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
import eng2
eng = G['eng']; IFT = G['IFT']; c = IFT.c; dev = 'cuda'
ref = json.load(open(os.environ.get('E2_REF', os.environ.get('E2_LIB', 'SET_E2_LIB') + '/runs/engine_ref/engine_reference.json')))
rs = random.Random(3)
ifp = [r for r in G['TESTSET'] if r['type'] in ('ifeval', 'combo')]; rs.shuffle(ifp); ifp = ifp[:32]
qi = [G['QID'](r) for r in ifp]
o = eng.generate(qi, G=1, greedy=True, max_new=256)
rows = list(zip(ref['q_ids'], ref['tokens'])) + list(zip(qi, o['tokens']))
B = len(rows); pre = eng.prefix_ids.tolist()
Tq = max(len(q) for q, _ in rows); Tr = max(len(r) for _, r in rows)
qids = torch.full((B, Tq), eng.eos, dtype=torch.long, device=dev); rids = torch.full((B, Tr), eng.eos, dtype=torch.long, device=dev)
for i, (q, r) in enumerate(rows):
    qids[i, Tq - len(q):] = torch.tensor(q, device=dev); rids[i, :len(r)] = torch.tensor(r, device=dev)
names = IFT.state_names
print('rows', B, 'Tq', Tq, 'Tr', Tr, flush=True)


def run(t0, seed=0):
    """IF circuit from reset with its global step counter starting at t0 (prompt memory written from column t0);
    prefix + left-padded question + teacher-forced response, full write at every response step (random h, fixed seed)."""
    c.reset(B); c.t.fill_(t0)
    g = torch.Generator(device=dev).manual_seed(seed)
    Ws, lat = [], []
    with torch.no_grad():
        for t in pre: c.step(torch.full((B,), t, device=dev))
        for k in range(Tq): c.step(qids[:, k])
        for k in range(Tr):
            c.step(rids[:, k])
            h = (torch.randn(B, 1024, device=dev, generator=g) * 3).to(torch.bfloat16)
            Ws.append(c.write(h).float().cpu())
            lat.append(torch.stack([x for x in c.L.values()], -1).cpu())
    return torch.stack(Ws), torch.stack(lat)


W0, L0 = run(0)
res = dict(rows=B, steps=Tr, nonzero_write_row_steps=int((W0.abs().sum(-1) > 0).sum()))
for t0 in (1, 37, 300, 2000):
    W1, L1 = run(t0)
    res[f'shift_{t0}'] = dict(write_mismatch_row_steps=int((W0 != W1).any(-1).sum()), latch_mismatch=int((L0 != L1).any(-1).sum()))
    print(t0, res[f'shift_{t0}'], flush=True)
print('IFSHIFT', json.dumps(res), flush=True)
json.dump(res, open(f'{OUT}/result.json', 'w'), indent=1)
