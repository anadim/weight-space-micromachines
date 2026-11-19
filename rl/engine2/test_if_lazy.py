"""Bitwise state check of the lazy IFEval write. Teacher-forced over real streams (reference rows + IFEval-type prompts
answered by the old engine): at every response step, from the same state, (A) the full write on all rows vs (B) the
subset write on the minimal capacity C = #rows that can write (+ optionally random extra rows). The write vector and
EVERY state tensor of the circuit must be bitwise equal."""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_env
OUT = os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/test_if_lazy'; os.makedirs(OUT, exist_ok=True)
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
c.reset(B)
with torch.no_grad():
    for t in pre: c.step(torch.full((B,), t, device=dev))
    for k in range(Tq): c.step(qids[:, k])


def snap():
    return {a: tree_flatten(getattr(c, a)) for a in names}


def clone(sn):
    return {a: ([x.clone() if torch.is_tensor(x) else x for x in lv], sp) for a, (lv, sp) in sn.items()}


def load(sn):
    for a, (lv, sp) in sn.items(): setattr(c, a, tree_unflatten([x.clone() if torch.is_tensor(x) else x for x in lv], sp))
    c.B = B


def bind(views, n):
    for a, (lv, sp, _) in views.items(): setattr(c, a, tree_unflatten(lv, sp))
    c.B = n


g = torch.Generator(device=dev).manual_seed(0)
res = dict(steps=0, write_mismatch=0, state_mismatch=0, can_rows_total=0, can_steps=0, extra_tests=0, nz_write_rows=0)
first_bad = None
with torch.no_grad():
    for k in range(Tr):
        c.step(rids[:, k])
        S0 = clone(snap())
        h = (torch.randn(B, 1024, device=dev, generator=g) * 3).to(torch.bfloat16)
        Wf = c.write(h).float(); Sf = clone(snap())
        for extra in (0, 7):
            load(S0)
            can = eng2.if_can_write(c); n = int(can.sum())
            C = min(B, max(1, n + extra))
            key = (~can).to(torch.int8) if extra == 0 else torch.where(can, 0, 1 + torch.randint(0, 5, (B,), device=dev, generator=g)).to(torch.int8)
            sel = torch.argsort(key, stable=True)[:C]
            v = {a: (lv, sp, [bool(torch.is_tensor(x) and x.dim() > 0 and x.shape[0] == B) for x in lv]) for a, (lv, sp) in snap().items()}
            ws = eng2.if_write_subset(c, v, h, sel, C, bind)
            Wl = torch.zeros_like(Wf); Wl[sel] = ws
            Sl = snap()
            wm = not torch.equal(Wf, Wl)
            sm = []
            for a in names:
                for x, y in zip(Sf[a][0], Sl[a][0]):
                    if torch.is_tensor(x) and not torch.equal(x, y): sm.append(a)
            res['write_mismatch'] += int(wm); res['state_mismatch'] += int(bool(sm)); res['extra_tests'] += int(extra > 0)
            if (wm or sm) and first_bad is None: first_bad = dict(step=k, extra=extra, write=wm, states=sorted(set(sm))[:10])
        res['steps'] += 1; res['can_rows_total'] += n; res['can_steps'] += int(n > 0)
        res['nz_write_rows'] += int((Wf.abs().sum(-1) > 0).sum())
        load(Sf)
res['rows'] = B; res['first_bad'] = first_bad
print('IFLAZY', json.dumps(res), flush=True)
json.dump(res, open(f'{OUT}/result.json', 'w'), indent=1)
