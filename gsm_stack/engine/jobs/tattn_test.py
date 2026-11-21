"""tattn test: correctness vs a dense fp32 reference (and flex's error on the same problem), timing at L=16k."""
import sys, os, time, json, torch
sys.path.insert(0, os.environ.get('STK_DIR', 'stk'))
import tattn as TA
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
torch._dynamo.config.cache_size_limit = 64
flex = torch.compile(flex_attention, dynamic=False)
SMALL = dict(BLOCK_M=32, BLOCK_N=32, num_stages=1, BLOCK_M1=16, BLOCK_N1=32, BLOCK_M2=32, BLOCK_N2=16)
dev = 'cuda'


def problem(seed, Lp, ngroups, segs=(1, 6), rl=(1, 600), padmax=300):
    import random
    r = random.Random(seed); g, d = [], []; did = 0
    for gi in range(ngroups):
        ql = r.randint(40, 200); g += [gi] * ql; d += [-1] * ql
        for _ in range(r.randint(*segs)):
            n = r.randint(*rl); g += [gi] * n; d += [did] * n; did += 1
    pad = r.randint(0, padmax); g += [-1] * pad; d += [-1] * pad
    L = len(g)
    gen = torch.Generator(device=dev).manual_seed(seed)
    mk = lambda *s: (torch.randn(*s, device=dev, generator=gen) * 1.5).to(torch.bfloat16)
    q, k, v = mk(L, 16, 128), mk(L, 8, 128), mk(L, 8, 128); Kp, Vp = mk(Lp, 8, 128), mk(Lp, 8, 128)
    return q, k, v, Kp, Vp, torch.tensor(g, dtype=torch.int32, device=dev), torch.tensor(d, dtype=torch.int32, device=dev)


_BM = {}


def flex_attn(q, k, v, Kp, Vp, grp, doc, ko):
    L = q.shape[0]; Lp = Kp.shape[0]
    key = (id(grp), L, Lp)
    if key not in _BM:
        gk = torch.cat([torch.full((Lp,), -7, dtype=torch.int32, device=dev), grp]); dk = torch.cat([torch.full((Lp,), -7, dtype=torch.int32, device=dev), doc])

        def mm(b, h, qi, kv):
            return (kv < Lp) | ((gk[kv] == grp[qi]) & (kv - Lp <= qi) & ((dk[kv] < 0) | (dk[kv] == doc[qi])))
        _BM[key] = create_block_mask(mm, 1, None, L, Lp + L, device=dev)
    bm = _BM[key]
    K = torch.cat([Kp, k], 0).transpose(0, 1)[None]; V = torch.cat([Vp, v], 0).transpose(0, 1)[None]
    o = flex(q.transpose(0, 1)[None], K, V, block_mask=bm, enable_gqa=True, kernel_options=ko)
    return o[0].transpose(0, 1)


def rel(a, b): return float((a.float() - b.float()).norm() / b.float().norm())


for seed in range(4):
    q, k, v, Kp, Vp, grp, doc = problem(seed, 900, 6)
    mt = TA.meta(grp, doc, q.shape[0])
    do = (torch.randn_like(q.float()) ).to(torch.bfloat16)
    qr, kr, vr = [x.float().requires_grad_(True) for x in (q, k, v)]
    o_ref = TA.reference(qr, kr, vr, Kp, Vp, grp, doc); o_ref.backward(do.float())
    res = dict(seed=seed, L=q.shape[0])
    for name, fn in (('triton', lambda q_, k_, v_: TA.group_attn(q_, k_, v_, Kp, Vp, grp, doc, mt)),
                     ('flex_small', lambda q_, k_, v_: flex_attn(q_, k_, v_, Kp, Vp, grp, doc, SMALL))):
        qt, kt, vt = [x.clone().requires_grad_(True) for x in (q, k, v)]
        o = fn(qt, kt, vt); o.backward(do)
        res[name] = dict(o=round(rel(o, o_ref), 5), dq=round(rel(qt.grad, qr.grad), 5), dk=round(rel(kt.grad, kr.grad), 5),
                         dv=round(rel(vt.grad, vr.grad), 5))
    res['o_bf16_floor'] = round(rel(o_ref.to(torch.bfloat16), o_ref), 5)
    print('TATTN', json.dumps(res), flush=True)

# timing at the real size: L ~ 16k, Lp 900, groups of 8 rows x ~4 segments
q, k, v, Kp, Vp, grp, doc = problem(7, 900, 3, segs=(20, 32), rl=(60, 400), padmax=50)
L = q.shape[0]; do = torch.randn_like(q.float()).to(torch.bfloat16)
print('TIME L', L, flush=True)


def bench(name, fn, n=10):
    qt, kt, vt = [x.clone().requires_grad_(True) for x in (q, k, v)]
    for _ in range(2): o = fn(qt, kt, vt); o.backward(do)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): o = fn(qt, kt, vt)
    torch.cuda.synchronize(); tf = (time.time() - t0) / n
    t0 = time.time()
    for _ in range(n): o = fn(qt, kt, vt); o.backward(do)
    torch.cuda.synchronize(); tfb = (time.time() - t0) / n
    print(f'TIME {name} fwd {tf * 1e3:.2f} ms  fwd+bwd {tfb * 1e3:.2f} ms', flush=True)


mt = TA.meta(grp, doc, L)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10): TA.meta(grp, doc, L)
torch.cuda.synchronize(); print(f'TIME meta {(time.time() - t0) * 100:.2f} ms', flush=True)
# useful flops: visible (q,k) pairs x H x HD x 4 (fwd)
i = torch.arange(L, device=dev)
pairs = 0
for c0 in range(0, L, 2048):
    ii = i[c0:c0 + 2048]
    vis = (grp[None, :] == grp[ii][:, None]) & (i[None, :] <= ii[:, None]) & ((doc[None, :] < 0) | (doc[None, :] == doc[ii][:, None]))
    pairs += int(vis.sum())
pairs += L * Kp.shape[0]
print(f'TIME visible pairs {pairs:.3e}  fwd TFLOP {pairs * 16 * 128 * 4 / 1e12:.3f}', flush=True)
bench('triton', lambda q_, k_, v_: TA.group_attn(q_, k_, v_, Kp, Vp, grp, doc, mt))
for cfg in json.loads(os.environ.get('TA_CFGS', '[]')):
    TA.CFG.update(cfg); bench('triton ' + json.dumps(cfg), lambda q_, k_, v_: TA.group_attn(q_, k_, v_, Kp, Vp, grp, doc, mt))
bench('flex_small', lambda q_, k_, v_: flex_attn(q_, k_, v_, Kp, Vp, grp, doc, SMALL))
for ko in (None, dict(BLOCK_M=64, BLOCK_N=64, num_stages=2, BLOCK_M1=32, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=32)):
    try: bench('flex ' + str(ko), lambda q_, k_, v_: flex_attn(q_, k_, v_, Kp, Vp, grp, doc, ko))
    except Exception as e: print('TIME flex', ko, 'FAILED', repr(e)[:300], flush=True)
