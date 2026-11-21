"""CPU fuzz test of the branch-token floor (min-p) in job/stk/c2_engine.py against a numpy reference.

The tested code (trunc_keep, C2Engine.set_trunc / _trunc) is lifted verbatim from c2_engine.py with ast, so the test does
not need the engine / triton.  Checks on random logits (bf16-valued, several scales, ties, -inf entries):
  1. kept set == numpy fp32 reference  (z - max z >= log m, z = logit / T)  exactly;
  2. kept set == float64 probability definition p_i >= m * p_max  except within 1e-5 (relative) of the boundary;
  3. renormalised probabilities softmax(truncated / T) == p * keep / sum(p * keep)  (fp64 reference, atol 1e-6);
  4. Gumbel-argmax sampling (the sample kernel's rule) never returns a dropped token, and the argmax is always kept;
  5. m = 0 (forced path) returns logits.float() bit-identically;  X gating: rows past X tokens of their attempt are untouched.
usage: python3 test_trunc.py   (needs torch + numpy; run on the node venv)
"""
import ast, math, os, sys
import numpy as np
import torch

SRC = next(p for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)), d, 'stk', 'c2_engine.py') for d in ('job', '.')] if os.path.exists(p))


def _lift():
    tree = ast.parse(open(SRC).read()); ns = {'torch': torch, 'math': math}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'trunc_keep']
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'C2Engine')
    meth = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('set_trunc', '_trunc')]
    assert len(body) == 1 and len(meth) == 2
    holder = ast.ClassDef(name='Eng', bases=[], keywords=[], body=meth, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=body + [holder], type_ignores=[])), SRC, 'exec'), ns)
    return ns['trunc_keep'], ns['Eng']


trunc_keep, Eng = _lift()


def make_eng(T, m, x=0, B=1, serial=False, off=None):
    e = Eng(); e.dev = 'cpu'; e.temp = torch.tensor(float(T)); e.set_trunc(True, m, x)
    if serial:
        e.rowid = torch.arange(B); e.astart = torch.zeros(B, dtype=torch.long)
        e.glen = torch.as_tensor(off, dtype=torch.long) - 1          # off = glen + 1 - astart
    return e


def ref_keep32(l, T, m):
    z = l.astype(np.float32) / np.float32(T)
    lm = np.float32(math.log(m)) if m > 0 else np.float32(-np.inf)
    return (z - z.max(-1, keepdims=True)) >= lm


def rand_logits(rng, B, V):
    sc = rng.choice([0.5, 2.0, 5.0, 12.0])
    l = rng.standard_normal((B, V)) * sc
    if rng.random() < 0.3: l[:, rng.integers(0, V, 3)] = l.max() + 4.0                # sharp peak
    if rng.random() < 0.3: l[:, :V // 7] = np.round(l[:, :V // 7])                     # ties
    if rng.random() < 0.2: l[:, rng.integers(0, V, 5)] = -np.inf                       # pre-masked tokens
    return torch.tensor(l, dtype=torch.float32).to(torch.bfloat16)                       # host logits are bf16


def main():
    rng = np.random.default_rng(0); torch.manual_seed(0)
    n_case = n_band = 0
    for it in range(300):
        B, V = int(rng.integers(1, 9)), int(rng.choice([7, 64, 1000, 4099]))
        T = float(rng.choice([0.5, 0.7, 1.0, 1.3])); m = float(rng.choice([0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]))
        lb = rand_logits(rng, B, V)
        e = make_eng(T, m, B=B)
        out = e._trunc(lb)
        assert out.dtype == torch.float32
        keep = torch.isfinite(out).numpy()
        l = lb.float().numpy()
        # 1. exact fp32 reference
        rk = ref_keep32(l, T, m) & np.isfinite(l)
        assert (keep == rk).all(), ('fp32 mismatch', it)
        np.testing.assert_array_equal(out.numpy()[keep], l[keep])                       # kept logits untouched
        # 2. probability definition (fp64), away from the boundary
        z64 = l.astype(np.float64) / T
        p = np.exp(z64 - z64.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        ratio = p / p.max(-1, keepdims=True)
        dk = (ratio >= m) & np.isfinite(l)
        far = np.abs(ratio - m) > 1e-5 * m if m > 0 else np.ones_like(dk)
        assert (keep[far] == dk[far]).all(), ('prob-def mismatch', it)
        n_band += int((~far).sum())
        # 3. renormalised probabilities
        q = torch.softmax(out.double() / T, -1).numpy()
        pr = p * keep; pr /= pr.sum(-1, keepdims=True)
        np.testing.assert_allclose(q, pr, atol=1e-6, rtol=0)
        # 4. sampling never picks a dropped token; argmax always kept
        assert keep[np.arange(B), l.argmax(-1)].all()
        for _ in range(4):
            u = torch.rand(B, V).clamp_(min=1e-20)
            s = (out / T - torch.log(-torch.log(u))).argmax(-1).numpy()
            assert keep[np.arange(B), s].all()
        # 5. m = 0 is the identity on the fp32 logits
        if m == 0.0:
            assert torch.equal(out, lb.float())
        # first-token path (_wave): same rule on tempered logits
        zf = lb.float() / T
        kf = trunc_keep(zf, e.tr_logm).numpy()
        assert (kf == (ref_keep32(l, T, m))).all()
        n_case += 1
    # X gating: rows whose offset in the attempt exceeds X are untouched
    lb = rand_logits(rng, 6, 500)
    off = [1, 5, 31, 32, 33, 400]
    e = make_eng(1.0, 0.2, x=32, B=6, serial=True, off=off)
    out = e._trunc(lb)
    rk = ref_keep32(lb.float().numpy(), 1.0, 0.2)
    for j, o in enumerate(off):
        k = torch.isfinite(out[j]).numpy()
        if o <= 32: assert (k == rk[j]).all()
        else: assert k.all() and torch.equal(out[j], lb[j].float())
    e0 = make_eng(1.0, 0.2, x=0, B=6, serial=True, off=off)                             # x=0: every token
    assert (torch.isfinite(e0._trunc(lb)).numpy() == rk).all()
    print(f'test_trunc OK: {n_case} random cases, {n_band} entries within 1e-5 of the p/p_max = m boundary, X gating OK')


if __name__ == '__main__':
    main()
