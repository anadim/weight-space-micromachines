"""Randomized check of MM15 (expr) and MM13 (sum_f) together against Python on BBH-multistep-style sub-expressions, plus
the cases each must leave alone. Machine alone, fp32; '=' is the last context token; the machine writes everything.
python test_expr.py N SEED"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import random, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import SPECS, first, Attached
tok = AutoTokenizer.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base'))
model = AutoModelForCausalLM.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base'), torch_dtype=torch.float32).eval()
MS = ('expr_n', 'sum_fn', 'calc_s4'); A = Attached(model, tok, machines=MS); E = model.get_input_embeddings()
def drive(name, ctx):
    M = A.nets[name]; I = A.all_ids[name]; outs = SPECS[name]['out']; ot = [first(I[s]) for s in outs]
    st = lambda t: M.step_sym(M.symbols(E(torch.tensor([t]))))[:, 0]
    ids = tok(ctx, add_special_tokens=False)['input_ids']; out = []
    with torch.no_grad():
        e = M.prime_sym(M.symbols(E(torch.tensor([ids]))))[:, -1]
        for _ in range(14):
            if e.max() < 0.5: break
            sym = outs[int(e.argmax())]
            if sym == 'OE': break
            out.append(ot[int(e.argmax())]); e = st(out[-1])
    return tok.decode(out)
r = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0); N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
def num(): v = r.randint(-9, 9); return str(v)
C = dict(exact=0, silent=0, WRONG=0)
for i in range(N):
    k = r.randint(2, 4); e = num()
    for _ in range(k - 1): e += ' ' + r.choice('+-*') + ' ' + num()
    if r.random() < 0.7: e = '(' + e + ')'
    want = str(eval(e))
    for pre in ('Let\'s calculate A = ', 'where B = '):
        got = {m: drive(m, pre + e + ' =') for m in MS if not (m == 'calc_s4')}
        g = [v for v in got.values() if v.strip()]
        if not g: C['silent'] += 1; continue
        if len(g) > 1: C['WRONG'] += 1; print('TWO MACHINES', e, got); continue
        if g[0].strip() == want: C['exact'] += 1
        else: C['WRONG'] += 1; print('WRONG', repr(pre + e), got, 'want', want, flush=True)
for e in ['2 * 3 + 4.5', '3 * (4 + 5)', '(2 + 3) * 4', '12 * 3 + 1', '9 * 5', 'x * 3 + 2']:
    got = {m: drive(m, 'A = ' + e + ' =') for m in ('expr_n', 'sum_fn')}
    print('must-be-silent', e, got)
print(N, 'exprs x 2 contexts', C, flush=True)
assert C['WRONG'] == 0, C
print('PASS')
