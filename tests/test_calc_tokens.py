"""v6 checks against Python: (1) 'expr =' -> exact digits (or silence); (2) 'expr = $' (the model wrote ' $') -> the digits
directly; (3) after the machine's number, a '.' written by the model -> a veto (OE), never a fraction digit; (4) calc_s4n:
silent whenever the exact result is negative.   MM=calc_s4n python test_v6.py N SEED"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os, random, sys, torch
from decimal import Decimal
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import SPECS, first, Attached
from micromachines.calcfmt import calc_ref
tok = AutoTokenizer.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base'))
model = AutoModelForCausalLM.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base'), torch_dtype=torch.float32).eval()
MM = os.environ.get('MM', 'calc_s4'); A = Attached(model, tok, machines=(MM,)); M = A.nets[MM]; I = A.all_ids[MM]
outs = SPECS[MM]['out']; out_tok = [first(I[s]) for s in outs]; E = model.get_input_embeddings()
sym = lambda ids: M.symbols(E(torch.tensor([ids])))
st = lambda t: M.step_sym(M.symbols(E(torch.tensor([t]))))[:, 0]
def run(ctx, model_os=None):
    """-> (text the machine writes, what it does on a following '.')"""
    ids = tok(ctx, add_special_tokens=False)['input_ids']
    with torch.no_grad():
        e = M.prime_sym(sym(ids))[:, -1]; out = []
        if model_os is not None:                                   # the model writes its own ' ' / ' $' after '='
            if e.max() < 0.5 or outs[int(e.argmax())] != 'OS': return '', None
            t = tok(model_os, add_special_tokens=False)['input_ids']; assert len(t) == 1
            e = st(t[0])
            if e.max() < 0.5: return '', None
        for _ in range(40):
            if e.max() < 0.5: return tok.decode(out).strip(), 'silent-before-end'
            s = outs[int(e.argmax())]
            if s == 'OE': break
            if s != 'OS': out.append(out_tok[int(e.argmax())])
            e = st(out_tok[int(e.argmax())])
        dot = tok('.', add_special_tokens=False)['input_ids']
        e2 = st(dot[0])
        after = 'none' if e2.max() < 0.5 else outs[int(e2.argmax())]
    return tok.decode(out).strip(), after
r = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0); N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
num = lambda: str(r.randint(1, 10 ** r.randint(1, 5))) if r.random() < 0.6 else f'{r.randint(0, 999)}.{r.randint(1, 99):02d}'.rstrip('0')
C = {'exact': 0, 'silent': 0, 'WRONG': 0, 'dot->OE': 0, 'dot->digit(BAD)': 0, 'dot->other': 0, 'neg spoken(BAD)': 0}
for i in range(N):
    a, b, o = num(), num(), r.choice('+-*/'); want = calc_ref(f'{a}{o}{b}')
    for ctx, mos in ((f'So it is {a} {o} {b} =', None), (f'So it is {a} {o} {b} =', ' $')):
        got, after = run(ctx, mos)
        if got == '': C['silent'] += 1; continue
        if MM == 'calc_s4n' and want.startswith('-'): C['neg spoken(BAD)'] += 1; print('NEG', a, o, b, got); continue
        if want and Decimal(got) == Decimal(want): C['exact'] += 1
        else: C['WRONG'] += 1; print('WRONG', a, o, b, mos, '| machine', got, '| exact', want)
        if after == 'OE': C['dot->OE'] += 1
        elif after in list('0123456789'): C['dot->digit(BAD)'] += 1; print('DOT-DIGIT', a, o, b, got)
        else: C['dot->other'] += 1
print(MM, N, 'expressions x 2 contexts:', C)
assert C['WRONG'] == 0 and (MM != 'calc_s4' or C['dot->digit(BAD)'] == 0), C
print('PASS')
