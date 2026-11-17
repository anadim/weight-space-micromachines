"""Machine-alone exactness tests for expr_gen / calc_gen / bool_gen on OUR random text (no host generation): the context ends
with '=', the machine's writes are fed back, compare with Python. Also must-be-silent cases and expr/calc disjointness.
python test_tools.py expr|bool N SEED"""
import os, sys, random, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..')); sys.path.insert(0, os.environ.get('WSM_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
import teamc_tools as TT
from micromachines import SPECS, first
which, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', torch_dtype=torch.float32).cuda().eval()
MS = ('expr_gen', 'calc_gen') if which == 'expr' else ('bool_gen',)
t0 = time.time(); A = TT.attach_gen(model, tok, MS, veto=None, oe='dotnl'); print('attach', time.time() - t0, 's', flush=True)
for n, M in A.nets.items(): print(n, 'units', M.stack1.net.n_units if hasattr(M.stack1, 'net') else M.stack1.n_units, M.stack2.n_units, 'NS', M.NS, flush=True)
E = model.get_input_embeddings()


def drive(name, ctx, maxn=18):
    M = A.nets[name]; I = A.all_ids[name]; outs = SPECS[name]['out']; ot = [first(I[s]) for s in outs]
    ids = tok(ctx, add_special_tokens=False)['input_ids']; out = []
    with torch.no_grad():
        e = M.prime_sym(M.symbols(E(torch.tensor([ids], device='cuda'))))[:, -1]
        for _ in range(maxn):
            if e.max() < 0.5: break
            j = int(e.argmax()); out.append(ot[j])
            if outs[j] in ('OE', 'VD'): break
            if name == 'bool_gen' and outs[j] in ('VT', 'VF'): break
            e = M.step_sym(M.symbols(E(torch.tensor([out[-1]], device='cuda'))))[:, 0]
    return tok.decode(out)


r = random.Random(seed)
C = dict(exact=0, silent_ok=0, WRONG=0, MISSED=0, BOTH=0)
if which == 'expr':
    def num():
        x = r.random()
        return str(r.randint(0, 9)) if x < 0.5 else str(r.randint(10, 999)) if x < 0.85 else str(r.randint(1000, 999999))
    def expr(d=0):
        n = r.randint(1, 4); parts = []
        for i in range(n):
            if d < 3 and r.random() < 0.25: a = '(' + expr(d + 1) + ')'
            else: a = num()
            if r.random() < 0.25: a = '-' + a
            parts.append(a)
            if i < n - 1: parts.append(r.choice(['+', '-', '*', '*', '×', 'x']) if r.random() < 0.05 else r.choice(['+', '-', '*']))
        return ' '.join(parts)
    PRE = ['A = ', 'So ', 'which is ', 'Total: ', 'Then ', '\n', 'we get ', 'cost = ', 'Step 2: ', 'equals ', '. Hence ']
    for i in range(N):
        e = expr()
        if not any(c in e for c in '+-*×x') or ' ' not in e: e = e + ' + ' + num()
        sp = r.random()
        txt = e.replace(' ', '') if sp < 0.2 else e.replace('( ', '(') if sp < 0.9 else e
        want = TT.expr_ref(txt.replace('x', '*').replace('×', '*'))
        ctx = r.choice(PRE) + txt + r.choice([' =', '='])
        got = {m: drive(m, ctx) for m in MS}
        ge = got['expr_gen'].replace('.\n', '').strip(); gc = got['calc_gen'].strip()
        if ge and gc: C['BOTH'] += 1; print('BOTH', repr(ctx), got); continue
        g = ge or gc
        if not want:
            if g: C['WRONG'] += 1; print('SHOULD BE SILENT', repr(ctx), got)
            else: C['silent_ok'] += 1
        elif not g: C['MISSED'] += 1; print('missed', repr(ctx), 'want', want)
        elif g == want: C['exact'] += 1
        else: C['WRONG'] += 1; print('WRONG', repr(ctx), got, 'want', want)
    print('must-be-silent / decimal / guard cases:')
    for ctx in ['3.5 + 2 =', 'x - 3 =', '1,000 + 5 =', '2 ** 3 - 1 =', '8 / 2 + 1 =', 'A = 7 =', 'A = (5) =', 'is 3 + =', 'is (3 + 4 =',
                'is 3 + 4) =', '2^3 - 1 =', 'value 5 - 2 =', 'So 12 * 3.5 =', 'So 21 / 7 =', 'is 3 4 + 1 =', '((((((1 + 2)))))) =', 'is 3 (4) =',
                'So 999999 * 999999 * 9 =', 'So 2 * 3 =', 'So (2 + 3) * 4 =', 'x = 7-(-3)=', 'We have 5 apples and 3 more, 5 + 3 =', 'so 3 - -4 =',
                '(9 - (-56)) + -9 =', 'A = ((-5 + 9 * -4 - 0) * (4 + -7 + 0 * -5)) =', 'thus -(2 + 3) * -4 =', 'So 1 + 1 + 1 + 1 + 1 + 2 + 3 + 1 + 1 + 1 =', 'is 5. 3 + 4 =', 'is 2.5 * 4 =', 'it was 1,250 - 50 =', 'sqrt(4+5)=', 'f(3) + 2 =', ')x7+1 =', 'abc2 + 3 =', 'A=3+4=', 'Total:3+4=', 'He scored 5 plus 3 * 4 =', 'then 10 divided by 2 + 3 =', 'Plus 3 * 4 =', 'The sum of 5 and 3 * 4 =']:
        print('  ', repr(ctx), {m: drive(m, ctx) for m in MS})
else:
    def bexpr(d=0):
        n = r.randint(1, 3); parts = []
        for i in range(n):
            a = '( ' + bexpr(d + 1) + ' )' if d < 2 and r.random() < 0.3 else r.choice(['True', 'False'])
            a = ' '.join(['not'] * r.choice([0, 0, 1, 2]) + [a]); parts.append(a)
            if i < n - 1: parts.append(r.choice(['and', 'or']))
        return ' '.join(parts)
    PRE = ['A = ', 'Z = ', 'so the value: x = ', 'result = ', 'we get: ', '\n', 'Then ', 'flag = ']
    for i in range(N):
        e = bexpr()
        if r.random() < 0.5: e = e.replace('( ', '(').replace(' )', ')')
        if r.random() < 0.1: e = e.replace('True', 'true').replace('False', 'false')
        ok_len = len(tok(e)['input_ids']) <= TT.K_BO - 3 and any(w in e.split() or w in e for w in ('not', 'and', 'or'))
        ctx = r.choice(PRE) + e + r.choice([' =', '='])
        g = drive('bool_gen', ctx).strip()
        want = TT.bool_ref(e) if ok_len else ''
        if not want:
            if g: C['WRONG'] += 1; print('SHOULD BE SILENT', repr(ctx), g)
            else: C['silent_ok'] += 1
        elif not g: C['MISSED'] += 1; print('missed', repr(ctx), 'want', want)
        elif g == want: C['exact'] += 1
        else: C['WRONG'] += 1; print('WRONG', repr(ctx), g, 'want', want)
    for ctx in ['x = True =', '!True and False =', 'a and not True =', 'True AND not False =', 'not ((True)) =', 'x if not True else False =',
                'We know that the switch is on, so True and not False =', 'print(True and False) =', '(not False) or (False) =', 'if (not True) or (False and True) then', 'cond = (True and not (False or not True))=', 'x = not(True) and (not(False)) =', '!(True) and False =', 'any(True and False) =', 'flag=True and not False=']:
        print('  ', repr(ctx), drive('bool_gen', ctx))
print(C, flush=True)
