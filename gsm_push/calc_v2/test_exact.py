"""exactness test of calc v2's tracker on 10k random expressions of each new form (model-free; real Qwen3 tokenizer).
The text 'The total is <expr> =' is fed token by token; a simulated host then proposes ' ' + WRONG digits + ' units.'.
Pass = the number written after '=' equals the reference value (independent evaluator: python ast over Fractions)."""
import ast, random, sys, os, json
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calc2
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
TXT = {}
def ttext(i):
    if i not in TXT: TXT[i] = tok.decode([i])
    return TXT[i]
def T(s): return tok(s, add_special_tokens=False)['input_ids']

def ref_eval(s):
    s = s.replace('$', '').replace('×', '*').replace('÷', '/')
    import re
    s = re.sub(r'(?<=\d),(?=\d{3})', '', s)
    s = re.sub(r'(?<![A-Za-z])[xX](?![A-Za-z])', '*', s)
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant): return Fraction(ast.get_source_segment(s, n))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -ev(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b}[type(n.op)]()
        raise ValueError(n)
    return ev(ast.parse(s, mode='eval'))

R = random.Random(int(os.environ.get('SEED', 0)))
OPS = ['+', '-', 'x', '*', '/', '×']
def num(dec=0.25, big=False, comma=False, dollar=0.1):
    if big or comma:
        v = R.randint(1000, 99999999)
        s = f'{v:,}' if comma else str(v)
    else:
        v = R.randint(0, 999 if R.random() < 0.8 else 99999)
        s = str(v)
    if R.random() < dec: s += '.' + str(R.randint(0, 9999)).rstrip('0') if True else ''
    s = s.rstrip('.')
    if R.random() < dollar: s = '$' + s
    return s
def chain(k):
    return ' '.join(sum([[num(), R.choice(OPS)] for _ in range(k - 1)], []) + [num()])
def tree(depth):
    if depth == 0 or R.random() < 0.3: return num()
    a, b = tree(depth - 1), tree(depth - 1)
    op = R.choice(OPS)
    e = f'{a} {op} {b}'
    return f'({e})' if R.random() < 0.6 else e
def gen(form):
    if form == 'chain': return chain(R.randint(4, 9))
    if form == 'paren':
        while True:
            e = tree(R.randint(2, 4))
            if '(' in e: return e
    if form == 'comma':
        k = R.randint(2, 5); parts = [num(comma=True) if i == 0 or R.random() < 0.5 else num() for i in range(k)]
        R.shuffle(parts)
        return ' '.join(sum([[p, R.choice(OPS)] for p in parts[:-1]], []) + [parts[-1]])
    if form == 'sign':
        return '-' + chain(R.randint(2, 4)).lstrip('$')
    if form == 'decimal':
        k = R.randint(2, 6)
        parts = [str(R.randint(0, 999)) + '.' + str(R.randint(1, 999999)) for _ in range(k)]
        return ' '.join(sum([[p, R.choice(OPS)] for p in parts[:-1]], []) + [parts[-1]])

def run_one(prefix, mode='ext', calc_sim=None):
    global MODE
    mode = MODE
    """feed the prefix tokens (host-written), then a simulated host answer. calc_sim: (tokens flagged as calc writes)."""
    row = calc2.Row(mode, True)
    ids = T(prefix)
    out = []
    row.policy = POLICY
    def feed_host(i, wf=0):
        a = row.feed(ttext(i), wf, T)
        if a is None or a[0] == 'snap': out.append(i); return
        if a[0] in ('replace', 'rollback'):
            nt = row.q.pop(0); row.forced(ttext(nt), not row.q); out.append(nt)
            while row.q:
                nt = row.q.pop(0); row.forced(ttext(nt), not row.q); out.append(nt)
            return
        if a[0] == 'veto':
            raise RuntimeError('veto in prefix')
    for i in ids: feed_host(i)
    # host proposes ' ' then wrong digits: first digit triggers the replacement; then the host tries more digits
    host = T(' ') + T('7') + T('1')
    for i in host: feed_host(i)
    if POLICY == 'look': feed_host(T(' units')[0])
    # after the value: the host tries to write '3' (vetoed) -> engine alt ' units'
    a = row.feed('3', 0, T)
    if a and a[0] == 'veto': row.vetoed(' units')
    return row

def check(form, n):
    global MODE
    MODE = 'all' if form == 'decimal' else 'ext'
    bad = 0; nofire = 0; ex = []
    for _ in range(n):
        while True:
            e = gen(form)
            try:
                v = ref_eval(e); break
            except ZeroDivisionError:
                continue
        row = run_one(f'So the total is {e} =')
        txt = row.text
        if not [v for v in row.events if v.get('act') in ('preempt', 'rewrite')]:
            nofire += 1
            if len(ex) < 5: ex.append(('nofire', e))
            continue
        got = txt.split('=')[-1].strip().split(' ')[0]
        want = calc2.fmt(v)
        if want == '71': continue      # the simulated host happened to be right
        indep = abs(Fraction(got) - v) <= Fraction(1, 2 * 10 ** 6) and len(got.split('.')[-1] if '.' in got else '') <= 6
        if got != want or not indep:
            bad += 1
            if len(ex) < 5: ex.append((e, got, want))
    return dict(form=form, n=n, wrong=bad, nofire=nofire, ex=ex)

def check_exactlit(n):
    """'a / b =' written by calc_s4 (truncated), then 'c x <that literal> =' -> v2 writes the exact c*a/b (rounded)."""
    bad = 0; nofire = 0; ex = []
    for _ in range(n):
        while True:
            a_, b_ = R.randint(1, 999), R.randint(2, 999)
            v = Fraction(a_, b_)
            if Fraction(calc2.trunc(v)) != v: break
        c_ = R.randint(1, 99999)
        row = calc2.Row('ext', True, POLICY)
        for i in T(f'The share is {a_} / {b_} ='): row.feed(ttext(i), 0, T)
        lit = calc2.trunc(v)
        for k, i in enumerate(T(' ' + lit)): row.feed(ttext(i), 1, T)          # calc_s4-written (wflag 1)
        pre = f'. So the cost is {c_} x {lit} ='
        for i in T(pre):
            a = row.feed(ttext(i), 0, T)
        for i in T(' 9 dollars'):
            a = row.feed(ttext(i), 0, T)
            if a and a[0] in ('replace', 'rollback'):
                while row.q: nt = row.q.pop(0); row.forced(ttext(nt), not row.q)
        want = calc2.fmt(c_ * v); exact_plain = calc2.fmt(c_ * Fraction(lit))
        if not [x for x in row.events if x.get('act') in ('preempt', 'rewrite')]:
            if want != exact_plain: nofire += 1; ex.append(('nofire', a_, b_, c_)) if len(ex) < 5 else None
            continue
        got = row.text.split('=')[-1].strip().split(' ')[0]
        if got != want:
            bad += 1
            if len(ex) < 5: ex.append((a_, b_, c_, got, want))
    return dict(form='exactlit', n=n, wrong=bad, nofire=nofire, ex=ex)

def check_restate(n):
    """look policy: the host restates ('= 60 + 30') or writes the right value -> the tracker must not write."""
    acts = 0
    for _ in range(n):
        a_, b_, c_, d_ = (R.randint(1, 999) for _ in range(4))
        row = calc2.Row('ext', True, 'look')
        right = R.random() < 0.5
        tail = f' {a_ * b_ + c_ * d_} points.' if right else f' {a_ * b_} + {c_ * d_} = '
        for i in T(f'So they have {a_} x {b_} + {c_} x {d_} =' + tail):
            x = row.feed(ttext(i), 0, T)
            if x and x[0] in ('replace', 'rollback'): acts += 1; break
    return dict(form='restate_or_agree', n=n, acted=acts)

def check_restate_wrong(n):
    """verify_rs: '= <wrong restated expression> =' -> rewind and write the exact value of the first expression;
    a correct restatement -> no action."""
    bad = 0; acted_right = 0; missed = 0
    for _ in range(n):
        a_, b_, c_, d_ = (R.randint(1, 999) for _ in range(4))
        good = R.random() < 0.5
        x, y = a_ * b_, c_ * d_ + (0 if good else R.choice([-10, -1, 1, 9, 100]))
        row = calc2.Row('verify_rs', True, 'look')
        fired = False
        for i in T(f'So they have {a_} x {b_} + {c_} x {d_} = {x} + {y} =' + f' {x + y} points.'):
            z = row.feed(ttext(i), 0, T)
            if z and z[0] == 'rollback':
                fired = True
                while row.q: nt = row.q.pop(0); row.forced(ttext(nt), not row.q)
                break
        if good and fired: acted_right += 1
        if not good:
            if not fired: missed += 1
            elif row.text.split('=')[-1].strip() != str(a_ * b_ + c_ * d_): bad += 1
    return dict(form='restate_wrong', n=n, wrong=bad, missed=missed, acted_on_right=acted_right)

def check_silent(n):
    global MODE
    MODE = 'ext'
    """basic calc_s4 forms and non-arithmetic text: the tracker must not act (ext mode)."""
    acts = 0; ex = []
    texts = []
    for _ in range(n):
        k = R.random()
        if k < 0.5: texts.append(f'So he has {R.randint(0,9999)} {R.choice(["+","-","x","/"])} {R.randint(1,9999)} =')
        elif k < 0.7: texts.append(f'Let x be the number. Then 3x + {R.randint(1,99)} =')
        elif k < 0.85: texts.append(f'{R.randint(2,9)}J - {R.randint(1,99)} =')
        else: texts.append(f'The total is {R.randint(1,99)} hours x {R.randint(1,9)} + 2 =')
    for t in texts:
        row = run_one(t)
        if row.events:
            acts += 1
            if len(ex) < 5: ex.append(t)
    return dict(form='silent', n=n, acted=acts, ex=ex)

MODE = 'ext'
POLICY = os.environ.get('POLICY', 'look')
if __name__ == '__main__':
    N = int(os.environ.get('N', 10000))
    res = [check(f, N) for f in ('chain', 'paren', 'comma', 'sign', 'decimal')] + [check_exactlit(N), check_restate(N), check_restate_wrong(N), check_silent(N)]
    for r in res: print(json.dumps(r))
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_exact_result.json'), 'w'), indent=1)
