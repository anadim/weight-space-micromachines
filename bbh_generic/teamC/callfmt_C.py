"""Call format and exact Python REFERENCES for the Team-C tools (tests and training-data writing only; never at inference --
at inference the compiled ReLU circuits in teamc_tools.py do the work).

Shared envelope (Qwen3 single tokens):  <tool_call>NAME ARGS</tool_call><tool_response>RESULT</tool_response>
The host writes up to and including </tool_call>; the tool writes <tool_response>RESULT</tool_response>.
RESULT empty = the call is malformed or out of range. (calc: a malformed call gets no response at all; see below.)

------------------------------------------------------------------------------------------------------------------------
expr   -- exact value of an INTEGER arithmetic expression.                         machine: expr_call
    expr E
    E    integers (any number of digits; |every intermediate value| <= 10^12), binary + - * (also × and ' x'),
         unary minus (also '--4', '+-4'), parentheses nested at most 4 deep; spaces anywhere between tokens.
         At least one binary operator. At most ~70 tokens in the whole call.
    RESULT  the value, e.g. -41
    e.g. expr (-5 + 9 * -4 - 0) * (4 + -7)  -> 123        expr 3 + 4 * 2 -> 11        expr 7 / 2 -> (empty: no division)
------------------------------------------------------------------------------------------------------------------------
bool   -- truth value of a propositional formula.                                   machine: bool_call
    bool B
    B    the literals True False (also true false), the operators not and or, parentheses nested at most 4 deep;
         Python precedence (not > and > or). At least one operator. At most ~45 tokens.
    RESULT  True or False
    e.g. bool not (True and False) or False -> True
------------------------------------------------------------------------------------------------------------------------
calc   -- exact a op b on signed decimals (the repo's calculator calc_s4).            machine: calc_call
    calc A op B
    A, B signed decimals (<= 18 integer digits, <= 6 fraction digits), op one of + - * /, spaces optional around op.
    RESULT  the exact value; division is truncated to 6 decimals; trailing zeros removed (calcfmt.calc_ref)
    e.g. calc 3.5 * 2 -> 7        calc 22 / 7 -> 3.142857        calc -1.25 + 0.5 -> -0.75
    A call calc cannot evaluate (1 / 0, three operations, parentheses, '$', letters, too many digits) gets an empty
    response <tool_response></tool_response>.
------------------------------------------------------------------------------------------------------------------------
Frozen-host (pattern-triggered) versions of the same functions: expr_gen ('EXPR =' -> ' VALUE.\n'), bool_gen
('BOOL =' -> ' True.' / ' False.'), calc_gen ('a op b =' with a decimal point or a division -> ' VALUE').
"""
import ast, random, re, sys, os
sys.path.insert(0, os.environ.get('WSM_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from micromachines.calcfmt import calc_ref as _calc_ref

CS, CE, OS, OE = '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>'
BV, DMAX = 10 ** 12, 4


def _depth(e):
    d = m = 0
    for c in e:
        d += (c == '(') - (c == ')'); m = max(m, d)
    return m


def expr_ref(E):
    e = E.replace('×', '*').replace(' x ', ' * ')
    if not re.fullmatch(r'[\d+\-*() ]+', e) or _depth(e) > DMAX or '**' in e: return ''
    try: t = ast.parse(e.strip(), mode='eval')
    except SyntaxError: return ''
    ok, binop = True, False
    def chk(n):
        nonlocal ok, binop
        if isinstance(n, ast.BinOp):
            binop = True; ok &= isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)); chk(n.left); chk(n.right)
        elif isinstance(n, ast.UnaryOp): ok &= isinstance(n.op, (ast.USub, ast.UAdd)); chk(n.operand)
        elif isinstance(n, ast.Constant): ok &= isinstance(n.value, int) and abs(n.value) <= BV
        else: ok = False
        # intermediate bound
        if ok and isinstance(n, (ast.BinOp, ast.UnaryOp)):
            v = eval(compile(ast.Expression(n), '', 'eval')); ok &= abs(v) <= BV
    chk(t.body)
    if not (ok and binop): return ''
    return str(eval(compile(t, '', 'eval')))


def bool_ref(B):
    b = B.replace('true', 'True').replace('false', 'False')
    if not re.fullmatch(r'(\s|True|False|not|and|or|\(|\))+', b) or _depth(b) > DMAX or not re.search(r'\b(not|and|or)\b', b): return ''
    try: v = eval(b, {'__builtins__': {}})
    except Exception: return ''
    return str(v) if isinstance(v, bool) else ''


def calc_ref(A):
    return _calc_ref(A.replace(' ', ''))


def run_ref(args):
    name, _, rest = args.partition(' ')
    return {'expr': expr_ref, 'bool': bool_ref, 'calc': calc_ref}.get(name, lambda x: '')(rest)


CALL_RE = re.compile(re.escape(CS) + r'(.*?)' + re.escape(CE))


def fill_calls(text):
    return CALL_RE.sub(lambda m: f'{CS}{m.group(1)}{CE}{OS}{run_ref(m.group(1))}{OE}', text)


# ---------------- random calls (tests / training data) ----------------
def rand_int_expr(r, d=0, nmax=4):
    n = r.randint(1 if d else 2, nmax); parts = []
    for i in range(n):
        if d < 2 and r.random() < 0.25: a = '(' + rand_int_expr(r, d + 1, 3) + ')'
        else:
            v = r.choice([r.randint(0, 9), r.randint(0, 99), r.randint(0, 9999)])
            a = ('-' if r.random() < 0.25 else '') + str(v)
        parts.append(a)
        if i < n - 1: parts.append(r.choice('+-*'))
    sp = r.random() < 0.7
    return (' ' if sp else '').join(parts)


def rand_bool(r, d=0):
    n = r.randint(1, 3); parts = []
    for i in range(n):
        a = '(' + rand_bool(r, d + 1) + ')' if d < 2 and r.random() < 0.3 else r.choice(['True', 'False'])
        parts.append(' '.join(['not'] * r.choice([0, 0, 1, 2]) + [a]))
        if i < n - 1: parts.append(r.choice(['and', 'or']))
    return ' '.join(parts)


def rand_dec(r):
    v = r.choice([str(r.randint(0, 999)), f'{r.randint(0, 99)}.{r.randint(0, 99):02d}', f'{r.randint(0, 9)}.{r.randint(1, 9)}'])
    return ('-' if r.random() < 0.2 else '') + v


def random_call(r):
    k = r.random()
    if k < 0.45:
        e = rand_int_expr(r)
        return 'expr ' + e
    if k < 0.75:
        e = rand_bool(r)
        if not re.search(r'\b(not|and|or)\b', e): e = 'not ' + e
        return 'bool ' + e
    return f'calc {rand_dec(r)} {r.choice("+-*/")} {rand_dec(r)}'


EDGE_CASES = ['expr 3 +', 'expr 2 ** 3', 'expr 7 / 2', 'expr 5', 'expr (1+2)*3', 'expr ((((1+1))))*2', 'expr -(2+3)*-4',
              'expr 999999*999999', 'expr 9999999*9999999', 'bool True', 'bool not (True and False) or False', 'bool (not true)',
              'bool True and', 'calc 1 / 0', 'calc 3.5 * 2', 'calc 22 / 7', 'calc 7*6', 'track 123:12?1', 'order 3:1<2']


if __name__ == '__main__':
    r = random.Random(0)
    for _ in range(8):
        c = random_call(r); print(c, '->', repr(run_ref(c)))
    for c in EDGE_CASES: print(c, '->', repr(run_ref(c)))
