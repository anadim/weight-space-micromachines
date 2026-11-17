"""Calculator reference semantics (calc_ref) and GSM8K solution helpers (strip_solution removes the <<a*b=c>> annotations).

The tag tokens CS/CE/OS/OE are the calculator's original call format from an earlier, fine-tuned phase of the project.
The natural-text calculator implants used here (specs.py: calc_s3 / calc_s4) override all four with ordinary text tokens.
"""
import re, ast
from decimal import Decimal, getcontext, ROUND_DOWN
getcontext().prec = 80

CS, CE, OS, OE = '<tool_call>', '</tool_call>', '<|box_start|>', '<|box_end|>'
NI, F, D = 18, 6, 6          # max integer digits / fraction digits of an operand; division decimals
NUM = r'(-?)(\d+)(?:\.(\d+))?'
CALL_RE = re.compile('^' + NUM + r'([+\-*/])' + NUM + '$')


def fmt(x: Decimal) -> str:
    s = format(x, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    if s in ('-0', ''):
        s = '0'
    return s


def calc_ref(expr: str) -> str:
    """Exact reference. Returns '' for a malformed / out-of-range call (the 'tool error')."""
    m = CALL_RE.match(expr)
    if not m:
        return ''
    sa, ai, af, op, sb, bi, bf = m.groups()
    af, bf = af or '', bf or ''
    if len(ai) > NI or len(bi) > NI or len(af) > F or len(bf) > F:
        return ''
    a, b = Decimal(sa + ai + ('.' + af if af else '')), Decimal(sb + bi + ('.' + bf if bf else ''))
    if op == '+': return fmt(a + b)
    if op == '-': return fmt(a - b)
    if op == '*': return fmt(a * b)
    if b == 0: return ''
    return fmt((a / b).quantize(Decimal(1).scaleb(-D), rounding=ROUND_DOWN))   # truncate to D decimals


# ---------- GSM8K conversion ----------
ANN = re.compile(r'<<([^=<>]*)=([^<>]*)>>')
OPS = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}


class Fail(Exception):
    pass


def _reorder(node):
    """x*(b/c) -> (x*b)/c ; (b/c)*x -> (b*x)/c  (keeps intermediates integral)."""
    if isinstance(node, ast.BinOp):
        node.left, node.right = _reorder(node.left), _reorder(node.right)
        if isinstance(node.op, ast.Mult):
            l, r = node.left, node.right
            if isinstance(r, ast.BinOp) and isinstance(r.op, ast.Div):
                return ast.BinOp(ast.BinOp(l, ast.Mult(), r.left), ast.Div(), r.right)
            if isinstance(l, ast.BinOp) and isinstance(l.op, ast.Div):
                return ast.BinOp(ast.BinOp(l.left, ast.Mult(), r), ast.Div(), l.right)
    return node


def _chain(expr: str, reorder=False):
    """Split an arithmetic expression into binary calls. Returns (calls, final_string)."""
    src = expr.replace(' ', '')
    try:
        tree = ast.parse(src, mode='eval').body
    except SyntaxError:
        raise Fail('syntax')
    if reorder:
        tree = _reorder(tree)
    calls = []

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            seg = ast.get_source_segment(src, n)
            if seg is not None and re.fullmatch(r'\.\d+', seg):
                seg = '0' + seg                      # .25 -> 0.25
            if seg is None or not re.fullmatch(r'\d+(\.\d+)?', seg):
                raise Fail('literal')
            return seg
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant):
            return '-' + ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            l, r = ev(n.left), ev(n.right)
            e = f'{l}{OPS[type(n.op)]}{r}'
            out = calc_ref(e)
            if out == '':
                raise Fail('range')
            calls.append((e, out))
            return out
        raise Fail('node')

    fin = ev(tree)
    if not calls:
        raise Fail('nocall')
    return calls, fin


def _agrees(fin: str, gold: str) -> bool:
    try:
        g, f = Decimal(gold.replace(',', '')), Decimal(fin)
    except Exception:
        return False
    if g == f:
        return True
    dec = -g.as_tuple().exponent if g.as_tuple().exponent < 0 else 0
    return abs(g - f) <= Decimal(1).scaleb(-dec) and dec < D   # gold is a rounding of ours


def convert_annotation(expr, gold):
    for ro in (False, True):
        try:
            calls, fin = _chain(expr, ro)
        except Fail:
            continue
        if _agrees(fin, gold):
            return calls
    return None


def convert_solution(answer: str, stats=None):
    """GSM8K answer field -> (body with calls, final answer)."""
    body, final = answer.split('####')

    def rep(m):
        calls = convert_annotation(m.group(1), m.group(2))
        if stats is not None:
            stats['ann'] += 1
            stats['conv'] += calls is not None
            stats['calls'] += len(calls or [])
        if calls is None:
            return ''
        return ''.join(f'{CS}{e}{CE}{OS}{o}{OE}' for e, o in calls)

    return ANN.sub(rep, body).strip(), final.strip().replace(',', '')


def strip_solution(answer: str):
    body, final = answer.split('####')
    return ANN.sub('', body).strip(), final.strip().replace(',', '')
