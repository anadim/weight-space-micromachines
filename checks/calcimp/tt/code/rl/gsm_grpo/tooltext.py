"""[35] arm X: a TEXT calculator tool (no implant, nothing in hidden states).

The model writes <calc>EXPR</calc>; the decode loop pauses that row, Python evaluates EXPR exactly (Fraction arithmetic),
and the result text is forced into the context (one decode step per result token = the same KV as a re-prefill); then
generation resumes. Prompt: the same 8 GSM8K train shots as common.prefix_ids, but GSM8K's own calculator annotations
<<EXPR=RESULT>>RESULT are kept as <calc>EXPR</calc>RESULT instead of being stripped (the GSM8K-paper calculator format,
text tags)."""
import ast, json, re
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP
import common as C

OPEN, CLOSE = '<calc>', '</calc>'
ANN = re.compile(r'<<([^=<>]*)=([^<>]*)>>')
MAX_CALLS = 16          # per answer; later </calc> are left to the model
MAX_EXPR = 120          # characters
DEC = 6                 # decimals of a non-integer result (rounded half-up, trailing zeros stripped)


def calc_prefix_ids(tok):
    shots = json.load(open(f'{C.REPO}/data/gsm8k/shots_v1.json'))
    T = lambda s: tok(s, add_special_tokens=False)['input_ids']
    ids = []
    for s in shots:
        body = ANN.sub(lambda m: OPEN + m.group(1) + CLOSE, s['solution']).strip()
        ids += T(f"Question: {s['question']}\nAnswer: {body}\n#### {str(s['answer']).replace(',', '')}\n\n")
    return ids, {s['question'] for s in shots}


def _ev(n):
    if isinstance(n, ast.Expression): return _ev(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return Fraction(str(n.value))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        v = _ev(n.operand); return -v if isinstance(n.op, ast.USub) else v
    if isinstance(n, ast.BinOp):
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Add): return a + b
        if isinstance(n.op, ast.Sub): return a - b
        if isinstance(n.op, ast.Mult): return a * b
        if isinstance(n.op, ast.Div):
            if b == 0: raise ZeroDivisionError
            return a / b
        if isinstance(n.op, ast.Pow):
            if b.denominator != 1 or abs(b) > 12: raise ValueError('pow')
            return a ** int(b)
        if isinstance(n.op, ast.Mod):
            if b == 0: raise ZeroDivisionError
            return a % b
    raise ValueError('node')


def fmt(v):
    if v.denominator == 1: return str(v.numerator)
    d = (Decimal(v.numerator) / Decimal(v.denominator)).quantize(Decimal(1).scaleb(-DEC), rounding=ROUND_HALF_UP)
    s = format(d, 'f').rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def calc_eval(expr):
    """Exact value of an arithmetic expression as text; 'ERR' if it is not one (or is out of range)."""
    if expr is None or len(expr) > MAX_EXPR: return 'ERR'
    e = re.sub(r'(?<=\d),(?=\d{3})', '', expr).replace('$', '').replace('×', '*').replace('x', '*').replace('^', '**')
    e = e.replace('%', '/100')
    if not re.fullmatch(r'[\d\s.+\-*/()]+', e): return 'ERR'
    try:
        v = _ev(ast.parse(e.strip(), mode='eval'))
    except Exception:
        return 'ERR'
    if abs(v) > 10 ** 18: return 'ERR'
    return fmt(v)


def calls_in(text):
    """Number of completed <calc>...</calc> in an answer's text (up to the scoring cut)."""
    return text.count(CLOSE)


if __name__ == '__main__':
    for e, want in [('48/2', '24'), ('6*60', '360'), ('10/3', '3.333333'), ('2,000*3', '6000'), ('(3+4)*2', '14'),
                    ('-5+2', '-3'), ('1/0', 'ERR'), ('abc', 'ERR'), ('12*0.5', '6'), ('2^3', '8'), ('50%*80', '40'),
                    ('0.1+0.2', '0.3'), ('2/3', '0.666667')]:
        got = calc_eval(e); assert got == want, (e, got, want)
    print('calc_eval ok')
