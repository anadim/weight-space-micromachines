"""Step extraction from attempt texts: computed quantities ('... = v' results, stated new numbers), exact Fraction
re-evaluation of every '<expr> = v', question-number reader.  Pure python (no numpy).

Text is handled as the concatenation of per-token strings so that every char offset maps to a token offset.
"""
import re
from fractions import Fraction

NUM_RE = re.compile(r'\$?(?<![\w.])([0-9]{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(%?)', re.ASCII)
OPS = {'+': '+', '-': '-', '−': '-', '*': '*', 'x': '*', 'X': '*', '×': '*', '·': '*', '/': '/', '÷': '/'}
LEX = re.compile(r"\$?(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?|[+\-−*×·/÷()]|(?<![A-Za-z])[xX](?![A-Za-z])|[A-Za-z']+|[^\sA-Za-z\d]", re.ASCII)
WORDNUM = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
           'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
           'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
           'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 10 ** 6,
           'dozen': 12, 'dozens': 12, 'twice': 2, 'double': 2, 'doubles': 2, 'triple': 3, 'triples': 3, 'thrice': 3,
           'half': Fraction(1, 2), 'halves': 2, 'quarter': Fraction(1, 4), 'quarters': 4, 'third': Fraction(1, 3),
           'thirds': 3, 'fourth': Fraction(1, 4), 'fifth': Fraction(1, 5), 'fifths': 5, 'tenth': Fraction(1, 10),
           'pair': 2, 'pairs': 2, 'couple': 2, 'week': 7, 'weeks': 7, 'year': 12, 'hour': 60, 'minutes': 60}
# conventional constants (units / calendar / counting): a step operand in this set not in the question = world knowledge
UNIT_CONST = {7, 12, 24, 30, 31, 52, 60, 365, 360, 100, 1000, 16, 8, 3600, 4, 2, 10, 1440, 5280, 36, 3, 28, 48, 128, 2000}


def num_val(s):
    """'$1,200.50' -> Fraction; '50%' -> Fraction(50) (percent flag returned separately)"""
    s = s.lstrip('$'); pc = s.endswith('%'); s = s.rstrip('%').replace(',', '')
    return Fraction(s), pc


def key(v):
    """value key for set matching (2-decimal rounding: 3.333 == 3.33 == 10/3)"""
    if v is None: return None
    return round(float(v), 2) + 0.0


def question_numbers(q):
    """set of value keys stated in the question (digits, percents as p and p/100, number words, fractions a/b)"""
    out = set()
    for m in NUM_RE.finditer(q):
        v, pc = num_val(m.group(0)); out.add(key(v))
        if pc: out.add(key(v / 100))
    for m in re.finditer(r'(\d+)\s*/\s*(\d+)', q):
        if int(m.group(2)): out.add(key(Fraction(int(m.group(1)), int(m.group(2)))))
    for w in re.findall(r'[a-z]+', q.lower()):
        if w in WORDNUM: out.add(key(WORDNUM[w]))
        if w.endswith('s') and w[:-1] in WORDNUM: out.add(key(WORDNUM[w[:-1]]))
    return out


class _P:
    """recursive-descent evaluator over lexed items (numbers as Fraction, ops, parens); implicit '*' before '('"""
    def __init__(self, it): self.it = it; self.i = 0

    def peek(self): return self.it[self.i] if self.i < len(self.it) else None

    def take(self): self.i += 1; return self.it[self.i - 1]

    def expr(self):
        v = self.term()
        while self.peek() in ('+', '-'):
            o = self.take(); w = self.term(); v = v + w if o == '+' else v - w
        return v

    def term(self):
        v = self.fac()
        while self.peek() in ('*', '/') or self.peek() == '(':
            o = self.take() if self.peek() != '(' else '*'
            w = self.fac()
            if o == '/' and w == 0: raise ZeroDivisionError
            v = v * w if o == '*' else v / w
        return v

    def fac(self):
        t = self.take()
        if t == '-': return -self.fac()
        if t == '(':
            v = self.expr()
            if self.take() != ')': raise ValueError
            return v
        if isinstance(t, Fraction): return t
        raise ValueError(t)


def lex(s):
    """-> list of (kind, value, start, end, pct): kind in num/op/lp/rp/word/var/other.  'x' is an operator only between
    a number/')' and a number/'$'/'('; letters glued to digits/parens ('2B', '(1/5)X') or capital singletons are variables."""
    out = []
    for m in LEX.finditer(s):
        t = m.group(0); a, b = m.start(), m.end()
        if t[0] in '0123456789' or (t[0] == '$' and len(t) > 1):
            v, pc = num_val(t); out.append(['num', v, a, b, pc]); continue
        if t in ('x', 'X'):
            pv = s[:a].rstrip(); nx = s[b:].lstrip()
            if pv and (pv[-1] in '0123456789)%') and nx and (nx[0] in '0123456789$('):
                out.append(['op', '*', a, b, False]); continue
            out.append(['var', t, a, b, False]); continue
        if t in OPS: out.append(['op', OPS[t], a, b, False])
        elif t == '(': out.append(['lp', '(', a, b, False])
        elif t == ')': out.append(['rp', ')', a, b, False])
        elif t[0].isalpha():
            glued = (a > 0 and s[a - 1] in '0123456789)(') or (b < len(s) and s[b] in '0123456789(')
            pv = s[:a].rstrip(); nx = s[b:].lstrip()
            opn = (pv[-1:] in ('+', '-', '*', '/', '(')) or (nx[:1] in ('+', '-', '*', '/', '=', ')'))
            if glued or (len(t) == 1 and (t not in ('a', 'A', 'I') or opn)):
                out.append(['var', t, a, b, False])
            else: out.append(['word', t.lower(), a, b, False])
        elif t[0] == "'": out.append(['word', t, a, b, False])
        else: out.append(['other', t, a, b, False])
    return [tuple(x) for x in out]


def _build(seq):
    """lexed span -> (ev items, operands, ops); None if it contains two adjacent numbers"""
    ev = []; opnds = []; ops = []
    for x in seq:
        if x[0] == 'num':
            if ev and (isinstance(ev[-1], Fraction)): return None
            if ev and ev[-1] == ')': ev.append('*'); ops.append('*')
            ev.append(x[1] / 100 if x[4] else x[1]); opnds.append(x[1])
        elif x[0] == 'op':
            ev.append(x[1])
            if not (x[1] == '-' and (len(ev) == 1 or ev[-2] in ('(', '+', '-', '*', '/'))): ops.append(x[1])
        elif x[0] == 'lp':
            if ev and (isinstance(ev[-1], Fraction) or ev[-1] == ')'): ev.append('*'); ops.append('*')
            ev.append('(')
        else: ev.append(')')
    return ev, opnds, ops


def trailing_expr(items):
    """longest trailing arithmetic span of lexed items (unit words after a number are skipped; 'p% of y' = p/100*y).
    -> dict(ev, operands, ops, start, wordy, var) or None"""
    seq = []; i = len(items) - 1; wordy = False; var = False; tail = False
    while i >= 0:
        k, v, s, e, pc = items[i]
        if k == 'var': var = True; break
        if k == 'word':
            j = i
            while j >= 0 and items[j][0] == 'word': j -= 1
            words = [items[x][1] for x in range(j + 1, i + 1)]
            right = seq[0] if seq else None
            if j >= 0 and items[j][0] == 'num' and items[j][4] and words == ['of'] and right is not None and right[0] in ('num', 'lp'):
                seq.insert(0, ('op', '*', items[j + 1][2], items[i][3], False)); i = j; continue
            if j >= 0 and items[j][0] in ('num', 'rp') and right is not None and right[0] in ('op', 'rp') and len(words) <= 3 \
                    and not {'of', 'is', 'and', 'so', 'then', 'are', 'was', 'has', 'have'} & set(words):
                wordy = True; i = j; continue
            if j >= 0 and items[j][0] in ('num', 'rp') and right is None and len(words) <= 3 and 'of' not in words:
                tail = True; i = j; continue                # trailing unit words of the last number
            break
        if k == 'other':
            if v in ('.', ',', ':', ';', '?', '!') and not seq: i -= 1; continue
            if v == '$': i -= 1; continue
            break
        if k == 'op' and v == '-' and i > 0 and items[i - 1][0] in ('word', 'other') and seq and seq[0][0] == 'num' \
                and (i < 2 or items[i - 2][0] != 'num'):
            if e == seq[0][2]: seq.insert(0, items[i])      # '-2' glued: a sign
            break                                           # '- ' after a word is a dash / bullet, not minus
        seq.insert(0, items[i]); i -= 1
    while seq and (seq[0][0] in ('op', 'rp')) and not (seq[0][1] == '-' and len(seq) > 1 and seq[1][0] == 'num'
                                                      and (i < 0 or seq[0][3] == seq[1][2])): seq.pop(0)
    while seq and seq[-1][0] in ('op', 'lp'): seq.pop()
    depth = 0; ok = []
    for x in seq:
        if x[0] == 'lp': depth += 1
        if x[0] == 'rp':
            if depth == 0: continue
            depth -= 1
        ok.append(x)
    seq = ok
    while seq and seq[0][0] == 'lp' and sum(x[0] == 'lp' for x in seq) > sum(x[0] == 'rp' for x in seq): seq.pop(0)
    if not seq or not any(x[0] == 'num' for x in seq): return None
    b = _build(seq)
    if b is None: return None
    return dict(ev=b[0], operands=b[1], ops=b[2], start=seq[0][2], wordy=wordy or tail, var=var)


def leading_expr(items):
    """leading arithmetic span of a part after '=' -> dict(ev, operands, ops, start, last, pct, var) or None"""
    i = 0
    while i < len(items) and items[i][0] == 'other' and items[i][1] in ('$',): i += 1
    seq = []; last = None; var = False
    while i < len(items):
        k, v, s, e, pc = items[i]
        if k == 'var': var = True; break
        if k in ('num', 'op', 'lp', 'rp'):
            if k == 'op' and v == '-' and not seq and not (i + 1 < len(items) and items[i + 1][0] == 'num'): break
            if k == 'rp' and sum(x[0] == 'lp' for x in seq) <= sum(x[0] == 'rp' for x in seq): break
            if k == 'num' and seq and seq[-1][0] == 'num': break
            seq.append(items[i]); last = i; i += 1; continue
        if k == 'other' and v == '$' and seq and seq[-1][0] == 'op': i += 1; continue
        if k == 'word' and seq and seq[-1][0] in ('num', 'rp'):
            j = i
            while j < len(items) and items[j][0] == 'word': j += 1
            words = [items[x][1] for x in range(i, j)]
            if words == ['of'] and seq[-1][0] == 'num' and seq[-1][4] and j < len(items) and items[j][0] == 'num':
                seq.append(('op', '*', s, e, False)); i = j; continue
            if j < len(items) and items[j][0] == 'op' and len(words) <= 3 and j + 1 < len(items) and \
                    items[j + 1][0] in ('num', 'lp'):
                i = j; continue
        break
    while seq and seq[-1][0] in ('op', 'lp'): seq.pop()
    if not seq or not any(x[0] == 'num' for x in seq): return None
    b = _build(seq)
    if b is None: return None
    nums = [x for x in seq if x[0] == 'num']
    return dict(ev=b[0], operands=b[1], ops=b[2], start=seq[0][2], last=last, pct=len(nums) == 1 and nums[0][4], var=var,
                end=seq[-1][3])


def evaluate(ev):
    try:
        p = _P(ev); v = p.expr()
        if p.i != len(ev): return None
        return v
    except (ValueError, ZeroDivisionError, IndexError, TypeError):
        return None


def stated_ok(val, stated, dec):
    """is `stated` (written with `dec` decimals) an acceptable rendering (rounded or truncated) of exact `val`?"""
    if val == stated: return True
    return (dec > 0 and abs(val - stated) < Fraction(1, 10 ** dec)) or abs(val - stated) <= Fraction(1, 10 ** 5) * abs(val)


def extract(text, qnums):
    """-> list of steps (dict) in order.  kind: 'eq' (single value after '='), 'restate' (an '= expr' whose value differs
    from the expression before it), 'stated' (number in a line without '=' that is neither a question number nor seen
    before), 'ans' (the '#### v' line).  Each step: kind, val (Fraction), key, pos (char offset in text), line, expr (text of
    the expression right before the '='), ev_val (its exact value), operands, ops, arith_ok (None if not checkable)."""
    steps = []; seen = set(); pos0 = 0
    for line in text.split('\n'):
        lpos = pos0; pos0 += len(line) + 1
        if not line.strip(): continue
        if line.lstrip().startswith('####'):
            m = NUM_RE.search(line)
            if m:
                v, _ = num_val(m.group(0)); v = -v if line[:m.start()].rstrip().endswith('-') else v
                steps.append(dict(kind='ans', val=v, key=key(v), pos=lpos + m.start(), line=line, expr=None, ev_val=None,
                                  operands=[], ops=[], arith_ok=None))
            continue
        if 'Wait, let me solve it again' in line: continue
        ln = line
        st = len(ln) - len(ln.lstrip())
        if ln[st:st + 2] == '- ': ln = ln[:st] + '  ' + ln[st + 2:]          # list bullet
        parts = ln.split('=')
        if len(parts) > 1:
            off = 0; lhs = None
            for pi, part in enumerate(parts):
                items = lex(part)
                if pi > 0:
                    le = leading_expr(items)
                    if le is not None:
                        val = evaluate(le['ev'])
                        lv = evaluate(lhs['ev']) if (lhs and not lhs['var']) else None
                        if le['var']: lv = None
                        single = not le['ops']
                        raw = part[le['start']:le['end']].rstrip('%')
                        dec = len(raw.split('.', 1)[1]) if (single and '.' in raw) else 0
                        a_ok = None
                        if lv is not None and val is not None and lhs['ops']:
                            a_ok = stated_ok(lv, val, dec) or (le['pct'] and stated_ok(lv, val * 100, dec))
                        if single and val is not None and not le['var']:
                            steps.append(dict(kind='eq', val=val, key=key(val), pos=lpos + off + le['start'], line=line,
                                              expr=lhs['text'] if lhs else None, ev_val=lv,
                                              operands=[float(o) for o in lhs['operands']] if lhs else [],
                                              ops=lhs['ops'] if lhs else [], arith_ok=a_ok, wordy=bool(lhs and lhs['wordy'])))
                            seen.add(key(val))
                            if le['pct']: seen.add(key(val * 100))
                        elif a_ok is False and le['operands'] and lv is not None and stated_ok(lv, le['operands'][0], 0):
                            v0 = le['operands'][0]                  # '= 15 stations * 25 = 375': 15 is a result
                            steps.append(dict(kind='eq', val=v0, key=key(v0), pos=lpos + off + le['start'], line=line,
                                              expr=lhs['text'], ev_val=lv, operands=[float(o) for o in lhs['operands']],
                                              ops=lhs['ops'], arith_ok=True, wordy=bool(lhs['wordy'])))
                            seen.add(key(v0))
                        elif a_ok is False and not lhs['wordy']:
                            steps.append(dict(kind='restate', val=val, key=key(val), pos=lpos + off + le['start'], line=line,
                                              expr=lhs['text'], ev_val=lv, operands=[float(o) for o in lhs['operands']],
                                              ops=lhs['ops'], arith_ok=False, wordy=False))
                        rest = [x for x in items[le['last'] + 1:] if x[0] not in ('word', 'other')]
                        if le['ops'] and not rest and pi < len(parts) - 1:
                            lhs = dict(le, text=part.strip(), wordy=False); off += len(part) + 1; continue
                te = trailing_expr(items)
                lhs = dict(te, text=part[te['start']:].strip()) if te else None
                off += len(part) + 1
        else:
            for m in NUM_RE.finditer(line):
                v, pc = num_val(m.group(0)); k = key(v)
                if k in qnums or k in seen or (pc and key(v / 100) in qnums): continue
                seen.add(k)
                steps.append(dict(kind='stated', val=v, key=k, pos=lpos + m.start(), line=line, expr=None, ev_val=None,
                                  operands=[], ops=[], arith_ok=None))
    return steps
