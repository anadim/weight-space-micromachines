"""calculator v2: exact evaluation of the arithmetic expression the host writes before '=' (the forms calc_s4 misses).

f(text before '=') -> exact value of the longest arithmetic expression that ends there:
  numbers   123 | 1,234,567 (comma groups of 3) | 12.5 | .5 | optional '$' prefix
  operators + - * / x X × ÷   with standard precedence (* / before + -), left associative
  brackets  ( ... ) nested, unary minus at the start / after '(' / after an operator
Exact rational arithmetic (Fraction). The result is printed exactly when it has <= 6 decimals, else rounded half-up to
6 decimals (and the exact rational is remembered behind that printed literal, see Row.exact_lit).

Row: the per-row token-level state machine the engine runs between decode steps (pure python, model-free):
  IDLE   --(host token ends with '=' [+ spaces/$], parse ok and fire rule)-->  ARMED
  ARMED  --(host token starts a number: digit / '-')--> replace it by the exact value's tokens  --> FORCING
         --(spaces / '$')--> ARMED (wait, <= 3 tokens)        --(anything else)--> IDLE
  FORCING  host tokens replaced by the queued value tokens; then VETO
  VETO   the host's next token may not start with a digit (masked argmax over non-digit tokens = the engine's alt);
         after '.' or ',' one more step of veto.
Fire rule (mode):
  'ext'   fire only on forms calc_s4 does not cover: >= 4 operands, brackets, comma-grouped numbers, leading minus,
          or an operand that is a rounded literal the implants printed earlier (exact value substituted).
  'all'   'ext' + any parsed expression the calc_s4 implant is silent on (decided at the number start from its wflag).
  'verify' (look policy only) every parsed expression: the number written after '=' (by the host or by calc_s4) is
          compared with the exact value and rewritten on a mismatch (calc_s4 itself is never wrong when it writes all
          the digits, but it writes only the '-' of a negative result and the host writes the digits).
Policy 'look' (default): the tracker does not pre-empt; it lets the number after '=' be written, then
  restated expression ('= 60 + 30', '= 171 / 3')  -> stands down (the next '=' gets checked again)
  value equal to the exact one (|diff| <= 1e-6)   -> nothing
  different value                                 -> the engine rewinds the row (KV slot, output position, calc_s4
                                                     state) to the number start and writes the exact value.
Policy 'preempt' (first version): writes the exact value as soon as the host starts a number.
"""
import re
from fractions import Fraction

DEC = 6
OPCH = {'+': '+', '-': '-', '*': '*', 'x': '*', 'X': '*', '×': '*', '/': '/', '÷': '/'}
NUM_RE = re.compile(r'\$?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)')


class ParseError(Exception):
    pass


def lex(s):
    """s -> list of (kind, value, text) or raise. kinds: num, op, lp, rp."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace(): i += 1; continue
        if c == '(': out.append(('lp', None, c)); i += 1; continue
        if c == ')': out.append(('rp', None, c)); i += 1; continue
        if c in OPCH:
            # 'x' / 'X' only as an operator when not glued to letters (e.g. 'xyz', 'box')
            if c in 'xX':     # 'x' is an operator only between spaces ('3 x 4') or between digits ('3x4')
                prev = s[i - 1] if i > 0 else ' '; nxt = s[i + 1] if i + 1 < n else ' '
                if not ((prev.isspace() and nxt.isspace()) or (prev.isdigit() and nxt.isdigit())): raise ParseError('x-word')
            out.append(('op', OPCH[c], c)); i += 1; continue
        m = NUM_RE.match(s, i)
        if m:
            t = m.group(0)
            j = m.end()
            if j < n and (s[j].isalnum() or s[j] in '.,%_'):     # 12abc, 1.2.3, 1,23, 12% -> not a plain number
                if not (s[j] in '.,' and (j + 1 >= n or not s[j + 1].isdigit())):
                    raise ParseError('num-tail')
            out.append(('num', t, t)); i = j; continue
        raise ParseError('char %r' % c)
    return out


class Expr:
    __slots__ = ('val', 'nops', 'paren', 'comma', 'lead_minus', 'unary', 'ops', 'nums')

    def __init__(self):
        self.nops = 0; self.paren = False; self.comma = False; self.lead_minus = False; self.unary = False
        self.ops = 0; self.nums = []


def parse(toks, exact_lit=None):
    """recursive descent over the token list (must consume all). Returns Expr with exact val (Fraction)."""
    E = Expr(); pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else (None, None, None)

    def number(t):
        raw = t.lstrip('$')
        if ',' in raw: E.comma = True
        v = Fraction(raw.replace(',', ''))
        E.nums.append(raw.replace(',', ''))
        if exact_lit and raw.replace(',', '') in exact_lit:
            v = exact_lit[raw.replace(',', '')]
        E.nops += 1
        return v

    def atom(first):
        k, v, _ = peek()
        if k == 'op' and v == '-':
            pos[0] += 1
            E.unary = True
            if first: E.lead_minus = True
            return -atom(False)
        if k == 'num':
            pos[0] += 1; return number(v)
        if k == 'lp':
            pos[0] += 1; E.paren = True
            r = expr(True)
            if peek()[0] != 'rp': raise ParseError('no )')
            pos[0] += 1
            return r
        raise ParseError('atom')

    def term(first):
        r = atom(first)
        while True:
            k, v, _ = peek()
            if k == 'op' and v in '*/':
                pos[0] += 1; E.ops += 1
                b = atom(False)
                if v == '*': r = r * b
                else:
                    if b == 0: raise ParseError('div0')
                    r = r / b
            else:
                return r

    def expr(first):
        r = term(first)
        while True:
            k, v, _ = peek()
            if k == 'op' and v in '+-':
                pos[0] += 1; E.ops += 1
                b = term(False)
                r = r + b if v == '+' else r - b
            else:
                return r

    E.val = expr(True)
    if pos[0] != len(toks): raise ParseError('trailing')
    if E.ops == 0: raise ParseError('no operator')
    return E


def fmt(v):
    """exact value -> text: integer, exact decimal (<= DEC places), or rounded half-up to DEC places."""
    neg = v < 0; a = -v if neg else v
    if a.denominator == 1:
        s = str(a.numerator)
    else:
        q = a * 10 ** DEC
        num, den = q.numerator, q.denominator
        r = (2 * num + den) // (2 * den)              # half-up
        s = str(r).rjust(DEC + 1, '0')
        s = s[:-DEC] + '.' + s[-DEC:]
        s = s.rstrip('0').rstrip('.')
    if s == '0': neg = False
    return ('-' if neg else '') + s


def trunc(v):
    """calc_s4's print of a quotient: truncated (toward 0) to DEC decimals, trailing zeros stripped."""
    neg = v < 0; a = -v if neg else v
    q = a * 10 ** DEC
    r = q.numerator // q.denominator
    s = str(r).rjust(DEC + 1, '0'); s = (s[:-DEC] + '.' + s[-DEC:]).rstrip('0').rstrip('.')
    return ('-' if neg and s != '0' else '') + s


def find_expr(line, exact_lit=None):
    """longest suffix of `line` (text before '=') that is a full arithmetic expression with a clean left boundary.
    Returns (Expr, start) or None. The left boundary: the char before the expression is not part of a number, and the
    previous non-space char is not a digit, ')', '.', '%', '$' or an operator (else the expression would continue to
    the left into something that is not arithmetic, e.g. '3J - 8 =' or '5 hours x 4 + 2 =')."""
    s = line.rstrip()
    for i in range(len(s)):
        c = s[i]
        if not (c.isdigit() or c in '$(-.'): continue
        if i > 0 and (s[i - 1].isalnum() or s[i - 1] in '.,$_'): continue
        j = i - 1
        while j >= 0 and s[j].isspace(): j -= 1
        if j >= 0:
            q = s[j]
            if q.isdigit() or q in ')]%$+-*/\u00d7\u00f7^': continue
            if q in ',.' and j >= 1 and s[j - 1].isdigit(): continue
            if q in 'xX' and (j == 0 or not s[j - 1].isalpha()): continue
        try:
            E = parse(lex(s[i:]), exact_lit)
        except (ParseError, ValueError, ZeroDivisionError):
            continue
        return E, i
    return None


def ltr_value(expr):
    """strict left-to-right value of a flat expression (no brackets, no unary minus); None otherwise."""
    try: toks = lex(expr)
    except ParseError: return None
    if not toks or toks[0][0] != 'num' or any(k in ('lp', 'rp') for k, _, _ in toks): return None
    v = Fraction(toks[0][1].lstrip('$').replace(',', ''))
    i = 1
    while i < len(toks):
        if toks[i][0] != 'op' or i + 1 >= len(toks) or toks[i + 1][0] != 'num': return None
        x = Fraction(toks[i + 1][1].lstrip('$').replace(',', '')); op = toks[i][1]
        if op == '/' and x == 0: return None
        v = v + x if op == '+' else v - x if op == '-' else v * x if op == '*' else v / x
        i += 2
    return v


def classify(E, exact_used=False):
    tags = []
    if E.nops >= 4: tags.append('chain')
    if E.paren: tags.append('paren')
    if E.comma: tags.append('comma')
    if E.lead_minus: tags.append('sign')
    if exact_used: tags.append('exactlit')
    return tags


# ---------------------------------------------------------------------------------------------- token state machine
IDLE, ARMED, FORCING, VETO, NUM = 0, 1, 2, 3, 4


def eq_trigger(t):
    """token text t: does it END an equation's left side? ('=' followed only by spaces / '$'; not '==', '<=' ...)"""
    k = t.rfind('=')
    if k < 0: return False
    if t[k + 1:].strip(' $') != '': return False
    if k > 0 and t[k - 1] in '=!<>:+-*/|&%^~': return False
    return True


OPSET = {'x', 'X', '*', '+', '-', '/', '\u00d7', '\u00f7', '=', '(', '^'}


def is_vetoed(t):
    """tokens the host may not write right after the exact value: a digit (would extend the number) or an operator /
    '=' / '(' (would continue the arithmetic, e.g. a restated '20 x 0.80' after the value)."""
    return t[:1].isdigit() or op_start(t)


def op_start(t):
    s = t.strip()
    return s in ('x', 'X') or (s[:1] in '+-*/\u00d7\u00f7=^(' and s != '')


def num_start(t):
    s = t.lstrip(' $')
    if not s: return None          # spaces / '$' only: keep waiting
    if s[0].isdigit(): return True
    if s in ('-',) : return True
    return False


class Row:
    """per-row tracker. feed(tok_text, wflag) after every booked token; returns an action for the engine."""

    def __init__(self, mode='ext', exact=True, policy='look'):
        self.mode = mode; self.exact = exact; self.policy = policy
        self.check_restate = mode in ('verify_rs', 'ext_rs', 'verify_rsl')
        self.ltr_ok = mode == 'verify_rsl'
        if mode in ('verify_rs', 'verify_rsl'): self.mode = 'verify'
        if mode == 'ext_rs': self.mode = 'ext'
        self.text = ''; self.st = IDLE; self.q = []; self.wait = 0; self.pend = None
        self.exact_lit = {}           # printed rounded literal -> exact Fraction (for later expressions)
        self.events = []; self.vetoes = 0

    def _line(self, upto):
        t = self.text[:upto]
        return t[t.rfind('\n') + 1:]

    def arm_on(self, t_before_eq):
        """t_before_eq: full row text up to (excluding) the trigger '='. Decide ARMED / IDLE."""
        line = self._line(len(t_before_eq))
        r = find_expr(line, self.exact_lit if self.exact else None)
        if r is None: return
        E, i = r
        used = self.exact and any(x in self.exact_lit for x in E.nums)
        tags = classify(E, used)
        if used:      # does substituting the exact value change the printed result?
            r0 = find_expr(line, None)
            if r0 is not None and fmt(r0[0].val) == fmt(E.val): tags.remove('exactlit')
        self.pend = dict(expr=line[i:].strip(), val=E.val, s=fmt(E.val), tags=tags, nops=E.nops)
        self.st = ARMED; self.wait = 0

    def feed(self, t, wflag, tokenize):
        self.tok_ = tokenize
        """t: text of the token just booked by the host; wflag: calc_s4 write flag of that token.
        Returns None (keep), ('replace', [ids...]) -> the engine replaces THIS token by ids[0] and queues ids[1:],
        ('veto',) -> the engine replaces this token by its masked (non-digit) alternative (then calls fix_last)."""
        if self.st == FORCING:      # the engine already replaced this token; text appended by the engine (fix_last)
            return None
        if self.st == VETO:
            if is_vetoed(t):
                self.vetoes += 1
                return ('veto',)
            self.text += t
            self.st = VETO if t in ('.', ',') and self.vwait == 0 else IDLE
            self.vwait = getattr(self, 'vwait', 0) + 1
            if self.st == IDLE: self._idle_check(t)
            return None
        if self.st == NUM:
            if t[:1].isdigit() and len(self.num) < 60:
                self.num += t; self.punct = False; self.text += t; return None
            if t in ('.', ',') and not self.punct and self.num[-1:].isdigit():
                self.num += t; self.punct = True; self.text += t; return None
            num = self.num.rstrip('.,').replace(',', '')
            p = self.pend; self.st = IDLE
            restate = op_start(t) and not self.punct
            try: hv = Fraction(num)
            except (ValueError, ZeroDivisionError): hv = None
            ev = dict(p, val=str(p['val']), host=num, host_tok=t, restate=restate)
            if restate or hv is None:
                ev['act'] = 'restate' if restate else 'nonum'; self.events.append(ev)
                if restate and self.check_restate:
                    self.rs = dict(p=p, snap_text=self.snap_text, lead=self.lead, ev=ev)
                self.text += t
                return self._idle_restate(t)
            if abs(hv - p['val']) <= Fraction(1, 10 ** 6):       # absolute: a truncated / rounded 6th decimal agrees
                ev['act'] = 'agree'; self.events.append(ev)
                self.text += t; self._idle_check(t); return None
            if self.ltr_ok:          # the host's value is the left-to-right reading of a flat expression
                lv = ltr_value(p['expr'])  # ('170 + 150 + 160 / 3 = 160' means the average): its grouping, not a slip
                if lv is not None and lv != p['val'] and abs(hv - lv) <= Fraction(1, 10 ** 6):
                    ev['act'] = 'agree_ltr'; self.events.append(ev)
                    self.text += t; self._idle_check(t); return None
            ev['act'] = 'rewrite'; self.events.append(ev)
            if self.exact and p['val'].denominator != 1 and fmt(p['val']) != str(p['val']):
                self.exact_lit[p['s'].lstrip('-')] = abs(p['val'])
            self.text = self.snap_text
            ids = tokenize(self.lead + p['s'])
            self.st = FORCING; self.q = list(ids)
            return ('rollback', ids)
        if self.st == ARMED:
            ns = num_start(t)
            if ns is None and self.wait < 3:
                self.text += t; self.wait += 1; return None
            if ns:
                p = self.pend
                fire = bool(p['tags']) or (self.mode == 'all' and not (wflag & 1)) or (self.mode == 'verify' and self.policy == 'look')
                if not fire and self.exact and (wflag & 1) and p['val'].denominator != 1:
                    tr = trunc(p['val'])           # calc_s4 prints the quotient truncated to 6 decimals
                    if Fraction(tr) != p['val']: self.exact_lit[tr.lstrip('-')] = abs(p['val'])
                if fire and self.policy == 'look':
                    # let the host write its own number; decide when it ends (restated expression -> stand down,
                    # equal value -> nothing, different value -> engine rolls back to here and writes the exact value)
                    self.lead = t[:len(t) - len(t.lstrip(' $'))]
                    self.snap_text = self.text
                    self.num = t.lstrip(' $'); self.punct = False
                    self.text += t; self.st = NUM
                    return ('snap',)
                if fire:
                    lead = t[:len(t) - len(t.lstrip(' $'))]
                    ids = tokenize(lead + p['s'])
                    ev = dict(p, val=str(p['val']), at=len(self.text), host_tok=t, calc_active=bool(wflag & 1), act='preempt')
                    self.events.append(ev)
                    if self.exact and p['val'].denominator != 1 and fmt(p['val']) != str(p['val']):
                        # printed a rounded value: remember the exact value behind the literal
                        self.exact_lit[p['s'].lstrip('-')] = abs(p['val'])
                    self.st = FORCING; self.q = list(ids)
                    return ('replace', ids)
            self.st = IDLE
        # IDLE
        self.text += t
        return self._idle_restate(t)

    def _idle_restate(self, t):
        """IDLE step with the restatement check: after '= <restated expression> =' the restated expression must have
        the exact value of the first one; if not, the engine rewinds to the first number start and writes that value."""
        rs = getattr(self, 'rs', None)
        if rs is not None:
            if '\n' in t: self.rs = None
            elif eq_trigger(t):
                self.rs = None
                k = len(self.text) - len(t) + t.rfind('=')
                seg = self.text[len(rs['snap_text']):k].strip()
                try:
                    E = parse(lex(seg), self.exact_lit if self.exact else None)
                    v = E.val
                except (ParseError, ValueError, ZeroDivisionError):
                    v = None
                if v is not None and abs(v - rs['p']['val']) > Fraction(1, 10 ** 6):
                    p = rs['p']
                    self.events.append(dict(p, val=str(p['val']), host=seg, host_tok=t, act='rewrite_restate'))
                    if self.exact and p['val'].denominator != 1 and fmt(p['val']) != str(p['val']):
                        self.exact_lit[p['s'].lstrip('-')] = abs(p['val'])
                    self.text = rs['snap_text']
                    ids = self.tok_(rs['lead'] + p['s'])
                    self.st = FORCING; self.q = list(ids)
                    return ('rollback', ids)
        self._idle_check(t)
        return None

    def _idle_check(self, t):
        if eq_trigger(t):
            k = len(self.text) - len(t) + t.rfind('=')
            self.arm_on(self.text[:k])

    def forced(self, t, last):
        """engine wrote token text t (from the queue); last: queue empty after it."""
        self.text += t
        if last: self.st = VETO; self.vwait = 0

    def vetoed(self, t):
        """engine replaced a digit by the masked alternative t."""
        self.text += t
        self.st = VETO if t in ('.', ',') and self.vwait == 0 else IDLE
        self.vwait += 1
        if self.st == IDLE: self._idle_check(t)

    def note_calc_written(self, t):
        pass
