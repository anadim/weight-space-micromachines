"""Find every explicit 'a op b (op c ...) = c' in a GSM-style trace and recompute it exactly.
Units between numbers are allowed ("6 hours * 60 minutes/hour = 360 minutes"); '%' or unbalanced parens -> unparsed."""
import re
from fractions import Fraction

NUM = r'\d[\d,]*(?:\.\d+)?'
TOK = re.compile(rf'(?P<num>{NUM})|(?P<op>(?<![A-Za-z])[+\-*/×÷x](?![A-Za-z]))|(?P<par>[()])|(?P<pct>%)|(?P<word>[A-Za-z][A-Za-z/\'-]*)|(?P<other>\S)')
RHS = re.compile(rf'\s*\$?\s*(?P<neg>-)?\s*\$?\s*(?P<c>{NUM})')
DELIM = re.compile(r'[=\n:;]|\.\s|,\s|\bis\b|\bare\b|\bso\b|\bthen\b', re.I)
OPMAP = {'x': '*', '×': '*', '÷': '/'}


def fnum(s):
    return Fraction(s.replace(',', ''))


def _eval(toks):
    """toks: list of ('n', Fraction) / ('o', op) / ('p', '(' or ')'). Standard precedence. None on failure."""
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else (None, None)

    def atom():
        k, v = peek()
        if k == 'n': pos[0] += 1; return v
        if k == 'p' and v == '(':
            pos[0] += 1; r = expr()
            if peek() != ('p', ')'): raise ValueError
            pos[0] += 1; return r
        raise ValueError

    def term():
        r = atom()
        while peek()[0] == 'o' and peek()[1] in '*/':
            op = toks[pos[0]][1]; pos[0] += 1; b = atom()
            r = r * b if op == '*' else r / b
        return r

    def expr():
        r = term()
        while peek()[0] == 'o' and peek()[1] in '+-':
            op = toks[pos[0]][1]; pos[0] += 1; b = term()
            r = r + b if op == '+' else r - b
        return r
    try:
        r = expr()
        return r if pos[0] == len(toks) else None
    except (ValueError, ZeroDivisionError):
        return None


def equations(text):
    """-> list of dict(lhs, ops, operands, claimed, exact, ok, eq_pos (index of '='), rhs_end)."""
    out = []
    for m in re.finditer('=', text):
        e = m.start()
        # segment: back to previous delimiter
        seg_start = 0
        for d in DELIM.finditer(text, 0, e):
            seg_start = d.end()
        seg = text[seg_start:e]
        r = RHS.match(text, e + 1)
        if not r: continue
        toks, bad = [], False
        for t in TOK.finditer(seg):
            k = t.lastgroup; v = t.group()
            if k == 'num': toks.append(('n', fnum(v)))
            elif k == 'op':
                if v == 'x' and not (toks and toks[-1][0] in 'np'): continue
                toks.append(('o', OPMAP.get(v, v)))
            elif k == 'par': toks.append(('p', v))
            elif k == 'pct': bad = True
            # words / '$' / others are skipped (units)
        # keep the trailing alternating run num (op num)* (with parens) ending at '='
        while toks and toks[0][0] == 'o': toks = toks[1:]
        # collapse adjacent numbers (unit words in between like "3 bags 5") -> keep last number run start
        run = []
        for t in toks:
            if run and t[0] == 'n' and run[-1][0] == 'n': run = [t]   # two numbers without op: restart
            else: run.append(t)
        toks = run
        nops = sum(1 for t in toks if t[0] == 'o')
        if nops == 0 or bad: continue
        if toks[-1][0] == 'o': continue
        if toks.count(('p', '(')) != toks.count(('p', ')')):
            toks = [t for t in toks if t[0] != 'p']
        v = _eval(toks)
        if v is None: continue
        c = fnum(r.group('c')) * (-1 if r.group('neg') else 1)
        ok = c == v or abs(float(c) - float(v)) <= max(0.0051, 1e-6 * abs(float(v)))   # allow 2-dp rounding
        operands = [t[1] for t in toks if t[0] == 'n']
        out.append(dict(ops=[t[1] for t in toks if t[0] == 'o'], operands=operands, claimed=c, exact=v, ok=bool(ok),
                        eq_pos=e, rhs_end=r.end(), lhs=text[seg_start:e].strip(), rhs=r.group('c')))
    return out


if __name__ == '__main__':
    for s in ["First: 6 hours * 60 minutes/hour = 360 minutes\nThen 360 minutes / 90 minutes/break = 4 breaks",
              "Twice is 20 x 2 = 40.", "So $5 + $3 = $9", "(3 + 4) * 2 = 14", "10% of 50 = 5", "1,200 - 300 = 900",
              "She spends 1/3 * 12 = 4", "2.5 * 4 = 10.0"]:
        print(s.replace('\n', ' | '), '->', [(e['lhs'], e['claimed'], e['exact'], e['ok']) for e in equations(s)])
