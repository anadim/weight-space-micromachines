"""Per-attempt features for the voter study (all computable inside the generation stream; see notes/voter.md).

usage: python3.11 voter/feats.py --sets dev dev2 [trainC]  -> voter/cache/att_{set}.pkl
Each attempt record: set, idx, seed, k, gid, gold, ok, val (register value or None), valid, bad, lg, nrep, committed,
plus text features (calc re-check, question coverage, intermediate-value set, answer shape) and logp features.
"""
import os, re, sys, gzip, json, math, pickle, argparse
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'atlas'))
from parse import NUM, correct          # scorer regex / C.correct, verbatim from gen/wiring/common.py

WAIT = "Wait, let me solve it again.\n"
NUMRE = re.compile(r'\d+(?:\.\d+)?')
ICAP = 16                               # intermediate register capacity per attempt


def norm_num(s):
    s = s.replace(',', '')
    try: f = Fraction(s)
    except (ValueError, ZeroDivisionError): return None
    return f


def text_numbers(t):
    t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', t)
    return [Fraction(m.group(0)) for m in NUMRE.finditer(t)]


# ---------------------------------------------------------------- exact calculator re-check (Fractions)
ARITH = r'[\d\.\s\+\-\*/x×÷\(\)\$%]'
SUFFIX = re.compile(ARITH + r'+$')
PREFIX = re.compile(r'^' + ARITH + r'+')
TOK = re.compile(r'\s*(\d+(?:\.\d+)?%?|[\+\-\*/\(\)])')


def _tokens(e):
    e = e.replace('$', '').replace('×', '*').replace('÷', '/')
    e = re.sub(r'(?<=[\d\)])\s*x\s*(?=[\d\(])', '*', e)
    if re.search(r'[a-wyzA-Z]', e) or 'x' in e: return None
    e = e.strip().rstrip('.').strip()
    out, p = [], 0
    while p < len(e):
        m = TOK.match(e, p)
        if not m: return None
        out.append(m.group(1)); p = m.end()
        if p < len(e) and not e[p:].strip(): break
    return out


def _parse(ts):
    """recursive descent: expr = term (+|- term)*, term = fac (*|/ fac)*, fac = num | (expr).  -> Fraction or None"""
    pos = [0]

    def peek(): return ts[pos[0]] if pos[0] < len(ts) else None

    def fac():
        t = peek()
        if t is None: raise ValueError
        if t == '(':
            pos[0] += 1; v = expr()
            if peek() != ')': raise ValueError
            pos[0] += 1; return v
        if t[0].isdigit():
            pos[0] += 1
            return Fraction(t[:-1]) / 100 if t.endswith('%') else Fraction(t)
        raise ValueError

    def term():
        v = fac()
        while peek() in ('*', '/'):
            op = ts[pos[0]]; pos[0] += 1; w = fac()
            v = v * w if op == '*' else v / w
        return v

    def expr():
        v = term()
        while peek() in ('+', '-'):
            op = ts[pos[0]]; pos[0] += 1; w = term()
            v = v + w if op == '+' else v - w
        return v
    try:
        v = expr()
        return v if pos[0] == len(ts) else None
    except (ValueError, ZeroDivisionError):
        return None


def _shown_ok(lhs, rhs_txt, rhs):
    """lhs (exact) vs the value written on the right; rounding to the written number of decimals is accepted."""
    if lhs == rhs: return True
    d = len(rhs_txt.split('.')[1]) if '.' in rhs_txt else 0
    if d == 0: return False                       # an integer written for a non-integer / different value
    return abs(lhs - rhs) <= Fraction(1, 2 * 10 ** d) + Fraction(1, 10 ** 9)


def calc_check(text):
    """-> (n_checked, n_wrong) over every 'expr = value' link (chains 'a = b = c' checked link by link)."""
    n = bad = 0
    t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', text)
    for line in t.split('\n'):
        parts = line.split('=')
        for a, b in zip(parts[:-1], parts[1:]):
            m = SUFFIX.search(a)
            if not m: continue
            le = m.group(0)
            if not re.search(r'\d', le): continue
            lt = _tokens(le)
            if not lt or lt[0] in '+-*/)' or not any(x in '+-*/' for x in lt): continue
            lv = _parse(lt)
            if lv is None: continue
            mb = PREFIX.match(b)
            if not mb: continue
            rt = _tokens(mb.group(0))
            if not rt or rt[0] in '+-*/)': continue
            rv = _parse(rt) if len(rt) > 1 else (Fraction(rt[0][:-1]) / 100 if rt[0].endswith('%') else Fraction(rt[0]))
            if rv is None:
                if not rt[0][0].isdigit(): continue
                rt = rt[:1]; rv = Fraction(rt[0].rstrip('%'))
            n += 1
            rtxt = rt[0].rstrip('%') if len(rt) == 1 else ''
            ok = _shown_ok(lv, rtxt, rv) if len(rt) == 1 else (lv == rv)
            if not ok and len(rt) == 1 and rt[0].endswith('%'): ok = _shown_ok(lv * 100, rtxt, Fraction(rtxt))
            bad += not ok
    return n, bad


# ---------------------------------------------------------------- per attempt
def body_text(a):
    t = a['text']
    if t.lstrip('\n').startswith(WAIT): t = t.lstrip('\n')[len(WAIT):]
    m = re.search(NUM, t)
    return (t[:m.start()] if m else t), t


def attempt_features(a, qnums):
    body, full = body_text(a)
    nums = text_numbers(body)
    eqv = [norm_num(m.group(1)) for m in re.finditer(r'=\s*\$?\s*(\d[\d,]*(?:\.\d+)?)', body)]
    eqv = [x for x in eqv if x is not None]
    qs = set(qnums)
    inter, seen = [], set()
    for x in nums:                                     # non-question numbers, in order of appearance, capped register
        if x not in qs and x not in seen:
            seen.add(x); inter.append(x)
    inter = inter[:ICAP]
    eqset = []
    for x in eqv:
        if x not in eqset: eqset.append(x)
    eqset = eqset[:ICAP]
    numset = set(nums)
    cov = (sum(1 for x in qs if x in numset) / len(qs)) if qs else 1.0
    nchk, nwrong = calc_check(body)
    lp, wf = a['seg_logp'], a['seg_wf']
    host = [l for l, w in zip(lp, wf) if w == 0]
    ho = a.get('hash_off')
    ans_lp = [l for l, w in list(zip(lp, wf))[ho + 1:] if w == 0] if ho is not None else []
    val = a['reg_val'] if a['reg_ok'] else None
    fv = Fraction(val).limit_denominator(10 ** 6) if val is not None else None
    return dict(
        ntok=len(a['seg_tokens']), nsteps=len(a['steps']), nprev=len(a['prev_segments']),
        lp_mean=sum(host) / max(1, len(host)), lp_min=min(host) if host else 0.0,
        lp_n1=sum(1 for l in host if l < math.log(0.1)), lp_n5=sum(1 for l in host if l < math.log(0.5)),
        lp_ans=sum(ans_lp), nhost=len(host),
        calc_n=nchk, calc_bad=nwrong, qcov=cov, nq=len(qs),
        inter=inter, eqset=eqset,
        ans_isq=int(fv is not None and fv in qs),
        ans_int=int(fv is not None and fv.denominator == 1),
        ans_log=math.log10(abs(val) + 1) if val is not None else 0.0,
        ans_in_body=int(fv is not None and fv in numset),          # answer already computed in the text
    )


def load(sset):
    path = os.path.join(WS, 'atlas', 'data', f'attempts_{sset}.jsonl.gz')
    out = []
    for line in gzip.open(path):
        a = json.loads(line)
        for k in ('seg_tokens', 'seg_wf', 'seg_logp'):
            if isinstance(a[k], str): a[k] = json.loads(a[k])
        qnums = set(text_numbers(a['q']))
        r = dict(set=a['set'], idx=a['idx'], seed=a['seed'], k=a['k'], gid=a['gid'], gold=a['gold'], ok=a['ok'],
                 ans=a['ans'], lg=a['lg'], bad=bool(a['bad']), nrep=a['nrep'], voted=a['voted'],
                 committed=bool(a['complete']) and a['reg_ok'] is not None,
                 valid=bool(a['reg_ok']) if a['reg_ok'] is not None else False,
                 rbad=bool(a['reg_bad']) if a['reg_bad'] is not None else True,
                 rlg=a['lg'] if a['lg'] is not None else -30.0,
                 val=a['reg_val'] if a['reg_ok'] else None)
        r['vok'] = int(r['val'] is not None and correct(str(r['val']), a['gold']))
        r.update(attempt_features(a, qnums))
        out.append(r)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--sets', nargs='+', default=['dev', 'dev2'])
    A = ap.parse_args()
    for s in A.sets:
        R = load(s)
        pickle.dump(R, open(os.path.join(HERE, 'cache', f'att_{s}.pkl'), 'wb'))
        print(s, len(R), 'calc_bad>0:', sum(r['calc_bad'] > 0 for r in R), 'calc_n mean:',
              sum(r['calc_n'] for r in R) / len(R), 'vok==ok:', sum(r['vok'] == r['ok'] for r in R if r['val'] is not None),
              '/', sum(r['val'] is not None for r in R))
