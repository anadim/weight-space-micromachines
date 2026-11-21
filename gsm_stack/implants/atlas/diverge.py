"""Right-vs-wrong divergence atlas on MIXED items (lane 1, todo 'divergence').

usage: python3 atlas/diverge.py [--sets dev dev2]  -> atlas/data/diverge_{set}.jsonl.gz (one line per attempt of a mixed
item, then one line per item with "rec": "item"), printed summary tables (atlas/report.py renders notes/atlas.md).

Per wrong attempt:
 1. semantic divergence: steps = computed values ('... = v', new stated numbers, '#### v') from steps.extract; right-value
    set R = values computed by >= thr right attempts of the item (thr = 1 if n_right <= 4 else 2), canonical = by >= 1/2;
    question numbers Q are always consistent.  First divergent step D = first step whose value is not in Q u R.
 2. heuristic category of D (see classify()).
 3. token branch point vs the right attempt (same visible context group: k0 / wait / rep) with the longest common token
    prefix; sampled logprob of both branch tokens (identical context => logp of each is under the same distribution).
"""
import os, sys, re, json, gzip, math, argparse, collections, itertools
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__)); WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from parse import Tok, correct, EOS_TXT          # noqa: E402
from steps import extract, question_numbers, key, UNIT_CONST, NUM_RE   # noqa: E402

CATS = ['operand', 'relation', 'dropped', 'unit', 'target', 'plan', 'arithmetic', 'early_stop', 'format', 'other']
CAT_NAME = {'operand': 'wrong operand copied', 'relation': 'wrong relation / direction', 'dropped': 'dropped quantity',
            'unit': 'unit / world knowledge', 'target': 'wrong question target', 'plan': 'plan (other decomposition)',
            'arithmetic': 'arithmetic', 'early_stop': 'early stop / truncation / no answer', 'format': 'rounding / format',
            'other': 'other'}
BANDS = [(1, 8), (9, 16), (17, 24), (25, 31)]
UNIT_STRONG = {7, 12, 24, 30, 31, 52, 60, 365, 1000, 16, 32, 3600, 1440, 5280, 36, 28, 2000}
UNIT_WORD = re.compile(r'\s*(?:days?|weeks?|months?|years?|hours?|minutes?|seconds?|fingers?|toes?|inches|feet|foot|yards?|miles?|ounces?|quarts?|gallons?|cups?|pints?|pounds?|grams?|kilograms?|meters?|cm|weekdays?|school days|sides?|legs?|wheels?|players?|people|persons?|members?)\b', re.I)


def band(n): return next(f'{a}-{b}' for a, b in BANDS if a <= n <= b)


def rd(fn):
    with gzip.open(fn, 'rt') as f:
        return [json.loads(l) for l in f]


def group_of(a):
    if a['phrase_len'] == 0: return 'k0'
    return 'rep' if a['nrep'] > 0 else f"wait{a['phrase_len']}"


def tok_type(s):
    t = s.strip()
    if s in ('\n',) or (s and not t and '\n' in s): return 'newline'
    if not t: return 'space'
    if t.isdigit(): return 'digit'
    if t in ('+', '-', '*', '/', 'x', '×', '÷', '=', '(', ')', '%'): return 'operator'
    if t in ('$', '.', ',', ':', ';', '?', '!') or not any(c.isalnum() for c in t): return 'punct'
    return 'word'


class Att:
    """attempt + its parsed steps with token offsets"""
    def __init__(self, a, tk, qn):
        self.a = a; ids = a['seg_tokens']
        pieces = ['' if t == tk.eos else tk.ts(t) for t in ids]
        self.text = ''.join(pieces)
        self.c2t = []
        for i, p in enumerate(pieces): self.c2t += [i] * len(p)
        self.n = len(ids); self.plen = a['phrase_len']
        st = extract(self.text, qn)
        st = [s for s in st if not (s['kind'] == 'stated' and _enum(s, self.text))]
        lines = self.text.split('\n'); ls = []; p = 0
        for l in lines: ls.append((p, p + len(l))); p += len(l) + 1
        for s in st:
            s['tok'] = self.c2t[min(s['pos'], len(self.c2t) - 1)] if self.c2t else 0
            li = next(i for i, (x, y) in enumerate(ls) if x <= s['pos'] <= y)
            s['li'] = li; s['line_tok'] = self.c2t[min(ls[li][0], len(self.c2t) - 1)] if self.c2t else 0
            s['rel'] = (s['tok'] - self.plen) / max(1, self.n - self.plen)
        self.steps = st
        self.keys = [s['key'] for s in st]
        self.kset = set(self.keys)
        self.used = set()                                   # values used as operands or mentioned anywhere
        for m in NUM_RE.finditer(self.text):
            try:
                v = Fraction(m.group(0).lstrip('$').rstrip('%').replace(',', '')); self.used.add(key(v))
                if m.group(0).endswith('%'): self.used.add(key(v / 100))
            except Exception: pass
        for s in st: self.used.update(key(o) for o in s['operands'])
        self.ok = a['ok']; self.ans = a['ans']
        self.anskey = key(Fraction(a['ans'])) if a['ans'] not in (None, '') else None


def _enum(s, text):
    """list enumerations ('1. ', 'Step 2:') are not quantities"""
    pre = text[max(0, s['pos'] - 6):s['pos']]; post = text[s['pos']:s['pos'] + 6]
    return bool(re.match(r'\d+[.)]\s', post) and (s['pos'] == 0 or text[s['pos'] - 1] == '\n' or pre.rstrip(' ').endswith('\n'))) \
        or bool(re.search(r'(?i)step\s*$', pre))


def close_digits(o, q):
    """o is a plausible mis-copy of question number q: one digit changed/inserted/dropped, transposed, x10 / /10"""
    if o == q: return False
    if q != 0 and (o == q * 10 or o * 10 == q): return True
    a, b = f'{o:g}', f'{q:g}'
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        if len(d) == 1 and len(a) >= 2: return True
        if len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]: return True
    if abs(len(a) - len(b)) == 1 and min(len(a), len(b)) >= 2:
        L, S = (a, b) if len(a) > len(b) else (b, a)
        return any(L[:i] + L[i + 1:] == S for i in range(len(L)))
    return False


def _ev(vals, ops):
    """left-assoc with precedence via python eval on Fractions"""
    try:
        e = ''.join(f'Fraction({Fraction(v).limit_denominator(10 ** 6).numerator},{Fraction(v).limit_denominator(10 ** 6).denominator})'
                    + (ops[i] if i < len(ops) else '') for i, v in enumerate(vals))
        return eval(e, {'Fraction': Fraction})
    except Exception:
        return None


def _close_val(o, q):
    return close_digits(float(o), float(q))


def match_right(D, ctx):
    """right eq step that best corresponds to the divergent step: max shared operands (multiset), value a right value"""
    wo = collections.Counter(key(o) for o in D['operands'])
    best = None
    for (ro, rops, rv), c in ctx['Rsteps'].items():
        if rv == D['key']: continue
        rc = collections.Counter(ro); sh = sum((wo & rc).values())
        if sh == 0: continue
        nontriv = any(k not in (1, 2, 100, 0.5) for k in (wo & rc).elements())
        if not nontriv and not (sh >= len(ro) - 1 and len(ro) >= 2 and sh >= len(wo) - 1): continue
        sc = (sh - 0.5 * (sum((wo - rc).values()) + sum((rc - wo).values())), c)
        if best is None or sc > best[0]: best = (sc, ro, rops, rv)
    return best


def classify(w, D, ctx):
    """-> (cat, sub, detail).  ctx: Q, Rs (strong right keys), Rc (counter), gold_key, used_r (question numbers used by
    >= 1/2 right attempts), canon, Rsteps (right eq steps: (operand keys, ops, value) -> count)"""
    Q, Rs, gk, canon = ctx['Q'], ctx['Rs'], ctx['gold_key'], ctx['canon']
    good = lambda k: k is not None and (k in Rs or k == gk)
    if w.ans is None:
        return 'early_stop', 'truncated' if not w.a['complete'] else 'no_answer', ''
    unused = sorted(k for k in ctx['used_r'] if k not in w.used)
    if D is None:                                            # every step consistent, answer is a right value
        if gk in w.kset: return 'target', 'overshoot', 'computes gold then continues'
        g = ctx['gold']; X = Fraction(w.anskey).limit_denominator(10 ** 6) if w.anskey is not None else None
        if ctx['cents'] and g is not None and X and (X == g * 100 or X * 100 == g): return 'format', 'x100', 'dollars/cents'
        if g is not None and X:
            for q in sorted({k for k in Q if k} | {k for k in UNIT_CONST}):
                qf = Fraction(q).limit_denominator(10 ** 6)
                if qf not in (0, 1) and (X * qf == g or X / qf == g):
                    return 'target', 'missing_last_scale', f'gold = {float(X):g} x/ {float(qf):g}'
        if ctx['cents'] and g is not None and X and (X == g * 100 or X * 100 == g): return 'format', 'x100', 'dollars/cents'
        if unused: return 'dropped', 'unused_q', f'unused {unused}'
        return 'dropped', 'missing_branch', 'answer = right partial value'
    v = D['val']; vk = D['key']
    if D['kind'] == 'ans':
        prev = [s for s in w.steps if s['kind'] in ('eq', 'stated')]
        if prev and prev[-1]['key'] != vk and good(prev[-1]['key']):
            return 'other', 'ans_ne_last', 'answer differs from last computed value'
    if D.get('arith_ok') is False:
        return 'arithmetic', 'restate' if D['kind'] == 'restate' else ('ev_right' if good(key(D['ev_val'])) else 'ev_novel'), \
            f"{D['expr']} = {float(v):g} (exact {float(D['ev_val']):g})"
    if D['kind'] == 'ans' and gk is not None:
        g = ctx['gold']
        if abs(v - g) < 1 and (math.floor(g) == v or math.ceil(g) == v or math.floor(v) == g or math.ceil(v) == g):
            return 'format', 'rounding', f'{float(v):g} vs gold {float(g):g}'
    if ctx['cents'] and (good(key(v * 100)) or good(key(v / 100))):
        return 'format', 'x100', f'{float(v):g} dollars/cents'
    opnds = [Fraction(o).limit_denominator(10 ** 6) for o in D['operands']]; ops = D['ops']
    for m_ in NUM_RE.finditer(D['line']):                  # a novel number written with a unit word: world fact / count
        try: x = Fraction(m_.group(0).lstrip('$').rstrip('%').replace(',', ''))
        except Exception: continue
        if key(x) not in Q and key(x) not in ctx['used_any_r'] and UNIT_WORD.match(D['line'], m_.end()) \
                and ((D['kind'] == 'stated' and x == v) or (x in opnds and x != v)) and x.denominator == 1 and 1 < x <= 3600:
            return 'unit', 'world_fact', f"{m_.group(0)}{UNIT_WORD.match(D['line'], m_.end()).group(0)}"
    for o in (opnds if opnds else ([v] if D['kind'] == 'stated' else [])):
        if key(o) in Q or key(o) in Rs: continue
        for q in ctx['Qv']:
            if _close_val(o, q): return 'operand', 'miscopy', f'{float(o):g} for {float(q):g}'
    if v.denominator == 1 and 1 <= abs(v) <= 20 and vk not in Q and (key(v + 1) in canon or key(v - 1) in canon) \
            and D['kind'] == 'stated':
        return 'unit', 'count_off_by_one', f'{float(v):g} vs right {float(v + 1) if key(v + 1) in canon else float(v - 1):g}'
    prior = set(s['key'] for s in w.steps if s['tok'] < D['tok'])
    uc = lambda x: x in UNIT_STRONG and x not in Q and x not in Rs and x not in prior and x not in ctx['Rsv']
    if opnds:
        m = match_right(D, ctx)
        if m is not None:
            _, ro, rops, rv = m
            wc = collections.Counter(key(o) for o in opnds); rc = collections.Counter(ro)
            miss, extra = list((rc - wc).elements()), list((wc - rc).elements())
            det = f"{D['expr']} = {float(v):g}  vs right {' '.join(map(lambda x: f'{x:g}', ro))} [{''.join(rops)}] = {rv:g}"
            if not miss and not extra: return 'relation', 'op_or_order', det
            if miss and not extra:
                if any(uc(x) for x in miss): return 'unit', 'missing_factor', det
                return 'dropped', 'missing_operand', det
            if extra and not miss: return 'relation', 'extra_term', det
            if len(miss) == 1 and len(extra) == 1:
                a, b = extra[0], miss[0]
                if uc(a) or uc(b): return 'unit', 'factor', det
                if a in Q and b in Q: return 'operand', 'wrong_q_number', det
                return 'relation', 'wrong_ref', det
            if any(uc(x) for x in miss + extra): return 'unit', 'factor', det
            return 'relation', 'restructured', det
        if len(ops) == len(opnds) - 1:
            for i, op in enumerate(ops):
                for a2 in ('+', '-', '*', '/'):
                    if a2 == op: continue
                    r = _ev(opnds, ops[:i] + [a2] + ops[i + 1:])
                    if r is not None and key(r) in canon: return 'relation', f'op{op}->{a2}', f"{D['expr']}: {op}->{a2}"
            if len(opnds) == 2 and ops[0] in ('-', '/'):
                r = _ev(opnds[::-1], ops)
                if r is not None and key(r) in canon: return 'relation', 'swap', f"{D['expr']}: operands swapped"
            for i, o in enumerate(opnds):
                cands = [(1 - o, 'complement'), (1 + o, 'pct_base')] if 0 < o < 1 else []
                for c, nm in cands:
                    r = _ev(opnds[:i] + [c] + opnds[i + 1:], ops)
                    if r is not None and key(r) in canon: return 'relation', nm, f"{D['expr']}: {float(o):g}->{float(c):g}"
        for o in opnds:
            if o in UNIT_STRONG and key(o) not in Q and key(o) not in ctx['used_any_r']:
                return 'unit', 'const', f'uses {float(o):g}'
    if D['kind'] in ('eq', 'ans'):
        for q in set(Q) | prior:
            if q is None or q == 0: continue
            qf = Fraction(q).limit_denominator(10 ** 6)
            for r in (v + qf, v - qf, v * qf, qf - v):
                if (key(r) in canon or key(r) == gk) and key(r) != q:
                    return 'dropped', 'missing_term', f'{float(v):g} ~ {float(r):g} via {q:g}'
    if unused: return 'dropped', 'unused_q', f'unused {unused}'
    if D['kind'] == 'stated': return 'plan', 'invented_quantity', f'states {float(v):g}'
    after = [s for s in w.steps if s['tok'] >= D['tok'] and s['kind'] in ('eq', 'stated')]
    share = sum(good(s['key']) for s in after) / len(after) if after else 0
    if share < 0.3: return 'plan', 'novel_path', f'{share:.2f} of later values are right values'
    return 'other', 'unexplained', f'{share:.2f}'


def lcp(x, y):
    n = min(len(x), len(y)); i = 0
    while i < n and x[i] == y[i]: i += 1
    return i


def analyse_item(it, atts, tk):
    Q = question_numbers(it['q']); Qv = sorted(q for q in Q if q is not None)
    W = [Att(a, tk, Q) for a in atts]
    right = [w for w in W if w.ok]; wrong = [w for w in W if not w.ok]
    nr = len(right); thr = 1 if nr <= 4 else 2
    Rc = collections.Counter(k for w in right for k in w.kset)
    try: gold = Fraction(str(it['gold']).replace(',', ''))
    except Exception: gold = None
    gk = key(gold)
    Rs = {k for k, c in Rc.items() if c >= thr}
    canon = {k for k, c in Rc.items() if c >= nr / 2}
    qd = set()                                              # question numbers written as digits
    for m in NUM_RE.finditer(it['q']):
        v = Fraction(m.group(0).lstrip('$').rstrip('%').replace(',', '')); qd.add(key(v))
    used_r = {q for q in qd if sum(q in w.used for w in right) >= nr / 2}
    used_any_r = {k for w in right for k in w.used}
    Rcomp = collections.Counter(k for w in right for k in {s['key'] for s in w.steps if s['kind'] in ('eq', 'restate', 'ans')})
    Rsteps = collections.Counter()
    for w in right:
        for st in {(tuple(sorted(key(o) for o in s['operands'])), tuple(s['ops']), s['key']) for s in w.steps
                   if s['kind'] in ('eq', 'restate') and s['ops'] and s['key'] in Rs}:
            Rsteps[st] += 1
    ctx = dict(Q=Q, Qv=Qv, Rs=Rs, Rc=Rc, gold_key=gk, gold=gold, used_r=used_r, used_any_r=used_any_r, canon=canon,
               Rsteps=Rsteps, Rsv={k for k in Rc}, cents='cent' in it['q'].lower())

    def cons_step(s):
        k = s['key']
        if k == gk or k in Rs and (k not in Q or Rcomp[k] >= thr): return True
        if k in Q:   # a question number re-derived by a non-trivial expression right attempts never compute is divergent
            return not (s['kind'] in ('eq', 'restate') and s['ops'] and all(key(o) != k for o in s['operands']))
        return False
    cons = lambda k: k in Q or k in Rs or k == gk
    out = []
    by_g = collections.defaultdict(list)
    for w in right: by_g[group_of(w.a)].append(w)
    for w in W:
        a = w.a
        rec = dict(rec='att', set=a['set'], idx=a['idx'], seed=a['seed'], k=a['k'], gid=a['gid'], ok=a['ok'], ans=a['ans'],
                   voted=a['voted'], group=group_of(a), ntok=w.n, plen=w.plen, nrep=a['nrep'], lg=a['lg'],
                   nsteps=len(w.steps), frac_right_vals=round(sum(cons(s['key']) for s in w.steps) / max(1, len(w.steps)), 3),
                   n_arith_err=sum(s.get('arith_ok') is False for s in w.steps))
        if not w.ok:
            D = next((s for s in w.steps if not cons_step(s)), None)
            cat, sub, det = classify(w, D, ctx)
            rec.update(cat=cat, sub=sub, detail=det)
            if D is not None:
                rec.update(div_kind=D['kind'], div_si=w.steps.index(D), div_tok=D['tok'], div_rel=round(D['rel'], 3),
                           div_line=D['line'].strip(), div_expr=D['expr'], div_val=float(D['val']), div_line_tok=D['line_tok'])
            else:
                A = next((s for s in w.steps if s['kind'] == 'ans'), None)
                if A is not None:
                    rec.update(div_kind='none', div_si=w.steps.index(A), div_tok=A['tok'], div_rel=round(A['rel'], 3),
                               div_line=A['line'].strip(), div_line_tok=A['line_tok'])
            # token branch: right attempt in the same context group with the longest common prefix
            g = group_of(a); best = None
            for r in by_g.get(g, []):
                L = lcp(a['seg_tokens'], r.a['seg_tokens'])
                if best is None or L > best[0] or (L == best[0] and r.a['seg_logp'][min(L, r.n - 1)] > best[1].a['seg_logp'][min(best[0], best[1].n - 1)]):
                    best = (L, r)
            if best is not None and best[0] < min(w.n, best[1].n):
                L, r = best
                rt = [x for x in by_g[g] if lcp(a['seg_tokens'], x.a['seg_tokens']) == L]
                rtoks = collections.Counter(x.a['seg_tokens'][L] for x in rt)
                pre_d = max((abs(p - q) for p, q in zip(a['seg_logp'][1:L], r.a['seg_logp'][1:L])), default=0.0)  # [0] scores the forced phrase start given the hidden previous attempt
                ws, rs = tk.ts(a['seg_tokens'][L]), tk.ts(r.a['seg_tokens'][L])
                bt = dict(partner=r.a['gid'], b_tok=L, b_rel=round((L - w.plen) / max(1, w.n - w.plen), 3),
                          b_wtok=ws, b_rtok=rs, b_type=tok_type(ws), b_rtype=tok_type(rs),
                          lp_w=a['seg_logp'][L], lp_r=r.a['seg_logp'][L], b_calc=bool(a['seg_wf'][L] & 1),
                          b_rcalc=bool(r.a['seg_wf'][L] & 1), n_partners=len(rt), n_rtoks=len(rtoks),
                          prefix_maxdlp=round(pre_d, 4),
                          b_line=w.text[:sum(len(tk.ts(t)) for t in a['seg_tokens'][:L])].split('\n')[-1][-80:])
                # left context of the branch token (to the line start) and the two continuations to the line end
                cw = ''.join(tk.ts(t) for t in a['seg_tokens'][L:L + 40]).split('\n')[0]
                cr = ''.join(tk.ts(t) for t in r.a['seg_tokens'][L:L + 40]).split('\n')[0]
                bt.update(b_wcont=cw[:80], b_rcont=cr[:80])
                if 'div_tok' in rec:
                    dt, dl = rec['div_tok'], rec['div_line_tok']
                    bt['b_vs_div'] = 'after' if L > dt else 'in_line' if L >= dl else 'before'
                    bt['b_lines_before'] = w.text[:sum(len(tk.ts(t)) for t in a['seg_tokens'][:L])].count('\n')
                    bt['div_lines'] = w.text[:w.steps[rec['div_si']]['pos']].count('\n')
                rec.update(bt)
            # semantic partner (for the corresponding right step): branch partner, else max value overlap
            sp = best[1] if best is not None else max(right, key=lambda r: len(r.kset & w.kset) - 0.01 * abs(r.n - w.n))
            rec['sem_partner'] = sp.a['gid']
            if D is not None:
                c = [s['key'] for s in w.steps if s['tok'] < D['tok'] and cons(s['key'])]
                rs_ = [s for s in sp.steps if s['kind'] in ('eq', 'stated', 'ans')]
                j = 0
                if c:
                    for i, s in enumerate(rs_):
                        if s['key'] == c[-1]: j = i + 1
                nxt = next((s for s in rs_[j:] if s['key'] not in w.kset or s['kind'] == 'ans'), rs_[-1] if rs_ else None)
                if nxt is not None: rec['right_line'] = nxt['line'].strip()
        out.append(rec)
    # item level
    wa = collections.Counter(w.ans for w in wrong)
    att, natt = wa.most_common(1)[0]
    try: attk = key(Fraction(att)) if att not in (None, '') else None
    except Exception: attk = None
    atype = ('none' if att is None else 'right_canonical_intermediate' if attk in canon else
             'right_intermediate' if attk in Rs else 'right_rare_value' if attk in Rc else
             'question_number' if attk in Q else 'novel')
    cats = collections.Counter(r['cat'] for r in out if not r['ok'])
    att_cats = collections.Counter(r['cat'] for r in out if not r['ok'] and r['ans'] == att)
    item = dict(rec='item', set=it['set'], idx=it['idx'], n_right=it['n_right'], n_att=it['n_att'], band=band(it['n_right']),
                gold=it['gold'], n_wrong=len(wrong), n_distinct_wrong=len(wa), attractor=att, attractor_n=natt,
                attractor_share_wrong=round(natt / len(wrong), 3), attractor_share_all=round(natt / len(W), 3),
                attractor_type=atype, attractor_cat=att_cats.most_common(1)[0][0],
                dominant_cat=cats.most_common(1)[0][0], cats=dict(cats), wrong_answers=dict(wa.most_common(6)),
                vote_ok=it['vote_ok'], vote_ans=it['vote_ans'], n_right_seed=it['n_right_seed'], greedy_ok=it.get('greedy_ok'),
                n_canon=len(canon), n_Q=len(Q))
    return out, item


def run(set_, tk, outd):
    A = rd(f'{HERE}/data/attempts_{set_}.jsonl.gz'); I = rd(f'{HERE}/data/items_{set_}.jsonl.gz')
    by = collections.defaultdict(list)
    for a in A: by[a['idx']].append(a)
    rows = []; items = []
    for it in I:
        if it['class'] != 'mixed': continue
        r, item = analyse_item(it, by[it['idx']], tk)
        rows += r; items.append(item)
    with gzip.open(f'{outd}/diverge_{set_}.jsonl.gz', 'wt') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
        for r in items: f.write(json.dumps(r) + '\n')
    return rows, items


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sets', nargs='+', default=['dev', 'dev2'])
    a = ap.parse_args(); tk = Tok()
    for s in a.sets:
        rows, items = run(s, tk, f'{HERE}/data')
        wr = [r for r in rows if not r['ok']]
        print(s, 'items', len(items), 'wrong attempts', len(wr))
        print(' attempt cats', collections.Counter(r['cat'] for r in wr).most_common())
        print(' item dominant', collections.Counter(r['dominant_cat'] for r in items).most_common())
        print(' subs', collections.Counter((r['cat'], r['sub']) for r in wr).most_common(30))


if __name__ == '__main__':
    main()
