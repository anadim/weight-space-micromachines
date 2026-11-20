"""Exact text-level triggers and nudges for the reading/bookkeeping implants (shared by the pod hook and local analysis)."""
import re
NUMQ = re.compile(r'(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)')
ANS = re.compile(r'####\s*(-?[\d,]*\.?\d+)')


def fnum(s):
    try: return float(s.replace(',', ''))
    except ValueError: return None


def qnums(q):
    """digit-written numbers of the question with their character span."""
    out = []
    for m in NUMQ.finditer(q):
        v = fnum(m.group(1).rstrip('.,'))
        if v is not None: out.append((v, m.start(1), m.end(1)))
    return out


def body(txt):
    return txt.replace('<|endoftext|>', '').split('\n\n')[0].replace('⟦', '').replace('⟧', '')


def answer(txt):
    m = ANS.search(body(txt))
    return fnum(m.group(1)) if m else None


def unused(q, txt, minv=0):
    """question numbers (value >= minv) that never appear (as a value) in the answer text before '####'."""
    pre = body(txt).split('####')[0]
    used = {fnum(m.group(1).rstrip('.,')) for m in NUMQ.finditer(pre)}
    return [(v, s, e) for v, s, e in qnums(q) if v >= minv and v not in used]


def clause(q, s, e):
    """the clause of the question around the char span [s,e): between the nearest sentence/comma boundaries."""
    a = max(q.rfind(c, 0, s) for c in '.?!,;') + 1
    bs = [q.find(c, e) for c in '.?!,;']; bs = [b for b in bs if b >= 0 and not (q[b] in '.,' and b + 1 < len(q) and q[b + 1].isdigit())]
    b = min(bs) if bs else len(q)
    return q[a:b].strip()


def t_unused(q, txt, minv=0):
    return len(unused(q, txt, minv)) > 0


def t_san(txt):
    """answer-sanity class: missing / negative / non-integer final answer."""
    a = answer(txt)
    return a is None or a < 0 or a != int(a)


def t_notlast(txt):
    """the final answer is not the last value computed with '=' (an intermediate or a restated number)."""
    b = body(txt); pre = b.split('####')[0]; a = answer(txt)
    eq = [fnum(x) for x in re.findall(r'=\s*\$?(-?\d[\d,]*(?:\.\d+)?)', pre)]
    eq = [x for x in eq if x is not None]
    return a is not None and bool(eq) and a != eq[-1]


def nudge_unused(q, txt, style='clause'):
    u = unused(q, txt)
    if not u: return None
    v, s, e = u[0]
    if style == 'clause':
        return f'Wait, I did not use this part of the question: "{clause(q, s, e)}". Let me solve it again and use every quantity.\n'
    return f'Wait, I did not use the number {q[s:e]}. Let me solve it again and use every quantity.\n'


def ask_sentence(q):
    """the question's last sentence (what is asked)."""
    s = re.split(r'(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bMrs\.)(?<=[.?!])\s+(?=[A-Z0-9$])', q.strip())
    qs = [x for x in s if x.endswith('?')]
    return (qs[-1] if qs else s[-1]).strip()
