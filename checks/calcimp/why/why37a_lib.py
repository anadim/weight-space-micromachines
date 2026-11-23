"""classifier helpers (EQ / ev / nops / scan / SL / SLC), copied unchanged from why37a.py for why38.py."""
import json, re, os, sys, collections
H = os.environ.get('TT_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../tooltext/out'))
SEEDS = [7, 11, 13]; TS = [0, 1, 2, 3, 4]
EQ = re.compile(r'([0-9][0-9.,\s+\-*/x×÷()]*[0-9)])\s*=\s*(?:<calc>[^<]*</calc>)?\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)')
def ev(lhs):
    e = lhs.replace(',', '').replace('×', '*').replace('x', '*').replace('÷', '/')
    if not re.fullmatch(r'[0-9.+\-*/() ]+', e) or not re.search(r'[+\-*/]', e.strip().lstrip('-')): return None
    if '**' in e: return None   # why38: guards eval against huge powers (why37a.py never met one)
    try: return eval(e, {'__builtins__': {}})
    except Exception: return None
def nops(lhs):
    return len(re.findall(r'(?<=[0-9) ])\s*[+\-*/x×÷]\s*(?=[0-9(])', lhs))
def scan(text):
    t = re.sub(r'<calc>([^<]*)</calc>', lambda m: '<calc>' + m.group(1) + '</calc>', text)
    slips = 0; neq = 0; multi = 0
    for m in EQ.finditer(t):
        v = ev(m.group(1));
        if v is None: continue
        neq += 1
        if nops(m.group(1)) > 1: multi += 1
        try:
            got = float(m.group(2).replace(',', '')); v = float(v)
        except OverflowError: continue
        if abs(got - v) > max(1e-6, 1e-4 * abs(v)) and abs(round(v, 2) - got) > 0.011: slips += 1; SL[nops(m.group(1)) > 1] += 1; SLC['calc' in m.group(0)] += 1
    return neq, slips, multi
SL = collections.Counter(); SLC = collections.Counter()
