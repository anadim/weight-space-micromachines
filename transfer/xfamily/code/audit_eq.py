"""Check every 'a op b = c' line segment in the outputs: is c the exact value? (implant writes the digits of c when it fires)"""
import json, re, sys
from decimal import Decimal, getcontext; getcontext().prec = 60
EQ = re.compile(r'(?<![\d.*/+x×\-])(?<![\d.*/+x×\-] )(?<![(])(-?\d+(?:\.\d+)?)\s*([+\-*/x×])\s*(-?\d+(?:\.\d+)?)\s*=\s*\$?(-?\d+(?:\.\d+)?)')
for f in sys.argv[1:]:
    d = json.load(open(f)); tot = ok = 0; runaway = 0; bad = []
    for r in d['rows']:
        for a, op, b, c in EQ.findall(r['out'].replace(',', '')):
            A, B, C = Decimal(a), Decimal(b), Decimal(c); tot += 1
            v = {'+': A + B, '-': A - B, '*': A * B, 'x': A * B, '×': A * B}.get(op)
            if op == '/': v = A / B if B else None
            good = v is not None and (C == v or (op == '/' and abs(C - v) < Decimal('1e-6')))
            ok += good
            if len(c.replace('-', '').replace('.', '')) > 20: runaway += 1
            if not good and len(bad) < 5: bad.append(f'{a}{op}{b}={c}')
    print(f"{f.split('/')[-1]}: equations {tot}, exact {ok} ({100*ok/max(tot,1):.1f}%), >20-digit results {runaway}; wrong e.g. {bad}")
