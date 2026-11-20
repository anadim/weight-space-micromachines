"""offline: scan dumped greedy traces (marked text, implant spans in ⟦⟧) for equations calc v2 would fire on."""
import json, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc2 import find_expr, classify, fmt, eq_trigger
from fractions import Fraction
def scan(txt):
    out = []
    raw = txt
    for m in re.finditer('=', raw):
        k = m.start()
        if k > 0 and raw[k-1] in '=!<>:+-*/|&%^~': continue
        if k + 1 < len(raw) and raw[k+1] == '=': continue
        before = raw[:k].replace('⟦', '').replace('⟧', '')
        line = before[before.rfind('\n') + 1:]
        r = find_expr(line)
        if r is None: continue
        E, i = r
        after = raw[k+1:k+40]
        calc = after.lstrip(' $')[:1] == '⟦' or after[:1] == '⟦'
        a2 = after.replace('⟦', '').replace('⟧', '')
        mm = re.match(r'\s*\$?\s*(-?[\d,]*\.?\d+)', a2)
        if mm and re.match(r'\s*[-+*/x\u00d7\u00f7(]', a2[mm.end():]): continue      # restated expression, not a result
        host = mm.group(1).replace(',', '') if mm else None
        ok = None
        if host is not None:
            try: ok = abs(Fraction(host) - E.val) <= Fraction(1, 10**6) * max(1, abs(E.val))
            except Exception: ok = None
        tags = classify(E)
        if E.val.denominator != 1 or any('.' in x for x in E.nums): tags = tags + ['decimal']
        out.append(dict(expr=line[i:].strip(), val=fmt(E.val), tags=tags, calc=calc, host=host, ok=ok))
    return out
if __name__ == '__main__':
    for fn in sys.argv[1:]:
        rows = [json.loads(l) for l in open(fn)]
        C = collections.Counter(); W = collections.Counter(); F = collections.Counter(); ex = []
        for r in rows:
            g = r['greedy']; txt = g['txt'].split('\n\n')[0]
            evs = scan(txt)
            bad_ext = False
            for e in evs:
                key = ('calc' if e['calc'] else 'host') + ':' + ('+'.join(e['tags']) or 'basic')
                C[key] += 1
                if e['ok'] is False: W[key] += 1
                if e['ok'] is False and not e['calc'] and e['tags']: bad_ext = True; ex.append((e, g['ok']))
            if bad_ext: F['ok' if g['ok'] else 'fail'] += 1
        print(fn, len(rows), 'items')
        for k, v in sorted(C.items(), key=lambda x: -x[1]): print(f'  {k:30s} {v:5d} wrong {W[k]}')
        print('  items with a wrong host ext equation:', dict(F))
        for e, ok in ex[:12]: print('   ', ok, e['expr'], '=', e['host'], '| exact', e['val'])
