"""analysis of the c2 runs: (1) equation coverage / failure attribution on the 'off' traces, (2) fixes / breaks."""
import json, sys, os, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_traces import scan
from fractions import Fraction
D = sys.argv[1]
L = lambda f: [json.loads(l) for l in open(f)]
def cut(t): return t.split('\n\n')[0]
def pred_of(txt):
    m = re.search(r'####\s*(-?[\d,]*\.?\d+)', txt.replace('⟦', '').replace('⟧', '').replace('«', '').replace('»', '').replace('‹', '').replace('›', ''))
    return m.group(1).replace(',', '') if m else None
for nm in ('dev', 'dev2', 'hard'):
    f = f'{D}/{nm}_off.jsonl'
    if not os.path.exists(f): continue
    off = L(f); n = len(off); fails = [r for r in off if not r['ok']]
    C = collections.Counter(); W = collections.Counter(); FT = collections.Counter(); near = 0
    for r in off:
        evs = scan(cut(r['txt']))
        types = set()
        for e in evs:
            k = ('calc' if e['calc'] else 'host') + ':' + ('+'.join(e['tags']) or 'basic')
            C[k] += 1
            if e['ok'] is False:
                W[k] += 1
                if not e['calc']:
                    for t in (e['tags'] or ['basic']): types.add(t)
        if not r['ok']:
            for t in types: FT[t] += 1
            if types: FT['any'] += 1
            p = pred_of(cut(r['txt']))
            try:
                if p is not None and abs(float(p) - float(r['gold'])) <= 1e-3 * max(1, abs(float(r['gold']))): near += 1
            except ValueError: pass
    print(f'== {nm}: n={n} acc={100*sum(r["ok"] for r in off)/n:.2f} fails={len(fails)}  near-miss (|pred-gold| <= 1e-3 rel, decimals) {near}')
    for k, v in sorted(C.items(), key=lambda x: -x[1]): print(f'   {k:28s} {v:5d}  wrong {W[k]:3d}')
    print('   failures containing a wrong HOST equation, by type:', dict(FT))
    for var in ('ext', 'all', 'ext_0', 'all_0'):
        g = f'{D}/{nm}_{var}.jsonl'
        if not os.path.exists(g): continue
        v = L(g)
        fx = [k for k in range(n) if v[k]['ok'] and not off[k]['ok']]; br = [k for k in range(n) if off[k]['ok'] and not v[k]['ok']]
        fr = sum(1 for x in v if x['fires']); ev = sum(x['fires'] for x in v)
        tg = collections.Counter(t for x in v for e in x['events'] for t in (e['tags'] or ['silentcalc']))
        print(f'  [{var}] acc {100*sum(x["ok"] for x in v)/n:.2f} (off {100*sum(x["ok"] for x in off)/n:.2f})  fixes {len(fx)} breaks {len(br)}  fire rows {fr} ({100*fr/n:.1f}%) events {ev} tags {dict(tg)} vetoes {sum(x["vetoes"] for x in v)}')
        if os.environ.get('SHOW'):
            for lab, ks in (('FIX', fx), ('BREAK', br)):
                for k in ks:
                    print(f'   --{lab} #{k} gold {off[k]["gold"]}')
                    for e in v[k]['events']: print('      ev', e['tags'], e['expr'], '=', e['s'], '| host first tok', repr(e['host_tok']))
                    if os.environ.get('SHOW') == '2':
                        print('      OFF:', cut(off[k]['txt'])[-700:].replace('\n', ' | '))
                        print('      V2 :', cut(v[k]['txt'])[-700:].replace('\n', ' | '))
