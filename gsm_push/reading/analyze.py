"""Local analysis of pod outputs: per variant acc on dev/dev2, fixes/breaks vs off, tokens; and conditional switches
(trigger computed on the off answer -> that row's answer is replaced by variant V's answer on the same row), which is
exactly what a restart-with-another-prompt implant would produce (pass 2 does not depend on pass 1's text)."""
import json, sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import rd_trig as RT
D = sys.argv[1]
sp0 = json.load(open(os.path.join(os.path.dirname(__file__), 'dev_split.json')))
tr = [json.loads(l) for l in open(os.path.join(os.environ.get('WSM_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')), 'data', 'gsm8k', 'train.jsonl'))]
Q = {i: tr[i]['question'] for i in range(len(tr))}
V = {os.path.basename(f)[:-6]: {(x['set'], x['i']): x for x in map(json.loads, open(f))} for f in sorted(glob.glob(f'{D}/*.jsonl'))}
off = V['off']
def txt(x): return x['txt']
TRIG = {'san': lambda k, x: RT.t_san(txt(x)), 'unused': lambda k, x: RT.t_unused(Q[k[1]], txt(x)),
        'unused3': lambda k, x: RT.t_unused(Q[k[1]], txt(x), 3), 'notlast': lambda k, x: RT.t_notlast(txt(x)), 'all': lambda k, x: True}
def line(name, rows, base, fired=None):
    out = []
    for s in ('dev', 'dev2'):
        ks = [k for k in base if k[0] == s and k in rows]
        if not ks: continue
        n = len(ks); acc = 100 * sum(rows[k]['ok'] for k in ks) / n; b = 100 * sum(base[k]['ok'] for k in ks) / n
        fx = sum(1 for k in ks if rows[k]['ok'] and not base[k]['ok']); br = sum(1 for k in ks if base[k]['ok'] and not rows[k]['ok'])
        tk = sum(rows[k]['ntok'] for k in ks) / sum(base[k]['ntok'] for k in ks) - 1
        fr = '' if fired is None else f' fire {100*sum(1 for k in ks if fired(k))/n:.1f}%'
        from math import comb
        m = fx + br; pv = min(1.0, 2 * sum(comb(m, j) for j in range(0, min(fx, br) + 1)) / 2 ** m) if m else 1.0
        out.append(f'{s} {acc:.2f} ({acc-b:+.2f}) fix {fx} brk {br} p={pv:.2f} tok {100*tk:+.1f}%{fr}')
    print(f'{name:28s} ' + ' | '.join(out))
print('off:', ' | '.join(f"{s} {100*sum(x['ok'] for k,x in off.items() if k[0]==s)/sum(1 for k in off if k[0]==s):.2f} (n={sum(1 for k in off if k[0]==s)})" for s in ('dev','dev2') if any(k[0]==s for k in off)))
for v, rows in V.items():
    if v == 'off': continue
    fired = (lambda k, rows=rows: rows[k].get('fire', 1)) if v.startswith('nudge') else None
    line(v, rows, off, fired)
print('--- conditional restart: trigger on off answer -> take variant answer (tokens = off + variant on fired rows)')
for tn, tf in TRIG.items():
    if tn == 'all': continue
    for v, rows in V.items():
        if v == 'off' or v.startswith('nudge'): continue
        sw = {}
        for k, x in off.items():
            if k in rows and tf(k, x):
                y = dict(rows[k]); y['ntok'] = x['ntok'] + y['ntok']; sw[k] = y
            else: sw[k] = x
        line(f'{tn}->{v}', sw, off, lambda k, tf=tf: tf(k, off[k]))
