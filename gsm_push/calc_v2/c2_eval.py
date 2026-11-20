# exec'd inside tw_dump_lean.py after the --eval_ckpt weights are loaded (names: eng, C, tok, a, shot_q, torch).
# Greedy only. Sets: dev (600 GSM8K train split B, seed 0), dev2 (next 300), hard (300 scaled-number split-B items).
# NEVER GSM8K test.
import json as _j, os as _o, sys as _s, time as _t, random as _r
_s.path.insert(0, _o.environ['C2_DIR'])
from c2_engine import C2Engine
OUT = _o.environ['C2_OUT']; _o.makedirs(OUT, exist_ok=True)
SETS = _o.environ.get('C2_SETS', 'dev,dev2,hard').split(',')
VARS = _o.environ.get('C2_VARIANTS', 'off,ext,all').split(',')
NLIM = int(_o.environ.get('C2_N', 0))
eng.__class__ = C2Engine
eng.c2_setup()
pool_ = [r for r in C.load_train('B', shot_q=shot_q) if len(C.q_ids(tok, r['q'])) + a.max_new <= 544]
g_ = [r for r in pool_ if r['kind'] == 'gsm']; s_ = [r for r in pool_ if r['kind'] == 'scaled']
rr = _r.Random(0); rr.shuffle(g_); rr2 = _r.Random(0); rr2.shuffle(s_)
sets = {'dev': g_[:600], 'dev2': g_[600:900], 'hard': s_[:300]}
if not _o.path.exists(f'{OUT}/splits.json'):
    _j.dump({k: [dict(q=x['q'], gold=x['gold']) for x in v] for k, v in sets.items()}, open(f'{OUT}/splits.json', 'w'))
def mark(t, w):
    s = ''; on = 0
    for x, f in zip(t, w):
        c = 2 if f & 2 else (3 if f & 4 else (1 if f & 1 else 0))
        if c != on:
            s += {1: '⟧', 2: '»', 3: '›'}.get(on, '')
            s += {1: '⟦', 2: '«', 3: '‹'}.get(c, '')
            on = c
        s += tok.decode([x])
    s += {1: '⟧', 2: '»', 3: '›'}.get(on, '')
    return s.replace('<|endoftext|>', '')
summ = {}
if _o.path.exists(f'{OUT}/summary.json'): summ = _j.load(open(f'{OUT}/summary.json'))
for nm in SETS:
    items = sets[nm][:NLIM] if NLIM else sets[nm]
    qids = [C.q_ids(tok, r['q']) for r in items]
    base = None
    if _o.path.exists(f'{OUT}/{nm}_off.jsonl'): base = [_j.loads(l) for l in open(f'{OUT}/{nm}_off.jsonl')]
    for var in VARS:
        mode, ex, pol = (var.split(':') + ['1', 'look'][len(var.split(':')) - 1:])[:3]
        eng.set_c2(mode, exact=ex == '1', policy=pol)
        t0 = _t.time()
        o = eng.generate(qids, G=1, greedy=True, max_new=a.max_new)
        dt = _t.time() - t0
        rows = []
        for r, t, w, c2 in zip(items, o['tokens'], o['wflag'], o['c2']):
            sc = C.score(tok, t, r['gold'])
            rows.append(dict(q=r['q'], gold=r['gold'], ok=sc[0], fmt=sc[1], ntok=len(t), fires=sum(1 for v in c2['events'] if v.get('act') in ('preempt', 'rewrite', 'rewrite_restate')),
                             vetoes=c2['vetoes'], events=c2['events'], tokens=t, wflag=w, txt=mark(t, w)))
        with open(f'{OUT}/{nm}_{var.replace(":", "_")}.jsonl', 'w') as f:
            for x in rows: f.write(_j.dumps(x) + '\n')
        n = len(rows); acc = sum(x['ok'] for x in rows) / n
        rec = dict(n=n, rollback_tok=o.get('rollback_tok', 0), acts=dict(__import__('collections').Counter(v.get('act') for x in rows for v in x['events'])), acc=round(100 * acc, 2), sec=round(dt, 1), ntok=sum(x['ntok'] for x in rows),
                   maxmem=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        if mode == 'off': base = rows
        elif base is not None:
            fired = [k for k, x in enumerate(rows) if x['fires'] > 0]
            silent = [k for k, x in enumerate(rows) if x['fires'] == 0]
            ident = sum(1 for k in silent if rows[k]['tokens'] == base[k]['tokens'])
            chg = [k for k in range(n) if rows[k]['ok'] != base[k]['ok']]
            rec.update(fire_rows=len(fired), fire_rate=round(100 * len(fired) / n, 2), events=sum(x['fires'] for x in rows),
                       vetoes=sum(x['vetoes'] for x in rows),
                       fixes=sum(1 for k in chg if rows[k]['ok']), breaks=sum(1 for k in chg if not rows[k]['ok']),
                       fixes_fired=sum(1 for k in chg if rows[k]['ok'] and k in set(fired)),
                       breaks_fired=sum(1 for k in chg if not rows[k]['ok'] and k in set(fired)),
                       silent_identical=f'{ident}/{len(silent)}',
                       extra_tok_total=sum(rows[k]['ntok'] - base[k]['ntok'] for k in range(n)),
                       delta=round(100 * (acc - sum(x['ok'] for x in base) / n), 2))
        summ[f'{nm}/{var}'] = rec
        print('C2', nm, var, _j.dumps(rec), flush=True)
        _j.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1)
print('C2DONE', flush=True)
