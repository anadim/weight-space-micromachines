"""Per-seed IFEval-541 table for seeds35 (rows 2 / 2b / 4 per train seed) + run facts (GPU, wall, idle identity, dev curve).
python3 perseed.py  (reads ../res/ifeval/{P,I}_s{0..5}/: items jsons (u0 / seeds 0-2 via ifitems.py), eval summary, reduced steps.jsonl, run_facts.json)"""
import json, os, glob, sys
sys.dont_write_bytecode = True
from ifitems import rows as _rows   # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.environ.get('AO', os.path.join(HERE, '..', 'res', 'ifeval'))


def sc(p):                                   # p = <run dir>/<eval>.items.json
    rows = _rows(os.path.basename(os.path.dirname(p)), os.path.basename(p)[:-len('.items.json')])
    if rows is None: return None
    n = len(rows)
    ps = 100 * sum(all(r['strict']) for r in rows) / n
    ist = 100 * sum(sum(r['strict']) for r in rows) / sum(len(r['strict']) for r in rows)
    tq = sum(r['ntok'] or 0 for r in rows) / n
    fired = sum(1 for r in rows if (r.get('circ') or {}).get('fired'))
    return ps, ist, tq, fired


def facts(d):
    f = {}
    if not os.path.isdir(d) or not os.listdir(d): return {'gpu': 'not back yet'}
    s = os.path.join(d, 'eval_u300.summary.json')
    if os.path.exists(s):
        j = json.load(open(s)); i = j.get('idle') or {}
        f['idle'] = f"{i.get('identical_subset_run', '?')}/{i.get('silent_in_subset_run', '?')}" if i else '-'
    st = os.path.join(d, 'steps.jsonl')
    if os.path.exists(st):
        L = [json.loads(l) for l in open(st) if l.strip()]
        U = {x['u']: x for x in L if 'u' in x and 'roll_s' in x}
        f['u'] = max(U) if U else 0
        dev = {}
        for x in L:
            if 'dev' in x: dev[x['dev']['u']] = x['dev']['dev_prompt_strict']
        f['dev'] = ' '.join(f'{k}:{v:.1f}' for k, v in sorted(dev.items()))
        if U:
            f['mem'] = max(x.get('mem_gb', 0) for x in U.values())
            f['upd'] = sum(x['update_s'] for x in U.values()) / len(U)
    rf = json.load(open(os.path.join(d, 'run_facts.json')))
    f['gpu'] = 'A100' if 'A100' in rf['gpu'] else 'B200'
    for k in ('start', 'train_start', 'train_end', 'end'):
        if k in rf: f[k] = rf[k]
    return f


print('| seed | row 2 (P plain) ps / inst / tok | row 2b (P + implant at eval) ps / fired | row 4 (I imp) ps / inst / tok / fired | P: gpu, u, idle | I: gpu, u, idle |')
print('|---|---|---|---|---|---|')
for s in range(6):
    p, i = os.path.join(O, f'P_s{s}'), os.path.join(O, f'I_s{s}')
    r2, r2b, r4 = sc(f'{p}/eval_u300.plain.items.json'), sc(f'{p}/eval_u300.imp.items.json'), sc(f'{i}/eval_u300.imp.items.json')
    fp, fi = facts(p), facts(i)
    c = lambda r, k=3: ' / '.join(f'{v:.2f}' if j < 2 else f'{v:.0f}' for j, v in enumerate(r[:k])) if r else 'pending'
    print(f"| {s} | {c(r2)} | {f'{r2b[0]:.2f} / {r2b[3]}' if r2b else 'pending'} | {c(r4)}{f' / {r4[3]}' if r4 else ''} | "
          f"{fp.get('gpu')}, u{fp.get('u', '?')}, {fp.get('idle', '-')} | {fi.get('gpu')}, u{fi.get('u', '?')}, {fi.get('idle', '-')} |")
print()
for s in range(3, 6):
    for a in 'PI':
        f = facts(os.path.join(O, f'{a}_s{s}'))
        print(f"- {a}_s{s}: " + ', '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in f.items()))
