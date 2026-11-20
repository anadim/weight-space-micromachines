# exec'd inside dig_san/tw_dump_lean.py after --eval_ckpt is loaded. Compiled forward-only sanity implant on split B
# (dev 600, dev2 300, dev3 562). One generate() call per variant; greedy.
import json as _j, os as _o, sys as _s, time as _t, re as _re
_s.path.insert(0, _o.environ['RD_DIR'])
from comp_sanity import CompEngine
import rd_trig as RT
OUT = _o.environ['RD_OUT']; _o.makedirs(OUT, exist_ok=True)
VARS = _o.environ.get('RD_VARIANTS', 'off,wntf').split(',')
NLIM = int(_o.environ.get('RD_N', 0)); EXTRA = int(_o.environ.get('RD_EXTRA', 320)); MAXF = int(_o.environ.get('RD_MAXF', 2))
sp = _j.load(open(_o.path.join(_o.environ['RD_DIR'], 'dev_split.json')))
tr = [_j.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')]
used = set(sp['dev']) | set(sp['dev2'])
d3 = [i for i, r_ in enumerate(tr) if C.part(r_['question']) == 'B' and r_['question'] not in shot_q and i not in used]
items = [dict(i=i, set=nm, q=tr[i]['question'], gold=C.gold_of(tr[i]['answer'])) for nm, L in (('dev', sp['dev']), ('dev2', sp['dev2']), ('dev3', d3)) for i in L]
items = [r for r in items if len(C.q_ids(tok, r['q'])) + a.max_new <= 544]
if NLIM: items = items[:NLIM]
eng.__class__ = CompEngine
eng.comp_setup('wntf', maxf=MAXF, cap0=a.max_new, extra=EXTRA, cache=os.environ.get('DFA_CACHE'))
GEN = a.max_new + MAXF * EXTRA
print('COMP setup: classes', eng.c_TC.shape[0], 'states', eng.c_nS, 'max_new', GEN, 'S_max', eng.S_max, flush=True)
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
qids = [C.q_ids(tok, r['q']) for r in items]
def score(t, w, gold):
    txt = tok.decode(t).replace('<|endoftext|>', '').split('\n\n')[0]
    last = max([k for k, x in enumerate(w) if x & 2], default=-1)
    if last < 0:
        m = _re.search(C.NUM, txt); pred = m.group(1).replace(',', '') if m else None
    else:
        pre = tok.decode(t[:last + 1]).replace('<|endoftext|>', '')
        m = _re.search(C.NUM, txt[len(pre):]) if txt.startswith(pre) else None
        pred = m.group(1).replace(',', '') if m else None
    return float(C.correct(pred, gold)), pred, txt
base = None
for var in VARS:
    t0 = _t.time()
    vb, _, mm = var.partition('@M'); dbg = vb.endswith('!dbg'); vb = vb.replace('!dbg', '')
    zero = vb.endswith('+z'); pv = vb[:-2] if zero else vb
    if var == 'off': eng.set_comp(False)
    else:
        eng.set_phrases(pv)
        eng.set_comp(True, zero=zero, forget=pv.startswith('f'), wbias=pv.startswith('w'), M=float(mm or 50), dbg=dbg)
    eng.c_wmax.zero_()
    torch.manual_seed(0); eng.seed.fill_(0)
    o = eng.generate(qids, G=1, greedy=True, max_new=GEN)
    rows = []
    for r, t, w in zip(items, o['tokens'], o['wflag']):
        ok, pred, txt = score(t, w, r['gold'])
        nf = sum(1 for x in w if x & 8)
        # forced-token fidelity: every written token equals the phrase token the pointer asked for
        wr = [x for x, f in zip(t, w) if f & 2]
        spans, cur = [], []
        for k, (x, f) in enumerate(zip(t, w)):
            if f & 2: cur.append(x)
            elif cur: spans.append(cur); cur = []
        if cur: spans.append(cur)
        rows.append(dict(i=r['i'], set=r['set'], gold=r['gold'], ok=ok, pred=pred, ntok=len(t), fires=nf, nwritten=len(wr),
                         written=tok.decode(wr), spans=spans, tokens=t, wflag=w, txt=txt))
    with open(f'{OUT}/{var.replace("!", "_")}.jsonl', 'w') as f:
        for x in rows: f.write(_j.dumps(x) + '\n')
    rec = {}
    for nm in ('dev', 'dev2', 'dev3'):
        ks = [k for k, x in enumerate(rows) if x['set'] == nm]
        if not ks: continue
        acc = 100 * sum(rows[k]['ok'] for k in ks) / len(ks)
        rr = dict(n=len(ks), acc=round(acc, 2), ntok=sum(rows[k]['ntok'] for k in ks))
        if base is not None:
            fired = [k for k in ks if rows[k]['fires']]; silent = [k for k in ks if not rows[k]['fires']]
            rr.update(fires=len(fired), fixes=sum(1 for k in fired if rows[k]['ok'] and not base[k]['ok']),
                      breaks=sum(1 for k in fired if base[k]['ok'] and not rows[k]['ok']),
                      silent_identical=f"{sum(1 for k in silent if rows[k]['tokens'] == base[k]['tokens'])}/{len(silent)}",
                      delta=round(acc - 100 * sum(base[k]['ok'] for k in ks) / len(ks), 2),
                      extra_tok_pct=round(100 * (sum(rows[k]['ntok'] for k in ks) / sum(base[k]['ntok'] for k in ks) - 1), 2))
        rec[nm] = rr
    if var != 'off':
        exp = [T(v) for v in __import__('comp_sanity').PHRASES[pv].values()]
        nsp = sum(len(x['spans']) for x in rows)
        bad = sum(1 for x in rows for sp_ in x['spans'] if sp_ not in exp and not any(e[:len(sp_)] == sp_ for e in exp))
        rec['written_spans'] = nsp; rec['spans_not_exact_phrase'] = bad
    if var != 'off' and getattr(eng, 'c_wbias', False):
        rec['max_weight_on_abandoned_keys'] = float(eng.c_wmax.max()) if dbg else None
        rec['M'] = eng.c_M
    rec['seconds'] = round(_t.time() - t0, 1); rec['decode_tok_s'] = round(eng.stats.get('tok_s', 0), 1)
    if var == 'off': base = rows
    print('COMP', var, _j.dumps(rec), f'{_t.time() - t0:.0f}s', flush=True)
print('COMPDONE', flush=True)
