# Forward-only, single-call implant stack (scope change 2026-11-20). exec'd inside dig_san/tw_dump_lean.py --eval_ckpt
# ... --fused 0. One engine: StackF = SanEngine (answer-sanity DFA + phrase writer, in-graph) over C2Engine (calc v2,
# policy 'preempt', mode 'fwd': writes the exact value at the number start after '=', never rewinds).
#   S0        calc v2 off, sanity off           (host + original calculator)
#   S1        calc v2 fwd                       (forward-only calculator v2)
#   S2_<set>  calc v2 fwd + sanity phrase <set> (forward-only: the end token is replaced by a phrase; same call)
# STK_SET in {dev, dev2, test, hard}; STK_CFGS comma list (S0,S1,S2_generic,S2_typed,S2_plan,S2_reread).
# GSM-Hard: set SAN_CLASSES=3 (sanity on missing answers only).
import json as _j, os as _o, sys as _s, time as _t, random as _r, re as _re
_s.path.insert(0, _o.environ['STK_DIR'])
import engine as EM
from c2_engine import C2Engine
import san_engine as SE
from san_engine import SanEngine
import rd_trig as RT
OUT = _o.environ['STK_OUT']; _o.makedirs(OUT, exist_ok=True)
SET = _o.environ['STK_SET']; CFGS = _o.environ.get('STK_CFGS', 'S0,S1,S2_generic,S2_typed,S2_plan,S2_reread').split(',')
NLIM = int(_o.environ.get('STK_N', 0)); CAP0 = a.max_new; EXTRA = int(_o.environ.get('SAN_EXTRA', 320)); MAXF = 2
GEN_MAX = CAP0 + MAXF * EXTRA
LOG = open(f'{OUT}/{SET}.log', 'a')
def P(*x):
    s = ' '.join(str(y) for y in x); print('STK', SET, s, flush=True); LOG.write(s + '\n'); LOG.flush()


def items_of(nm):
    f = lambda it: [r for r in it if len(C.q_ids(tok, r['q'])) + CAP0 <= 544]
    if nm in ('dev', 'dev2'):
        it = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(0).shuffle(it)
        return it[:600] if nm == 'dev' else it[600:900]
    if nm in ('test', 'hard'):
        al = C.load_test(nm, shot_q=shot_q); P('n_all', len(al)); return f(al)
    raise ValueError(nm)


items = items_of(SET)
if NLIM: items = items[:NLIM]
P('items', len(items), 'S_max', eng.S_max, 'gen_max', GEN_MAX, 'SAN_CLASSES', _o.environ.get('SAN_CLASSES'))
qids = [C.q_ids(tok, r['q']) for r in items]

# ---------------------------------------------------------------- the stacked engine


class StackF(SanEngine, C2Engine):
    def _wave(self, qs, G, T, max_new, greedy):      # C2Engine's loop (sync every step, no compaction); reset sanity state
        self.s_dfa.zero_(); self.s_nf.zero_(); self.s_fr.fill_(-1); self.s_fp.zero_(); self.s_rst.zero_()
        return C2Engine._wave(self, qs, G, T, max_new, greedy)

    def _mut(self):
        return SanEngine._mut(self)


eng.__class__ = StackF
eng.c2_setup()
eng.san_setup(maxf=MAXF, room=24, cache=f"{_o.environ['STK_DIR']}/dfa_tables{'_zero' if SE.ZERO else ''}.pt", cap0=CAP0, extra=EXTRA)
SE.HK = EM.HK            # sanity step calls the sampler through calc v2's proxy (ALT token for calc v2's veto)
assert type(EM.HK).__name__ == '_HKProxy'
assert _o.environ.get('C2_NOCOMPACT') == '1'


def flag_of(txt):
    v = RT.answer(txt)
    if v is None: return 3
    if v < 0: return 1
    if v != int(v): return 2
    return 0


def score(ids, wf, gold):
    """C.score when sanity never fired; else the first '#### number' after the last fire (text cut at the first blank line)."""
    fires = [i for i, w in enumerate(wf) if w & 8]
    if not fires:
        ok, fmt, txt = C.score(tok, ids, gold); return ok, txt, 0
    f = fires[-1]
    txt = tok.decode(ids).replace('<|endoftext|>', '').split('\n\n')[0]
    pre = tok.decode(ids[:f]).replace('<|endoftext|>', '')
    tail = txt[len(pre):]
    m = _re.search(C.NUM, tail); pred = m.group(1).replace(',', '') if m else None
    return float(C.correct(pred, gold)), tail, len(fires)


def mark(t, w):
    s = ''; on = 0
    for x, f in zip(t, w):
        if f & 8: s += {1: '⟧', 2: '»', 3: '›'}.get(on, '') + '【SAN】'; on = 0
        c = 2 if f & 2 else (3 if f & 4 else (1 if f & 1 else 0))
        if c != on:
            s += {1: '⟧', 2: '»', 3: '›'}.get(on, ''); s += {1: '⟦', 2: '«', 3: '‹'}.get(c, ''); on = c
        s += tok.decode([x])
    s += {1: '⟧', 2: '»', 3: '›'}.get(on, '')
    return s.replace('<|endoftext|>', '')


def nfires(c2): return sum(1 for v in c2['events'] if v.get('act') in ('preempt', 'rewrite', 'rewrite_restate'))


base = {}
for cfg in CFGS:
    c2m = 'off' if cfg == 'S0' else 'fwd'
    eng.set_c2(c2m, exact=False, policy='preempt')
    if cfg.startswith('S2_'): eng.set_san('phrase', cfg[3:])
    else: eng.set_san('off')
    t0 = _t.time()
    o = eng.generate(qids, G=1, T=1.0, greedy=True, max_new=GEN_MAX)
    dt = _t.time() - t0
    rows = []
    for r, t, w, c in zip(items, o['tokens'], o['wflag'], o['c2']):
        ok, txt, sf = score(t, w, r['gold'])
        ok0, _, txt0 = C.score(tok, t, r['gold'])
        rows.append(dict(gold=r['gold'], ok=ok, pred=RT.answer(txt), flag=flag_of(txt0), ntok=len(t), c2fires=nfires(c), san_fires=sf,
                         c2events=[e for e in c['events'] if e.get('act') == 'preempt'], tokens=t, txt=mark(t, w)))
    with open(f'{OUT}/{SET}_{cfg}.jsonl', 'w') as f:
        for x in rows: f.write(_j.dumps(x) + '\n')
    n = len(rows); acc = 100 * sum(x['ok'] for x in rows) / n
    msg = [cfg, f'acc={acc:.2f}', 'tok/q', round(sum(x['ntok'] for x in rows) / n, 1), f'sec={dt:.0f}', 'rollback_tok', o.get('rollback_tok'),
           'c2_fire_rows', sum(x['c2fires'] > 0 for x in rows), 'san_fire_rows', sum(x['san_fires'] > 0 for x in rows)]
    base[cfg] = rows
    for ref in ('S0', 'S1'):
        if ref in base and ref != cfg:
            b = base[ref]
            fx = sum(1 for x, y in zip(rows, b) if x['ok'] and not y['ok']); br = sum(1 for x, y in zip(rows, b) if not x['ok'] and y['ok'])
            if cfg == 'S1' and ref == 'S0': sil = [k for k in range(n) if rows[k]['c2fires'] == 0]
            elif cfg.startswith('S2') and ref == 'S1': sil = [k for k in range(n) if rows[k]['san_fires'] == 0]
            else: sil = None
            msg += [f'vs{ref}', f'+{fx}/-{br}']
            if sil is not None: msg += ['silent_identical', f"{sum(rows[k]['tokens'] == b[k]['tokens'] for k in sil)}/{len(sil)}"]
    P(*msg)
P('DONE')
print('STKDONE', flush=True)
