# exec'd inside dig_san/tw_dump_lean.py --eval_ckpt ... --fused 0 (names in scope: eng, C, tok, a, shot_q, torch).
# The combined implant stack on one engine:  StackEngine = ProbeEngine(latch + L18 capture) over C2Engine(calc v2).
#   g0    greedy, calc v2 off                         -> S0
#   g1    greedy, calc v2 verify_rsl                  -> S1
#   casc  on S1 rows flagged by the answer-sanity trigger: restart plan1 (greedy) > re2q (greedy) > plain T=1   -> S2
#   samp  K samples at T=1 with calc v2 on, hidden state at '####' captured (L18)                               -> S3/S3a
#   rep   every flagged sample restarted (plan1 T=1 > re2q T=1 > plain T=1) before voting                      -> S4
# STK_SET in {dev, dev2, test, hard, trainC}; STK_STEPS comma list; STK_K (16); STK_MISSING_ONLY=1 (GSM-Hard rule).
import json as _j, os as _o, sys as _s, time as _t, random as _r
_s.path.insert(0, _o.environ['STK_DIR'])
import engine as EM
from c2_engine import C2Engine
import probe_engine as PE
from probe_engine import ProbeEngine
import rd_trig as RT
OUT = _o.environ['STK_OUT']; _o.makedirs(OUT, exist_ok=True)
SET = _o.environ['STK_SET']; STEPS = _o.environ.get('STK_STEPS', 'g0,g1,casc,samp,rep').split(',')
K = int(_o.environ.get('STK_K', 16)); MISS = _o.environ.get('STK_MISSING_ONLY', '0') == '1'
NLIM = int(_o.environ.get('STK_N', 0)); MAXN = a.max_new
LOG = open(f'{OUT}/{SET}.log', 'a')
def P(*x):
    s = ' '.join(str(y) for y in x); print('STK', SET, s, flush=True); LOG.write(s + '\n'); LOG.flush()


def items_of(nm):
    f = lambda it: [r for r in it if len(C.q_ids(tok, r['q'])) + MAXN <= 544]
    if nm in ('dev', 'dev2'):
        it = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(0).shuffle(it)
        return it[:600] if nm == 'dev' else it[600:900]
    if nm == 'trainC': return f(C.load_train('C', which=('gsm',), shot_q=shot_q))
    if nm in ('test', 'hard'):
        al = C.load_test(nm, shot_q=shot_q); P('n_all', len(al)); return f(al)
    raise ValueError(nm)


items = items_of(SET)
if NLIM: items = items[:NLIM]
P('items', len(items))
T_ = lambda s: tok(s, add_special_tokens=False)['input_ids']
PROMPTS = {'off': lambda q: f"Question: {q}\nAnswer:",
           're2q': lambda q: f"Question: {q}\nRead the question again: {q}\nAnswer:",
           'plan1': lambda q: f"Question: {q}\nAnswer: We need to find"}
assert PROMPTS['off']('x') == tok.decode(C.q_ids(tok, 'x'))


def flag_of(txt):          # 0 ok, 1 negative, 2 non-integer, 3 missing (scorer's view); GSM-Hard rule: missing only
    v = RT.answer(txt)
    if v is None: return 3
    if MISS: return 0
    if v < 0: return 1
    if v != int(v): return 2
    return 0


def mark(t, w):
    s = ''; on = 0
    for x, f in zip(t, w):
        c = 2 if f & 2 else (3 if f & 4 else (1 if f & 1 else 0))
        if c != on:
            s += {1: '⟧', 2: '»', 3: '›'}.get(on, ''); s += {1: '⟦', 2: '«', 3: '‹'}.get(c, ''); on = c
        s += tok.decode([x])
    s += {1: '⟧', 2: '»', 3: '›'}.get(on, '')
    return s.replace('<|endoftext|>', '')


def nfires(c2): return sum(1 for v in c2['events'] if v.get('act') in ('preempt', 'rewrite', 'rewrite_restate'))


def gen(prompts, mode, greedy, seed=0, G=1):
    eng.set_c2(mode, exact=True, policy='look')
    if not greedy: torch.manual_seed(seed); eng.seed.fill_(seed)
    t0 = _t.time()
    o = eng.generate(prompts, G=G, T=1.0, greedy=greedy, max_new=MAXN)
    o['sec'] = _t.time() - t0
    return o


def rowrec(r, t, w, c2):
    ok, fmt, txt = C.score(tok, t, r['gold'])
    return dict(gold=r['gold'], ok=ok, pred=RT.answer(txt), flag=flag_of(txt), ntok=len(t), c2fires=nfires(c2), tokens=t, txt=mark(t, w))


def wjsonl(fn, rows):
    with open(fn, 'w') as f:
        for x in rows: f.write(_j.dumps(x) + '\n')


def rjsonl(fn): return [_j.loads(l) for l in open(fn)]


# ---------------------------------------------------------------- optional silence check: stock engine vs StackEngine(off)
stock = None
if _o.environ.get('STK_CHECK') and 'g0' in STEPS:
    o = eng.generate([C.q_ids(tok, r['q']) for r in items], G=1, greedy=True, max_new=MAXN)
    stock = o['tokens']
# ---------------------------------------------------------------- build the stacked engine
eng.__class__ = type('StackEngine', (ProbeEngine, C2Engine), {})
eng.c2_setup()
eng.probe_setup(layers=(18,))
PE.HK = EM.HK            # the probe step must call the sampler through calc v2's proxy (ALT token for the veto)
assert getattr(EM.HK, 'sample_kernel', None) is not None and type(EM.HK).__name__ == '_HKProxy'
qids = [T_(PROMPTS['off'](r['q'])) for r in items]
assert all(x == C.q_ids(tok, r['q']) for x, r in zip(qids[:20], items[:20]))

if 'g0' in STEPS:
    o = gen(qids, 'off', True)
    rows = [rowrec(r, t, w, c) for r, t, w, c in zip(items, o['tokens'], o['wflag'], o['c2'])]
    wjsonl(f'{OUT}/{SET}_S0.jsonl', rows)
    P('S0', f"acc={100*sum(x['ok'] for x in rows)/len(rows):.2f}", 'tok', sum(x['ntok'] for x in rows), f"sec={o['sec']:.0f}",
      'probe_capture', int((o['hpos'] >= 0).sum()))
    if stock is not None:
        P('SILENCE stock_engine==stack_off', sum(int(x == y['tokens']) for x, y in zip(stock, rows)), '/', len(rows),
          'same_ok', sum(int(C.score(tok, x, y['gold'])[0] == y['ok']) for x, y in zip(stock, rows)))
if 'g1' in STEPS:
    o = gen(qids, 'verify_rsl', True)
    rows = [rowrec(r, t, w, c) for r, t, w, c in zip(items, o['tokens'], o['wflag'], o['c2'])]
    wjsonl(f'{OUT}/{SET}_S1.jsonl', rows)
    msg = ['S1', f"acc={100*sum(x['ok'] for x in rows)/len(rows):.2f}", 'tok', sum(x['ntok'] for x in rows), f"sec={o['sec']:.0f}",
           'rollback_tok', o.get('rollback_tok')]
    if _o.path.exists(f'{OUT}/{SET}_S0.jsonl'):
        b = rjsonl(f'{OUT}/{SET}_S0.jsonl'); sil = [k for k, x in enumerate(rows) if x['c2fires'] == 0]
        msg += ['fire_rows', len(rows) - len(sil), 'silent_identical', f"{sum(rows[k]['tokens'] == b[k]['tokens'] for k in sil)}/{len(sil)}"]
    P(*msg)
if 'casc' in STEPS:                 # S2 = S1 + sanity trigger -> reworded restart cascade (all attempts with calc v2 on)
    base = rjsonl(f'{OUT}/{SET}_S1.jsonl')
    rows = [dict(gold=x['gold'], ok=x['ok'], pred=x['pred'], flag=x['flag'], ntok=x['ntok'], fire=0, path='S1', base_ok=x['ok'],
                 txt=x['txt']) for x in base]
    for st in ('plan1', 're2q', 'T1s0'):
        todo = [k for k, x in enumerate(rows) if x['flag'] != 0]
        if not todo: break
        pk = 'off' if st.startswith('T1s') else st
        pr = [T_(PROMPTS[pk](items[k]['q'])) for k in todo]
        o = gen(pr, 'verify_rsl', not st.startswith('T1s'), seed=0)
        for k, t, w, c in zip(todo, o['tokens'], o['wflag'], o['c2']):
            rr = rowrec(items[k], t, w, c)
            rows[k].update(ok=rr['ok'], pred=rr['pred'], flag=rr['flag'], ntok=rows[k]['ntok'] + rr['ntok'], fire=rows[k]['fire'] + 1,
                           path=rows[k]['path'] + '>' + st, txt=rows[k]['txt'] + f'【RESTART {st}】' + rr['txt'])
        P('casc step', st, 'rows', len(todo), f"sec={o['sec']:.0f}")
    wjsonl(f'{OUT}/{SET}_S2.jsonl', rows)
    P('S2', f"acc={100*sum(x['ok'] for x in rows)/len(rows):.2f}", 'tok', sum(x['ntok'] for x in rows), 'fire_rows', sum(x['fire'] > 0 for x in rows))
if 'samp' in STEPS:                 # K samples at T=1, calc v2 on, L18 state at '####'
    o = gen(qids, 'verify_rsl', False, seed=1000, G=K)
    S = []
    for j, (t, w, c) in enumerate(zip(o['tokens'], o['wflag'], o['c2'])):
        r = items[j // K]; ok, fmt, txt = C.score(tok, t, r['gold'])
        S.append(dict(ok=ok, pred=RT.answer(txt), flag=flag_of(txt), ntok=len(t), c2fires=nfires(c), txt=mark(t, w) if j % K == 0 else None))
    torch.save(dict(K=K, gold=[r['gold'] for r in items], q=[r['q'] for r in items], rows=S, hcap=o['hcap'][:, 0].clone(),
                    hpos=o['hpos'].clone(), sec=o['sec']), f'{OUT}/{SET}_samp.pt')
    wjsonl(f'{OUT}/{SET}_samp.jsonl', [dict(q=j // K, k=j % K, **{k_: v for k_, v in x.items() if k_ != 'txt' or v}) for j, x in enumerate(S)])
    P('samp', 'rows', len(S), f"mean_ok={100*sum(x['ok'] for x in S)/len(S):.2f}", f"tok/row={sum(x['ntok'] for x in S)/len(S):.1f}",
      'cap', float((o['hpos'] >= 0).float().mean()), 'flagged', sum(x['flag'] != 0 for x in S), f"sec={o['sec']:.0f}")
if 'rep' in STEPS:                  # S4: flagged samples restarted with the reworded cascade (sampled) before the vote
    D = torch.load(f'{OUT}/{SET}_samp.pt'); S = D['rows']; Kd = D['K']
    hcap = D['hcap'].clone(); hpos = D['hpos'].clone()
    R = [dict(ok=x['ok'], pred=x['pred'], flag=x['flag'], ntok=x['ntok'], fire=0) for x in S]
    for si, st in enumerate(('plan1', 're2q', 'off')):
        todo = [j for j, x in enumerate(R) if x['flag'] != 0]
        if not todo: break
        pr = [T_(PROMPTS[st](items[j // Kd]['q'])) for j in todo]
        o = gen(pr, 'verify_rsl', False, seed=5000 + si)
        for n_, (j, t, w, c) in enumerate(zip(todo, o['tokens'], o['wflag'], o['c2'])):
            ok, fmt, txt = C.score(tok, t, items[j // Kd]['gold'])
            R[j].update(ok=ok, pred=RT.answer(txt), flag=flag_of(txt), ntok=R[j]['ntok'] + len(t), fire=R[j]['fire'] + 1)
            hcap[j] = o['hcap'][n_, 0]; hpos[j] = o['hpos'][n_]
        P('rep step', st, 'rows', len(todo), f"sec={o['sec']:.0f}")
    torch.save(dict(K=Kd, gold=D['gold'], rows=R, hcap=hcap, hpos=hpos), f'{OUT}/{SET}_rep.pt')
    wjsonl(f'{OUT}/{SET}_rep.jsonl', [dict(q=j // Kd, k=j % Kd, **x) for j, x in enumerate(R)])
    P('rep done', 'repaired', sum(x['fire'] > 0 for x in R), 'still_flagged', sum(x['flag'] != 0 for x in R))
P('DONE')
print('STKDONE', flush=True)
