# exec'd inside tw_dump_lean.py after --eval_ckpt weights are loaded (sync_weights() done).
# Generates K samples per question with the RL engine + calculator implant, and captures the host residual at the '####'
# token (ProbeEngine latch). Writes one .pt per job: tokens, logp, pred, ok, sanity flag, hidden captures.
# SEL_JOBS = comma list of  set:mode:G:seed   set in {dev,dev2,trainC}, mode in {greedy, T<temp>}
import os as _o, sys as _s, time as _t, random as _r
_s.path.insert(0, _o.environ['SEL_DIR'])
from probe_engine import ProbeEngine
import torch as _torch
OUT = _o.environ['SEL_OUT']; _o.makedirs(OUT, exist_ok=True)
JOBS = _o.environ['SEL_JOBS'].split(',')
NCAP = int(_o.environ.get('SEL_NCAP', 0))
MAXN = a.max_new


def _flag(pred):           # 0 ok, 1 negative, 2 non-integer, 3 missing (scorer's view)
    if pred is None: return 3
    try: v = float(pred)
    except ValueError: return 3
    if v < 0: return 1
    if v != int(v): return 2
    return 0


def _split(name):
    if name in ('dev', 'dev2'):
        it = C.load_train('B', which=('gsm',), shot_q=shot_q)
        it = [r for r in it if len(C.q_ids(tok, r['q'])) + MAXN <= eng.S_max]
        _r.Random(0).shuffle(it)
        return it[:600] if name == 'dev' else it[600:900]
    if name == 'trainC':
        it = C.load_train('C', which=('gsm',), shot_q=shot_q)
        it = [r for r in it if len(C.q_ids(tok, r['q'])) + MAXN <= eng.S_max]
        return it
    if name == 'devS':                     # split-B scaled-number copies (GSM-Hard-like; golds are noisy, see failures §2a)
        it = C.load_train('B', which=('scaled',), shot_q=shot_q)
        it = [r for r in it if len(C.q_ids(tok, r['q'])) + MAXN <= eng.S_max]
        _r.Random(0).shuffle(it)
        return it[:300]
    raise ValueError(name)


import re as _re
if _o.environ.get('SEL_CHECK'):           # silence check: ProbeEngine greedy == plain engine greedy
    items = _split('dev')[:int(_o.environ['SEL_CHECK'])]
    qids = [C.q_ids(tok, r['q']) for r in items]
    o0 = eng.generate(qids, G=1, greedy=True, max_new=MAXN)
eng.__class__ = ProbeEngine
eng.probe_setup(layers=tuple(int(x) for x in _o.environ.get('SEL_LAYERS', '14,18,22').split(',')))
if _o.environ.get('SEL_CHECK'):
    o1 = eng.generate(qids, G=1, greedy=True, max_new=MAXN)
    same = sum(int(a_ == b_) for a_, b_ in zip(o0['tokens'], o1['tokens']))
    print('SELCHECK identical', same, '/', len(qids), 'captured', int((o1['hpos'] >= 0).sum()), flush=True)
for job in JOBS:
    nm, mode, G, sd = job.split(':')
    G, sd = int(G), int(sd)
    fn = f'{OUT}/{nm}_{mode}_G{G}_s{sd}.pt'
    if _o.path.exists(fn): print('skip', fn, flush=True); continue
    items = _split(nm)
    if NCAP: items = items[:NCAP]
    qids = [C.q_ids(tok, r['q']) for r in items]
    torch.manual_seed(1000 + sd); eng.seed.fill_(1000 + sd)
    greedy = mode == 'greedy'; T = 1.0 if greedy else float(mode[1:])
    t0 = _t.time()
    o = eng.generate(qids, G=G, T=T, greedy=greedy, max_new=MAXN)
    dt = _t.time() - t0
    preds, oks, flags, ntok, lpm = [], [], [], [], []
    for j, (t, lp) in enumerate(zip(o['tokens'], o['logp'])):
        r = items[j // G]
        txt = tok.decode(t).replace('<|endoftext|>', '').split('\n\n')[0]
        m = _re.search(C.NUM, txt); pred = m.group(1).replace(',', '') if m else None
        preds.append(pred); oks.append(int(C.correct(pred, r['gold']))); flags.append(_flag(pred))
        ntok.append(len(t)); lpm.append(sum(lp) / max(1, len(lp)))
    # does the capture position hold the '####' that the scorer reads? (first '####' in the text; same block)
    D = dict(set=nm, mode=mode, G=G, seed=sd, T=T, q=[r['q'] for r in items], gold=[r['gold'] for r in items],
             tokens=o['tokens'], pred=preds, ok=oks, flag=flags, ntok=ntok, lp_mean=lpm,
             hcap=o['hcap'], hpos=o['hpos'], layers=eng.cap_layers, sec=dt, stats=o['stats'])
    _torch.save(D, fn)
    n = len(items)
    acc1 = sum(oks[j * G] for j in range(n)) / n
    print('SELJOB', job, f'n={n} rows={n*G} acc(sample0)={100*acc1:.2f} mean_ok={100*sum(oks)/len(oks):.2f} '
          f'tok/row={sum(ntok)/len(ntok):.1f} cap={(o["hpos"]>=0).float().mean():.3f} sec={dt:.0f}', flush=True)
print('SELDONE', flush=True)
