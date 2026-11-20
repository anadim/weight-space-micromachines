# exec'd inside tw_dump_lean.py --eval_ckpt (needs SAN_SMAX large, e.g. 1700).
# SER_JOBS = comma list of  set:mode:N:vmode[:thr]   mode greedy | T0.7 ; vmode 0 count 1 probe 2 last ; thr -> adaptive
import os as _o, sys as _s, time as _t, random as _r, re as _re
_s.path.insert(0, _o.environ['SEL_DIR'])
from serial_engine import SerialEngine
import torch as _torch
OUT = _o.environ['SEL_OUT']; _o.makedirs(OUT, exist_ok=True)
ACAP = a.max_new
NCAP = int(_o.environ.get('SEL_NCAP', 0))


def _split(name):
    if name in ('dev', 'dev2'):
        it = C.load_train('B', which=('gsm',), shot_q=shot_q)
        it = [r for r in it if len(C.q_ids(tok, r['q'])) + ACAP <= 544]
        _r.Random(0).shuffle(it)
        return it[:600] if name == 'dev' else it[600:900]
    if name == 'trainC':
        it = C.load_train('C', which=('gsm',), shot_q=shot_q)
        return [r for r in it if len(C.q_ids(tok, r['q'])) + ACAP <= 544]
    raise ValueError(name)


PR = _torch.load(_o.environ['SER_PROBE'])
eng.__class__ = SerialEngine
eng.serial_setup(acap=ACAP, layer=PR.get('layer', 18), **({'wait': _o.environ['SER_WAIT'].replace('\\n', '\n')} if _o.environ.get('SER_WAIT') else {}))
print('SER S_max', eng.S_max, 'wait', eng.q_wait, flush=True)
for job in _o.environ['SER_JOBS'].split(','):
    f = job.split(':'); nm, mode, N, vm = f[0], f[1], int(f[2]), int(f[3])
    thr = float(f[4]) if len(f) > 4 else None
    tag = f'{nm}_{mode}_N{N}_v{vm}' + (f'_ad{thr:g}' if thr is not None else '')
    tag += _o.environ.get('SER_TAG', '')
    fn = f'{OUT}/ser_{tag}.pt'
    if _o.path.exists(fn): print('skip', fn, flush=True); continue
    items = _split(nm)
    if NCAP: items = items[:NCAP]
    qids = [C.q_ids(tok, r['q']) for r in items]
    eng.set_serial(N=N, adaptive=thr is not None, thr=thr if thr is not None else 1e9, vmode=vm, w=PR['w'].cuda(), b=PR['b'], forget=_o.environ.get('SER_FORGET') == '1', posfix=_o.environ.get('SER_POSFIX') == '1')
    torch.manual_seed(7); eng.seed.fill_(7)
    greedy = mode == 'greedy'; T = 1.0 if greedy else float(mode[1:])
    maxn = int(_o.environ.get('SER_NMAXGEN', 4)) * (ACAP + 16) + 40          # fixed for all jobs (one output buffer, one graph pool)
    assert max(map(len, qids)) + maxn <= eng.S_max, (max(map(len, qids)), maxn, eng.S_max)
    t0 = _t.time()
    o = eng.generate(qids, G=1, T=T, greedy=greedy, max_new=maxn)
    dt = _t.time() - t0
    rows = []
    for j, (t, w) in enumerate(zip(o['tokens'], o['wflag'])):
        txt = tok.decode(t).replace('<|endoftext|>', '')
        ms = list(_re.finditer(C.NUM, txt))
        last = ms[-1].group(1).replace(',', '') if ms else None
        plain = C.score(tok, t, items[j]['gold'])
        rows.append(dict(gold=items[j]['gold'], ntok=len(t), last=last, ok_last=int(C.correct(last, items[j]['gold'])),
                         ok_plain=int(plain[0]), nw=sum(1 for x in w if x & 8)))
    D = dict(job=job, q=[r['q'] for r in items], rows=rows, tokens=o['tokens'], wflag=o['wflag'], reg=o['reg'], sec=dt)
    _torch.save(D, fn)
    n = len(rows)
    print('SERJOB', tag, f'n={n} acc_last={100*sum(x["ok_last"] for x in rows)/n:.2f} tok/item={sum(x["ntok"] for x in rows)/n:.1f} '
          f'natt={o["reg"]["natt"].float().mean():.2f} sec={dt:.0f}', flush=True)
print('SERDONE', flush=True)
