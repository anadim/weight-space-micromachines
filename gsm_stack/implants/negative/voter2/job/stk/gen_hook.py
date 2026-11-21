# S3-serial: the selection agent's serial tries + forget + position register + register/probe voter, ported onto the
# final stack S2F (calc v2 fwd inside every attempt; the S2F sanity repair inside every attempt; weight-space forget).
# exec'd inside dig_san/tw_dump_lean.py --eval_ckpt ... --fused 0, with C2_NOCOMPACT=1 and SAN_SMAX large.
# STK_SER = ';'-list of  N:R:vmode:san:repmiss:mode   (R = sanity repairs per attempt; vmode 0 count / 1 probe;
#            san 1 = voter sanity filter; repmiss 1 = repair only missing answers; mode greedy | T1.0)
import json as _j, os as _o, sys as _s, time as _t, random as _r, re as _re
_s.path.insert(0, _o.environ['STK_DIR'])
import engine as EM
from c2_engine import C2Engine
import calc2
import serial_stack as SS
from serial_stack import SerialEngine
import rd_trig as RT
OUT = _o.environ['STK_OUT']; _o.makedirs(OUT, exist_ok=True)
SET = _o.environ['STK_SET']; NLIM = int(_o.environ.get('STK_N', 0)); ACAP = a.max_new
JOBS = _o.environ['STK_SER'].split(';')
SEEDS = [int(x) for x in _o.environ.get('STK_SEEDS', '7').split(',')]
LOG = open(f'{OUT}/{SET}.log', 'a')
def P(*x):
    s = ' '.join(str(y) for y in x); print('STK', SET, s, flush=True); LOG.write(s + '\n'); LOG.flush()


def items_of(nm):
    f = lambda it: [r for r in it if len(C.q_ids(tok, r['q'])) + ACAP <= 544]
    if nm in ('dev', 'dev2'):
        it = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(0).shuffle(it)
        return it[:600] if nm == 'dev' else it[600:900]
    if nm == 'dev3':
        it = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(0).shuffle(it); return it[900:]
    if nm == 'trainC':
        return f(C.load_train('C', which=('gsm',), shot_q=shot_q))
    if nm in ('test', 'hard'):
        al = C.load_test(nm, shot_q=shot_q); P('n_all', len(al)); return f(al)
    raise ValueError(nm)


items = items_of(SET)
if _o.environ.get('STK_SLICE'):                       # 'k/m': the k-th of m interleaved shards (multi-GPU)
    k_, m_ = map(int, _o.environ['STK_SLICE'].split('/')); items = items[k_::m_]
if NLIM: items = items[:NLIM]
qids = [C.q_ids(tok, r['q']) for r in items]
P('items', len(items), 'S_max', eng.S_max, 'B_max', eng.B_max)


class StackS(SerialEngine, C2Engine):
    def _wave(self, qs, G, T, max_new, greedy):
        self._n_wave = len(qs) * G
        return SerialEngine._wave(self, qs, G, T, max_new, greedy)

    def _stops(self, row, nt, glen, max_new):          # the serial writer decides every end (calc v2 never ends a row)
        return nt == self.eos

    def _c2_apply(self, acts, max_new):
        if acts: C2Engine._c2_apply(self, acts, max_new)
        n = self._n_wave
        sn = self.seg_new[:n].nonzero()[:, 0].tolist()
        if sn:                                          # calc v2 tracker reset at each new attempt / repair
            R = self.c2_rows[-n:]
            for j in sn:
                old = R[j]; nr = calc2.Row(self.c2_mode, self.c2_exact, self.c2_policy)
                nr.events, nr.vetoes = old.events, old.vetoes
                R[j].__dict__.update(nr.__dict__)
            self.seg_new[:n] = False


PR = torch.load(f"{_o.environ['STK_DIR']}/verifier_probe_L18.pt")
eng.__class__ = StackS
eng.c2_setup()
eng.serial_setup(acap=ACAP, layer=PR.get('layer', 18))
assert eng.ws, 'SER_WS=1 required (weight-space forget)'
assert type(EM.HK).__name__ == '_HKProxy' and _o.environ.get('C2_NOCOMPACT') == '1'


def nfires(c2): return sum(1 for v in c2['events'] if v.get('act') in ('preempt', 'rewrite', 'rewrite_restate'))


for job, SEED in [(j_, s_) for j_ in JOBS for s_ in SEEDS]:
    N, Rr, vm, san, rm, mode = job.split(':'); N, Rr, vm, san, rm = int(N), int(Rr), int(vm), san == '1', rm == '1'
    tag = f'N{N}_R{Rr}_v{vm}_s{int(san)}_m{int(rm)}_{mode}' + f'_seed{SEED}' + _o.environ.get('STK_TAG', '')
    if eng.v2: tag += '_v2'
    eng.set_c2(_o.environ.get('STK_C2MODE', 'fwd'), exact=False, policy='preempt')
    eng.set_serial(N=N, vmode=vm, w=PR['w'].cuda(), b=PR['b'], forget=True, posfix=True, R=Rr, repmiss=rm, san=san)
    torch.manual_seed(SEED); eng.seed.fill_(SEED)
    greedy = mode == 'greedy'; T = 1.0 if greedy else float(mode[1:])
    maxn = int(_o.environ.get('STK_MAXN', 5416))          # fixed for all jobs (one output buffer, one graph pool)
    assert N * (1 + Rr) * (ACAP + 16) + 40 <= maxn
    assert max(map(len, qids)) + maxn <= eng.S_max, (maxn, eng.S_max)
    t0 = _t.time()
    o = eng.generate(qids, G=1, T=T, greedy=greedy, max_new=maxn)
    dt = _t.time() - t0
    reg = o['reg']; rows = []
    for j, (t, w, c, lpp) in enumerate(zip(o['tokens'], o['wflag'], o['c2'], o['logp'])):
        txt = tok.decode(t).replace('<|endoftext|>', '')
        ms = list(_re.finditer(C.NUM, txt)); last = ms[-1].group(1).replace(',', '') if ms else None
        rows.append(dict(tokens=list(map(int, t)), wf=[int(v) for v in w], logp=[round(float(v), 4) for v in lpp], c2ev=[{k: v for k, v in e.items() if isinstance(v, (int, float, str))} for e in c['events']], idx=j, q=items[j]['q'], gold=items[j]['gold'], ok=int(C.correct(last, items[j]['gold'])), last=last, ntok=len(t),
                         natt=int(reg['natt'][j]), nseg=int(reg['nseg'][j]), pick=int(reg['pick'][j]), c2fires=nfires(c),
                         val=reg['val'][j].tolist(), rok=reg['ok'][j].tolist(), bad=reg['bad'][j].tolist(), lg=reg['lg'][j].tolist(),
                         end=reg['end'][j].tolist(), txt=txt))
        if eng.v2: rows[-1].update(v2=[round(float(v), 5) for v in reg['v2'][j]], pick0=int(reg['pick0'][j]))
    with open(f'{OUT}/{SET}_{tag}.jsonl', 'w') as f:
        for x in rows: f.write(_j.dumps(x) + '\n')
    n = len(rows)
    P(tag, f"acc={100*sum(x['ok'] for x in rows)/n:.2f}", f"tok/q={sum(x['ntok'] for x in rows)/n:.1f}",
      f"natt={sum(x['natt'] for x in rows)/n:.2f}", f"repairs={sum(x['nseg']-x['natt'] for x in rows)}", f"c2rows={sum(x['c2fires']>0 for x in rows)}", f'sec={dt:.0f}')
P('DONE')
print('STKDONE', flush=True)
