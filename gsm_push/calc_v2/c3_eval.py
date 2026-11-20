# exec'd inside tw_dump_lean.py after the --eval_ckpt weights are loaded (names: eng, imp, C, tok, a, shot_q, torch, W).
# calc v2 COMPILED: calc_s4 alone vs calc_s4 + chain2 (forward-only micromachine), ONE generate call per set, greedy.
import json as _j, os as _o, sys as _s, time as _t, random as _r
_s.path.insert(0, _o.environ['C2_DIR'])
import chain2
chain2.register(tok)
from implant import CalcImplant
from implant2 import DualImplant
from c2_engine import C2Engine
from transformers import AutoModelForCausalLM
OUT = _o.environ['C2_OUT']; _o.makedirs(OUT, exist_ok=True)
SETS = _o.environ.get('C2_SETS', 'dev,dev2,hard').split(',')
NLIM = int(_o.environ.get('C2_N', 0))
_hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to('cuda').eval()
ch = CalcImplant(_hf, tok, a.B_max, machine='chain2', fused=False)       # built against the base model, as calc_s4 is
del _hf; torch.cuda.empty_cache()
# reader exactness on the HOST's embedding rows (the engine feeds the RL host's own embeddings)
with torch.no_grad():
    Eh = eng.W['E'].float()
    symh = ch.M.symbols(Eh)
    ok = bool(((symh == 0) | (symh == 1)).all())
    ids = ch.A.all_ids['chain2']; want = torch.zeros_like(symh)
    for i, s_ in enumerate(chain2.SYMS):
        t_ = ids[s_] if isinstance(ids[s_], list) else [ids[s_]]
        want[t_, i] = 1
    print('C3 reader on host E: binary', ok, 'exact', bool(torch.equal(symh, want)), 'mismatch rows', int((symh != want).any(-1).sum()), flush=True)
dual = DualImplant(imp, ch)
import copy as _cp
_v = torch.load(_o.path.join(_o.environ['C2_DIR'], 'veto_op_qwen3.pt'))['v'].to(imp.D)
def _opv(m, oe):
    m2 = _cp.copy(m); m2.D = m.D.clone(); m2.D[oe] = _v; return m2
imp_op = _opv(imp, 13)                      # calc_s4 OUT: digits, '.', '-', OS, OE
ch_op = _opv(ch, chain2.OUT.index('OE'))
dual_op = DualImplant(imp_op, ch_op)
VARS = {'calc': imp, 'calc+chain2': dual, 'calc+opveto': imp_op, 'calc+chain2+opveto': dual_op}
VLIST = _o.environ.get('C3_VARIANTS', 'calc,calc+chain2,calc+opveto,calc+chain2+opveto').split(',')


class C3Engine(C2Engine):
    def _mut(self):
        m = super()._mut()
        if isinstance(self.imp, DualImplant): m += [self.imp.chain.sym_h, self.imp.chain.S_h, self.imp.chain.key_h, self.imp.chain.fst]
        return m


eng.__class__ = C3Engine
eng.c2_setup(); eng.set_c2('off')
pool_ = [r for r in C.load_train('B', shot_q=shot_q) if len(C.q_ids(tok, r['q'])) + a.max_new <= 544]
g_ = [r for r in pool_ if r['kind'] == 'gsm']; s_ = [r for r in pool_ if r['kind'] == 'scaled']
rr = _r.Random(0); rr.shuffle(g_); rr2 = _r.Random(0); rr2.shuffle(s_)
sets = {'dev': g_[:600], 'dev2': g_[600:900], 'hard': s_[:300]}


def mark(t, w, f2):
    s = ''; on = 0
    for k, (x, f) in enumerate(zip(t, w)):
        c = 2 if f2[k] else (1 if f & 1 else 0)
        if c != on:
            s += {1: '⟧', 2: '»'}.get(on, ''); s += {1: '⟦', 2: '«'}.get(c, ''); on = c
        s += tok.decode([x])
    s += {1: '⟧', 2: '»'}.get(on, '')
    return s.replace('<|endoftext|>', '')


@torch.no_grad()
def chain_flags(qids, toks):
    """teacher-forced chain2 output over prefix tail + question + answer: flag[k] = chain2 wrote answer token k."""
    pre = eng.prefix_ids.tolist(); KW = ch.K + ch.W
    out = []
    for q, t in zip(qids, toks):
        seq = (pre + q)[-(KW + len(q)):] + t
        e = ch.M.emit(eng.W['E'][torch.tensor(seq, device='cuda')][None])[0][0]
        nz = (e.abs().sum(-1) > 0).tolist()
        base = len(seq) - len(t)            # answer token k is produced from position base + k - 1
        out.append([bool(nz[base + k - 1]) for k in range(len(t))])
    return out


summ = {}
for nm in SETS:
    items = sets[nm][:NLIM] if NLIM else sets[nm]
    qids = [C.q_ids(tok, r['q']) for r in items]
    res = {}
    for var in VLIST:
        im = VARS[var]
        eng.imp = im; eng.graphs = {}; eng.pool = None
        t0 = _t.time(); o = eng.generate(qids, G=1, greedy=True, max_new=a.max_new); dt = _t.time() - t0
        fl = chain_flags(qids, o['tokens'])
        rows = []
        for r, t, w, f2 in zip(items, o['tokens'], o['wflag'], fl):
            sc = C.score(tok, t, r['gold'])
            rows.append(dict(q=r['q'], gold=r['gold'], ok=sc[0], ntok=len(t), chain_tok=sum(f2), tokens=t, txt=mark(t, w, f2)))
        res[var] = rows
        with open(f'{OUT}/{nm}_{var.replace("+", "_")}.jsonl', 'w') as f:
            for x in rows: f.write(_j.dumps(x) + '\n')
        print('C3', nm, var, 'acc', round(100 * sum(x['ok'] for x in rows) / len(rows), 2), 'sec', round(dt, 1), flush=True)
    b = res['calc']; n = len(b)
    for var in VLIST[1:]:
        v = res[var]
        fired = [k for k in range(n) if v[k]['chain_tok'] > 0] if 'chain2' in var else [k for k in range(n) if v[k]['tokens'] != b[k]['tokens']]
        sil = [k for k in range(n) if k not in fired]
        rec = dict(n=n, acc_calc=round(100 * sum(x['ok'] for x in b) / n, 2), acc=round(100 * sum(x['ok'] for x in v) / n, 2),
                   fixes=sum(1 for k in range(n) if v[k]['ok'] > b[k]['ok']), breaks=sum(1 for k in range(n) if v[k]['ok'] < b[k]['ok']),
                   fire_rows=len(fired), fire_rate=round(100 * len(fired) / n, 2), chain_tokens=sum(x['chain_tok'] for x in v),
                   silent_identical=f"{sum(1 for k in sil if v[k]['tokens'] == b[k]['tokens'])}/{len(sil)}",
                   extra_tok=sum(v[k]['ntok'] - b[k]['ntok'] for k in range(n)))
        summ[f'{nm}/{var}'] = rec; print('C3SUM', nm, var, _j.dumps(rec), flush=True)
    _j.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1)
print('C2DONE', flush=True)
