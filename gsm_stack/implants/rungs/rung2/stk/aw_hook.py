# lane aw-base: the ALL-WEIGHTS serial stack (comp_serial.SerialWD, no RoPE position reset), built exactly as
# gsm_stack/eval/trees/v42/stk/fe_hook.py (= rl_hook.py's dev-eval path: DualImplant calc_s4 + chain2, fastroll, T=1),
# rows written in gsm_stack/implants/gen/out format so atlas/ and voter/ tools work.
# exec'd in tw_dump_lean.py --eval_ckpt ckpt_a2s2 --arm 2 --fused 0.
# env: AW_ADAPT spec 'A=0.7/16/4:S=4/0.5:B=3:C=400' (aw_adapt.py; needs SW_NS>=Nmax, SW_MAXN, SAN_SMAX), AW_OUT dir, AW_SETS (dev,dev2), AW_SEEDS (7,11,13,17), AW_N (8), AW_R (1 repair/attempt), AW_VM (1 probe vote),
#      AW_SAN (1), AW_RM (0), AW_PR (0: no reset; 1 = programmatic reset, reference only), AW_NLIM, AW_TAG.
import json as _j, os as _o, sys as _s, time as _t, random as _r, re as _re, gzip as _gz
_s.path.insert(0, _o.environ['STK_DIR']); _o.environ.setdefault('SAN_DIR', _o.environ['STK_DIR'])
import engine as EM
E = _o.environ
OUT = E['AW_OUT']; _o.makedirs(OUT, exist_ok=True)
SETS = E.get('AW_SETS', 'dev,dev2').split(','); SEEDS = [int(x) for x in E.get('AW_SEEDS', '7,11,13,17').split(',')]
N = int(E.get('AW_N', 8)); R_ = int(E.get('AW_R', 1)); PR = E.get('AW_PR', '0') == '1'
VM = int(E.get('AW_VM', 1)); SAN = E.get('AW_SAN', '1') == '1'; RM = E.get('AW_RM', '0') == '1'
ACAP = a.max_new
assert E.get('C2_NOCOMPACT') == '1'


def P(*x): print('AW', *x, flush=True)


import chain2
chain2.register(tok)
from implant import CalcImplant
from implant2 import DualImplant
from transformers import AutoModelForCausalLM
_hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to('cuda').eval()
eng.imp = DualImplant(imp, CalcImplant(_hf, tok, a.B_max, machine='chain2', fused=False))
del _hf; torch.cuda.empty_cache()
import comp_serial as CSR
eng.__class__ = CSR.SerialWD
eng.c2_setup(); eng.set_c2('off')
eng.sw_setup(N=N, acap=ACAP, R=R_, posreset=PR, probe=f"{E['STK_DIR']}/verifier_probe_L18.pt",
             cache=f"{E['STK_DIR']}/dfa_tables.pt", vmode=VM, san=SAN, repmiss=RM)
import aw_adapt as AWA
SPEC = E.get('AW_ADAPT', '')
AWA.setup(eng, SPEC)                       # A/S/B/C knobs (empty: rung-0 stack untouched)
P('adapt', SPEC or 'off', 'NS', CSR.NS, 'acap', eng.acap, 'maxn', eng.sw_maxn, 'S_max', eng.S_max)
GEN_MAX = eng.sw_maxn
eng.graphs = {}; eng.pool = None; eng._outshape = None
if E.get('RL_FASTROLL', '0') == '1':
    import fastroll as _FR
    _FR.enable(eng)
REP0 = tok(CSR.REP_PH, add_special_tokens=False)['input_ids'][0]


def items_of(nm):
    f = lambda it: [r for r in it if len(C.q_ids(tok, r['q'])) + ACAP <= 544]
    it = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(0).shuffle(it)
    return {'dev': it[:600], 'dev2': it[600:900], 'dev3': it[900:]}[nm]


def ends_of(t, w):
    """token index of the last token of each attempt: the token before every non-repair queue start (wf bit 8)."""
    return [i - 1 for i, v in enumerate(w) if v & 8 and int(t[i]) != REP0]


if E.get('AW_PRECAP', '0') == '1':
    # harness-only (default off): the engine's graph warm-up (2 eager steps per new (B, S) graph) advances eng.seed, which is
    # not restored, so the FIRST run in a process samples a shifted RNG stream wherever it captures a new bucket. Capture every
    # bucket up front (after a 1-item warm generate for lazy allocations) so each run, first or later, samples the same stream:
    # then a run in a fresh process must be bitwise equal to the same seed run after others (state-leak test).
    _q0 = [C.q_ids(tok, items_of(SETS[0])[0]['q'])]
    eng.generate(_q0, G=1, T=1.0, greedy=False, max_new=GEN_MAX)
    for _b in [b for b in EM.B_BUCKETS if b <= eng.B_max]: eng._graph(_b, eng.S_max)
    P('precap', sorted(eng.graphs))


for SET in SETS:
    items = items_of(SET)
    if int(E.get('AW_NLIM', 0)): items = items[:int(E['AW_NLIM'])]
    qids = [C.q_ids(tok, r['q']) for r in items]
    assert max(map(len, qids)) + GEN_MAX <= eng.S_max, (GEN_MAX, eng.S_max)
    for SEED in SEEDS:
        torch.manual_seed(SEED); eng.seed.fill_(SEED)
        t0 = _t.time()
        o = eng.generate(qids, G=1, T=1.0, greedy=False, max_new=GEN_MAX)
        dt = _t.time() - t0
        Rg = o['reg']; rows = []
        for j, (t, w, lpp) in enumerate(zip(o['tokens'], o['wflag'], o['logp'])):
            t = [int(x) for x in t]; w = [int(x) for x in w]
            txt = tok.decode(t).replace('<|endoftext|>', '')
            ms = list(_re.finditer(C.NUM, txt)); last = ms[-1].group(1).replace(',', '') if ms else None
            val = [(CSR.SerialW._val(Rg['sym'][j, i].tolist()) or 0.0) if bool(Rg['ok'][j, i]) else 0.0 for i in range(CSR.NS)]
            en = ends_of(t, w); en += [-1] * (CSR.NS - len(en))
            rows.append(dict(tokens=t, wf=w, logp=[round(float(v), 4) for v in lpp], c2ev=[], idx=j, q=items[j]['q'],
                             gold=items[j]['gold'], ok=int(C.correct(last, items[j]['gold'])), last=last, ntok=len(t),
                             natt=int(Rg['natt'][j]), nseg=int(Rg['nseg'][j]), pick=int(Rg['win'][j]), c2fires=0,
                             nrep=int(Rg['nrep'][j]), val=val, rok=Rg['ok'][j].tolist(), bad=Rg['bad'][j].tolist(),
                             lg=Rg['lg'][j].tolist(), end=en[:CSR.NS], txt=txt))
        sp = ('_' + SPEC.replace('/', '-').replace(':', '_').replace('=', '')) if SPEC else ''
        tag = f"{SET}_N{N}_R{R_}_v{VM}_s{int(SAN)}_m{int(RM)}_pr{int(PR)}{'_wpr' if E.get('WPR') == '1' else ''}{sp}_T1.0_seed{SEED}{E.get('AW_TAG', '')}"
        with _gz.open(f'{OUT}/{tag}.jsonl.gz', 'wt') as fh:
            for x in rows: fh.write(_j.dumps(x) + '\n')
        n = len(rows)
        P('done', tag, f"n={n} acc={100*sum(x['ok'] for x in rows)/n:.2f} tok/q={sum(x['ntok'] for x in rows)/n:.1f}",
          f"natt={sum(x['natt'] for x in rows)/n:.2f} rep/q={sum(x['nrep'] for x in rows)/n:.2f} sec={dt:.0f}")
print('AWDONE', flush=True)
