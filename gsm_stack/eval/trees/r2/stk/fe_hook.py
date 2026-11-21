# task [6] final evaluation: one config of an RL arm's stack on one item set, built exactly as rl_hook.py builds it
# (DualImplant = calc_s4 + chain2, SerialWD for A/A1/Ar/C or StackHD for B, fastroll loop as in the arms' dev evals).
# exec'd in tw_dump_lean.py --eval_ckpt ckpt_a2s2 [--host_ckpt base | RUN/step_000300] --arm 2 --fused 0.
# env: FE_OUT dir, FE_SET (test | hard | dev | dev2), FE_KIND (S serial | B | ladder rungs H host only, C4 calc_s4, D calc_s4+chain2), FE_N, FE_PR (posreset 0/1), FE_VM (1 probe vote,
#      0 plain majority), FE_SAN (1/0 sanity filter in the vote), FE_RM (1: repair only missing answers), FE_MODE (T1 | greedy),
#      FE_SEED (7), FE_TAG, FE_MISS (B/SC: off | hook | w = missing-answer trigger: none / stop-token swap / compiled), FE_IMP (both | calc | chain2 | none: [14] implant ablation), FE_FORGET (0: forget mask off), FE_COMP (B/SC 0: compiled triggers off). Rows: ana_rltest.py format (serial: sym / rok / bad / lg per slot; B: fire).
import json as _j, os as _o, sys as _s, time as _t, random as _r, re as _re
_s.path.insert(0, _o.environ['STK_DIR']); _o.environ.setdefault('SAN_DIR', _o.environ['STK_DIR'])
import engine as EM
E = _o.environ
OUT = E['FE_OUT']; _o.makedirs(OUT, exist_ok=True)
SET = E['FE_SET']; KIND = E.get('FE_KIND', 'S'); N = int(E.get('FE_N', 8)); PR = E.get('FE_PR', '0') == '1'
VM = int(E.get('FE_VM', 1)); SAN = E.get('FE_SAN', '1') == '1'; RM = E.get('FE_RM', '0') == '1'
MODE = E.get('FE_MODE', 'T1'); SEED = int(E.get('FE_SEED', 7)); ACAP = a.max_new
assert E.get('C2_NOCOMPACT') == '1'


def P(*x): print('FE', *x, flush=True)


import chain2
chain2.register(tok)
from implant import CalcImplant
from implant2 import DualImplant
from transformers import AutoModelForCausalLM
_hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to('cuda').eval()
if KIND != 'H' and KIND != 'C4':
    eng.imp = DualImplant(imp, CalcImplant(_hf, tok, a.B_max, machine='chain2', fused=False))
del _hf; torch.cuda.empty_cache()
FIMP = E.get('FE_IMP', 'both')      # task [14]5 implant ablation: both | calc (calc_s4 only) | chain2 (chain2 only) | none
if KIND not in ('H', 'C4') and FIMP != 'both':
    if FIMP == 'calc': eng.imp = eng.imp.calc
    elif FIMP == 'chain2': eng.imp = eng.imp.chain
    else: assert FIMP == 'none'; eng.imp_on = False
if KIND in ('H', 'C4', 'D'):          # task [8] ladder rungs: plain Engine (stop: eos / blank line / '\n\n' / budget)
    eng.imp_on = KIND != 'H'          # H: host only;  C4: calc_s4 alone;  D: calc_s4 + chain2 (DualImplant)
    GEN_MAX = ACAP
elif KIND in ('B', 'SC'):            # SC (task [10] control): N independent B-stack samples per item, plain majority
    import stackh as SH
    eng.__class__ = SH.StackHD
    eng.c2_setup(); eng.set_c2('off')
    eng.comp_setup('fwntf', maxf=2, cap0=ACAP, extra=ACAP, cache=f"{E['STK_DIR']}/dfa_tables.pt")
    eng.stack_setup()
    eng.set_comp(E.get('FE_COMP', '1') == '1', forget=E.get('FE_FORGET', '1') == '1'); eng.m_on = False   # [14]5 ablations
    MISS = E.get('FE_MISS', 'off')    # [12] item 1: missing-answer trigger off | hook (stop-token swap, programmatic) | w (compiled)
    if MISS != 'off': eng.m_on = True; eng.m_w = MISS == 'w'
    GEN_MAX = ACAP + 2 * ACAP
else:
    import comp_serial as CSR
    eng.__class__ = CSR.SerialWD
    eng.c2_setup(); eng.set_c2('off')
    eng.sw_setup(N=N, acap=ACAP, posreset=PR, probe=f"{E['STK_DIR']}/verifier_probe_L18.pt",
                 cache=f"{E['STK_DIR']}/dfa_tables.pt", vmode=VM, san=SAN, repmiss=RM)
    import aw_adapt as AWA            # [17](2) rung 2: A/S/B/C on the registers (empty AW_ADAPT: no-op, rung-1 path)
    AWA.setup(eng, E.get('AW_ADAPT', ''))
    P('adapt', E.get('AW_ADAPT', '') or 'off', 'NS', CSR.NS, 'acap', eng.acap, 'maxn', eng.sw_maxn, 'S_max', eng.S_max)
    GEN_MAX = eng.sw_maxn
    if E.get('FE_FORGET', '1') == '0':   # task [10] ablation: forget mask off (needs FAST_ATT3=0: attn3 always forgets)
        assert E.get('FAST_ATT3', '0') == '0'
        eng.c_forget = False
eng.graphs = {}; eng.pool = None; eng._outshape = None
if E.get('RL_FASTROLL', '0') == '1' and KIND in ('S', 'B', 'SC'):
    import fastroll as _FR
    _FR.enable(eng)
f = lambda it: [r for r in it if len(C.q_ids(tok, r['q'])) + ACAP <= 544]
if SET in ('dev', 'dev2'):
    it = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(0).shuffle(it); items = it[:600] if SET == 'dev' else it[600:900]
else:
    al = C.load_test(SET, shot_q=shot_q); items = f(al)
    P('set', SET, 'n_all', len(al), 'n_kept', len(items), 'dropped_idx', [r['i'] for r in al if r not in items][:20])
if int(E.get('FE_NLIM', 0)): items = items[:int(E['FE_NLIM'])]
qids = [C.q_ids(tok, r['q']) for r in items]
assert max(map(len, qids)) + GEN_MAX <= eng.S_max, (GEN_MAX, eng.S_max)
torch.manual_seed(SEED); eng.seed.fill_(SEED)
t0 = _t.time()
GS = N if KIND == 'SC' else 1
o = eng.generate(qids, G=GS, T=1.0, greedy=MODE == 'greedy', max_new=GEN_MAX)
dt = _t.time() - t0
rows = []
if KIND == 'SC':
    from collections import Counter
    for j in range(len(items)):
        la_ = []
        for t in o['tokens'][j * GS:(j + 1) * GS]:
            ms = list(_re.finditer(C.NUM, tok.decode(t).replace('<|endoftext|>', '')))
            la_.append(ms[-1].group(1).replace(',', '') if ms else None)
        val = [x for x in la_ if x is not None]
        maj = Counter(val).most_common(1)[0][0] if val else None      # ties: the earliest sample's answer (Counter order)
        rows.append(dict(i=items[j].get('i', j), gold=items[j]['gold'], ok=int(C.correct(maj, items[j]['gold'])), last=maj,
                         ntok=sum(len(t) for t in o['tokens'][j * GS:(j + 1) * GS]), answers=la_,
                         att_ok=[int(C.correct(x, items[j]['gold'])) for x in la_]))
for j, (t, w) in enumerate(zip(o['tokens'], o['wflag']) if KIND != 'SC' else []):
    txt = tok.decode(t).replace('<|endoftext|>', '')
    ms = list(_re.finditer(C.NUM, txt)); la = ms[-1].group(1).replace(',', '') if ms else None
    x = dict(i=items[j].get('i', j), gold=items[j]['gold'], ok=int(C.correct(la, items[j]['gold'])), last=la, ntok=len(t), tokens=t)
    if KIND in ('H', 'C4', 'D'):
        x['nw'] = int(sum(1 for v in w if v & 1))
    elif KIND in ('B', 'SC'):
        x['fire'] = int(any(v & 8 for v in w))
    else:
        R = o['reg']
        x.update(natt=int(R['natt'][j]), nseg=int(R['nseg'][j]), win=int(R['win'][j]), sym=R['sym'][j].tolist(),
                 rok=R['ok'][j].tolist(), bad=R['bad'][j].tolist(), lg=R['lg'][j].tolist())
    rows.append(x)
tag = E.get('FE_TAG') or (f"{KIND}_{MODE}_s{SEED}" if KIND != 'S' else f"N{N}_pr{int(PR)}_{MODE}_v{VM}_s{int(SAN)}_m{int(RM)}_s{SEED}")
with open(f'{OUT}/{SET}_{tag}.jsonl', 'w') as fh:
    for x in rows: fh.write(_j.dumps(x) + '\n')
n = len(rows)
P('done', SET, tag, f"n={n} acc={100*sum(x['ok'] for x in rows)/n:.2f} tok/q={sum(x['ntok'] for x in rows)/n:.1f} sec={dt:.0f}")
print('FEDONE', flush=True)
