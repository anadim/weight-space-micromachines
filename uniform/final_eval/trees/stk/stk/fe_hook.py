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
SET = E['FE_SET']; KIND = E.get('FE_KIND', 'S'); KB = 'H' if KIND == 'SCH' else KIND   # SCH ([19]b): N independent host-only samples + plain majority
SCK = KIND in ('SC', 'SCH'); N = int(E.get('FE_N', 8)); PR = E.get('FE_PR', '0') == '1'
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
if KB != 'H' and KB != 'C4':
    eng.imp = DualImplant(imp, CalcImplant(_hf, tok, a.B_max, machine='chain2', fused=False))
del _hf; torch.cuda.empty_cache()
FIMP = E.get('FE_IMP', 'both')      # task [14]5 implant ablation: both | calc (calc_s4 only) | chain2 (chain2 only) | none
if KB not in ('H', 'C4') and FIMP != 'both':
    if FIMP == 'calc': eng.imp = eng.imp.calc
    elif FIMP == 'chain2': eng.imp = eng.imp.chain
    else: assert FIMP == 'none'; eng.imp_on = False
STALE = E.get('FE_STALE', '0') == '1'   # task [20] R3: stack setup on the u0 host weights (as in an RL process), then the host ckpt
if STALE:
    assert a.host_ckpt and a.host_ckpt != 'base'
    _sd0 = load_file(f"{a.eval_ckpt}/model.safetensors", device="cpu")
    with torch.no_grad():
        for k in names: P32[k].copy_(_sd0[k].float())
    sync_weights(); del _sd0
    P('stale: stack setup on', a.eval_ckpt)
if KB in ('H', 'C4', 'D'):          # task [8] ladder rungs: plain Engine (stop: eos / blank line / '\n\n' / budget)
    eng.imp_on = KB != 'H'          # H: host only;  C4: calc_s4 alone;  D: calc_s4 + chain2 (DualImplant)
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
    GEN_MAX = eng.sw_maxn
    if E.get('FE_FORGET', '1') == '0':   # task [10] ablation: forget mask off (needs FAST_ATT3=0: attn3 always forgets)
        assert E.get('FAST_ATT3', '0') == '0'
        eng.c_forget = False
if STALE:
    _sd1 = load_file(f"{a.host_ckpt}/model.safetensors", device="cpu")
    with torch.no_grad():
        for k in names: P32[k].copy_(_sd1[k].float())
    sync_weights(); del _sd1
    P('stale: host weights now', a.host_ckpt)
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
if int(E.get('FE_WARM', 0)) and KIND == 'S':   # task [20] R3: process history as in an RL run (training-shaped rollouts first)
    _wi = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(1).shuffle(_wi)
    _WD = E.get('FE_WDBG', '0') == '1'           # R3 debug: which engine state does the warm-up leave behind
    if _WD:
        def _walk(o, pre, d, depth):
            for k, v in list(vars(o).items()):
                if k in ('W', 'graphs', 'pool') or k.startswith('__'): continue
                nm = pre + k
                if isinstance(v, torch.Tensor): d.setdefault(id(v), (nm, v))
                elif isinstance(v, (list, tuple)) and v and all(isinstance(x, torch.Tensor) for x in v):
                    for i, x in enumerate(v): d.setdefault(id(x), (f'{nm}[{i}]', x))
                elif isinstance(v, dict) and v and all(isinstance(x, torch.Tensor) for x in v.values()):
                    for i, x in v.items(): d.setdefault(id(x), (f'{nm}[{i}]', x))
                elif depth < 2 and hasattr(v, '__dict__') and not isinstance(v, type) and type(v).__module__ not in ('builtins', 'torch', 'transformers'):
                    try: _walk(v, nm + '.', d, depth + 1)
                    except TypeError: pass
        _TS = {}; _walk(eng, '', _TS, 0)
        _SN = {}; _big = {}
        for i_, (nm, v) in _TS.items():
            if v.numel() * v.element_size() <= 4e8: _SN[i_] = v.detach().to('cpu', copy=True)
            else: _big[i_] = float(v.detach().double().sum())
        _PY = {k: v for k, v in vars(eng).items() if isinstance(v, (int, float, bool, str))}
        P('wdbg: snapshot', len(_SN), 'tensors,', len(_big), 'big (checksum only)')
    for _w in range(int(E['FE_WARM'])):
        eng.set_sw(N=4); torch.manual_seed(100 + _w); eng.seed.fill_(100 + _w)
        eng.generate([C.q_ids(tok, r['q']) for r in _wi[_w * 64:(_w + 1) * 64]], G=8, T=1.0, greedy=False, max_new=GEN_MAX)
    eng.set_sw(N=N); P('warm: ran', E['FE_WARM'], 'rollouts of 64 x 8 at N=4 before the eval')
    if _WD:
        _df = []
        for i_, (nm, v) in _TS.items():
            if i_ in _SN:
                a_ = _SN[i_]; b_ = v.detach().cpu()
                if a_.shape != b_.shape or not torch.equal(a_, b_):
                    nd = int((a_ != b_).sum()) if a_.shape == b_.shape else -1
                    _df.append(nm); P('wdbg diff', nm, tuple(v.shape), v.dtype, 'n_neq', nd)
            elif float(v.detach().double().sum()) != _big[i_]: _df.append(nm); P('wdbg diff BIG', nm, tuple(v.shape), v.dtype)
        for k, v in _PY.items():
            if getattr(eng, k, None) != v: P('wdbg diff py', k, v, '->', getattr(eng, k, None))
        _RS = E.get('FE_RESTORE', '')               # all | comma list of name prefixes | -prefixes (all but these)
        if _RS:
            neg = _RS.startswith('-'); pref = [x for x in _RS.lstrip('-').split(',') if x and x != 'all']
            nr = 0
            for i_, (nm, v) in _TS.items():
                if i_ not in _SN: continue
                hit = any(nm.startswith(p_) for p_ in pref)
                if (_RS == 'all') or (hit != neg):
                    with torch.no_grad(): v.copy_(_SN[i_].to(v.device)); nr += 1
            P('wdbg: restored', nr, 'tensors (', _RS, ')')
        if E.get('FE_GRAPHRESET', '0') == '1':
            eng.graphs = {}; eng.pool = None; eng._outshape = None; P('wdbg: graphs reset')
if E.get('FE_SEQ') and KIND == 'S':   # R3 debug: E=eval T1, G=eval greedy, R=drop graphs, W4/W8=2 warm rollouts at N
    import hashlib as _hl
    _wi2 = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(1).shuffle(_wi2)
    _ref = {}
    for st_ in E['FE_SEQ'].split(','):
        if st_ == 'R': eng.graphs = {}; eng.pool = None; eng._outshape = None; P('seq R: graphs dropped'); continue
        if st_.startswith('W'):
            for _w in range(2):
                eng.set_sw(N=int(st_[1:])); torch.manual_seed(100 + _w); eng.seed.fill_(100 + _w)
                eng.generate([C.q_ids(tok, r['q']) for r in _wi2[_w * 64:(_w + 1) * 64]], G=8, T=1.0, greedy=False, max_new=GEN_MAX)
            eng.set_sw(N=N); P('seq', st_, ': 2 warm rollouts at N', st_[1:], 'graphs now', sorted(eng.graphs)); continue
        gr = st_ == 'G'
        torch.manual_seed(SEED); eng.seed.fill_(SEED)
        o_ = eng.generate(qids, G=1, T=1.0, greedy=gr, max_new=GEN_MAX)
        ok_ = []
        for j, t in enumerate(o_['tokens']):
            ms = list(_re.finditer(C.NUM, tok.decode(t).replace('<|endoftext|>', ''))); la = ms[-1].group(1).replace(',', '') if ms else None
            ok_.append(int(C.correct(la, items[j]['gold'])))
        h_ = _hl.md5(_j.dumps(o_['tokens']).encode()).hexdigest()[:12]
        nd = sum(a != b for a, b in zip(_ref[st_], o_['tokens'])) if st_ in _ref else None
        _ref.setdefault(st_, o_['tokens'])
        P('seq', st_, f"acc={100*sum(ok_)/len(ok_):.2f} md5={h_} rows_differing_from_first_{st_}={nd} seed_after={int(eng.seed)} steps={o_['stats']['decode_steps']} graphs={sorted(eng.graphs)}")
    raise SystemExit(0)
torch.manual_seed(SEED); eng.seed.fill_(SEED)
t0 = _t.time()
GS = N if SCK else 1
o = eng.generate(qids, G=GS, T=1.0, greedy=MODE == 'greedy', max_new=GEN_MAX)
dt = _t.time() - t0
rows = []
if SCK:
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
    if E.get('FE_SCV', '0') == '1':      # task [19](a): the rung-2 voter applied offline to the N samples (scvote.py)
        import scvote
        scvote.run(eng, tok, C, items, qids, o, GS, rows, E['STK_DIR'], P)
for j, (t, w) in enumerate(zip(o['tokens'], o['wflag']) if not SCK else []):
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
