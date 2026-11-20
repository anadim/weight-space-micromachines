# compiled serial stack (comp_serial.SerialWD) on dev / dev2: equivalence + no-reset measurements. exec'd in tw_dump_lean.
import json as _j, os as _o, sys as _s, time as _t, random as _r, re as _re
_s.path.insert(0, _o.environ['STK_DIR']); _o.environ.setdefault('SAN_DIR', _o.environ['STK_DIR'])
OUT = _o.environ['STK_OUT']; _o.makedirs(OUT, exist_ok=True)
SET = _o.environ['STK_SET']; NLIM = int(_o.environ.get('STK_N', 0)); ACAP = a.max_new
if _o.environ.get('SW_CHAIN2') == '1':
    import chain2; chain2.register(tok)
    from implant import CalcImplant
    from implant2 import DualImplant
    from transformers import AutoModelForCausalLM
    _hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to('cuda').eval()
    eng.imp = DualImplant(imp, CalcImplant(_hf, tok, a.B_max, machine='chain2', fused=False)); del _hf; torch.cuda.empty_cache()
import comp_serial as CSR
f = lambda it: [r for r in it if len(C.q_ids(tok, r['q'])) + ACAP <= 544]
if SET in ('dev', 'dev2'):
    it = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(0).shuffle(it); items = it[:600] if SET == 'dev' else it[600:900]
else: items = f(C.load_test(SET, shot_q=shot_q))
if NLIM: items = items[:NLIM]
qids = [C.q_ids(tok, r['q']) for r in items]
eng.__class__ = CSR.SerialWD
eng.c2_setup(); eng.set_c2('off')
for job in _o.environ['SW_JOBS'].split(';'):            # N:posreset:mode:vmode:san:repmiss
    N, pr, mode, vm, sn, rm = job.split(':'); N = int(N)
    eng.sw_setup(N=N, acap=ACAP, posreset=pr == '1', probe=f"{_o.environ['STK_DIR']}/verifier_probe_L18.pt",
                 cache=_o.environ.get('DFA_CACHE', f"{_o.environ['STK_DIR']}/dfa_tables.pt"), vmode=int(vm), san=sn == '1', repmiss=rm == '1')
    torch.manual_seed(7); eng.seed.fill_(7)
    t0 = _t.time()
    o = eng.generate(qids, G=1, T=1.0, greedy=mode == 'greedy', max_new=eng.sw_maxn)
    dt = _t.time() - t0
    rows = []
    for j, (t, w) in enumerate(zip(o['tokens'], o['wflag'])):
        txt = tok.decode(t).replace('<|endoftext|>', '')
        ms = list(_re.finditer(C.NUM, txt)); la = ms[-1].group(1).replace(',', '') if ms else None
        R = o['reg']
        rows.append(dict(gold=items[j]['gold'], ok=int(C.correct(la, items[j]['gold'])), last=la, ntok=len(t), tokens=t,
                         natt=int(R['natt'][j]), nseg=int(R['nseg'][j]), win=int(R['win'][j]),
                         sym=R['sym'][j].tolist(), rok=R['ok'][j].tolist(), bad=R['bad'][j].tolist(), lg=R['lg'][j].tolist()))
    tag = f"N{N}_pr{pr}_{mode}_v{vm}_s{sn}_m{rm}" + _o.environ.get('STK_TAG', '')
    with open(f'{OUT}/{SET}_{tag}.jsonl', 'w') as fh:
        for x in rows: fh.write(_j.dumps(x) + '\n')
    def rollout(qd, greedy=False, Gr=1, seed=None):
        if seed is not None: torch.manual_seed(seed); eng.seed.fill_(seed)
        return eng.generate(qd, G=Gr, T=1.0, greedy=greedy, max_new=eng.sw_maxn)
    n = len(rows)
    print('SW', SET, tag, f"acc={100*sum(x['ok'] for x in rows)/n:.2f} tok/q={sum(x['ntok'] for x in rows)/n:.1f} "
          f"natt={sum(x['natt'] for x in rows)/n:.2f} reps={sum(x['nseg']-x['natt'] for x in rows)} nowin={sum(x['win']<0 for x in rows)} sec={dt:.0f}", flush=True)
print('SWDONE', flush=True)
