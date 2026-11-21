# RL_STACK arm B (S2F control, one attempt) on a test set: the same compiled stack rl_hook.py builds for ARM == 'B'.
# exec'd in tw_dump_lean.py --eval_ckpt ckpt_a2s2 --host_ckpt rl/B/step_000300 --arm 2 --fused 0 (C2_NOCOMPACT=1).
# env: STK_OUT, STK_SET (test | dev | dev2), STK_N (0 = all), RLB_MODE (T1 | greedy), RLB_SEED (7).
import json as _j, os as _o, sys as _s, time as _t, random as _r, re as _re
_s.path.insert(0, _o.environ['STK_DIR']); _o.environ.setdefault('SAN_DIR', _o.environ['STK_DIR'])
OUT = _o.environ['STK_OUT']; _o.makedirs(OUT, exist_ok=True)
SET = _o.environ['STK_SET']; NLIM = int(_o.environ.get('STK_N', 0)); ACAP = a.max_new
MODE = _o.environ.get('RLB_MODE', 'T1'); SEED = int(_o.environ.get('RLB_SEED', 7))
import chain2; chain2.register(tok)
from implant import CalcImplant
from implant2 import DualImplant
from transformers import AutoModelForCausalLM
_hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to('cuda').eval()
eng.imp = DualImplant(imp, CalcImplant(_hf, tok, a.B_max, machine='chain2', fused=False)); del _hf; torch.cuda.empty_cache()
import stackh as SH
eng.__class__ = SH.StackHD
eng.c2_setup(); eng.set_c2('off')
eng.comp_setup('fwntf', maxf=2, cap0=ACAP, extra=ACAP, cache=f"{_o.environ['STK_DIR']}/dfa_tables.pt")
eng.stack_setup()
eng.set_comp(True, forget=True); eng.m_on = False
GEN_MAX = ACAP + 2 * ACAP
eng.graphs = {}; eng.pool = None; eng._outshape = None
assert _o.environ.get('C2_NOCOMPACT') == '1'
f = lambda it: [r for r in it if len(C.q_ids(tok, r['q'])) + ACAP <= 544]
if SET in ('dev', 'dev2'):
    it = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(0).shuffle(it); items = it[:600] if SET == 'dev' else it[600:900]
else: items = f(C.load_test(SET, shot_q=shot_q))
if NLIM: items = items[:NLIM]
qids = [C.q_ids(tok, r['q']) for r in items]
torch.manual_seed(SEED); eng.seed.fill_(SEED) if hasattr(eng, 'seed') else None
t0 = _t.time()
o = eng.generate(qids, G=1, T=1.0, greedy=MODE == 'greedy', max_new=GEN_MAX)
dt = _t.time() - t0
rows = []
for j, (t, w) in enumerate(zip(o['tokens'], o['wflag'])):
    txt = tok.decode(t).replace('<|endoftext|>', '')
    ms = list(_re.finditer(C.NUM, txt)); la = ms[-1].group(1).replace(',', '') if ms else None
    rows.append(dict(gold=items[j]['gold'], ok=int(C.correct(la, items[j]['gold'])), last=la, ntok=len(t), tokens=t,
                     fire=int(any(x & 8 for x in w))))
tag = f"B_{MODE}_s{SEED}" + _o.environ.get('STK_TAG', '')
with open(f'{OUT}/{SET}_{tag}.jsonl', 'w') as fh:
    for x in rows: fh.write(_j.dumps(x) + '\n')
n = len(rows)
print('RLB', SET, tag, f"n={n} acc={100*sum(x['ok'] for x in rows)/n:.2f} tok/q={sum(x['ntok'] for x in rows)/n:.1f} "
      f"fires={100*sum(x['fire'] for x in rows)/n:.1f} sec={dt:.0f}", flush=True)
print('STKDONE', flush=True)
