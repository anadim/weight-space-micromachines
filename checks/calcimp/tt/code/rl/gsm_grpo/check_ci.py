"""[38] arm I checks on the base host (calc prefix): (1) bit identity with arm X (text tool) up to the first '</calc>'
(the wiring is silent until a call fires), sampled and greedy; (2) the exact core vs calc_eval on every live call;
(3) written-symbol fidelity (host's sample == the symbol the wiring writes); (4) hand-direction report.
python check_ci.py  (CHK_OUT=path.json)"""
import os, sys, json, torch
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q, armi as AI
from engine import Engine
from tooltext import calc_prefix_ids, calc_eval, CLOSE
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, shot_q = calc_prefix_ids(tok)
W = Q.fuse(Q.load_hf_params(dtype=torch.bfloat16))
sym_ids = AI.sym_token_ids(tok)
W0, rep = AI.calib_W0(Q, W, pre, sym_ids, 'cuda')
items = C.load_test('test', shot_q=shot_q)[:256]
qids = [C.q_ids(tok, r['q']) for r in items]
res = dict(hand=rep)
close_ids = None


def upto_close(ids):
    """index of the first token whose decoded prefix contains '</calc>' (inclusive), else len."""
    for k in range(len(ids)):
        if CLOSE in tok.decode(ids[:k + 1]): return k + 1
    return len(ids)


for arm in ('I0', 'Itok', 'I'):
    ci = AI.ArmI(arm, W0, sym_ids)
    S = 320 + 128 + max(map(len, qids)) + 64
    eX = Engine(W, pre, tok, B_max=256, S_max=S, tool=True, sync_every=4, cap_extra=128)
    eI = Engine(W, pre, tok, B_max=256, S_max=S, tool=True, sync_every=4, cap_extra=128, ci=ci)
    for greedy in (False, True):
        oo = {}
        for nm, e in (('X', eX), ('I', eI)):
            torch.manual_seed(7); e.seed.fill_(7 * 1000003); e.tool_log = []
            oo[nm] = (e.generate(qids, G=1, T=1.0, max_new=320, greedy=greedy), list(e.tool_log))
        (oX, lX), (oI, lI) = oo['X'], oo['I']
        same_pre = same_full = 0; first_div = []
        for a_, b_ in zip(oX['tokens'], oI['tokens']):
            k = upto_close(a_)
            same_pre += a_[:k] == b_[:k]; same_full += a_ == b_
            if a_[:k] != b_[:k]: first_div.append(next(i for i in range(min(len(a_), len(b_))) if a_[i] != b_[i]) if a_[:min(len(a_), len(b_))] != b_[:min(len(a_), len(b_))] else -1)
        # core vs calc_eval on every call of the I run
        agree = sum(1 for e in lI if calc_eval(e[0]) == e[1])
        bad = [(e[0], calc_eval(e[0]), e[1]) for e in lI if calc_eval(e[0]) != e[1]][:20]
        # written fidelity: token sampled at flag-2 positions == the queued symbol's token
        wr = ok = 0
        for t, w, s in zip(oI['tokens'], oI['wflag'], oI['sym']):
            for x, f, sy in zip(t, w, s):
                if f == 2: wr += 1; ok += x == sym_ids[sy]
        acc = lambda o: 100 * sum(C.score(tok, t, r['gold'], 'first')[0] for t, r in zip(o['tokens'], items)) / len(items)
        key = f'{arm}_{"greedy" if greedy else "T1"}'
        res[key] = dict(n=len(items), same_upto_first_close=same_pre, same_full=same_full, first_div=first_div[:20],
                        calls_X=len(lX), calls_I=len(lI), admitted=sum(1 for e in lI if e[2]), core_agree=agree, core_bad=bad,
                        written=wr, written_ok=ok, acc_X=acc(oX), acc_I=acc(oI),
                        stats_I={k: v for k, v in oI['stats'].items() if isinstance(v, (int, float))})
        print(key, json.dumps(res[key]), flush=True)
    del eX, eI; torch.cuda.empty_cache()
json.dump(res, open(os.environ.get('CHK_OUT', 'check_ci.json'), 'w'), indent=1)
print('DONE', flush=True)
