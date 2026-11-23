"""[35] evaluation: arm 1 (plain RL), arm 2 (RL + calculator implant, one attempt) and arm X (RL + text <calc> tool),
GSM8K test / GSM-Hard / GSM-Plus testmini, T = 1, one attempt per question, eval seeds given, max_new 320, same engine.
Per item: correct, parsed, generated tokens (model-sampled vs tool-written), tool calls. Timing: batched wall-clock per
question (B_max rows in flight; for X it includes the pauses, the Python calls and the forced result tokens = re-prefill)
and single-stream latency on the first --lat_n test questions.
python fe_tool.py --ckpt RUN/step_000300|base --arm X --out DIR"""
import os, sys, re, json, time, argparse, torch
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--arm', required=True, choices=['1', '2', 'X'])
ap.add_argument('--out', required=True); ap.add_argument('--sets', default='test,hard,gsmplus')
ap.add_argument('--seeds', default='7,11,13'); ap.add_argument('--B_max', type=int, default=512)
ap.add_argument('--max_new', type=int, default=320); ap.add_argument('--sync_every', type=int, default=16)
ap.add_argument('--lat_n', type=int, default=100); ap.add_argument('--fused', type=int, default=1)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
if a.arm == 'X':
    from tooltext import calc_prefix_ids
    pre, shot_q = calc_prefix_ids(tok)
else:
    pre, shot_q = C.prefix_ids(tok)
sd = Q.load_hf_params(dtype=torch.bfloat16)
if a.ckpt != 'base':
    from safetensors.torch import load_file
    ck = load_file(f'{a.ckpt}/model.safetensors', device='cuda')
    for k in sd: sd[k].copy_(ck[k])
W = Q.fuse(sd)
imp = None
if a.arm == '2':
    from implant import CalcImplant
    hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
    imp = CalcImplant(hf, tok, a.B_max, fused=bool(a.fused))
    del hf


def load(name):
    if name == 'gsmplus':
        rows = [json.loads(l) for l in open(f'{C.REPO}/data/gsmplus/testmini.jsonl')]
        return [dict(i=i, q=r['question'], gold=C.gold_of(r['answer']), ptype=r['ptype']) for i, r in enumerate(rows)
                if r['question'] not in shot_q]
    return C.load_test(name, shot_q=shot_q)


def _ndec(x):
    x = str(x).strip(); return len(x.split('.')[1]) if '.' in x and 'e' not in x.lower() else 0


def correct(pred, gold, round2):
    """common.correct; GSM-Plus also accepts agreement after rounding to the shorter (>= 2) decimal count (final_eval rule)."""
    if C.correct(pred, gold): return True
    if not round2 or pred is None: return False
    from decimal import Decimal, ROUND_HALF_UP
    try:
        d = min(_ndec(pred), _ndec(gold))
        if d < 2: return False
        q = Decimal(1).scaleb(-d)
        return Decimal(str(pred).strip()).quantize(q, ROUND_HALF_UP) == Decimal(str(gold).strip()).quantize(q, ROUND_HALF_UP)
    except Exception:
        return False


def calls_of(wf):
    return sum(1 for i in range(len(wf)) if wf[i] and (i == 0 or not wf[i - 1]))


sets = {s: load(s) for s in a.sets.split(',')}
maxq = max(len(C.q_ids(tok, r['q'])) for v in sets.values() for r in v)
S_max = ((maxq + a.max_new + 63) // 64) * 64
eng = Engine(W, pre, tok, implant=imp, B_max=a.B_max, S_max=S_max, tool=(True if a.arm == 'X' else None), sync_every=a.sync_every)
print(f'arm {a.arm} ckpt {a.ckpt} prefix {len(pre)} S_max {S_max}', flush=True)
summ = dict(arm=a.arm, ckpt=a.ckpt, T=1.0, max_new=a.max_new, B_max=a.B_max, sync_every=a.sync_every, prefix_tokens=len(pre))
# warm-up (graph capture) so timing excludes compilation
eng.generate([C.q_ids(tok, r['q']) for r in sets[list(sets)[0]][:64]], G=1, T=1.0, max_new=a.max_new)
for name, items in sets.items():
    qids = [C.q_ids(tok, r['q']) for r in items]
    for seed in [int(x) for x in a.seeds.split(',')]:
        torch.manual_seed(seed); eng.seed.fill_(seed * 1000003)
        if a.arm == 'X': eng.tool_log = []
        out = eng.generate(qids, G=1, T=1.0, max_new=a.max_new)
        rows = []
        for j, (t, r) in enumerate(zip(out['tokens'], items)):
            txt = tok.decode(t).replace('<|endoftext|>', '').split('\n\n')[0]
            m = re.findall(C.NUM, txt); pred = m[0].replace(',', '') if m else None
            wf = out['wflag'][j]
            ntok = len(t); nforced = int(sum(wf)) if a.arm == 'X' else 0
            calls = out['calls'][j] if a.arm == 'X' else (calls_of(wf) if a.arm == '2' else 0)
            rows.append(dict(i=r['i'], ok=int(correct(pred, r['gold'], name == 'gsmplus')), parsed=int(pred is not None),
                             pred=pred, gold=r['gold'], ntok=ntok, ntok_model=ntok - nforced, nforced=nforced, calls=calls,
                             calls_in_answer=txt.count('</calc>') if a.arm == 'X' else calls, text=txt))
        tag = f'{name}_s{seed}'
        with open(f'{a.out}/{tag}.jsonl', 'w') as f:
            for row in rows: f.write(json.dumps(row) + '\n')
        n = len(rows); st = out['stats']
        S = dict(n=n, acc=100 * sum(x['ok'] for x in rows) / n, fmt=100 * sum(x['parsed'] for x in rows) / n,
                 tok_q=sum(x['ntok'] for x in rows) / n, tok_q_model=sum(x['ntok_model'] for x in rows) / n,
                 calls_q=sum(x['calls'] for x in rows) / n, s_per_q_batched=st['seconds'] / n, seconds=st['seconds'],
                 decode_steps=st['decode_steps'])
        if a.arm == 'X':
            S['tool_err'] = sum(1 for _, res in eng.tool_log if res == 'ERR') / max(1, len(eng.tool_log))
            S['tool_examples'] = eng.tool_log[:20]
        summ[tag] = S
        print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in S.items() if k != 'tool_examples'}, flush=True)
        json.dump(summ, open(f'{a.out}/summary.json', 'w'), indent=1)
if a.lat_n > 0:                                   # single-stream latency (one question in flight), test, seed 7
    items = sets.get('test') or sets[list(sets)[0]]
    torch.manual_seed(7); eng.seed.fill_(7)
    eng.generate([C.q_ids(tok, items[0]['q'])], G=1, T=1.0, max_new=a.max_new)
    ts = []; toks = []
    for r in items[:a.lat_n]:
        torch.cuda.synchronize(); t0 = time.time()
        o = eng.generate([C.q_ids(tok, r['q'])], G=1, T=1.0, max_new=a.max_new)
        torch.cuda.synchronize(); ts.append(time.time() - t0); toks.append(len(o['tokens'][0]))
    summ['latency_B1'] = dict(n=len(ts), s_per_q=sum(ts) / len(ts), tok_q=sum(toks) / len(toks))
    print('latency_B1', summ['latency_B1'], flush=True)
json.dump(summ, open(f'{a.out}/summary.json', 'w'), indent=1)
print('DONE', flush=True)
