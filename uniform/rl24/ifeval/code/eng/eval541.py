"""IFEval (541) / dev generation on the fast engine for the 4-row table: greedy, max_new 2048 (the full14 harness), same prompt
format for every row. Writes run jsonl files (key, response, finish, ntok, circ) for score_reduce.py, and the idle
bit-identity check (implant attached but silent == host alone).

python eval541.py --model M [--ckpt DIR/model.safetensors] [--sd SD.pt] --modes imp,plain --out PREFIX [--data ifeval.jsonl]
  modes: imp (implant attached; needs --sd), plain (host alone)
  --idle 1 (needs both modes): the prompts on which the implant never wrote in the 'imp' run are generated again as their own
  batch with and without the implant (identical schedules as long as the implant stays silent) -> token identity per prompt.
"""
import argparse, json, os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'stk'), os.path.join(HERE, '..', 'f14')]
import fastqwen as Q
from transformers import AutoTokenizer
from ifengine import IFEngine, STOPS
import common_if as CI

ap = argparse.ArgumentParser(); ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base'); ap.add_argument('--ckpt', default=None)
ap.add_argument('--sd', default=None); ap.add_argument('--modes', default='imp,plain'); ap.add_argument('--out', required=True)
ap.add_argument('--data', default=os.path.join(HERE, '..', 'ifdata', 'ifeval.jsonl')); ap.add_argument('--max_new', type=int, default=2048)
ap.add_argument('--fmt', default='chat'); ap.add_argument('--B_max', type=int, default=256); ap.add_argument('--idle', type=int, default=1)
ap.add_argument('--n', type=int, default=0); ap.add_argument('--mem_frac', type=float, default=0)
a = ap.parse_args()
if a.mem_frac: torch.cuda.set_per_process_memory_fraction(a.mem_frac)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
modes = a.modes.split(',')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
docs = [json.loads(l) for l in open(a.data)]
if a.n: docs = docs[:a.n]
pre, full = CI.format_prompts(tok, [d['prompt'] for d in docs], a.fmt)
qids = [f[len(pre):] for f in full]
P32 = Q.load_hf_params(a.model)
if a.ckpt:
    from safetensors.torch import load_file
    ck = load_file(a.ckpt)
    assert set(ck) == set(P32), (set(ck) ^ set(P32))
    for k in P32: P32[k] = ck[k].to(P32[k].device, torch.float32)
W = Q.fuse(P32, torch.bfloat16); del P32
imp = None
if 'imp' in modes:
    from gexec import GExec
    imp = GExec(torch.load(a.sd))
eng = IFEngine(W, tok, pre, imp=imp, B_max=a.B_max, S_max=max(map(len, qids)) + a.max_new + 8, N_cap=len(docs),
               max_new_cap=a.max_new)
V = eng.V
summ = dict(model=a.model, ckpt=a.ckpt, sd=a.sd, max_new=a.max_new, fmt=a.fmt, n=len(docs))


def run(on, idx):
    eng.imp_on = on
    o = eng.generate([qids[i] for i in idx], G=1, max_new=a.max_new, greedy=True, full_ids=[full[i] for i in idx])
    return o


def dump(o, idx, path):
    with open(path, 'w') as f:
        for t, w, i in zip(o['tokens'], o['widx'], idx):
            fin = t[-1] in STOPS; body = t[:-1] if fin else t
            f.write(json.dumps({'key': docs[i]['key'], 'response': tok.decode(body).strip(), 'finish': 'stop' if fin else 'length',
                                'ntok': len(body), 'circ': {'nwrites': sum(x != V for x in w), 'fired': int(any(x != V for x in w))}}) + '\n')


res = {}
for m in modes:
    idx = list(range(len(docs)))
    o = run(m == 'imp', idx); res[m] = o
    dump(o, idx, f'{a.out}.{m}.jsonl')
    summ[m] = dict(o['stats'], tok_q=sum(len(t) - (t[-1] in STOPS) for t in o['tokens']) / len(idx),
                   fired=sum(any(x != V for x in w) for w in o['widx']))
    print(m, json.dumps(summ[m]), flush=True)
if a.idle and 'imp' in res and 'plain' in res:
    idle = [i for i, w in enumerate(res['imp']['widx']) if all(x == V for x in w)]
    oi = run(True, idle); op = run(False, idle)
    still = [j for j, w in enumerate(oi['widx']) if all(x == V for x in w)]
    same = sum(oi['tokens'][j] == op['tokens'][j] for j in still)
    same_full = sum(res['imp']['tokens'][i] == res['plain']['tokens'][i] for i in idle)
    summ['idle'] = dict(n_idle=len(idle), silent_in_subset_run=len(still), identical_subset_run=same,
                        identical_in_full_runs=same_full)
    print('idle', json.dumps(summ['idle']), flush=True)
json.dump(summ, open(f'{a.out}.summary.json', 'w'), indent=1)
