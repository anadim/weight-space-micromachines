"""task [31](b)(ii): measure the candidate early-zero rule (ifreward.degenerate: no letters / pure repetition -> reward 0)
against the official lm_eval checker (prompt-level strict, seeded) on sampled RL-shape rollouts (train prompts, 64 x G8, T 1,
max_new 1024, chat format) and on existing greedy 541 outputs. The rule would be a reward change unless the official reward
is already 0 on every flagged output; on unflagged outputs it is the official reward by construction.

python degen.py --cfg NAME:MODEL:CKPT|-:imp|plain[:NBATCH] ... [--sd SD] [--jsonl NAME:RUN.jsonl ...] --out PREFIX
"""
import argparse, collections, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'stk'), os.path.join(HERE, '..', 'f14')]
import ifreward as RW

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='*', default=[]); ap.add_argument('--sd', default=None)
ap.add_argument('--jsonl', nargs='*', default=[]); ap.add_argument('--out', required=True)
ap.add_argument('--train', default=os.path.join(HERE, '..', 'gen', 'train.jsonl'))
ap.add_argument('--ifeval', default=os.path.join(HERE, '..', 'ifdata', 'ifeval.jsonl'))
ap.add_argument('--max_new', type=int, default=1024); ap.add_argument('--B_max', type=int, default=256)
ap.add_argument('--workers', type=int, default=8); ap.add_argument('--mem_frac', type=float, default=0)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
POOL = RW.Pool(a.workers)
rows = []                                                           # (source, doc, text)


def add(src, docs, texts):
    sc = POOL.map(docs, texts)
    for d, t, s in zip(docs, texts, sc):
        rows.append(dict(src=src, key=d['key'], ids=d['instruction_id_list'], r=s[0], rl=s[3], deg=RW.degenerate(t),
                         n=len(t), head=t[:160]))


IFD = {d['key']: d for d in map(json.loads, open(a.ifeval))}
for spec in a.jsonl:                                                # greedy 541 outputs (eval541 dumps)
    name, path = spec.split(':', 1)
    if not os.path.exists(path): print('missing', path, flush=True); continue
    R = [json.loads(l) for l in open(path)]
    add(name, [IFD[r['key']] for r in R], [r['response'] for r in R]); print('jsonl', name, len(R), flush=True)

if a.cfg:
    import torch
    if a.mem_frac: torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    import fastqwen as Q
    from transformers import AutoTokenizer
    from ifengine import IFEngine
    import common_if as CI
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    TRAIN = [json.loads(l) for l in open(a.train)]
    pre, tr_full = CI.format_prompts(tok, [d['prompt'] for d in TRAIN], 'chat')
    Lp = len(pre); maxq = max(len(f) for f in tr_full) - Lp
    for spec in a.cfg:
        p = spec.split(':'); name, model, ck, mode = p[:4]; nb = int(p[4]) if len(p) > 4 else 2
        if ck != '-' and not os.path.exists(ck): print('missing', ck, flush=True); continue
        P32 = Q.load_hf_params(model)
        if ck != '-':
            from safetensors.torch import load_file
            c = load_file(ck)
            for k in P32: P32[k] = c[k].to(P32[k].device, torch.float32)
        W = Q.fuse(P32, torch.bfloat16); del P32
        imp = None
        if mode == 'imp':
            from gexec import GExec
            imp = GExec(torch.load(a.sd))
        eng = IFEngine(W, tok, pre, imp=imp, B_max=a.B_max, S_max=maxq + a.max_new + 8, N_cap=512, max_new_cap=a.max_new)
        for b in range(nb):
            bi = random.Random(4242 + b).sample(range(len(TRAIN)), 64)
            full = [tr_full[i] for i in bi]
            o = eng.generate([f[Lp:] for f in full], G=8, T=1.0, max_new=a.max_new, greedy=False, full_ids=full, seed=777 + b)
            add(name, [TRAIN[bi[j // 8]] for j in range(len(o['tokens']))], [tok.decode(t) for t in o['tokens']])
            print('cfg', name, 'batch', b, json.dumps({k: round(v, 2) for k, v in o['stats'].items()}), flush=True)
        del eng, W, imp; torch.cuda.empty_cache()

with open(a.out + '.rows.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')


def summ(R):
    fl = [r for r in R if r['deg']]
    bad = [r for r in fl if r['r'] > 0]
    return dict(n=len(R), flagged=len(fl), flagged_noletter=sum(r['deg'] == 'noletter' for r in fl),
                flagged_repeat=sum(r['deg'] == 'repeat' for r in fl), flagged_official_nonzero=len(bad),
                flagged_official_loose_nonzero=sum(r['rl'] > 0 for r in fl),
                reward_changed_frac=len(bad) / max(1, len(R)), mean_official=sum(r['r'] for r in R) / max(1, len(R)),
                mean_rule=sum(r['r'] for r in R if not r['deg']) / max(1, len(R)),
                unflagged_identical=True,
                nonzero_inst_ids=collections.Counter(','.join(r['ids']) for r in bad).most_common(10))


S = {'ALL': summ(rows), 'sampled': summ([r for r in rows if not r['src'].startswith('g541')]),
     'greedy541': summ([r for r in rows if r['src'].startswith('g541')])}
for s in sorted(set(r['src'] for r in rows)): S[s] = summ([r for r in rows if r['src'] == s])
S['examples_flagged_nonzero'] = [dict(src=r['src'], ids=r['ids'], deg=r['deg'], head=r['head']) for r in rows
                                 if r['deg'] and r['r'] > 0][:12]
json.dump(S, open(a.out + '.summary.json', 'w'), indent=1, ensure_ascii=False)
for k, v in S.items():
    if k != 'examples_flagged_nonzero':
        print(k, {kk: vv for kk, vv in v.items() if kk != 'nonzero_inst_ids'}, flush=True)
