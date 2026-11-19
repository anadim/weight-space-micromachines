"""Non-BBH no-tool generalization check (coordinator request): ARC-Easy multiple-choice in a 3-shot CoT format,
v2 tools attached. No ARC item is used for probe training. Report call rate and accuracy for none vs gate vs breaker.
python cg_arc.py OUT.json --mode none|gate|breaker --probe P.pt [--n 400]
"""
import os, sys, re, json, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _cgpaths import CKPT, RES, BBH, COT, CODE
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import integ
from cg_gate_gen import generate

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--model', default=CKPT)
ap.add_argument('--mode', default='none'); ap.add_argument('--probe', default=f'{RES}/probe_v3.pt')
ap.add_argument('--n', type=int, default=400); ap.add_argument('--bs', type=int, default=32); ap.add_argument('--max_new', type=int, default=400)
a = ap.parse_args()
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
net = integ.attach_all(model, tok, integ.TOOLS_V2, veto=integ.VETO)
probe = torch.load(a.probe) if a.mode == 'gate' else None


def fmt(e):
    opts = '\n'.join(f'({l}) {t}' for l, t in zip(e['choices']['label'], e['choices']['text']))
    return f"Q: {e['question']}\nOptions:\n{opts}\nA: Let's think step by step."


tr = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='train')
val = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='validation')
shots = [tr[i] for i in (0, 1, 2)]
PREFIX = '\n\n'.join(fmt(s) + f"\nThe answer is ({s['answerKey']})." for s in shots) + '\n\n'
val = [e for e in val if len(e['choices']['label']) <= 5][:a.n]


def extract(text):
    t = text.split('\n\nQ:')[0]
    m = re.findall(r'[Tt]he answer is\s*\(?([A-E1-5])\)?', t)
    return m[-1] if m else ''


rows = []; t0 = time.time()
for b in range(0, len(val), a.bs):
    ch = val[b:b + a.bs]
    for _k in range(40):
            try: gens, stats = generate(model, tok, net, [PREFIX + fmt(e) for e in ch], a.max_new, dev, mode=a.mode, probe=probe); break
            except torch.OutOfMemoryError: torch.cuda.empty_cache(); print('OOM retry', _k + 1, flush=True); time.sleep(30)
    for e, g, st in zip(ch, gens, stats):
        txt = tok.decode(g).replace('<|endoftext|>', ''); p = extract(txt)
        rows.append({'gold': e['answerKey'], 'pred': p, 'ok': p == e['answerKey'], 'called': '<tool_call>' in txt, 'ncall': st['ncall'], 'blocked': st['blocked'], 'out': txt})
    print(f'{b+len(ch)}/{len(val)} | {time.time()-t0:.0f}s', flush=True)
acc = 100 * sum(r['ok'] for r in rows) / len(rows); cr = 100 * sum(r['called'] for r in rows) / len(rows)
blk = 100 * sum(r['blocked'] for r in rows) / len(rows); nc = sum(r['ncall'] for r in rows) / len(rows)
print(f'ARC-Easy mode={a.mode}: acc={acc:.1f} call_rate={cr:.1f} blk={blk:.1f} mean_ncall={nc:.2f} (n={len(rows)})')
json.dump({'mode': a.mode, 'acc': acc, 'call_rate': cr, 'blk': blk, 'mean_ncall': nc, 'n': len(rows), 'rows': rows}, open(a.out, 'w'), indent=1)
