"""LATENT action (dev only). Candidate decline vectors per layer:
  v_sys: mean over dev positives of [state | system-prompt fence, host declined] - [state | no system prompt]   (topic-decline delta)
  v_ref: mean(AdvBench) - mean(Alpaca dev)                                                                    (generic refusal)
Sweep (vector, layer L2, scale) with the gate forced ON for 64 dev positives; measure decline (string) and degeneration.
usage: s3_latent.py F out.pt"""
import sys, json, random, torch, numpy as np, pandas as pd
from fc import *
F, outp = sys.argv[1:3]
rows = load_jsonl('data/dev.jsonl')
def role(r):
    c = r['clf_' + F]
    return 'pos' if (r['fence'] == F and r['label'] == 'pos' and c > 0) else ('neg' if (r['fence'] != F and r['label'] in ('neg',) and c < -2) else None)
rng = random.Random(0)
pos = [r['text'] for r in rows if role(r) == 'pos' and r['source'] == 'gen7b']; rng.shuffle(pos)
neg = [r['text'] for r in rows if role(r) == 'neg']; rng.shuffle(neg)
alp = [r['text'] for r in rows if r['source'] == 'alpaca_train' and r['clf_math'] < -2 and r['clf_ml'] < -2][:200]
adv = pd.read_csv('data/advbench.csv')['goal'].tolist()[:200]
m, tok = load(HOST)
P = pos[:200]
# v_sys was dropped: Qwen3-1.7B declines only 3% of dev positives under the system prompt (too few declining states).
# v_pf: host states while it READS a teacher-forced decline vs a teacher-forced compliance opener (mean over opener tokens)
@torch.no_grad()
def opener_states(prompts, opener, bs=16):
    out = []
    for i in range(0, len(prompts), bs):
        txt = [c + opener for c in chat(tok, prompts[i:i + bs])]
        enc = tok(txt, return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')
        n = len(tok(opener, add_special_tokens=False)['input_ids'])
        hs = m(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -n:].float().mean(1) for h in hs], 1).cpu())
    return torch.cat(out)
v_pf = (opener_states(P, PHRASE[F]) - opener_states(P, "Sure! Let's work through this step by step.")).mean(0)
Hn = last_states(m, tok, P).float()
v_ref = last_states(m, tok, adv).float().mean(0) - last_states(m, tok, alp).float().mean(0)
nrm = Hn.norm(dim=-1).mean(0)                                     # typical last-token norm per layer
torch.save(dict(v_pf=v_pf, v_ref=v_ref, nrm=nrm), outp)
print('|v_pf|/|h|', (v_pf.norm(dim=-1) / nrm)[::2].numpy().round(2)); print('|v_ref|/|h|', (v_ref.norm(dim=-1) / nrm)[::2].numpy().round(2))
T = pos[200:264]
base = generate(m, tok, T, max_new=60)
print('base decline', np.mean([refusal_str(g['text']) for g in base]), flush=True)
res = []
for vn, V in [('v_ref', v_ref), ('v_pf', v_pf)]:
    for L2 in [8, 10, 12, 14, 16, 18, 20]:
        for a in [1, 2, 4, 8]:
            fz = Fence(m, tok, 'latent', vec=V[L2] * a, L2=L2)
            g = generate(m, tok, T, fence=fz, gates=[True] * len(T), max_new=60); fz.detach()
            d = np.mean([refusal_str(x['text']) for x in g])
            uniq = np.mean([len(set(x['text'].split())) / max(1, len(x['text'].split())) for x in g])
            res.append(dict(v=vn, L2=L2, a=a, decline=float(d), uniq=float(uniq), ex=[x['text'][:160] for x in g[:3]]))
            print(vn, L2, a, f'decline {d:.2f} uniq {uniq:.2f} |', g[0]['text'][:110].replace('\n', ' '), flush=True)
json.dump(res, open(outp.replace('.pt', '_sweep.json'), 'w'), indent=1)
