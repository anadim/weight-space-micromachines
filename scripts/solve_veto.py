"""Solve the calculator's 'no more digits' write (the veto) against the frozen host.

One residual direction v (added before the final norm) that makes every digit token lose while leaving the model's own
choice among the other tokens as it was. Fitted with Adam on real hidden states at the ends of numbers in GSM8K training
solutions (rows 2000-2299 of the train split). No weight of the host changes; only v is chosen.
Runs on CPU in fp32 (as in the original run), a few minutes.

python scripts/solve_veto.py [--model Qwen/Qwen3-0.6B-Base] [--out runs/veto_qwen3.pt] [--train data/gsm8k/train.jsonl]
-> saves {'v': (d_model,)} and prints how often the model's non-digit choice survives
"""
import argparse, json, os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--out', default='runs/veto_qwen3.pt')
ap.add_argument('--train', default='data/gsm8k/train.jsonl')
ap.add_argument('--device', default='cpu')          # 'cuda' to collect hidden states on GPU (fp32), much faster for 1.7B/4B
ap.add_argument('--nrows', type=int, default=300)   # number of GSM8K-train solutions (rows 2000..2000+nrows) used
a = ap.parse_args()
torch.set_grad_enabled(True)
dev = a.device
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
digits = [tok(str(d), add_special_tokens=False)['input_ids'][0] for d in range(10)]
norm, head = model.model.norm, model.lm_head
rows = [json.loads(l) for l in open(a.train)][2000:2000 + a.nrows]
H, nxt = [], []
with torch.no_grad():
    for r in rows:
        body = re.sub(r'<<[^>]*>>', '', r['answer'].split('####')[0])
        ids = tok(f"Question: {r['question']}\nAnswer: {body}", return_tensors='pt').input_ids.to(dev)
        cap = {}
        hk = norm.register_forward_pre_hook(lambda m, x: cap.__setitem__('h', x[0][0]))
        model.model(input_ids=ids); hk.remove(); hs = cap['h']                # the residual stream right before the final norm
        t = ids[0].tolist()
        for p in range(len(t) - 1):
            if t[p] in digits and t[p + 1] not in digits:              # last digit of a number; the true next token is not a digit
                H.append(hs[p]); nxt.append(t[p + 1])
H = torch.stack(H); print('number endings', len(H))
with torch.no_grad():
    base = head(norm(H)); mask = torch.zeros(base.shape[1], dtype=torch.bool, device=dev); mask[digits] = True
    nondig_choice = base.masked_fill(mask, -1e9).argmax(-1)
    print('model already avoids a digit at', f'{100 * (~mask[base.argmax(-1)]).float().mean():.1f}%', 'of number endings')
torch.manual_seed(0); perm = torch.randperm(len(H), device=dev); tr, va = perm[:int(0.8 * len(H))], perm[int(0.8 * len(H)):]
g = norm.weight.detach(); E = head.weight.detach()
v = torch.zeros(E.shape[1], device=dev, requires_grad=True)
opt = torch.optim.Adam([v], lr=0.02)
F = torch.nn.functional
for step in range(1500):                    # hinge: no digit may win (margin 2); otherwise stay close to the model's own distribution
    b = tr[torch.randint(len(tr), (256,), device=dev)]
    lg = head(norm(H[b] + v)); lb = base[b]
    dmax = lg[:, digits].max(-1).values; ndmax = lg.masked_fill(mask, -1e9).max(-1).values
    hinge = F.relu(dmax - ndmax + 2).mean()
    kl = F.kl_div(F.log_softmax(lg.masked_fill(mask, -1e9), -1), F.log_softmax(lb.masked_fill(mask, -1e9), -1), log_target=True, reduction='batchmean')
    loss = hinge + 0.5 * kl
    opt.zero_grad(); loss.backward(); opt.step()
v = v.detach()
with torch.no_grad():
    for name, idx in (('train', tr), ('held-out', va)):
        lg = torch.cat([head(norm(H[idx[i:i + 512]] + v)) for i in range(0, len(idx), 512)])
        print(f'{name}: no digit {100 * (~mask[lg.argmax(-1)]).float().mean():.2f}% | own non-digit choice kept {100 * (lg.argmax(-1) == nondig_choice[idx]).float().mean():.2f}% | |v| rms {v.pow(2).mean().sqrt():.2f}')
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
torch.save({'v': v.detach().cpu()}, a.out)
print('saved', a.out)
