"""Solve the rule-keeper write directions against the FROZEN Qwen3-0.6B (instruct). Nothing of the host changes.

States: pre-final-norm hidden states of the host teacher-forced on its own baseline answers to DEV (even-key) prompts.
  veto[S]      : one vector v; adding it makes no token of set S win, while keeping the host's choice among the rest
  swap[S->t]   : one vector; where the host's top token is in S (e.g. end-of-turn) token t wins instead; elsewhere top-1 kept
  force[t]     : a direction that makes t win (rms 1; used with a huge gain) -- E_t/g, checked over the whole vocab
  kill         : -beta * E_t/g (rms-normalised), beta calibrated here
python solve_dirs.py OUT.pt
"""
import sys, json, torch, torch.nn.functional as F
from common import load, vocab_strings, token_sets, EOS_IDS, chat
torch.manual_seed(0)
tok, model = load()
norm, head = model.model.norm, model.lm_head
g = norm.weight.detach().float(); E = head.weight.detach().float()
V = E.shape[0]
strs = vocab_strings(tok)
sets = token_sets(strs)
rows = [json.loads(l) for l in open('../data/q3_nothink.jsonl') if json.loads(l)['key'] % 2 == 0]

H, NXT = [], []
with torch.no_grad():
    for r in rows:
        p = tok(chat(tok, r['prompt']), add_special_tokens=False).input_ids
        a = tok(r['response'], add_special_tokens=False).input_ids[:1500] + [EOS_IDS[0]]
        ids = torch.tensor([p + a], device='cuda')
        cap = {}
        hk = norm.register_forward_pre_hook(lambda m, x: cap.__setitem__('h', x[0][0]))
        model.model(input_ids=ids); hk.remove()
        H.append(cap['h'][len(p) - 1:-1].float()); NXT.append(torch.tensor(a, device='cuda'))
H = torch.cat(H); NXT = torch.cat(NXT)
print('states', H.shape, 'h rms', H.pow(2).mean(-1).sqrt().median().item())


def logits(x): return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g) @ E.T


with torch.no_grad():
    base_top = torch.cat([logits(H[i:i + 1024]).argmax(-1) for i in range(0, len(H), 1024)])
perm = torch.randperm(len(H), device='cuda'); ntr = int(0.8 * len(H)); TR, VA = perm[:ntr], perm[ntr:]


def solve_veto(name, mask, steps=1500, margin=2.0, lam=0.5, bs=384):
    v = torch.zeros(E.shape[1], device='cuda', requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.05)
    # oversample states whose top token is in S (those are where the veto has to work)
    hit = TR[mask[base_top[TR]]]
    for st in range(steps):
        b = torch.cat([TR[torch.randint(len(TR), (bs // 2,), device='cuda')], hit[torch.randint(len(hit), (bs // 2,), device='cuda')]]) if len(hit) else TR[torch.randint(len(TR), (bs,), device='cuda')]
        with torch.no_grad(): lb = logits(H[b])
        lg = logits(H[b] + v)
        smax = lg.masked_fill(~mask, -1e9).max(-1).values; nmax = lg.masked_fill(mask, -1e9).max(-1).values
        hinge = F.relu(smax - nmax + margin).mean()
        kl = F.kl_div(F.log_softmax(lg.masked_fill(mask, -1e9), -1), F.log_softmax(lb.masked_fill(mask, -1e9), -1), log_target=True, reduction='batchmean')
        loss = hinge + lam * kl
        opt.zero_grad(); loss.backward(); opt.step()
    v = v.detach()
    with torch.no_grad():
        ok = keep = n = 0; okh = nh = 0
        for i in range(0, len(VA), 1024):
            b = VA[i:i + 1024]; lb = logits(H[b]); lg = logits(H[b] + v)
            top = lg.argmax(-1); ok += (~mask[top]).sum().item(); n += len(b)
            keep += (top == lb.masked_fill(mask, -1e9).argmax(-1)).sum().item()
            h = mask[lb.argmax(-1)]; okh += (~mask[top[h]]).sum().item(); nh += h.sum().item()
    print(f'veto {name:6s} |S|={mask.sum().item():6d} held-out: S never wins {100*ok/n:.2f}% (on S-top states {100*okh/max(nh,1):.1f}% of {nh}) | host non-S choice kept {100*keep/n:.2f}% | rms {v.pow(2).mean().sqrt():.1f}')
    return v


def solve_swap(name, src_mask, t, steps=1500, margin=2.0, lam=0.5, bs=256):
    v = torch.zeros(E.shape[1], device='cuda', requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.05)
    pos = TR[src_mask[base_top[TR]]]; neg = TR[~src_mask[base_top[TR]]]
    for st in range(steps):
        bp = pos[torch.randint(len(pos), (bs // 2,), device='cuda')]; bn = neg[torch.randint(len(neg), (bs // 2,), device='cuda')]
        lp = logits(H[bp] + v); others = lp.clone(); others[:, t] = -1e9
        l1 = F.relu(others.max(-1).values - lp[:, t] + margin).mean()
        with torch.no_grad(): lbn = logits(H[bn]); tn = lbn.argmax(-1)
        ln = logits(H[bn] + v); own = ln.gather(1, tn[:, None])[:, 0]; o2 = ln.scatter(1, tn[:, None], -1e9)
        l2 = F.relu(o2.max(-1).values - own + margin).mean()
        kl = F.kl_div(F.log_softmax(ln, -1), F.log_softmax(lbn, -1), log_target=True, reduction='batchmean')
        loss = l1 + l2 + lam * kl
        opt.zero_grad(); loss.backward(); opt.step()
    v = v.detach()
    with torch.no_grad():
        tb = torch.cat([logits(H[VA[i:i + 1024]]).argmax(-1) for i in range(0, len(VA), 1024)]); tg = torch.cat([logits(H[VA[i:i + 1024]] + v).argmax(-1) for i in range(0, len(VA), 1024)])
        s = src_mask[tb]
    print(f'swap {name:10s} held-out: S-top states -> t {100*(tg[s]==t).float().mean():.1f}% of {s.sum().item()} | other states top kept {100*(tg[~s]==tb[~s]).float().mean():.2f}% | rms {v.pow(2).mean().sqrt():.1f}')
    return v


import os
out = torch.load(sys.argv[1] + '.partial') if os.path.exists(sys.argv[1] + '.partial') else {'veto': {}, 'swap': {}}
torch.set_grad_enabled(True)
for name in ['comma', 'upper', 'lower', 'eos', 'nn', 'star3']:
    if name not in out['veto']: out['veto'][name] = solve_veto(name, sets[name]); torch.save(out, sys.argv[1] + '.partial')
for name, t in [('eos>quote', tok('"', add_special_tokens=False).input_ids[0]), ('eos>nn', 271), ('eos>star6', 6824)]:
    out['swap'][name] = solve_swap(name, sets['eos'], t)
torch.set_grad_enabled(False)
# force: does E_t/g win for every token under a huge gain?
D = E / g
win = torch.zeros(V, dtype=torch.bool, device='cuda')
for i in range(0, V, 4096):
    x = D[i:i + 4096]; win[i:i + 4096] = logits(x * 1e4).argmax(-1) == torch.arange(i, min(i + 4096, V), device='cuda')
print(f'force E_t/g wins for {100*win.float().mean():.2f}% of the vocabulary ({(~win).sum().item()} fail)')
out['force_ok'] = win.cpu()
# kill calibration: on states whose top token is a word token t, add -beta * rmsnorm(E_t/g)
Dn = D * torch.rsqrt(D.pow(2).mean(-1, keepdim=True))
wordy = torch.tensor([s.strip().isalpha() and len(s.strip()) > 2 for s in strs], device='cuda')
idx = VA[wordy[base_top[VA]]][:3000]
lb = logits(H[idx]); tb = lb.argmax(-1); second = lb.scatter(1, tb[:, None], -1e9).argmax(-1)
for beta in [5, 10, 20, 40, 80]:
    lg = logits(H[idx] - beta * Dn[tb]); t2 = lg.argmax(-1)
    print(f'kill beta={beta:3d}: target loses {100*(t2!=tb).float().mean():.1f}% | new top == host 2nd choice {100*(t2==second).float().mean():.1f}%')
torch.save(out, sys.argv[1])
