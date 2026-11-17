"""Solve the fixed write vectors against the FROZEN host (nothing of the host changes), on host states from OUR data only:
the host teacher-forced on its own plain answers to our generated prompts (runs/*_plain*.jsonl of data/dev.jsonl / train2.jsonl).
  veto[S]  one vector v: adding it makes no token of set S the top token, keeping the host's choice among the rest
  kill_M   d x d map: kill write for token t = M e_t (e_t its embedding row); fitted so the killed token loses and nothing else changes
  force_fix  for tokens where the rms-normalised E_t/g direction (x 1e4) does not win, a solved direction that does
python solve_vecs.py OUT.pt runs/dev_plain.s0.jsonl ...
"""
import sys, json, re, torch, torch.nn.functional as F
from common import load, vocab_strings, EOS_IDS
from tables import build
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
tok, model = load(); strs = vocab_strings(tok); T = build(tok, strs)
norm = model.model.norm; g = norm.weight.float(); E = model.lm_head.weight.float(); Vn, d = E.shape
data = {}
for f in ['data/dev.jsonl', 'data/train2.jsonl']:
    for l in open(f): r = json.loads(l); data[r['key']] = r
rows = [json.loads(l) for p in sys.argv[2:] for l in open(p)]
print('answers', len(rows))
def logits(x): return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g) @ E.T
def states(conv=lambda s: s, n=250, cap=600):
    H = []
    with torch.no_grad():
        for r in rows[:n]:
            p = tok(tok.apply_chat_template([{'role': 'user', 'content': data[r['key']]['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False).input_ids
            a = tok(conv(r['response']), add_special_tokens=False).input_ids[:cap] + [EOS_IDS[0]]
            cap_ = {}; hk = norm.register_forward_pre_hook(lambda m, x: cap_.__setitem__('h', x[0][0]))
            model.model(input_ids=torch.tensor([p + a], device='cuda')); hk.remove(); H.append(cap_['h'][len(p) - 1:-1].float())
    return torch.cat(H)
def solve_veto(Hs, mask, name, steps=600, margin=2.0, lam=0.5, bs=256):
    n = len(Hs); perm = torch.randperm(n, device='cuda'); TR, VA = perm[:int(.8 * n)], perm[int(.8 * n):]
    with torch.no_grad(): top = torch.cat([logits(Hs[i:i + 2048]).argmax(-1) for i in range(0, n, 2048)])
    hit = TR[mask[top[TR]] > 0]
    v = torch.zeros(d, device='cuda', requires_grad=True); opt = torch.optim.Adam([v], lr=0.05); mb = mask > 0
    with torch.enable_grad():
        for st in range(steps):
            b = torch.cat([TR[torch.randint(len(TR), (bs // 2,), device='cuda')], hit[torch.randint(len(hit), (bs // 2,), device='cuda')]]) if len(hit) > 10 else TR[torch.randint(len(TR), (bs,), device='cuda')]
            with torch.no_grad(): lb = logits(Hs[b])
            lg = logits(Hs[b] + v)
            hinge = F.relu(lg.masked_fill(~mb, -1e9).max(-1).values - lg.masked_fill(mb, -1e9).max(-1).values + margin).mean()
            kl = F.kl_div(F.log_softmax(lg.masked_fill(mb, -1e9), -1), F.log_softmax(lb.masked_fill(mb, -1e9), -1), log_target=True, reduction='batchmean')
            loss = hinge + lam * kl; opt.zero_grad(); loss.backward(); opt.step()
    v = v.detach()
    with torch.no_grad():
        ok = keep = okh = nh = 0
        for i in range(0, len(VA), 1024):
            b = VA[i:i + 1024]; lb = logits(Hs[b]); t2 = logits(Hs[b] + v).argmax(-1)
            ok += (~mb[t2]).sum().item(); keep += (t2 == lb.masked_fill(mb, -1e9).argmax(-1)).sum().item()
            hh = mb[lb.argmax(-1)]; okh += (~mb[t2[hh]]).sum().item(); nh += hh.sum().item()
    print(f'veto {name:8s} |S|={int(mask.sum()):6d} held-out: S never top {100*ok/len(VA):.2f}% (on S-top states {100*okh/max(nh,1):.1f}% of {nh}) | host choice kept {100*keep/len(VA):.2f}% | rms {v.pow(2).mean().sqrt():.1f}', flush=True)
    return v.cpu()
H = states()
print('states', H.shape, flush=True)
eosm = torch.zeros(Vn, device='cuda'); eosm[EOS_IDS] = 1
out = {'veto': {}, 'force_fix': {}}
for name, m in [('comma', T['COMMASET']), ('star3', T['STAR3SET']), ('nn', T['NN']), ('nlstart', T['STARTNL']), ('eos', eosm), ('bulst', T['BULST'])]:
    out['veto'][name] = solve_veto(H, m.float(), name)
out['veto']['upper'] = solve_veto(states(str.lower, n=150), T['UPPERSET'].float(), 'upper')       # in regime: lowercase text
out['veto']['lower'] = solve_veto(states(str.upper, n=150), T['LOWERSET'].float(), 'lower')       # in regime: uppercase text
torch.save(out, sys.argv[1])
# kill map
strs_ok = torch.tensor([s.strip().isalpha() and len(s.strip()) > 2 for s in strs], device='cuda')
n = len(H); perm = torch.randperm(n, device='cuda'); TR, VA = perm[:int(.8 * n)], perm[int(.8 * n):]
with torch.no_grad(): tb = torch.cat([logits(H[i:i + 2048]).argmax(-1) for i in range(0, n, 2048)])
TRw = TR[strs_ok[tb[TR]]]; VAw = VA[strs_ok[tb[VA]]][:3000]; words = torch.nonzero(strs_ok)[:, 0]
M = torch.zeros(d, d, device='cuda')
with torch.no_grad(): M += torch.diag(-6.0 / g * torch.rsqrt((E / g).pow(2).mean()))
M.requires_grad_(True); opt = torch.optim.Adam([M], lr=1e-3)
with torch.enable_grad():
    for st in range(1000):
        b = TRw[torch.randint(len(TRw), (256,), device='cuda')]; t0 = tb[b]; rnd = words[torch.randint(len(words), (256,), device='cuda')]
        lg = logits(H[b] + E[t0] @ M.T); own = lg.gather(1, t0[:, None])[:, 0]; oth = lg.scatter(1, t0[:, None], -1e9).max(-1).values
        l1 = F.relu(own - oth + 2).mean()
        with torch.no_grad(): lb = logits(H[b])
        lg2 = logits(H[b] + E[rnd] @ M.T); keep = rnd != t0
        kl = F.kl_div(F.log_softmax(lg2[keep], -1), F.log_softmax(lb[keep], -1), log_target=True, reduction='batchmean')
        kl2 = F.kl_div(F.log_softmax(lg.scatter(1, t0[:, None], -1e9), -1), F.log_softmax(lb.scatter(1, t0[:, None], -1e9), -1), log_target=True, reduction='batchmean')
        loss = l1 + kl + 0.5 * kl2; opt.zero_grad(); loss.backward(); opt.step()
M = M.detach()
with torch.no_grad():
    t0 = tb[VAw]; lb = logits(H[VAw]); sec = lb.scatter(1, t0[:, None], -1e9).argmax(-1)
    t2 = logits(H[VAw] + E[t0] @ M.T).argmax(-1); rnd = words[torch.randint(len(words), (len(VAw),), device='cuda')]
    t3 = logits(H[VAw] + E[rnd] @ M.T).argmax(-1)
    print(f'kill map: target loses {100*(t2!=t0).float().mean():.1f}% | new top = host 2nd {100*(t2==sec).float().mean():.1f}% | unrelated kill keeps top {100*(t3==t0).float().mean():.1f}%', flush=True)
out['kill_M'] = M.cpu(); torch.save(out, sys.argv[1])
# force: rms-normalised E_t/g x 1e4 must win, alone and added to real states
D = E / g; D = D * torch.rsqrt(D.pow(2).mean(-1, keepdim=True))
fail = []
with torch.no_grad():
    for i in range(0, Vn, 4096):
        idx = torch.arange(i, min(i + 4096, Vn), device='cuda')
        hs = H[torch.randint(len(H), (len(idx),), device='cuda')]
        w1 = logits(D[idx] * 1e4).argmax(-1) == idx; w2 = logits(hs + D[idx] * 1e4).argmax(-1) == idx
        fail += idx[~(w1 & w2)].tolist()
fail = [t for t in fail if t < 151643 or t == 151645]
print('force fails for', len(fail), 'tokens', flush=True)
fx = torch.tensor(fail, device='cuda')
Dp = D[fx].clone().requires_grad_(True); opt = torch.optim.Adam([Dp], lr=0.05)
with torch.enable_grad():
    for st in range(600):
        hs = H[torch.randint(len(H), (len(fx),), device='cuda')]
        dn = Dp * torch.rsqrt(Dp.pow(2).mean(-1, keepdim=True))
        l = F.cross_entropy(logits(dn * 1e4), fx) + F.cross_entropy(logits(hs + dn * 1e4), fx)
        opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad():
    dn = Dp * torch.rsqrt(Dp.pow(2).mean(-1, keepdim=True))
    ok = (logits(dn * 1e4).argmax(-1) == fx) & (logits(H[torch.randint(len(H), (len(fx),), device='cuda')] + dn * 1e4).argmax(-1) == fx)
print('force fixed', int(ok.sum()), 'of', len(fail), 'still failing', [t for t, o in zip(fail, ok.tolist()) if not o][:30])
out['force_fix'] = {t: dn[i].cpu() for i, t in enumerate(fail) if ok[i]}
torch.save(out, sys.argv[1]); print('saved')
