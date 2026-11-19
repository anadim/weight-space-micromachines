"""Re-solve ONLY the force_fix rows (the one host-tied tensor circuit.py reads) for a new host. The code below is the
force_fix part of Team D's solve_vecs2.py, verbatim except: the host is chosen by WSM_MODEL (common.py), and the unused
veto / kill-map solves are skipped (circuit v2 never reads them). Data = the host's OWN plain answers to OUR dev prompts
(data/dev.jsonl, seed 0), teacher-forced; nothing from IFEval.
WSM_MODEL=Qwen/Qwen3-1.7B python solve_forcefix.py vecs_1.7B.pt runs/q17_dev_plain.s0.jsonl runs/q17_dev_plain.s1.jsonl"""
import sys, json, torch, torch.nn.functional as F
from common import load, vocab_strings, EOS_IDS, MODEL
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
tok, model = load()
norm = model.model.norm; g = norm.weight.float(); E = model.lm_head.weight.float(); Vn, d = E.shape
print(MODEL, 'd =', d, flush=True)
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
H = states()
print('states', H.shape, 'state rms', H.pow(2).mean(-1).sqrt().median().item(), 'max |h|', H.abs().max().item(), flush=True)
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
# extra diagnostic (not used by the circuit): with the solved rows in place, does every forced token win on fresh state draws?
with torch.no_grad():
    allok = 0; tot = 0
    for i in range(0, Vn, 4096):
        idx = torch.arange(i, min(i + 4096, Vn), device='cuda'); idx = idx[(idx < 151643) | (idx == 151645)]
        Dd = D[idx].clone(); m = torch.isin(idx, fx)
        if m.any():
            pos = torch.searchsorted(fx.sort().values, idx[m]); srt = fx.sort().indices
            Dd[m] = dn[srt[pos]]
        hs = H[torch.randint(len(H), (len(idx),), device='cuda')]
        allok += (logits(hs + Dd * 1e4).argmax(-1) == idx).sum().item(); tot += len(idx)
print(f'diag: forced token wins on a fresh random state for {allok}/{tot} vocab tokens', flush=True)
out = {'force_fix': {t: dn[i].cpu() for i, t in enumerate(fail) if ok[i]}, 'source': f'solve_forcefix.py on {MODEL}, ' + ' '.join(sys.argv[2:])}
torch.save(out, sys.argv[1]); print('saved', len(out['force_fix']))
