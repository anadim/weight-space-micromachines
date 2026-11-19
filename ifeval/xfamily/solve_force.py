"""force_fix only (the v2 circuit uses no veto/kill vectors): for tokens whose rms-normalised E_t/g x 1e4 write does not win on
this host, solve a direction that does. Host states = host teacher-forced on its OWN plain answers to OUR dev prompts
(data/dev.jsonl; never IFEval). Copied from ifeval/heldout/solve_vecs2.py (force part); only the special-token filter
uses the tokenizer's special set instead of Qwen's id boundary.  python solve_force.py OUT.pt runs/dev_plain.jsonl"""
import sys, json, torch, torch.nn.functional as F
from common import load, EOS_IDS, special_ids, H as HC
torch.manual_seed(0)
tok, model = load(); SP = special_ids(tok)
norm = model.model.norm; g = norm.weight.float(); E = model.lm_head.weight.float(); Vn, d = E.shape
data = {json.loads(l)['key']: json.loads(l) for l in open('data/dev.jsonl')}
rows = [json.loads(l) for p in sys.argv[2:] for l in open(p)]
def logits(x): return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g) @ E.T
Hs = []
with torch.no_grad():
    for r in rows[:250]:
        p = tok(tok.apply_chat_template([{'role': 'user', 'content': data[r['key']]['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False).input_ids
        a = tok(r['response'], add_special_tokens=False).input_ids[:600] + [EOS_IDS[0]]
        cap = {}; hk = norm.register_forward_pre_hook(lambda m, x: cap.__setitem__('h', x[0][0]))
        model.model(input_ids=torch.tensor([p + a], device='cuda')); hk.remove(); Hs.append(cap['h'][len(p) - 1:-1].float())
H = torch.cat(Hs); print('states', len(H), flush=True)
D = E / g; D = D * torch.rsqrt(D.pow(2).mean(-1, keepdim=True))
fail = []
with torch.no_grad():
    for i in range(0, Vn, 4096):
        idx = torch.arange(i, min(i + 4096, Vn), device='cuda')
        hs = H[torch.randint(len(H), (len(idx),), device='cuda')]
        w1 = logits(D[idx] * 1e4).argmax(-1) == idx; w2 = logits(hs + D[idx] * 1e4).argmax(-1) == idx
        fail += idx[~(w1 & w2)].tolist()
fail = [t for t in fail if t not in SP or t == HC['im_end']]
print('force fails for', len(fail), 'tokens', flush=True)
out = {'veto': {}, 'force_fix': {}}
if fail:
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
