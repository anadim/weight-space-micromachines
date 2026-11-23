"""GRPO-family RL on Qwen3-0.6B-Base with the GSM calculator implant (calc_s4 + veto), fast engine.

Arms:  1 = no implant | 2 = implant attached, frozen | 3 = implant + trainable links (feature = implant one-hot e,
       gate init 0, designed write kept) | 3b = implant feature only through links with RANDOM init, designed write OFF.
Env:   GSM8K train split A + its scaled-number (GSM-Hard-like) copies. Dev: split B (same mix). Test never read here.
Update: P prompts x G samples, one optimizer step (on-policy). Old logp from the rollout pass. Shared prefix KV computed
once per update WITH grad (prefix-leaf trick: micro-batches accumulate dL/dK_prefix, one backward through the prefix).
"""
import os, sys, time, json, math, random, argparse, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='1', choices=['1', '2', '3', '3b', 'R'])
ap.add_argument('--rand', default='stack1', help="arm R: which stacks get their nonzero values shuffled")
ap.add_argument('--run_dir', required=True)
ap.add_argument('--updates', type=int, default=100)
ap.add_argument('--prompts', type=int, default=64); ap.add_argument('--G', type=int, default=8)
ap.add_argument('--T', type=float, default=1.0); ap.add_argument('--max_new', type=int, default=320)
ap.add_argument('--lr', type=float, default=1e-6); ap.add_argument('--link_lr', type=float, default=1e-3)
ap.add_argument('--kl', type=float, default=0.0); ap.add_argument('--loss', default='drgrpo', choices=['drgrpo', 'cispo'])
ap.add_argument('--clip', type=float, default=0.2); ap.add_argument('--cispo_max', type=float, default=4.0)
ap.add_argument('--tok_budget', type=int, default=16384, help='packed suffix tokens per training micro-batch')
ap.add_argument('--link_layers', default='6,12,18,final'); ap.add_argument('--link_std', type=float, default=0.01)
ap.add_argument('--dev_every', type=int, default=25); ap.add_argument('--dev_n', type=int, default=400)
ap.add_argument('--ckpt_every', type=int, default=100)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--B_max', type=int, default=512)
ap.add_argument('--warmup', type=int, default=5); ap.add_argument('--fused', type=int, default=1)
a = ap.parse_args()
os.makedirs(a.run_dir, exist_ok=True)
json.dump(vars(a), open(f'{a.run_dir}/config.json', 'w'), indent=1)
random.seed(a.seed); torch.manual_seed(a.seed)
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'))
pre, shot_q = C.prefix_ids(tok); Lp = len(pre)
P32 = Q.load_hf_params(dtype=torch.float32)
for v in P32.values(): v.requires_grad_(True)
names = list(P32)
W = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16)      # rollout copy (updated in place)
W_ref = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16) if a.kl > 0 else None
imp = None
if a.arm != '1':
    hf = AutoModelForCausalLM.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'), dtype=torch.bfloat16).to(dev).eval()
    imp = CalcImplant(hf, tok, a.B_max, fused=bool(a.fused), random=(a.rand if a.arm == 'R' else None))
    del hf; torch.cuda.empty_cache()


class Link(nn.Module):
    def __init__(self, nf, d, std, gate0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, nf, device=dev) * std)
        self.gate = nn.Parameter(torch.full((d,), float(gate0), device=dev))

    def forward(self, f):
        return self.gate * F.linear(f.float(), self.W)


links = None
if a.arm in ('3', '3b'):
    torch.manual_seed(1000 + a.seed)
    links = {}
    for l in a.link_layers.split(','):
        key = 'final' if l == 'final' else int(l)
        links[key] = Link(imp.n_out, Q.D, a.link_std, 0.0 if a.arm == '3' else 1.0)
    torch.manual_seed(a.seed)
link_params = [p for lk in (links or {}).values() for p in lk.parameters()]
groups = [{'params': [P32[k] for k in names], 'lr': a.lr}]
if link_params: groups.append({'params': link_params, 'lr': a.link_lr})
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
base_lrs = [g['lr'] for g in opt.param_groups]
eng = Engine(W, pre, tok, implant=imp, links=links, B_max=a.B_max, S_max=544, use_write=(a.arm != '3b'))

pool = C.load_train('A', shot_q=shot_q)
pool = [r for r in pool if len(C.q_ids(tok, r['q'])) + a.max_new <= 544]
devset = C.load_train('B', shot_q=shot_q)
rng_dev = random.Random(123)
dg = [r for r in devset if r['kind'] == 'gsm']; ds = [r for r in devset if r['kind'] == 'scaled']
rng_dev.shuffle(dg); rng_dev.shuffle(ds)
devset = dg[:a.dev_n // 2] + ds[:a.dev_n // 2]
print(f'pool {len(pool)} (gsm {sum(r["kind"]=="gsm" for r in pool)}), dev {len(devset)}, prefix {Lp}', flush=True)


def log(obj):
    with open(f'{a.run_dir}/steps.jsonl', 'a') as f: f.write(json.dumps(obj) + '\n')


@torch.no_grad()
def sync_weights():
    Wn = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16)
    for k in ('E', 'lnf'): W[k].copy_(Wn[k])
    for l, ln in zip(W['L'], Wn['L']):
        for k in l: l[k].copy_(ln[k])
    eng.refresh()


def calls_of(wf):
    return sum(1 for i in range(len(wf)) if wf[i] and (i == 0 or not wf[i - 1]))


def evaluate(items, use_implant=True, tag='dev'):
    """Greedy eval (run_gsm protocol). use_implant=False: host only (implant detached, links off)."""
    torch.cuda.empty_cache()
    e2 = eng; eng.imp_on = use_implant
    qids = [C.q_ids(tok, r['q']) for r in items]
    out = e2.generate(qids, G=1, greedy=True, max_new=a.max_new)
    sc = [C.score(tok, t, r['gold']) for t, r in zip(out['tokens'], items)]
    acc = lambda kind: 100 * sum(s[0] for s, r in zip(sc, items) if r['kind'] == kind) / max(1, sum(r['kind'] == kind for r in items))
    res = {f'{tag}_acc': 100 * sum(s[0] for s in sc) / len(sc), f'{tag}_gsm': acc('gsm'), f'{tag}_scaled': acc('scaled'),
           f'{tag}_fmt': 100 * sum(s[1] for s in sc) / len(sc),
           f'{tag}_writes': sum(sum(w) for w in out['wflag']) / len(sc), f'{tag}_calls': sum(calls_of(w) for w in out['wflag']) / len(sc),
           f'{tag}_len': sum(map(len, out['tokens'])) / len(sc)}
    eng.imp_on = True
    return res


def save(tag):
    d = f'{a.run_dir}/{tag}'; os.makedirs(d, exist_ok=True)
    from safetensors.torch import save_file
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in P32.items()}
    sd['lm_head.weight'] = sd['model.embed_tokens.weight'].clone()
    save_file(sd, f'{d}/model.safetensors')
    if links: torch.save({str(k): lk.state_dict() for k, lk in links.items()}, f'{d}/links.pt')
    open(f'{d}/DONE', 'w').close()


PACK_SIZES = [n for n in (2048, 4096, 8192, 16384, 32768) if n <= a.tok_budget] or [a.tok_budget]
packers = {n: Q.Packer(n, Lp, dev) for n in PACK_SIZES}


def lp_of_hidden(Wt, hn, tgt):
    E = Wt['E']
    def chunk_lp(hc, tc):
        lg = F.linear(hc, E).float() / a.T
        return torch.log_softmax(lg, -1).gather(-1, tc[:, None])[:, 0]
    return torch.cat([checkpoint(chunk_lp, hn[c:c + 4096], tgt[c:c + 4096], use_reentrant=False) for c in range(0, hn.shape[0], 4096)])


def train_step(batch_q, batch_r, old_lp, adv, n_total_rows):
    """Packed sequences: each micro-batch = rows concatenated into one stream of exactly tok_budget tokens (flex attention,
    block mask = shared prefix + own causal suffix). Prefix KV computed ONCE per update with grad; micro-batches
    accumulate dL/d(prefix KV) into leaf tensors, then one backward through the prefix forward."""
    t0 = time.time()
    W0 = Q.fuse(P32, torch.bfloat16)
    pkv = Q.prefix_kv_of(W0, eng.prefix_ids, eng.cos_t, eng.sin_t)
    leaves = [(k.detach().requires_grad_(True), v.detach().requires_grad_(True)) for k, v in pkv]
    pkv_ref = None
    if W_ref is not None:
        with torch.no_grad(): pkv_ref = Q.prefix_kv_of(W_ref, eng.prefix_ids, eng.cos_t, eng.sin_t)
    order = sorted(range(len(batch_q)), key=lambda i: -(len(batch_q[i]) + len(batch_r[i])))
    micro, cur, cur_n = [], [], 0                         # first-fit-decreasing into bins of tok_budget tokens
    bins = []
    for i in order:
        L = len(batch_q[i]) + len(batch_r[i])
        for bn in bins:
            if bn[1] + L <= a.tok_budget: bn[0].append(i); bn[1] += L; break
        else: bins.append([[i], L])
    micro = [bn[0] for bn in bins]
    tot_tok = sum(len(r) for r in batch_r)
    norm = (n_total_rows * a.max_new) if a.loss == 'drgrpo' else tot_tok
    st = dict(loss=0.0, kl=0.0, clipfrac=0.0, ratio=0.0, ntok=tot_tok, n_micro=len(micro),
              pack_fill=sum(bn[1] for bn in bins) / (len(bins) * a.tok_budget))
    Kt = imp.K + imp.W if imp is not None else 0
    for mb in micro:
        Wt = Q.fuse(P32, torch.bfloat16)
        seqs = [batch_q[i] + batch_r[i] for i in mb]
        lens = [len(x) for x in seqs]
        Ltot = next(n for n in PACK_SIZES if n >= sum(lens)); packer = packers[Ltot]
        flat = sum(seqs, []); flat += [eng.eos] * (Ltot - len(flat))
        ids = torch.tensor(flat, device=dev)
        bm = packer.pack(lens)
        # predictor positions (pos j predicts token j+1) of the response tokens, in packed coordinates
        sel, o = [], 0
        for j, i in enumerate(mb):
            ql, rl = len(batch_q[i]), len(batch_r[i])
            sel.append(torch.arange(o + ql - 1, o + ql + rl - 1)); o += lens[j]
        sel = torch.cat(sel).to(dev); tgt = ids[sel + 1]
        feat = fwrite = None
        if imp is not None:
            with torch.no_grad():
                Tm = max(lens)                               # implant output at every suffix position (exact decode values)
                full = torch.tensor([x + [eng.eos] * (Tm - len(x)) for x in seqs], device=dev)
                full = torch.cat([eng.prefix_ids[-Kt:][None].expand(len(seqs), -1), full], 1)
                eb = imp.full(W['E'][full])[:, Kt:]
                es = [eb[j, :n] for j, n in enumerate(lens)]
                e = torch.cat(es + [torch.zeros(Ltot - sum(lens), imp.n_out, device=dev)])[None]
            feat = e
            if a.arm != '3b': fwrite = e @ imp.D
        hn = Q.fwd_packed(Wt, ids, packer.pos, bm, eng.cos_t, eng.sin_t, leaves, links=links, feat=feat, final_write=fwrite)
        lp = lp_of_hidden(Wt, hn[sel], tgt)
        old = torch.cat([torch.tensor(old_lp[i], device=dev) for i in mb])
        A = torch.cat([torch.full((len(batch_r[i]),), adv[i], device=dev) for i in mb])
        ratio = torch.exp(torch.clamp(lp - old, -20, 20))
        if a.loss == 'cispo':
            per = -torch.clamp(ratio, max=a.cispo_max).detach() * A * lp
        else:
            per = -torch.minimum(ratio * A, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A)
        if W_ref is not None:
            with torch.no_grad():
                hr = Q.fwd_packed(W_ref, ids, packer.pos, bm, eng.cos_t, eng.sin_t, pkv_ref, links=links, feat=feat, final_write=fwrite)
                rlp = lp_of_hidden(W_ref, hr[sel], tgt)
            d = rlp - lp
            klt = torch.exp(d) - d - 1
            per = per + a.kl * klt
            st['kl'] += float(klt.detach().sum()) / tot_tok
        loss = per.sum() / norm
        loss.backward()
        st['loss'] += float(loss.detach())
        with torch.no_grad():
            st['ratio'] += float(ratio.sum()) / tot_tok
            st['clipfrac'] += float(((ratio - 1).abs() > a.clip).float().sum()) / tot_tok
    torch.autograd.backward([t for kv in pkv for t in kv], [t.grad if t.grad is not None else torch.zeros_like(t) for kv in leaves for t in kv])
    gn = torch.nn.utils.clip_grad_norm_([P32[k] for k in names] + link_params, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    st['grad_norm'] = float(gn); st['train_s'] = time.time() - t0
    return st


# ------------------------------------------------------------------ main loop
t_start = time.time()
res0 = evaluate(devset); res0.update(u=0, t=0); log({'dev': res0}); print('dev@0', res0, flush=True)
if imp is not None:
    r = evaluate(devset, use_implant=False, tag='devhost'); r.update(u=0); log({'dev_host_curve': r}); print('host-only@0', r, flush=True)
for u in range(1, a.updates + 1):
    for g, b in zip(opt.param_groups, base_lrs): g['lr'] = b * min(1.0, u / a.warmup)
    batch = random.sample(pool, a.prompts)
    qids = [C.q_ids(tok, r['q']) for r in batch]
    t0 = time.time()
    out = eng.generate(qids, G=a.G, T=a.T, max_new=a.max_new)
    t_roll = time.time() - t0
    R = []; fmt = 0
    for j, t in enumerate(out['tokens']):
        s = C.score(tok, t, batch[j // a.G]['gold']); R.append(s[0]); fmt += s[1]
    Rt = torch.tensor(R).view(a.prompts, a.G)
    mean = Rt.mean(1, keepdim=True); std = Rt.std(1, keepdim=True)
    keep = (std[:, 0] > 0)
    if a.loss == 'drgrpo':
        advt = Rt - mean
    else:
        advt = (Rt - mean) / (Rt[keep].std() + 1e-4 if keep.any() else 1.0)
    idx = [p * a.G + g for p in range(a.prompts) if keep[p] for g in range(a.G)]
    kinds = [batch[j // a.G]['kind'] for j in range(len(R))]
    rec = dict(u=u, t=time.time() - t_start, reward=float(Rt.mean()), fmt=fmt / len(R),
               r_gsm=sum(r for r, k in zip(R, kinds) if k == 'gsm') / max(1, kinds.count('gsm')),
               r_scaled=sum(r for r, k in zip(R, kinds) if k == 'scaled') / max(1, kinds.count('scaled')),
               kept_groups=int(keep.sum()), len=sum(map(len, out['tokens'])) / len(R),
               writes=sum(sum(w) for w in out['wflag']) / len(R), calls=sum(calls_of(w) for w in out['wflag']) / len(R),
               roll_s=t_roll, roll_tok_s=out['stats']['tok_s'], gen_tokens=out['stats']['tokens'],
               mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    if idx:
        st = train_step([qids[j // a.G] for j in idx], [out['tokens'][j] for j in idx], [out['logp'][j] for j in idx],
                        [float(advt.view(-1)[j]) for j in idx], len(R))
        rec.update(st)
        sync_weights()
    if links:
        rec['link_gate_l2'] = {str(k): float(lk.gate.detach().norm()) for k, lk in links.items()}
        rec['link_W_l2'] = {str(k): float(lk.W.detach().norm()) for k, lk in links.items()}
    log(rec)
    if u % 5 == 0 or u == 1:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    if u % a.dev_every == 0 or u == a.updates:
        r = evaluate(devset); r.update(u=u, t=time.time() - t_start); log({'dev': r}); print('dev', r, flush=True)
        if imp is not None and u != a.updates:
            r = evaluate(devset, use_implant=False, tag='devhost'); r.update(u=u); log({'dev_host_curve': r})
    if u % a.ckpt_every == 0 or u == a.updates:
        save(f'step_{u:06d}')
if imp is not None:
    r = evaluate(devset, use_implant=False, tag='devhost'); r.update(u=a.updates); log({'dev_hostonly': r}); print('host-only', r, flush=True)
log({'DONE_RUN': time.time() - t_start}); print('DONE', time.time() - t_start, flush=True)
