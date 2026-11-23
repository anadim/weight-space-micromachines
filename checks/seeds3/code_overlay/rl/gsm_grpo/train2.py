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
ap.add_argument('--arm', default='1', choices=['1', '2', '3', '3b', 'R', '4', '4c', '4cRL', '4s'])
ap.add_argument('--soft_init', default='A', choices=['A', 'B']); ap.add_argument('--soft_lr', type=float, default=1e-3)
ap.add_argument('--det_lr', type=float, default=1e-4); ap.add_argument('--det_T', type=float, default=1.0)
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
ap.add_argument('--S_max', type=int, default=544)
ap.add_argument('--warmup', type=int, default=5); ap.add_argument('--fused', type=int, default=1)
ap.add_argument('--resume_every', type=int, default=0, help='seeds3 patch: rolling resume state every N updates (0 = off)')
ap.add_argument('--resume_dir', default='', help='seeds3 patch: resume state dir (default run_dir/resume)')
a = ap.parse_args()
os.makedirs(a.run_dir, exist_ok=True)
json.dump(vars(a), open(f'{a.run_dir}/config.json' if not os.path.exists(f'{a.run_dir}/config.json') else f'{a.run_dir}/config_resumed.json', 'w'), indent=1)
random.seed(a.seed); torch.manual_seed(a.seed)
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, shot_q = C.prefix_ids(tok); Lp = len(pre)
P32 = Q.load_hf_params(dtype=torch.float32)
for v in P32.values(): v.requires_grad_(True)
names = list(P32)
W = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16)      # rollout copy (updated in place)
W_ref = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16) if a.kl > 0 else None
imp = None; rt = None
if a.arm != '1':
    hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to(dev).eval()
    imp = CalcImplant(hf, tok, a.B_max, fused=bool(a.fused), random=(a.rand if a.arm == 'R' else None))
    rt = None
    if a.arm in ('4', '4c', '4cRL', '4s'):             # + compiled mistake detector / rethink (gated hybrid, cap 300)
        from rethink_step import RethinkStep
        rt = RethinkStep(hf, tok, a.B_max)
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
det_opt = None
softV = soft_opt = softV_b = None
if a.arm == '4s':                                      # soft rethink: K trainable input vectors replace the phrase embeddings
    E0 = Q.load_hf_params(dtype=torch.float32)['model.embed_tokens.weight'] if False else P32['model.embed_tokens.weight'].detach()
    ph = torch.tensor(rt.phrase_ids, device=dev)
    init = E0[ph].clone()
    if a.soft_init == 'B':
        g = torch.Generator(device='cpu').manual_seed(4321)
        r_ = torch.randn(init.shape, generator=g).to(dev)
        init = r_ / r_.norm(dim=-1, keepdim=True) * init.norm(dim=-1, keepdim=True).mean()
    softV = init.clone().requires_grad_(True); softV0 = init.clone()
    soft_opt = torch.optim.Adam([softV], lr=a.soft_lr, eps=1e-12)
    softV_b = softV.detach().to(torch.bfloat16).contiguous()
if a.arm in ('4c', '4cRL', '4s'):                       # co-trained detector (its own Adam)
    for p in rt.params(): p.requires_grad_(True)
    det_init = [p.detach().double().clone() for p in rt.params()]
    det_opt = torch.optim.Adam(rt.params(), lr=a.det_lr)
    rt.temp.fill_(a.det_T)
eng = Engine(W, pre, tok, implant=imp, links=links, B_max=a.B_max, S_max=a.S_max, use_write=(a.arm != '3b'), rethink=rt, soft_V=softV_b)
WHICH = 'last' if a.arm.startswith('4') else 'first'          # rethink arm: its second solution's '####' counts (run_gsm_v2)

pool = C.load_train('A', shot_q=shot_q)
pool = [r for r in pool if len(C.q_ids(tok, r['q'])) + a.max_new <= a.S_max]
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


def rethink_stats(toks, rtf, golds):
    """Per answer: did the rethink circuit write; if it wrote AFTER a first '####' answer (end path): did the final ('last')
    answer differ, fix or break the first one. Fires before any '####' (hash path) have no first answer to compare."""
    import re as _re
    n = len(toks); fired = hashf = endf = changed = fixed = broke = 0
    for ids, f, g in zip(toks, rtf, golds):
        if not any(f): continue
        fired += 1
        txt = tok.decode(ids).replace('<|endoftext|>', '').split('\n\n')[0]
        ms = _re.findall(C.NUM, txt)
        if len(ms) >= 2:
            endf += 1
            a0, a1 = ms[0].replace(',', ''), ms[-1].replace(',', '')
            c0, c1 = C.correct(a0, g), C.correct(a1, g)
            changed += (a0 != a1); fixed += (not c0 and c1); broke += (c0 and not c1)
        else:
            hashf += 1
    return dict(rt_fire=fired / n, rt_fire_hash=hashf / n, rt_fire_end=endf / n, rt_changed=changed / n,
                rt_fixed=fixed / n, rt_broke=broke / n)


def soft_report(u):
    with torch.no_grad():
        E_ = P32['model.embed_tokens.weight'].detach()
        En = E_ / E_.norm(dim=-1, keepdim=True)
        out = []
        for p in range(softV.shape[0]):
            v = softV[p].detach(); c = En @ (v / v.norm())
            top = torch.topk(c, 5)
            out.append(dict(p=p, init_tok=tok.decode([rt.phrase_ids[p]]), drift=float((v - softV0[p]).norm()), norm=float(v.norm()),
                            nearest=[(tok.decode([int(i)]), round(float(s_), 3)) for s_, i in zip(top.values, top.indices)]))
        return dict(u=u, vectors=out)


def calls_of(wf):
    return sum(1 for i in range(len(wf)) if wf[i] and (i == 0 or not wf[i - 1]))


def _auroc(pos, neg):
    if not pos or not neg: return float('nan')
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg]); r = 0.0; cn = 0
    for v, lab in allv:
        if lab == 0: cn += 1
        else: r += cn
    return r / (len(pos) * len(neg))


def detector_audit(items):
    """Current detector on the current host's FIRST answers (rethink read-only, calc on): AUROC + fire on wrong/right."""
    rt.passive = True
    out = eng.generate([C.q_ids(tok, r['q']) for r in items], G=1, greedy=True, max_new=a.max_new)
    rt.passive = False
    rr = out['rt_rec']; res = {}
    if rr is None: return res
    ok = [C.score(tok, t_, r['gold'], 'first')[0] for t_, r in zip(out['tokens'], items)]
    idx = [i for i in range(len(items)) if rr['has'][i] > 0.5]
    sB = rr['sB'].tolist(); s0 = rr['s0'].tolist(); fire = [bool(rr['dB'][i] > 0.5) or bool(rr['dA'][i] > 0.5) for i in range(len(items))]
    W_ = [i for i in idx if not ok[i]]; R_ = [i for i in idx if ok[i]]
    res = dict(aud_auroc_sB=_auroc([sB[i] for i in W_], [sB[i] for i in R_]), aud_auroc_s0=_auroc([s0[i] for i in R_], [s0[i] for i in W_]),
               aud_fire_wrong=sum(fire[i] for i in W_) / max(1, len(W_)), aud_fire_right=sum(fire[i] for i in R_) / max(1, len(R_)),
               aud_acc_first=100 * len(R_) / max(1, len(idx)))
    return res


def evaluate(items, use_implant=True, tag='dev'):
    """Greedy eval (run_gsm protocol). use_implant=False: host only (implant detached, links off)."""
    torch.cuda.empty_cache()
    e2 = eng; eng.imp_on = use_implant
    qids = [C.q_ids(tok, r['q']) for r in items]
    out = e2.generate(qids, G=1, greedy=True, max_new=a.max_new)
    sc = [C.score(tok, t, r['gold'], WHICH) for t, r in zip(out['tokens'], items)]
    acc = lambda kind: 100 * sum(s[0] for s, r in zip(sc, items) if r['kind'] == kind) / max(1, sum(r['kind'] == kind for r in items))
    res = {f'{tag}_acc': 100 * sum(s[0] for s in sc) / len(sc), f'{tag}_gsm': acc('gsm'), f'{tag}_scaled': acc('scaled'),
           f'{tag}_fmt': 100 * sum(s[1] for s in sc) / len(sc),
           f'{tag}_writes': sum(sum(w) for w in out['wflag']) / len(sc), f'{tag}_calls': sum(calls_of(w) for w in out['wflag']) / len(sc),
           f'{tag}_len': sum(map(len, out['tokens'])) / len(sc),
           f'{tag}_rethink_frac': sum(1 for w in out['rtflag'] if any(w)) / len(sc)}
    if rt is not None and use_implant:
        res.update({f'{tag}_{k}': v for k, v in rethink_stats(out['tokens'], out['rtflag'], [r['gold'] for r in items]).items()})
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
    return torch.cat([checkpoint(chunk_lp, hn[c:c + 2048], tgt[c:c + 2048], use_reentrant=False) for c in range(0, hn.shape[0], 2048)])


def train_step(batch_q, batch_r, old_lp, adv, n_total_rows, forced=None, sidx=None):
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
        soft = None
        if softV is not None and sidx is not None:
            sp = sum([[-1] * len(batch_q[i]) + list(sidx[i]) for i in mb], []); sp += [-1] * (Ltot - len(sp))
            soft = (torch.tensor(sp, device=dev), softV)
        hn = Q.fwd_packed(Wt, ids, packer.pos, bm, eng.cos_t, eng.sin_t, leaves, links=links, feat=feat, final_write=fwrite, soft=soft)
        lp = lp_of_hidden(Wt, hn[sel], tgt)
        old = torch.cat([torch.tensor(old_lp[i], device=dev) for i in mb])
        A = torch.cat([torch.full((len(batch_r[i]),), adv[i], device=dev) for i in mb])
        if forced is not None:                         # tokens written by the rethink implant are environment, not policy
            A = A * (1 - torch.cat([torch.tensor(forced[i], device=dev, dtype=torch.float) for i in mb]))
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
    if soft_opt is not None:
        st['soft_grad'] = float(softV.grad.norm()) if softV.grad is not None else 0.0
        soft_opt.step(); soft_opt.zero_grad(set_to_none=True)
        with torch.no_grad(): softV_b.copy_(softV.detach().to(torch.bfloat16))
    st['grad_norm'] = float(gn); st['train_s'] = time.time() - t0
    return st


# ------------------------------------------------------------------ resume (seeds3 patch; arms 1/2/R only: no links / detector / soft state)
RES_BASE = a.resume_dir or f'{a.run_dir}/resume'


def save_resume(u):
    """fp32 params + full AdamW state + python/torch/cuda RNG + engine sampling seed, written to a temp dir then swapped in."""
    import shutil
    assert not links and det_opt is None and soft_opt is None, 'resume implemented for arms 1/2/R only'
    d = RES_BASE + '_new'; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    torch.save({k: P32[k].detach().cpu() for k in names}, f'{d}/params.pt')
    torch.save(opt.state_dict(), f'{d}/opt.pt')
    torch.save(dict(u=u, py_rng=random.getstate(), t_rng=torch.get_rng_state(), c_rng=torch.cuda.get_rng_state_all(),
                    eng_seed=eng.seed.detach().cpu().clone() if torch.is_tensor(getattr(eng, 'seed', None)) else None), f'{d}/meta.pt')
    open(f'{d}/DONE', 'w').close()
    shutil.rmtree(RES_BASE, ignore_errors=True); os.rename(d, RES_BASE)
    log({'resume_saved': u}); print('RESUME saved u', u, flush=True)


def load_resume():
    if not (a.resume_every and os.path.exists(f'{RES_BASE}/DONE')): return 0
    meta = torch.load(f'{RES_BASE}/meta.pt', weights_only=False)
    src = torch.load(f'{RES_BASE}/params.pt')
    with torch.no_grad():
        for k in names: P32[k].copy_(src[k].to(P32[k].device))
    del src
    opt.load_state_dict(torch.load(f'{RES_BASE}/opt.pt', weights_only=False))
    sync_weights()
    if meta['eng_seed'] is not None: eng.seed.copy_(meta['eng_seed'].to(eng.seed.device))
    random.setstate(meta['py_rng']); torch.set_rng_state(meta['t_rng']); torch.cuda.set_rng_state_all(meta['c_rng'])
    log({'resumed_from': meta['u']}); print('RESUMED from u', meta['u'], flush=True)
    return meta['u']


# ------------------------------------------------------------------ main loop
t_start = time.time()
START_U = load_resume()
if not START_U:
    res0 = evaluate(devset); res0.update(u=0, t=0)
    if softV is not None: log({'soft_vectors': soft_report(0)})
    if rt is not None: res0.update(detector_audit(devset))
    log({'dev': res0}); print('dev@0', res0, flush=True)
    if imp is not None:
        r = evaluate(devset, use_implant=False, tag='devhost'); r.update(u=0); log({'dev_host_curve': r}); print('host-only@0', r, flush=True)
for u in range(START_U + 1, a.updates + 1):
    for g, b in zip(opt.param_groups, base_lrs): g['lr'] = b * min(1.0, u / a.warmup)
    batch = random.sample(pool, a.prompts)
    qids = [C.q_ids(tok, r['q']) for r in batch]
    t0 = time.time()
    if det_opt is not None: rt.stoch.fill_(True)
    out = eng.generate(qids, G=a.G, T=a.T, max_new=a.max_new)
    if det_opt is not None: rt.stoch.fill_(False)
    t_roll = time.time() - t0
    R = []; fmt = 0
    for j, t in enumerate(out['tokens']):
        s = C.score(tok, t, batch[j // a.G]['gold'], WHICH); R.append(s[0]); fmt += s[1]
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
               rethink_rows=sum(1 for w in out['rtflag'] if any(w)) / len(R),
               **(rethink_stats(out['tokens'], out['rtflag'], [batch[j // a.G]['gold'] for j in range(len(R))]) if rt is not None else {}),
               roll_s=t_roll, roll_tok_s=out['stats']['tok_s'], gen_tokens=out['stats']['tokens'],
               mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    if idx:
        st = train_step([qids[j // a.G] for j in idx], [out['tokens'][j] for j in idx], [out['logp'][j] for j in idx],
                        [float(advt.view(-1)[j]) for j in idx], len(R), [out['rtflag'][j] for j in idx],
                        [out['sidx'][j] for j in idx])
        rec.update(st)
        sync_weights()
    if det_opt is not None and out['rt_rec'] is not None:
        rr = out['rt_rec']; has = rr['has'] > 0.5
        if has.any():
            rows = torch.nonzero(has)[:, 0]
            sB, sA, s0 = rt.heads(rr['feat'][rows])
            T_ = float(a.det_T); rc = rt.rc
            lpB = torch.nn.functional.logsigmoid((sB - rc.thB) / T_); lnB = torch.nn.functional.logsigmoid(-(sB - rc.thB) / T_)
            lpA = torch.nn.functional.logsigmoid((sA - rc.thA) / T_); lnA = torch.nn.functional.logsigmoid(-(sA - rc.thA) / T_)
            dB, dA = rr['dB'][rows].double(), rr['dA'][rows].double()
            logp = dB * lpB + (1 - dB) * lnB + (1 - dB) * (dA * lpA + (1 - dA) * lnA)
            Arow = advt.view(-1).to(dev)[rows].double()
            loss_d = -(Arow * logp).mean()
            st_aux = 0.0
            if a.arm in ('4c', '4s'):                      # auxiliary refit of the "right as it stands" head
                firsts = torch.tensor([C.score(tok, out['tokens'][int(j)], batch[int(j) // a.G]['gold'], 'first')[0] for j in rows], device=dev).double()
                mk = (dB == 0)
                if mk.any():
                    aux = torch.nn.functional.binary_cross_entropy_with_logits(s0[mk], firsts[mk])
                    loss_d = loss_d + aux; st_aux = float(aux)
            det_opt.zero_grad(set_to_none=True); loss_d.backward(); det_opt.step()
            rec.update(det_loss=float(loss_d), det_aux=st_aux, det_rows=int(len(rows)), det_fireB=float(dB.mean()),
                       det_W1_drift=float(sum((p.detach().double() - p0).norm() ** 2 for p, p0 in zip(rt.params(), det_init)) ** 0.5))
    if links:
        rec['link_gate_l2'] = {str(k): float(lk.gate.detach().norm()) for k, lk in links.items()}
        rec['link_W_l2'] = {str(k): float(lk.W.detach().norm()) for k, lk in links.items()}
    log(rec)
    if u % 5 == 0 or u == 1:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    if u % a.dev_every == 0 or u == a.updates:
        r = evaluate(devset); r.update(u=u, t=time.time() - t_start)
        if rt is not None and a.arm in ('4', '4c', '4cRL', '4s'): r.update(detector_audit(devset))
        log({'dev': r}); print('dev', r, flush=True)
        if softV is not None and (u % 100 == 0):
            log({'soft_vectors': soft_report(u)})
        if imp is not None and u != a.updates:
            r = evaluate(devset, use_implant=False, tag='devhost'); r.update(u=u); log({'dev_host_curve': r})
    if u % a.ckpt_every == 0 or u == a.updates:
        save(f'step_{u:06d}')
    if a.resume_every and (u % a.resume_every == 0 or u == START_U + 1 or u == a.updates): save_resume(u)
if imp is not None:
    r = evaluate(devset, use_implant=False, tag='devhost'); r.update(u=a.updates); log({'dev_hostonly': r}); print('host-only', r, flush=True)
log({'DONE_RUN': time.time() - t_start}); print('DONE', time.time() - t_start, flush=True)
