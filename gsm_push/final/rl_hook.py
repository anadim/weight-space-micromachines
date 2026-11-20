# GRPO / Dr.GRPO RL of the HOST with a frozen single-call implant stack attached (gsm_push RL_STACK study).
# exec'd inside dig_san/tw_dump_lean.py --eval_ckpt CKPT [--host_ckpt base] --arm 2 --fused 0 --tok_budget 16384
# (names in scope: eng, imp, P32, names, opt, base_lrs, sync_weights, lp_of_hidden, packers, PACK_SIZES, Lp, pool, C, tok, a,
#  shot_q, Q, W, dev).  Same hyper-parameters as the a2_u300 runs: lr 1e-6 (warmup 5), AdamW (0.9, 0.95), clip 0.2, KL 0,
#  drgrpo, grad-norm clip 1.0, T = 1, split-A pool (gsm + scaled), G = 8.
# RL_ARM:
#   B     S2F control: calc_s4 + compiled chain2 + compiled sanity (neg / non-int, input token) + weight-space forget,
#         one attempt per row; RL_PROMPTS x G rows (token-matched to the serial arms).
#   A     serial stack (comp_serial.SerialW), N attempts, reward = final voted answer.
#   A1    = A' : + attempt-level advantage (each attempt's own answer, within the group) added to the final-vote advantage.
# Teacher forcing reproduces the rollout's attention exactly: every segment (attempt / repair) is its own packed document
# [question + segment tokens] (the forget hides all other segments), with the rollout's RoPE positions. Tokens written by
# an implant (calculator digits / veto, phrases, voter line: wflag bits 1|2|4) are masked out of the loss.
import json as _j, os as _o, sys as _s, time as _t, random as _r, re as _re, math as _m
_s.path.insert(0, _o.environ['STK_DIR']); _o.environ.setdefault('SAN_DIR', _o.environ['STK_DIR'])
import engine as EM
from torch.utils.checkpoint import checkpoint as _ckpt
import torch.nn.functional as F
ARM = _o.environ['RL_ARM']; RUN = _o.environ['RL_RUN']; _o.makedirs(RUN, exist_ok=True)
UPD = int(_o.environ.get('RL_UPDATES', 300)); PROMPTS = int(_o.environ.get('RL_PROMPTS', 64)); G = int(_o.environ.get('RL_G', 8))
NATT = int(_o.environ.get('RL_N', 4)); DEV_EVERY = int(_o.environ.get('RL_DEV_EVERY', 25)); SEED = int(_o.environ.get('RL_SEED', 0))
RESUME = f"{_o.environ.get('WSM_RESUME_DIR', '/dev/shm')}/rlstack_{_o.path.basename(RUN.rstrip('/'))}.pt"
RESUME_EVERY = int(_o.environ.get('RL_RESUME_EVERY', 5))
ACAP = a.max_new
LOGF = open(f'{RUN}/steps.jsonl', 'a')
def LOG(o): LOGF.write(_j.dumps(o) + '\n'); LOGF.flush()
def P(*x): print('RL', ARM, *x, flush=True)


# ------------------------------------------------------------------------------------------------ the frozen implant stack
def dual_implant():
    import chain2
    chain2.register(tok)
    from implant import CalcImplant
    from implant2 import DualImplant
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to('cuda').eval()
    ch = CalcImplant(hf, tok, a.B_max, machine='chain2', fused=False)
    del hf; torch.cuda.empty_cache()
    return DualImplant(imp, ch)


DUAL = dual_implant()
with torch.no_grad():                    # the lean harness keeps the fp32 host on the CPU: train it on the GPU
    for k in names: P32[k] = P32[k].detach().to(dev).requires_grad_(True)
opt = torch.optim.AdamW([{'params': [P32[k] for k in names], 'lr': a.lr}], betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
base_lrs = [a.lr]
eng.imp = DUAL
if ARM == 'B':
    import stackh as SH
    eng.__class__ = SH.StackHD
    eng.c2_setup(); eng.set_c2('off')
    eng.comp_setup('fwntf', maxf=2, cap0=ACAP, extra=ACAP, cache=_o.environ.get('DFA_CACHE', f"{_o.environ['STK_DIR']}/dfa_tables.pt"))
    eng.stack_setup()
    eng.set_comp(True, forget=True); eng.m_on = False            # compiled triggers only (no stop-token hook)
    GEN_MAX = ACAP + 2 * ACAP
else:
    import comp_serial as CSR
    eng.__class__ = CSR.SerialWD
    eng.c2_setup(); eng.set_c2('off')
    eng.sw_setup(N=NATT, acap=ACAP, posreset=_o.environ.get('RL_POSRESET', '0') == '1',
                 probe=f"{_o.environ['STK_DIR']}/verifier_probe_L18.pt", cache=_o.environ.get('DFA_CACHE', f"{_o.environ['STK_DIR']}/dfa_tables.pt"))
    GEN_MAX = eng.sw_maxn
eng.graphs = {}; eng.pool = None; eng._outshape = None
assert _o.environ.get('C2_NOCOMPACT') == '1'
assert max(len(C.q_ids(tok, r['q'])) for r in pool) + GEN_MAX <= eng.S_max, (GEN_MAX, eng.S_max)
P('arm', ARM, 'gen_max', GEN_MAX, 'S_max', eng.S_max, 'prompts', PROMPTS, 'G', G, 'pool', len(pool))


def last_answer(t):
    txt = tok.decode(t).replace('<|endoftext|>', '')
    ms = list(_re.finditer(C.NUM, txt)); return ms[-1].group(1).replace(',', '') if ms else None


def segments(t, w):
    """split one output stream at the segment starts (fire bit 8 that starts a new attempt / repair). Returns list of
    (start, end) index ranges; a segment made only of implant-written tokens (the voter line) is merged into the previous."""
    st = [0] + [i for i, x in enumerate(w) if (x & 8) and i > 0]
    rng = [(s_, e_) for s_, e_ in zip(st, st[1:] + [len(t)])]
    out = []
    for s_, e_ in rng:
        if out and all(w[i] & 7 for i in range(s_, e_)) and ARM != 'B' and eng.sw_is_final(t, w, s_):
            out[-1] = (out[-1][0], e_)
        else: out.append((s_, e_))
    return out


def seg_answers(t, w, sg):
    """each segment's own '#### n' (first in the segment), for attempt-level credit."""
    res = []
    for s_, e_ in sg:
        txt = tok.decode(t[s_:e_]).replace('<|endoftext|>', '')
        m = _re.search(C.NUM, txt); res.append(m.group(1).replace(',', '') if m else None)
    return res


# ---------------------------------------------------------------------------------------------------------- training
def train(docs, n_rows, norm_len):
    """docs: list of dict(q, r, pos (absolute RoPE positions of r, or None = contiguous), lp, mask, adv). Dr.GRPO."""
    t0 = _t.time()
    W0 = Q.fuse(P32, torch.bfloat16)
    pkv = Q.prefix_kv_of(W0, eng.prefix_ids, eng.cos_t, eng.sin_t)
    leaves = [(k.detach(), v.detach()) for k, v in pkv]
    del W0
    order = sorted(range(len(docs)), key=lambda i: -(len(docs[i]['q']) + len(docs[i]['r'])))
    bins = []
    for i in order:
        L = len(docs[i]['q']) + len(docs[i]['r'])
        assert L <= a.tok_budget
        for bn in bins:
            if bn[1] + L <= a.tok_budget: bn[0].append(i); bn[1] += L; break
        else: bins.append([[i], L])
    norm = n_rows * norm_len
    ntrain = sum(sum(d['mask']) for d in docs)
    st = dict(loss=0.0, clipfrac=0.0, ratio=0.0, ntok_train=ntrain, n_docs=len(docs), n_micro=len(bins))
    for mb, _ in bins:
        Wt = Q.fuse(P32, torch.bfloat16)
        seqs = [docs[i]['q'] + docs[i]['r'] for i in mb]; lens = [len(x) for x in seqs]
        Ltot = next(n for n in PACK_SIZES if n >= sum(lens)); packer = packers[Ltot]
        flat = sum(seqs, []); flat += [eng.eos] * (Ltot - len(flat))
        ids = torch.tensor(flat, device=dev)
        bm = packer.pack(lens)
        pos = packer.pos.clone()
        sel, o, old, A, M = [], 0, [], [], []
        for j, i in enumerate(mb):
            d = docs[i]; ql, rl = len(d['q']), len(d['r'])
            if d['pos'] is not None: pos[o + ql:o + ql + rl] = torch.tensor(d['pos'], device=dev)
            sel.append(torch.arange(o + ql - 1, o + ql + rl - 1)); o += lens[j]
            old += d['lp']; A += [d['adv']] * rl; M += d['mask']
        packer.pos.copy_(pos)
        sel = torch.cat(sel).to(dev); tgt = ids[sel + 1]
        hn = Q.fwd_packed(Wt, ids, packer.pos, bm, eng.cos_t, eng.sin_t, leaves)
        lp = lp_of_hidden(Wt, hn[sel], tgt)
        old = torch.tensor(old, device=dev); A = torch.tensor(A, device=dev); M = torch.tensor(M, device=dev, dtype=torch.bool)
        ratio = torch.exp(torch.clamp(lp - old, -20, 20))
        per = -torch.minimum(ratio * A, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A)
        loss = (per * M).sum() / norm
        loss.backward()
        with torch.no_grad():
            st['loss'] += float(loss.detach())
            st['ratio'] += float((ratio * M).sum()) / max(1, ntrain)
            st['clipfrac'] += float(((((ratio - 1).abs() > a.clip) & M).float()).sum()) / max(1, ntrain)
    tp = [P32[k] for k in names]
    gn = torch.nn.utils.clip_grad_norm_(tp, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    sync_weights()
    st['grad_norm'] = float(gn); st['train_s'] = _t.time() - t0
    return st


# ---------------------------------------------------------------------------------------------------------- rollouts
def rollout(qids, greedy=False, Gr=1, seed=None):
    if seed is not None: torch.manual_seed(seed); eng.seed.fill_(seed)
    eng.graphs = eng.graphs                      # (graphs are kept across calls: same max_new every call)
    return eng.generate(qids, G=Gr, T=1.0, greedy=greedy, max_new=GEN_MAX)


def build_docs(qids, out, adv_rows, adv_seg=None):
    docs = []
    for j, (t, w, lp) in enumerate(zip(out['tokens'], out['wflag'], out['logp'])):
        q = qids[j // G]; sg = segments(t, w)
        for k, (s_, e_) in enumerate(sg):
            r = t[s_:e_]; mk = [int((x & 7) == 0) for x in w[s_:e_]]
            if not any(mk): continue
            pos = [eng.Lp + len(q) + i for i in range(s_, e_)] if not getattr(eng, 'sw_posreset', True) and ARM != 'B' else None
            if ARM == 'B': pos = [eng.Lp + len(q) + i for i in range(s_, e_)]   # S2F has no position reset
            ad = adv_rows[j] + (adv_seg[j][k] if adv_seg is not None else 0.0)
            if ad == 0.0: continue
            docs.append(dict(q=q, r=r, pos=pos, lp=lp[s_:e_], mask=mk, adv=ad))
    return docs


# ---------------------------------------------------------------------------------------------------------- dev eval
def dev_items(nm):
    f = lambda it: [r for r in it if len(C.q_ids(tok, r['q'])) + ACAP <= 544]
    it = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(0).shuffle(it)
    return it[:600] if nm == 'dev' else it[600:900]


DEV = {nm: dev_items(nm)[:int(_o.environ.get('RL_DEVN', 100000))] for nm in ('dev', 'dev2')}


def evaluate(u):
    rec = dict(u=u)
    for nm, items in DEV.items():
        qd = [C.q_ids(tok, r['q']) for r in items]
        if ARM == 'B':
            o = rollout(qd, greedy=True)
            oks = [float(C.correct(last_answer(t), r['gold'])) for t, r in zip(o['tokens'], items)]
            rec[f'{nm}_acc'] = 100 * sum(oks) / len(oks)
            rec[f'{nm}_tok'] = sum(map(len, o['tokens'])) / len(oks)
            rec[f'{nm}_fires'] = 100 * sum(any(x & 8 for x in w) for w in o['wflag']) / len(oks)
        else:
            rec.update({f'{nm}_{k}': v for k, v in eng.sw_dev_eval(qd, items, rollout, C, tok, seed=4242).items()})
    LOG({'dev': rec}); P('dev', _j.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()}))
    return rec


# ---------------------------------------------------------------------------------------------------------- main loop
def resume_save(u):
    st = dict(u=u, host={k: P32[k].detach().cpu() for k in names}, opt=opt.state_dict(),
              rng=dict(py=_r.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    torch.save(st, RESUME + '.tmp'); _o.replace(RESUME + '.tmp', RESUME)


def save_ckpt(tag):
    from safetensors.torch import save_file
    d = f'{RUN}/{tag}'; _o.makedirs(d, exist_ok=True)
    save_file({k: P32[k].detach().to(torch.bfloat16).cpu().contiguous() for k in names}, f'{d}/model.safetensors')
    P('saved', d)


START = 1
_r.seed(SEED); torch.manual_seed(SEED)
if _o.path.exists(RESUME):
    st = torch.load(RESUME, map_location='cpu', weights_only=False)
    with torch.no_grad():
        for k in names: P32[k].copy_(st['host'][k].to(P32[k].device))
    opt.load_state_dict(st['opt']); _r.setstate(st['rng']['py']); torch.set_rng_state(st['rng']['torch'])
    torch.cuda.set_rng_state_all(st['rng']['cuda']); START = st['u'] + 1; del st
    sync_weights(); LOG({'resumed_from': START - 1}); P('RESUMED', START - 1)
if START == 1: evaluate(0)
for u in range(START, UPD + 1):
    for g_, b_ in zip(opt.param_groups, base_lrs): g_['lr'] = b_ * min(1.0, u / a.warmup)
    batch = _r.sample(pool, PROMPTS); qids = [C.q_ids(tok, r['q']) for r in batch]
    t0 = _t.time(); out = rollout(qids, Gr=G, seed=10000 + u); t_roll = _t.time() - t0
    n = len(out['tokens'])
    Rf = [float(C.correct(last_answer(t), batch[j // G]['gold'])) for j, t in enumerate(out['tokens'])]
    Rt = torch.tensor(Rf).view(PROMPTS, G); advt = (Rt - Rt.mean(1, keepdim=True)).view(-1).tolist()
    adv_seg = None; rec = dict(u=u, reward=float(Rt.mean()))
    if ARM != 'B':
        rec.update(eng.sw_roll_stats(out))
    if ARM == 'A1':               # attempt-level credit: each segment's own answer vs the group's mean attempt reward
        segA = []
        for j, (t, w) in enumerate(zip(out['tokens'], out['wflag'])):
            sg = segments(t, w); ans = seg_answers(t, w, sg)
            segA.append([float(C.correct(x, batch[j // G]['gold'])) for x in ans])
        adv_seg = []
        for p_ in range(PROMPTS):
            allr = [x for j in range(p_ * G, (p_ + 1) * G) for x in segA[j]]
            mu = sum(allr) / max(1, len(allr))
            for j in range(p_ * G, (p_ + 1) * G): adv_seg.append([x - mu for x in segA[j]])
        rec['attempt_reward'] = sum(sum(x) for x in segA) / max(1, sum(len(x) for x in segA))
    docs = build_docs(qids, out, advt, adv_seg)
    rec.update(len=sum(map(len, out['tokens'])) / n, roll_s=t_roll, gen_tokens=out['stats']['tokens'],
               kept_groups=int(((Rt.std(1) > 0)).sum()), mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    if docs:
        rec.update(train(docs, n, ACAP * NATT if ARM != 'B' else ACAP))   # same per-token weight for every arm
    LOG(rec)
    if u % 5 == 0 or u <= 2: P(_j.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
    if u % DEV_EVERY == 0 or u == UPD: evaluate(u)
    if u % 100 == 0 or u == UPD: save_ckpt(f'step_{u:06d}')
    if u % RESUME_EVERY == 0 and u < UPD: resume_save(u)
LOG({'DONE_RUN': True}); P('DONE')
if _o.path.exists(RESUME): _o.remove(RESUME)
print('RLDONE', flush=True)
