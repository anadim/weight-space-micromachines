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
RESUME = f"{_o.environ.get('RL_RESUME_DIR', '/dev/shm')}/rlstack_{_o.path.basename(RUN.rstrip('/'))}.pt"
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
    eng.comp_setup('fwntf', maxf=2, cap0=ACAP, extra=ACAP, cache=f"{_o.environ['STK_DIR']}/dfa_tables.pt")
    eng.stack_setup()
    eng.set_comp(True, forget=True); eng.m_on = False            # compiled triggers only (no stop-token hook)
    GEN_MAX = ACAP + 2 * ACAP
else:
    import comp_serial as CSR
    eng.__class__ = CSR.SerialWD
    eng.c2_setup(); eng.set_c2('off')
    eng.sw_setup(N=NATT, acap=ACAP, posreset=_o.environ.get('RL_POSRESET', '0') == '1',
                 probe=f"{_o.environ['STK_DIR']}/verifier_probe_L18.pt", cache=f"{_o.environ['STK_DIR']}/dfa_tables.pt")
    GEN_MAX = eng.sw_maxn
eng.graphs = {}; eng.pool = None; eng._outshape = None
assert _o.environ.get('C2_NOCOMPACT') == '1'
FASTROLL = _o.environ.get('RL_FASTROLL', '0') == '1'      # continuous-batching rollout loop (fastroll.py); default off
if FASTROLL:
    import fastroll as _FR
    _FR.enable(eng)
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
def train_grads(docs, n_rows, norm_len):
    """docs: list of dict(q, r, pos (absolute RoPE positions of r, or None = contiguous), lp, mask, adv). Dr.GRPO.
    Accumulates the gradient into P32[k].grad; returns the stats (the reference trainer)."""
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
    return st


FT_TRAINER = None
if _o.environ.get('RL_FASTTRAIN', '0') == '1':      # fast trainer (fasttrain.py); default off = the reference trainer
    import fasttrain as _FT
    FT_TRAINER = _FT.from_env(P32, names, eng, a)


def _fused_adam_on():
    """fused AdamW (one kernel over all params): switch the param group and move the step counters to the GPU
    (fused requires device steps; states loaded by torch.load(map_location='cpu') have CPU steps)."""
    for g_ in opt.param_groups: g_['fused'] = True; g_['foreach'] = False
    for s_ in opt.state.values():
        if 'step' in s_: s_['step'] = s_['step'].to(P32[names[0]].device, torch.float32)


def train(docs, n_rows, norm_len):
    t0 = _t.time()
    st = FT_TRAINER.grads(docs, n_rows, norm_len) if FT_TRAINER is not None else train_grads(docs, n_rows, norm_len)
    tp = [P32[k] for k in names]
    gn = torch.nn.utils.clip_grad_norm_(tp, 1.0)
    if _o.environ.get('FT_FUSED_ADAM', '0') == '1': _fused_adam_on()
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
ASYNC_SAVE = _o.environ.get('RL_ASYNC_SAVE', '0') == '1'
_SAVE_BUF, _SAVE_TH = {}, [None]


def _snap(x, key):
    """GPU tensor -> reused pinned host buffer, CPU tensor -> copy; containers recursively (RL_ASYNC_SAVE=1)."""
    if isinstance(x, torch.Tensor):
        if not x.is_cuda: return x.detach().clone()
        b = _SAVE_BUF.get(key)
        if b is None or b.shape != x.shape or b.dtype != x.dtype:
            b = _SAVE_BUF[key] = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        b.copy_(x.detach(), non_blocking=True); return b
    if isinstance(x, dict): return {k: _snap(v, (key, k)) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(_snap(v, (key, i)) for i, v in enumerate(x))
    return x


def resume_wait():
    if _SAVE_TH[0] is not None: _SAVE_TH[0].join(); _SAVE_TH[0] = None


_SAVE_INFO = {}


def resume_save(u):
    import threading as _th
    if ASYNC_SAVE and _SAVE_TH[0] is not None and _SAVE_TH[0].is_alive():
        _SAVE_INFO['save_skipped'] = u; return      # (previous write still in flight: skip; the file on disk stays complete)
    resume_wait()                                   # (the pinned buffers are reused: the previous write must be done)
    if not ASYNC_SAVE:
        st = dict(u=u, host={k: P32[k].detach().cpu() for k in names}, opt=opt.state_dict(),
                  rng=dict(py=_r.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
        torch.save(st, RESUME + '.tmp'); _o.replace(RESUME + '.tmp', RESUME); return
    st = dict(u=u, host=_snap({k: P32[k] for k in names}, 'h'), opt=_snap(opt.state_dict(), 'o'),
              rng=dict(py=_r.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    torch.cuda.synchronize()                        # snapshot complete before the next update mutates params / moments
    def w():
        t0 = _t.time(); _t.sleep(float(_o.environ.get('RL_SAVE_SLOW', 0)))    # (RL_SAVE_SLOW: test hook)
        torch.save(st, RESUME + '.tmp'); _o.replace(RESUME + '.tmp', RESUME); _SAVE_INFO['save_w_s'] = _t.time() - t0
    _SAVE_TH[0] = _th.Thread(target=w, daemon=False); _SAVE_TH[0].start()


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


def fastcheck(nq):
    """greedy dev: the old wave loop vs fastroll, same weights; token identity, accuracy, first-divergence log-probs."""
    items = DEV['dev'][:nq]; qd = [C.q_ids(tok, r['q']) for r in items]
    fast_cls = eng.__class__; old_cls = fast_cls.__bases__[0]
    def run(cls, c2on):
        eng.__class__ = cls; eng.c2_on = c2on; eng.graphs = {}; eng.pool = None; eng._outshape = None
        if cls is old_cls: eng.imp = eng.f_dual                  # old path: torch implants
        t0 = _t.time(); o = rollout(qd, greedy=True); return o, _t.time() - t0
    oo, t_o = run(old_cls, True)
    of, t_f = run(fast_cls, False)
    acc = lambda o: 100 * sum(float(C.correct(last_answer(t), r['gold'])) for t, r in zip(o['tokens'], items)) / nq
    div = []
    for j, (x, y) in enumerate(zip(oo['tokens'], of['tokens'])):
        if x != y:
            k = next((i for i, (u_, v_) in enumerate(zip(x, y)) if u_ != v_), min(len(x), len(y)))
            div.append(dict(row=j, pos=k, lp_old=oo['logp'][j][k] if k < len(x) else None,
                            lp_fast=of['logp'][j][k] if k < len(y) else None))
    dl = max((max((abs(p_ - q_) for p_, q_ in zip(a_, b_)), default=0.0) for a_, b_, x, y in
              zip(oo['logp'], of['logp'], oo['tokens'], of['tokens']) if x == y), default=0.0)
    rec = dict(fastcheck=dict(u=START - 1, n=nq, tok_identical=sum(x == y for x, y in zip(oo['tokens'], of['tokens'])),
                              wflag_identical=sum(x == y for x, y in zip(oo['wflag'], of['wflag'])),
                              max_abs_dlogp_identical_rows=dl, acc_old=acc(oo), acc_fast=acc(of),
                              s_old=t_o, s_fast=t_f, tok_s_old=oo['stats']['tok_s'], tok_s_fast=of['stats']['tok_s'],
                              n_div=len(div), divergences=div[:40]))
    if 'reg' in oo: rec['fastcheck']['reg_identical'] = {k: bool(torch.equal(oo['reg'][k], of['reg'][k])) for k in oo['reg']}
    LOG(rec); P('FASTCHECK', _j.dumps({k: v for k, v in rec['fastcheck'].items() if k != 'divergences'}))
    eng.__class__ = fast_cls; eng.c2_on = False; eng.graphs = {}; eng.pool = None; eng._outshape = None


if FASTROLL:
    LOG({'engine': 'fastroll', 'from_u': START - 1, 'K': eng.f_K, 'admit': eng.f_admit, 'B_max': eng.B_max,
         'tk_implants': eng.f_tk is not None, 'forget_skip': eng.f_fskip,
         'flags': {k: _o.environ.get(k) for k in ('FAST_COMPACT', 'FAST_OVL', 'FAST_ATT3', 'FAST_TK', 'FLEX_SMALL')}})
    P('ENGINE fastroll from u', START - 1)
    if int(_o.environ.get('RL_FASTCHECK', 0)) > 0: fastcheck(int(_o.environ['RL_FASTCHECK']))
_first_ratio_logged = False
if START == 1 and _o.environ.get('RL_SKIP_EVAL0', '0') != '1': evaluate(0)      # (skip: smoke tests only)
elif START > 1 and _o.environ.get('RL_EVAL_AT_START', '0') == '1': evaluate(START - 1)   # (debug: dev eval of a resumed state)
_prev = {}
for u in range(START, UPD + 1):
    tu0 = _t.time()
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
    rec.update(_prev); rec.update(_SAVE_INFO); _SAVE_INFO.clear(); rec['post_s'] = _t.time() - t0 - t_roll
    rec.update(len=sum(map(len, out['tokens'])) / n, roll_s=t_roll, gen_tokens=out['stats']['tokens'],
               kept_groups=int(((Rt.std(1) > 0)).sum()), mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    if _o.environ.get('RL_TRAIN_BENCH'):          # (bench: time / compare trainers on this real batch, then exit)
        exec(open(f"{_o.environ['STK_DIR']}/train_bench.py").read()); raise SystemExit(0)
    if docs:
        rec.update(train(docs, n, ACAP * NATT if ARM != 'B' else ACAP))   # same per-token weight for every arm
    if FASTROLL: rec['engine'] = 'fastroll'
    rec['pre_s'] = t0 - tu0; t_log = _t.time()
    LOG(rec)
    if not _first_ratio_logged and 'ratio' in rec:
        _first_ratio_logged = True; P('FIRST_UPDATE_RATIO', u, round(rec['ratio'], 6), 'fastroll' if FASTROLL else 'old')
    if u % 5 == 0 or u <= 2: P(_j.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
    if u % DEV_EVERY == 0 or u == UPD: evaluate(u)
    if u % 100 == 0 or u == UPD: save_ckpt(f'step_{u:06d}')
    if u % RESUME_EVERY == 0 and u < UPD: resume_save(u)
    _prev = dict(prev_wall_s=_t.time() - tu0, prev_tail_s=_t.time() - t_log)   # (tail = dev eval + ckpt + resume save)
resume_wait()
LOG({'DONE_RUN': True}); P('DONE')
if _o.path.exists(RESUME) and _o.environ.get('RL_KEEP_RESUME', '0') != '1': _o.remove(RESUME)
print('RLDONE', flush=True)
