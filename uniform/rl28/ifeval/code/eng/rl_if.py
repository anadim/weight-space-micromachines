"""GRPO (Dr.GRPO) RL of the Qwen3 host on fresh IFEval-like prompts on the fast stack (rl24/ifeval, task [24b]/[26]).

Arms: P = plain host (no implant); I = host + the compiled 14-rule IFEval implant (full export, all-weights: registers updated
by fixed linear / ReLU / table-gather / hard-attention / vocab-readout ops; its only output is a row of a fixed table added to
the residual stream before the final norm; see gexec.py / ifengine.py). Reward = lm_eval IFEval checkers, prompt-level strict
(ifreward.py, seeded). Recipe (rl_hook / a2): 64 prompts x G8, T 1, AdamW (0.9, 0.95, wd 0, eps 1e-8) lr 1e-6 warmup 5,
grad-norm clip 1, ratio clip 0.2, KL 0, Dr.GRPO (sum / (rows * max_new)), adv = R - group mean, docs with adv != 0 only.
Trainer: fasttrain.Trainer (packed varlen Triton attention, fused LP head); arm I adds the rollout's recorded implant write
F[widx] at every sampled position (the same bf16 add as the rollout), tokens written by the implant (widx != V) are masked.

python rl_if.py --model M --arm P|I [--sd SD] --run DIR --seed S [--updates 300 ...]
"""
import argparse, json, os, random, shutil, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'stk'), os.path.join(HERE, '..', 'f14')]
import ifreward as RW

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base'); ap.add_argument('--arm', required=True, choices=['P', 'I'])
ap.add_argument('--sd', default=None); ap.add_argument('--run', required=True); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--updates', type=int, default=300); ap.add_argument('--prompts', type=int, default=64); ap.add_argument('--G', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-6); ap.add_argument('--warmup', type=int, default=5); ap.add_argument('--T', type=float, default=1.0)
ap.add_argument('--clip', type=float, default=0.2); ap.add_argument('--max_new', type=int, default=1024)
ap.add_argument('--dev_every', type=int, default=25); ap.add_argument('--dev_n', type=int, default=415)
ap.add_argument('--resume_every', type=int, default=2); ap.add_argument('--ckpt_every', type=int, default=100)
ap.add_argument('--fmt', default='chat'); ap.add_argument('--B_max', type=int, default=512); ap.add_argument('--tok_budget', type=int, default=32768)
ap.add_argument('--train', default=os.path.join(HERE, '..', 'gen', 'train.jsonl'))
ap.add_argument('--dev', default=os.path.join(HERE, '..', 'gen', 'dev.jsonl'))
ap.add_argument('--stop_after', type=int, default=0, help='exit after this many updates in this process (smoke / timing)')
ap.add_argument('--skip_eval0', type=int, default=0); ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--mem_frac', type=float, default=0, help='per-process CUDA memory cap (packed jobs, task [31])')
a = ap.parse_args()
assert (a.arm == 'I') == (a.sd is not None)
os.makedirs(a.run, exist_ok=True)
POOL = RW.Pool(a.workers)                          # (forked before CUDA is initialised)

import torch
if a.mem_frac: torch.cuda.set_per_process_memory_fraction(a.mem_frac)
import fastqwen as Q
import fasttrain as FT
from transformers import AutoTokenizer
from ifengine import IFEngine
import common_if as CI

LOGF = open(f'{a.run}/steps.jsonl', 'a')
def LOG(o): LOGF.write(json.dumps(o) + '\n'); LOGF.flush()
def P(*x): print('RL', a.arm, *x, flush=True)

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')            # (same vocab / special tokens as -Base; has the chat template)
TRAIN = [json.loads(l) for l in open(a.train)]
DEVS = [json.loads(l) for l in open(a.dev)][:a.dev_n]
pre, tr_full = CI.format_prompts(tok, [d['prompt'] for d in TRAIN], a.fmt)
_, dv_full = CI.format_prompts(tok, [d['prompt'] for d in DEVS], a.fmt)
Lp = len(pre)
maxq = max(len(f) for f in tr_full + dv_full) - Lp
dev = 'cuda'

P32 = Q.load_hf_params(a.model, device=dev)
names = list(P32)
with torch.no_grad():
    for k in names: P32[k] = P32[k].detach().requires_grad_(True)
W = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16)
imp = None
if a.arm == 'I':
    from gexec import GExec
    imp = GExec(torch.load(a.sd))
eng = IFEngine(W, tok, pre, imp=imp, B_max=a.B_max, S_max=maxq + a.max_new + 8,
               N_cap=max(a.prompts * a.G, len(DEVS)), max_new_cap=a.max_new)
for k_, v_ in dict(FT_ATTN='triton', FT_CKPT='1', FT_LP='1', FT_DEDUP='1', FT_COMPILE='1', FT_FUSED_ADAM='1').items():
    os.environ.setdefault(k_, v_)
trainer = FT.from_env(P32, names, eng, a)
trainer.impF = imp.F if imp is not None else None
opt = torch.optim.AdamW([{'params': [P32[k] for k in names], 'lr': a.lr}], betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
V = eng.V
NORM = a.max_new
RESUME = f'{a.run}/resume.pt'


def sync_weights():
    Wn = Q.fuse({k: v.detach() for k, v in P32.items()}, torch.bfloat16)
    for k in ('E', 'lnf'): W[k].copy_(Wn[k])
    for l, ln in zip(W['L'], Wn['L']):
        for k in l: l[k].copy_(ln[k])
    eng.refresh()


def rewards(docs, toks):
    texts = [tok.decode(t) for t in toks]
    return POOL.map(docs, texts, a.fmt), texts


# --------------------------------------------------------------------------------------------------------------- dev
def evaluate(u):
    t0 = time.time()
    qids = [f[Lp:] for f in dv_full]
    o = eng.generate(qids, G=1, T=1.0, max_new=a.max_new, greedy=True, full_ids=dv_full, seed=0)
    sc, _ = rewards(DEVS, o['tokens'])
    n = len(sc)
    rec = dict(u=u, dev_prompt_strict=100 * sum(s[0] for s in sc) / n,
               dev_inst_strict=100 * sum(sum(s[1]) for s in sc) / sum(len(s[1]) for s in sc),
               dev_prompt_loose=100 * sum(s[3] for s in sc) / n,
               dev_tok=sum(map(len, o['tokens'])) / n,
               dev_trunc=100 * sum(1 for t in o['tokens'] if t[-1] not in (151645, 151643)) / n,
               dev_fired=100 * sum(any(w != V for w in ws) for ws in o['widx']) / n,
               dev_s=time.time() - t0, dev_tok_s=o['stats']['tok_s'])
    LOG({'dev': rec}); P('dev', json.dumps({k: round(v, 3) for k, v in rec.items()}))
    return rec


# ------------------------------------------------------------------------------------------------------ train step
def _fused_adam_on():
    for g_ in opt.param_groups: g_['fused'] = True; g_['foreach'] = False
    for s_ in opt.state.values():
        if 'step' in s_: s_['step'] = s_['step'].to(dev, torch.float32)


def train(docs, n_rows):
    t0 = time.time()
    st = trainer.grads(docs, n_rows, NORM)
    tp = [P32[k] for k in names]
    gn = torch.nn.utils.clip_grad_norm_(tp, 1.0)
    if os.environ.get('FT_FUSED_ADAM', '1') == '1': _fused_adam_on()
    opt.step(); opt.zero_grad(set_to_none=True)
    sync_weights()
    st['grad_norm'] = float(gn); st['train_s'] = time.time() - t0
    return st


# ---------------------------------------------------------------------------------------------------------- resume
_BUF, _TH = {}, [None]


def _snap(x, key):
    if isinstance(x, torch.Tensor):
        if not x.is_cuda: return x.detach().clone()
        b = _BUF.get(key)
        if b is None or b.shape != x.shape or b.dtype != x.dtype:
            b = _BUF[key] = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        b.copy_(x.detach(), non_blocking=True); return b
    if isinstance(x, dict): return {k: _snap(v, (key, k)) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(_snap(v, (key, i)) for i, v in enumerate(x))
    return x


_SV = dict(block_s=0.0, write_s=None, skipped=0, saved_u=None)       # save timing, logged in the next update record


def resume_wait():
    if _TH[0] is not None: _TH[0].join(); _TH[0] = None


def resume_save(u, force=False):
    """Non-blocking resume save: state -> pinned CPU buffers, file written by a background thread (.tmp + rename).
    If the previous write is still running the save is skipped (no stall; the next due update tries again), unless force."""
    t0 = time.time()
    if _TH[0] is not None and _TH[0].is_alive() and not force:
        _SV['skipped'] += 1; _SV['block_s'] += time.time() - t0; return
    resume_wait()
    if shutil.disk_usage(a.run).free - sum(P32[k].numel() for k in names) * 12 < (250 << 30):   # [30](d): /work free >= 250 GB after the .tmp write
        _SV['skipped'] += 1; _SV['disk_skip'] = _SV.get('disk_skip', 0) + 1; _SV['block_s'] += time.time() - t0
        P('resume save SKIPPED: /work free would drop below 250 GB'); return
    st = dict(u=u, host=_snap({k: P32[k] for k in names}, 'h'), opt=_snap(opt.state_dict(), 'o'),
              rng=dict(batch=BRNG.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    torch.cuda.synchronize()
    def w():
        tw = time.time()
        torch.save(st, RESUME + '.tmp'); os.replace(RESUME + '.tmp', RESUME)
        _SV['write_s'] = time.time() - tw; _SV['saved_u'] = u
    _TH[0] = threading.Thread(target=w, daemon=False); _TH[0].start()
    _SV['block_s'] += time.time() - t0


def save_ckpt(tag):
    from safetensors.torch import save_file
    d = f'{a.run}/{tag}'; os.makedirs(d, exist_ok=True)
    save_file({k: P32[k].detach().to(torch.bfloat16).cpu().contiguous() for k in names}, f'{d}/model.safetensors')
    P('saved', d)


BRNG = random.Random(1000 + a.seed)                # batch order: same per seed for both arms (paired)
torch.manual_seed(a.seed)
START = 1
if os.path.exists(RESUME):
    st = torch.load(RESUME, map_location='cpu', weights_only=False)
    with torch.no_grad():
        for k in names: P32[k].copy_(st['host'][k].to(dev))
    opt.load_state_dict(st['opt']); BRNG.setstate(st['rng']['batch'])
    torch.set_rng_state(st['rng']['torch']); torch.cuda.set_rng_state_all(st['rng']['cuda'])
    START = st['u'] + 1; del st
    sync_weights(); LOG({'resumed_from': START - 1}); P('RESUMED', START - 1)
else:
    LOG({'config': vars(a), 'Lp': Lp, 'maxq': maxq, 'n_train': len(TRAIN), 'n_dev': len(DEVS), 'V': V,
         'env': {k: v for k, v in os.environ.items() if k.startswith(('FT_', 'RL_'))}})
P('arm', a.arm, 'model', a.model, 'start', START, 'S_max', eng.S_max, 'maxq', maxq)
if START == 1 and not a.skip_eval0: evaluate(0)


def set_lr(u):
    for g_ in opt.param_groups: g_['lr'] = a.lr * min(1.0, u / a.warmup)


done_here = 0
for u in range(START, a.updates + 1):
    t0 = time.time(); set_lr(u)
    bi = BRNG.sample(range(len(TRAIN)), a.prompts)
    batch = [TRAIN[i] for i in bi]; full = [tr_full[i] for i in bi]; qids = [f[Lp:] for f in full]
    out = eng.generate(qids, G=a.G, T=a.T, max_new=a.max_new, greedy=False, full_ids=full, seed=1_000_000 * a.seed + u)
    t_roll = time.time() - t0
    n = len(out['tokens'])
    sc, _ = rewards([batch[j // a.G] for j in range(n)], out['tokens'])
    Rt = torch.tensor([s[0] for s in sc]).view(a.prompts, a.G)
    adv = (Rt - Rt.mean(1, keepdim=True)).view(-1).tolist()
    docs = []
    for j in range(n):
        if adv[j] == 0.0: continue
        t, lp, w = out['tokens'][j], out['logp'][j], out['widx'][j]
        mk = [int(x == V) for x in w]
        if not any(mk): continue
        docs.append(dict(q=qids[j // a.G], r=t, pos=None, lp=lp, mask=mk, adv=adv[j], widx=w))
    t_rew = time.time() - t0 - t_roll
    rec = dict(u=u, reward=float(Rt.mean()), inst_strict=sum(sum(s[1]) for s in sc) / sum(len(s[1]) for s in sc),
               len=sum(map(len, out['tokens'])) / n, trunc=sum(1 for t in out['tokens'] if t[-1] not in (151645, 151643)) / n,
               fired=sum(any(x != V for x in w) for w in out['widx']) / n,
               forced_tok=sum(sum(x != V for x in w) for w in out['widx']), kept_groups=int((Rt.std(1) > 0).sum()),
               roll_s=t_roll, reward_s=t_rew, gen_tokens=out['stats']['tokens'], roll_tok_s=out['stats']['tok_s'],
               lr=opt.param_groups[0]['lr'])
    if docs: rec.update(train(docs, n))
    rec['update_s'] = time.time() - t0; rec['mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
    rec['wall'] = time.time(); rec['save_block_s'] = _SV['block_s']; rec['save_write_s'] = _SV['write_s']
    rec['save_skipped'] = _SV['skipped']; rec['saved_u'] = _SV['saved_u']; _SV['block_s'] = 0.0
    LOG(rec)
    if u <= 2 or u % 5 == 0: P(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
    if u % a.dev_every == 0 or u == a.updates: evaluate(u)
    if u % a.ckpt_every == 0 or u == a.updates: save_ckpt(f'step_{u:06d}')
    if u % a.resume_every == 0 and u < a.updates: resume_save(u)
    done_here += 1
    if a.stop_after and done_here >= a.stop_after: resume_wait(); P('STOP_AFTER', u); sys.exit(0)
resume_wait()
LOG({'DONE_RUN': True}); P('DONE')
if os.path.exists(RESUME): os.remove(RESUME)
print('RLDONE', flush=True)
