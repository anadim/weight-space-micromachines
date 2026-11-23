"""Exactness / speed checks of the fast IFEval engine against the reference harness (f14/run_c.py decode loop: HF model + eager
Executor, left padding, bs 24, greedy, hook on the final-norm input during generation).

python check.py --model M --sd SD.pt --data DEV.jsonl --n 128 --max_new 1024 --out DIR [--fmt chat]
 1. reference greedy (HF + Executor) on the first n prompts: tokens, widx per step, top1-top2 logit gap per step; tok/s
 2. implant fuzz: the first reference batch's (ids, h) streams replayed through GExec
      (a) padded, same order as the reference           -> registers, memory, widx bitwise every call
      (b) unpadded prompts (masked steps), rows permuted -> widx on every live step (padding invariance)
 3. fast engine greedy (implant on), same prompts: token identity vs reference; first divergences with the reference gap
 4. fast engine timing at the RL shape (64 prompts x G8, T=1, sampled), implant on and off
"""
import argparse, json, os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'stk'), os.path.join(HERE, '..', 'f14')]
from execu import Executor
from transformers import AutoTokenizer, AutoModelForCausalLM
import fastqwen as Q
from gexec import GExec
from ifengine import IFEngine
import common_if as CI

ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--sd', required=True)
ap.add_argument('--data', required=True); ap.add_argument('--n', type=int, default=128); ap.add_argument('--max_new', type=int, default=1024)
ap.add_argument('--out', required=True); ap.add_argument('--fmt', default='chat'); ap.add_argument('--bs', type=int, default=24)
ap.add_argument('--skip_ref', type=int, default=0); ap.add_argument('--rl_shape', type=int, default=1)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LOG = open(f'{a.out}/check.jsonl', 'a')
def log(**k): print(json.dumps(k), flush=True); LOG.write(json.dumps(k) + '\n'); LOG.flush()

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
docs = [json.loads(l) for l in open(a.data)][:a.n]
pre, full = CI.format_prompts(tok, [d['prompt'] for d in docs], a.fmt)
sd = torch.load(a.sd)
EOS = [151645, 151643]

# ------------------------------------------------------------------------------------------------------- 1. reference
ref = {}
if not a.skip_ref:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    EXr = Executor(sd, device='cuda')
    state = {'on': False, 'rec': None}
    def _hook(mod, x):
        if not state['on']: return None
        h = x[0]; hl = h[:, -1]
        if state['rec'] is not None: state['rec'].append(('w', hl.clone()))
        W = EXr.write(hl)
        if not state.get('ig_on', True): return (h + W[:, None].to(h.dtype),)
        x = hl.float(); x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EXr.sd['voc.eps']) * EXr.sd['voc.g']
        lr = (x.to(EXr.E.dtype) @ EXr.E.T).float(); blk = EXr._last_blk.float()
        t2 = lr.topk(2, -1); gt = t2.values[:, 0] - t2.values[:, 1]
        a2 = (lr + blk * -1e30).topk(2, -1).values; ga = a2[:, 0] - a2[:, 1]
        btop = blk.gather(1, t2.indices[:, :1])[:, 0] > 0
        state['ig'] = torch.where(btop, torch.minimum(gt, ga), gt)   # implant readout near-tie gap (top, or alt if top blocked)
        return (h + W[:, None].to(h.dtype),)
    model.model.norm.register_forward_pre_hook(_hook)
    def cstep(ids):
        if state['rec'] is not None: state['rec'].append(('s', ids.clone()))
        EXr.step(ids)
    order = sorted(range(len(docs)), key=lambda i: (len(full[i]), docs[i]['key']))
    batches = [order[b:b + a.bs] for b in range(0, len(order), a.bs)]
    R_tok, R_w, R_gap, R_ig = {}, {}, {}, {}
    torch.cuda.synchronize(); t0 = time.time(); ntok = 0
    with torch.no_grad():
        for bi, idx in enumerate(batches):
            P = [full[i] for i in idx]; B = len(P); L = max(map(len, P))
            x = torch.tensor([[EOS[1]] * (L - len(p)) + p for p in P], device='cuda')
            att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in P], device='cuda')
            if bi == 0: state['rec'] = []; rec_idx = idx
            out = [[] for _ in range(B)]; wl = [[] for _ in range(B)]; gp = [[] for _ in range(B)]; ig = [[] for _ in range(B)]
            done = torch.zeros(B, dtype=torch.bool, device='cuda')
            EXr.reset(B)
            for t in range(L): cstep(x[:, t])
            state['on'] = True
            o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
            lg = model.lm_head(o.last_hidden_state[:, -1]).float()
            for step in range(a.max_new):
                nxt = lg.argmax(-1); t2 = lg.topk(2, -1).values; gap = (t2[:, 0] - t2[:, 1]).tolist()
                wi = torch.round(EXr.wout['widx']).long().tolist(); igl = state['ig'].tolist()
                nl, dl = nxt.tolist(), done.tolist()
                for i in range(B):
                    if not dl[i]: out[i].append(nl[i]); wl[i].append(wi[i]); gp[i].append(gap[i]); ig[i].append(igl[i])
                fin = (nxt == EOS[0]) | (nxt == EOS[1])
                done = done | fin
                if bool(done.all()): break
                if step == a.max_new - 1: break
                nxt = torch.where(done, torch.full_like(nxt, EOS[1]), nxt)
                att = torch.cat([att, att.new_ones(B, 1)], 1)
                cstep(nxt)
                o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
                lg = model.lm_head(o.last_hidden_state[:, -1]).float()
            state['on'] = False
            if bi == 0: rec = state['rec']; state['rec'] = None; rec_P = P
            for j, i in enumerate(idx):
                R_tok[i], R_w[i], R_gap[i], R_ig[i] = out[j], wl[j], gp[j], ig[j]; ntok += len(out[j])
    torch.cuda.synchronize(); dt = time.time() - t0
    log(stage='reference', n=len(docs), bs=a.bs, max_new=a.max_new, seconds=dt, tokens=ntok, tok_s=ntok / dt,
        rows_written=sum(any(w < EXr.E.shape[0] for w in R_w[i]) for i in R_w))
    ref = dict(tok=R_tok, w=R_w, gap=R_gap, ig=R_ig)
    if a.rl_shape:                                    # reference speed at the RL shape: 64 train prompts x G8, T 1, one batch
        tr = [json.loads(l) for l in open(os.path.join(HERE, '..', 'gen', 'train.jsonl'))][:64]
        _, ftr = CI.format_prompts(tok, [d['prompt'] for d in tr], a.fmt)
        P = [f for f in ftr for _ in range(8)]; B = len(P); L = max(map(len, P))
        state['ig_on'] = False; torch.manual_seed(0); torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            x = torch.tensor([[EOS[1]] * (L - len(p)) + p for p in P], device='cuda')
            att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in P], device='cuda')
            EXr.reset(B); done = torch.zeros(B, dtype=torch.bool, device='cuda'); ln = torch.zeros(B, dtype=torch.long, device='cuda')
            for t in range(L): EXr.step(x[:, t])
            state['on'] = True
            o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
            lg = model.lm_head(o.last_hidden_state[:, -1]).float()
            for step in range(a.max_new):
                nxt = torch.multinomial(torch.softmax(lg, -1), 1)[:, 0]
                ln += (~done).long(); done = done | (nxt == EOS[0]) | (nxt == EOS[1])
                if bool(done.all()) or step == a.max_new - 1: break
                nxt = torch.where(done, torch.full_like(nxt, EOS[1]), nxt)
                att = torch.cat([att, att.new_ones(B, 1)], 1); EXr.step(nxt)
                o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
                lg = model.lm_head(o.last_hidden_state[:, -1]).float()
            state['on'] = False
        torch.cuda.synchronize(); dt = time.time() - t0; nt = int(ln.sum())
        log(stage='reference_rl_shape', rows=B, bs=B, T=1.0, max_new=a.max_new, seconds=dt, tokens=nt, tok_s=nt / dt,
            mean_len=nt / B, steps=step + 1)
    del model, past; torch.cuda.empty_cache()

# --------------------------------------------------------------------------------------------------------- 2. fuzz
imp = GExec(sd); imp.vcheck = imp.fused_vocab
if not a.skip_ref:
    B = len(rec_P); L = max(map(len, rec_P))
    # (a) padded, same layout as the reference
    EXa = Executor(sd, device='cuda'); EXa.reset(B)
    st, mem = imp.new_state(B)
    mm = dict(steps=0, writes=0, state_mm=0, mem_mm=0, widx_mm=0)
    for kind, v in rec:
        if kind == 's':
            EXa.step(v); imp.gstep(st, mem, v); mm['steps'] += 1
        else:
            EXa.write(v); wi, _ = imp.gwrite(st, mem, v); mm['writes'] += 1
            mm['widx_mm'] += int(not torch.equal(torch.round(EXa.wout['widx']).long(), wi))
        mm['state_mm'] += int(not torch.equal(EXa.state, st)); mm['mem_mm'] += int(not torch.equal(EXa.mem, mem))
    log(stage='fuzz_padded', **mm)
    # (b) unpadded, masked prompt steps, rows permuted
    perm = list(range(B))[::-1]
    stb, memb = imp.new_state(B)
    ids_p = [rec_P[j] for j in perm]
    Lb = max(map(len, ids_p))
    idt = torch.tensor([p + [EOS[1]] * (Lb - len(p)) for p in ids_p], device='cuda'); lens = torch.tensor([len(p) for p in ids_p], device='cuda')
    for t in range(Lb): imp.gstep(stb, memb, idt[:, t], active=lens > t)
    pi = torch.tensor(perm, device='cuda')
    first_w = next(i for i, (k, _) in enumerate(rec) if k == 'w')
    live = torch.ones(B, dtype=torch.bool, device='cuda'); nmm = 0; nlive = 0; nw = 0
    stc, memc = imp.new_state(B)                      # padded replay again, for the per-step comparison
    for kind, v in rec[:first_w]: imp.gstep(stc, memc, v)
    for kind, v in rec[first_w:]:
        if kind == 's':
            imp.gstep(stb, memb, v[pi]); imp.gstep(stc, memc, v)
            live = live & ~((v == EOS[0]) | (v == EOS[1]))
        else:
            wb, _ = imp.gwrite(stb, memb, v[pi]); wc, _ = imp.gwrite(stc, memc, v)
            lv = live[pi]
            nmm += int(((wb != wc[pi]) & lv).sum()); nlive += int(lv.sum()); nw += int(((wc[pi] < imp.V) & lv).sum())
    log(stage='fuzz_unpadded_vs_padded', live_writes=nlive, implant_writes=nw, widx_mismatch=nmm)

# --------------------------------------------------------------------------------------------------------- 3. fast greedy
P32 = Q.load_hf_params(a.model)
W = Q.fuse(P32, torch.bfloat16); del P32
S_max = max(len(f) for f in full) - len(pre) + a.max_new + 8
eng = IFEngine(W, tok, pre, imp=imp, B_max=256, S_max=S_max, N_cap=max(1024, len(docs) * 8), max_new_cap=max(a.max_new, 1024))
qids = [f[len(pre):] for f in full]
for rep in range(2):                                  # (first call: graph captures)
    o = eng.generate(qids, G=1, max_new=a.max_new, greedy=True, full_ids=full)
    log(stage=f'fast_greedy_call{rep}', **o['stats'])
if ref:
    nid = 0; divs = []
    for i in range(len(docs)):
        ft = o['tokens'][i]; rt = ref['tok'][i]
        if ft == rt and o['widx'][i] == ref['w'][i]: nid += 1; continue
        k = next((j for j, (x, y) in enumerate(zip(ft, rt)) if x != y), min(len(ft), len(rt)))
        divs.append(dict(i=i, pos=k, len_fast=len(ft), len_ref=len(rt), ref_gap=ref['gap'][i][k] if k < len(rt) else None,
                         imp_gap=ref['ig'][i][k] if k < len(rt) else None,
                         w_ref=ref['w'][i][k] if k < len(rt) else None, w_fast=o['widx'][i][k] if k < len(o['widx'][i]) else None,
                         tok_mismatch=ft != rt))
    log(stage='greedy_identity', n=len(docs), identical=nid, divergent=len(divs),
        near_tie_le_0p25=sum(1 for d in divs if d['ref_gap'] is not None and d['ref_gap'] <= 0.25),
        # a divergence is a bf16 near-tie if the host logit gap or (for implant-written / implant-changed steps) the implant's
        # readout gap at the first differing step is <= 0.25 (2 bf16 ulps at |logit| ~ 16-32)
        near_tie_any_le_0p25=sum(1 for d in divs if d['ref_gap'] is not None and
                                 (d['ref_gap'] <= 0.25 if d['w_ref'] == d['w_fast'] == imp.V else min(d['ref_gap'], d['imp_gap']) <= 0.25)),
        divs=divs)
json.dump(dict(tokens=o['tokens'], widx=o['widx']), open(f'{a.out}/fast_greedy.json', 'w'))
log(stage='fused_vocab_vs_executor', fused=imp.fused_vocab, rows_compared=int(imp.vn), mismatches=int(imp.vmm))
imp.vcheck = False; eng.graphs.clear(); eng.pool = None; torch.cuda.synchronize()   # graphs with the check dropped before timing

# --------------------------------------------------------------------------------------------------------- 4. RL shape timing
if a.rl_shape:
    tr = [json.loads(l) for l in open(os.path.join(HERE, '..', 'gen', 'train.jsonl'))][:64]
    _, ftr = CI.format_prompts(tok, [d['prompt'] for d in tr], a.fmt)
    qtr = [f[len(pre):] for f in ftr]
    for on in (True, False):
        eng.imp_on = on
        for rep in range(2):
            o = eng.generate(qtr, G=8, T=1.0, max_new=a.max_new, greedy=False, full_ids=ftr, seed=rep)
            L = [len(t) for t in o['tokens']]
            log(stage=f'rl_shape_imp{int(on)}_call{rep}', mean_len=sum(L) / len(L), max_len=max(L),
                trunc=sum(1 for t in o['tokens'] if t[-1] not in EOS), **o['stats'])
    eng.imp_on = True
log(stage='DONE')
