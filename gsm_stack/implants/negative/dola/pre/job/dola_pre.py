"""DoLa prescreen (GPU, the B200 node): teacher-force the host (Qwen3-0.6B-Base + ckpt_a2s2, bf16 sdpa) on pre(8-shot) + q_ids + seg_tokens
of every job attempt (dev/dev2 mixed items: all 32 attempts; always-right items: seed-7 attempts), and at every segment
position o >= plen (predicting seg_tokens[o]) compute
  z   = lm_head(norm(h_28))                      (h_28 = final-norm input; the host's logits)
  z_l = lm_head(norm(h_l)), l in LAYS            (h_l = layer-l input residual, as the side-net reads it)
  dyn : per position l* = argmax_l JS(softmax z || softmax z_l)  (DoLa dynamic premature layer)
  z'  = z + alpha * (z - z_l)  ; APC(beta): z'[i] = -inf where p(z)_i < beta * max p(z)
For each config (l or dyn) x alpha x beta: lp'(x_o) (fp16), argmax'(z') != argmax(z), argmax'(z') == x_o.
Fork positions (probe/branch trie forks, dev/dev2): candidate lp' and argmax' ids per config.
Outputs -> ./out/{S}_pos.npz, {S}_fork.npz, {S}_att.json, {S}_summary.json
"""
import os, sys, json, gzip, time, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--sets', default='dev,dev2')
ap.add_argument('--ckpt', default='$WSM_DATA/gsm_sel/ckpt_a2s2')
ap.add_argument('--layers', default='14,18,22,24')
ap.add_argument('--alphas', default='0.1,0.25,0.5,1.0')
ap.add_argument('--betas', default='0,0.1')
ap.add_argument('--tok_budget', type=int, default=65536)
ap.add_argument('--chunk', type=int, default=2048)
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
dev = 'cuda'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, _ = C.prefix_ids(tok); Lp = len(pre)
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
sd = load_file(f'{a.ckpt}/model.safetensors', device='cpu')
msd = m.state_dict()
names = [k for k in msd if k != 'lm_head.weight']
assert not [k for k in names if k not in sd]
with torch.no_grad():
    for k in names: msd[k].copy_(sd[k].float().to(torch.bfloat16).to(dev))
    if 'lm_head.weight' in sd and m.lm_head.weight.data_ptr() != m.model.embed_tokens.weight.data_ptr():
        m.lm_head.weight.copy_(sd['lm_head.weight'].float().to(torch.bfloat16).to(dev))
print('host loaded; tied', m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr(), 'Lp', Lp, flush=True)
del sd
PL = [int(x) for x in a.layers.split(',')]; NLY = m.config.num_hidden_layers; D = m.config.hidden_size
LAY = PL + [NLY]
ALPHAS = [float(x) for x in a.alphas.split(',')]; BETAS = [float(x) for x in a.betas.split(',')]
CH = [str(l) for l in PL] + ['dyn']
CFG = [(c, al, be) for c in CH for al in ALPHAS for be in BETAS]
NC = len(CFG)
print('configs', NC, CFG, flush=True)
cap = {}
def _pre(L):
    def f(mod, args, kwargs):
        cap[L] = args[0] if args else kwargs['hidden_states']
    return f
for L in LAY:
    (m.model.layers[L] if L < NLY else m.model.norm).register_forward_pre_hook(_pre(L), with_kwargs=True)
W = m.lm_head.weight; normf = m.model.norm
os.makedirs(f'{HERE}/out', exist_ok=True)


@torch.no_grad()
def chunk_eval(Hs, idx, xt, fork_rows, fork_cands):
    """Hs: dict L -> (N, D) bf16 flat states; idx: flat rows (n,); xt: target token ids (n,).
    fork_rows: local row indices (into idx) that are fork positions, fork_cands (nf, 8) ids (-1 pad)."""
    z = (normf(Hs[NLY].index_select(0, idx)) @ W.T).float()
    lse0 = torch.logsumexp(z, -1, keepdim=True); lp0 = z - lse0
    a0 = z.argmax(-1)
    lpx0 = lp0.gather(1, xt[:, None])[:, 0]
    top1 = lp0.max(-1).values
    zl = {}
    js = []
    p0 = lp0.exp()
    for l in PL:
        zz = (normf(Hs[l].index_select(0, idx)) @ W.T).float()
        zl[str(l)] = zz
        lpl = torch.log_softmax(zz, -1)
        mm = torch.log(0.5 * (p0 + lpl.exp()) + 1e-30)
        js.append(0.5 * (p0 * (lp0 - mm)).sum(-1) + 0.5 * (lpl.exp() * (lpl - mm)).sum(-1))
        del lpl, mm
    js = torch.stack(js, 1); lstar = js.argmax(1)
    zd = torch.zeros_like(z)
    for i, l in enumerate(PL):
        zd += (lstar == i).float()[:, None] * zl[str(l)]
    zl['dyn'] = zd
    apc = {be: (lp0 < (np.log(be) + top1)[:, None]) for be in BETAS if be > 0}
    LPX = torch.empty(len(idx), NC, device=dev); AMC = torch.empty(len(idx), NC, dtype=torch.bool, device=dev)
    AMX = torch.empty(len(idx), NC, dtype=torch.bool, device=dev)
    nf = len(fork_rows)
    FLP = torch.empty(nf, NC, 8, device=dev); FAM = torch.empty(nf, NC, dtype=torch.long, device=dev)
    if nf:
        fr = torch.tensor(fork_rows, device=dev); fc = torch.tensor(fork_cands, device=dev).clamp(min=0)
    for ci, (c, al, be) in enumerate(CFG):
        zp = z + al * (z - zl[c])
        if be > 0: zp = zp.masked_fill(apc[be], float('-inf'))
        lse = torch.logsumexp(zp, -1)
        am = zp.argmax(-1)
        LPX[:, ci] = zp.gather(1, xt[:, None])[:, 0] - lse
        AMC[:, ci] = am != a0; AMX[:, ci] = am == xt
        if nf:
            FLP[:, ci] = zp[fr].gather(1, fc) - lse[fr][:, None]; FAM[:, ci] = am[fr]
        del zp
    out = dict(lpx0=lpx0, a0=a0, top1=top1, lstar=lstar, js=js, LPX=LPX, AMC=AMC, AMX=AMX)
    fo = dict(FLP=FLP, FAM=FAM)
    if nf:
        fo['lp0c'] = lp0[fr].gather(1, fc); fo['a0'] = a0[fr]
    return out, fo


def run(S):
    t0 = time.time()
    with gzip.open(f'{HERE}/tables/jobs_{S}.jsonl.gz', 'rt') as f: J = [json.loads(l) for l in f]
    if a.limit: J = J[:a.limit]
    qids = {}
    for j in J:
        if j['idx'] not in qids: qids[j['idx']] = C.q_ids(tok, j['q'])
    P = {k: [] for k in ['att', 'o', 'tok', 'wf', 'slp', 'lpx0', 'a0', 'top1', 'lstar', 'js', 'LPX', 'AMC', 'AMX']}
    F = {k: [] for k in ['fid', 'att', 'L', 'cand', 'lp0c', 'a0', 'FLP', 'FAM']}
    order = sorted(range(len(J)), key=lambda i: len(qids[J[i]['idx']]) + len(J[i]['seg_tokens']))
    b0 = 0; ntok = 0
    while b0 < len(order):
        b1 = b0
        while b1 < len(order):
            T = Lp + len(qids[J[order[b1]]['idx']]) + len(J[order[b1]]['seg_tokens'])
            if (b1 - b0 + 1) * T > a.tok_budget and b1 > b0: break
            b1 += 1
        bat = order[b0:b1]; b0 = b1
        seqs = [pre + qids[J[i]['idx']] + J[i]['seg_tokens'] for i in bat]
        T = max(map(len, seqs)); B = len(seqs); ntok += sum(map(len, seqs))
        ids = torch.full((B, T), tok.pad_token_id or 0, dtype=torch.long)
        for b, s in enumerate(seqs): ids[b, :len(s)] = torch.tensor(s)
        with torch.no_grad():
            m.model(input_ids=ids.to(dev), use_cache=False)
            Hs = {L: cap[L].reshape(B * T, D) for L in LAY}
            fl, meta, fk = [], [], {}
            for b, i in enumerate(bat):
                j = J[i]; qb = Lp + len(qids[j['idx']])
                fpos = {L: (fid, cand) for fid, L, cand in j['forks']}
                for o in range(j['plen'], len(j['seg_tokens'])):
                    if o in fpos: fk[len(fl)] = (i, o) + fpos[o]
                    fl.append(b * T + qb + o - 1); meta.append((i, o))
            for c0 in range(0, len(fl), a.chunk):
                c1 = min(len(fl), c0 + a.chunk)
                idx = torch.tensor(fl[c0:c1], device=dev)
                xt = torch.tensor([J[i]['seg_tokens'][o] for i, o in meta[c0:c1]], device=dev)
                frows = [r - c0 for r in range(c0, c1) if r in fk]
                fc = [(fk[r + c0][3] + [-1] * 8)[:8] for r in frows]
                out, fo = chunk_eval(Hs, idx, xt, frows, fc)
                for i, o in meta[c0:c1]:
                    j = J[i]; P['att'].append(i); P['o'].append(o); P['tok'].append(j['seg_tokens'][o]); P['wf'].append(j['seg_wf'][o])
                    P['slp'].append(j['seg_logp'][o])
                for k in ['lpx0', 'a0', 'top1', 'lstar', 'js', 'LPX', 'AMC', 'AMX']: P[k].append(out[k].cpu().numpy())
                for q, r in enumerate(frows):
                    i, o, fid, cand = fk[r + c0]
                    F['fid'].append(fid); F['att'].append(i); F['L'].append(o); F['cand'].append(fc[q])
                if frows:
                    for k in ['lp0c', 'a0', 'FLP', 'FAM']: F[k].append(fo[k].cpu().numpy())
        del Hs
        cap.clear()
    cat = lambda k: np.concatenate(P[k], 0)
    LPX = np.clip(cat('LPX'), -99, 0).astype(np.float16)
    np.savez_compressed(f'{HERE}/out/{S}_pos.npz', att=np.array(P['att'], np.int32), o=np.array(P['o'], np.int16),
                        tok=np.array(P['tok'], np.int32), wf=np.array(P['wf'], np.int16), slp=np.array(P['slp'], np.float32),
                        lpx0=cat('lpx0').astype(np.float32), a0=cat('a0').astype(np.int32), top1=cat('top1').astype(np.float16),
                        lstar=cat('lstar').astype(np.int8), js=cat('js').astype(np.float16), LPX=LPX,
                        AMC=np.packbits(cat('AMC'), axis=1), AMX=np.packbits(cat('AMX'), axis=1))
    if F['fid']:
        np.savez_compressed(f'{HERE}/out/{S}_fork.npz', fid=np.array(F['fid']), att=np.array(F['att'], np.int32), L=np.array(F['L'], np.int32),
                            cand=np.array(F['cand'], np.int32), lp0c=np.concatenate(F['lp0c']).astype(np.float32),
                            a0=np.concatenate(F['a0']).astype(np.int32), FLP=np.clip(np.concatenate(F['FLP']), -99, 0).astype(np.float32),
                            FAM=np.concatenate(F['FAM']).astype(np.int32))
    json.dump(dict(gid=[j['gid'] for j in J], idx=[j['idx'] for j in J], ok=[j['ok'] for j in J], cls=[j['cls'] for j in J],
                   plen=[j['plen'] for j in J], cfg=CFG), open(f'{HERE}/out/{S}_att.json', 'w'))
    wf = np.array(P['wf']); d = np.abs(cat('lpx0') - np.array(P['slp']))[(wf & 3) == 0]
    s = dict(set=S, n_att=len(J), n_pos=len(P['att']), n_fork=len(F['fid']), tokens=ntok, secs=round(time.time() - t0, 1),
             lp_mae_wf0=float(d.mean()), lp_p90=float(np.quantile(d, .9)), cfg=CFG)
    print(json.dumps(s), flush=True)
    json.dump(s, open(f'{HERE}/out/{S}_summary.json', 'w'))


for S in a.sets.split(','): run(S)
print('DOLA_PRE DONE', flush=True)
