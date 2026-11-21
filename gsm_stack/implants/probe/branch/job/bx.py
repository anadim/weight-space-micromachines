"""Branch-probe extraction (GPU, the B200 node). Teacher-forces the host (Qwen3-0.6B-Base + ckpt_a2s2, bf16, sdpa) on the visible
context pre (8-shot) + C.q_ids(q) + seg_tokens of each job attempt (positions 0..len-1, as probe/extract.py) and keeps
  forks:    at every listed fork offset L: h at the position predicting seg_tokens[L] (abs qb+L-1), layers LAY,
            top-5 (id, logp) and logp of the fork's candidate tokens (fp32 log_softmax of the host's logits)
  controls: (ctrl attempts) up to NCTRL random positions o >= plen with wf[o]==0 and p2/p1 >= RATIO (host top-2), h + top-5
  pos:      (ctrl attempts) every position o >= plen: p1, p2, id1, id2, logp of the sampled token, wf
LAY = layers 14,18,22,27 (HF hidden_states[L] = layer-L input) + 28 (final-norm input). Unembedding rows of all ids seen.
Outputs -> ./out/ (small) and h arrays -> $WSM_DATA/implants/branch/ + ./out/.
"""
import os, sys, json, gzip, time, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--sets', default='trainC,dev,dev2')
ap.add_argument('--ckpt', default='$WSM_DATA/gsm_sel/ckpt_a2s2')
ap.add_argument('--big', default='$WSM_DATA/implants/branch')
ap.add_argument('--layers', default='14,18,22,27,28')
ap.add_argument('--tok_budget', type=int, default=65536)
ap.add_argument('--nctrl', type=int, default=2)
ap.add_argument('--ratio', type=float, default=0.3)
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
tied = m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
print('host loaded; tied', tied, 'lm_head in ckpt', 'lm_head.weight' in sd, 'Lp', Lp, flush=True)
del sd
LAY = [int(x) for x in a.layers.split(',')]; NLY = m.config.num_hidden_layers; D = m.config.hidden_size
cap = {}
def _pre(L):
    def f(mod, args, kwargs):
        cap[L] = args[0] if args else kwargs['hidden_states']
    return f
for L in LAY:
    (m.model.layers[L] if L < NLY else m.model.norm).register_forward_pre_hook(_pre(L), with_kwargs=True)
W = m.lm_head.weight
seen = set()
os.makedirs(a.big, exist_ok=True); os.makedirs(f'{HERE}/out', exist_ok=True)


def run(S):
    t0 = time.time()
    with gzip.open(f'{HERE}/tables/jobs_{S}.jsonl.gz', 'rt') as f: J = [json.loads(l) for l in f]
    if a.limit: J = J[:a.limit]
    qids = {}
    for j in J:
        if j['idx'] not in qids: qids[j['idx']] = C.q_ids(tok, j['q'])
    rng = np.random.default_rng(0)
    FH, FM, CH, CM = [], dict(fid=[], att=[], L=[], top_id=[], top_lp=[], cand=[], cand_lp=[], cand_slp=[]), [], \
        dict(att=[], o=[], tok=[], top_id=[], top_lp=[], lp_tok=[], slp=[])
    PS = dict(att=[], o=[], p1=[], p2=[], id1=[], id2=[], lp_tok=[], slp=[], wf=[])
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
            H = {L: cap[L].reshape(B * T, D) for L in LAY}
            normf = m.model.norm
            # rows: (b, o) -> flat index of the position predicting seg_tokens[o]
            fl_fork, fk = [], []
            fl_seg, sk = [], []
            for b, i in enumerate(bat):
                j = J[i]; qb = Lp + len(qids[j['idx']])
                for fid, L, cand in j['forks']:
                    fl_fork.append(b * T + qb + L - 1); fk.append((i, fid, L, cand))
                if j['ctrl']:
                    for o in range(j['plen'], len(j['seg_tokens'])):
                        fl_seg.append(b * T + qb + o - 1); sk.append((i, o))
            def logits_at(fl):
                out = []
                for c0 in range(0, len(fl), 4096):
                    x = H[NLY].index_select(0, torch.tensor(fl[c0:c0 + 4096], device=dev))
                    out.append(torch.log_softmax((normf(x) @ W.T).float(), -1))
                return out
            if fl_fork:
                lps = torch.cat(logits_at(fl_fork), 0)
                tl, ti = lps.topk(5, -1)
                idx_t = torch.tensor(fl_fork, device=dev)
                FH.append(torch.stack([H[L].index_select(0, idx_t) for L in LAY], 1).to(torch.float16).cpu().numpy())
                for r, (i, fid, L, cand) in enumerate(fk):
                    c8 = (cand + [-1] * 8)[:8]
                    FM['fid'].append(fid); FM['att'].append(i); FM['L'].append(L)
                    FM['top_id'].append(ti[r].tolist()); FM['top_lp'].append(tl[r].tolist())
                    FM['cand'].append(c8); FM['cand_lp'].append([float(lps[r, c]) if c >= 0 else -99. for c in c8])
                    FM['cand_slp'].append(J[i]['seg_logp'][L])
                    seen.update(ti[r].tolist()); seen.update(cand)
                del lps
            if fl_seg:
                chunks = logits_at(fl_seg); r0 = 0; sel = []
                for lp in chunks:
                    n = lp.shape[0]; tl, ti = lp.topk(5, -1)
                    tt = torch.tensor([J[sk[r0 + q][0]]['seg_tokens'][sk[r0 + q][1]] for q in range(n)], device=dev)
                    ltok = lp.gather(1, tt[:, None])[:, 0]
                    tl, ti, ltok = tl.cpu().numpy(), ti.cpu().numpy(), ltok.cpu().numpy()
                    for q in range(n):
                        i, o = sk[r0 + q]; j = J[i]
                        PS['att'].append(i); PS['o'].append(o); PS['p1'].append(tl[q, 0]); PS['p2'].append(tl[q, 1])
                        PS['id1'].append(ti[q, 0]); PS['id2'].append(ti[q, 1]); PS['lp_tok'].append(ltok[q])
                        PS['slp'].append(j['seg_logp'][o]); PS['wf'].append(j['seg_wf'][o])
                        if j['seg_wf'][o] == 0 and tl[q, 1] - tl[q, 0] >= np.log(a.ratio):
                            sel.append((r0 + q, ti[q], tl[q], ltok[q]))
                    r0 += n
                del chunks
                # choose up to nctrl per attempt
                bya = {}
                for s_ in sel: bya.setdefault(sk[s_[0]][0], []).append(s_)
                pick = []
                for i, L_ in bya.items():
                    for k in rng.permutation(len(L_))[:a.nctrl]: pick.append(L_[k])
                if pick:
                    idx_t = torch.tensor([fl_seg[p[0]] for p in pick], device=dev)
                    CH.append(torch.stack([H[L].index_select(0, idx_t) for L in LAY], 1).to(torch.float16).cpu().numpy())
                    for p in pick:
                        i, o = sk[p[0]]
                        CM['att'].append(i); CM['o'].append(o); CM['tok'].append(J[i]['seg_tokens'][o])
                        CM['top_id'].append(p[1].tolist()); CM['top_lp'].append(p[2].tolist()); CM['lp_tok'].append(float(p[3]))
                        CM['slp'].append(J[i]['seg_logp'][o]); seen.update(p[1].tolist())
        del H
        cap.clear()
    gids = np.array([j['gid'] for j in J])
    if FH:
        Fh = np.concatenate(FH, 0); np.save(f'{a.big}/{S}_fork_h.npy', Fh); np.save(f'{HERE}/out/{S}_fork_h.npy', Fh)
        np.savez(f'{HERE}/out/{S}_fork_meta.npz', fid=np.array(FM['fid']), att=np.array(FM['att'], np.int32), L=np.array(FM['L'], np.int32),
                 top_id=np.array(FM['top_id'], np.int32), top_lp=np.array(FM['top_lp'], np.float32), cand=np.array(FM['cand'], np.int32),
                 cand_lp=np.array(FM['cand_lp'], np.float32), cand_slp=np.array(FM['cand_slp'], np.float32), gids=gids)
    if CH:
        Ch = np.concatenate(CH, 0); np.save(f'{a.big}/{S}_ctrl_h.npy', Ch); np.save(f'{HERE}/out/{S}_ctrl_h.npy', Ch)
        np.savez(f'{HERE}/out/{S}_ctrl_meta.npz', att=np.array(CM['att'], np.int32), o=np.array(CM['o'], np.int32),
                 tok=np.array(CM['tok'], np.int32), top_id=np.array(CM['top_id'], np.int32), top_lp=np.array(CM['top_lp'], np.float32),
                 lp_tok=np.array(CM['lp_tok'], np.float32), slp=np.array(CM['slp'], np.float32), gids=gids,
                 ok=np.array([j['ok'] for j in J], np.int8), b=np.array([j['b'] for j in J], np.int32))
    if PS['att']:
        np.savez_compressed(f'{HERE}/out/{S}_pos.npz', att=np.array(PS['att'], np.int32), o=np.array(PS['o'], np.int16),
                            lp1=np.array(PS['p1'], np.float16), lp2=np.array(PS['p2'], np.float16), id1=np.array(PS['id1'], np.int32),
                            id2=np.array(PS['id2'], np.int32), lp_tok=np.array(PS['lp_tok'], np.float16), slp=np.array(PS['slp'], np.float16),
                            wf=np.array(PS['wf'], np.int16))
    # sanity: tf logp of fork children vs stored sampled logp (first child = rep's own token)
    s = {}
    if FM['fid']:
        rep_tok = np.array([J[i]['seg_tokens'][L] for i, L in zip(FM['att'], FM['L'])])
        c = np.array(FM['cand']); cl = np.array(FM['cand_lp'])
        own = cl[np.arange(len(c)), (c == rep_tok[:, None]).argmax(1)]
        d = np.abs(own - np.array(FM['cand_slp']))
        s['fork_own_lp_mae'] = float(d.mean()); s['fork_own_lp_p90'] = float(np.quantile(d, .9))
    if PS['att']:
        d = np.abs(np.array(PS['lp_tok'], np.float32) - np.array(PS['slp'], np.float32))[np.array(PS['wf']) == 0]
        s['pos_lp_mae_wf0'] = float(d.mean()); s['pos_lp_p90'] = float(np.quantile(d, .9))
    s.update(set=S, n_att=len(J), n_fork=len(FM['fid']), n_ctrl=len(CM['att']), n_pos=len(PS['att']), tokens=ntok, secs=round(time.time() - t0, 1))
    print(json.dumps(s), flush=True)
    json.dump(s, open(f'{HERE}/out/{S}_summary.json', 'w'))


for S in a.sets.split(','): run(S)
ids_ = np.array(sorted(seen), np.int64)
np.savez(f'{HERE}/out/unemb.npz', ids=ids_, U=W[torch.tensor(ids_, device=dev)].float().cpu().numpy().astype(np.float16),
         norm_w=m.model.norm.weight.float().cpu().numpy(), eps=np.float32(m.model.norm.variance_epsilon), layers=np.array(LAY))
print('BX DONE', len(ids_), flush=True)
