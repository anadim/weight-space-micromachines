"""Side-net features: per-attempt hidden states of the rung-3 (avf) stack attempts, teacher-forced in their in-stack-equivalent
context  pre (8-shot prefix) + C.q_ids(q) + seg_tokens  (positions 0..len-1; forget mask hides earlier attempts, WPR makes the
question adjacent) through the frozen host (Qwen3-0.6B-Base + gsm_sel/ckpt_a2s2, bf16, sdpa).

Per attempt, layers 10..27 = layer INPUT (HF hidden_states[L]) and 28 = final-norm input, fp16:
  H[l, a] = state at the token completing '####' (hash_off; the stack's L18 verifier read). No '####': last token, has_hash=0.
  M[l, a] = mean of the states at model-written line ends: tokens of seg_tokens containing '\\n' with phrase_len <= off < hash_off
            (no '####': < len). 0 if none (nnl=0).
Output (node): OUT_ROOT/{set}/H.npy, M.npy (nL, n, 1024) fp16 layer-major; meta.npz (idx, seed, k, ok, lg, bad, nrep, has_hash,
nnl, seg_len, pick/voted).  Validation (L18 hash state . verifier probe vs the row register lg) -> ./out/extract_{set}.json
"""
import os, sys, json, gzip, time, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--sets', default='dev,dev2,trainC')
ap.add_argument('--data', default=f'{HERE}/tables')
ap.add_argument('--out_root', default='$WSM_DATA/implants/sidenet')
ap.add_argument('--ckpt', default='$WSM_DATA/gsm_sel/ckpt_a2s2')
ap.add_argument('--probe', default=f'{HERE}/verifier_probe_L18.pt')
ap.add_argument('--layers', default='10-28')
ap.add_argument('--tok_budget', type=int, default=131072)
ap.add_argument('--limit_items', type=int, default=0)
ap.add_argument('--tag', default='')
ap.add_argument('--extra', type=int, default=0, help='1: only A (last seg token) and E (last line end before ####) -> A.npy, E.npy')
a = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
dev = 'cuda'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, shot_q = C.prefix_ids(tok); Lp = len(pre)
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
sd = load_file(f'{a.ckpt}/model.safetensors', device='cpu')
msd = m.state_dict()
names = [k for k in msd if k != 'lm_head.weight']
assert all(k in sd for k in names)
with torch.no_grad():
    for k in names: msd[k].copy_(sd[k].float().to(torch.bfloat16).to(dev))
del sd
print('host loaded, Lp', Lp, flush=True)
lo, hi = map(int, a.layers.split('-')); LAY = list(range(lo, hi + 1)); nL = len(LAY); D = m.config.hidden_size
NLY = m.config.num_hidden_layers
PR = torch.load(a.probe); pw = PR['w'].float().to(dev); pb = float(PR['b']); pL = int(PR.get('layer', 18))
NLTOK = np.array(['\n' in tok.decode([i]) for i in range(len(tok))] + [False] * (m.config.vocab_size - len(tok)))

cap = {}; cur = {}
def _pre(L):
    def f(mod, args, kwargs):
        h = args[0] if args else kwargs['hidden_states']
        cap[L] = h.reshape(-1, D).index_select(0, cur['flat'])
    return f
for L in LAY:
    (m.model.layers[L] if L < NLY else m.model.norm).register_forward_pre_hook(_pre(L), with_kwargs=True)


def read_table(p):
    with gzip.open(p, 'rt') as f: return [json.loads(l) for l in f]


def run_set(S):
    t0 = time.time()
    rows = read_table(f'{a.data}/attempts_{S}.jsonl.gz')
    if a.limit_items:
        ks = set(sorted({r['idx'] for r in rows})[:a.limit_items]); rows = [r for r in rows if r['idx'] in ks]
    n = len(rows); od = f'{a.out_root}/{S}{a.tag}'; os.makedirs(od, exist_ok=True)
    H = np.lib.format.open_memmap(f'{od}/{"A" if a.extra else "H"}.npy', 'w+', np.float16, (nL, n, D))
    M = np.lib.format.open_memmap(f'{od}/{"E" if a.extra else "M"}.npy', 'w+', np.float16, (nL, n, D))
    qids = {}
    for r in rows:
        if r['idx'] not in qids: qids[r['idx']] = C.q_ids(tok, r['q'])
    has_hash = np.zeros(n, np.int8); nnl = np.zeros(n, np.int16); lg_ours = np.full(n, np.nan, np.float32)
    POS = []
    for r in rows:
        st = r['seg_tokens']; ho = r.get('hash_off')
        hp = int(ho) if ho is not None else len(st) - 1
        end = int(ho) if ho is not None else len(st)
        pl = int(r.get('phrase_len') or 0)
        nl = [o for o in range(pl, end) if NLTOK[st[o]]]
        if a.extra: hp, nl = len(st) - 1, nl[-1:]
        POS.append((hp, nl))
    order = sorted(range(n), key=lambda i: Lp + len(qids[rows[i]['idx']]) + len(rows[i]['seg_tokens']))
    b0 = 0; ntok = 0
    while b0 < n:
        b1 = b0
        while b1 < n:
            T = Lp + len(qids[rows[order[b1]]['idx']]) + len(rows[order[b1]]['seg_tokens'])
            if (b1 - b0 + 1) * T > a.tok_budget and b1 > b0: break
            b1 += 1
        bat = order[b0:b1]; b0 = b1
        seqs = [pre + qids[rows[i]['idx']] + rows[i]['seg_tokens'] for i in bat]
        T = max(map(len, seqs)); B = len(seqs)
        ids = torch.full((B, T), tok.pad_token_id or 0, dtype=torch.long)
        flat, seg = [], []                                  # seg: attempt-in-batch index for each nl position
        for b, (i, s) in enumerate(zip(bat, seqs)):
            ids[b, :len(s)] = torch.tensor(s)
            qb = Lp + len(qids[rows[i]['idx']])
            flat.append(b * T + qb + POS[i][0])
        for b, i in enumerate(bat):
            qb = Lp + len(qids[rows[i]['idx']])
            for o in POS[i][1]: flat.append(b * T + qb + o); seg.append(b)
        ntok += sum(map(len, seqs))
        cur['flat'] = torch.tensor(flat, device=dev)
        with torch.no_grad():
            m.model(input_ids=ids.to(dev), use_cache=False)
            Cc = torch.stack([cap[L] for L in LAY], 0)       # (nL, B + nnl, D) bf16
            h = Cc[:, :B]
            lgb = (cap[pL][:B].float() @ pw + pb).cpu().numpy()
            sg = torch.tensor(seg, device=dev, dtype=torch.long)
            sm = torch.zeros(nL, B, D, device=dev, dtype=torch.float32)
            if len(seg): sm.index_add_(1, sg, Cc[:, B:].float())
            cnt = torch.bincount(sg, minlength=B).float() if len(seg) else torch.zeros(B, device=dev)
            mn = sm / cnt.clamp(min=1)[None, :, None]
        bi = np.array(bat)
        srt = np.argsort(bi); bs = bi[srt]
        H[:, bs] = h.to(torch.float16).cpu().numpy()[:, srt]
        M[:, bs] = mn.to(torch.float16).cpu().numpy()[:, srt]
        nnl[bi] = cnt.cpu().numpy().astype(np.int16)
        lg_ours[bi] = lgb
        for b, i in enumerate(bat): has_hash[i] = rows[i].get('hash_off') is not None
    H.flush(); M.flush(); del H, M
    meta = dict(idx=np.array([r['idx'] for r in rows], np.int32), seed=np.array([r['seed'] for r in rows], np.int32),
                k=np.array([r['k'] for r in rows], np.int16), ok=np.array([int(r['ok']) for r in rows], np.int8),
                lg=np.array([np.nan if r.get('lg') is None else float(r['lg']) for r in rows], np.float32),
                bad=np.array([int(bool(r.get('bad'))) for r in rows], np.int8), nrep=np.array([int(r.get('nrep') or 0) for r in rows], np.int16),
                has_hash=has_hash, nnl=nnl, seg_len=np.array([len(r['seg_tokens']) for r in rows], np.int16),
                voted=np.array([int(r.get('voted') or 0) for r in rows], np.int8), lg_ours=lg_ours, layers=np.array(LAY))
    if a.extra:
        print('extra done', S, nnl.sum(), flush=True); return
    np.savez(f'{od}/meta.npz', **meta)
    t = meta['lg']; ok_ = has_hash.astype(bool) & np.isfinite(t) & (t > -29.5)
    d = np.abs(t[ok_] - lg_ours[ok_])
    val = dict(set=S, n=n, n_cmp=int(ok_.sum()), corr=float(np.corrcoef(t[ok_], lg_ours[ok_])[0, 1]), mae=float(d.mean()),
               p99=float(np.quantile(d, .99)), max=float(d.max()), frac_gt_0p5=float((d > .5).mean()),
               nrep0_mae=float(np.abs(t - lg_ours)[ok_ & (meta['nrep'] == 0)].mean()),
               nrep_mae=float(np.abs(t - lg_ours)[ok_ & (meta['nrep'] > 0)].mean()) if (ok_ & (meta['nrep'] > 0)).any() else None,
               no_hash=int((has_hash == 0).sum()), nnl0=int((nnl == 0).sum()), tokens=ntok, secs=round(time.time() - t0, 1),
               size_gb=round(2 * nL * n * D * 2 / 1e9, 2))
    os.makedirs(f'{HERE}/out', exist_ok=True)
    json.dump(val, open(f'{HERE}/out/extract_{S}{a.tag}.json', 'w'), indent=1)
    print('VALIDATION', json.dumps(val), flush=True)


for S in a.sets.split(','):
    run_set(S)
print('EXTRACT DONE', flush=True)
