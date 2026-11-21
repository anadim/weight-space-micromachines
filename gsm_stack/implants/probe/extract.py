"""Hidden-state extraction for the RL+calc host (lanes 2/3). Runs on the B200 node (see run.sh, README.md).

For every attempt of attempts_{set}.jsonl.gz we teacher-force its visible context
    ids = pre (8-shot prefix, C.prefix_ids) + C.q_ids(q) + seg_tokens,   positions 0..len-1
(= what the S3-serial stack shows the host: forget mask hides earlier segments, the position register restarts the
segment right after the question) through the host (HF Qwen3-0.6B-Base + gsm_sel/ckpt_a2s2 weights, bf16) and keep,
at layers 10..27 the layer INPUT (= HF hidden_states[L]) and as "layer 28" the final-norm INPUT (layer-27 output,
pre-norm; HF hidden_states[28] is post-norm), fp16, at
    kind 1 'nl'   every token of seg_tokens whose text contains '\\n' (line / step ends; j = newline ordinal)
    kind 2 'hash' the token completing '####' (hash_off) -- where the stack reads the L18 verifier logit lg
    kind 3 'last' the last token of seg_tokens (end of the '#### n' line)
and once per item kind 0 'qend' = the last question token ('Answer:' ':'), identical for every attempt of the item.
The calc implant writes only after layer 27 (before the final norm) and its acts are token replacements, so the
teacher-forced stream reproduces the host's residual stream up to bf16 noise; validated via lg (see --check).

Output: OUT_ROOT/{set}{tag}/ : info.json, shard_XXX.npy (nL, n, 1024) fp16, pos_meta.npz, qend.npy (nL, n_items, D),
qend_meta.npz, attempts.npz (attempt-level fields), attempts_light.jsonl.gz (table minus seg_tokens).
Small validation summary -> ./out/extract_{set}{tag}.json
"""
import os, sys, json, gzip, time, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--sets', default='dev,dev2')
ap.add_argument('--data', default=f'{HERE}/tables')
ap.add_argument('--out_root', default='$WSM_DATA/implants/hs')
ap.add_argument('--ckpt', default='$WSM_DATA/gsm_sel/ckpt_a2s2')
ap.add_argument('--probe', default=f'{HERE}/verifier_probe_L18.pt')
ap.add_argument('--layers', default='10-28', help="'a-b' inclusive; 28 = final-norm input")
ap.add_argument('--tok_budget', type=int, default=131072, help='padded tokens per forward batch')
ap.add_argument('--shard', type=int, default=4096, help='attempts per shard file')
ap.add_argument('--limit_items', type=int, default=0, help='smoke: only the first N item idx of each set')
ap.add_argument('--tag', default='')
ap.add_argument('--dtype', default='bf16', help='bf16 (as the engine) | fp32 (bf16 weights upcast, fp32 compute)')
ap.add_argument('--inspect', action='store_true', help='load model/ckpt, print key names, exit (CPU ok)')
ap.add_argument('--check_hs', type=int, default=1, help='compare hook captures with output_hidden_states on batch 0')
a = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, shot_q = C.prefix_ids(tok); Lp = len(pre)
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
sd = load_file(f'{a.ckpt}/model.safetensors', device='cpu')
msd = m.state_dict()
names = [k for k in msd if k != 'lm_head.weight']
missing = [k for k in names if k not in sd]; extra = [k for k in sd if k not in msd]
print('ckpt keys', len(sd), 'e.g.', list(sd)[:3], 'dtypes', sorted({str(v.dtype) for v in sd.values()}),
      '| missing', missing[:5], len(missing), '| extra', extra[:5], len(extra), flush=True)
assert not missing, missing
with torch.no_grad():
    drift = {}
    for k in names:
        new = sd[k].float().to(torch.bfloat16)        # engine: fp32 P32 copy -> bf16 fuse
        if k.endswith(('layers.0.mlp.down_proj.weight', 'layers.18.self_attn.q_proj.weight', 'embed_tokens.weight')):
            drift[k] = float((new.float() - msd[k].float().cpu()).norm() / msd[k].float().norm())
        msd[k].copy_(new.to(dev))
    tied = m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
print('loaded host from', a.ckpt, '| rel drift vs base', drift, '| tied lm_head', tied, '| Lp', Lp, flush=True)
del sd
if a.dtype == 'fp32':
    m = m.float(); torch.backends.cuda.matmul.allow_tf32 = False
if a.inspect:
    print('config', m.config.num_hidden_layers, m.config.hidden_size, m.config._attn_implementation)
    sys.exit(0)

lo, hi = map(int, a.layers.split('-')); LAY = list(range(lo, hi + 1)); nL = len(LAY); D = m.config.hidden_size
NLY = m.config.num_hidden_layers                       # 28
assert hi <= NLY
PR = torch.load(a.probe); pw = PR['w'].float().to(dev); pb = float(PR['b']); pL = int(PR.get('layer', 18))
assert pL in LAY
NLTOK = torch.tensor(['\n' in tok.decode([i]) for i in range(len(tok))] + [False] * (m.config.vocab_size - len(tok)))

cap = {}; cur = {}
def _pre(L):
    def f(mod, args, kwargs):
        if cur.get('off'): return
        h = args[0] if args else kwargs['hidden_states']
        cap[L] = h.reshape(-1, D).index_select(0, cur['flat'])
    return f
hooks = [m.model.layers[L].register_forward_pre_hook(_pre(L), with_kwargs=True) for L in LAY if L < NLY]
if NLY in LAY: hooks.append(m.model.norm.register_forward_pre_hook(_pre(NLY), with_kwargs=True))


def read_table(p):
    with gzip.open(p, 'rt') as f: return [json.loads(l) for l in f]


def positions(r):
    """[(kind, j, off)] of one attempt, offsets into seg_tokens."""
    st = r['seg_tokens']; out = []
    nl = NLTOK[torch.tensor(st, dtype=torch.long)].nonzero()[:, 0].tolist() if st else []
    out += [(1, j, o) for j, o in enumerate(nl)]
    if r.get('hash_off') is not None: out.append((2, 0, int(r['hash_off'])))
    if st: out.append((3, 0, len(st) - 1))
    return out


def run_set(S):
    t_set = time.time()
    rows = read_table(f'{a.data}/attempts_{S}.jsonl.gz')
    if a.limit_items:
        keep = sorted({r['idx'] for r in rows})[:a.limit_items]; ks = set(keep)
        rows = [r for r in rows if r['idx'] in ks]
    od = f'{a.out_root}/{S}{a.tag}'; os.makedirs(od, exist_ok=True)
    qids = {}
    for r in rows:
        if r['idx'] not in qids: qids[r['idx']] = C.q_ids(tok, r['q'])
    items = sorted(qids); item_row = {i: n for n, i in enumerate(items)}
    qend = np.zeros((nL, len(items), D), np.float16); qend_got = np.zeros(len(items), bool); qend_dev = []
    P = [positions(r) for r in rows]
    pm = dict(att=[], kind=[], j=[], off=[], pos=[], tok=[], shard=[], row=[])
    lg_pairs = []                                      # (att, lg_table, lg_bf16, lg_fp16)
    shards = []; n_tok = 0; n_pad = 0; t_fwd = 0.0; checked = not a.check_hs; hs_check = None
    for s0 in range(0, len(rows), a.shard):
        A = list(range(s0, min(len(rows), s0 + a.shard)))
        base = {}; n = 0
        for i in A: base[i] = n; n += len(P[i])
        buf = np.zeros((nL, n, D), np.float16)
        order = sorted(A, key=lambda i: Lp + len(qids[rows[i]['idx']]) + len(rows[i]['seg_tokens']))
        b0 = 0
        while b0 < len(order):
            b1 = b0
            while b1 < len(order):
                T = Lp + len(qids[rows[order[b1]]['idx']]) + len(rows[order[b1]]['seg_tokens'])
                if (b1 - b0 + 1) * T > a.tok_budget and b1 > b0: break
                b1 += 1
            bat = order[b0:b1]; b0 = b1
            seqs = [pre + qids[rows[i]['idx']] + rows[i]['seg_tokens'] for i in bat]
            T = max(map(len, seqs)); B = len(seqs)
            ids = torch.full((B, T), tok.pad_token_id or 0, dtype=torch.long)
            flat, where = [], []                       # where: (i, slot_in_attempt or -1 for qend)
            for b, (i, s) in enumerate(zip(bat, seqs)):
                ids[b, :len(s)] = torch.tensor(s)
                qb = Lp + len(qids[rows[i]['idx']])
                flat.append(b * T + qb - 1); where.append((i, -1))
                for p_, (kd, j, o) in enumerate(P[i]):
                    flat.append(b * T + qb + o); where.append((i, p_))
            n_tok += sum(map(len, seqs)); n_pad += B * T
            cur['flat'] = torch.tensor(flat, device=dev)
            sync(); t0 = time.time()
            with torch.no_grad():
                if not checked:
                    hs = m.model(input_ids=ids.to(dev), use_cache=False, output_hidden_states=True).hidden_states
                    fl = cur['flat']
                    d_in = max(float((hs[L].reshape(-1, D).index_select(0, fl) - cap[L]).abs().max()) for L in LAY if L < NLY)
                    cur['off'] = True
                    d_norm = float((hs[NLY].reshape(-1, D).index_select(0, fl) - m.model.norm(cap[NLY])).abs().max()) if NLY in LAY else None
                    cur['off'] = False
                    hs_check = dict(max_abs_diff_layer_inputs=d_in, max_abs_diff_norm=d_norm); checked = True; del hs
                    print('hook check', hs_check, flush=True)
                else:
                    m.model(input_ids=ids.to(dev), use_cache=False)
            sync(); t_fwd += time.time() - t0
            Cc = torch.stack([cap[L] for L in LAY], 0)              # (nL, npos, D) bf16
            # verifier logit from the bf16 state (as the engine reads x.float() @ w + b)
            Lc = cap[pL].float() @ pw + pb
            C16 = Cc.to(torch.float16).cpu().numpy()
            Lc16 = (Cc[LAY.index(pL)].to(torch.float16).float() @ pw + pb).cpu().numpy()
            Lc = Lc.cpu().numpy()
            for col, (i, p_) in enumerate(where):
                if p_ < 0:
                    it = item_row[rows[i]['idx']]
                    if not qend_got[it]: qend[:, it] = C16[:, col]; qend_got[it] = True
                    else: qend_dev.append(float(np.abs(qend[:, it].astype(np.float32) - C16[:, col].astype(np.float32)).max()))
                    continue
                buf[:, base[i] + p_] = C16[:, col]
                if P[i][p_][0] == 2: lg_pairs.append((i, rows[i].get('lg'), float(Lc[col]), float(Lc16[col])))
        k = len(shards); fn = f'shard_{k:03d}.npy'
        np.save(f'{od}/{fn}', buf); shards.append(dict(file=fn, n=n, att0=A[0], att1=A[-1] + 1))
        for i in A:
            qb = Lp + len(qids[rows[i]['idx']])
            for p_, (kd, j, o) in enumerate(P[i]):
                pm['att'].append(i); pm['kind'].append(kd); pm['j'].append(j); pm['off'].append(o); pm['pos'].append(qb + o)
                pm['tok'].append(rows[i]['seg_tokens'][o]); pm['shard'].append(k); pm['row'].append(base[i] + p_)
        print(f'{S} shard {k}: attempts {A[0]}..{A[-1]} positions {n} | fwd {t_fwd:.0f}s tok {n_tok} pad-eff {n_tok / max(1, n_pad):.2f}', flush=True)
        if not np.isfinite(buf).all(): print('WARNING non-finite fp16 values in', fn, flush=True)
    dt = {'att': np.int32, 'kind': np.int8, 'j': np.int16, 'off': np.int16, 'pos': np.int32, 'tok': np.int32, 'shard': np.int16, 'row': np.int32}
    np.savez(f'{od}/pos_meta.npz', **{k_: np.array(v, dtype=dt[k_]) for k_, v in pm.items()})
    np.save(f'{od}/qend.npy', qend)
    np.savez(f'{od}/qend_meta.npz', idx=np.array(items, np.int32), qlen=np.array([len(qids[i]) for i in items], np.int32),
             pos=np.array([Lp + len(qids[i]) - 1 for i in items], np.int32))
    hof = lambda r: -1 if r.get('hash_off') is None else int(r['hash_off'])
    np.savez(f'{od}/attempts.npz', idx=np.array([r['idx'] for r in rows], np.int32), seed=np.array([r['seed'] for r in rows], np.int32),
             k=np.array([r['k'] for r in rows], np.int16), ok=np.array([int(bool(r['ok'])) for r in rows], np.int8),
             lg=np.array([np.nan if r.get('lg') is None else float(r['lg']) for r in rows], np.float32),
             bad=np.array([int(r.get('bad') or 0) for r in rows], np.int8), nrep=np.array([int(r.get('nrep') or 0) for r in rows], np.int16),
             hash_off=np.array([hof(r) for r in rows], np.int16), seg_len=np.array([len(r['seg_tokens']) for r in rows], np.int16),
             gid=np.array([str(r.get('gid', '')) for r in rows]))
    with gzip.open(f'{od}/attempts_light.jsonl.gz', 'wt') as f:
        for r in rows: f.write(json.dumps({k_: v for k_, v in r.items() if k_ != 'seg_tokens'}) + '\n')
    for fn in ('items', 'attempts'):
        src = f'{a.data}/{fn}_{S}.jsonl.gz'
        if fn == 'items' and os.path.exists(src): os.system(f'cp {src} {od}/')
    # ---- validation: L18 state at hash_off . verifier probe vs the register logit lg
    v = [(i, t, x, y) for i, t, x, y in lg_pairs if t is not None]
    t_ = np.array([q[1] for q in v], float); x_ = np.array([q[2] for q in v]); y_ = np.array([q[3] for q in v])
    set_ = t_ > -29.5                                   # -30 = slot logit never set in the register
    nrep = np.array([int(rows[q[0]].get('nrep') or 0) for q in v])
    def cmp(tt, xx):
        if len(tt) < 2: return None
        d = np.abs(tt - xx)
        return dict(n=int(len(tt)), corr=float(np.corrcoef(tt, xx)[0, 1]), mae=float(d.mean()), p99=float(np.quantile(d, .99)),
                    max=float(d.max()), frac_gt_0p1=float((d > 0.1).mean()), frac_gt_0p5=float((d > 0.5).mean()))
    val = dict(bf16=cmp(t_[set_], x_[set_]), fp16_stored=cmp(t_[set_], y_[set_]),
               bf16_norep=cmp(t_[set_ & (nrep == 0)], x_[set_ & (nrep == 0)]), bf16_rep=cmp(t_[set_ & (nrep > 0)], x_[set_ & (nrep > 0)]),
               n_hash=len(lg_pairs), n_lg_unset_with_hash=int((~set_).sum()))
    worst = sorted(v, key=lambda q: -abs(q[1] - q[2]))[:15]
    val['worst'] = [dict(att=i, idx=rows[i]['idx'], seed=rows[i]['seed'], k=rows[i]['k'], nrep=rows[i].get('nrep'), lg=t, ours=round(x, 4)) for i, t, x, _ in worst]
    npos = len(pm['att'])
    info = dict(set=S, tag=a.tag, layers=LAY, layer_note='L<28: HF hidden_states[L] (layer-L input); 28: final-norm input (pre-norm)',
                D=D, dtype='float16', Lp=Lp, n_attempts=len(rows), n_items=len(items), n_pos=npos,
                kinds={'0': 'qend (qend.npy)', '1': 'nl', '2': 'hash', '3': 'last'}, shards=shards, ckpt=a.ckpt,
                compute_dtype=a.dtype, created=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
    json.dump(info, open(f'{od}/info.json', 'w'), indent=1)
    gb = (npos + len(items)) * nL * D * 2 / 1e9
    res = dict(info={k_: v_ for k_, v_ in info.items() if k_ != 'shards'}, n_shards=len(shards), out_dir=od, size_gb=round(gb, 2),
               timing=dict(total_s=round(time.time() - t_set, 1), fwd_s=round(t_fwd, 1), tokens=n_tok, padded=n_pad,
                           tok_per_s_fwd=round(n_tok / max(t_fwd, 1e-9)), attempts_per_s=round(len(rows) / (time.time() - t_set), 1)),
               hook_check=hs_check, qend_consistency_maxabs=(max(qend_dev) if qend_dev else None),
               kind_counts={str(k_): int(np.sum(np.array(pm['kind']) == k_)) for k_ in (1, 2, 3)}, validation=val)
    os.makedirs(f'{HERE}/out', exist_ok=True)
    json.dump(res, open(f'{HERE}/out/extract_{S}{a.tag}.json', 'w'), indent=1)
    print(json.dumps({k_: v_ for k_, v_ in res.items() if k_ != 'validation'}, indent=1))
    print('VALIDATION', json.dumps({k_: v_ for k_, v_ in val.items() if k_ != 'worst'}, indent=1), flush=True)
    print('worst', json.dumps(val['worst'][:8]), flush=True)


for S in a.sets.split(','):
    run_set(S)
print('EXTRACT DONE', flush=True)
