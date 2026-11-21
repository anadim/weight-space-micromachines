"""lane prmstep: PER-LINE residual features (layers 14/18/22/27 = sidenet c4 layer inputs) at every model-written line end of
each attempt (same line-end definition and teacher-forced context as sidenet extract_sn.py: tokens of seg_tokens containing '\\n'
with phrase_len <= off < hash_off (no '####': < len)), plus a step label per line:
  line text = decode(seg_tokens[prev_line_end+1 .. off]); result = first number after the LAST '=' in the line (None -> label -1);
  label 1 if result matches a gold intermediate value (GSM8K train annotation <<..=c>>) or the gold final answer (tables/goldvals.json,
  rel tol 1e-3 or equal after rounding to 2 dp), else 0.
Out (node): OUT_ROOT/{set}/X.npy (nlines, 4, 1024) fp16, lines.npz (att, off, lab, val); meta.npz per attempt (as extract_sn)."""
import os, sys, json, gzip, time, argparse, re
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
ap = argparse.ArgumentParser()
ap.add_argument('--sets', default='trainC,dev,dev2')
ap.add_argument('--out_root', default='$WSM_DATA/implants/prmstep')
ap.add_argument('--ckpt', default='$WSM_DATA/gsm_sel/ckpt_a2s2')
ap.add_argument('--probe', default=f'{HERE}/verifier_probe_L18.pt')
ap.add_argument('--tok_budget', type=int, default=131072)
ap.add_argument('--limit_items', type=int, default=0)
ap.add_argument('--tag', default='')
a = ap.parse_args()
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
dev = 'cuda'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, _ = C.prefix_ids(tok); Lp = len(pre)
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
sd = load_file(f'{a.ckpt}/model.safetensors', device='cpu'); msd = m.state_dict()
with torch.no_grad():
    for k in msd:
        if k != 'lm_head.weight': msd[k].copy_(sd[k].float().to(torch.bfloat16).to(dev))
del sd
LAY = [14, 18, 22, 27]; D = m.config.hidden_size
PR = torch.load(a.probe); pw = PR['w'].float().to(dev); pb = float(PR['b'])
NLTOK = np.array(['\n' in tok.decode([i]) for i in range(len(tok))] + [False] * (m.config.vocab_size - len(tok)))
GV = json.load(open(f'{HERE}/tables/goldvals.json'))
NUMRE = re.compile(r'-?\$?\d[\d,]*(?:\.\d+)?')
cap = {}; cur = {}
def _pre(L):
    def f(mod, args, kwargs):
        h = args[0] if args else kwargs['hidden_states']
        cap[L] = h.reshape(-1, D).index_select(0, cur['flat'])
    return f
for L in sorted(set(LAY + [18])): m.model.layers[L].register_forward_pre_hook(_pre(L), with_kwargs=True)


def label(txt, gv):
    if '=' not in txt: return -1, np.nan
    mm = NUMRE.search(txt.rsplit('=', 1)[1])
    if not mm: return -1, np.nan
    try: v = float(mm.group(0).replace('$', '').replace(',', ''))
    except ValueError: return -1, np.nan
    ok = any(abs(v - g) <= 1e-3 * max(1.0, abs(g)) or round(v, 2) == round(g, 2) for g in gv)
    return int(ok), v


def run_set(S):
    t0 = time.time()
    rows = [json.loads(l) for l in gzip.open(f'{HERE}/tables/attempts_{S}.jsonl.gz', 'rt')]
    if a.limit_items:
        ks = set(sorted({r['idx'] for r in rows})[:a.limit_items]); rows = [r for r in rows if r['idx'] in ks]
    n = len(rows); od = f'{a.out_root}/{S}{a.tag}'; os.makedirs(od, exist_ok=True)
    qids = {}
    for r in rows:
        if r['idx'] not in qids: qids[r['idx']] = C.q_ids(tok, r['q'])
    POS, LA, LO, LL, LV, ST = [], [], [], [], [], []
    for i, r in enumerate(rows):
        st = r['seg_tokens']; ho = r.get('hash_off'); end = int(ho) if ho is not None else len(st)
        pl = int(r.get('phrase_len') or 0)
        nl = [o for o in range(pl, end) if NLTOK[st[o]]]
        hp = int(ho) if ho is not None else len(st) - 1
        POS.append((hp, nl)); ST.append(len(LA)); prev = pl
        for o in nl:
            lab, v = label(tok.decode(st[prev:o + 1]), GV[r['q']]); prev = o + 1
            LA.append(i); LO.append(o); LL.append(lab); LV.append(v)
    ST.append(len(LA)); NLN = len(LA); ST = np.array(ST)
    X = np.lib.format.open_memmap(f'{od}/X.npy', 'w+', np.float16, (NLN, len(LAY), D))
    lg_ours = np.full(n, np.nan, np.float32)
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
        flat, dst = [], []
        for b, (i, s) in enumerate(zip(bat, seqs)):
            ids[b, :len(s)] = torch.tensor(s); flat.append(b * T + Lp + len(qids[rows[i]['idx']]) + POS[i][0])
        for b, i in enumerate(bat):
            qb = Lp + len(qids[rows[i]['idx']])
            for j, o in enumerate(POS[i][1]): flat.append(b * T + qb + o); dst.append(ST[i] + j)
        ntok += sum(map(len, seqs)); cur['flat'] = torch.tensor(flat, device=dev)
        with torch.no_grad():
            m.model(input_ids=ids.to(dev), use_cache=False)
            lg_ours[np.array(bat)] = (cap[18][:B].float() @ pw + pb).cpu().numpy()
            if dst:
                Cc = torch.stack([cap[L][B:] for L in LAY], 1).to(torch.float16).cpu().numpy()
                dd = np.array(dst); s_ = np.argsort(dd); X[dd[s_]] = Cc[s_]
    X.flush(); del X
    LL = np.array(LL, np.int8)
    np.savez(f'{od}/lines.npz', att=np.array(LA, np.int32), off=np.array(LO, np.int16), lab=LL, val=np.array(LV, np.float64), start=ST)
    meta = dict(idx=np.array([r['idx'] for r in rows], np.int32), seed=np.array([r['seed'] for r in rows], np.int32),
                k=np.array([r['k'] for r in rows], np.int16), ok=np.array([int(r['ok']) for r in rows], np.int8),
                lg=np.array([np.nan if r.get('lg') is None else float(r['lg']) for r in rows], np.float32),
                has_hash=np.array([r.get('hash_off') is not None for r in rows], np.int8), lg_ours=lg_ours)
    np.savez(f'{od}/meta.npz', **meta)
    t = meta['lg']; okm = meta['has_hash'].astype(bool) & np.isfinite(t) & (t > -29.5)
    okA = meta['ok'][LA].astype(bool) if NLN else np.zeros(0, bool)
    val = dict(set=S, n=n, lines=NLN, labeled=int((LL >= 0).sum()), good=int((LL == 1).sum()),
               good_frac_in_right_att=float((LL[okA] == 1).sum() / max(1, (LL[okA] >= 0).sum())),
               good_frac_in_wrong_att=float((LL[~okA] == 1).sum() / max(1, (LL[~okA] >= 0).sum())),
               corr_lg=float(np.corrcoef(t[okm], lg_ours[okm])[0, 1]), mae_lg=float(np.abs(t[okm] - lg_ours[okm]).mean()),
               tokens=ntok, secs=round(time.time() - t0, 1))
    os.makedirs(f'{HERE}/out', exist_ok=True)
    json.dump(val, open(f'{HERE}/out/extract_{S}{a.tag}.json', 'w'), indent=1)
    print('VALIDATION', json.dumps(val), flush=True)


for S in a.sets.split(','): run_set(S)
print('EXTRACT DONE', flush=True)
