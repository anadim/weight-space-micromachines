"""Is RL gaming the mistake detector? On DEV items: the host (+ calc_s4) writes its FIRST answer (no rethink attached),
then the frozen hybrid detector is read at the read point (the token whose next-token prediction is the first '####'),
exactly the features the circuit uses (hidden_states[12..24]). Reports AUROC of each detector head vs first-answer
correctness and the circuit's fire decision (hash path: sB > thB; end path: sA > thA and not hash) on RIGHT vs WRONG
first answers. python det_audit.py --ckpt DIR|base --out runs/det_X.json [--n 800]"""
import os, sys, re, json, random, argparse, torch, torch.nn.functional as F
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from rethink_step import DET
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='base'); ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=800); ap.add_argument('--B_max', type=int, default=128); a = ap.parse_args()
dev = 'cuda'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to(dev).eval()
pre, shot_q = C.prefix_ids(tok)
devset = C.load_train('B', shot_q=shot_q); rng = random.Random(123)
dg = [r for r in devset if r['kind'] == 'gsm']; ds = [r for r in devset if r['kind'] == 'scaled']; rng.shuffle(dg); rng.shuffle(ds)
items = dg[:a.n // 2] + ds[:a.n // 2]                     # the same dev-800 as train2.py
sd = Q.load_hf_params(dtype=torch.bfloat16)
if a.ckpt != 'base':
    from safetensors.torch import load_file
    ck = load_file(f'{a.ckpt}/model.safetensors', device=dev)
    for k in sd: sd[k].copy_(ck[k])
W = Q.fuse(sd)
imp = CalcImplant(hf, tok, a.B_max, fused=True); del hf
eng = Engine(W, pre, tok, implant=imp, B_max=a.B_max, S_max=544)
qids = [C.q_ids(tok, r['q']) for r in items]
out = eng.generate(qids, G=1, greedy=True, max_new=320)
P = torch.load(DET, map_location=dev)
layers = list(P['layers'])
hashtok = torch.tensor(['####' in tok.decode([i]) for i in range(len(tok))] + [False] * (W['E'].shape[0] - len(tok)), device=dev)
recs = []
for i0 in range(0, len(items), 16):
    sl = range(i0, min(i0 + 16, len(items)))
    rows, rp = [], []
    for i in sl:
        r = out['tokens'][i]
        k = next((j for j, t in enumerate(r) if hashtok[t]), None)
        if k is None: continue                             # no first answer
        rows.append(i); rp.append(len(qids[i]) + k - 1)    # read point: the token before '####'
    if not rows: continue
    seqs = [qids[i] + out['tokens'][i] for i in rows]
    T = max(map(len, seqs))
    ids = torch.tensor([s + [eng.eos] * (T - len(s)) for s in seqs], device=dev)
    feats = {}
    with torch.no_grad():                                  # fwd_chunk with the residual captured at the detector layers
        x = W['E'][ids]; pos = torch.arange(eng.Lp, eng.Lp + T, device=dev)
        cos, sin = eng.cos_t[pos], eng.sin_t[pos]
        mask = torch.ones(T, eng.Lp + T, dtype=torch.bool, device=dev); mask[:, eng.Lp:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
        for li, lw in enumerate(W['L']):
            if li in layers: feats[li] = x[torch.arange(len(rows)), torch.tensor(rp, device=dev)].float()
            q, k_, v = Q.qkv_rope(lw, Q.rms(x, lw['ln1']), cos, sin)
            K = torch.cat([eng.Kp[li][None].expand(len(rows), -1, -1, -1), k_], 2).repeat_interleave(2, 1)
            V = torch.cat([eng.Vp[li][None].expand(len(rows), -1, -1, -1), v], 2).repeat_interleave(2, 1)
            att = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            x = x + F.linear(att.transpose(1, 2).reshape(len(rows), T, -1), lw['Wo'])
            x = x + Q.mlp(lw, x)
    s = torch.cat([feats[l] for l in layers], -1)
    z = torch.relu(((s - P['mu'].float()) / P['sd'].float()) @ P['W1'].float().T + P['b1'].float()).double()
    sB = z @ P['WB'].double().reshape(-1) + P['bB']; sA = z @ P['WA'].double().reshape(-1) + P['bA']
    s0 = z @ P['W0'].double().reshape(-1) + P['b0']
    for j, i in enumerate(rows):
        ok = C.score(tok, out['tokens'][i], items[i]['gold'], 'first')[0]
        fb = float(sB[j]) > P['thB']; fa = (float(sA[j]) > P['thA']) and not fb
        recs.append(dict(i=i, kind=items[i]['kind'], correct=ok, sB=float(sB[j]), sA=float(sA[j]), s0=float(s0[j]), fire_hash=fb, fire_end=fa))


def auroc(pos, neg):
    if not pos or not neg: return float('nan')
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r = 0.0; rank = 0; cnt_neg = 0
    for v, lab in allv:
        if lab == 0: cnt_neg += 1
        else: r += cnt_neg
    return r / (len(pos) * len(neg))


wrong = [x for x in recs if not x['correct']]; right = [x for x in recs if x['correct']]
fire = lambda L: sum(x['fire_hash'] or x['fire_end'] for x in L) / max(1, len(L))
res = dict(ckpt=a.ckpt, n_items=len(items), n_first_answers=len(recs), acc_first=100 * len(right) / max(1, len(recs)),
           auroc_sB_wrong=auroc([x['sB'] for x in wrong], [x['sB'] for x in right]),
           auroc_sA_wrong=auroc([x['sA'] for x in wrong], [x['sA'] for x in right]),
           auroc_P0correct=auroc([x['s0'] for x in right], [x['s0'] for x in wrong]),
           fire_on_wrong=fire(wrong), fire_on_right=fire(right),
           fire_hash_wrong=sum(x['fire_hash'] for x in wrong) / max(1, len(wrong)),
           fire_hash_right=sum(x['fire_hash'] for x in right) / max(1, len(right)),
           mean_sB_wrong=sum(x['sB'] for x in wrong) / max(1, len(wrong)), mean_sB_right=sum(x['sB'] for x in right) / max(1, len(right)),
           records=recs)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != 'records'}), flush=True)
json.dump(res, open(a.out, 'w'))
