"""selcheck feasibility (step 1), runs on the B200 node (bash run.sh python3 fz.py ...).
For every candidate (item, answer x) of data/cands_{set}.jsonl the RL+calc host (HF Qwen3-0.6B-Base + ckpt_a2s2, bf16, NO calc
implant) scores x with several check formats. Context always = 8-shot prefix (C.prefix_ids) + check text (no attempts visible,
as with the forget mask).
 teacher-forced (one forward):
  v2   'Question: q\\nProposed answer: x\\nIs the proposed answer correct? Answer yes or no:'   -> lse(yes,Yes) - lse(no,No)
  v3   4 verdict shots + 'Question: q\\nProposed answer: x\\nCorrect:'                           -> same
  dir  'Question: q\\nAnswer: #### x\\n' : mean / sum logp of the ' x' tokens (no-CoT answer likelihood)
 sampled check (K samples, T=1, until a '\\n\\n' token or max_new), then teacher-forced verdict suffix:
  fwd  'Question: q\\nAnswer: The answer is x. Let's verify it step by step.\\n' + check
  bwd  'Question: q\\nAnswer: Check: if the answer is x, then' + check
       verdict suffix '\\nVerdict: the answer x is' -> logit(' correct') - lse(' incorrect',' wrong')
       agree = the check's last '#### y' equals x
 hidden states (layers LAY, layer input; 28 = final-norm input) at the scoring token of each format are kept on the node
 ($WSM_DATA/implants/selcheck/hs_{fmt}_{set}.pt) and linear probes are fit on trainC (grouped 5-fold OOF, full fit -> dev/dev2).
Output: out/fz_{set}.jsonl.gz (per candidate), out/fz_probe.json."""
import os, sys, json, gzip, time, argparse, re, math
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--sets', default='trainC,dev,dev2')
ap.add_argument('--K', type=int, default=4)
ap.add_argument('--max_new', type=int, default=224)
ap.add_argument('--bs', type=int, default=192)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--fmts', default='v2,v3,dir,fwd,bwd')
ap.add_argument('--hs_dir', default='$WSM_DATA/implants/selcheck')
ap.add_argument('--ckpt', default='$WSM_DATA/gsm_sel/ckpt_a2s2')
ap.add_argument('--tag', default='')
ap.add_argument('--part', default='')            # 'i/n': items rows[i::n] (tag gets _p{i})
a = ap.parse_args()
os.makedirs(a.hs_dir, exist_ok=True); os.makedirs(f'{HERE}/out', exist_ok=True)

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from safetensors.torch import load_file
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, _ = C.prefix_ids(tok); Lp = len(pre)
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
sd = load_file(f'{a.ckpt}/model.safetensors', device='cpu'); msd = m.state_dict()
with torch.no_grad():
    for k in msd:
        if k in sd: msd[k].copy_(sd[k].float().to(torch.bfloat16).to(dev))
del sd
m.generation_config.top_k = None; m.generation_config.top_p = None
print('host loaded, Lp', Lp, flush=True)
LAY = [10, 14, 18, 22, 26, 28]; NLY = m.config.num_hidden_layers; D = m.config.hidden_size
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
def tid(s):
    i = T(s); assert len(i) == 1, (s, i); return i[0]
YES = [tid(' yes'), tid(' Yes')]; NO = [tid(' no'), tid(' No')]
COR = [tid(' correct')]; INC = [tid(' incorrect'), tid(' wrong')]
V = len(tok)
NN = torch.tensor(['\n\n' in tok.decode([i]) for i in range(V)] + [False] * (m.config.vocab_size - V), device=dev)
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0

cap = {}; cur = {}
def _pre(L):
    def f(mod, args, kwargs):
        if cur.get('flat') is None: return
        h = args[0] if args else kwargs['hidden_states']
        cap[L] = h.reshape(-1, D).index_select(0, cur['flat']).float()
    return f
for L in LAY:
    (m.model.layers[L] if L < NLY else m.model.norm).register_forward_pre_hook(_pre(L), with_kwargs=True)

SHOTS = json.load(open(f"{os.environ.get('WSM_ROOT', HERE + '/../../../../..')}/data/gsm8k/shots_v1.json"))
VSH = ''.join(f"Question: {SHOTS[i]['question']}\nProposed answer: {x}\nCorrect: {v}\n\n"
              for i, x, v in [(1, 62, 'yes'), (2, 56, 'no'), (4, 7, 'yes'), (6, 60, 'no')])
CTX = {
    'v2': lambda q, x: f"Question: {q}\nProposed answer: {x}\nIs the proposed answer correct? Answer yes or no:",
    'v3': lambda q, x: VSH + f"Question: {q}\nProposed answer: {x}\nCorrect:",
    'dir': lambda q, x: f"Question: {q}\nAnswer: ####",
    'fwd': lambda q, x: f"Question: {q}\nAnswer: The answer is {x}. Let's verify it step by step.\n",
    'bwd': lambda q, x: f"Question: {q}\nAnswer: Check: if the answer is {x}, then",
}
VSUF = lambda x: f"\nVerdict: the answer {x} is"


def lse(lo, ids): return torch.logsumexp(lo[:, ids], -1)


@torch.no_grad()
def forward_score(seqs, pos):
    """right-padded teacher-forced batch; seqs list of id lists; pos list of lists of positions to read.
    returns logits (fp32, log_softmax) at those positions and hidden states (nL, npos, D) fp16 on cpu."""
    B = len(seqs); Tm = max(map(len, seqs))
    ids = torch.full((B, Tm), PAD, dtype=torch.long)
    for b, s in enumerate(seqs): ids[b, :len(s)] = torch.tensor(s)
    flat = [b * Tm + p for b, ps in enumerate(pos) for p in ps]
    cur['flat'] = torch.tensor(flat, device=dev)
    out = m.model(input_ids=ids.to(dev), use_cache=False).last_hidden_state
    cur['flat'] = None
    h = out.reshape(-1, D).index_select(0, torch.tensor(flat, device=dev))
    lp = torch.log_softmax(m.lm_head(h).float(), -1)
    hs = torch.stack([cap[L] for L in LAY], 0).half().cpu()
    return lp, hs


def batches(n, lens, budget):
    order = sorted(range(n), key=lambda i: lens[i]); i = 0
    while i < n:
        j = i
        while j < n and (j - i + 1) * lens[order[j]] <= budget: j += 1
        j = max(j, i + 1); yield order[i:j]; i = j


class StopNN(StoppingCriteria):
    def __init__(self, L0): self.L0 = L0
    def __call__(self, ids, scores, **kw):
        return NN[ids[:, self.L0:]].any(1)


@torch.no_grad()
def generate(ctxs, seed):
    """left-padded sampled generation, T=1, full vocab. returns list of generated id lists (truncated at the '\\n\\n' token, inclusive)."""
    torch.manual_seed(seed)
    B = len(ctxs); Tm = max(map(len, ctxs))
    ids = torch.full((B, Tm), PAD, dtype=torch.long); am = torch.zeros((B, Tm), dtype=torch.long)
    for b, s in enumerate(ctxs): ids[b, Tm - len(s):] = torch.tensor(s); am[b, Tm - len(s):] = 1
    g = m.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                   max_new_tokens=a.max_new, pad_token_id=PAD, stopping_criteria=StoppingCriteriaList([StopNN(Tm)]))
    out = []
    for b in range(B):
        r = g[b, Tm:].tolist(); cut = len(r)
        for j, t in enumerate(r):
            if NN[t]: cut = j + 1; break
        out.append(r[:cut])
    return out


NUMRE = re.compile(r'####\s*(-?[\d,]*\.?\d+)')
def ans_of(txt):
    mm = NUMRE.findall(txt)
    return mm[-1].replace(',', '') if mm else None


def run_set(S):
    rows = [json.loads(l) for l in open(f'{HERE}/cdata/cands_{S}.jsonl')]
    if a.limit: rows = rows[:a.limit]
    if a.part: pi, pn = map(int, a.part.split('/')); rows = rows[pi::pn]
    C_ = [(r, c) for r in rows for c in r['cands']]
    n = len(C_); print(S, 'cands', n, flush=True)
    res = [dict(idx=r['idx'], x=c['x'], ok=c['ok'], n=c['n']) for r, c in C_]
    HS = {}
    t0 = time.time()
    for F in [f for f in a.fmts.split(',') if f in ('v2', 'v3')]:
        seqs = [pre + T(CTX[F](r['q'], c['x'])) for r, c in C_]
        H = torch.zeros(len(LAY), n, D, dtype=torch.half)
        for bat in batches(n, [len(s) for s in seqs], 160000):
            lp, hs = forward_score([seqs[i] for i in bat], [[len(seqs[i]) - 1] for i in bat])
            sc = (lse(lp, YES) - lse(lp, NO)).tolist()
            for j, i in enumerate(bat): res[i][F] = sc[j]
            H[:, bat] = hs
        HS[F] = H; print(S, F, f'{time.time() - t0:.0f}s', flush=True)
    if 'dir' in a.fmts:
        seqs, pos = [], []
        for r, c in C_:
            ctx = pre + T(CTX['dir'](r['q'], c['x'])); xs = T(f" {c['x']}\n")
            seqs.append(ctx + xs); pos.append(list(range(len(ctx) - 1, len(ctx) + len(xs) - 1)))
        H = torch.zeros(len(LAY), n, D, dtype=torch.half)
        for bat in batches(n, [len(s) for s in seqs], 160000):
            lp, hs = forward_score([seqs[i] for i in bat], [pos[i] for i in bat])
            k = 0
            for j, i in enumerate(bat):
                tg = seqs[i][pos[i][0] + 1: pos[i][-1] + 2]
                v = [float(lp[k + t, tg[t]]) for t in range(len(tg))]
                res[i]['dir_sum'] = sum(v[:-1]); res[i]['dir_mean'] = sum(v[:-1]) / max(1, len(v) - 1)
                H[:, i] = hs[:, k + len(tg) - 1]          # state at the last x token (predicting '\n')
                k += len(tg)
        HS['dir'] = H; print(S, 'dir', f'{time.time() - t0:.0f}s', flush=True)
    for F in [f for f in a.fmts.split(',') if f in ('fwd', 'bwd')]:
        ctxs = [pre + T(CTX[F](r['q'], c['x'])) for r, c in C_]
        Hc = torch.zeros(a.K, len(LAY), n, D, dtype=torch.half); Hv = torch.zeros(a.K, len(LAY), n, D, dtype=torch.half)
        for i in range(n): res[i][F] = []
        for k in range(a.K):
            gens = [None] * n
            for bi, bat in enumerate(batches(n, [len(s) + a.max_new for s in ctxs], a.bs * 1500)):
                gg = generate([ctxs[i] for i in bat], seed=1000 * k + 17 * bi + (0 if F == 'fwd' else 7))
                for j, i in enumerate(bat): gens[i] = gg[j]
            seqs = []; pos = []
            for i, (r, c) in enumerate(C_):
                ce = ctxs[i] + gens[i]; vs = T(VSUF(c['x']))
                seqs.append(ce + vs); pos.append([len(ce) - 1, len(ce) + len(vs) - 1])
            for bat in batches(n, [len(s) for s in seqs], 160000):
                lp, hs = forward_score([seqs[i] for i in bat], [pos[i] for i in bat])
                vv = (lse(lp[1::2], COR) - lse(lp[1::2], INC)).tolist()
                for j, i in enumerate(bat):
                    txt = tok.decode(gens[i]); y = ans_of(txt)
                    res[i][F].append(dict(v=vv[j], y=y, agree=int(y is not None and C.correct(y, res[i]['x'])), nt=len(gens[i]),
                                          txt=txt[:600]))
                    Hc[k, :, i] = hs[:, 2 * j]; Hv[k, :, i] = hs[:, 2 * j + 1]
            print(S, F, 'k', k, f'{time.time() - t0:.0f}s', flush=True)
        HS[F + '_c'] = Hc; HS[F + '_v'] = Hv
    torch.save(dict(hs=HS, LAY=LAY, idx=[r['idx'] for r in res], ok=[r['ok'] for r in res]), f'{a.hs_dir}/hs_{S}{a.tag}.pt')
    with gzip.open(f'{HERE}/out/fz_{S}{a.tag}.jsonl.gz', 'wt') as f:
        for r in res: f.write(json.dumps(r) + '\n')
    print(S, 'done', f'{time.time() - t0:.0f}s', flush=True)


for S in a.sets.split(','):
    run_set(S)
print('FZ DONE', flush=True)
