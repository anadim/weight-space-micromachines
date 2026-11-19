"""Direct stress of the forbidden-word / anti-repeat blocked set (the hash paths) on random register states.

For each trial: random 24-row state in circuit.py (forbid gate on, random slot hashes = hashes of real words / random values,
hcur = hash of a real word prefix / random / 0, random anti-repeat registers); the same registers are copied into the
compiled executor; both run write(h) with the same random h; the full 151,936-entry blocked mask, top/alt/chosen, the
emitted token and all registers are compared.
python hash_state_test.py --trials 400 --device cpu
"""
import argparse, os, random, sys, json, time, torch
ap = argparse.ArgumentParser(); ap.add_argument('--trials', type=int, default=400); ap.add_argument('--device', default='cpu'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
os.environ['CIRC_DEV'] = a.device
sys.setrecursionlimit(100000); torch.set_num_threads(8)
from transformers import AutoTokenizer, AutoModelForCausalLM
from circuit import Circuit
from execu import Executor
import export as EXP, lockstep_util as LU
dev = a.device; dt = torch.float32 if dev == 'cpu' else torch.bfloat16
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=dt).to(dev).eval()
circ = Circuit(tok, model, vecs='vecs2_forcefix.pt')
sd = torch.load('implant14_sd.pt')
sd['voc.E'] = model.lm_head.weight.detach().clone(); sd['voc.g'] = model.model.norm.weight.detach().float().clone()
sd['force.F'] = EXP.force_table(model.lm_head.weight, model.model.norm.weight, torch.load('vecs2_forcefix.pt')['force_fix'], dev).cpu()
ex = Executor(sd, device=dev, check=True)
T = circ.T; HP = 2147483647
R = random.Random(a.seed); g = torch.Generator().manual_seed(a.seed)
vocab = tok.convert_ids_to_tokens(list(range(151643)))
import re
words = sorted({t.lstrip('Ġ').lower() for t in vocab if re.fullmatch(r'Ġ?[a-zA-Z]{3,10}', t)})


def hseq(ids):
    """circuit.py's answer-side rolling hash over a token sequence (lines 632-634) -> list of hcur after each token."""
    h = 0; out = []
    for i in ids:
        i = torch.tensor([i], device=dev)
        hn = torch.where(T['LEAD'][i] > 0.5, T['HT'][i], (torch.tensor([h], device=dev) * T['POWT'][i] + T['HT'][i]) % HP)
        hn = torch.where((T['HASAN'][i] < 0.5) | (T['TRAIL'][i] > 0.5), torch.zeros_like(hn), hn)
        h = int(hn); out.append(h)
    return out


def rand_word_ids():
    w = R.choice(words) if R.random() < 0.6 else ''.join(R.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(R.randint(4, 14)))
    w = R.choice([w, w.capitalize(), w.upper()])
    return tok(R.choice(['', ' ']) + w, add_special_tokens=False).input_ids


st = dict(trials=0, rows=0, blk_mm=0, dec_mm=0, reg_mm=0, blocked_tokens=0, top_blocked=0, live_rows=0)
t0 = time.time()
B = 24
for trial in range(a.trials):
    circ.reset(B)
    circ.ans[:] = 1; circ.body[:] = 1
    circ.L['forbid'][:] = 1.0
    for k in ['para', 'high', 'two']: circ.L[k][:] = (torch.rand(B, generator=g) < 0.3).float().to(dev)
    circ.N_par[:] = 3; circ.N_high[:] = 2
    hc = []; tgt_ids = []
    for b in range(B):
        ws = [rand_word_ids() for _ in range(6)]
        for s in range(6):
            r = R.random()
            circ.slot_h[b, s] = hseq(ws[s])[-1] if r < 0.7 else (R.randrange(HP) if r < 0.85 else 0)
            circ.slot_ok[b, s] = float(R.random() < 0.7)
        w = R.choice(ws); hs = hseq(w); k = R.randrange(len(w))
        r = R.random()
        circ.hcur[b] = hs[k - 1] if (k > 0 and r < 0.6) else (R.randrange(HP) if r < 0.8 else 0)
        tgt_ids.append(w[k])                                          # the next piece of that word: make it the host's top token
        circ.cur_pf[b] = R.choice([-1, hs[0]]); circ.prev_pf[b] = R.choice([-1, hs[0], R.randrange(HP)]); circ.ans_pf[b] = R.choice([-1, hs[0]])
        circ.forced_brk[b] = float(R.random() < 0.5); circ.seg[b] = float(R.random() < 0.5)
        circ.k6[b] = float(R.random() < 0.5); circ.w6l[b] = float(R.random() < 0.5) * 3
    ex.reset(B)
    LU.REGS = ex.regs
    Rr = LU.ref_regs(circ)
    ex.state = torch.stack([Rr[n] for n in ex.regs], 1).to(dev).double()
    tgt = torch.tensor(tgt_ids, device=dev)
    h = (model.lm_head.weight[tgt].float() * 4 + torch.randn(B, model.lm_head.weight.shape[1], generator=g).to(dev) * 0.3).to(dt)
    blk_r = LU.ref_blk(circ)
    Wr = circ.write(h); tok_r = circ.emit_token(); act_r = Wr.abs().sum(-1) > 0
    Wc = ex.write(h)
    st['trials'] += 1; st['rows'] += B
    if not torch.equal(ex._last_blk.float().cpu(), blk_r.float().cpu()): st['blk_mm'] += 1
    act_c = ex.wout['active'] > 0.5; tok_c = ex.wout['tokf'].long()
    if not (torch.equal(act_c.cpu(), act_r.cpu()) and torch.equal(torch.where(act_c, tok_c, -1).cpu(), torch.where(act_r, tok_r, -1).cpu())): st['dec_mm'] += 1
    if LU.cmp_regs(circ, ex): st['reg_mm'] += 1
    st['blocked_tokens'] += int(blk_r.sum()); st['top_blocked'] += int(ex.wout['blocked'].sum())
    if trial % 50 == 0: print(trial, json.dumps(st), f'{time.time() - t0:.0f}s', flush=True)
print('FINAL', json.dumps(st), flush=True)
