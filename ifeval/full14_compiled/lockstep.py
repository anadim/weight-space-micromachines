"""Lockstep equivalence test: circuit.py (reference) and the compiled executor receive IDENTICAL inputs; after every
step()/write() all 414 registers, the full blocked-token mask (151,936 entries; the hash check), top/alt/chosen, the
emitted token and the write row are compared exactly.

python lockstep.py --device cpu|cuda --batches 0,5,22 [--h synth|model] [--closed 1] [--check 1]
Inputs: IFEval prompts (chat template, run.py batches of 24, left padding) + the reference implant answers (teacher tokens).
"""
import argparse, json, os, sys, time, torch
ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cpu'); ap.add_argument('--batches', default='0'); ap.add_argument('--h', default='synth')
ap.add_argument('--closed', type=int, default=1); ap.add_argument('--check', type=int, default=1); ap.add_argument('--maxans', type=int, default=4000)
ap.add_argument('--data', default='data_ifeval.jsonl'); ap.add_argument('--answers', default='ref_answers.jsonl')
ap.add_argument('--sd', default=''); ap.add_argument('--bs', type=int, default=24); ap.add_argument('--log', default='')
ap.add_argument('--seqs', default='')                      # optional: jsonl of {key, prompt_ids, answer_ids} (stress sets)
a = ap.parse_args()
os.environ['CIRC_DEV'] = a.device
sys.setrecursionlimit(100000)
torch.set_num_threads(8)
from transformers import AutoTokenizer, AutoModelForCausalLM
from circuit import Circuit, S_W
from execu import Executor
import export as EXP

dev = a.device
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
dt = torch.float32 if dev == 'cpu' else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=dt).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
VECS = 'vecs2_forcefix.pt'
circ = Circuit(tok, model, vecs=VECS)
if a.sd and os.path.exists(a.sd):
    sd = torch.load(a.sd)
else:
    sd, meta, _, _ = EXP.make(dev, dt)
# the lockstep host may be fp32 (CPU): use ITS head rows / force table so both sides see the same weights
sd['voc.E'] = model.lm_head.weight.detach().clone()
sd['voc.g'] = model.model.norm.weight.detach().float().clone()
sd['force.F'] = EXP.force_table(model.lm_head.weight, model.model.norm.weight, torch.load(VECS)['force_fix'], dev).cpu()
ex = Executor(sd, device=dev, check=bool(a.check))
from tables import CLASSES
CI = {c: i for i, c in enumerate(CLASSES)}
V = model.lm_head.weight.shape[0]


def ref_regs(c):
    out = {}
    B = c.B
    for n in ex.regs:
        if n == 't': v = torch.full((B,), float(c.t))
        elif n.startswith('win'): v = c.win[:, int(n[3:])]
        elif n.startswith('cl.'): v = c.cl[:, CI[n[3:]]]
        elif n.startswith('se.'): v = c.se[:, CI[n[3:]]]
        elif n.startswith('L.'): v = c.L[n[2:]]
        elif n.startswith('Ls.'): v = c.Ls[n[3:]]
        elif n.startswith('done.'): v = c.done[n[5:]]
        elif n.startswith('cand.'): v = c.cand[n[5:]]
        elif n.startswith('cue.'):
            _, k, w = n.split('.'); v = c.cueR[k][0 if w == 'clc' else 1] if k in c.cueR else torch.zeros(B)
        elif n.startswith('slot_h'): v = c.slot_h[:, int(n[6:])]
        elif n.startswith('slot_ok'): v = c.slot_ok[:, int(n[7:])]
        elif n.startswith('pre') and n[3:].isdigit(): v = c.pre[:, int(n[3:])]
        elif n.startswith('suf') and n[3:].isdigit(): v = c.suf[:, int(n[3:])]
        else: v = getattr(c, n)
        out[n] = v.double().to('cpu') if torch.is_tensor(v) else torch.full((B,), float(v))
    return out


def ref_blk(c):
    """verbatim copy of circuit.py write() lines 702-728 (the blocked set, before any write)."""
    g = c.gates(); a_ = c.ans; T = c.T
    cnt = torch.where(g['p3'] > 0.5, c.d3, c.kp)
    short = g['para'] * (cnt < g['npar'] - 1).float()
    eosb = torch.clamp(g['two'] * (c.k6 > 0).float() * (c.w6 < 5).float() * (c.n6 < 40).float() + c.hforced * c.hl_open * (c.hl_cont == 0).float()
                       + short * (1 - c.seg) + g['bul'] * c.ln_bul * (c.ln_cont == 0).float(), 0, 1)
    nnb = g['pn'] * (c.kp >= g['npar'] - 1).float()
    blk = (g['nc'][:, None] * T['COMMASET'][None]
           + torch.clamp(g['two'] * (c.k6 > 0).float() + g['p3'] + g['two'] * (c.k6 == 0).float() * (c.wa < 5).float(), 0, 1)[:, None] * T['STAR3SET'][None]
           + nnb[:, None] * T['NN'][None] + (nnb * c.at_ls)[:, None] * T['STARTNL'][None]
           + eosb[:, None] * c.EOSMASK[None]
           + (g['bul'] * c.at_ls * (c.bcount >= c.N_bul).float())[:, None] * T['BULST'][None]
           + (g['bul'] * c.ln_bul * (c.ln_cont == 0).float() * (1 - c.at_ls))[:, None] * T['NL'][None])
    HP = 2147483647
    cand = torch.where(T['LEAD'][None] > 0.5, T['HT'][None], (c.hcur[:, None] * T['POWT'][None] + T['HT'][None]) % HP)
    live = (c.slot_ok > 0.5) & (g['forbid'][:, None] > 0.5)
    tgt = torch.where(live, c.slot_h, -5)
    blk = blk + ((cand[:, None, :] == tgt[:, :, None]).any(1) & (T['HASAN'][None] > 0.5)).float()
    rep_on = torch.clamp(g['para'] + g['high'], 0, 1) * c.forced_brk * (1 - c.seg) * (c.prev_pf >= 0).float()
    two_on = g['two'] * (c.k6 > 0).float() * (c.w6l == 0).float() * (c.ans_pf >= 0).float()
    pf = torch.where(two_on > 0.5, c.ans_pf, torch.where(rep_on > 0.5, c.prev_pf, -5))
    blk = blk + ((T['HT'][None] == pf[:, None]) & (T['HASAN'][None] > 0.5) & (T['LEAD'][None] + (1 - c.hcur.clamp(max=1).float())[:, None] > 0)).float()
    return torch.clamp(blk, 0, 1), (live.any(1) | (pf >= 0)).cpu()


stats = dict(steps=0, writes=0, reg_mismatch=0, blk_mismatch=0, dec_mismatch=0, W_mismatch=0, W_bf16_mismatch=0, active=0,
             blocked=0, hash_live_writes=0, hash_blocked_tokens=0)
logf = open(a.log, 'a') if a.log else None


def log(*s):
    msg = ' '.join(str(x) for x in s); print(msg, flush=True)
    if logf: logf.write(msg + '\n'); logf.flush()


def cmp_regs(where_):
    import lockstep_util as LU
    bad = LU.cmp_regs(circ, ex)
    stats['sat'] = dict(LU.SAT)
    if bad:
        stats['reg_mismatch'] += 1
        if stats['reg_mismatch'] <= 20: log('REG MISMATCH', where_, bad[:6])
    return not bad


def run_batch(P, A, tag):
    B = len(P); L = max(map(len, P))
    x = torch.tensor([[151643] * (L - len(p)) + p for p in P], device=dev)
    circ.reset(B); ex.reset(B)
    for t in range(L):
        circ.step(x[:, t]); ex.step(x[:, t]); stats['steps'] += 1
        if not cmp_regs(f'{tag} prompt t={t}'): return False
    ptr = [0] * B; done = torch.zeros(B, dtype=torch.bool, device=dev)
    E = model.lm_head.weight
    gen = torch.Generator(device='cpu').manual_seed(1234)
    for s in range(a.maxans):
        tgt = torch.tensor([A[i][ptr[i]] if ptr[i] < len(A[i]) else 151645 for i in range(B)], device=dev)
        noise = torch.randn(B, E.shape[1], generator=gen).to(dev)
        h = (E[tgt].float() * 4 + noise * 0.3).to(dt)
        blk_r, hashlive = ref_blk(circ)
        Wr = circ.write(h); tok_r = circ.emit_token(); act_r = (Wr.abs().sum(-1) > 0)
        Wc = ex.write(h)
        stats['writes'] += 1
        bm = ex._last_blk.float()
        if not torch.equal(bm.cpu(), blk_r.float().cpu()):
            stats['blk_mismatch'] += 1
            if stats['blk_mismatch'] <= 20:
                rr = torch.nonzero(bm.cpu() != blk_r.float().cpu())[:5].tolist(); log('BLK MISMATCH', tag, s, rr)
        stats['hash_live_writes'] += int(hashlive.sum()); stats['hash_blocked_tokens'] += int(blk_r.sum())
        act_c = ex.wout['active'] > 0.5; tok_c = ex.wout['tokf'].long()
        if not (torch.equal(act_c.cpu(), act_r.cpu()) and torch.equal(torch.where(act_c, tok_c, -1).cpu(), torch.where(act_r, tok_r, -1).cpu())):
            stats['dec_mismatch'] += 1
            if stats['dec_mismatch'] <= 20: log('DECISION MISMATCH', tag, s, act_r.tolist(), act_c.tolist(), tok_r.tolist(), tok_c.tolist())
        if not torch.equal(Wc.float().cpu(), Wr.float().cpu()): stats['W_mismatch'] += 1
        if not torch.equal(Wc.to(torch.bfloat16).cpu(), Wr.to(torch.bfloat16).cpu()): stats['W_bf16_mismatch'] += 1
        stats['active'] += int(act_r.sum()); stats['blocked'] += int(ex.wout['blocked'].sum())
        if not cmp_regs(f'{tag} write s={s}'): return False
        # next token: the forced token if the circuit writes (closed loop), else the teacher token
        nxt = torch.where(act_r & torch.tensor(bool(a.closed), device=dev), tok_r, tgt)
        for i in range(B):
            if not (a.closed and bool(act_r[i])): ptr[i] += 1
        fin = (nxt == 151645) | (nxt == 151643)
        done = done | fin
        if bool(done.all()): break
        nxt = torch.where(done, torch.full_like(nxt, 151643), nxt)
        circ.step(nxt); ex.step(nxt); stats['steps'] += 1
        if not cmp_regs(f'{tag} ans s={s}'): return False
    return True


def main():
    if a.seqs:
        rows = [json.loads(l) for l in open(a.seqs)]
        batches = [rows[i:i + a.bs] for i in range(0, len(rows), a.bs)]
        sel = range(len(batches)) if a.batches == 'all' else [int(b) for b in a.batches.split(',')]
        for bi in sel:
            t0 = time.time(); ok = run_batch([r['prompt_ids'] for r in batches[bi]], [r['answer_ids'] for r in batches[bi]], f'seqs b{bi}')
            log(f'seq batch {bi} ok={ok} {time.time() - t0:.0f}s', json.dumps(stats))
        return
    docs = [json.loads(l) for l in open(a.data)]
    ans = {}
    for l in open(a.answers):
        r = json.loads(l); ans[r['key']] = r['response']
    P_all = [tok(tok.apply_chat_template([{'role': 'user', 'content': d['prompt']}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False), add_special_tokens=False).input_ids for d in docs]
    order = sorted(range(len(docs)), key=lambda i: (len(P_all[i]), docs[i]['key']))
    batches = [order[b:b + a.bs] for b in range(0, len(order), a.bs)]
    sel = range(len(batches)) if a.batches == 'all' else [int(b) for b in a.batches.split(',')]
    for bi in sel:
        idx = batches[bi]
        A = [tok(ans[docs[i]['key']], add_special_tokens=False).input_ids for i in idx]
        t0 = time.time()
        ok = run_batch([P_all[i] for i in idx], A, f'b{bi}')
        log(f'batch {bi} ok={ok} {time.time() - t0:.0f}s', json.dumps(stats))


main()
log('FINAL', json.dumps(stats))
