"""run.py (frozen Team D decode loop, identical batches/padding/hook) with the COMPILED implant (exported state_dict + executor).

python run_c.py DATA.jsonl OUT.jsonl --sd implant14_sd.pt [--shadow 1] [--bs 24] [--shard i/n] [--max_new 2048]
--shadow 1: circuit.py runs alongside on the same inputs (it does NOT drive generation); every step()/write() its 414 registers,
its blocked set, its emitted token and its bf16 write are compared with the compiled implant's; mismatches -> OUT.shadow.log.
"""
import argparse, json, os, pickle, sys, time, torch
sys.setrecursionlimit(100000)
from common import load, EOS_IDS
from execu import Executor

ap = argparse.ArgumentParser()
ap.add_argument('data'); ap.add_argument('out'); ap.add_argument('--sd', default='implant14_sd.pt'); ap.add_argument('--shadow', type=int, default=0)
ap.add_argument('--bs', type=int, default=24); ap.add_argument('--max_new', type=int, default=2048)
ap.add_argument('--shard', default='0/1'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--keys', default='')
ap.add_argument('--batches', default='')
a = ap.parse_args()

tok, model = load()
docs = [json.loads(l) for l in open(a.data)]
if a.keys: ks = set(map(int, a.keys.split(','))); docs = [d for d in docs if d['key'] in ks]
if a.limit: docs = docs[:a.limit]


def msgs(d):
    m = []
    if d.get('system'): m.append({'role': 'system', 'content': d['system']})
    m += d.get('history') or []
    m.append({'role': 'user', 'content': d['prompt']})
    return m


P_all = [tok(tok.apply_chat_template(msgs(d), tokenize=False, add_generation_prompt=True, enable_thinking=False),
             add_special_tokens=False).input_ids for d in docs]
order = sorted(range(len(docs)), key=lambda i: (len(P_all[i]), docs[i]['key']))
batches = [order[b:b + a.bs] for b in range(0, len(order), a.bs)]
si, sn = map(int, a.shard.split('/'))
mine = [bi for bi in range(len(batches)) if bi % sn == si]
if a.batches: mine = [int(b) for b in a.batches.split(',')]


class Compiled:
    """circuit.py interface on top of the executor (reset / step / write / report)."""

    def __init__(self, sd):
        self.ex = Executor(sd, device='cuda')
        self.E = self.ex.sd['voc.E']; self.g = self.ex.sd['voc.g']; self.eps = float(self.ex.sd['voc.eps'])

    def reset(self, B): self.ex.reset(B); self.ffail = torch.zeros(B, device='cuda')

    def step(self, ids): self.ex.step(ids)

    def write(self, h):
        W = self.ex.write(h)
        # diagnostic only (as circuit.py): did the forced token win on the host readout after the write?
        act = self.ex.wout['active'] > 0.5; tokf = self.ex.wout['tokf'].long()
        x = (h + W).float(); x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g
        self.ffail += (act & ((x.to(self.E.dtype) @ self.E.T).argmax(-1) != tokf)).float()
        return W

    def report(self):
        """same fields as circuit.Circuit.report (decoding of captured spans is diagnostic, outside the implant)"""
        ex = self.ex; R = lambda n: ex.reg(n).long().tolist()
        rules = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'pps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul', 'star3']
        Lv = {k: R('L.' + k) for k in rules}
        mem = ex.mem.long()
        out = []
        for i in range(ex.B):
            r = {k: 1 for k in rules if Lv[k][i] > 0}
            for k in ['N_par', 'N_sec', 'N_bul', 'N_high', 'nth_k', 'nth_last']: r[k] = R(k)[i]
            r['nth_word'] = tok.decode(mem[i, R('nth_s')[i]:R('nth_s')[i] + R('nth_len')[i]].tolist()) if r.get('nth') else ''
            ex_ = R('end_x')[i]
            r['end_phrase'] = (tok.decode(mem[i, R('end_s')[i]:R('end_s')[i] + R('end_len')[i]].tolist()) +
                               (tok.decode([ex_]) if ex_ >= 0 else '')) if r.get('end') else ''
            r['rep'] = tok.decode(mem[i, R('rep_s')[i]:R('rep_s')[i] + R('rep_len')[i]].tolist()) if r.get('repeat') else ''
            r['forbid_h'] = [R(f'slot_h{s}')[i] for s in range(6) if R(f'slot_ok{s}')[i] > 0]
            r['fires'] = R('fires')[i]; r['ffail'] = int(self.ffail[i])
            out.append(r)
        return out


sd = torch.load(a.sd)
CIRC = Compiled(sd)
REF = None
slog = open(a.out + '.shadow.log', 'a') if a.shadow else None
SH = dict(steps=0, writes=0, reg_mm=0, blk_mm=0, dec_mm=0, W_mm=0)
if a.shadow:
    os.environ.setdefault('VECS', 'vecs2_forcefix.pt')
    from circuit import Circuit
    REF = Circuit(tok, model)
    import lockstep_util as LU
state = {'on': False}


def _hook(mod, x):
    if not state['on']: return None
    h = x[0]
    hl = h[:, -1]
    if REF is not None:
        blk_r = LU.ref_blk(REF)
        Wr = REF.write(hl); tok_r = REF.emit_token(); act_r = Wr.abs().sum(-1) > 0
    W = CIRC.write(hl)
    if REF is not None:
        SH['writes'] += 1
        ex = CIRC.ex
        if not torch.equal(ex._last_blk.float(), blk_r.float()):
            SH['blk_mm'] += 1; slog.write(f'BLK step={SH["writes"]} rows={torch.nonzero((ex._last_blk.float() != blk_r.float()).any(1))[:, 0].tolist()}\n')
        act_c = ex.wout['active'] > 0.5; tok_c = ex.wout['tokf'].long()
        if not (torch.equal(act_c, act_r) and torch.equal(torch.where(act_c, tok_c, -1), torch.where(act_r, tok_r, -1))):
            SH['dec_mm'] += 1; slog.write(f'DEC step={SH["writes"]} ref={act_r.tolist()} {tok_r.tolist()} comp={act_c.tolist()} {tok_c.tolist()}\n')
        if not torch.equal(W.to(h.dtype), Wr.to(h.dtype)): SH['W_mm'] += 1
        bad = LU.cmp_regs(REF, ex)
        if bad: SH['reg_mm'] += 1; slog.write(f'REG(write) {bad[:6]}\n')
        slog.flush()
    return (h + W[:, None].to(h.dtype),)


model.model.norm.register_forward_pre_hook(_hook)


def cstep(ids):
    CIRC.step(ids)
    if REF is not None:
        REF.step(ids); SH['steps'] += 1
        bad = LU.cmp_regs(REF, CIRC.ex)
        if bad: SH['reg_mm'] += 1; slog.write(f'REG(step) {bad[:6]}\n'); slog.flush()


@torch.no_grad()
def run(idx):
    P = [P_all[i] for i in idx]; B = len(P); L = max(map(len, P))
    x = torch.tensor([[EOS_IDS[1]] * (L - len(p)) + p for p in P], device='cuda')
    att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in P], device='cuda')
    out = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device='cuda')
    CIRC.reset(B)
    if REF is not None: REF.reset(B)
    for t in range(L): cstep(x[:, t])
    state['on'] = True
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    for step in range(a.max_new):
        nxt = lg.argmax(-1)
        fin = (nxt == EOS_IDS[0]) | (nxt == EOS_IDS[1])
        for i in torch.nonzero(~done & ~fin)[:, 0].tolist(): out[i].append(nxt[i].item())
        done = done | fin
        if bool(done.all()): break
        nxt = torch.where(done, torch.full_like(nxt, EOS_IDS[1]), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        cstep(nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    state['on'] = False
    info = CIRC.report()
    if REF is not None:
        rinfo = REF.report()
        for j in range(B):
            if rinfo[j] != info[j]: slog.write(f'REPORT key={docs[idx[j]]["key"]} ref={rinfo[j]} comp={info[j]}\n')
        slog.write(f'batch done {json.dumps(SH)} sat={json.dumps(LU.SAT)}\n'); slog.flush()
    return [(docs[i], tok.decode(out[j]), bool(done[j]), len(out[j]), info[j]) for j, i in enumerate(idx)]


part = a.out + '.partial'; res = {}
if os.path.exists(part): res = pickle.load(open(part, 'rb')); print('resumed', len(res), flush=True)
t0 = time.time()
for bi in mine:
    if bi in res: continue
    while True:
        try: res[bi] = run(batches[bi]); break
        except torch.OutOfMemoryError:
            state['on'] = False
            torch.cuda.empty_cache(); print('OOM batch', bi, flush=True); time.sleep(20)
    pickle.dump(res, open(part, 'wb'))
    print(f'{len(res)}/{len(mine)} batches {time.time() - t0:.0f}s shadow={json.dumps(SH)}', flush=True)
rows = sorted([r for v in res.values() for r in v], key=lambda r: r[0]['key'])
with open(a.out, 'w') as f:
    for d, resp, fin, n, info in rows:
        f.write(json.dumps({'key': d['key'], 'response': resp, 'finish': 'stop' if fin else 'length', 'ntok': n, 'circ': info}) + '\n')
os.remove(part); print('done', time.time() - t0, json.dumps(SH))
