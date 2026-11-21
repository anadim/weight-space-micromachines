# TK prime check: TKDual.prime_ids_seq (Lt sequential kernel steps) vs prime_ids_par (one step over U*W scratch rows).
# Bitwise comparison of every TK state tensor of the target rows, then 64 decode steps on real streams from both states.
# exec'd by tw_dump_lean.py like rl_hook.py, RL_FASTROLL=1 (RL_RUN = scratch dir).
import os as _o5
_src = open(_o5.path.join(_o5.environ['STK_DIR'], 'rl_hook.py')).read()
_src = _src[:_src.index('\ndef fastcheck')]
exec(_src)
import json as _js5, time as _tm5
OUT = _o5.path.join(RUN, 'prime_check.jsonl')
def PL(o): print('PRIME', _js5.dumps(o), flush=True); open(OUT, 'a').write(_js5.dumps(o) + '\n')
tkd = eng.f_tk; assert tkd is not None
TK = tkd.TK; dev = eng.dev; Bp = TK.Bp
pre = eng.prefix_ids.tolist(); KW = tkd.KW
tkd.refresh()
items = DEV['dev'] + DEV['dev2']
g = torch.Generator(device='cpu').manual_seed(0)
dig = [tok(str(d), add_special_tokens=False)['input_ids'][0] for d in range(10)]
ops = [tok(x, add_special_tokens=False)['input_ids'][0] for x in [' +', ' -', ' *', ' /', ' =', '=', ' ', '.', ',', '\n', ' $', '-', '+', ' x']]
pool_ = torch.tensor(dig * 6 + ops * 3)


def tails_for(kind, U, Gr, off):
    if kind == 'real':
        t = [(pre + C.q_ids(tok, items[(off + i) % len(items)]['q']))[-KW:] for i in range(U)]
        t = torch.tensor(t)
    else:
        t = pool_[torch.randint(0, len(pool_), (U, KW), generator=g)]
        m = torch.rand(U, KW, generator=g) < 0.15
        t = torch.where(m, torch.randint(0, 150000, (U, KW), generator=g), t)
    return t.repeat_interleave(Gr, 0).to(dev)


def snap(rows):
    return [x.view(-1, Bp)[:, rows].clone() for x in TK._state_ids]


def junk(rows):
    for x in TK._state_ids:
        v = x.view(-1, Bp); v[:, rows] = torch.randint(0, 7, v[:, rows].shape, device=dev).to(v.dtype)


def run_steps(rows, toks):
    outs = []
    for t in range(toks.shape[1]):
        tb = torch.zeros(eng.B_max, dtype=torch.long, device=dev); tb[rows] = toks[:, t]
        outs.append(tkd.step(None, eng.B_max, tb)[rows].clone())
    return torch.stack(outs)


res = []
for kind, U, Gr in [('real', 48, 8), ('real', 3, 8), ('real', 24, 16), ('fuzz', 48, 8), ('fuzz', 7, 3), ('real', 1, 8)]:
    for rep in range(2):
        tails = tails_for(kind, U, Gr, 50 * rep + U)
        N = tails.shape[0]
        rows = torch.randperm(eng.B_max, device=dev)[:N]
        junk(rows); torch.cuda.synchronize(); t0 = _tm5.time()
        tkd.prime_ids_seq(tails, rows); torch.cuda.synchronize(); ts = _tm5.time() - t0
        A = snap(rows)
        stream = tails[:, torch.randint(0, KW, (64,), generator=g).to(dev)]
        oa = run_steps(rows, stream); A2 = snap(rows)
        junk(rows); torch.cuda.synchronize(); t0 = _tm5.time()
        tkd.prime_ids_par(tails, rows); torch.cuda.synchronize(); tp = _tm5.time() - t0
        B_ = snap(rows)
        ob = run_steps(rows, stream); B2 = snap(rows)
        neq = [i for i, (a, b) in enumerate(zip(A, B_)) if not torch.equal(a, b)]
        neq2 = [i for i, (a, b) in enumerate(zip(A2, B2)) if not torch.equal(a, b)]
        r = dict(kind=kind, U=U, G=Gr, rows=N, state_tensors=len(A), state_mismatch=neq, after64_mismatch=neq2,
                 out_equal=bool(torch.equal(oa, ob)), firing=int((oa != 0).any(-1).sum()), seq_ms=round(1e3 * ts, 1),
                 par_ms=round(1e3 * tp, 1), flag=int(TK.flag.item()))
        PL(r); res.append(r)
# prime_emit (engine prefill fast path): output at the last tail token vs the torch M.emit, target-row state vs prime_ids_seq;
# some prompts get no target rows (exact prefill computes e for the whole group, admits only a few prompts)
dual = eng.f_dual; E_ = eng.W['E']
for kind, U, Gr, reps in [('real', 24, 8, 2), ('real', 48, 4, 1), ('fuzz', 64, 2, 6), ('fuzz', 40, 3, 4), ('fuzz', 5, 8, 2)]:
    for rep in range(reps):
        tails = tails_for(kind, U, Gr, 100 * rep + U + 7)
        tp = tails.view(U, Gr, KW)[:, 0]
        keep = torch.arange(U, device=dev) % 3 != 1                           # prompts 1, 4, ... : no target rows
        src = torch.arange(U, device=dev).repeat_interleave(Gr)[keep.repeat_interleave(Gr)]
        rows = torch.randperm(eng.B_max, device=dev)[:src.numel()]
        junk(rows); tkd.prime_ids_seq(tp[src], rows); A = snap(rows)
        e_ref = dual.M.emit(E_[tp])[0][:, -1].float()
        junk(rows); torch.cuda.synchronize(); t0 = _tm5.time()
        e_tk = tkd.prime_emit(tp, src, rows); torch.cuda.synchronize(); te = _tm5.time() - t0
        B_ = snap(rows)
        neq = [i for i, (a, b) in enumerate(zip(A, B_)) if not torch.equal(a, b)]
        r = dict(tag='pemit', kind=kind, U=U, G=Gr, rows=int(rows.numel()), state_mismatch=neq,
                 e_equal=bool(torch.equal(e_ref, e_tk)), e_rows_diff=int((e_ref != e_tk).any(-1).sum()),
                 e_firing=int((e_ref != 0).any(-1).sum()), ms=round(1e3 * te, 1), flag=int(TK.flag.item()))
        PL(r); res.append(dict(r, after64_mismatch=[], out_equal=r['e_equal']))
# firing tails: cut real rollout streams (prefix + question + generated) at random positions, keep tails whose last token
# makes the implant write (torch M.emit) plus some that do not
o_ = rollout([C.q_ids(tok, r['q']) for r in DEV['dev'][:24]], Gr=8, seed=11)
strm = [pre + C.q_ids(tok, DEV['dev'][j // 8]['q']) + list(t) for j, t in enumerate(o_['tokens'])]
cand = []
for j, s_ in enumerate(strm):
    L0 = len(pre) + len(C.q_ids(tok, DEV['dev'][j // 8]['q']))
    for c in torch.randint(L0, max(L0 + 1, len(s_)), (24,), generator=g).tolist(): cand.append(s_[c + 1 - KW:c + 1])
cand = torch.tensor(cand, device=dev)
fire = torch.cat([(dual.M.emit(E_[cand[i:i + 256]])[0][:, -1] != 0).any(-1) for i in range(0, len(cand), 256)])
sel_ = torch.cat([fire.nonzero()[:, 0][:120], (~fire).nonzero()[:, 0][:40]])
PL(dict(tag='firing_tails', candidates=len(cand), firing=int(fire.sum()), used=int(sel_.numel())))
for rep in range(2):
    tp = cand[sel_[torch.randperm(sel_.numel(), generator=g).to(dev)]]
    U = tp.shape[0]; src = torch.arange(U, device=dev).repeat_interleave(2)[: eng.B_max]
    rows = torch.randperm(eng.B_max, device=dev)[:src.numel()]
    junk(rows); tkd.prime_ids_seq(tp[src], rows); A = snap(rows)
    e_ref = dual.M.emit(E_[tp])[0][:, -1].float()
    junk(rows); e_tk = tkd.prime_emit(tp, src, rows); B_ = snap(rows)
    neq = [i for i, (a, b) in enumerate(zip(A, B_)) if not torch.equal(a, b)]
    r = dict(tag='pemit_firing', U=U, rows=int(rows.numel()), state_mismatch=neq, e_equal=bool(torch.equal(e_ref, e_tk)),
             e_rows_diff=int((e_ref != e_tk).any(-1).sum()), e_firing=int((e_ref != 0).any(-1).sum()), flag=int(TK.flag.item()))
    PL(r); res.append(dict(r, after64_mismatch=[], out_equal=r['e_equal']))
ok = all(not r['state_mismatch'] and not r['after64_mismatch'] and r['out_equal'] for r in res)
PL(dict(tag='summary', ok=ok, cases=len(res)))
print('PRIMEDONE', flush=True)
