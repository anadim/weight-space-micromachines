# Implant exactness fuzz + timing: torch DualImplant (calc_s4 + chain2 CSR stacks) vs TKDual (engine2 tools kernel) on real
# token streams (sampled rollouts of the current host + random-token fuzz), teacher-forced step by step.
# exec'd by tw_dump_lean.py like rl_hook.py, with RL_FASTROLL=1 (RL_RUN = scratch dir).  CHK_ROWS rows, CHK_STEPS steps.
import os as _o4
_src = open(_o4.path.join(_o4.environ['STK_DIR'], 'rl_hook.py')).read()
_src = _src[:_src.index('\ndef fastcheck')]
exec(_src)
import json as _js4, time as _tm4, random as _rr4
OUT = _o4.path.join(RUN, 'tk_check.jsonl')
def CL(o): print('TKCHK', _js4.dumps(o), flush=True); open(OUT, 'a').write(_js4.dumps(o) + '\n')
dual, tkd = eng.f_dual, eng.f_tk
assert tkd is not None
NR = int(_o4.environ.get('CHK_ROWS', min(eng.B_max, 256))); NS_ = int(_o4.environ.get('CHK_STEPS', 600))
NR = min(NR, eng.B_max)
items = DEV['dev'][:max(1, NR // G)]
o = rollout([C.q_ids(tok, r['q']) for r in items], Gr=G, seed=7)
streams = [C.q_ids(tok, items[j // G]['q']) for j in range(len(o['tokens']))]
pre = eng.prefix_ids.tolist(); KW = dual.K + dual.W
dev = eng.dev; E = eng.W['E']


def run(kind, rows_tok, tails):
    """rows_tok: (N, T) token ids fed one per step after priming with tails (N, KW). Returns list of e (N, n) per step."""
    N = rows_tok.shape[0]; rows = torch.arange(N, device=dev)
    outs = []
    torch.cuda.synchronize(); t0 = _tm4.time()
    if kind == 'torch':
        dual.prime_rows(E[tails], rows)
        for t in range(rows_tok.shape[1]):
            tk_ = rows_tok[:, t]; outs.append(dual.step(E[tk_], N, tk_).clone())
    else:
        tkd.refresh(); tkd.prime_ids(tails, rows)
        for t in range(rows_tok.shape[1]):
            tk_ = rows_tok[:, t]; outs.append(tkd.step(None, N, tk_).clone())
    torch.cuda.synchronize()
    return outs, _tm4.time() - t0


def compare(tag, rows_tok, tails):
    a, ta = run('torch', rows_tok, tails); b, tb = run('tk', rows_tok, tails)
    A = torch.stack(a); Bt = torch.stack(b)
    mism = int((A != Bt).any(-1).sum()); nz = int((A != 0).any(-1).sum())
    CL(dict(tag=tag, rows=rows_tok.shape[0], steps=rows_tok.shape[1], row_steps=int(A.shape[0] * A.shape[1]),
            firing_row_steps=nz, mismatching_row_steps=mism, max_abs_diff=float((A - Bt).abs().max()),
            torch_ms_per_step=1e3 * ta / rows_tok.shape[1], tk_ms_per_step=1e3 * tb / rows_tok.shape[1],
            tk_flag=int(tkd.TK.flag.item())))


# 1) real streams: tail of prefix+question, then the generated tokens (padded with eos)
T = min(NS_, max(map(len, o['tokens'])))
rt = torch.full((len(streams), T), eng.eos, dtype=torch.long, device=dev)
for j, t in enumerate(o['tokens']): rt[j, :min(T, len(t))] = torch.tensor(t[:T], device=dev)
tails = torch.tensor([(pre + q)[-KW:] for q in streams], device=dev)
compare('rollout_streams', rt, tails)
# 2) fuzz: random mixes of digits / operators / '=' / spaces / random vocab
g = torch.Generator(device='cpu').manual_seed(0)
dig = [tok(str(d), add_special_tokens=False)['input_ids'][0] for d in range(10)]
ops = [tok(x, add_special_tokens=False)['input_ids'][0] for x in [' +', ' -', ' *', ' /', ' =', '=', ' ', '.', ',', '\n', ' $', '-', '+']]
pool_ = torch.tensor(dig * 6 + ops * 3, device=dev)
for rep in range(int(_o4.environ.get('CHK_FUZZ', 3))):
    N = NR; Tf = NS_
    r = torch.randint(0, len(pool_), (N, Tf), generator=g).to(dev); ft = pool_[r]
    rnd = torch.randint(0, len(tok), (N, Tf), generator=g).to(dev); m = torch.rand(N, Tf, generator=g).to(dev) < 0.15
    ft = torch.where(m, rnd, ft)
    ftails = pool_[torch.randint(0, len(pool_), (N, KW), generator=g).to(dev)]
    compare(f'fuzz{rep}', ft, ftails)
# 3) prime cost
for N in (8, 64, min(256, eng.B_max)):
    tails_ = tails[:1].expand(N, -1).contiguous(); rows = torch.arange(N, device=dev)
    torch.cuda.synchronize(); t0 = _tm4.time(); tkd.prime_ids(tails_, rows); torch.cuda.synchronize()
    t1 = _tm4.time(); dual.prime_rows(E[tails_], rows); torch.cuda.synchronize()
    CL(dict(tag='prime_ms', N=N, KW=KW, tk=1e3 * (t1 - t0), torch=1e3 * (_tm4.time() - t1)))
print('TKCHKDONE', flush=True)
