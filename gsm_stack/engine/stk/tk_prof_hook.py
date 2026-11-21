# TK (tools kernel) timing on real token streams vs batch size, plus stack geometry (layers / units / nnz per tool).
# exec'd by tw_dump_lean.py like rl_hook.py with RL_FASTROLL=1 (RL_RUN = scratch dir).
import os as _o6
_src = open(_o6.path.join(_o6.environ['STK_DIR'], 'rl_hook.py')).read()
_src = _src[:_src.index('\ndef fastcheck')]
exec(_src)
import json as _js6
OUT = _o6.path.join(RUN, 'tkprof.jsonl')
def PL(o): print('TKPROF', _js6.dumps(o), flush=True); open(OUT, 'a').write(_js6.dumps(o) + '\n')
tkd = eng.f_tk; TK = tkd.TK; dev = eng.dev
PL(dict(tag='info', info=TK.info, Bp=TK.Bp))
items = DEV['dev'][:max(1, eng.B_max // G)]
o = rollout([C.q_ids(tok, r['q']) for r in items], Gr=G, seed=7)
pre = eng.prefix_ids.tolist(); KW = tkd.KW
streams = [C.q_ids(tok, items[j // G]['q']) for j in range(len(o['tokens']))]
T = 256
rt = torch.full((len(streams), T), eng.eos, dtype=torch.long, device=dev)
for j, t in enumerate(o['tokens']): rt[j, :min(T, len(t))] = torch.tensor(t[:T], device=dev)
tails = torch.tensor([(pre + q)[-KW:] for q in streams], device=dev)
for B in [int(x) for x in _o6.environ.get('TKP_BS', '1,8,32,64,128,192,384').split(',') if int(x) <= eng.B_max]:
    rows = torch.arange(B, device=dev)
    tkd.prime_ids(tails[:B], rows)
    tb = torch.zeros(eng.B_max, dtype=torch.long, device=dev)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): tkd.step(None, B, tb)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g): tkd.step(None, B, tb)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 * T)]
    for t in range(T):
        tb[:B] = rt[:B, t]
        ev[2 * t].record(); g.replay(); ev[2 * t + 1].record()
    torch.cuda.synchronize()
    ms = sorted(ev[2 * t].elapsed_time(ev[2 * t + 1]) for t in range(T))
    PL(dict(tag='step', B=B, mean_ms=round(sum(ms) / T, 4), p50=round(ms[T // 2], 4), p90=round(ms[int(T * .9)], 4), max=round(ms[-1], 4)))
print('TKPROFDONE', flush=True)
