# Graph-replay decode-step profile per bucket (exec'd by tw_dump_lean.py like rl_hook.py; RL_RUN = scratch dir).
# PROF_BS (comma list of buckets), PROF_CTX (slot length), PROF_SEG (rows SEG tokens into a later segment).
# Per bucket: ms/step of graph replay, and a CUPTI kernel timeline of 10 replays: kernel-busy ms per stream, idle gaps on the
# main stream, kernels per step, top kernels.
import os as _o3
_src = open(_o3.path.join(_o3.environ['STK_DIR'], 'rl_hook.py')).read()
_src = _src[:_src.index('\ndef fastcheck')]
exec(_src)
import json as _js3, time as _tm3
from torch.profiler import profile, ProfilerActivity
BS = [int(x) for x in _o3.environ.get('PROF_BS', '384,192,128,64,32,16,8').split(',') if int(x) <= eng.B_max]
CTX = int(_o3.environ.get('PROF_CTX', 768)); SEG = int(_o3.environ.get('PROF_SEG', 170))
OUT = _o3.path.join(RUN, 'profg.jsonl')
def PL(o): print('PROF', _js3.dumps(o), flush=True); open(OUT, 'a').write(_js3.dumps(o) + '\n')
qd = [C.q_ids(tok, r['q']) for r in DEV['dev'][:max(1, eng.B_max // G)]]
o = rollout(qd, Gr=G, seed=1)
PL(dict(tag='warm_rollout', **{k: v for k, v in o['stats'].items()}))


def set_state(B, ctx):
    eng.done[:B] = False; eng.slot[:B] = ctx; eng.glen[:B] = 0; eng.tokb[:B] = 220
    if hasattr(eng, 's_fin'): eng.s_fin[:B] = False
    if SEG and hasattr(eng, 's_qlen'):
        eng.s_qlen[:B] = 150; eng.s_fr[:B] = max(151, ctx - SEG) if ctx - SEG > 150 else -1
        eng.s_nf[:B] = 3 if ctx - SEG > 150 else 0
    eng.maxnew.fill_(10 ** 6)


NR = 10
for B in BS:
    set_state(B, CTX); g = eng._graph(B, eng.S_max)
    for _ in range(3): g.replay()
    set_state(B, CTX); torch.cuda.synchronize(); t0 = _tm3.time()
    for _ in range(30): g.replay()
    torch.cuda.synchronize(); ms = 1e3 * (_tm3.time() - t0) / 30
    set_state(B, CTX); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(NR): g.replay()
        torch.cuda.synchronize()
    ev = []
    for e in prof.profiler.kineto_results.events():
        if 'cuda' in str(e.device_type()).lower() and e.duration_ns() > 0 and 'memcpy' not in e.name().lower() and 'memset' not in e.name().lower():
            ev.append((e.start_ns(), e.duration_ns(), e.device_resource_id(), e.name()))
    if not ev:
        PL(dict(tag='graph_step', B=B, ctx=CTX, ms=ms, note='no CUPTI kernel records')); continue
    ev.sort()
    t_lo = ev[0][0]; t_hi = max(s + d for s, d, _, _ in ev)
    per_stream = {}
    for s, d, st, n in ev: per_stream[st] = per_stream.get(st, 0) + d
    main = max(per_stream, key=per_stream.get)
    mev = [(s, d) for s, d, st, _ in ev if st == main]
    gaps = 0; gl = []; end = mev[0][0]
    for s, d in mev:
        if s > end: gaps += s - end; gl.append(s - end)
        end = max(end, s + d)
    by = {}
    for s, d, st, n in ev:
        k = (n[:90], st == main); a = by.setdefault(k, [0, 0]); a[0] += d; a[1] += 1
    top = sorted(by.items(), key=lambda kv: -kv[1][0])[:30]
    gl.sort()
    PL(dict(tag='graph_step', B=B, ctx=CTX, seg=SEG, ms=round(ms, 3), tok_s=round(B / ms * 1e3),
            prof_wall_ms=round((t_hi - t_lo) / 1e6 / NR, 3),
            kern_ms_by_stream={str(k): round(v / 1e6 / NR, 3) for k, v in per_stream.items()},
            main_stream=str(main), main_gap_ms=round(gaps / 1e6 / NR, 3), main_kernels_per_step=len(mev) / NR,
            gap_us_p50=round(gl[len(gl) // 2] / 1e3, 2) if gl else 0, gap_us_p90=round(gl[int(len(gl) * .9)] / 1e3, 2) if gl else 0,
            kernels_per_step=len(ev) / NR,
            top=[(round(v[0] / 1e6 / NR, 4), v[1] // NR, 'M' if k[1] else 'S', k[0]) for k, v in top]))
print('PROFDONE', flush=True)
