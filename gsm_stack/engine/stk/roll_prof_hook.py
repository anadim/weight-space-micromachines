# Kernel profile of REAL rollouts (exec'd by tw_dump_lean.py like rl_hook.py; RL_RUN = scratch dir).
# One warm rollout of the arm batch under CUPTI; each graph replay runs in a record_function range; kernels map to it by CUPTI correlation id, so
# every kernel is attributed to its decode step and bucket. Per bucket: steps, wall ms/step, main-stream busy / gap ms per
# step, side-stream busy ms per step, and the top kernels (us per step, launches per step).
import os as _o6
_src = open(_o6.path.join(_o6.environ['STK_DIR'], 'rl_hook.py')).read()
_src = _src[:_src.index('\ndef fastcheck')]
exec(_src)
import json as _js6, time as _tm6, random as _rr6, bisect
from torch.profiler import profile, ProfilerActivity, record_function
OUT = _o6.path.join(RUN, 'rprof.jsonl')
def PL(o): print('RPROF', _js6.dumps(o), flush=True); open(OUT, 'a').write(_js6.dumps(o) + '\n')
BP = int(_o6.environ.get('BENCH_PROMPTS', PROMPTS))
batch = _rr6.Random(123).sample(pool, BP); qids = [C.q_ids(tok, r['q']) for r in batch]
rollout(qids, Gr=G, seed=1)                                          # graphs captured, everything warm
marks = []
_orig = eng._graph


class _W:
    def __init__(self, g, B): self.g, self.B = g, B
    def replay(self):
        marks.append((_tm6.monotonic_ns(), self.B))
        with record_function('RSTEP%d_%d' % (len(marks) - 1, self.B)): self.g.replay()


eng._graph = lambda B, S: _W(_orig(B, S), B)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    o = rollout(qids, Gr=G, seed=2)
    torch.cuda.synchronize()
eng._graph = _orig
t_end = _tm6.monotonic_ns()
PL(dict(tag='rollout', **{k: v for k, v in o['stats'].items() if not isinstance(v, dict)}))
# kernel -> (correlation id) launching runtime call -> enclosing RSTEP range (all on the kineto clock)
allev = list(prof.profiler.kineto_results.events())
rng = sorted((e.start_ns(), e.start_ns() + e.duration_ns(), int(e.name()[5:].split('_')[0])) for e in allev if e.name().startswith('RSTEP'))
rs = [r[0] for r in rng]
corr = {}
for e in allev:
    if 'cpu' in str(e.device_type()).lower() and e.name().startswith('cuda'):
        corr[e.correlation_id()] = e.start_ns()
ev = []; nostep = 0
for e in allev:
    if 'cuda' not in str(e.device_type()).lower() or e.duration_ns() <= 0: continue
    t = corr.get(e.correlation_id()); i = -1
    if t is not None:
        j = bisect.bisect_right(rs, t) - 1
        if j >= 0 and t <= rng[j][1]: i = rng[j][2]
    if i < 0: nostep += 1
    ev.append((i, e.start_ns(), e.duration_ns(), e.device_resource_id(), e.name()))
ev.sort(key=lambda x: (x[0], x[1]))
ts = [m[0] for m in marks] + [t_end]
PL(dict(tag='events', n=len(ev), steps=len(marks), ranges=len(rng), outside_steps=nostep))
st_busy = {}
for i, s, d, st, n in ev: st_busy[st] = st_busy.get(st, 0) + d
main = max(st_busy, key=st_busy.get)
agg = {}                                    # bucket -> stats
for i, (t0, B) in enumerate(marks):
    a = agg.setdefault(B, dict(steps=0, wall=0, main=0, side=0, gap=0, k={}))
    a['steps'] += 1; a['wall'] += ts[i + 1] - t0
lastend = {}
for i, s, d, st, n in ev:
    if i < 0 or i >= len(marks): continue                          # admission / prefill / TK outside a replay are skipped
    B = marks[i][1]; a = agg[B]
    if st == main:
        a['main'] += d
        le = lastend.get(i)
        if le is not None and s > le: a['gap'] += s - le
        lastend[i] = max(le or 0, s + d)
    else: a['side'] += d
    kk = (n[:80], 'main' if st == main else 'side'); x = a['k'].setdefault(kk, [0, 0]); x[0] += d; x[1] += 1
for B in sorted(agg, reverse=True):
    a = agg[B]; ns = a['steps']
    top = sorted(a['k'].items(), key=lambda kv: -kv[1][0])[:int(_o6.environ.get('RP_TOP', 25))]
    PL(dict(tag='bucket', B=B, steps=ns, wall_ms=round(a['wall'] / ns / 1e6, 3), main_busy_ms=round(a['main'] / ns / 1e6, 3),
            main_gap_ms=round(a['gap'] / ns / 1e6, 3), side_busy_ms=round(a['side'] / ns / 1e6, 3),
            kernels_per_step=round(sum(v[1] for v in a['k'].values()) / ns, 1),
            top=[(k[0], k[1], round(v[0] / ns / 1e3, 1), round(v[1] / ns, 2)) for k, v in top]))
print('RPROFDONE', flush=True)
