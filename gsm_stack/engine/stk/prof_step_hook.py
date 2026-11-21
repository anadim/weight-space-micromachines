# Per-decode-step profile of the serial stack engine (exec'd by tw_dump_lean.py like rl_hook.py; RL_RUN = scratch dir).
# PROF_B (bucket, default B_max), PROF_CTX (comma list of slot lengths).  Prints ms/step (graph replay) for:
#   full | no implants (calc_s4 + chain2 off) | host only (implants off, side circuit write dropped is NOT possible ->
#   reported via eager profiler categories), plus an eager torch.profiler kernel table per category.
import os as _o3
_src = open(_o3.path.join(_o3.environ['STK_DIR'], 'rl_hook.py')).read()
_src = _src[:_src.index('\ndef fastcheck')]
exec(_src)
import json as _js3, time as _tm3
from torch.profiler import profile, ProfilerActivity
B = int(_o3.environ.get('PROF_B', eng.B_max)); CTX = [int(x) for x in _o3.environ.get('PROF_CTX', '256,768,1536').split(',')]
OUT = _o3.path.join(RUN, 'prof.jsonl')
def PL(o): print('PROF', _js3.dumps(o), flush=True); open(OUT, 'a').write(_js3.dumps(o) + '\n')
# warm state: one short real rollout so every buffer exists, then hand-set a live state
qd = [C.q_ids(tok, r['q']) for r in DEV['dev'][:max(1, B // G)]]
o = rollout(qd, Gr=G, seed=1)
PL(dict(tag='warm_rollout', **o['stats']))


def set_state(ctx):
    eng.done[:B] = False; eng.slot[:B] = ctx; eng.glen[:B] = 0; eng.tokb[:B] = 220
    if hasattr(eng, 's_fin'): eng.s_fin[:B] = False
    SEG = int(_o3.environ.get('PROF_SEG', 0))          # >0: rows sit SEG tokens into a later segment (forget span before)
    if SEG and hasattr(eng, 's_qlen'):
        eng.s_qlen[:B] = 150; eng.s_fr[:B] = max(151, ctx - SEG) if ctx - SEG > 150 else -1
        eng.s_nf[:B] = 3 if ctx - SEG > 150 else 0
    eng.maxnew.fill_(10 ** 6)


def time_graph(ctx, n=30):
    set_state(ctx); g = eng._graph(B, eng.S_max)
    for _ in range(3): g.replay()
    set_state(ctx); torch.cuda.synchronize(); t0 = _tm3.time()
    for _ in range(n): g.replay()
    torch.cuda.synchronize(); return 1e3 * (_tm3.time() - t0) / n


for ctx in CTX:
    eng.imp_on = True; full = time_graph(ctx)
    eng.imp_on = False; noimp = time_graph(ctx)
    eng.imp_on = True
    PL(dict(tag='step_ms', B=B, ctx=ctx, full=full, no_implants=noimp, tok_s_full=B / full * 1e3))

# eager kernel breakdown at the middle context
ctx = CTX[len(CTX) // 2]
set_state(ctx)
with torch.no_grad():
    for _ in range(2): eng._step(B, eng.S_max)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): eng._step(B, eng.S_max)
        torch.cuda.synchronize()
rows = []
for ev in prof.key_averages():
    if ev.device_type is not None and 'CUDA' in str(ev.device_type) and ev.self_device_time_total > 0:
        rows.append((ev.self_device_time_total / 5 / 1e3, ev.count // 5, ev.key[:110]))
rows.sort(reverse=True)
cat = {}
def catof(k):
    kl = k.lower()
    if 'attn_suffix' in kl: return 'attn_suffix(forget bias)'
    if 'attn_prefix' in kl: return 'attn_prefix'
    if 'sample' in kl: return 'sample'
    if 'gemm' in kl or 'sm90' in kl or 'sm80' in kl or 'cutlass' in kl or 'ampere' in kl or 'splitk' in kl or 'gemv' in kl: return 'gemm'
    if 'csr' in kl or 'sparse' in kl or 'cusparse' in kl: return 'sparse(implants)'
    if 'rms' in kl or 'rope' in kl or 'silu' in kl: return 'host_elementwise'
    return 'other(torch elementwise: implants/_sw/book)'
for ms, c, k in rows: cat[catof(k)] = cat.get(catof(k), 0) + ms
PL(dict(tag='eager_kernel_ms_per_step_by_cat', B=B, ctx=ctx, total=sum(cat.values()), cats=cat))
PL(dict(tag='eager_top_kernels', B=B, ctx=ctx, top=[(round(a, 3), b, k) for a, b, k in rows[:40]]))
print('PROFDONE', flush=True)
