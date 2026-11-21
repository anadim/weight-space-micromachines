# Rollout time breakdown (exec'd by tw_dump_lean.py like rl_hook.py; RL_RUN = scratch dir).
# One cold rollout (graph capture), then BR_REPS warm rollouts of the arm's batch (64 prompts x G) with FAST_TIME=1 (admission
# and compaction synced and timed), then one warm rollout under cProfile (Python-side cost: admission / prefill / priming /
# harvest).  Reports total, decode (sum of bucket seconds), admit, compact, the rest, and the top cProfile entries.
import os as _o4
_src = open(_o4.path.join(_o4.environ['STK_DIR'], 'rl_hook.py')).read()
_src = _src[:_src.index('\ndef fastcheck')]
exec(_src)
import json as _js4, time as _tm4, random as _rr4, cProfile, pstats, io
import fastroll
OUT = _o4.path.join(RUN, 'rbreak.jsonl')
def PL(o): print('RBREAK', _js4.dumps(o), flush=True); open(OUT, 'a').write(_js4.dumps(o) + '\n')
BP = int(_o4.environ.get('BENCH_PROMPTS', PROMPTS))
batch = _rr4.Random(123).sample(pool, BP); qids = [C.q_ids(tok, r['q']) for r in batch]


def rec(tag, o):
    s = o['stats']; dec = sum(s['bucket_s'].values())
    PL(dict(tag=tag, seconds=round(s['seconds'], 3), tokens=s['tokens'], tok_s=round(s['tok_s']), decode_s=round(dec, 3),
            admit_s=s.get('admit_s'), admits=s.get('admits'), compact_s=s.get('compact_s'), compactions=s.get('compactions'),
            other_s=round(s['seconds'] - dec - (s.get('admit_s') or 0) - (s.get('compact_s') or 0), 3),
            steps=s['decode_steps'], occ=round(s['occupancy'], 3), bucket_occ=round(s['bucket_occ'], 3),
            buckets={k: (s['bucket_steps'][k], s['bucket_s'][k], round(1e3 * s['bucket_s'][k] / s['bucket_steps'][k], 2))
                     for k in sorted(s['bucket_s'], key=int, reverse=True)}))


rec('cold', rollout(qids, Gr=G, seed=1))
for k in range(int(_o4.environ.get('BR_REPS', 2))):
    rec(f'warm{k}', rollout(qids, Gr=G, seed=2 + k))
eng.f_time = False
rec('warm_notime', rollout(qids, Gr=G, seed=7))
pr = cProfile.Profile(); pr.enable()
o = rollout(qids, Gr=G, seed=8)
pr.disable(); rec('warm_cprofile', o)
sio = io.StringIO(); pstats.Stats(pr, stream=sio).sort_stats('cumulative').print_stats(45)
print(sio.getvalue()[:12000])
sio = io.StringIO(); pstats.Stats(pr, stream=sio).sort_stats('tottime').print_stats(30)
print(sio.getvalue()[:8000])
if _o4.environ.get('BR_SECT', '1') == '1':
    # synced section timing inside admission prefill (GPU time hidden behind the bool-mask sync otherwise)
    import collections, engine as _E4
    acc = collections.defaultdict(float); cnt = collections.Counter()
    def wrap(obj, name, tag):
        f = getattr(obj, name)
        def g(*a, **k):
            torch.cuda.synchronize(); t = _tm4.perf_counter(); r = f(*a, **k); torch.cuda.synchronize()
            acc[tag] += _tm4.perf_counter() - t; cnt[tag] += 1; return r
        setattr(obj, name, g)
    wrap(_E4.Q, 'fwd_chunk', 'fwd_chunk')
    imp = eng.imp
    for nm in ('emit_ids', 'full', 'prime_rows', 'prime_rows_ids'):
        if hasattr(imp, nm): wrap(imp, nm, 'imp.' + nm)
    if hasattr(imp, 'M') and hasattr(imp.M, 'emit'): wrap(imp.M, 'emit', 'imp.M.emit')
    wrap(_E4.Engine, '_prefill', 'prefill_total'); wrap(type(eng), '_f_admit', 'admit_total')
    o = rollout(qids, Gr=G, seed=9); rec('warm_sect', o)
    PL(dict(tag='sections', **{k: (cnt[k], round(v, 3), round(1e3 * v / cnt[k], 2)) for k, v in sorted(acc.items())}))
print('RBREAKDONE', flush=True)
