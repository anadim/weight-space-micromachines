# Bench / equivalence check of fastroll vs the old wave loop.  exec'd by tw_dump_lean.py like rl_hook.py
# (RL_RUN must be a scratch dir).  BENCH_STEPS: comma list of  old | wave | fast | devgreedy | ratio
import os as _o2
_src = open(_o2.path.join(_o2.environ['STK_DIR'], 'rl_hook.py')).read()
_src = _src[:_src.index('\nSTART = 1')]
exec(_src)
import fastroll, json as _js, time as _tm, random as _rr
OUTF = _o2.path.join(RUN, 'bench.jsonl')
def BL(o): print('BENCH', _js.dumps(o), flush=True); open(OUTF, 'a').write(_js.dumps(o) + '\n')
steps = _o2.environ.get('BENCH_STEPS', 'old,wave,fast,ratio').split(',')
BP = int(_o2.environ.get('BENCH_PROMPTS', PROMPTS))
batch = _rr.Random(123).sample(pool, BP); qids = [C.q_ids(tok, r['q']) for r in batch]
SEED_ = 10001


def summ(tag, out, extra=None):
    st = dict(out['stats']); st['tag'] = tag
    st['len_max'] = max(map(len, out['tokens'])); st['len_mean'] = sum(map(len, out['tokens'])) / len(out['tokens'])
    if 'reg' in out: st.update(eng.sw_roll_stats(out))
    Rf = [float(C.correct(last_answer(t), batch[j // G]['gold'])) for j, t in enumerate(out['tokens'])]
    st['reward'] = sum(Rf) / len(Rf)
    if extra: st.update(extra)
    BL(st)


def same(a, b):
    n = len(a['tokens']); eq = sum(x == y for x, y in zip(a['tokens'], b['tokens']))
    eqw = sum(x == y for x, y in zip(a['wflag'], b['wflag']))
    dl = max((max((abs(u - v) for u, v in zip(x, y)), default=0.0) for x, y, t1, t2 in
              zip(a['logp'], b['logp'], a['tokens'], b['tokens']) if t1 == t2), default=0.0)
    r = dict(rows=n, tok_identical=eq, wflag_identical=eqw, max_abs_dlogp_on_identical=dl)
    if 'reg' in a and 'reg' in b:
        r['reg_identical'] = {k: bool(torch.equal(a['reg'][k], b['reg'][k])) for k in a['reg']}
    return r


res = {}
if 'old' in steps:
    torch.cuda.reset_peak_memory_stats()
    res['old'] = rollout(qids, Gr=G, seed=SEED_); summ('old_sampled', res['old'], dict(mem_gb=torch.cuda.max_memory_allocated() / 1e9))
fastroll.enable(eng)
if 'wave' in steps:
    eng.fast_config(K=1, wave=True)
    res['wave'] = rollout(qids, Gr=G, seed=SEED_); summ('fast_wavemode_sampled', res['wave'])
    if 'old' in res: BL(dict(tag='old_vs_wavemode_sampled', **same(res['old'], res['wave'])))
eng.fast_config(K=int(_o2.environ.get('FAST_K', 8)), admit=int(_o2.environ.get('FAST_ADMIT', 0)) or None, wave=False)
if 'fast' in steps:
    torch.cuda.reset_peak_memory_stats()
    res['fast'] = rollout(qids, Gr=G, seed=SEED_); summ('fast_sampled', res['fast'], dict(mem_gb=torch.cuda.max_memory_allocated() / 1e9))
    res['fast2'] = rollout(qids, Gr=G, seed=SEED_ + 1); summ('fast_sampled_2', res['fast2'])
    _reps = int(_o2.environ.get('BENCH_REPS', 1))
    if _reps > 1:                      # more warm runs (other seeds): steady-state throughput = mean over warm runs
        ws = [res['fast2']['stats']]
        for k_ in range(2, _reps + 1):
            o_ = rollout(qids, Gr=G, seed=SEED_ + k_); summ(f'fast_sampled_{k_ + 1}', o_); ws.append(o_['stats'])
        BL(dict(tag='fast_warm_mean', runs=len(ws), tok_s=sum(s['tokens'] for s in ws) / sum(s['seconds'] for s in ws),
                s_per_run=sum(s['seconds'] for s in ws) / len(ws), tokens_per_run=sum(s['tokens'] for s in ws) / len(ws)))
if 'devgreedy' in steps:
    DN = int(_o2.environ.get('BENCH_DEVN', 600))
    items = DEV['dev'][:DN]; qd = [C.q_ids(tok, r['q']) for r in items]
    N_ = int(_o2.environ.get('RL_N', 4))
    eng.fast_config(K=int(_o2.environ.get('FAST_K', 8)), wave=False)
    eng.__class__.__bases__[0]  # (old path = the wrapped class)
    t0 = _tm.time(); og = rollout(qd, greedy=True); t_f = _tm.time() - t0
    cls_fast = eng.__class__; eng.__class__ = cls_fast.__bases__[0]; eng.graphs = {}; eng.pool = None; eng._outshape = None
    eng.c2_on = False
    t0 = _tm.time(); oo = rollout(qd, greedy=True); t_o = _tm.time() - t0
    eng.__class__ = cls_fast; eng.graphs = {}; eng.pool = None; eng._outshape = None
    rec = dict(tag='devgreedy_old_vs_fast', n=DN, t_fast=t_f, t_old=t_o, fast_tok_s=og['stats']['tok_s'], old_tok_s=oo['stats']['tok_s'],
               **same(oo, og))
    acc = lambda o: 100 * sum(float(C.correct(last_answer(t), r['gold'])) for t, r in zip(o['tokens'], items)) / DN
    rec['acc_old'] = acc(oo); rec['acc_fast'] = acc(og)
    # divergence analysis: first differing position, old-path logprob of both tokens there is not available -> report pos
    div = []
    for j, (x, y) in enumerate(zip(oo['tokens'], og['tokens'])):
        if x != y:
            k = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
            div.append(dict(row=j, pos=k, lp_old=oo['logp'][j][k] if k < len(x) else None, lp_fast=og['logp'][j][k] if k < len(y) else None,
                            tok_old=x[k] if k < len(x) else None, tok_fast=y[k] if k < len(y) else None))
    rec['divergences'] = div[:60]; rec['n_div'] = len(div)
    BL(rec)
    import pickle; pickle.dump(dict(old=oo, fast=og, items=[r['q'] for r in items]), open(_o2.path.join(RUN, 'devgreedy.pkl'), 'wb'))
if 'fastgreedy' in steps:          # fast path only, dev greedy tokens dumped (compare offline across configs)
    DN = int(_o2.environ.get('BENCH_DEVN', 600))
    items = DEV['dev'][:DN]; qd = [C.q_ids(tok, r['q']) for r in items]
    t0 = _tm.time(); og = rollout(qd, greedy=True); t_f = _tm.time() - t0
    acc = 100 * sum(float(C.correct(last_answer(t), r['gold'])) for t, r in zip(og['tokens'], items)) / DN
    BL(dict(tag='fastgreedy', n=DN, t_fast=t_f, fast_tok_s=og['stats']['tok_s'], acc_fast=acc))
    import pickle; pickle.dump(dict(fast=dict(tokens=og['tokens'], logp=og['logp'], wflag=og['wflag'])),
                               open(_o2.path.join(RUN, 'fastgreedy.pkl'), 'wb'))
if 'ratio' in steps:
    o = res.get('fast') or rollout(qids, Gr=G, seed=SEED_)
    Rf = [float(C.correct(last_answer(t), batch[j // G]['gold'])) for j, t in enumerate(o['tokens'])]
    Rt = torch.tensor(Rf).view(BP, G); advt = (Rt - Rt.mean(1, keepdim=True)).view(-1).tolist()
    docs = build_docs(qids, o, advt, None)
    st = train(docs, len(o['tokens']), ACAP * NATT if ARM != 'B' else ACAP)
    BL(dict(tag='teacher_forcing_first_update_fast', **{k: v for k, v in st.items()}))
print('BENCHDONE', flush=True)
