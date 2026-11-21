# Trainer bench (exec'd by rl_hook at the first update when RL_TRAIN_BENCH=1; names in scope: docs, n, ARM, NATT, ACAP,
# train_grads, P32, names, opt, eng, a, sync_weights, PACK_SIZES). Real rollout batch; every variant's gradient is
# compared with the reference trainer's (rl_hook.train_grads) on the same weights. Prints BENCH / PROF lines, then exits.
import json as _bj, time as _bt, fasttrain as FT
_norm_len = ACAP * NATT if ARM != 'B' else ACAP


def _B(*x): print('BENCH', *x, flush=True)


def _snap():
    g = {k: P32[k].grad.detach().clone() for k in names}
    for k in names: P32[k].grad = None
    return g


def _run(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
    t0 = _bt.time(); st = fn(); torch.cuda.synchronize(); dt = _bt.time() - t0
    return st, dt, (torch.cuda.max_memory_allocated() - m0) / 1e9


def _cmp(G, R):
    num = sum(float((G[k].double() - R[k].double()).pow(2).sum()) for k in names)
    den = sum(float(R[k].double().pow(2).sum()) for k in names)
    worst = max(((float((G[k] - R[k]).double().norm() / max(R[k].double().norm(), 1e-30)), k) for k in names))
    return dict(rel=(num / den) ** 0.5, worst_rel=round(worst[0], 6), worst=worst[1].replace('model.', ''),
                bitwise=all(torch.equal(G[k], R[k]) for k in names))


def _prof(fn, tag, top=22):
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        fn(); torch.cuda.synchronize()
    for k in names: P32[k].grad = None
    ev = sorted(pr.key_averages(), key=lambda e: -e.device_time_total)
    tot = sum(e.self_device_time_total for e in ev)
    print(f'PROF {tag} total_kernel_ms {tot / 1e3:.0f}', flush=True)
    for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:top]:
        print(f'PROF {tag} {e.self_device_time_total / 1e3:9.1f} ms {100 * e.self_device_time_total / max(tot, 1):5.1f}% n={e.count:6d} {e.key[:110]}', flush=True)


_ntok_q = sum(len(d['q']) for d in docs); _ntok_r = sum(len(d['r']) for d in docs)
_B('batch', _bj.dumps(dict(arm=ARM, u=u, docs=len(docs), rows=n, q_tokens=_ntok_q, r_tokens=_ntok_r,
                            uniq_q=len({tuple(d['q']) for d in docs}), Lp=int(eng.prefix_ids.numel()),
                            mem_alloc_gb=round(torch.cuda.memory_allocated() / 1e9, 1))))
for k in names: P32[k].grad = None
_REF = None
_res = {}
_reps = int(_o.environ.get('TB_REPS', 2))
_PG = _o.environ.get('TB_PG', '0') == '1'
if _PG:      # PG surrogate: the reference = fasttrain's reference-equivalent path (bitwise = train_grads, bench j23172)
    FT.Trainer.PG = True
    _reftr = FT.Trainer(P32, names, eng, T=a.T, clip=a.clip, dedup=False, ckpt=True, fused_lp=False, kopts='small', sizes=list(PACK_SIZES))
    train_grads = lambda d_, n_, l_: _reftr.grads(d_, n_, l_)
    _B('PG mode: loss = -A*lp (no ratio/clip); ref = reference-equivalent fasttrain path')
for _r_ in range(_reps):
    st, dt, mem = _run(lambda: train_grads(docs, n, _norm_len))
    _B('ref', f'run{_r_}', f'{dt:.2f}s', f'peak+{mem:.1f}GB', _bj.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in st.items()}))
    G = _snap()
    if _REF is None: _REF = (G, st)
    else: _B('ref', 'repeat', _bj.dumps(_cmp(G, _REF[0])))
_res['ref'] = dt
if _o.environ.get('TB_PROF', '1') == '1': _prof(lambda: train_grads(docs, n, _norm_len), 'ref')

# high-precision truth: the reference math (same layout) with fp32 weights/activations, no TF32. Every bf16 trainer
# (the reference included) is scored by its gradient error vs this truth.
_TRUTH = None
if _o.environ.get('TB_TRUTH', '1') == '1':
    try:
        _tf = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        _tr = FT.Trainer(P32, names, eng, T=a.T, clip=a.clip, dedup=False, ckpt=True, fused_lp=False, kopts='small',
                         sizes=list(PACK_SIZES), dtype=torch.float32)
        st, dt, mem = _run(lambda: _tr.grads(docs, n, _norm_len)); _TRUTH = _snap(); del _tr
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = _tf
        _B('truth_fp32', f'{dt:.2f}s', f'peak+{mem:.1f}GB', _bj.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in st.items()}))
        _B('ref', 'vs_truth', _bj.dumps(_cmp(_REF[0], _TRUTH)))
    except Exception as _e:
        import traceback; traceback.print_exc(); _B('truth_fp32', 'FAILED', repr(_e)[:400])
        for k in names: P32[k].grad = None
    torch.cuda.empty_cache()

_B200 = dict(BLOCK_M=128, BLOCK_N=64, num_stages=2, num_warps=8, BLOCK_M1=32, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=32)
_VARIANTS = [
    ('nodedup_ckpt_small', dict(dedup=False, ckpt=True, fused_lp=False, kopts='small', sizes=list(PACK_SIZES))),
    # noise floor: the reference math with a different packing (8k budget) -> only bf16 summation order changes
    ('floor_nodedup_ckpt_small_8k', dict(dedup=False, ckpt=True, fused_lp=False, kopts='small', sizes=list(PACK_SIZES), budget=8192)),
    ('dedup_ckpt_small', dict(dedup=True, ckpt=True, fused_lp=False, kopts='small')),
    ('dedup_nockpt_small', dict(dedup=True, ckpt=False, fused_lp=False, kopts='small')),
    ('dedup_nockpt_small_flp', dict(dedup=True, ckpt=False, fused_lp=True, kopts='small')),
    ('dedup_nockpt_default_flp', dict(dedup=True, ckpt=False, fused_lp=True, kopts=None)),
    ('dedup_nockpt_b200_flp', dict(dedup=True, ckpt=False, fused_lp=True, kopts=_B200)),
    ('dedup_nockpt_small_flp_32k', dict(dedup=True, ckpt=False, fused_lp=True, kopts='small', budget=32768)),
    ('tri_nodedup_ckpt', dict(dedup=False, ckpt=True, fused_lp=False, attn='triton')),
    ('tri_dedup_ckpt', dict(dedup=True, ckpt=True, fused_lp=False, attn='triton')),
    ('tri_dedup_nockpt_flp', dict(dedup=True, ckpt=False, fused_lp=True, attn='triton')),
    ('tri_dedup_ckpt_flp_32k', dict(dedup=True, ckpt=True, fused_lp=True, attn='triton', budget=32768)),
    ('tri_dedup_auto_flp', dict(dedup=True, ckpt='auto', fused_lp=True, attn='triton')),
    ('tri_dedup_auto_flp_ew', dict(dedup=True, ckpt='auto', fused_lp=True, attn='triton', compile_ew=True)),
    ('tri_dedup_ckpt_flp_ew_32k', dict(dedup=True, ckpt=True, fused_lp=True, attn='triton', compile_ew=True, budget=32768)),
]
if _o.environ.get('TB_VARIANTS'):
    _want = _o.environ['TB_VARIANTS'].split(',')
    _VARIANTS = [v for v in _VARIANTS if v[0] in _want]
for _name, _kw in _VARIANTS:
    try:
        _tr = FT.Trainer(P32, names, eng, T=a.T, clip=a.clip, **_kw)
        for _r_ in range(_reps):
            st, dt, mem = _run(lambda: _tr.grads(docs, n, _norm_len))
            G = _snap(); c = _cmp(G, _REF[0])
            _B(_name, f'run{_r_}', f'{dt:.2f}s', f'peak+{mem:.1f}GB', _bj.dumps(c),
               _bj.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in st.items()}))
            if _TRUTH is not None and _r_ == 0: _B(_name, 'vs_truth', _bj.dumps(_cmp(G, _TRUTH)))
            del G
        _res[_name] = dt
        if _o.environ.get('TB_PROF', '1') == '1' and _name in _o.environ.get('TB_PROFV', 'dedup_nockpt_small_flp'):
            _prof(lambda: _tr.grads(docs, n, _norm_len), _name)
        del _tr
    except Exception as _e:
        import traceback; traceback.print_exc()
        _B(_name, 'FAILED', repr(_e)[:400])
        for k in names: P32[k].grad = None
    torch.cuda.empty_cache()

# optimizer step: torch foreach AdamW (as now) vs fused AdamW, on the reference gradient (weights change: bench only)
for _fused in (False, True):
  try:
    for k in names: P32[k].grad = _REF[0][k].clone()
    for g_ in opt.param_groups: g_['fused'] = _fused; g_['foreach'] = not _fused
    if _fused:
        for s_ in opt.state.values(): s_['step'] = s_['step'].to(dev, torch.float32)
    _ts = []
    for _r_ in range(3):
        torch.cuda.synchronize(); t0 = _bt.time()
        torch.nn.utils.clip_grad_norm_([P32[k] for k in names], 1.0); opt.step(); torch.cuda.synchronize(); _ts.append(_bt.time() - t0)
    _B('adamw', 'fused' if _fused else 'foreach', ' '.join(f'{x * 1e3:.0f}ms' for x in _ts))
  except Exception as _e:
    _B('adamw', _fused, 'FAILED', repr(_e)[:300])
torch.cuda.synchronize(); t0 = _bt.time(); sync_weights(); torch.cuda.synchronize(); _B('sync_weights', f'{(_bt.time() - t0) * 1e3:.0f}ms')
_B('summary', _bj.dumps({k: round(v, 2) for k, v in _res.items()}))
print('RLDONE', flush=True)
