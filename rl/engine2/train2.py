"""engine2 trainer, step 1: train_step2 = the old train_step with the tool features taken from the rollout records
(engine2 res['E'] / res['Eq'], exact 0/1 machine outputs) instead of recomputing them teacher-forced with CSR stacks
(32 of the 44 s on the A40), and the per-token Python suppression loop replaced by a vectorised scan (cummax).
Everything else (packed flex forward, adapters, in-wire decision log-probs, Dr.GRPO loss, backward, clip, AdamW,
decoupled gate L1) is the old code, so loss / grads are identical whenever the features are.
make_patch(src) returns the train_library source with train_step2 added (exec'd by ref_env.load)."""

NEW_BLOCK = '''        feat = fwrite = None; dpos, didx, dsgn, dtool = [], [], [], []
        if imp is not None and a.arm == 'LIB' and not a.plain:
            with torch.no_grad():
                NO_ = eng.TK.nout
                Eall = torch.zeros(Ltot, NO_, device=dev)
                Dall = torch.zeros(Ltot, NT, dtype=torch.int64, device=dev)
                Rm = torch.zeros(Ltot, dtype=torch.bool, device=dev)
                ridx = torch.zeros(Ltot, dtype=torch.long, device=dev)
                segs = torch.zeros(Ltot, dtype=torch.long, device=dev)
                o2 = si = 0
                for j, i in enumerate(mb):
                    ql, rl = len(batch_q[i]), len(batch_r[i]); dl = batch_dec[i]
                    Eall[o2:o2 + ql] = batch_E_q[i].to(dev, non_blocking=True).float()
                    Eall[o2 + ql:o2 + ql + rl] = batch_E[i][1:rl + 1].to(dev, non_blocking=True).float()
                    dlr = dl[:rl].to(torch.int64) if len(dl) >= rl else torch.cat([dl.to(torch.int64), torch.zeros(rl - len(dl), NT, dtype=torch.int64)])
                    p0 = o2 + ql - 1
                    Dall[p0:p0 + rl] = dlr.to(dev); Rm[p0:p0 + rl] = True
                    ridx[p0:p0 + rl] = torch.arange(si, si + rl, device=dev); segs[p0:p0 + rl] = p0
                    o2 += lens[j]; si += rl
                Es = {}
                for k_, (nm_, (oo, nn_)) in enumerate(zip(TNAMES, eng.TK.outoff)):
                    Es[nm_] = Eall[:, oo:oo + nn_] * eng.tool_on[k_]
                if IFT is not None:                                # recorded IF writes (sparse) at response positions
                    wif = torch.zeros(Ltot, Q.D, device=dev); o2 = 0
                    for j, i in enumerate(mb):
                        ql = len(batch_q[i]); pz, vz = batch_ifw[i]
                        if len(pz): wif[o2 + ql - 1 + pz.to(dev)] = vz.to(dev)
                        o2 += lens[j]
                    Es['ifeval'] = wif
                nzall = torch.stack([(Es[n].abs().sum(-1) > 0) for n in TNAMES], -1)       # (Ltot, NT)
                # suppression scan (old per-token loop): sup_t = the last decision at t' <= t (same response) was a veto
                # AND the tool output stayed nonzero on [t', t]
                pos = torch.arange(Ltot, device=dev)[:, None].expand(Ltot, NT)
                lastd = torch.cummax(torch.where((Dall > 0) & Rm[:, None], pos, -1), 0).values
                lastz = torch.cummax(torch.where((~nzall) & Rm[:, None], pos, -1), 0).values
                vet = Dall.gather(0, lastd.clamp(min=0)) == 2
                sup = (lastd >= segs[:, None]) & vet & (lastz < lastd)
                keep = torch.where(Rm[:, None], ~sup, torch.ones_like(sup))
                dm = (Dall > 0) & Rm[:, None]
                nzp = dm.nonzero()
                dpos = nzp[:, 0].tolist(); dtool = nzp[:, 1].tolist(); didx = ridx[nzp[:, 0]].tolist()
                dsgn = torch.where(Dall[nzp[:, 0], nzp[:, 1]] == 1, 1.0, -1.0).tolist()
                for k_, n in enumerate(TNAMES): Es[n] = Es[n] * keep[:, k_:k_ + 1].to(Es[n].dtype)
'''


def make_patch(src):
    a = src.index('def train_step('); b = src.index('@torch.no_grad()\ndef hf_check')
    f = src[a:b]
    s0 = f.index("        feat = fwrite = None; dpos, didx, dsgn, dtool = [], [], [], []\n")
    s1 = f.index("                fwrite = torch.zeros(Ltot, Q.D, device=dev)\n")
    g = f[:s0] + NEW_BLOCK + f[s1:]
    g = g.replace('def train_step(batch_q, batch_r, old_lp, adv, n_total_rows, batch_dec=None, batch_ifw=None):',
                  'def train_step2(batch_q, batch_r, old_lp, adv, n_total_rows, batch_dec=None, batch_ifw=None, batch_E=None, batch_E_q=None):')
    assert 'def train_step2(' in g
    return src[:b] + g + src[b:]


FUSE_ONCE_SETUP = '''    # fuse the host weights ONCE per update (old code: once per micro-batch); bf16 leaves, their per-micro grads
    # accumulated in fp32 in micro order (= what autograd did into P32.grad), unfused into P32.grad before the prefix
    # backward (same order as before) -> bitwise-identical gradients
    Wt_once = Q.fuse({k_: v_.detach() for k_, v_ in P32.items()}, torch.bfloat16)
    leaves_W = [Wt_once['E'], Wt_once['lnf']] + [t_ for l_ in Wt_once['L'] for t_ in l_.values()]
    for t_ in leaves_W: t_.requires_grad_(not a.freeze_host)
    accW = None if a.freeze_host else [torch.zeros(t_.shape, dtype=torch.float32, device=dev) for t_ in leaves_W]
    for mb in micro:
'''
ACC_AFTER_BWD = '''        loss.backward()
        if accW is not None:
            for t_, ac_ in zip(leaves_W, accW):
                if t_.grad is not None: ac_ += t_.grad.float(); t_.grad = None
'''
UNFUSE = '''    if accW is not None:
        H_, KVH_, HD_, FF_ = Q.H, Q.KVH, Q.HD, Q.FF
        gd = {'model.embed_tokens.weight': accW[0], 'model.norm.weight': accW[1]}
        for i_ in range(Q.NL):
            ln1, qn, kn, ln2, Wqkv, Wo, Wgu, Wd = accW[2 + 8 * i_: 10 + 8 * i_]
            p_ = f'model.layers.{i_}.'
            gd[p_ + 'input_layernorm.weight'] = ln1; gd[p_ + 'self_attn.q_norm.weight'] = qn; gd[p_ + 'self_attn.k_norm.weight'] = kn
            gd[p_ + 'post_attention_layernorm.weight'] = ln2
            gd[p_ + 'self_attn.q_proj.weight'] = Wqkv[:H_ * HD_]; gd[p_ + 'self_attn.k_proj.weight'] = Wqkv[H_ * HD_:(H_ + KVH_) * HD_]
            gd[p_ + 'self_attn.v_proj.weight'] = Wqkv[(H_ + KVH_) * HD_:]
            gd[p_ + 'self_attn.o_proj.weight'] = Wo; gd[p_ + 'mlp.gate_proj.weight'] = Wgu[:FF_]; gd[p_ + 'mlp.up_proj.weight'] = Wgu[FF_:]
            gd[p_ + 'mlp.down_proj.weight'] = Wd
        for k_ in names:
            g_ = gd[k_].contiguous()
            P32[k_].grad = g_ if P32[k_].grad is None else P32[k_].grad + g_
    if not a.freeze_host:
        torch.autograd.backward([t for kv in pkv'''


def make_patch_fast(src):
    """train_step2 + fuse once per update (bitwise identical)."""
    src = make_patch(src)
    a = src.index('def train_step2('); b = src.index('@torch.no_grad()\ndef hf_check')
    f = src[a:b]
    for x, y in [('    for mb in micro:\n', FUSE_ONCE_SETUP), ('        Wt = Q.fuse(P32, torch.bfloat16)\n', '        Wt = Wt_once\n'),
                 ('        loss.backward()\n', ACC_AFTER_BWD), ('    if not a.freeze_host:\n        torch.autograd.backward([t for kv in pkv', UNFUSE)]:
        assert f.count(x) == 1, x
        f = f.replace(x, y)
    return src[:a] + f + src[b:]


def install_nockpt(Q):
    """packed forward without per-layer activation checkpointing (same ops; activations kept): bitwise identical."""
    import inspect
    if getattr(Q, '_nockpt', False): return
    src = inspect.getsource(Q.fwd_packed).replace('if torch.is_grad_enabled():', 'if False:')
    ns = dict(Q.__dict__); exec(src, ns); Q.fwd_packed = ns['fwd_packed']; Q._nockpt = True


def make_patch_fused(src):
    """make_patch_fast + fused log-prob head (numerically, not bitwise, equal: fp32 softmax reduction order)."""
    src = make_patch_fast(src)
    a = src.index('def train_step2('); b = src.index('@torch.no_grad()\ndef hf_check')
    f = src[a:b]
    x = '        lp = lp_of_hidden(Wt, hn[sel], tgt)\n'
    assert f.count(x) == 1
    f = f.replace(x, '        lp = __import__("lmhead").fused_lp(hn[sel], Wt["E"], tgt, a.T)\n')
    return src[:a] + f + src[b:]
