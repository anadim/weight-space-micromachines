"""Packed GRPO trainer for the BBH engine (rl24/bbh): the fasttrain.Trainer triton path (token-major layers, tattn group
attention, fused log-prob head) with Lp = 0 (no shared prefix: each group = [prompt, r_1, .., r_G], prompt tokens shared
by the G responses = question dedup) plus the two things the BBH rollout adds to the policy:

1. tool writes: the frozen micromachines' writes are added to the residual right before the final norm, exactly as in the
   rollout (rl/library train_step convention: every generated token is trained, the write is teacher-forced). The write
   at a position depends only on the last K+W tokens through the frozen E0 snapshot, so it is a constant here
   (Micromachine.emit math over [prompt tail (K+W) + response], same windows as the rollout's prime + step).
2. masked tokens: the rollout masks eos for the first `min_tokens` tokens and <tool_call> once the call gate latched;
   the sampled token's log-prob is then log softmax over the remaining vocabulary = lp - log(1 - sum p(masked)).

Loss: Dr.GRPO, per-token -min(r A, clip(r) A), summed and divided by n_rows * max_new (train_library 'drgrpo').
"""
import torch, torch.nn.functional as F
import fastqwen as Q
import tattn as TA
from fasttrain import FusedLP, _compiled_ew
from torch.utils.checkpoint import checkpoint

H, KVH, HD, FF, NL = Q.H, Q.KVH, Q.HD, Q.FF, Q.NL


@torch.no_grad()
def emit_tf(tool, ids_2d, E0):
    """teacher-forced Micromachine.emit for a BBHTool over rows ids_2d (B, T) -> e (B, T, n_out) fp32.
    Same math as machine.Micromachine.emit, with the tool's (exactness-checked) DenseStack for deep stack-1 nets."""
    from micromachines.machine import BIG, GAMMA
    M = tool.M
    sym = M.symbols(E0[ids_2d])
    win = M.windows(sym)
    S = tool.stack1(win)
    C = M.attend(S, sym[..., M.key])
    return M.stack2(torch.cat([win, C], -1)).float()


class BBHTrainer:
    NOCKPT_BYTES_PER_TOK = 2.8e6

    def __init__(self, P32, names, eng, T=1.0, clip=0.2, budget=16384, ckpt='auto', compile_ew=False):
        self.P32, self.names, self.eng, self.T, self.clip = P32, names, eng, T, clip
        self.budget, self.ckpt = budget, ckpt
        self.dev = P32[names[0]].device
        self.ew = _compiled_ew() if compile_ew else None
        self._kvbuf = torch.zeros(1, KVH, HD, device=self.dev, dtype=torch.bfloat16)

    # ---------------------------------------------------------------------------------------------- packing
    def plan(self, docs):
        B = self.budget; units = []
        byq = {}
        for i, d in enumerate(docs): byq.setdefault(d['qi'], []).append(i)
        for qk, idx in byq.items():
            ql = len(docs[idx[0]]['q']); cur, L = [], ql
            for i in sorted(idx, key=lambda i: -len(docs[i]['r'])):
                rl = len(docs[i]['r']); assert ql + rl <= B
                if cur and L + rl > B: units.append((qk, cur, L)); cur, L = [], ql
                cur.append(i); L += rl
            if cur: units.append((qk, cur, L))
        units.sort(key=lambda u: -u[2])
        bins = []
        for u in units:
            for bn in bins:
                if bn[1] + u[2] <= B: bn[0].append(u); bn[1] += u[2]; break
            else: bins.append([[u], u[2]])
        return bins

    def build(self, units, docs):
        ids, pos, grp, doc, sel, tgt, old, A, M, mpos, mtok = [], [], [], [], [], [], [], [], [], [], []
        o = 0; ns = 0
        wsel = []                                         # (packed sel index, doc i, response index t) for writes
        for g, (qk, idx, _) in enumerate(units):
            q = docs[idx[0]]['q']; ql = len(q); qlast = o + ql - 1
            ids += q; pos += range(ql); grp += [g] * ql; doc += [-1] * ql; o += ql
            for i in idx:
                d = docs[i]; r = d['r']; rl = len(r)
                ids += r; pos += range(ql, ql + rl); grp += [g] * rl; doc += [i] * rl
                sel.append(qlast); sel += range(o, o + rl - 1); tgt += r
                old += d['lp']; A += [d['adv']] * rl; M += d.get('mask', [1] * rl)
                for t, mt in d.get('masked', ()):
                    mpos.append(ns + t); mtok.append(mt)
                wsel.append((ns, i, rl))
                o += rl; ns += rl
        T_ = lambda x, dt=torch.long: torch.tensor(x, dtype=dt).pin_memory().to(self.dev, non_blocking=True)
        g_, d_ = T_(grp, torch.int32), T_(doc, torch.int32)
        return dict(ids=T_(ids), pos=T_(pos), grp=g_, doc=d_, mt=TA.meta(g_, d_, o), sel=T_(sel), tgt=T_(tgt),
                    old=T_(old, torch.float32), A=T_(A, torch.float32), M=T_(M, torch.bool), n=o, ns=ns,
                    mpos=T_(mpos), mtok=T_(mtok), wsel=wsel)

    # ---------------------------------------------------------------------------------------------- writes
    @torch.no_grad()
    def writes(self, mb, docs):
        """fp32 (ns, D) tool write at each selected (predictor) position, or None without tools."""
        eng = self.eng
        if not eng.tools_on: return None, None
        Kt = eng.Kt; ns = mb['ns']
        W = torch.zeros(ns, Q.D, device=self.dev); wf = torch.zeros(ns, dtype=torch.bool, device=self.dev)
        rows = []                                          # predictor tokens per doc: prompt tail (Kt) + r[:-1]
        for s0, i, rl in mb['wsel']:
            q = docs[i]['q']; r = docs[i]['r']
            pre = ([eng.eos] * Kt + list(q))[-Kt:]
            rows.append((s0, rl, pre + list(r[:-1])))
        for c in range(0, len(rows), 16):
            ch = rows[c:c + 16]; Tm = max(len(x[2]) for x in ch)
            ids = torch.tensor([x[2] + [eng.eos] * (Tm - len(x[2])) for x in ch], device=self.dev)
            outs = {k: emit_tf(t, ids, eng.E0) for k, t in enumerate(eng.tools)}
            wr, f = eng._tool_write({k: v.reshape(-1, v.shape[-1]) for k, v in outs.items()})
            wr = wr.view(len(ch), Tm, Q.D); f = f.view(len(ch), Tm)
            for j, (s0, rl, _) in enumerate(ch):
                W[s0:s0 + rl] = wr[j, Kt - 1:Kt - 1 + rl]; wf[s0:s0 + rl] = f[j, Kt - 1:Kt - 1 + rl]
        return W, wf

    # ---------------------------------------------------------------------------------------------- forward
    def fwd(self, W, mb, ckpt):
        L = mb['n']
        cos, sin = self.eng.cos_t[mb['pos']][:, None], self.eng.sin_t[mb['pos']][:, None]
        grp, doc, mt = mb['grp'], mb['doc'], mb['mt']
        Kp = Vp = self._kvbuf[:0]

        def layer(x, lw):
            if self.ew is not None:
                q, k, v = self.ew[0](x, lw['ln1'], lw['Wqkv'], lw['qn'], lw['kn'], cos, sin)
                a = TA.group_attn(q.contiguous(), k.contiguous(), v.contiguous(), Kp, Vp, grp, doc, mt)
                return self.ew[1](x, a, lw['Wo'], lw['ln2'], lw['Wgu'], lw['Wd'])
            qkv = F.linear(Q.rms(x, lw['ln1']), lw['Wqkv'])
            q = Q.rms(qkv[:, :H * HD].view(L, H, HD), lw['qn'])
            k = Q.rms(qkv[:, H * HD:(H + KVH) * HD].view(L, KVH, HD), lw['kn'])
            v = qkv[:, (H + KVH) * HD:].reshape(L, KVH, HD).contiguous()
            q = q * cos + Q.rot(q) * sin; k = k * cos + Q.rot(k) * sin
            a = TA.group_attn(q, k, v, Kp, Vp, grp, doc, mt)
            x = x + F.linear(a.view(L, H * HD), lw['Wo'])
            return x + Q.mlp(lw, x)
        x = W['E'][mb['ids']]
        for lw in W['L']:
            x = checkpoint(layer, x, lw, use_reentrant=False) if ckpt else layer(x, lw)
        return x                                           # pre-final-norm residual (L, D)

    def lp_of(self, W, xs, mb, wr):
        if wr is not None: xs = xs + wr.to(xs.dtype)
        h = Q.rms(xs, W['lnf'])
        lp = FusedLP.apply(h, W['E'], mb['tgt'], self.T)
        if mb['mpos'].numel():
            lpm = FusedLP.apply(h[mb['mpos']], W['E'], mb['mtok'], self.T)
            ps = torch.zeros_like(lp).index_add(0, mb['mpos'], torch.exp(lpm))
            lp = lp - torch.log1p(-ps.clamp(max=1 - 1e-7))
        return lp

    # ---------------------------------------------------------------------------------------------- gradients
    def grads(self, docs, n_rows, norm_len, check_only=False):
        """docs: dict(qi, q, r, lp (old), adv, masked=[(t, tok)], mask). Sets P32[k].grad (fp32); returns stats.
        check_only: no backward, returns (stats, per-doc new lp lists, per-doc teacher-forced write flags)."""
        P32 = self.P32
        W = Q.fuse({k: P32[k].detach() for k in self.names}, torch.bfloat16)
        leaves = [W['E'], W['lnf']] + [t for l in W['L'] for t in l.values()]
        if not check_only:
            for t in leaves: t.requires_grad_(True)
        acc = [torch.zeros(t.shape, dtype=torch.float32, device=t.device) for t in leaves] if not check_only else None
        ckpt = self.ckpt
        if ckpt == 'auto':
            free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            ckpt = free < self.NOCKPT_BYTES_PER_TOK * self.budget + 4e9
        bins = self.plan(docs)
        norm = n_rows * norm_len
        ntrain = sum(sum(d.get('mask', [1] * len(d['r']))) for d in docs)
        st = dict(loss=0.0, clipfrac=0.0, ratio=0.0, ntok_train=ntrain, n_docs=len(docs), n_micro=len(bins),
                  ntok_packed=sum(b[1] for b in bins), ckpt=int(bool(ckpt)), max_dlp=0.0, wf_mismatch=0)
        new_lp = {} if check_only else None; tf_wf = {} if check_only else None
        for units, _ in bins:
            mb = self.build(units, docs)
            wr, wf = self.writes(mb, docs)
            ctx = torch.no_grad() if check_only else torch.enable_grad()
            with ctx:
                x = self.fwd(W, mb, ckpt and not check_only)
                lp = self.lp_of(W, x[mb['sel']], mb, wr)
                Mk = mb['M']; A = mb['A']
                ratio = torch.exp(torch.clamp(lp - mb['old'], -20, 20))
                per = -torch.minimum(ratio * A, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * A)
                loss = (per * Mk).sum() / norm
            if not check_only:
                loss.backward()
                for t, ac in zip(leaves, acc):
                    if t.grad is not None: ac += t.grad.float(); t.grad = None
            with torch.no_grad():
                st['loss'] += float(loss.detach())
                st['ratio'] += float((ratio * Mk).sum()) / max(1, ntrain)
                st['clipfrac'] += float(((((ratio - 1).abs() > self.clip) & Mk).float()).sum()) / max(1, ntrain)
                st['max_dlp'] = max(st['max_dlp'], float((lp - mb['old']).abs().max()) if lp.numel() else 0.0)
                if check_only:
                    lpl = lp.tolist(); wfl = wf.tolist() if wf is not None else None
                    for s0, i, rl in mb['wsel']:
                        new_lp[i] = lpl[s0:s0 + rl]
                        if wfl is not None: tf_wf[i] = [int(v) for v in wfl[s0:s0 + rl]]
            del x, lp, ratio, per, loss, mb
        if check_only: return st, new_lp, tf_wf
        gd = {'model.embed_tokens.weight': acc[0], 'model.norm.weight': acc[1]}
        for i in range(NL):
            ln1, qn, kn, ln2, Wqkv, Wo, Wgu, Wd = acc[2 + 8 * i: 10 + 8 * i]
            p = f'model.layers.{i}.'
            gd[p + 'input_layernorm.weight'] = ln1; gd[p + 'self_attn.q_norm.weight'] = qn; gd[p + 'self_attn.k_norm.weight'] = kn
            gd[p + 'post_attention_layernorm.weight'] = ln2
            gd[p + 'self_attn.q_proj.weight'] = Wqkv[:H * HD]; gd[p + 'self_attn.k_proj.weight'] = Wqkv[H * HD:(H + KVH) * HD]
            gd[p + 'self_attn.v_proj.weight'] = Wqkv[(H + KVH) * HD:]
            gd[p + 'self_attn.o_proj.weight'] = Wo; gd[p + 'mlp.gate_proj.weight'] = Wgu[:FF]; gd[p + 'mlp.up_proj.weight'] = Wgu[FF:]
            gd[p + 'mlp.down_proj.weight'] = Wd
        for k in self.names:
            g = gd[k].contiguous()
            P32[k].grad = g if P32[k].grad is None else P32[k].grad + g
        return st
