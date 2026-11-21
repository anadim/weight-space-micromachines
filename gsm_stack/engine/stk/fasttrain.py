"""Fast trainer for rl_hook (task[5].1): the same Dr.GRPO loss / gradients as rl_hook.train_grads, computed faster.

Reference semantics (rl_hook.train_grads): every segment doc = [question + segment tokens] is its own packed document
(the forget hides all other segments: its -1e4 bias is exactly 0 after the fp32 softmax), attending to the shared 8-shot
prefix (KV detached: no gradient through the prefix) + itself causally.

What changes (all exact up to fp32 summation order unless noted):
- question dedup: all docs of one question share ONE copy of the question tokens in the pack (group layout
  [q, seg_1, seg_2, ...]); a segment token attends to prefix + its group's question + its own segment (causal). The
  question's activations / KV are identical in every doc, so the gradient is the sum the duplicated layout computes.
- the host is fused to bf16 once per update (bf16 leaves; per-micro grads accumulated in fp32, as P32.grad did).
- optional: no activation checkpointing (ckpt=False), fused log-prob head (lmhead.cu: bf16 logits kept, one pass for
  softmax stats / gather / dlogits; no fp32 full-vocab logits), flex kernel options sized for B200, larger micro.
"""
import os, ctypes, torch, torch.nn.functional as F
import fastqwen as Q
import tattn as TA
from torch.utils.checkpoint import checkpoint
HERE = os.path.dirname(os.path.abspath(__file__))
H, KVH, HD, FF, NL = Q.H, Q.KVH, Q.HD, Q.FF, Q.NL
SMALL = dict(BLOCK_M=32, BLOCK_N=32, num_stages=1, BLOCK_M1=16, BLOCK_N1=32, BLOCK_M2=32, BLOCK_N2=16)   # FLEX_SMALL

# ------------------------------------------------------------------------------- fused log-prob head (engine2 lmhead.cu)
_K = {}


def _lmk(name):
    if not _K:
        import tk
        cap = torch.cuda.get_device_capability()
        cub, _ = tk.compile_cubin(open(os.path.join(HERE, 'lmhead.cu')).read(), cap[0] * 10 + cap[1])
        for f in ('lp_fwd', 'lp_bwd'): _K[f] = tk.Kernel(cub, f)
    return _K[name]


class FusedLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, E, tgt, T):
        lg = F.linear(h, E).contiguous()                                 # (N, V) bf16 (same GEMM as the reference)
        N, V = lg.shape
        tgt = tgt.contiguous().to(torch.int64)
        lp = torch.empty(N, device=h.device); lse = torch.empty(N, device=h.device)
        if N: _lmk('lp_fwd')(N, 512, [lg.data_ptr(), tgt.data_ptr(), lp.data_ptr(), lse.data_ptr(), ctypes.c_int(V), ctypes.c_float(1.0 / T)])
        ctx.save_for_backward(h, E, tgt, lse, lg); ctx.T = T
        return lp

    @staticmethod
    def backward(ctx, g):
        h, E, tgt, lse, lg = ctx.saved_tensors
        N, V = lg.shape; g = g.contiguous().float()
        if N: _lmk('lp_bwd')(N, 512, [lg.data_ptr(), tgt.data_ptr(), lse.data_ptr(), g.data_ptr(), ctypes.c_int(V), ctypes.c_float(1.0 / ctx.T)])
        dh = lg @ E if ctx.needs_input_grad[0] else None
        dE = lg.t() @ h if ctx.needs_input_grad[1] else None
        return dh, dE, None, None


def lp_ref(E, hn, tgt, T):                    # == tw_dump_lean.lp_of_hidden (fp32 logits per 4096 rows, recomputed)
    def chunk_lp(hc, tc):
        lg = F.linear(hc, E).float() / T
        return torch.log_softmax(lg, -1).gather(-1, tc[:, None])[:, 0]
    return torch.cat([checkpoint(chunk_lp, hn[c:c + 4096], tgt[c:c + 4096], use_reentrant=False) for c in range(0, hn.shape[0], 4096)])


# ----------------------------------------------------------------------------------------------------- packed buffers
_EW = []


def _compiled_ew():
    """torch.compile'd (inductor) token-major layer halves around the attention: rms + qkv + q/k-norm + RoPE, and
    o-proj + residual + rms + SwiGLU MLP + residual (pointwise ops fused, GEMMs stay cuBLAS). Dynamic in L."""
    if _EW: return _EW
    def pre(x, ln1, Wqkv, qn, kn, cos, sin):
        L = x.shape[0]
        qkv = F.linear(Q.rms(x, ln1), Wqkv)
        q = Q.rms(qkv[:, :H * HD].view(L, H, HD), qn)
        k = Q.rms(qkv[:, H * HD:(H + KVH) * HD].view(L, KVH, HD), kn)
        v = qkv[:, (H + KVH) * HD:].reshape(L, KVH, HD).contiguous()
        return q * cos + Q.rot(q) * sin, k * cos + Q.rot(k) * sin, v

    def post(x, a, Wo, ln2, Wgu, Wd):
        x = x + F.linear(a.view(x.shape[0], H * HD), Wo)
        gu = F.linear(Q.rms(x, ln2), Wgu)
        g, u = gu.chunk(2, -1)
        return x + F.linear(F.silu(g) * u, Wd)
    _EW.extend([torch.compile(pre, dynamic=True), torch.compile(post, dynamic=True)])
    return _EW


class GroupPack:
    """Fixed packed length Ltot. Per token: grp (group = one question copy + its segments), doc (segment id, -1 on the
    question tokens and pads). Buffers are reused so the mask_mod closure (and the compiled flex kernel) never change."""

    def __init__(self, Ltot, Lp, device):
        self.Ltot, self.Lp = Ltot, Lp
        self.gq = torch.full((Ltot,), -1, dtype=torch.int32, device=device)
        self.dq = torch.full((Ltot,), -1, dtype=torch.int32, device=device)
        self.gk = torch.full((Lp + Ltot,), -7, dtype=torch.int32, device=device)
        self.dk = torch.full((Lp + Ltot,), -7, dtype=torch.int32, device=device)
        gq, dq, gk, dk = self.gq, self.dq, self.gk, self.dk

        def mask_mod(b, h, q, kv):
            return (kv < Lp) | ((gk[kv] == gq[q]) & (kv - Lp <= q) & ((dk[kv] < 0) | (dk[kv] == dq[q])))
        self.mask_mod = mask_mod

    def set(self, grp, doc):
        from torch.nn.attention.flex_attention import create_block_mask
        self.gq.copy_(grp); self.dq.copy_(doc); self.gk[self.Lp:].copy_(grp); self.dk[self.Lp:].copy_(doc)
        return create_block_mask(self.mask_mod, 1, None, self.Ltot, self.Lp + self.Ltot, device=self.gq.device)


class Trainer:
    # activation memory without checkpointing (triton path, fused LP head): ~2.6 MB per packed token (bench j23183)
    NOCKPT_BYTES_PER_TOK = 2.8e6
    PG = False          # bench only: loss = -A*lp (fixed per-token weights, no ratio/clip) to compare raw gradient numerics

    def __init__(self, P32, names, eng, T=1.0, clip=0.2, budget=16384, ckpt=True, fused_lp=False, dedup=True,
                 kopts='small', sizes=None, attn='flex', compile_ew=False, dtype=torch.bfloat16):
        self.P32, self.names, self.eng, self.T, self.clip = P32, names, eng, T, clip
        self.budget, self.ckpt, self.fused_lp, self.dedup, self.attn = budget, ckpt, fused_lp, dedup, attn
        self._ckpt_now = ckpt is True; self.dtype = dtype   # (dtype float32: the high-precision bench truth only)
        self.ew = _compiled_ew() if compile_ew else None
        self.kopts = SMALL if kopts == 'small' else kopts
        self.Lp = int(eng.prefix_ids.numel()); self.dev = P32[names[0]].device
        self.sizes = sizes or sorted({max(1024, (budget * k // 8 + 1023) // 1024 * 1024) for k in (2, 4, 6, 8)})
        self.packs = {}
        from torch.nn.attention.flex_attention import flex_attention
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        self.flex = torch.compile(flex_attention, dynamic=False)

    # ------------------------------------------------------------------------------------------- planning / packing
    def plan(self, docs):
        """units = (question tokens, [doc idx], tokens); a question's docs are split over several units only if they
        do not fit one micro. Without dedup every doc is its own unit (= the reference layout). First-fit decreasing."""
        B = self.budget; units = []
        if self.dedup:
            byq = {}
            for i, d in enumerate(docs): byq.setdefault(tuple(d['q']), []).append(i)
            for qk, idx in byq.items():
                ql = len(qk); cur, L = [], ql
                for i in sorted(idx, key=lambda i: -len(docs[i]['r'])):
                    rl = len(docs[i]['r']); assert ql + rl <= B
                    if cur and L + rl > B: units.append((qk, cur, L)); cur, L = [], ql
                    cur.append(i); L += rl
                if cur: units.append((qk, cur, L))
        else:
            units = [(tuple(d['q']), [i], len(d['q']) + len(d['r'])) for i, d in enumerate(docs)]
        units.sort(key=lambda u: -u[2])
        bins = []
        for u in units:
            for bn in bins:
                if bn[1] + u[2] <= B: bn[0].append(u); bn[1] += u[2]; break
            else: bins.append([[u], u[2]])
        return bins

    def build(self, units, docs):
        Lp = self.Lp
        ids, pos, grp, doc, sel, tgt, old, A, M = [], [], [], [], [], [], [], [], []
        o = 0
        for g, (qk, idx, _) in enumerate(units):
            q = list(qk); ql = len(q); qlast = o + ql - 1
            ids += q; pos += range(Lp, Lp + ql); grp += [g] * ql; doc += [-1] * ql; o += ql
            for i in idx:
                d = docs[i]; r = d['r']; rl = len(r)
                ids += r; pos += (d['pos'] if d['pos'] is not None else range(Lp + ql, Lp + ql + rl))
                grp += [g] * rl; doc += [i] * rl
                sel.append(qlast); sel += range(o, o + rl - 1); tgt += r
                old += d['lp']; A += [d['adv']] * rl; M += d['mask']; o += rl
        T_ = lambda x, dt=torch.long: torch.tensor(x, dtype=dt).pin_memory().to(self.dev, non_blocking=True)
        if self.attn == 'triton':                       # runtime-L kernels: no padding, no block mask
            g_, d_ = T_(grp, torch.int32), T_(doc, torch.int32)
            return dict(ids=T_(ids), pos=T_(pos), grp=g_, doc=d_, mt=TA.meta(g_, d_, o), sel=T_(sel), tgt=T_(tgt),
                        old=T_(old, torch.float32), A=T_(A, torch.float32), M=T_(M, torch.bool), n=o, Ltot=o)
        Ltot = next(n for n in self.sizes if n >= o)
        pad = Ltot - o
        ids += [self.eng.eos] * pad; pos += [Lp] * pad; grp += [-1] * pad; doc += [-1] * pad
        if Ltot not in self.packs: self.packs[Ltot] = GroupPack(Ltot, Lp, self.dev)
        pk = self.packs[Ltot]
        bm = pk.set(T_(grp, torch.int32), T_(doc, torch.int32))
        return dict(ids=T_(ids), pos=T_(pos), bm=bm, sel=T_(sel), tgt=T_(tgt), old=T_(old, torch.float32),
                    A=T_(A, torch.float32), M=T_(M, torch.bool), n=o, Ltot=Ltot)

    # ----------------------------------------------------------------------------------------------------- forward
    def fwd(self, W, mb, pkv):
        if self.attn == 'triton': return self.fwd_tok(W, mb, pkv)
        cos_t, sin_t = self.eng.cos_t, self.eng.sin_t
        x = W['E'][mb['ids']][None]
        cos, sin = cos_t[mb['pos']][None], sin_t[mb['pos']][None]
        bm, flex, ko = mb['bm'], self.flex, self.kopts

        def layer(x, lw, Kp, Vp):
            q, k, v = Q.qkv_rope(lw, Q.rms(x, lw['ln1']), cos, sin)
            K = torch.cat([Kp[None], k], 2); V = torch.cat([Vp[None], v], 2)
            a = flex(q, K, V, block_mask=bm, enable_gqa=True, kernel_options=ko)
            x = x + F.linear(a.transpose(1, 2).reshape(1, -1, H * HD), lw['Wo'])
            return x + Q.mlp(lw, x)
        for i, lw in enumerate(W['L']):
            Kp, Vp = pkv[i]
            x = checkpoint(layer, x, lw, Kp, Vp, use_reentrant=False) if self.ckpt else layer(x, lw, Kp, Vp)
        return Q.rms(x, W['lnf'])[0]

    def fwd_tok(self, W, mb, pkv):
        """token-major (L, D) layers with the Triton group attention; pkv[i] = (Kp, Vp) as (Lp, KVH, HD)."""
        L = mb['n']
        cos, sin = self.eng.cos_t[mb['pos']][:, None], self.eng.sin_t[mb['pos']][:, None]
        grp, doc, mt = mb['grp'], mb['doc'], mb['mt']

        def layer(x, lw, Kp, Vp):
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
        ckpt = self._ckpt_now
        for i, lw in enumerate(W['L']):
            Kp, Vp = pkv[i]
            x = checkpoint(layer, x, lw, Kp, Vp, use_reentrant=False) if ckpt else layer(x, lw, Kp, Vp)
        return Q.rms(x, W['lnf'])

    # ----------------------------------------------------------------------------------------------------- gradients
    def grads(self, docs, n_rows, norm_len):
        """sets P32[k].grad (fp32) like rl_hook.train_grads; returns the stats dict."""
        P32, eng = self.P32, self.eng
        W = Q.fuse({k: P32[k].detach() for k in self.names}, self.dtype)
        leaves = [W['E'], W['lnf']] + [t for l in W['L'] for t in l.values()]
        for t in leaves: t.requires_grad_(True)
        pkv = Q.prefix_kv_of(W, eng.prefix_ids, eng.cos_t, eng.sin_t)
        pkv = [(k.detach(), v.detach()) for k, v in pkv]
        if self.attn == 'triton': pkv = [(k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()) for k, v in pkv]
        acc = [torch.zeros(t.shape, dtype=torch.float32, device=t.device) for t in leaves]
        if self.ckpt == 'auto':
            free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            self._ckpt_now = free < self.NOCKPT_BYTES_PER_TOK * self.budget + 4e9
        bins = self.plan(docs)
        norm = n_rows * norm_len
        ntrain = sum(sum(d['mask']) for d in docs)
        st = dict(loss=0.0, clipfrac=0.0, ratio=0.0, ntok_train=ntrain, n_docs=len(docs), n_micro=len(bins),
                  ntok_packed=sum(b[1] for b in bins), ckpt=int(bool(self._ckpt_now if self.attn == 'triton' else self.ckpt)))
        for units, _ in bins:
            mb = self.build(units, docs)
            hn = self.fwd(W, mb, pkv)
            h_sel = hn[mb['sel']]
            lp = FusedLP.apply(h_sel, W['E'], mb['tgt'], self.T) if self.fused_lp else lp_ref(W['E'], h_sel, mb['tgt'], self.T)
            M = mb['M']; A = mb['A']
            ratio = torch.exp(torch.clamp(lp - mb['old'], -20, 20))
            per = -torch.minimum(ratio * A, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * A) if not self.PG else -lp * A
            loss = (per * M).sum() / norm
            loss.backward()
            for t, ac in zip(leaves, acc):
                if t.grad is not None: ac += t.grad.float(); t.grad = None
            with torch.no_grad():
                st['loss'] += float(loss.detach())
                st['ratio'] += float((ratio * M).sum()) / max(1, ntrain)
                st['clipfrac'] += float(((((ratio - 1).abs() > self.clip) & M).float()).sum()) / max(1, ntrain)
            del hn, h_sel, lp, ratio, per, loss, mb
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


def from_env(P32, names, eng, a):
    """RL_FASTTRAIN=1 -> Trainer with FT_* options: FT_ATTN (flex|triton), FT_CKPT (1|0|auto), FT_LP, FT_DEDUP,
    FT_COMPILE (inductor-fused elementwise, triton path), FT_BUDGET, FT_KOPTS (flex: json|small|default)."""
    import json
    ko = os.environ.get('FT_KOPTS', 'small')
    ko = 'small' if ko == 'small' else (None if ko == 'default' else json.loads(ko))
    ck = os.environ.get('FT_CKPT', '1'); ck = 'auto' if ck == 'auto' else ck == '1'
    return Trainer(P32, names, eng, T=a.T, clip=a.clip, budget=int(os.environ.get('FT_BUDGET', a.tok_budget)),
                   ckpt=ck, fused_lp=os.environ.get('FT_LP', '0') == '1',
                   dedup=os.environ.get('FT_DEDUP', '1') == '1', kopts=ko, attn=os.environ.get('FT_ATTN', 'flex'),
                   compile_ew=os.environ.get('FT_COMPILE', '0') == '1')
