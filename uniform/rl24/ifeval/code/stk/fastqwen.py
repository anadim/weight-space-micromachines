"""Hand-written Qwen3-0.6B-Base forward (bf16, same op order as HF Qwen3) for rollouts and training.

Shared-prefix layout: every sequence = PREFIX (the fixed 8-shot prompt, Lp tokens, KV computed once per weight version)
+ SUFFIX (question + answer). Attention of a suffix token = over all prefix keys + the causal suffix keys.

fwd_chunk : many suffix tokens at once (question prefill, teacher-forced training). SDPA (memory-efficient kernel).
DecodeStep: one token per row, static buffers, CUDA-graph capturable; prefix attention is one shared GEMM over all rows.
"""
import math, torch, torch.nn.functional as F

H, KVH, HD, D, FF, NL, EPS, THETA = 16, 8, 128, 1024, 3072, 28, 1e-6, 1e6
LAYER_KEYS = ['input_layernorm.weight', 'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
              'self_attn.o_proj.weight', 'self_attn.q_norm.weight', 'self_attn.k_norm.weight',
              'post_attention_layernorm.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']


def load_hf_params(name='Qwen/Qwen3-0.6B-Base', device='cuda', dtype=torch.float32):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    sd = {k: v.to(device=device, dtype=dtype).contiguous() for k, v in m.state_dict().items() if k != 'lm_head.weight'}
    return sd


def rms(x, w):                                   # == Qwen3RMSNorm (fp32 statistics, cast back, weight * x)
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return w * xf.to(dt)


def rope_tables(maxpos, device, dtype=torch.bfloat16):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / HD))
    fr = torch.arange(maxpos, device=device, dtype=torch.float)[:, None] * inv[None]
    emb = torch.cat([fr, fr], -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rot(x):
    return torch.cat([-x[..., HD // 2:], x[..., :HD // 2]], -1)


def fuse(sd, dtype=torch.bfloat16):
    """HF state dict -> per-layer fused tensors (Wqkv, Wgu) in dtype. Differentiable if sd requires grad."""
    L = []
    for i in range(NL):
        g = lambda k: sd[f'model.layers.{i}.{k}']
        L.append(dict(ln1=g('input_layernorm.weight').to(dtype), qn=g('self_attn.q_norm.weight').to(dtype),
                      kn=g('self_attn.k_norm.weight').to(dtype), ln2=g('post_attention_layernorm.weight').to(dtype),
                      Wqkv=torch.cat([g('self_attn.q_proj.weight'), g('self_attn.k_proj.weight'), g('self_attn.v_proj.weight')], 0).to(dtype),
                      Wo=g('self_attn.o_proj.weight').to(dtype),
                      Wgu=torch.cat([g('mlp.gate_proj.weight'), g('mlp.up_proj.weight')], 0).to(dtype),
                      Wd=g('mlp.down_proj.weight').to(dtype)))
    return dict(E=sd['model.embed_tokens.weight'].to(dtype), lnf=sd['model.norm.weight'].to(dtype), L=L)


def qkv_rope(lw, h, cos, sin):
    """h (..., T, D) -> q (..., H, T, HD), k, v (..., KVH, T, HD)."""
    qkv = F.linear(h, lw['Wqkv'])
    sh = h.shape[:-1]
    q = rms(qkv[..., :H * HD].view(*sh, H, HD), lw['qn']).transpose(-3, -2)
    k = rms(qkv[..., H * HD:(H + KVH) * HD].view(*sh, KVH, HD), lw['kn']).transpose(-3, -2)
    v = qkv[..., (H + KVH) * HD:].view(*sh, KVH, HD).transpose(-3, -2)
    c, s = cos.unsqueeze(-3), sin.unsqueeze(-3)
    return q * c + rot(q) * s, k * c + rot(k) * s, v


def mlp(lw, x):
    gu = F.linear(rms(x, lw['ln2']), lw['Wgu'])
    g, u = gu.chunk(2, -1)
    return F.linear(F.silu(g) * u, lw['Wd'])


def fwd_chunk(W, ids, pos0, cos_t, sin_t, prefix_kv=None, links=None, feat=None, final_write=None, return_kv=False,
              return_hidden=False, capture=None, return_pre=False):
    """ids (N,T) suffix tokens (right padding allowed: causal attention never looks at later pads).
    prefix_kv: list of (K,V) per layer, each (KVH, Lp, HD) shared by all rows, or None (no prefix).
    links: optional dict layer -> callable(feat)->(N,T,D) residual add at that layer's input ('final' = before final norm).
    feat: (N,T,nf) implant feature for links;  final_write: (N,T,D) fp32 implant write added before the final norm.
    Returns final hidden after norm (N,T,D) [+ list of (k,v) suffix per layer]."""
    N, T = ids.shape
    x = W['E'][ids]
    pos = torch.arange(pos0, pos0 + T, device=ids.device)
    cos, sin = cos_t[pos], sin_t[pos]
    Lp = 0 if prefix_kv is None else prefix_kv[0][0].shape[-2]
    if Lp:
        mask = torch.ones(T, Lp + T, dtype=torch.bool, device=ids.device)
        mask[:, Lp:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))
    kvs = []; hcap = None
    for i, lw in enumerate(W['L']):
        if i == capture: hcap = x                     # layer-`capture` input, before any link add (in-wire read)
        if links is not None and i in links:
            x = x + links[i](feat, x).to(x.dtype)
        q, k, v = qkv_rope(lw, rms(x, lw['ln1']), cos, sin)
        if return_kv: kvs.append((k, v))
        if Lp:
            Kp, Vp = prefix_kv[i]
            K = torch.cat([Kp.unsqueeze(0).expand(N, -1, -1, -1), k], 2)
            V = torch.cat([Vp.unsqueeze(0).expand(N, -1, -1, -1), v], 2)
            K = K.repeat_interleave(H // KVH, 1); V = V.repeat_interleave(H // KVH, 1)
            a = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        else:
            a = F.scaled_dot_product_attention(q, k.repeat_interleave(H // KVH, 1), v.repeat_interleave(H // KVH, 1), is_causal=True)
        x = x + F.linear(a.transpose(1, 2).reshape(N, T, H * HD), lw['Wo'])
        x = x + mlp(lw, x)
    if links is not None and 'final' in links:
        x = x + links['final'](feat, x).to(x.dtype)
    if return_pre: return (x, kvs) if return_kv else x          # (rl24/ifeval: pre-norm residual, the implant reads it)
    if final_write is not None:
        x = x + final_write.to(x.dtype)
    hn = rms(x, W['lnf'])
    if capture is not None: return (hn, kvs, hcap) if return_kv else (hn, hcap)
    return (hn, kvs) if return_kv else hn


def prefix_kv_of(W, prefix_ids, cos_t, sin_t):
    """KV of the shared prefix: list of (K (KVH,Lp,HD), V) per layer."""
    _, kvs = fwd_chunk(W, prefix_ids[None], 0, cos_t, sin_t, return_kv=True)
    return [(k[0], v[0]) for k, v in kvs]


# ---------------- packed training forward (flex attention, no prefix duplication, no padding between rows) -------------
_flex = None
import os as _os
# small-shared-memory GPUs (A40/A100-class sm8x with <=100 KB): smaller flex tiles
_FLEX_OPTS = (dict(BLOCK_M=32, BLOCK_N=32, num_stages=1, BLOCK_M1=16, BLOCK_N1=32, BLOCK_M2=32, BLOCK_N2=16)
              if _os.environ.get('FLEX_SMALL') else None)


def _get_flex():
    global _flex
    if _flex is None:
        from torch.nn.attention.flex_attention import flex_attention
        _flex = torch.compile(flex_attention, dynamic=False)
    return _flex


class Packer:
    """Fixed packed length Ltot (one compiled flex kernel); doc/pos buffers reused so mask_mod never changes identity."""

    def __init__(self, Ltot, Lp, device):
        self.Ltot, self.Lp, self.dev = Ltot, Lp, device
        self.doc = torch.full((Ltot,), -1, dtype=torch.int32, device=device)
        self.docp = torch.full((Lp + Ltot,), -2, dtype=torch.int32, device=device)
        self.pos = torch.zeros(Ltot, dtype=torch.long, device=device)
        doc, docp = self.doc, self.docp

        def mask_mod(b, h, q, kv):
            return (kv < Lp) | ((docp[kv] == doc[q]) & (kv - Lp <= q))
        self.mask_mod = mask_mod

    def pack(self, lens):
        from torch.nn.attention.flex_attention import create_block_mask
        assert sum(lens) <= self.Ltot
        doc = torch.full((self.Ltot,), -1, dtype=torch.int32)
        pos = torch.full((self.Ltot,), self.Lp, dtype=torch.long)
        o = 0
        for i, n in enumerate(lens):
            doc[o:o + n] = i; pos[o:o + n] = torch.arange(self.Lp, self.Lp + n); o += n
        self.doc.copy_(doc.to(self.dev)); self.docp[self.Lp:].copy_(self.doc); self.pos.copy_(pos.to(self.dev))
        return create_block_mask(self.mask_mod, 1, None, self.Ltot, self.Lp + self.Ltot, device=self.dev)


def fwd_packed(W, ids, pos, bm, cos_t, sin_t, prefix_kv, links=None, feat=None, final_write=None, capture=None, pre_write=False):
    """ids (Ltot,) packed suffix tokens; prefix_kv list of (K,V) (KVH,Lp,HD). Returns normalized final hidden (Ltot, D)."""
    flex = _get_flex()
    x = W['E'][ids][None]                                  # (1,L,D)
    cos, sin = cos_t[pos][None], sin_t[pos][None]
    from torch.utils.checkpoint import checkpoint

    def layer(x, lw, Kp, Vp):
        q, k, v = qkv_rope(lw, rms(x, lw['ln1']), cos, sin)            # (1,H,L,HD), (1,KVH,L,HD)
        K = torch.cat([Kp[None], k], 2); V = torch.cat([Vp[None], v], 2)
        a = flex(q, K, V, block_mask=bm, enable_gqa=True, kernel_options=_FLEX_OPTS)
        x = x + F.linear(a.transpose(1, 2).reshape(1, -1, H * HD), lw['Wo'])
        return x + mlp(lw, x)
    hcap = None
    for i, lw in enumerate(W['L']):
        if i == capture: hcap = x[0]                  # (L, D) layer-`capture` input, before any link add
        if links is not None and i in links:
            x = x + links[i](feat, x).to(x.dtype)
        Kp, Vp = prefix_kv[i]
        if torch.is_grad_enabled():
            x = checkpoint(layer, x, lw, Kp, Vp, use_reentrant=False)
        else:
            x = layer(x, lw, Kp, Vp)
    if links is not None and 'final' in links:
        x = x + links['final'](feat, x).to(x.dtype)
    x_pre = x                                         # host-only residual (before the calculator's final write)
    if final_write is not None:
        x = x + final_write.to(x.dtype)
    if pre_write: return rms(x, W['lnf'])[0], rms(x_pre, W['lnf'])[0]
    if capture is not None: return rms(x, W['lnf'])[0], hcap
    return rms(x, W['lnf'])[0]
