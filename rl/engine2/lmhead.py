"""Fused log-prob head (custom kernels, NVRTC): lp = log_softmax(float(h @ E^T) / T)[tgt] without fp32 logits.
Logits come from the same cuBLAS bf16 GEMM as F.linear; softmax statistics, target gather and the logit gradient are
one pass each over the bf16 logits (kept for backward instead of recomputed). dh / dE: cuBLAS GEMMs as in F.linear."""
import os, ctypes, torch
import tk
HERE = os.path.dirname(os.path.abspath(__file__))
_K = {}


def _k(name):
    if not _K:
        cap = torch.cuda.get_device_capability()
        cub, _ = tk.compile_cubin(open(os.path.join(HERE, 'lmhead.cu')).read(), cap[0] * 10 + cap[1])
        for f in ('lp_fwd', 'lp_bwd'): _K[f] = tk.Kernel(cub, f)
    return _K[name]


class FusedLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, E, tgt, T):
        lg = torch.nn.functional.linear(h, E).contiguous()               # (N, V) bf16, as the reference
        N, V = lg.shape; assert V % 2 == 0
        tgt = tgt.contiguous().to(torch.int64)
        lp = torch.empty(N, device=h.device); lse = torch.empty(N, device=h.device)
        _k('lp_fwd')(N, 512, [lg.data_ptr(), tgt.data_ptr(), lp.data_ptr(), lse.data_ptr(), ctypes.c_int(V), ctypes.c_float(1.0 / T)])
        ctx.save_for_backward(h, E, tgt, lse, lg); ctx.T = T
        return lp

    @staticmethod
    def backward(ctx, g):
        h, E, tgt, lse, lg = ctx.saved_tensors
        N, V = lg.shape; g = g.contiguous().float()
        _k('lp_bwd')(N, 512, [lg.data_ptr(), tgt.data_ptr(), lse.data_ptr(), g.data_ptr(), ctypes.c_int(V), ctypes.c_float(1.0 / ctx.T)])
        dh = lg @ E if ctx.needs_input_grad[0] else None
        dE = lg.t() @ h if ctx.needs_input_grad[1] else None
        return dh, dE, None, None


def fused_lp(h, E, tgt, T):
    return FusedLP.apply(h, E, tgt, T)
