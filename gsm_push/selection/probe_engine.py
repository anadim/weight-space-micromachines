"""ProbeEngine: the RL engine (rl_wiring/engine.py) + an in-graph latch that captures the host's residual stream at the
token that completes '####' (the answer marker), at a few layers. Everything happens inside the captured CUDA-graph decode
step with tensor ops (no Python state on data):

  hash counter   hc_t = (hc_{t-1} + nh[tok_t]) if nh[tok_t] > 0 else 0      (nh = number of '#' in an all-'#' token)
  latch          fire_t = (hc_t >= 4) & ~got ; got |= fire_t
  capture        hcap[row, j] = x^{(L_j)}_t  (layer-L_j input = HF hidden_states[L_j]) when fire_t

The verifier probe is then one dot product per sample on hcap (w . h + b), and the group vote is an exact reduction over
the K rows of a question. When no row fires, decoding is token-for-token the plain engine (the capture only reads x).
"""
import torch, triton, torch.nn.functional as F
import fastqwen as Q
import hostkern as HK
from engine import Engine


class ProbeEngine(Engine):
    def probe_setup(self, layers=(14, 18, 22)):
        self.cap_layers = list(layers)
        dev = self.dev; Bp = self.B_max + 1
        V = self.W['E'].shape[0]
        txt = [self.tok.decode([i]) for i in range(len(self.tok))] + [''] * max(0, V - len(self.tok))
        nh = []
        for t in txt:
            s = t.strip()
            nh.append(len(s) if (s and set(s) == {'#'}) else 0)
        self.t_nh = torch.tensor(nh, device=dev, dtype=torch.int32)
        self.p_hc = torch.zeros(Bp, device=dev, dtype=torch.int32)        # by rowid
        self.p_got = torch.zeros(Bp, device=dev, dtype=torch.bool)        # by rowid (stable within a wave)
        self.p_pos = torch.full((Bp,), -1, device=dev, dtype=torch.long)  # by rowid: index in out_tok of the '####' token
        self.p_cap = torch.zeros(Bp, len(self.cap_layers), Q.D, device=dev, dtype=self.W['E'].dtype)
        self.graphs = {}; self.pool = None       # fresh graph pool (the old graphs are dropped)

    def _mut(self):
        m = super()._mut()
        if hasattr(self, 'p_cap'): m += [self.p_hc, self.p_got, self.p_pos, self.p_cap]
        return m

    def _step_triton(self, B, S):
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        imp = self.imp if self.imp_on else None
        links = self.links if self.imp_on else None
        emb = W['E'][tok]
        feat = write = wf = None; dec = lpd = None
        assert self.inwire is None and links is None
        if imp is not None:
            e = imp.step(emb, B, tok) * self.row_on[:B, None]
            feat = e
            if self.use_write:
                write = (e @ self.Dp) if self.Dp is not None else (imp.last_w if imp.fk is not None else e @ imp.D)
            wf = (e.abs().sum(-1) > 0).to(torch.int8)
        # ---- latch (reads only the input token) ----
        nh = self.t_nh[tok]
        r = self.rowid[:B]
        hc = torch.where(nh > 0, self.p_hc[r] + nh, torch.zeros_like(nh))
        self.p_hc[r] = hc
        fire = (hc >= 4) & ~self.p_got[r] & ~self.done[:B]
        self.p_got[r] = self.p_got[r] | fire
        self.p_pos[r] = torch.where(fire, self.glen[:B], self.p_pos[r])
        caps = []
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        x.copy_(emb)
        pos = self.Lp + slot
        ieee = x.dtype == torch.float32
        sc = 1.0 / (Q.HD ** 0.5)
        delta = None
        for i, lw in enumerate(W['L']):
            if i in self.cap_layers:
                if delta is not None: x += delta; delta = None
                caps.append(x.clone())
            HK.add_rms_kernel[(B,)](x, delta if delta is not None else x, lw['ln1'], h, D=Q.D, HAS_DELTA=delta is not None, eps=Q.EPS)
            qkv = F.linear(h, lw['Wqkv'])
            Kc, Vc = self.Kc[i], self.Vc[i]
            HK.qk_rope_kernel[(B,)](qkv, lw['qn'], lw['kn'], self.cos_t, self.sin_t, pos, q, Kc, Vc, slot,
                                    Kc.stride(0), Kc.stride(1), Kc.stride(2), H=Q.H, KVH=Q.KVH, HD=Q.HD, eps=Q.EPS)
            rbk = 8 if ieee else 32; ns = 1 if ieee else 2
            HK.attn_prefix_kernel[(Q.KVH, triton.cdiv(B, rbk))](q, self.Kp[i], self.Vp[i], self.mp, self.lpf, self.accp, B,
                                                                self.Lp, sc, H=Q.H, HD=Q.HD, RBK=rbk, BN=64 if not ieee else 32,
                                                                IEEE=ieee, num_warps=4, num_stages=ns)
            HK.attn_suffix_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.mp, self.lpf, self.accp, att,
                                              Kc.stride(0), Kc.stride(1), Kc.stride(2), sc, H=Q.H, HD=Q.HD, BN=64 if not ieee else 32,
                                              IEEE=ieee, num_warps=4, num_stages=1 if ieee else 2)
            o = F.linear(att.view(B, Q.H * Q.HD), lw['Wo'])
            HK.add_rms_kernel[(B,)](x, o, lw['ln2'], h, D=Q.D, HAS_DELTA=True, eps=Q.EPS)
            gu = F.linear(h, lw['Wgu'])
            act = self.actb[:B]
            HK.silu_mul_kernel[(B, Q.FF // 1024)](gu, act, FF=Q.FF, BLK=1024)
            delta = F.linear(act, lw['Wd'])
        x += delta
        cs = torch.stack(caps, 1)                                            # (B, nL, D)
        self.p_cap[r] = torch.where(fire[:, None, None], cs, self.p_cap[r])
        if write is not None:
            x += write.to(x.dtype)
        HK.add_rms_kernel[(B,)](x, x, W['lnf'], h, D=Q.D, HAS_DELTA=False, eps=Q.EPS)
        logits = F.linear(h, W['E'])
        nxt, lp = self.nxt_b[:B], self.lp_b[:B]
        HK.sample_kernel[(B,)](logits, self.temp, self.greedy, self.seed, nxt, lp, logits.shape[1], BV=2048, num_warps=8)
        self.seed += 1
        self._book(B, tok, nxt, lp, wf, imp, dec, None)

    def _wave(self, qs, G, T, max_new, greedy):
        self.p_hc.zero_(); self.p_got.zero_(); self.p_pos.fill_(-1); self.p_cap.zero_()
        st = super()._wave(qs, G, T, max_new, greedy)
        n = len(qs) * G
        self._caps.append(self.p_cap[:n].cpu()); self._poss.append(self.p_pos[:n].cpu())
        return st

    def generate(self, qids, **kw):
        self._caps, self._poss = [], []
        o = super().generate(qids, **kw)
        o['hcap'] = torch.cat(self._caps); o['hpos'] = torch.cat(self._poss)
        return o


def group_select(vals, valid, flag, hcap, w=None, b=0.0, G=8, sanity=True):
    """The selection implant's reduction, as tensor ops over the G rows of each question (no Python on data).
    vals (N,) float64 answer values; valid (N,) bool (a parsable '#### number'); flag (N,) int (0 ok, 1 neg, 2 non-int,
    3 missing); hcap (N, D) captured state at '####' (the probe's layer); w (D,), b: verifier probe (None = majority).
    Returns (Q,) index of the chosen row per question (-1 if no row has an answer)."""
    N = vals.shape[0]; Qn = N // G
    p = torch.ones(N, dtype=torch.float64, device=vals.device) if w is None else \
        torch.sigmoid(hcap.double() @ w.double() + b)
    v = vals.view(Qn, G); ok = valid.view(Qn, G); P = (p * valid).view(Qn, G)
    if sanity:                                             # drop impossible answers unless none is left
        sane = ok & (flag.view(Qn, G) == 0)
        keep = torch.where(sane.any(1, keepdim=True), sane, ok)
    else:
        keep = ok
    P = P * keep
    same = (v[:, :, None] - v[:, None, :]).abs() <= 1e-6 * torch.clamp(v[:, None, :].abs(), min=1.0)
    score = (same & keep[:, None, :]).double() @ P[:, :, None]          # (Q,G,1): weight of each row's answer
    score = torch.where(keep, score[..., 0], torch.full_like(P, -1.0))
    # earliest-first tie break: subtract a tiny rank term
    score = score - 1e-9 * torch.arange(G, device=vals.device, dtype=torch.float64)[None]
    pick = score.argmax(1)
    return torch.where(keep.any(1), pick, torch.full_like(pick, -1))
