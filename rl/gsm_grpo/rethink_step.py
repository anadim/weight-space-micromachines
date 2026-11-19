"""The compiled mistake-detector + rethink implant (micromachines/rethink_circuit_v2.py, gated 'hybrid' configuration with
the fitted detector results/rethink_v2/detector_hybrid.pt, cap 300) as a one-token decode step with static per-row state,
so it runs inside the engine's CUDA graph after the calculator.

Same math as RethinkCircuitV2.forward for n=1; every "value at the most recent event" hard-attention head over the whole
history is kept as a per-row scalar register (exact: the head's softmax is one-hot in fp32 for sequences < 3,333 tokens).
State at the first decode position is all zeros: the prompt ends with the key "\\nAnswer:", so the most recent event of
every head is that key (value 0), and nothing fires inside the prompt (checked by `prefill_check`).

Inputs per step: the token at this position, the host's pre-norm hidden state (calculator write included), the host
logits from that state (the circuit's LM-head readout uses a copy of the same rows), and the residual stream after
layers 11,13,...,23 (= hidden_states[12..24], the detector's read layers)."""
import os, sys, torch, torch.nn.functional as Fn
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WSM_ROOT', os.path.dirname(os.path.dirname(HERE)))   # repo root
sys.path.insert(0, REPO)
from micromachines.rethink_circuit import step_ge, step_gt, AND, BIG, K_END        # noqa: E402
from micromachines.rethink_circuit_v2 import RethinkCircuitV2, K_DET              # noqa: E402
from micromachines.rethink_circuit import PHRASE as PHRASE_TXT                       # noqa: E402

DET = os.path.join(REPO, 'results', 'rethink_v2', 'detector_hybrid.pt')     # fitted mistake detector (released)
DIRS = os.path.join(REPO, 'results', 'rethink_v2', 'write_dirs_all.pt')      # solved rethink write directions (released)


class RethinkStep:
    def __init__(self, hf_model, tok, B_max, mode='gated', intervene='hybrid', cap=300):
        rc = RethinkCircuitV2(hf_model, tok, mode=mode, detector=DET if mode == 'gated' else None, intervene=intervene,
                              cap=cap, dirs=DIRS)
        rc.detach()
        self.rc = rc; dev = rc.dev; self.dev = dev
        self.mode, self.intervene, self.cap = mode, intervene, cap
        self.tbl = rc.tbl
        V = rc.tbl.shape[0]
        ix = lambda l: torch.tensor(l, device=dev, dtype=torch.long)
        if not hasattr(rc, 'idx_A'):
            raise RuntimeError('RethinkCircuitV2 lacks idx_A/idx_N/idx_H: use the repo version of micromachines/rethink_circuit_v2.py '
                               '(adds one line to micromachines/rethink_circuit_v2.py that keeps the index lists)')
        self.iA, self.iN, self.iH = ix(rc.idx_A), ix(rc.idx_N), ix(rc.idx_H)
        mS = torch.zeros(V, dtype=torch.bool, device=dev); mS[self.iA] = True; mS[self.iN] = True; mS[self.iH] = True
        self.maskS = mS
        self.WT, self.R, self.L = rc.WT, rc.R, rc.L
        self.det_layers = rc.det_layers
        z = lambda *s: torch.zeros(*s, device=dev)
        self.fl_tail = z(B_max, 2, 4)
        self.s_first, self.s_gate, self.s_cand, self.s_fire, self.s_pf = (z(B_max) for _ in range(5))
        self.frow_tail = z(B_max, self.L - 1, self.R)
        self.last_w = None
        # ---- co-training support (arm 4c): detector weights as parameters (same storage the graph reads),
        # stochastic fire decisions in rollouts (off = the hard threshold, bit-identical), per-row records by rowid
        if mode == 'gated':
            for k in ('W1', 'b1', 'WB', 'WA', 'W0'):
                if getattr(rc, k, None) is not None and not isinstance(getattr(rc, k), torch.nn.Parameter):
                    setattr(rc, k, torch.nn.Parameter(getattr(rc, k).to(dev), requires_grad=False))
            P = torch.load(DET, map_location=dev)
            if not hasattr(rc, 'W0') or rc.W0 is None:
                rc.W0 = torch.nn.Parameter(P['W0'].double().to(dev).reshape(-1), requires_grad=False)
            rc.b0 = float(P['b0'])
        self.stoch = torch.zeros((), device=dev, dtype=torch.bool); self.temp = torch.ones((), device=dev)
        nf = 1024 * len(self.det_layers)
        self.rec_feat = z(B_max + 1, nf); self.rec_has = z(B_max + 1); self.rec_dB = z(B_max + 1); self.rec_dA = z(B_max + 1)
        self.rec_sB = z(B_max + 1); self.rec_s0 = z(B_max + 1)
        self.passive = False
        # ---- arm 4s (soft rethink): which phrase position each forced (sequence, offset) is; -1 = not a phrase token
        ph = rc.T(PHRASE_TXT); self.K = len(ph); self.phrase_ids = ph
        Ms = torch.full((self.R, self.L), -1.0)
        for r, sq in enumerate(rc.seqs):
            if len(sq) >= self.K and sq[-self.K:] == ph:
                for j in range(len(sq) - self.K, len(sq)): Ms[r, j] = j - (len(sq) - self.K)
        self.Msoft = (Ms + 1).to(dev)                       # +1 so that "none" = 0
        self.last_sidx = None

    def reset_rows(self, rows, last2_tokens):
        """Decode start for these rows: zero registers; the flags of the last two prompt tokens."""
        for t in (self.s_first, self.s_gate, self.s_cand, self.s_fire, self.s_pf): t[rows] = 0
        self.frow_tail[rows] = 0
        self.fl_tail[rows] = self.tbl[last2_tokens]

    def gather(self, idx, n):
        for t in (self.fl_tail, self.s_first, self.s_gate, self.s_cand, self.s_fire, self.s_pf, self.frow_tail):
            t[:n] = t[idx]

    def detector(self, feats, s=None):
        rc = self.rc
        if s is None: s = torch.cat([feats[l].float() for l in self.det_layers], -1)
        z = (s - rc.mu) / rc.sd
        z = torch.relu(z @ rc.W1.T + rc.b1)
        if self.intervene == 'hybrid':
            return z.double() @ rc.WB + rc.bB, z.double() @ rc.WA + rc.bA
        return z.double() @ rc.W2 + rc.b2, None

    def heads(self, s):
        """Differentiable heads from stored features s (N, 7168): sB, sA, s0 (P(correct now) logit)."""
        rc = self.rc
        z = torch.relu(((s - rc.mu) / rc.sd) @ rc.W1.T + rc.b1).double()
        return z @ rc.WB + rc.bB, z @ rc.WA + rc.bA, z @ rc.W0 + rc.b0

    def params(self):
        rc = self.rc
        return [rc.W1, rc.b1, rc.WB, rc.WA, rc.W0]

    def reset_records(self):
        for t in (self.rec_has, self.rec_dB, self.rec_dA, self.rec_sB, self.rec_s0): t.zero_()

    def step(self, tok, logits, feats, pos, B, rowid=None, live=None):
        """tok (B,) token at this position; logits (B,V) host logits before this implant's write (fp32 view of bf16);
        feats {layer: (B,D)}; pos (B,) absolute positions. Returns write (B,D) fp32 (zero rows where silent)."""
        rc = self.rc
        fl = self.tbl[tok]                                                      # (B,4)
        prev = self.fl_tail[:B]
        key = AND(fl[:, 0], prev[:, 1, 1], prev[:, 0, 2])
        hsh, endsnl = fl[:, 3], fl[:, 2]
        lg = logits.float()
        zA, zN, zH = lg[:, self.iA], lg[:, self.iN], lg[:, self.iH]
        mC = lg.masked_fill(self.maskS, float('-inf')).amax(-1)
        negb = torch.full_like(mC, -BIG)
        mA = zA.amax(-1) if zA.shape[1] else negb
        mN = zN.amax(-1) if zN.shape[1] else negb
        mH = zH.amax(-1)
        hn = step_gt(mH - torch.maximum(torch.maximum(mC, mA), mN), K_END)
        if self.mode == 'gated':
            sfeat = torch.cat([feats[l].float() for l in self.det_layers], -1)
            sB, sA = self.detector(feats, sfeat)
            if self.intervene == 'hybrid':
                pasB = step_gt(sB - rc.thB, K_DET).float(); pasA = step_gt(sA - rc.thA, K_DET).float()
                # arm 4c rollouts: Bernoulli(sigmoid((score - th)/T)) in place of the hard threshold
                uB = torch.rand_like(pasB); uA = torch.rand_like(pasA)
                pasB = torch.where(self.stoch, (uB < torch.sigmoid((sB - rc.thB) / self.temp)).float(), pasB)
                pasA = torch.where(self.stoch, (uA < torch.sigmoid((sA - rc.thA) / self.temp)).float(), pasA)
                pas = pasB
            else:
                pas = step_ge(sB - rc.thr, K_DET).float()
        else:
            pas = torch.ones_like(key)
        # head A: first read point (value of HN at the most recent {key, hn} strictly before t)
        first = AND(hn, 1 - self.s_first[:B])
        if rowid is not None and self.mode == 'gated':                       # record the decision at the read point
            rec = (first > 0.5) & live
            r = rowid
            self.rec_feat[r] = torch.where(rec[:, None], sfeat, self.rec_feat[r])
            self.rec_has[r] = torch.where(rec, torch.ones_like(first), self.rec_has[r])
            self.rec_dB[r] = torch.where(rec, pasB, self.rec_dB[r])
            self.rec_dA[r] = torch.where(rec, pasA, self.rec_dA[r])
            self.rec_sB[r] = torch.where(rec, sB.float(), self.rec_sB[r])
            s0 = (torch.relu(((sfeat - rc.mu) / rc.sd) @ rc.W1.T + rc.b1).double() @ rc.W0 + rc.b0).float()
            self.rec_s0[r] = torch.where(rec, s0, self.rec_s0[r])
        ev = torch.clamp(key + hn, max=1)
        self.s_first[:B] = torch.where(ev > 0, hn, self.s_first[:B])
        frow = torch.zeros(B, self.R, device=self.dev)
        if self.intervene in ('hash', 'hybrid'):
            fire = AND(first, pas)
            frow[:, rc.r_ph_nl] += fire * endsnl; frow[:, rc.r_ph] += fire * (1 - endsnl)
            if self.cap:
                pf_new = fire * (pos.float() + rc.len_ph_nl * endsnl + rc.len_ph * (1 - endsnl))
                kk = torch.clamp(key + fire + hsh, max=1)
                self.s_fire[:B] = torch.where(kk > 0, fire, self.s_fire[:B])
                self.s_pf[:B] = torch.where(kk > 0, pf_new, self.s_pf[:B])
                fired, pf = self.s_fire[:B], self.s_pf[:B]
                d = pos.float() - pf - self.cap
                eq = step_ge(d, 1e3) - step_ge(d - 1, 1e3)
                capf = AND(fired, eq)
                frow[:, rc.r_cap_nl] += capf * endsnl; frow[:, rc.r_cap] += capf * (1 - endsnl)
        if self.intervene in ('end', 'hybrid'):
            pasE = pasA * (1 - pasB) if self.intervene == 'hybrid' else pas
            e = torch.maximum(mA, mN - BIG * (1 - endsnl))
            cmax = torch.maximum(torch.maximum(mC, mH), mN - BIG * endsnl)
            end = step_gt(e - cmax, K_END)
            sE = torch.cat([zA, zN - BIG * (1 - endsnl)[:, None]], -1)
            oh = step_ge(sE - e[:, None], K_END)
            row = torch.relu(oh @ rc.M_any - endsnl[:, None]) + torch.relu(oh @ rc.M_nl - (1 - endsnl)[:, None])
            row = Fn.pad(row, (0, self.R - row.shape[-1]))
            evg = torch.clamp(key + first, max=1)                              # head B (at or before t)
            self.s_gate[:B] = torch.where(evg > 0, first * pasE, self.s_gate[:B])
            cand = AND(end, self.s_gate[:B])
            fireA = AND(cand, 1 - self.s_cand[:B])                             # head C (strictly before t)
            evc = torch.clamp(key + cand, max=1)
            self.s_cand[:B] = torch.where(evc > 0, cand, self.s_cand[:B])
            frow = frow + row * fireA[:, None]
        w = frow @ self.WT[0]
        tail = self.frow_tail[:B]
        sid = frow @ self.Msoft[:, 0]                                            # phrase index (+1) of the forced token
        for j in range(1, self.L):
            sid = sid + tail[:, self.L - 1 - j] @ self.Msoft[:, j]
        si = sid.round().long() - 1
        self.last_sidx = torch.where((si >= 0) & (si < self.K), si, torch.full_like(si, -1))   # overlap/none -> -1
        for j in range(1, self.L):
            w = w + tail[:, self.L - 1 - j] @ self.WT[j]
        self.frow_tail[:B] = torch.cat([tail[:, 1:], frow[:, None]], 1)
        self.fl_tail[:B] = torch.cat([prev[:, 1:], fl[:, None]], 1)
        if self.passive:                                                     # read-only mode (AUROC evals)
            w = torch.zeros_like(w); self.last_sidx = torch.full_like(self.last_sidx, -1)
        self.last_w = w
        return w
