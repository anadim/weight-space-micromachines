"""Learned per-attempt vote weights w = sigmoid(theta . f) (listwise: softmax over answer clusters of the weighted vote)
and pointwise (logistic regression of attempt ok).  numpy + scipy L-BFGS."""
import numpy as np
from scipy.optimize import minimize


def _pack(streams):
    """pad to 8 slots: F (S,8,d), K keep, M same&keep (S,8,8), R rep, G gold rep slot one-hot, informative mask"""
    S = len(streams); d = streams[0].F.shape[1]
    F = np.zeros((S, 8, d)); K = np.zeros((S, 8)); M = np.zeros((S, 8, 8)); R = np.zeros((S, 8), bool)
    G = np.zeros((S, 8)); inf = np.zeros(S, bool)
    for s, st in enumerate(streams):
        n = len(st.S)
        if n == 0: continue
        F[s, :n] = st.F; K[s, :n] = st.keep
        M[s, :n, :n] = st.same & st.keep[None, :]
        R[s, :n] = st.rep
        g = st.rep & (st.ok > 0)
        if g.any() and st.rep.sum() > 1:
            G[s, :n] = g; inf[s] = True
    return F[inf], K[inf], M[inf], R[inf], G[inf]


class Listwise:
    def __init__(self, l2=1e-2, beta=None, kind='sig'):
        self.l2 = l2; self.beta = 1.0 if kind == 'exp' else beta; self.kind = kind

    def fit(self, streams):
        F, K, M, R, G = _pack(streams)
        self.mu = F.reshape(-1, F.shape[2])[K.reshape(-1) > 0].mean(0)
        self.sd = F.reshape(-1, F.shape[2])[K.reshape(-1) > 0].std(0) + 1e-6
        Z = (F - self.mu) / self.sd
        d = Z.shape[2]; l2 = self.l2; fixb = self.beta; ex = self.kind == 'exp'

        def fg(p):
            th, b0 = p[:d], p[d]
            lb = p[d + 1] if fixb is None else np.log(fixb)
            beta = np.exp(lb)
            z = Z @ th + b0
            w = np.exp(np.clip(z, -30, 30)) if ex else 1 / (1 + np.exp(-z))   # (S,8)
            sc = np.einsum('sij,sj->si', M, w * K)                      # cluster score at every slot
            lo = np.where(R, beta * sc, -1e9)
            lo = lo - lo.max(1, keepdims=True)
            P = np.exp(lo) * R; P /= P.sum(1, keepdims=True)
            L = -(np.log((P * G).sum(1) + 1e-300)).sum() / len(Z)
            dsc = beta * (P - G) / len(Z)                                # dL/dsc (S,8)
            dw = np.einsum('si,sij->sj', dsc, M) * K
            dz = dw * w if ex else dw * w * (1 - w)
            gth = np.einsum('sj,sjd->d', dz, Z) + 2 * l2 * th
            gb0 = dz.sum()
            glb = ((P - G) * sc * R).sum() / len(Z) * beta
            L += l2 * (th ** 2).sum()
            g = np.concatenate([gth, [gb0]] + ([[glb]] if fixb is None else []))
            return L, g
        p0 = np.zeros(d + (2 if fixb is None else 1))
        if fixb is None: p0[-1] = np.log(4.0)
        r = minimize(fg, p0, jac=True, method='L-BFGS-B', options=dict(maxiter=2000))
        self.th = r.x[:d] / self.sd; self.b0 = r.x[d] - (self.mu * r.x[:d] / self.sd).sum()
        self.beta_ = np.exp(r.x[d + 1]) if fixb is None else fixb
        self.loss = r.fun
        return self

    def w(self, st):
        if not len(st.S): return np.zeros(0)
        z = st.F @ self.th + self.b0
        return np.exp(np.clip(z, -30, 30)) if self.kind == 'exp' else 1 / (1 + np.exp(-z))


class Pointwise:
    """logistic regression of slot ok on features (kept slots); weight = predicted probability"""
    def __init__(self, l2=1e-2): self.l2 = l2

    def fit(self, streams):
        X = np.concatenate([st.F[st.keep] for st in streams if len(st.S)])
        y = np.concatenate([st.ok[st.keep] for st in streams if len(st.S)])
        mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; l2 = self.l2

        def fg(p):
            z = Z @ p[:-1] + p[-1]; q = 1 / (1 + np.exp(-z))
            L = -(y * np.log(q + 1e-12) + (1 - y) * np.log(1 - q + 1e-12)).mean() + l2 * (p[:-1] ** 2).sum()
            g = np.concatenate([Z.T @ (q - y) / len(y) + 2 * l2 * p[:-1], [(q - y).mean()]])
            return L, g
        r = minimize(fg, np.zeros(Z.shape[1] + 1), jac=True, method='L-BFGS-B')
        self.th = r.x[:-1] / sd; self.b0 = r.x[-1] - (mu * r.x[:-1] / sd).sum()
        return self

    def w(self, st):
        return 1 / (1 + np.exp(-(st.F @ self.th + self.b0))) if len(st.S) else np.zeros(0)


class ClusterLin:
    """answer-cluster scorer: s(v) = theta . g(v), g = [count/8, log count, sum sigmoid(lg), max lg, mean f, max f, min f]
    over the kept slots of the cluster; conditional logit over clusters (convex). In-graph: same-matrix sums / masked max."""
    def __init__(self, l2=1e-3, lgcol=0): self.l2 = l2; self.lgcol = lgcol

    def agg(self, st):
        n = len(st.S)
        if n == 0: return np.zeros((0, 1))
        M = (st.same & st.keep[None, :]).astype(float)
        c = M.sum(1, keepdims=True); c1 = np.maximum(c, 1)
        F = st.F; lg = F[:, self.lgcol]
        big = np.where(M[:, :, None] > 0, F[None], -1e9).max(1)
        small = np.where(M[:, :, None] > 0, F[None], 1e9).min(1)
        g = np.concatenate([c / 8, np.log(c1), M @ (1 / (1 + np.exp(-lg)))[:, None], (M @ F) / c1, big, small], 1)
        return np.where(st.keep[:, None], g, 0)

    def fit(self, streams):
        Gs, Rs, Ys = [], [], []
        for st in streams:
            if not len(st.S): continue
            g = st.rep & (st.ok > 0)
            if not g.any() or st.rep.sum() < 2: continue
            Gs.append(self.agg(st)[st.rep]); Ys.append(g[st.rep])
        X = np.concatenate(Gs); mu = X.mean(0); sd = X.std(0) + 1e-6
        Gz = [(x - mu) / sd for x in Gs]
        off = np.cumsum([0] + [len(x) for x in Gz]); Z = np.concatenate(Gz); Y = np.concatenate(Ys).astype(float)
        seg = np.repeat(np.arange(len(Gz)), [len(x) for x in Gz]); S = len(Gz); l2 = self.l2

        def fg(th):
            z = Z @ th
            m = np.full(S, -1e18); np.maximum.at(m, seg, z)
            e = np.exp(z - m[seg]); den = np.bincount(seg, e, S)
            p = e / den[seg]
            L = -(np.log(np.bincount(seg, p * Y, S) + 1e-300)).mean() + l2 * (th ** 2).sum()
            # gradient: sum over clusters (p - y/ P(gold)) ... with single gold cluster: (p - Y) per row
            g = Z.T @ (p - Y) / S + 2 * l2 * th
            return L, g
        r = minimize(fg, np.zeros(Z.shape[1]), jac=True, method='L-BFGS-B', options=dict(maxiter=3000))
        self.th = r.x / sd; self.c0 = -(mu * r.x / sd).sum()
        return self

    def score(self, st):
        return self.agg(st) @ self.th + self.c0


def pick_cluster(model, st):
    if not st.keep.any(): return None
    sc = model.score(st)
    sc = np.where(st.keep, sc - 1e-9 * np.arange(len(sc)), -1e18)
    return int(np.argmax(sc))
