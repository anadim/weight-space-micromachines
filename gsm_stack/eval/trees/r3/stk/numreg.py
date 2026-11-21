"""Question-number / body-number register for the vfeat voter (SER_VF=1).

A token-level automaton that reads the attempt body (host tokens before '####') and extracts the numbers exactly as
voter/feats.text_numbers does on the decoded text:  re '\\d+(?:\\.\\d+)?' after removing thousands commas
'(?<=\\d),(?=\\d{3}(?!\\d))'.  The Qwen tokenizer emits every digit as its own token, so a token class table
(digit d / ',' / '.' / other) is enough.  Values are mantissa M and decimal count E (value = M / 10^E, exact for
equal decimals).  Each extracted number is inserted into a per-row distinct set (capacity K).
All ops are batched torch (same code on GPU in the engine and on CPU in the tests).

States: 0 idle, 1 integer digits, 2 after '.', 3 fraction digits, 4 after ',' (pc pending digits P, pfr = comma inside fraction).
"""
import torch


class NumReg:
    def __init__(self, Bp, K, dev):
        z = lambda *s, dt=torch.long: torch.zeros(*s, device=dev, dtype=dt)
        self.K = K
        self.st, self.E, self.pc = z(Bp), z(Bp), z(Bp)
        self.M, self.P = z(Bp, dt=torch.float64), z(Bp, dt=torch.float64)
        self.pfr = z(Bp, dt=torch.bool)
        self.S = z(Bp, K, dt=torch.float64); self.n = z(Bp)
        self.ar = torch.arange(K, device=dev)[None]

    def tensors(self):
        return [self.st, self.E, self.pc, self.M, self.P, self.pfr, self.S, self.n]

    def reset(self, r, m):
        for t in (self.st, self.E, self.pc, self.n):
            t[r] = torch.where(m, torch.zeros_like(t[r]), t[r])
        for t in (self.M, self.P):
            t[r] = torch.where(m, torch.zeros_like(t[r]), t[r])
        self.pfr[r] = self.pfr[r] & ~m

    def _insert(self, r, emit, v):
        S, n = self.S[r], self.n[r]
        have = ((S == v[:, None]) & (self.ar < n[:, None])).any(1)
        ins = emit & ~have & (n < self.K)
        oh = (self.ar == n[:, None]) & ins[:, None]
        self.S[r] = torch.where(oh, v[:, None], S)
        self.n[r] = n + ins.long()

    def step(self, r, c, d, act):
        """r: row index (B,), c: token class (B,) 3 digit / 4 ',' / 5 '.' / other, d: digit value (float64), act: rows reading"""
        st, M, E, P, pc, pfr = self.st[r], self.M[r], self.E[r], self.P[r], self.pc[r], self.pfr[r]
        dg, cm, dt = c == 3, c == 4, c == 5
        ot = ~(dg | cm | dt)
        s0, s1, s2, s3, s4 = st == 0, st == 1, st == 2, st == 3, st == 4
        full = pc == 3; pz = pc == 0; pp = ~full & ~pz
        p10 = torch.pow(torch.full_like(M, 10.0), E.double())
        cur = M / p10                                                    # value of the current number
        Mm = M * 1000 + P; Em = E + torch.where(pfr, 3, 0)              # merged across a removed thousands comma
        mer = Mm / torch.pow(torch.full_like(M, 10.0), Em.double())
        # ---- emits
        e1 = (s1 & ot) | (s2 & ~dg) | (s3 & (dt | ot)) | (s4 & ((dg & full) | (cm & ~full) | (dt & (~full | pfr)) | ot))
        v1 = torch.where(s4 & full & ((dt & pfr) | ot), mer, cur)
        e2 = s4 & pp & ot
        # ---- next state
        nst, nM, nE, nP, npc, npfr = st.clone(), M.clone(), E.clone(), P.clone(), pc.clone(), pfr.clone()
        dd = d
        # st0
        a = s0 & dg; nst = torch.where(a, 1, nst); nM = torch.where(a, dd, nM); nE = torch.where(a, 0, nE)
        # st1
        a = s1 & dg; nM = torch.where(a, M * 10 + dd, nM)
        nst = torch.where(s1 & dt, 2, nst)
        a = s1 & cm; nst = torch.where(a, 4, nst); nP = torch.where(a, 0.0, nP); npc = torch.where(a, 0, npc); npfr = torch.where(a, False, npfr)
        nst = torch.where(s1 & ot, 0, nst)
        # st2
        a = s2 & dg; nst = torch.where(a, 3, nst); nM = torch.where(a, M * 10 + dd, nM); nE = torch.where(a, E + 1, nE)
        nst = torch.where(s2 & ~dg, 0, nst)
        # st3
        a = s3 & dg; nM = torch.where(a, M * 10 + dd, nM); nE = torch.where(a, E + 1, nE)
        nst = torch.where(s3 & (dt | ot), 0, nst)
        a = s3 & cm; nst = torch.where(a, 4, nst); nP = torch.where(a, 0.0, nP); npc = torch.where(a, 0, npc); npfr = torch.where(a, True, npfr)
        # st4
        a = s4 & dg & ~full; nP = torch.where(a, P * 10 + dd, nP); npc = torch.where(a, pc + 1, npc)
        a = s4 & dg & full; nst = torch.where(a, 1, nst); nM = torch.where(a, P * 10 + dd, nM); nE = torch.where(a, 0, nE)
        a = s4 & cm & full; nM = torch.where(a, Mm, nM); nE = torch.where(a, Em, nE); nP = torch.where(a, 0.0, nP); npc = torch.where(a, 0, npc)
        a = s4 & cm & pp; nM = torch.where(a, P, nM); nE = torch.where(a, 0, nE); nP = torch.where(a, 0.0, nP); npc = torch.where(a, 0, npc); npfr = torch.where(a, False, npfr)
        nst = torch.where(s4 & cm & pz, 0, nst)
        a = s4 & dt & full & ~pfr; nst = torch.where(a, 2, nst); nM = torch.where(a, Mm, nM); nE = torch.where(a, Em, nE)
        nst = torch.where(s4 & dt & full & pfr, 0, nst)
        a = s4 & dt & pp; nst = torch.where(a, 2, nst); nM = torch.where(a, P, nM); nE = torch.where(a, 0, nE)
        nst = torch.where(s4 & dt & pz, 0, nst)
        nst = torch.where(s4 & ot, 0, nst)
        # ---- commit (only reading rows)
        self.st[r] = torch.where(act, nst, st); self.M[r] = torch.where(act, nM, M); self.E[r] = torch.where(act, nE, E)
        self.P[r] = torch.where(act, nP, P); self.pc[r] = torch.where(act, npc, pc); self.pfr[r] = torch.where(act, npfr, pfr)
        self._insert(r, act & e1, v1)
        self._insert(r, act & e2, P)

    def flush(self, r, act):
        z = torch.zeros_like(r)
        self.step(r, z, z.double(), act)

    def run(self, r, C, D, L):
        """read whole sequences: C, D (B, T) classes/digits, L (B,) lengths; then flush"""
        for t in range(C.shape[1]):
            self.step(r, C[:, t], D[:, t], t < L)
        self.flush(r, torch.ones_like(L, dtype=torch.bool))


def token_tables(txt):
    """class / digit tables for a decoded vocabulary list (same rule as serial_stack r_cls)"""
    cls, dig = [], []
    for t in txt:
        c = 3 if (len(t) == 1 and t in '0123456789') else 4 if t == ',' else 5 if t == '.' else 0
        cls.append(c); dig.append(int(t) if c == 3 else 0)
    return cls, dig


def ref_numbers(cls, dig):
    """pure-python reference: decode the class stream to a string and apply the offline regexes"""
    import re
    from fractions import Fraction
    s = ''.join(str(d) if c == 3 else ',' if c == 4 else '.' if c == 5 else ' ' for c, d in zip(cls, dig))
    s = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', s)
    return {Fraction(m.group(0)) for m in re.finditer(r'\d+(?:\.\d+)?', s)}
