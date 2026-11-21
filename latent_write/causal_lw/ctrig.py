"""Compiled trigger for the causal latent write (task [5] caveat): a token DFA (fixed table TT[token, state], as the serial
stack's '#### number' reader) plus 4 integer registers, run over the host's token stream (prompt AND generated tokens, one
token per step). It fires at the closing token ')+' of '(a*b)+' and at the tail tokens '?\n' 'A' ':' ' ' after c, and
emits (slot, P) where slot indexes the trained side map row and P = a*b comes from a fixed product table MUL[ra, rb].

registers: s (DFA state), ra, rb (operand digits: r <- 10 r + DIG[token], a fixed affine op selected by the state), P (product
latched at ')+'), n (saturating count of '\\n\\n' tokens = segment counter, used only by the gated variant).
gates (build-time constants, i.e. part of the table): colon = '(' only opens after ':' (answer expressions 'A: (a*b)+c');
nsep = fire only when n >= nsep (after the few-shot block; reproduces the old Python query-only positions exactly).

Pure python (no numpy/torch) so it fuzzes on the sandbox; cw_ct.py loads the same tables into torch on the node.
"""
IDLE, OPEN, A1, A2, A3, STAR, B1, B2, CLOSE, C1, C2, C3, T1, T2, T3, T4, COLON = range(17)
NSTATE = 17
# token classes
K_OTHER, K_OPEN, K_DIG, K_STAR, K_CLOSE, K_QNL, K_A, K_COLON, K_SPACE, K_SEP = range(10)
NCLS = 10
TAIL_SLOT = {T1: 4, T2: 3, T3: 2, T4: 1}     # Wt row order of cw.py: [')+', ' ', ':', 'A', '?\n'] (FE = [7, 1, 2, 3, 4])


def tok_class(s):
    if s in (" (", "("): return K_OPEN
    if len(s) == 1 and s in "0123456789": return K_DIG
    if s == "*": return K_STAR
    if s == ")+": return K_CLOSE
    if s == "?\n": return K_QNL
    if s == "A": return K_A
    if s == ":": return K_COLON
    if s == " ": return K_SPACE
    if s == "\n\n": return K_SEP
    return K_OTHER


def build(colon=False, tail=True):
    """class-level table TC[cls][state] -> next state (the token table is TT[v][s] = TC[cls[v]][s])."""
    TC = [[IDLE] * NSTATE for _ in range(NCLS)]
    for s in range(NSTATE):
        TC[K_OPEN][s] = OPEN if (not colon or s in (COLON, T3)) else IDLE
        TC[K_COLON][s] = T3 if (tail and s == T2) else COLON
        TC[K_DIG][s] = {OPEN: A1, A1: A2, A2: A3, STAR: B1, B1: B2, CLOSE: C1, C1: C2, C2: C3}.get(s, IDLE)
        TC[K_STAR][s] = STAR if s in (A1, A2, A3) else IDLE
        TC[K_CLOSE][s] = CLOSE if s in (B1, B2) else IDLE
        if tail:
            TC[K_QNL][s] = T1 if s in (C1, C2, C3) else IDLE
            TC[K_A][s] = T2 if s == T1 else IDLE
            TC[K_SPACE][s] = T4 if s == T3 else IDLE
    # register ops per state entered: 0 hold, 1 ra <- d, 2 ra <- 10 ra + d, 3 rb <- d, 4 rb <- 10 rb + d
    ROP = [0] * NSTATE
    ROP[A1], ROP[A2], ROP[A3], ROP[B1], ROP[B2] = 1, 2, 2, 3, 4
    FSLOT = [-1] * NSTATE
    FSLOT[CLOSE] = 0
    if tail:
        for s, k in TAIL_SLOT.items(): FSLOT[s] = k
    return TC, ROP, FSLOT


def vocab_tables(txt):
    """txt: list of token strings (index = token id) -> CLS[v], DIG[v] (digit value or 0)."""
    CLS = [tok_class(s) for s in txt]
    DIG = [int(s) if c == K_DIG else 0 for s, c in zip(txt, CLS)]
    return CLS, DIG


def mul_valid(ra, rb):
    """fixed product table MUL[ra, rb] (ra < 1000, rb < 100) and validity table VALID[P] = 100 <= P <= 999."""
    P = ra * rb
    return P, 100 <= P <= 999


class Trig:
    """table-driven scan (the reference implementation of the compiled circuit, one token per step)."""
    def __init__(self, txt, colon=False, tail=True, nsep=0):
        self.TC, self.ROP, self.FSLOT = build(colon, tail)
        self.CLS, self.DIG = vocab_tables(txt)
        self.nsep = nsep
        self.reset()

    def reset(self):
        self.s = IDLE; self.ra = self.rb = self.P = self.n = 0

    def step(self, t):
        c = self.CLS[t]; d = self.DIG[t]
        s = self.TC[c][self.s]
        op = self.ROP[s]
        if op == 1: self.ra = d
        elif op == 2: self.ra = 10 * self.ra + d
        elif op == 3: self.rb = d
        elif op == 4: self.rb = 10 * self.rb + d
        if s == CLOSE:
            P, ok = mul_valid(self.ra, self.rb)
            if ok: self.P = P
            else: s = IDLE
        self.n = min(self.n + (c == K_SEP), max(self.nsep, 1))
        self.s = s
        slot = self.FSLOT[s]
        if slot >= 0 and self.n < self.nsep: slot = -1
        return slot, (self.P if slot >= 0 else 0)

    def scan(self, ids):
        self.reset()
        return [self.step(t) for t in ids]


def ref_find(toks, colon=False, tail=True, nsep=0):
    """independent Python reference (pattern search over token strings, not a DFA): list of (pos, slot, P)."""
    isd = lambda s: len(s) == 1 and s in "0123456789"
    out = []
    for j, t in enumerate(toks):
        if t != ")+": continue
        i = j - 1; bd = ""
        while i >= 0 and isd(toks[i]): bd = toks[i] + bd; i -= 1
        if not 1 <= len(bd) <= 2 or i < 0 or toks[i] != "*": continue
        i -= 1; ad = ""
        while i >= 0 and isd(toks[i]): ad = toks[i] + ad; i -= 1
        if not 1 <= len(ad) <= 3 or i < 0 or toks[i] not in (" (", "("): continue
        if colon and not (i >= 1 and toks[i - 1] == ":"): continue
        P = int(ad) * int(bd)
        if not 100 <= P <= 999: continue
        hits = [(j, 0)]
        if tail:
            k = j + 1; m = 0
            while k < len(toks) and isd(toks[k]): k += 1; m += 1
            if 1 <= m <= 3:
                for q, (want, slot) in enumerate([("?\n", 4), ("A", 3), (":", 2), (" ", 1)]):
                    if k + q < len(toks) and toks[k + q] == want: hits.append((k + q, slot))
                    else: break
        for p, slot in hits:
            if toks[:p + 1].count("\n\n") >= nsep: out.append((p, slot, P))
    return out


def dfa_find(trig, ids):
    return [(p, s, P) for p, (s, P) in enumerate(trig.scan(ids)) if s >= 0]
