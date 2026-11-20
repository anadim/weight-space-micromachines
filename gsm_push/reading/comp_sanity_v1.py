"""Compiled, forward-only answer-sanity implant for the RL engine (one generation call, no rewind, no restart).

Per decode step, from the INPUT token only (the token the host wrote last), with fixed tensors:
  reader   CLS[tok]            gather of a fixed row: token -> one of 26 token classes (tokens with identical behaviour
                               on the 34-state '#### number' DFA of dig/sanity/san_engine.py)
  DFA      s' = s @ TC[CLS]    one-hot state register times a fixed 0/1 transition matrix (34 x 34 per class)
  certain  cN = s' . CN, cF = s' . CF   0/1 rows over states: every continuation of the text gives a NEGATIVE (CN) or a
                               NON-INTEGER (CF) value for the scorer's first '#### number' (computed by exhaustive
                               reachability over the char DFA; checked against the regex by fuzzing, see fuzz())
  gates    fire = relu(c + idle + under + room + on - 4)  on 0/1 wires (idle: pointer register at 0; under: fires < maxf;
                               room: position counter below the budget; on: implant switch)
  writer   pointer register p (one-hot over the phrase rows of both banks + idle):  p' = p @ SHIFT, or START_bank on fire;
           write = G * p'[1:] @ DIR  added to the host's residual before the final norm (DIR[k] = solved direction whose
           readout argmax is phrase token k); the host's own sampler then emits the phrase token.
  reset    s' = S0 on fire (the next '####' after the phrase is read fresh).
Silent rows: write is exactly 0 and nothing else changes -> bit-identical to the engine without the implant.
Disclosed non-weight pieces: the decode loop and its per-row token budget (+EXTRA tokens per fire, as the sanity engine);
registers are stored as indices between steps (argmax of the one-hot) for the engine's row compaction.
Not covered: a MISSING answer (needs reading the host's output stop token; 8 of 76 sanity flags on split B)."""
import os, sys, random, re, torch, torch.nn.functional as F
sys.path.insert(0, os.environ.get('SAN_DIR', os.path.dirname(os.path.abspath(__file__))))
import san_engine as SE
from san_engine import SanEngine

PHRASES = {
    'wntf': {1: "\nWait, that can't be right. We need to find", 2: "\nWait, that can't be right. We need to find"},
    'reread': {1: "\nHmm, wait. Let me read the question again. We need to find", 2: "\nHmm, wait. Let me read the question again. We need to find"},
    'typed': {1: "\nWait, the answer can't be negative. We need to find", 2: "\nWait, the answer should be a whole number. We need to find"},
    'rethink': {1: "\nHmm, wait. Let me rethink this.\n", 2: "\nHmm, wait. Let me rethink this.\n"},
}
ALPHA = ['#', '-', '.', ',', '0', '1', ' ', '\n', 'a', '$']


def certain_sets(S):
    """CN / CF: states from which EVERY continuation ends with scorer class 1 (negative) / 2 (non-integer)."""
    ix = {s: i for i, s in enumerate(S)}
    reach = []
    for s in S:
        seen, st = {s}, [s]
        while st:
            u = st.pop()
            for c in ALPHA:
                v = SE.char_step(u, c)
                if v not in seen: seen.add(v); st.append(v)
        reach.append({SE.flag_of(u) for u in seen})
    CN = torch.tensor([float(r == {1}) for r in reach]); CF = torch.tensor([float(r == {2}) for r in reach])
    return CN, CF


class CompEngine(SanEngine):
    def comp_setup(self, phrase_set='wntf', maxf=2, gain=None, cap0=320, extra=320, cache=None):
        tok, dev = self.tok, self.dev
        V = self.W['E'].shape[0]
        txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
        TT, FLAG, S = SE.build_tables(txt, cache)
        nS = len(S)
        rows, CLS = {}, torch.empty(V, dtype=torch.long)
        for v in range(V):
            k = tuple(TT[v].tolist()); CLS[v] = rows.setdefault(k, len(rows))
        TC = torch.zeros(len(rows), nS, nS)
        for k, c in rows.items():
            for s, s2 in enumerate(k): TC[c, s, s2] = 1.0
        CN, CF = certain_sets(S)
        self.c_CLS, self.c_TC, self.c_CN, self.c_CF = CLS.to(dev), TC.to(dev), CN.to(dev), CF.to(dev)
        self.c_nS, self.c_S0 = nS, F.one_hot(torch.tensor(0), nS).float().to(dev)
        self.c_tables = (TT, CLS, TC, CN, CF, S)
        self.c_maxf = maxf; self.s_cap0 = cap0; self.s_extra = extra
        self.c_on = torch.zeros((), device=dev)
        self.set_phrases(phrase_set, gain)
        Bp = self.B_max + 1
        z = lambda: torch.zeros(Bp, device=dev, dtype=torch.long)
        # reuse SanEngine's compacted registers: s_dfa = DFA state, s_fp = pointer (0 idle), s_nf = fires
        self.s_dfa, self.s_nf, self.s_fr, self.s_fp = z(), z(), z(), z(); self.s_fr.fill_(-1)
        self.s_mode = torch.zeros((), device=dev, dtype=torch.long)       # SanEngine's own circuit stays OFF
        self.s_z0 = torch.zeros(Bp, V, device=dev); self.s_qlen = z(); self.s_rst = torch.zeros(Bp, device=dev, dtype=torch.bool)
        self.graphs = {}; self.pool = None; self._outshape = None
        return self

    def set_phrases(self, phrase_set, gain=None):
        tok, dev, W = self.tok, self.dev, self.W
        T = lambda s: tok(s, add_special_tokens=False)['input_ids']
        banks = {c: T(PHRASES[phrase_set][c]) for c in (1, 2)}
        ids = banks[1] + banks[2]; P = len(ids)
        SH = torch.zeros(P + 1, P + 1); SH[0, 0] = 1
        st1, st2 = 1, 1 + len(banks[1])
        for k in range(1, P + 1):
            last = (k == len(banks[1])) or (k == P)
            SH[k, 0 if last else k + 1] = 1
        START = {1: F.one_hot(torch.tensor(st1), P + 1).float(), 2: F.one_hot(torch.tensor(st2), P + 1).float()}
        DIR = self._solve_dirs(ids)
        self.c_ids = torch.tensor(ids, device=dev); self.c_SH = SH.to(dev); self.c_ST1 = START[1].to(dev); self.c_ST2 = START[2].to(dev)
        self.c_DIR = DIR.to(dev); self.c_P = P; self.c_Lmax = max(len(banks[1]), len(banks[2]))
        self.c_gain = float(gain or 3e4)
        self.c_phrase = phrase_set
        self.graphs = {}; self.pool = None; self._outshape = None

    def _solve_dirs(self, ids):
        """unit-rms direction d_k with argmax(E @ (lnf * d_k / rms(d_k))) == ids[k] by a margin (start: E[t] / lnf)."""
        E = self.W['E'].float(); g = self.W['lnf'].float()
        D = []
        for t in ids:
            d = (E[t] / g).clone().requires_grad_(True)
            opt = torch.optim.Adam([d], lr=1e-2)
            for it in range(400):
                h = g * d / d.pow(2).mean().sqrt()
                lg = E @ h
                oth = lg.clone(); oth[t] = -1e9
                m = lg[t] - oth.max()
                if m.item() > 8.0 and it > 0: break
                opt.zero_grad(); (-(lg[t] - torch.logsumexp(oth, 0))).backward(); opt.step()
            d = d.detach(); d = d / d.pow(2).mean().sqrt(); D.append(d)
            h = g * d; lg = E @ h; assert int(lg.argmax()) == t, ('direction failed', t)
        self.c_margin = []
        return torch.stack(D)

    def set_comp(self, on=True):
        self.c_on.fill_(1.0 if on else 0.0)

    def _sanity(self, B, tok, nxt, lp, wf, logits):          # SanEngine's decode-time circuit is not used
        return nxt, lp, wf

    def _comp(self, B, tok):
        s = F.one_hot(self.s_dfa[:B], self.c_nS).float()
        s2 = torch.bmm(s[:, None], self.c_TC[self.c_CLS[tok]])[:, 0]
        cN = s2 @ self.c_CN; cF = (s2 @ self.c_CF) * (1 - cN)
        p = F.one_hot(self.s_fp[:B], self.c_P + 1).float()
        idle = p[:, 0]
        under = (self.s_nf[:B] < self.c_maxf).float()
        room = ((self.glen[:B] + self.c_Lmax + 24) < self.s_cap0 + self.s_extra * self.s_nf[:B]).float() * (~self.done[:B]).float()
        fN = F.relu(cN + idle + under + room + self.c_on - 4)
        fF = F.relu(cF + idle + under + room + self.c_on - 4)
        fire = fN + fF
        p2 = (p @ self.c_SH) * (1 - fire)[:, None] + fN[:, None] * self.c_ST1 + fF[:, None] * self.c_ST2
        write = (p2[:, 1:] @ self.c_DIR) * self.c_gain
        s3 = s2 * (1 - fire)[:, None] + fire[:, None] * self.c_S0
        self.s_dfa[:B] = s3.argmax(-1); self.s_fp[:B] = p2.argmax(-1); self.s_nf[:B] += fire.long()
        wbits = ((1 - p2[:, 0]) * 2 + fire * 8).to(torch.int8)
        return write, wbits, p2

    def _step_triton(self, B, S):
        import fastqwen as Q, hostkern as HK, triton
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        imp = self.imp if self.imp_on else None
        links = self.links if self.imp_on else None
        emb = W['E'][tok]
        feat = write = wf = None; dec = lpd = None
        if imp is not None:
            e = imp.step(emb, B, tok) * self.row_on[:B, None]
            feat = e
            if self.use_write:
                write = (e @ self.Dp) if self.Dp is not None else (imp.last_w if imp.fk is not None else e @ imp.D)
            wf = (e.abs().sum(-1) > 0).to(torch.int8)
        cw, wbits, p2 = self._comp(B, tok)
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        x.copy_(emb)
        pos = self.Lp + slot
        ieee = x.dtype == torch.float32
        sc = 1.0 / (Q.HD ** 0.5)
        delta = None
        for i, lw in enumerate(W['L']):
            if links is not None and i in links:
                if delta is not None: x += delta; delta = None
                x += links[i](feat, x).to(x.dtype)
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
        if links is not None and 'final' in links:
            x += links['final'](feat, x).to(x.dtype)
        if write is not None:
            x += write.to(x.dtype)
        x += cw.to(x.dtype)                                   # the compiled sanity write (exactly 0 when silent)
        HK.add_rms_kernel[(B,)](x, x, W['lnf'], h, D=Q.D, HAS_DELTA=False, eps=Q.EPS)
        logits = F.linear(h, W['E'])
        nxt, lp = self.nxt_b[:B], self.lp_b[:B]
        HK.sample_kernel[(B,)](logits, self.temp, self.greedy, self.seed, nxt, lp, logits.shape[1], BV=2048, num_warps=8)
        self.seed += 1
        wf = wbits if wf is None else (wf | wbits)
        g0 = self.glen[:B].clone()
        self._book(B, tok, nxt, lp, wf, imp if imp is not None else True, dec, None)
        self.done[:B] |= (g0 + 2) >= (self.s_cap0 + self.s_extra * self.s_nf[:B])


# ------------------------------------------------------------------------------------------------------ exactness
def fuzz(tok, TT, CLS, TC, CN, CF, S, n=200000, seed=0):
    """(1) token-level compiled DFA (class gather + one-hot matmul) == char DFA at every token boundary;
       (2) certain flags are sound: for every prefix with cN/cF = 1, the regex class of prefix + random suffix + '\\n'
           is 1 / 2; and complete: a string whose final regex class is 1/2 has cN/cF = 1 at some prefix."""
    rnd = random.Random(seed)
    pieces = ['####', '#', '-', '.', ',', ' ', '\n', '$', 'x', 'The answer is', '0', '00'] + [str(d) for d in range(10)] * 3
    nS = len(S); ix = {s: i for i, s in enumerate(S)}
    bad1 = bad_sound = bad_compl = 0; nfire = 0
    for it in range(n):
        s = ''.join(rnd.choice(pieces) for _ in range(rnd.randint(1, 14)))
        if it % 2 == 0:                       # half the strings: a '####' answer line with sign / commas / decimals
            s = ''.join(rnd.choice(pieces) for _ in range(rnd.randint(0, 4))) + '####' + rnd.choice(['', ' ', '  ', ' $', '\n']) + \
                rnd.choice(['', '-', '--', '-.']) + ''.join(rnd.choice('0123456789,.') for _ in range(rnd.randint(0, 7))) + \
                ''.join(rnd.choice(pieces) for _ in range(rnd.randint(0, 3)))
        ids = tok(s, add_special_tokens=False)['input_ids']
        st = F.one_hot(torch.tensor(0), nS).float(); cs = S[0]; fired = 0; text = ''
        for t in ids:
            st = st @ TC[CLS[t]]
            for c in tok.decode([t]):
                cs = SE.char_step(cs, c)
            text += tok.decode([t])
            if int(st.argmax()) != ix[cs]: bad1 += 1
            cn, cf = float(st @ CN), float(st @ CF)
            if (cn or cf) and not fired:
                fired = 1 if cn else 2; nfire += 1
                for _ in range(3):
                    suf = ''.join(rnd.choice(pieces) for _ in range(rnd.randint(0, 5)))
                    if SE.py_flag(text + suf + '\n') != fired: bad_sound += 1
        fin = SE.py_flag(s + '\n')
        if fin in (1, 2) and fired != fin: bad_compl += 1
    return dict(strings=n, state_mismatch=bad1, unsound=bad_sound, incomplete=bad_compl, fired=nfire)


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
    # sanity-DFA tables over the host vocabulary (Qwen3-0.6B: 151,936 embedding rows), built on the CPU on first use
    txt = [tok.decode([i]) for i in range(len(tok))]; txt += [''] * (int(os.environ.get('HOST_VOCAB', 151936)) - len(txt))
    TT, FLAG, S = SE.build_tables(txt, os.environ.get('DFA_CACHE'))
    rows, CLS = {}, torch.empty(TT.shape[0], dtype=torch.long)
    for v in range(TT.shape[0]):
        k = tuple(TT[v].tolist()); CLS[v] = rows.setdefault(k, len(rows))
    TC = torch.zeros(len(rows), len(S), len(S))
    for k, c in rows.items():
        for s, s2 in enumerate(k): TC[c, s, s2] = 1.0
    CN, CF = certain_sets(S)
    print('classes', len(rows), 'states', len(S), 'certain-neg states', int(CN.sum()), 'certain-nonint states', int(CF.sum()))
    print([S[i] for i in range(len(S)) if CN[i]], [S[i] for i in range(len(S)) if CF[i]])
    print(fuzz(tok, TT, CLS, TC, CN, CF, S, n=int(sys.argv[1]) if len(sys.argv) > 1 else 20000))
