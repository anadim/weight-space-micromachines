"""[38] arm I: the exact calculator CORE of the implant -- a fixed (parameter-free, frozen) batched int64 tensor machine.

Input: the <calc>EXPR</calc> expression of each firing row as unicode code points (N, L) + lengths. Output: the result as
output symbols (0-9 digits, 10 '.', 11 '-', 12 'ERR'), at most FMAX, + register features. Semantics = tooltext.calc_eval
(the arm-T tool) exactly, except where an intermediate leaves int64 / a float literal has > 17 significant digits
(flag 'inexact'; the result is then ERR, counted as a mismatch in verify_core.py). Float literals are read as exact
decimals; calc_eval reads Fraction(str(float(lit))), the same value whenever lit is a shortest float repr (<= 15
significant digits always; 16-17 digits when the model copied a Python float) -- verify_core.py counts the rest.

Stages (all tensor ops, rows in lockstep; no Python per row):
  1 reader    length check (> 120 code points: ERR); calc_eval's text rewrites as a vectorised transducer:
              ',' between a digit and 3 digits dropped, '$' dropped, 'x'/'×' -> '*', '^' -> '**', '%' -> '/100';
              strip outer whitespace.
  2 machine   shift-reduce evaluator over chars with Python's grammar for + - * / ** unary +/- ( ) and numeric
              literals (leading-zero ints, '1.2.3', '2(3)', '//', newline outside parens ... are errors like
              ast.parse); exact rationals (num, den > 0, gcd-reduced) in int64 with overflow guards; one action per
              row per iteration (consume a char, finalize a number, push an operator or reduce once).
  3 writer    |v| > 1e18 or den > 9e17: ERR; else integer digits, or half-up rounding at 6 decimals with trailing zeros stripped
              ('-0' -> '0'); symbol string cut at FMAX (= the tool's forced-token cap).
"""
import torch

FMAX = 24
MAX_EXPR = 120
SYM_DOT, SYM_NEG, SYM_ERR = 10, 11, 12
N_SYM = 13
D_STK = 128                        # >= any 120-char expression's operator / value stack
BIG = 4.0e18                       # int64 guard (2^63 ~ 9.22e18) on float64 magnitude estimates

# op codes
ADD, SUB, MUL, DIV, POW, UNEG, UPOS, LPAR = 1, 2, 3, 4, 5, 6, 7, 8
PREC = [0, 1, 1, 2, 2, 4, 3, 3, 0]      # unary binds below ** and above * /
# char classes after the reader
C_PAD, C_DIG, C_DOT, C_PLUS, C_MINUS, C_STAR, C_SLASH, C_LP, C_RP, C_SP, C_NL, C_BAD, C_END, C_WS2 = range(14)
# other str.isspace() characters: removed by calc_eval's strip() at the ends, a syntax error inside (reader maps them to 11)
WS2 = [c for c in range(0x3001) if chr(c).isspace() and chr(c) not in ' \t\f\n\r']
# register feature layout (R_DIM): result sign/err/int, #int digits (16), #frac digits (7), #operands (5),
# max operand int digits (8), operators present (+ - * / ( **)
R_DIM = 3 + 16 + 7 + 5 + 8 + 6


def _cls_table(dev):
    t = torch.full((128,), C_BAD, dtype=torch.long)
    for c in '0123456789': t[ord(c)] = C_DIG
    t[ord('.')] = C_DOT; t[ord('+')] = C_PLUS; t[ord('-')] = C_MINUS; t[ord('*')] = C_STAR; t[ord('/')] = C_SLASH
    t[ord('(')] = C_LP; t[ord(')')] = C_RP
    for c in ' \t\f': t[ord(c)] = C_SP
    for c in '\n\r': t[ord(c)] = C_NL
    for c in WS2:
        if c < 128: t[c] = C_WS2
    return t.to(dev)


class ExactCore:
    def __init__(self, dev):
        self.dev = dev
        self.cls = _cls_table(dev)
        self.p10 = torch.tensor([10 ** k for k in range(19)], dtype=torch.long, device=dev)
        self.prec = torch.tensor(PREC, dtype=torch.long, device=dev)
        self.ws2 = torch.tensor(WS2, dtype=torch.long, device=dev)

    # ------------------------------------------------------------------ 1 reader
    def read(self, cp, ln):
        """cp (N, L) int code points (0 pad), ln (N,). -> chars (N, L2) ASCII codes after the rewrites, ln2, err (N,)."""
        N, L = cp.shape
        dev = self.dev
        ar = torch.arange(L, device=dev)[None]
        valid = ar < ln[:, None]
        err = ln > MAX_EXPR
        isd = (cp >= 48) & (cp <= 57) & valid
        # ',' with a digit before and >= 3 digits after
        def sh(t, k):                                            # t shifted: out[:, i] = t[:, i + k] (0 outside)
            z = torch.zeros(N, 4, dtype=t.dtype, device=dev)
            return torch.cat([z, t, z], 1)[:, 4 + k:4 + k + L]
        dropc = (cp == ord(',')) & sh(isd, -1) & sh(isd, 1) & sh(isd, 2) & sh(isd, 3)
        drop = (dropc | (cp == ord('$'))) & valid
        c = torch.where((cp == ord('x')) | (cp == 0xD7), torch.full_like(cp, ord('*')), cp)
        cnt = torch.where(drop | ~valid, 0, torch.where(c == ord('^'), 2, torch.where(c == ord('%'), 4, 1)))
        start = torch.cumsum(cnt, 1) - cnt
        ln2 = cnt.sum(1)
        L2 = int(ln2.max()) if N else 0
        L2 = max(L2, 1)
        out = torch.zeros(N, L2 + 1, dtype=torch.long, device=dev)
        rows = torch.arange(N, device=dev)[:, None].expand(N, L)
        # sub-character k of each input char's expansion
        sub = [torch.where(c == ord('^'), ord('*'), torch.where(c == ord('%'), ord('/'), c)),
               torch.where(c == ord('^'), ord('*'), ord('1')), torch.full_like(c, ord('0')), torch.full_like(c, ord('0'))]
        for k in range(4):
            m = cnt > k
            idx = torch.where(m, start + k, torch.full_like(start, L2))
            out[rows[m], idx[m]] = sub[k][m].long()
        out = out[:, :L2]
        # non-ASCII code points that survive (other than the rewritten ones) are errors
        isws2 = torch.isin(out, self.ws2)
        out = torch.where(isws2, 11, torch.where(out > 127, 1, out))      # 11 -> class WS2, 1 -> class BAD
        return out, ln2, err

    # ------------------------------------------------------------------ helpers (int64 rationals)
    @staticmethod
    def _red(n, d):
        g = torch.gcd(n, d).clamp(min=1)
        return n // g, d // g

    @staticmethod
    def _ovf_mul(x, y):
        return x.double().abs() * y.double().abs() > BIG

    def _mul(self, n1, d1, n2, d2):
        g1 = torch.gcd(n1, d2).clamp(min=1); g2 = torch.gcd(n2, d1).clamp(min=1)
        a, b = n1 // g1, n2 // g2
        c, d = d1 // g2, d2 // g1
        ov = self._ovf_mul(a, b) | self._ovf_mul(c, d)
        return a * b, c * d, ov

    def _add(self, n1, d1, n2, d2, sign):
        g = torch.gcd(d1, d2).clamp(min=1)
        f1, f2 = d2 // g, d1 // g
        ov = self._ovf_mul(n1, f1) | self._ovf_mul(n2, f2) | self._ovf_mul(d1, f1)
        a, b = n1 * f1, n2 * f2 * sign
        ov = ov | ((a.double() + b.double()).abs() > BIG)
        n, d = self._red(a + b, d1 * f1)
        return n, d, ov

    # ------------------------------------------------------------------ 2 machine
    def evaluate(self, ch, ln2):
        """ch (N, L2) ASCII codes, ln2 (N,). -> num, den (N,), err (N,), inexact (N,), regs dict."""
        N, L2 = ch.shape
        dev = self.dev; i64 = torch.long
        cl = self.cls[ch.clamp(max=127)]
        ar = torch.arange(L2, device=dev)[None]
        inb = ar < ln2[:, None]
        cl = torch.where(inb, cl, torch.full_like(cl, C_END))
        ws = (cl == C_SP) | (cl == C_NL) | (cl == C_WS2)
        nonws = inb & ~ws
        any_nw = nonws.any(1)
        first = torch.where(any_nw, nonws.float().argmax(1), torch.zeros(N, dtype=i64, device=dev))
        last = torch.where(any_nw, L2 - 1 - nonws.flip(1).float().argmax(1), torch.full((N,), -1, dtype=i64, device=dev))
        # strip(): positions outside [first, last] become END / skipped
        cl = torch.where((ar > last[:, None]), torch.full_like(cl, C_END), cl)
        cl = torch.where(cl == C_WS2, C_BAD, cl)
        cl = torch.cat([cl, torch.full((N, 1), C_END, dtype=i64, device=dev)], 1)
        dg = torch.cat([(ch - 48).clamp(0, 9), torch.zeros(N, 1, dtype=i64, device=dev)], 1)
        z = lambda: torch.zeros(N, dtype=i64, device=dev)
        b = lambda: torch.zeros(N, dtype=torch.bool, device=dev)
        pos = first.clone(); mode = z(); mant = z(); nfrac = z(); nsig = z(); nint = z()
        hasint = b(); hasfrac = b(); first0 = b(); nz_int = b()
        pend = z(); wop = z()
        vn = torch.zeros(N, D_STK, dtype=i64, device=dev); vd = torch.ones(N, D_STK, dtype=i64, device=dev)
        ops = torch.zeros(N, D_STK, dtype=i64, device=dev)
        vsp = z(); osp = z(); depth = z()
        err = ~any_nw; inex = b(); fin = b()
        n_opd = z(); max_opd = z(); opset = torch.zeros(N, 6, dtype=torch.bool, device=dev)
        rows = torch.arange(N, device=dev)
        it = 0
        max_it = 6 * (L2 + 2) + 8
        while True:
            act = ~fin & ~err
            if not bool(act.any()) or it > max_it: break
            it += 1
            c = cl.gather(1, pos.clamp(max=L2)[:, None])[:, 0]
            d = dg.gather(1, pos.clamp(max=L2)[:, None])[:, 0]
            top = ops.gather(1, (osp - 1).clamp(0, D_STK - 1)[:, None])[:, 0]
            top = torch.where(osp > 0, top, z())
            innum = (mode == 1) | (mode == 2)
            # ---- choose one action per row
            a_fin = act & innum & ~((c == C_DIG) | ((c == C_DOT) & (mode == 1)))
            rest = act & ~a_fin
            a_pend = rest & (pend != 0)
            rest = rest & ~a_pend
            red_cond = (top != 0) & (top != LPAR)
            ptop = self.prec[top]; pw = self.prec[wop]
            a_wop = rest & (wop != 0)
            a_wred = a_wop & red_cond & ((ptop > pw) | ((ptop == pw) & (wop != POW)))
            a_wpush = a_wop & ~a_wred
            rest = rest & ~a_wop
            a_rp = rest & (c == C_RP) & (mode == 3)
            a_rpred = a_rp & red_cond
            a_rppop = a_rp & (top == LPAR)
            a_rperr = a_rp & (osp == 0)
            rest = rest & ~a_rp
            a_end = rest & (c == C_END)
            a_endred = a_end & (mode == 3) & red_cond
            a_endpar = a_end & (mode == 3) & (top == LPAR)
            a_endok = a_end & (mode == 3) & (osp == 0)
            a_enderr = a_end & ((mode == 0) | a_endpar)
            rest = rest & ~a_end
            a_chr = rest                          # consume one char by mode
            # ---- finalize number
            bad_num = (mode == 2) & ~hasint & ~hasfrac | (mode == 1) & first0 & nz_int
            fin_inex = ((mode == 2) & (nsig > 17)) | (nsig > 18) | (nfrac > 18)     # floats: <= 17 sig. digits (repr length)
            fd = self.p10[nfrac.clamp(0, 18)]
            fn, fdd = self._red(mant, fd)
            m = a_fin
            err = err | (m & bad_num) | (m & (vsp >= D_STK))
            inex = inex | (m & fin_inex)
            si = vsp.clamp(max=D_STK - 1)
            vn[rows[m], si[m]] = fn[m]; vd[rows[m], si[m]] = fdd[m]
            vsp = vsp + m.long()
            n_opd = n_opd + m.long(); max_opd = torch.where(m, torch.maximum(max_opd, nint), max_opd)
            mode = torch.where(m, 3, mode)
            # ---- resolve pending '*' / '/'
            m = a_pend
            star2 = m & (pend == 1) & (c == C_STAR)
            sl2 = m & (pend == 2) & (c == C_SLASH)
            err = err | sl2
            wop = torch.where(m, torch.where(pend == 1, torch.where(star2, POW, MUL), DIV), wop)
            pos = pos + star2.long()
            pend = torch.where(m, 0, pend)
            # ---- reduce (for a pending binary push, a ')' or the end)
            m = a_wred | a_rpred | a_endred
            if bool(m.any()):
                vn, vd, vsp, osp, err, inex = self._reduce(m, top, vn, vd, vsp, osp, err, inex)
            # ---- push the pending binary op
            m = a_wpush
            oi = osp.clamp(max=D_STK - 1)
            err = err | (m & (osp >= D_STK))
            ops[rows[m], oi[m]] = wop[m]
            for k, o in enumerate((ADD, SUB, MUL, DIV, LPAR, POW)):
                opset[:, k] |= m & (wop == o)
            osp = osp + m.long(); wop = torch.where(m, 0, wop); mode = torch.where(m, 0, mode)
            # ---- ')' pops '('
            m = a_rppop
            osp = osp - m.long(); depth = depth - m.long(); pos = pos + m.long()
            err = err | a_rperr | a_enderr
            fin = fin | (a_endok & (vsp == 1))
            err = err | (a_endok & (vsp != 1))
            # ---- consume one char
            m = a_chr
            m0 = m & (mode == 0); m1 = m & innum; m3 = m & (mode == 3)
            # mode 0: operand expected
            st_num = m0 & (c == C_DIG); st_dot = m0 & (c == C_DOT)
            pu = m0 & ((c == C_MINUS) | (c == C_PLUS) | (c == C_LP))
            uop = torch.where(c == C_MINUS, UNEG, torch.where(c == C_PLUS, UPOS, LPAR))
            oi = osp.clamp(max=D_STK - 1)
            err = err | (pu & (osp >= D_STK))
            ops[rows[pu], oi[pu]] = uop[pu]
            osp = osp + pu.long(); depth = depth + (pu & (c == C_LP)).long()
            opset[:, 4] |= pu & (c == C_LP)
            sp0 = m0 & ((c == C_SP) | ((c == C_NL) & (depth > 0)))
            err = err | (m0 & ~(st_num | st_dot | pu | sp0))
            # number start
            mode = torch.where(st_num, 1, torch.where(st_dot, 2, mode))
            mant = torch.where(st_num, d, torch.where(st_dot, 0, mant))
            nfrac = torch.where(st_num | st_dot, 0, nfrac)
            nsig = torch.where(st_num, (d > 0).long(), torch.where(st_dot, 0, nsig))
            nint = torch.where(st_num, 1, torch.where(st_dot, 0, nint))
            hasint = torch.where(st_num, True, torch.where(st_dot, False, hasint))
            hasfrac = torch.where(st_num | st_dot, False, hasfrac)
            first0 = torch.where(st_num, d == 0, torch.where(st_dot, False, first0))
            nz_int = torch.where(st_num | st_dot, False, nz_int)
            # in number: digit / '.'
            m1d = m1 & (c == C_DIG); m1p = m1 & (c == C_DOT)
            sig = m1d & ((nsig > 0) | (d > 0))
            nsig = nsig + sig.long()
            acc = sig & (nsig <= 18)
            mant = torch.where(acc, mant * 10 + d, mant)
            fr = m1d & (mode == 2)
            # digits beyond 18 significant: value inexact (flag), keep the scale right for ints
            nfrac = nfrac + (fr & (nsig <= 18)).long()
            nint = nint + (m1d & (mode == 1)).long()
            nz_int = nz_int | (m1d & (mode == 1) & (d > 0))
            hasfrac = hasfrac | fr
            mode = torch.where(m1p, 2, mode)
            # mode 3: operator expected
            pm = m3 & ((c == C_PLUS) | (c == C_MINUS))
            wop = torch.where(pm, torch.where(c == C_PLUS, ADD, SUB), wop)
            pend = torch.where(m3 & (c == C_STAR), 1, torch.where(m3 & (c == C_SLASH), 2, pend))
            sp3 = m3 & ((c == C_SP) | ((c == C_NL) & (depth > 0)))
            ok3 = pm | (m3 & ((c == C_STAR) | (c == C_SLASH))) | sp3
            err = err | (m3 & ~ok3)
            pos = pos + m.long()
        err = err | ~fin
        n = vn[:, 0]; dd = vd[:, 0]
        regs = dict(n_opd=n_opd, max_opd=max_opd, opset=opset)
        return n, dd, err, inex, regs

    def _reduce(self, m, top, vn, vd, vsp, osp, err, inex):
        N = vn.shape[0]; dev = self.dev; rows = torch.arange(N, device=dev)
        un = (top == UNEG) | (top == UPOS)
        ib = (vsp - 1).clamp(0, D_STK - 1); ia = (vsp - 2).clamp(0, D_STK - 1)
        nb, db = vn.gather(1, ib[:, None])[:, 0], vd.gather(1, ib[:, None])[:, 0]
        na, da = vn.gather(1, ia[:, None])[:, 0], vd.gather(1, ia[:, None])[:, 0]
        err = err | (m & ~un & (vsp < 2)) | (m & un & (vsp < 1))
        # unary
        un_n = torch.where(top == UNEG, -nb, nb)
        # binary
        s_n, s_d, s_o = self._add(na, da, nb, db, torch.where(top == SUB, -1, 1))
        m_n, m_d, m_o = self._mul(na, da, nb, db)
        zb = nb == 0
        inv_n = torch.where(nb < 0, -db, db); inv_d = nb.abs().clamp(min=1)
        q_n, q_d, q_o = self._mul(na, da, inv_n, inv_d)
        # power: exponent integer, |e| <= 12
        pe_ok = (db == 1) & (nb.abs() <= 12)
        e = nb.abs().clamp(max=12)
        p_n = torch.ones_like(na); p_d = torch.ones_like(da); p_o = torch.zeros_like(zb)
        for k in range(12):
            mk = k < e
            t_n, t_d, t_o = self._mul(p_n, p_d, na, da)
            p_n = torch.where(mk, t_n, p_n); p_d = torch.where(mk, t_d, p_d); p_o = p_o | (mk & t_o)
        pz = (nb < 0) & (p_n == 0)
        pi_n = torch.where(nb < 0, torch.where(p_n < 0, -p_d, p_d), p_n)
        pi_d = torch.where(nb < 0, p_n.abs().clamp(min=1), p_d)
        r_n = torch.where(top == MUL, m_n, torch.where(top == DIV, q_n, torch.where(top == POW, pi_n, s_n)))
        r_d = torch.where(top == MUL, m_d, torch.where(top == DIV, q_d, torch.where(top == POW, pi_d, s_d)))
        r_o = torch.where(top == MUL, m_o, torch.where(top == DIV, q_o, torch.where(top == POW, p_o, s_o)))
        r_e = ((top == DIV) & zb) | ((top == POW) & (~pe_ok | pz))
        mb = m & ~un
        mu = m & un
        err = err | (mb & r_e)
        inex = inex | (mb & r_o)
        vn = vn.clone(); vd = vd.clone()
        vn[rows[mb], ia[mb]] = r_n[mb]; vd[rows[mb], ia[mb]] = r_d[mb]
        vn[rows[mu], ib[mu]] = un_n[mu]
        vsp = vsp - mb.long()
        osp = osp - m.long()
        return vn, vd, vsp, osp, err, inex

    # ------------------------------------------------------------------ 3 writer
    def write(self, n, d, err, inex):
        """-> sym (N, FMAX) long (-1 pad), slen (N,), and result registers."""
        N = n.shape[0]; dev = self.dev; i64 = torch.long
        err = err | inex
        na = n.abs()
        I = na // d.clamp(min=1); r = na % d.clamp(min=1)
        big = (I > 10 ** 18) | ((I == 10 ** 18) & (r > 0))
        err = err | big
        dsafe = d.clamp(min=1)
        err = err | ((d > 1) & (dsafe > 9 * 10 ** 17))           # long division below needs 10 * den < 2^63
        rr = r.clone(); frac = torch.zeros_like(r)
        for _ in range(6):                                       # 6 decimals by long division, then half-up on the rest
            rr = rr * 10; dig = rr // dsafe; rr = rr - dig * dsafe; frac = frac * 10 + dig
        frac = frac + ((d > 1) & (2 * rr >= dsafe)).long()
        carry = frac >= 10 ** 6
        I = I + carry.long(); frac = torch.where(carry, frac - 10 ** 6, frac)
        isint = d == 1
        frac = torch.where(isint, torch.zeros_like(frac), frac)
        neg = (n < 0) & ((I > 0) | (frac > 0))
        # integer digits (most significant first)
        p10 = self.p10
        nd_int = torch.ones(N, dtype=i64, device=dev)
        for k in range(1, 19):
            nd_int = nd_int + (I >= p10[k]).long()
        # fractional digits: 6 places, trailing zeros stripped
        fd = torch.stack([(frac // (10 ** (5 - k))) % 10 for k in range(6)], 1)       # (N, 6)
        nz = fd != 0
        nd_frac = torch.where(nz.any(1), 6 - nz.flip(1).float().argmax(1).long(), torch.zeros(N, dtype=i64, device=dev))
        # assemble: [-] I digits [. frac digits]
        L = 1 + 19 + 1 + 6
        sym = torch.full((N, L), -1, dtype=i64, device=dev)
        k = torch.arange(L, device=dev)[None]
        off = neg.long()[:, None]
        sym = torch.where(neg[:, None] & (k == 0), SYM_NEG, sym)
        j = k - off                                             # index into the integer digits
        Ib = I.clamp(min=0)
        pw = (nd_int[:, None] - 1 - j).clamp(0, 18)
        idig = (Ib[:, None] // p10[pw]) % 10
        in_int = (j >= 0) & (j < nd_int[:, None])
        sym = torch.where(in_int, idig, sym)
        dot = nd_frac > 0
        kd = off[:, 0] + nd_int
        sym = torch.where(dot[:, None] & (k == kd[:, None]), SYM_DOT, sym)
        jf = k - kd[:, None] - 1
        in_f = dot[:, None] & (jf >= 0) & (jf < nd_frac[:, None])
        sym = torch.where(in_f, fd.gather(1, jf.clamp(0, 5)), sym)
        slen = off[:, 0] + nd_int + dot.long() + nd_frac
        sym = torch.where(err[:, None], torch.where(k == 0, SYM_ERR, -1), sym)
        slen = torch.where(err, torch.ones_like(slen), slen)
        sym = sym[:, :FMAX]; slen = slen.clamp(max=FMAX)
        res = dict(neg=neg & ~err, err=err, isint=isint & ~err, nd_int=torch.where(err, 0, nd_int),
                   nd_frac=torch.where(err, 0, nd_frac))
        return sym, slen, res

    # ------------------------------------------------------------------ all
    @torch.no_grad()
    def __call__(self, cp, ln):
        ch, ln2, err0 = self.read(cp, ln)
        n, d, err, inex, regs = self.evaluate(ch, ln2)
        err = err | err0
        sym, slen, res = self.write(n, d, err, inex)
        return sym, slen, self.features(regs, res), inex & ~err0

    def features(self, regs, res):
        """register features (N, R_DIM) float, multi-hot."""
        N = res['err'].shape[0]; dev = self.dev
        oh = lambda v, n: torch.nn.functional.one_hot(v.clamp(0, n - 1), n).float()
        f = [res['neg'].float()[:, None], res['err'].float()[:, None], res['isint'].float()[:, None],
             oh(res['nd_int'] - 1, 16) * (~res['err']).float()[:, None], oh(res['nd_frac'], 7) * (~res['err']).float()[:, None],
             oh(regs['n_opd'] - 1, 5) * (regs['n_opd'] > 0).float()[:, None],
             oh(regs['max_opd'] - 1, 8) * (regs['max_opd'] > 0).float()[:, None], regs['opset'].float()]
        out = torch.cat(f, 1)
        assert out.shape[1] == R_DIM
        return out

    def encode(self, exprs):
        """python strings (None -> ERR) -> (cp, ln) tensors (the engine's text-side hand-off)."""
        L = max([len(e) for e in exprs if e is not None] + [1])
        L = min(L, MAX_EXPR + 1)
        cp = torch.zeros(len(exprs), L, dtype=torch.long)
        ln = torch.zeros(len(exprs), dtype=torch.long)
        for i, e in enumerate(exprs):
            if e is None:
                ln[i] = MAX_EXPR + 1; continue                      # no open tag: ERR (as calc_eval(None))
            ln[i] = len(e)
            s = e[:L]
            if s: cp[i, :len(s)] = torch.tensor([ord(ch) for ch in s], dtype=torch.long)
        return cp.to(self.dev), ln.to(self.dev)


SYM_TXT = [str(i) for i in range(10)] + ['.', '-', 'ERR']


def sym_str(sym, slen):
    return [''.join(SYM_TXT[int(s)] for s in row[:int(n)]) for row, n in zip(sym.tolist(), slen.tolist())]
