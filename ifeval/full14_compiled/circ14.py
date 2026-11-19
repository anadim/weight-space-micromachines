"""Line-by-line transliteration of ifeval/heldout/circuit.py (md5 cd15769c...) into the symbolic wire DSL (sym.py).
Building it produces two netlists: STEP (state, token id) -> state and WRITE (state, host readout) -> (state, write row index).
Line numbers in comments refer to circuit.py. Registers are named after the circuit's attributes.

Departures from circuit.py (all exact, see NOTEBOOK):
 (a) int64 rolling hash (h*POWT + HT) mod (2^31-1): computed exactly with bits of h (ReLU staircase), per-bit residue
     constants (fixed table PCOEF[power][bit] = 2^bit * power mod p) and a ReLU staircase for the final mod -> the SAME hash
     values (collisions included), in float64.
 (b) torch.sort hole compaction: prefix counts + one-hot selection.
 (c) readout: copy of the LM-head rows + final-norm gain inside the implant (vocab node).
 numbers: the digit accumulator saturates at CAP=10^6 (circuit.py: float32, all uses compare with <= 21); see NOTEBOOK.
"""
import torch
from collections import OrderedDict
from sym import Graph, Sym, S, const, where, maximum, minimum, clamp, rem2, ge, assume, mux, AND, BIG

HP = 2147483647
MAXMEM = 8192
EOS = 151645
BIGAGE = 99
S_W = 6
CAP = 10 ** 6
V = 151936
RULES = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'pps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul', 'star3']
CUES = ['two', 'title', 'bul', 'par', 'high', 'nth', 'hightimes']
FLAGS = ['NUMV', 'ISNUM', 'ISDIG', 'ORDV', 'ORDSUF', 'NONWS', 'SENTB', 'CLAUSEB', 'NL', 'NN', 'ENDNL', 'STARTNL', 'PRECNN', 'POSTNN',
         'STAR3', 'LEADST', 'TRAILST', 'ALLST', 'DQC', 'BTC', 'DQOPEN', 'DQCLOSE', 'QMARK', 'STAR6', 'STAR6X', 'ANGTOK', 'CLOSETITLE',
         'COLON', 'STARC', 'BULST', 'LETTERS', 'WSTART', 'ENDALNUM', 'ALPHA', 'CONT', 'QE', 'QX', 'OPENDQ', 'OPENSQ', 'CLOSEDQ',
         'CLOSESQ', 'IMSTART', 'IMEND', 'LEAD', 'TRAIL', 'HASAN', 'LOWERSET', 'UPPERSET', 'COMMASET', 'STAR3SET']
INTS = ['PRE', 'LOW', 'NOSP', 'CAP1', 'CANON', 'HT']


# ------------------------------------------------------------------------------------------------ fixed tables
def build_tables(T, E_rows=V):
    """From circuit.py's tables.pt dict -> the implant's fixed tables (numpy/torch, CPU)."""
    CLASSES = None
    cols, data = [], []
    for k in FLAGS: cols.append(k); data.append(T[k].double())
    for k in INTS: cols.append(k); data.append(T[k].double())
    POWS = torch.unique(T['POWT'])                                       # 36 distinct powers 131^len mod p
    pidx = torch.searchsorted(POWS, T['POWT'])
    cols.append('PIDX'); data.append(pidx.double())
    for i in range(6): cols.append(f'UPSEQ{i}'); data.append(T['UPSEQ'][:, i].double())
    for i in range(6): cols.append(f'LOWSEQ{i}'); data.append(T['LOWSEQ'][:, i].double())
    eosm = torch.zeros(V, dtype=torch.float64); eosm[[151645, 151643]] = 1
    nlm = T['NL'].clone().double(); nlm[T['CLOSETITLE'] > 0] = 0
    cols += ['EOSMASK', 'NLMASK']; data += [eosm, nlm]
    from tables import CLASSES
    for j, c in enumerate(CLASSES): cols.append('T.' + c); data.append(T['T'][:, j].double())
    FEAT = torch.stack(data, 1)
    # pattern matcher: token -> code (0 = in no pattern); index = id + 2 (window holds -2 before the first token)
    P = T['P']
    pid = torch.unique(P[P >= 0])
    code = torch.zeros(V + 2, dtype=torch.float64)
    code[pid + 2] = torch.arange(1, len(pid) + 1, dtype=torch.float64)
    NCODE = len(pid) + 1
    PCOEF = torch.tensor([[(pow(2, i) * int(p)) % HP for i in range(31)] for p in POWS.tolist()], dtype=torch.float64)
    pcode = torch.where(P >= 0, code[(P + 2).clamp(min=0)].long(), torch.full_like(P, -1))      # (Np, 5) code or -1 wildcard
    return dict(FEAT=(FEAT, cols), CODE2=(code[:, None], ['CODE']), PCOEF=(PCOEF, [f'c{i}' for i in range(31)]),
                NCODE=NCODE, pcode=pcode, PC=T['PC'], POWS=POWS, CLASSES=CLASSES, TOK=T['TOK'])


# ------------------------------------------------------------------------------------------------ registers
def register_spec(CLASSES):
    """name -> (lo, hi, init). Bounds are invariants (counters: assumed < BIG, checked at runtime in debug mode)."""
    R = OrderedDict()
    def b(n, init=0): R[n] = (0, 1, init)
    def i(n, lo, hi, init=0): R[n] = (lo, hi, init)
    i('t', 0, BIG)
    for k in range(5): i(f'win{k}', -2, V - 1, -2)
    for c in CLASSES: b('cl.' + c); b('se.' + c)
    for n in ['cb_prev', 'sb_prev', 'asst', 'prevdig', 'in_sys', 'nth_last', 'e_cf', 's_klast', 's_hseen', 'rep_below', 's_negrep',
              'rep_seen', 'rg', 'rq', 'rsawq', 'rqonly', 's_abort', 'lastsep', 'rnegA', 'rfirst', 'aw_neg', 'aw_app',
              'ans', 'body', 'lq', 'lx', 'seg', 'at_ls', 'hl_open', 'hforced', 't_open', 'ps_seen', 'end_done', 'last_brk', 'ln_bul',
              'ln_cont', 'se_star', 'inq', 'insq', 'inbt', 'se_data', 'lw_neg', 'lw_pos', 'cw_neg', 'cw_pos', 'lw_seen', 'cw_seen',
              'lw_post', 'cw_post', 's_impany', 'qsent', 'qpend', 's_sayrep', 'flip_c', 'end_sent', 'se_s3', 'forced_brk', 'se_ang',
              'cpend', 'cvalid', 'capsall', 'se_six', 'numdet', 'prevdig_before', 'imp_end', 'wlead', 'wlead_now', 'imp_any', 'negopen']:
        b(n)
    b('clause_sent', 1); b('imp_pos', 1)
    for n in ['age_neg', 'age_num', 'age_para', 'aw', 'age_you']: i(n, 0, BIGAGE, BIGAGE)
    for n in ['acc', 'num_recent', 'N_par', 'N_sec', 'N_bul', 'N_high', 'nth_k', 's_k', 's_hn_first', 's_hn_last', 'sc_bul', 'sc_par', 'sc_sec']:
        i(n, 0, CAP)
    i('s_commit', 0, 2)                      # aw_neg*aw_app + inaw*NEGADJ*(aw>0) can reach 2 (circuit.py line 446)
    i('sys_nslot', 0, S_W); i('nslot', 0, S_W); i('nbase', 0, S_W)
    i('ec', 0, 3); i('eq_t', 0, 2); i('end_s', 0, BIG); i('end_len', -BIG, BIG); i('end_x', -1, V - 1, -1)
    i('wc', 0, 3); i('nth_s', 0, BIG); i('nth_len', 0, BIG)
    for n in ['lnw_pos', 'prev_end', 'msg_start', 'rep_s', 'rep_len', 'rq_age', 'n', 'kp', 'd3', 'k6', 'w6', 'n6', 'w6l', 'bcount',
              'hl_cont', 'hcount', 'end_m', 'wa', 'pj', 'sp_s', 'fires', 'strun']:
        i(n, 0, BIG)
    i('rep_wait', 0, 2); i('rqt', 0, 2); i('dm', 0, 2); i('strail', 0, 6)
    for k in range(S_W): i(f'slot_h{k}', 0, HP - 1); b(f'slot_ok{k}')
    i('hcur', 0, HP - 1)
    for n in ['cur_pf', 'prev_pf', 'ans_pf']: i(n, -1, HP - 1, -1)
    i('prevtok', -1, V - 1, -1); i('prog_prev', 0, 15); i('prog', 0, 15); i('cj', 0, 5); i('fmap', 0, 2)
    for k in range(4): i(f'pre{k}', -1, V - 1, 0); i(f'suf{k}', -1, V - 1, 0)          # circuit.reset: zeros
    i('pre_len', 0, 4); i('suf_len', 0, 4); i('sp_len', -BIG, BIG)
    for k in ['qopen', 'rep', 'title', 'nth']: b('done.' + k)
    for k in ['nc', 'lower', 'caps', 'quote', 'ps']: b('cand.' + k)
    for k in CUES: b(f'cue.{k}.clc'); b(f'cue.{k}.sneg')
    for r in RULES: b('L.' + r); b('Ls.' + r)
    return R


class Build:
    """Holds registers as Syms; methods mirror circuit.Circuit."""

    def __init__(self, G, tabs, spec):
        self.G, self.tb, self.spec = G, tabs, spec
        self.CLASSES = tabs['CLASSES']; self.CI = {c: i for i, c in enumerate(self.CLASSES)}
        self.tk = tabs['TOK']
        for n, (lo, hi, init) in spec.items():
            self.__dict__['R_' + n] = G.inp(n, lo, hi)
        # convenient views
        self.win = [self.R(f'win{k}') for k in range(5)]
        self.cl = [self.R('cl.' + c) for c in self.CLASSES]; self.se = [self.R('se.' + c) for c in self.CLASSES]
        self.L = {r: self.R('L.' + r) for r in RULES}; self.Ls = {r: self.R('Ls.' + r) for r in RULES}
        self.done = {k: self.R('done.' + k) for k in ['qopen', 'rep', 'title', 'nth']}
        self.cand = {k: self.R('cand.' + k) for k in ['nc', 'lower', 'caps', 'quote', 'ps']}
        self.cueR = {k: [self.R(f'cue.{k}.clc'), self.R(f'cue.{k}.sneg')] for k in CUES}
        self.slot_h = [self.R(f'slot_h{k}') for k in range(S_W)]; self.slot_ok = [self.R(f'slot_ok{k}') for k in range(S_W)]
        self.pre = [self.R(f'pre{k}') for k in range(4)]; self.suf = [self.R(f'suf{k}') for k in range(4)]

    def R(self, n): return self.__dict__['R_' + n]

    def __getattr__(self, n):
        if 'R_' + n in self.__dict__: return self.__dict__['R_' + n]
        raise AttributeError(n)

    def __setattr__(self, n, v):
        if 'R_' + n in self.__dict__: self.__dict__['R_' + n] = v
        else: self.__dict__[n] = v

    def outputs(self):
        """current register values -> OrderedDict name -> Sym (bounded by the declared invariants)."""
        cur = dict()
        for k in range(5): cur[f'win{k}'] = self.win[k]
        for j, c in enumerate(self.CLASSES): cur['cl.' + c] = self.cl[j]; cur['se.' + c] = self.se[j]
        for r in RULES: cur['L.' + r] = self.L[r]; cur['Ls.' + r] = self.Ls[r]
        for k, v in self.done.items(): cur['done.' + k] = v
        for k, v in self.cand.items(): cur['cand.' + k] = v
        for k, (a, b_) in self.cueR.items(): cur[f'cue.{k}.clc'] = a; cur[f'cue.{k}.sneg'] = b_
        for k in range(S_W): cur[f'slot_h{k}'] = self.slot_h[k]; cur[f'slot_ok{k}'] = self.slot_ok[k]
        for k in range(4): cur[f'pre{k}'] = self.pre[k]; cur[f'suf{k}'] = self.suf[k]
        out = OrderedDict(); self.assumed = []
        for n, (lo, hi, init) in self.spec.items():
            v = cur[n] if n in cur else self.R(n)
            v = S(self.G, v)
            if v.lo < lo or v.hi > hi:
                self.assumed.append((n, v.lo, v.hi)); v = assume(v, lo, hi)
            out[n] = v
        return out

    # -------------------------------------------------------------- table helpers
    def feat(self, idx): return self.G.gather('FEAT', idx)

    def hashmul(self, x, pidx):
        """(x * POWS[pidx]) mod p, exact: bits of x, fixed per-bit residues PCOEF[pidx][bit], ReLU staircase mod."""
        bits = bits31(x)
        crow = self.G.gather('PCOEF', pidx)
        y = None
        for i in range(31):
            term = bits[i] * crow(f'c{i}')
            y = term if y is None else y + term
        return modp(y)

    # -------------------------------------------------------------- circuit.Circuit methods
    def classes(self, ids, tr):
        prev = clamp(self.win[3], lo=0)
        wst = clamp(tr('WSTART') + (1 - self.feat(prev)('ENDALNUM')), 0, 1)
        f = [tr('T.' + c) * wst for c in self.CLASSES]                                      # line 115
        # pattern matcher (line 116-118): AND of position-shifted one-hot compares over the 5-token window
        oh = []
        for k in range(5):
            code = self.G.gather('CODE2', self.win[k] + 2)('CODE')
            oh.append(self.G.gather('ONEHOT', code))
        pc, PC = self.tb['pcode'], self.tb['PC']
        cls_of = PC.argmax(1).tolist()
        acc = [[] for _ in self.CLASSES]
        for p in range(pc.shape[0]):
            terms = [oh[k](int(pc[p, k])) for k in range(5) if pc[p, k] >= 0]
            acc[cls_of[p]].append(AND(*terms))
        X = []
        for j in range(len(self.CLASSES)):
            s = f[j]
            for m in acc[j]: s = s + m
            X.append(clamp(s, 0, 1))
        return X

    def start_prog(self, gate, kind, pre=(), span=None, suf=(), fmap=0):
        G = self.G
        g = (S(G, gate) > 0.5) & (self.prog == 0)
        def pad(lst):
            out = [S(G, -1)] * 4
            for i, x in enumerate(lst): out[i] = S(G, x)
            return out
        P = pad(pre); Sx = pad(suf)
        pl = sum((x >= 0 for x in P), S(G, 0)); sl = sum((x >= 0 for x in Sx), S(G, 0))
        ss, sn = span if span is not None else (0, 0)
        self.prog = where(g, kind, self.prog); self.pj = where(g, 0, self.pj); self.cj = where(g, 0, self.cj)
        self.pre = [where(g, P[i], self.pre[i]) for i in range(4)]; self.pre_len = where(g, pl, self.pre_len)
        self.suf = [where(g, Sx[i], self.suf[i]) for i in range(4)]; self.suf_len = where(g, sl, self.suf_len)
        self.sp_s = where(g, ss, self.sp_s); self.sp_len = where(g, sn, self.sp_len)
        self.fmap = where(g, fmap, self.fmap)
        return g

    def prog_token(self):
        G = self.G
        j = self.pj; pl, sl = self.pre_len, self.sp_len
        a = mux(self.pre, clamp(j, 0, 3))
        k = clamp(j - pl, lo=0)
        pos = clamp(self.sp_s + k, 0, MAXMEM - 1)
        b = G.attn(pos, MAXMEM, 0, V - 1)                                                   # copy head (line 144)
        first = (k == 0) & (j >= pl)
        b = where(first & (self.fmap == 1), self.feat(b)('NOSP'), b)
        b = where(first & (self.fmap == 2), self.feat(b)('CAP1'), b)
        c = mux(self.suf, clamp(j - pl - sl, 0, 3))
        t = where(j < pl, a, where(j < pl + sl, b, c))
        return clamp(t, lo=0)

    def case_seq(self, t):
        gg = self.gates()
        r = self.feat(t)
        need_u = (gg['caps'] > 0.5) & (r('LOWERSET') > 0.5)
        need_l = (gg['lower'] > 0.5) & (r('UPPERSET') > 0.5)
        one = [t] + [S(self.G, -1)] * 5
        seq = [where(need_u, r(f'UPSEQ{i}'), where(need_l, r(f'LOWSEQ{i}'), one[i])) for i in range(6)]
        c0 = seq[0] < 0
        seq = [where(c0, one[i], seq[i]) for i in range(6)]
        return seq, sum((x >= 0 for x in seq), S(self.G, 0))

    def emit_token(self):
        seq, n = self.case_seq(self.prog_token())
        return clamp(mux(seq, minimum(self.cj, 5)), lo=0)

    def gates(self):
        L, a = self.L, self.ans
        g = {k: L[k] * a for k in L}
        g['caps'] = g['caps'] * (1 - g['lower'])
        npar = where((L['star3'] > 0.5) & (self.N_par == 0), self.N_sec, self.N_par)
        g['para'] = g['para'] * (npar >= 2) * (1 - g['two'])
        g['npar'] = npar
        g['p3'] = g['para'] * L['star3']; g['pn'] = g['para'] * (1 - L['star3'])
        g['nth'] = g['nth'] * g['pn']
        g['bul'] = g['bul'] * (self.N_bul >= 1)
        g['high'] = g['high'] * (self.N_high >= 1)
        return g

    def _cue_neg(self, name, cue, negx, negc):
        clc, sneg = self.cueR[name]
        clc = maximum(cue, (1 - self.cb_prev) * clc)
        sneg = maximum(maximum(cue * negc, clc * negx), (1 - self.sb_prev) * sneg)
        self.cueR[name] = [clc, sneg]
        return sneg

    def _starc_se(self, tr):
        self.se_star = maximum(tr('STARC'), (1 - self.sb_prev) * self.se_star)
        return self.se_star

    # ================================================================== step (circuit.py lines 167-531)
    def step(self, ids):
        G = self.G; CI = self.CI; t = self.t
        tr = self.feat(ids); f = tr
        G.memwrite(minimum(t, MAXMEM - 1), ids)                                              # line 171
        L_before = dict(self.L)
        self.win = self.win[1:] + [ids]
        X = self.classes(ids, tr)
        inq_prev = self.inq
        self.inq = where(f('DQOPEN') > 0.5, 1, where(f('DQCLOSE') > 0.5, 0, rem2(self.inq + f('DQC'))))
        self.inq = where((f('NL') + f('IMEND')) > 0.5, 0, self.inq)
        insq_prev = self.insq
        self.insq = where(self.insq > 0.5, 1 - f('CLOSESQ'), f('OPENSQ'))
        self.insq = where((f('NL') + f('IMEND')) > 0.5, 0, self.insq)
        inbt_prev = self.inbt
        self.inbt = where((f('NL') + f('IMEND')) > 0.5, 0, rem2(self.inbt + f('BTC')))
        Xr = lambda c: tr('T.' + c)
        dm_prev = self.dm
        arm = f('COLON') * clamp(self.se_data, 0, 1)
        self.dm = where(arm > 0.5, 1, self.dm)
        self.dm = where((self.dm == 1) & (f('NONWS') > 0.5) & (arm < 0.5), 2, self.dm)
        self.dm = where(((self.dm == 2) & (f('NL') > 0.5)) | (f('NN') > 0.5) | (f('IMEND') > 0.5), 0, self.dm)
        mask = maximum(maximum(maximum(inq_prev, self.inq), maximum(insq_prev, self.insq)),
                       maximum(maximum(inbt_prev, self.inbt), (dm_prev > 0)))
        self.se_data = maximum(clamp(Xr('DATAV') + Xr('DATAW'), 0, 1) * (1 - mask), (1 - self.sb_prev) * self.se_data)
        X = [x * (1 - mask) for x in X]
        x = lambda c: X[CI[c]]
        cb = clamp(f('CLAUSEB') + f('IMEND') + x('ANDBUT'), 0, 1)
        sb = clamp(f('SENTB') + f('IMEND'), 0, 1)
        lead = maximum(self.cb_prev, self.imp_pos)
        prevrow = self.feat(clamp(self.win[3], lo=0))                                       # previous token's row
        wst = maximum(f('WSTART'), 1 - prevrow('ENDALNUM'))
        self.wlead = where(wst > 0.5, lead, self.wlead)
        self.wlead_now = self.wlead * (1 - x('IMPPRE'))
        self.imp_pos = where((f('NONWS') * (1 - x('IMPPRE'))) > 0.5, 0, lead)
        opener = self.wlead_now * x('INSTV')
        self.clause_sent = where(self.cb_prev > 0.5, self.sb_prev, self.clause_sent)
        self.imp_any = maximum(opener, (1 - self.cb_prev) * self.imp_any)
        auxopen = self.wlead_now * x('AUXW') * maximum(self.clause_sent, self.sb_prev)
        nxt_word = (self.qpend > 0.5) & (wst > 0.5) & (f('ALPHA') > 0.5) & (auxopen < 0.5)
        self.qsent = where(self.sb_prev > 0.5, 0, self.qsent)
        self.qsent = where(nxt_word, 1 - x('YOUW'), self.qsent)
        self.qpend = where(auxopen > 0.5, 1, where(nxt_word | (self.sb_prev > 0.5), 0, self.qpend))
        self.cl = [maximum(X[j], (1 - self.cb_prev) * self.cl[j]) for j in range(len(X))]
        self.se = [maximum(X[j], (1 - self.sb_prev) * self.se[j]) for j in range(len(X))]
        cl = lambda c: self.cl[CI[c]]; se = lambda c: self.se[CI[c]]
        nonws = f('NONWS')
        qm = f('QMARK') * (1 - maximum(maximum(inq_prev, self.inq), maximum(insq_prev, self.insq)))
        nonreq_q = qm * (1 - maximum(se('PLEASEW'), se('YOUW') * se('MODALQ')))
        self.se_ang = maximum(f('ANGTOK'), (1 - self.sb_prev) * self.se_ang)
        # roles (224-231)
        is_role_tok = self.prevtok == 151644
        self.asst = where(is_role_tok > 0.5, ids == 77091, self.asst)
        user = (1 - self.asst) * (1 - f('IMSTART')) * (1 - is_role_tok)
        self.msg_start = where((self.win[2] == 151644) & (ids == 198) & (self.asst < 0.5), t + 1, self.msg_start)
        ans_start = (self.win[3] == 151668) & (ids == 271) & (self.asst > 0.5)
        new_turn = f('IMSTART')
        # numbers (233-251)
        isdig, isnum = f('ISDIG'), f('ISNUM')
        acc_prev = self.acc; self.prevdig_before = self.prevdig
        accn = minimum(self.prevdig * self.acc.scale(10) + f('NUMV'), CAP)                  # saturating (see header)
        self.acc = isdig * accn
        numval = isdig * self.acc + (1 - isdig) * f('NUMV')
        ordv = f('ORDV') + (f('ORDSUF') * self.prevdig) * acc_prev
        self.prevdig = isdig
        self.num_recent = where(isnum > 0.5, numval, self.num_recent)
        self.age_num = where(isnum > 0.5, 0, minimum(self.age_num + 1, BIGAGE))
        self.age_num = where((self.sb_prev + f('ORDSUF')) > 0.5, BIGAGE, self.age_num)
        w2 = clamp(self.win[3], lo=0)
        pw = clamp(where(self.feat(w2)('NONWS') > 0.5, self.win[3], self.win[2]), lo=0)
        detprev = self.feat(pw)('T.POSSW')
        self.numdet = where(isnum > 0.5, detprev * (1 - self.prevdig_before), self.numdet)
        self.age_num = where(x('UNITW') > 0.5, BIGAGE, self.age_num)
        self.age_neg = where(x('NEG') > 0.5, 0, where((x('CONJ') + self.sb_prev) > 0.5, BIGAGE, minimum(self.age_neg + 1, BIGAGE)))
        self.age_neg = where(x('FORGETW') > 0.5, BIGAGE, self.age_neg)
        self.age_para = where(x('PARAW') > 0.5, 0, minimum(self.age_para + 1, BIGAGE))
        u = user
        rqc = (1 - nonreq_q) * (1 - maximum(cl('OTHERS'), cl('DESCV')))
        rqs = (1 - nonreq_q) * (1 - maximum(se('OTHERS'), se('DESCV')))
        negc = cl('NEG') * (1 - cl('FORGETW')); negs = se('NEG') * (1 - se('FORGETW'))
        prevw = lambda c: prevrow('T.' + c)
        prev_inv = prevw('INVW')
        valid_c = self.cpend * (1 - f('ALPHA') * (1 - x('STOPF')))
        self.cpend = x('COMMA') * (1 - prev_inv)
        self.cvalid = maximum(maximum(valid_c, x('COMMAP') * (1 - prev_inv)), (1 - self.cb_prev) * self.cvalid)
        m_nc = self.cvalid * maximum(negc, cl('FREEW')) * (1 - cl('FREQ2')) * (1 - cl('REMOVEV'))
        self.capsall = maximum(x('CAPSW') * maximum(prevw('ALLQ'), prevw('INW')), (1 - self.cb_prev) * self.capsall)
        capletters = clamp(cl('UPPERW') + cl('CAPW') * cl('LETTER') + self.capsall, 0, 1)
        nocap = clamp(cl('UPPERW') + cl('CAPW') * cl('LETTER') + cl('CAPSW'), 0, 1)
        scope = clamp(cl('ALLQ') + cl('INW') + cl('WHOLE2') + cl('LETTER'), 0, 1)
        self.lw_neg = maximum(x('LOWERW') * cl('NEG'), (1 - self.cb_prev) * self.lw_neg)
        self.lw_pos = maximum(x('LOWERW') * (1 - cl('NEG')), (1 - self.cb_prev) * self.lw_pos)
        capword = clamp(x('UPPERW') + x('CAPW') + x('CAPSW'), 0, 1)
        self.cw_neg = maximum(capword * cl('NEG'), (1 - self.cb_prev) * self.cw_neg)
        self.cw_pos = maximum(capword * (1 - cl('NEG')), (1 - self.cb_prev) * self.cw_pos)
        self.lw_seen = maximum(x('LOWERW'), (1 - self.cb_prev) * self.lw_seen)
        self.cw_seen = maximum(capword, (1 - self.cb_prev) * self.cw_seen)
        self.lw_post = maximum(x('NEG') * self.lw_seen * (1 - x('LOWERW')), (1 - self.cb_prev) * self.lw_post)
        self.cw_post = maximum(x('NEG') * self.cw_seen * (1 - capword), (1 - self.cb_prev) * self.cw_post)
        self.flip_c = maximum(x('APPEARS') * prevw('NEG'), (1 - self.cb_prev) * self.flip_c)
        flip = self.flip_c
        startx = cl('STARTW') * cl('FIRSTEACH')
        lowA = self.lw_pos * scope * (1 - self.lw_neg) * (1 - self.lw_post) * (1 - cl('TIMES')) * (1 - startx) * (1 - cl('PARTW'))
        lowB = nocap * scope * maximum(self.cw_neg, self.cw_post * flip) * (1 - cl('FREQ2')) * (1 - cl('TIMES'))
        capA = capletters * self.cw_pos * scope * (1 - self.cw_neg) * (1 - self.cw_post) * (1 - cl('TIMES')) * (1 - startx) * (1 - cl('FREQ2')) * (1 - cl('PARTW'))
        capB = scope * cl('LETTER') * maximum(self.lw_neg, self.lw_post * flip) * (1 - cl('FREQ2')) * (1 - cl('TIMES'))
        m_q = cl('QUOTEW') * (1 - cl('SINGLE')) * maximum(cl('QMARKW'), cl('WRAPS')) * maximum(cl('WRAPV'), cl('INW')) \
            * cl('WHOLE2') * (1 - cl('ENDW') * (1 - se('STARTW'))) * (1 - negc)
        negopen = x('NEG') * self.wlead_now * self.clause_sent
        self.negopen = maximum(negopen, (1 - self.cb_prev) * self.negopen)
        reqm = clamp(cl('REQANY') + self.negopen + self.imp_any, 0, 1)
        ce = cb * rqc * reqm
        self.s_impany = maximum(maximum(self.imp_any, self.negopen), (1 - self.sb_prev) * self.s_impany)
        sayrep = x('SAYW') * (1 - self.wlead_now) * (self.age_neg > 2) * (1 - prevw('STOPF'))
        self.s_sayrep = maximum(sayrep, (1 - self.sb_prev) * self.s_sayrep)
        topic = clamp(se('EXPLAINV') * se('WHQ') + se('REPORTV') + self.s_sayrep + se('CODEW'), 0, 1)
        topic = clamp(topic + self.qsent, 0, 1)
        reqs = clamp(se('REQANY') + self.s_impany, 0, 1) * (1 - topic)
        negx = x('NEG') * (1 - cl('FORGETW'))
        ng = lambda nm, cue: self._cue_neg(nm, cue, negx, negc)
        n_two = ng('two', clamp(x('ASTER') + f('STAR6'), 0, 1)); n_title = ng('title', clamp(x('TITLEW') + f('ANGTOK') + x('ANGW'), 0, 1))
        n_bul = ng('bul', clamp(x('BULLETS') + x('BULLET1'), 0, 1)); n_par = ng('par', clamp(x('PARAW') + x('SECTW'), 0, 1))
        n_high = ng('high', clamp(x('HIGHW') + x('ITALW'), 0, 1)); n_nth = ng('nth', x('STARTV'))
        t_high = self._cue_neg('hightimes', clamp(x('HIGHW') + x('ITALW'), 0, 1), x('TIMES'), cl('TIMES'))
        run = f('LEADST') + where(f('ALLST') > 0.5, self.strun, self.strail * (f('LEADST') > 0))
        self.strun = where(f('ALLST') > 0.5, run, f('TRAILST')); self.strail = f('TRAILST')
        lit3 = (run >= 3) * (run < 6) * (1 - inq_prev) * (1 - insq_prev) * (1 - inbt_prev)
        lit6 = (run >= 6) * (1 - inq_prev) * (1 - insq_prev) * (1 - inbt_prev) * (1 - self.inbt)
        six = (self.num_recent == 6) * (self.age_num <= 3)
        three = (self.num_recent == 3) * (self.age_num <= 3)
        self.se_six = maximum(maximum(maximum(f('STAR6') * (1 - inq_prev) * (1 - insq_prev) * (1 - inbt_prev) * (1 - self.inbt), lit6), x('ASTER') * six), (1 - self.sb_prev) * self.se_six)
        m_two = self.se_six * maximum(se('SEPV'), se('TWOW')) * sb * rqs * reqs * (1 - n_two)
        self.se_s3 = maximum(maximum(maximum(maximum(f('STAR3') * (1 - inq_prev), lit3), x('ASTER') * three), cl('DIVW') * cl('MARKDOWN')),
                             (1 - self.sb_prev) * self.se_s3)
        m_s3 = self.se_s3 * sb * rqs * reqs
        m_title = maximum(self.se_ang * se('TITLEW'), se('ANGW') * se('BRACKW') * clamp(se('TITLEW') + se('YOUW') + se('WHOLE2'), 0, 1)) \
            * sb * rqs * reqs * (1 - n_title)
        near = (self.age_num <= 3) * (1 - f('ISNUM')) * (1 - self.numdet) * (self.num_recent <= 20) * (self.num_recent >= 1)
        prev_b1 = prevw('BULLET1')
        b_hit = maximum(x('BULLETS') * near, x('POINTW') * prev_b1 * (self.age_num <= 4) * (1 - self.numdet) * (self.num_recent <= 20))
        p_hit = x('PARAW') * near
        s_hit = x('SECTW') * near * (1 - clamp(se('HIGHW') + se('ITALW') + se('BULLETW'), 0, 1))
        self.sc_bul = where(b_hit > 0.5, self.num_recent, self.sc_bul)
        self.sc_par = where(p_hit > 0.5, self.num_recent, self.sc_par)
        self.sc_sec = where(s_hit > 0.5, self.num_recent, self.sc_sec)
        okc = sb * u * rqs * reqs
        okb = okc * (1 - n_bul); okp = okc * (1 - n_par)
        self.N_bul = where((okb * (self.sc_bul > 0)) > 0.5, self.sc_bul, self.N_bul)
        self.N_par = where((okp * (self.sc_par > 0)) > 0.5, self.sc_par, self.N_par)
        self.N_sec = where((okp * (self.sc_sec > 0)) > 0.5, self.sc_sec, self.N_sec)
        m_bul = okb * (self.sc_bul > 0)
        m_para = okp * maximum((self.sc_par > 0), (self.sc_sec > 0))
        for k, m in [('nc', m_nc * ce), ('lower', maximum(lowA, lowB) * ce), ('caps', maximum(capA, capB) * ce), ('quote', m_q * ce),
                     ('ps', cl('PSW') * cl('PSV') * (1 - negc) * (1 - cl('EXPLAINV')) * ce)]:
            self.cand[k] = maximum(self.cand[k], m * u)
            self.L[k] = maximum(self.L[k], self.cand[k] * maximum(sb * (1 - nonreq_q), f('COLON')) * (1 - topic))
            self.cand[k] = where((sb + f('COLON')) > 0.5, 0, self.cand[k])
        for k, m in [('pps', cl('PPS')), ('two', m_two), ('star3', m_s3), ('title', m_title), ('bul', m_bul), ('para', m_para)]:
            self.L[k] = maximum(self.L[k], m * u * (1 - topic))
        # end phrase capture (352-387)
        self.imp_end = maximum(x('ENDW') * self.wlead, (1 - self.cb_prev) * self.imp_end)
        endok = maximum(cl('ENDV'), cl('LASTW') * clamp(cl('WITHW') + cl('BEW'), 0, 1))
        armed = endok * clamp(cl('REQMOD') + self.imp_any + self.imp_end * cl('WITHW'), 0, 1) * (1 - negc) * rqc \
            * (1 - cl('EACH')) * (1 - se('PARAW')) * u * (1 - self.L['end'])
        opq = maximum(f('OPENDQ'), f('OPENSQ')); qtype = f('OPENDQ') + (1 - f('OPENDQ')) * f('OPENSQ') * 2
        idle = self.ec == 0
        go_q = idle * armed * opq + (self.ec == 1) * self.e_cf * opq * u
        colon_ok = clamp(prevw('WITHW') + prevw('PHRASEW') + prevw('ENDV') + prevw('BEW'), 0, 1)
        go_c = idle * armed * f('COLON') * colon_ok * (1 - go_q)
        closer = where(self.eq_t == 1, f('CLOSEDQ'), f('CLOSESQ'))
        pre = f('PRE')
        in_q = (self.ec == 2) * u; in_c = (self.ec == 1) * (1 - self.e_cf) * u
        fin_q = in_q * closer
        fin_c_nl = in_c * maximum(f('NL'), f('IMEND'))
        fin_c_p = in_c * f('SENTB') * (1 - f('NL'))
        L_q = t - self.end_s
        self.end_x = where(fin_q > 0.5, pre, where(fin_c_nl > 0.5, where(f('NL') > 0.5, pre, -1), self.end_x))
        new_len = where(fin_c_p > 0.5, L_q + 1, L_q)
        fin = clamp(fin_q + fin_c_nl + fin_c_p, 0, 1)
        okl = ((new_len + (self.end_x >= 0)) >= 1) & (new_len <= 40)
        self.end_len = where(fin > 0.5, new_len, self.end_len)
        self.L['end'] = maximum(self.L['end'], fin * okl)
        self.end_sent = maximum(self.end_sent, fin * okl)
        undo = self.end_sent * nonreq_q
        self.L['end'] = where(undo > 0.5, 0, self.L['end']); self.ec = where(undo > 0.5, 0, self.ec)
        self.end_sent = where(sb > 0.5, 0, self.end_sent)
        self.ec = where(fin > 0.5, where(okl, 3, 0), self.ec)
        self.ec = where((in_q * (L_q > 40)) > 0.5, 0, self.ec)
        self.ec = where(go_q > 0.5, 2, where(go_c > 0.5, 1, self.ec))
        self.eq_t = where(go_q > 0.5, qtype, self.eq_t)
        self.end_s = where((go_q + go_c) > 0.5, t + 1, self.end_s)
        self.end_x = where((go_q + go_c) > 0.5, -1, self.end_x)
        self.e_cf = go_c
        # nth word capture (388-401)
        armw = (self.wc == 0) * se('STARTV') * x('WITHW') * u
        skip = clamp(x('SKIPNTH') + opq + f('COLON'), 0, 1)
        cap0 = (self.wc == 1) * f('ALPHA') * (1 - skip) * u
        cont = (self.wc == 2) * f('CONT')
        stop2 = (self.wc == 2) * (1 - f('CONT'))
        self.nth_s = where(cap0 > 0.5, t, self.nth_s)
        self.nth_len = where(cap0 > 0.5, 1, where(cont > 0.5, self.nth_len + 1, self.nth_len))
        self.wc = where(armw > 0.5, 1, where(cap0 > 0.5, 2, where(stop2 > 0.5, 3, self.wc)))
        self.wc = where((self.wc == 1) & (skip < 0.5) & (f('ALPHA') < 0.5) & (f('NONWS') > 0.5), 0, self.wc)
        self.wc = where((self.wc == 2) & (self.nth_s == t) & (x('ARTW') > 0.5), 0, self.wc)
        kval = where(ordv > 0, ordv, where((isnum > 0.5) & (self.age_para <= 2), numval, 0))
        self.s_k = where((kval * (u * (1 - cap0) * (1 - cont))) > 0, kval, self.s_k)
        self.s_klast = maximum(self.s_klast, x('LASTW') * u)
        # highlight count (402-406)
        hw = maximum(x('HIGHW'), x('ITALW'))
        self.s_hn_first = where((isnum * self.s_hseen * (self.s_hn_first == 0)) > 0.5, numval, self.s_hn_first)
        self.s_hn_last = where((isnum * (1 - self.s_hseen)) > 0.5, numval, self.s_hn_last)
        self.s_hseen = maximum(self.s_hseen, hw)
        # forbidden words (407-456)
        opener = opq
        cls_tok = where(self.rqt == 1, f('CLOSEDQ'), f('CLOSESQ'))
        sepish = clamp(x('SEPW') + (ids == 11) + f('COLON') + opener, 0, 1)
        open_w = (self.rg == 0) * x('WORDTERM') * u
        det_before = prevw('DETW') > 0.5
        open_q = (self.rg == 0) * opener * (self.age_neg <= 6) * u * (1 - open_w)
        inr = (self.rg == 1) * u
        self.rq_age = where(self.rq > 0.5, self.rq_age + 1, 0)
        q_abort = inr * self.rq * (self.rq_age > 12)
        self.s_abort = maximum(self.s_abort, q_abort)
        q_close = inr * self.rq * cls_tok
        q_open = inr * (1 - self.rq) * opener
        q_item = inr * self.rq * (1 - cls_tok) * f('LETTERS')
        u_item = inr * (1 - self.rq) * (1 - opener) * f('ALPHA') * (1 - x('STOPF')) * (1 - x('SEPW')) * (1 - self.rqonly) * (1 - self.rsawq)
        u_sep = inr * (1 - self.rq) * (1 - opener) * sepish
        firstbad = self.rfirst * inr * (1 - self.rq) * (1 - opener) * (1 - f('COLON')) * (1 - u_item)
        item = clamp(q_item + u_item, 0, 1) * (1 - firstbad)
        neww = item * clamp(f('WSTART') + self.lastsep, 0, 1)
        contw = item * (1 - clamp(f('WSTART') + self.lastsep, 0, 1))
        si = clamp(where(neww > 0.5, self.nslot, self.nslot - 1), 0, S_W - 1)
        wr = (((neww > 0.5) & (self.nslot < S_W)) | ((contw > 0.5) & (self.nslot > 0))) & (f('HASAN') > 0.5)
        cur = mux(self.slot_h, si)
        nh = where(neww > 0.5, f('HT'), modp(self.hashmul(cur, f('PIDX')) + f('HT')))     # (a): exact mod-p hash
        self.slot_h = [where(wr & (si == k), nh, self.slot_h[k]) for k in range(S_W)]
        self.nslot = where(wr & (neww > 0.5), self.nslot + 1, self.nslot)
        endreg = inr * (1 - self.rq) * (1 - opener) * (1 - u_item) * (1 - u_sep) * (1 - q_close)
        endreg = maximum(endreg, firstbad)
        rq_new = where((open_q + q_open) > 0.5, 1, where((q_close + q_abort) > 0.5, 0, self.rq))
        endreg = maximum(endreg, q_abort)
        endreg = maximum(endreg, inr * (1 - rq_new) * sb)
        self.aw = where(endreg > 0.5, 0, minimum(self.aw + 1, BIGAGE))
        inaw = self.aw <= 6
        self.aw_neg = where(endreg > 0.5, x('NEG'), maximum(self.aw_neg, inaw * x('NEG')))
        self.aw_app = where(endreg > 0.5, x('APPEARS'), maximum(self.aw_app, inaw * x('APPEARS')))
        self.s_commit = maximum(self.s_commit, self.aw_neg * self.aw_app + inaw * x('NEGADJ') * (self.aw > 0))       # can be 2
        self.s_commit = maximum(self.s_commit, endreg * self.rnegA)
        st_open = clamp(open_w + open_q, 0, 1)
        self.rnegA = where(st_open > 0.5, self.age_neg <= 8, self.rnegA)
        self.rq = rq_new
        self.rsawq = where(st_open > 0.5, open_q, maximum(self.rsawq, q_open))
        self.rqonly = where(open_w > 0.5, 1 - det_before, where(st_open > 0.5, 0, self.rqonly))
        self.rqt = where((open_q + q_open) > 0.5, f('OPENDQ') + (1 - f('OPENDQ')).scale(2), self.rqt)
        self.lastsep = where((st_open + q_open + q_close + u_sep) > 0.5, 1, where(item > 0.5, 0, self.lastsep))
        self.rfirst = where(open_w > 0.5, 1, where(inr > 0.5, 0, self.rfirst))
        self.rg = where(st_open > 0.5, 1, where(endreg > 0.5, 0, self.rg))
        # repeat (457-460)
        self.s_negrep = maximum(self.s_negrep, x('REPEATV') * (self.age_neg <= 4))
        self.age_you = where(x('YOURW') > 0.5, 0, minimum(self.age_you + 1, BIGAGE))
        self.s_negrep = maximum(self.s_negrep, x('REQTERM') * (self.age_you <= 2))
        # sentence finalisation (461-505)
        sbu = sb * u
        rep_hit = sbu * rqs * reqs * (1 - self.in_sys) * se('REPEATV') * se('REQTERM') * se('VERBATIM') * (1 - se('REFMT')) \
            * (1 - self.s_negrep) * (1 - self.rep_seen)
        self.rep_seen = maximum(self.rep_seen, rep_hit)
        below = se('BELOWD') * (1 - se('ABOVED'))
        self.rep_s = where(rep_hit * (1 - below) > 0.5, self.msg_start, self.rep_s)
        self.rep_len = where(rep_hit * (1 - below) > 0.5, clamp(self.prev_end - self.msg_start + 1, lo=0), self.rep_len)
        self.rep_below = where(rep_hit > 0.5, below, self.rep_below)
        self.rep_wait = where(rep_hit * below > 0.5, 1, self.rep_wait)
        nl_now = rep_hit * below * f('NL')
        self.rep_s = where(nl_now > 0.5, t + 1, self.rep_s)
        self.rep_wait = where(nl_now > 0.5, 2, self.rep_wait)
        self.L['repeat'] = maximum(self.L['repeat'], rep_hit * (1 - below) * (self.prev_end >= self.msg_start))
        nth_hit = sbu * rqs * reqs * (1 - n_nth) * se('STARTV') * se('PARAW') * (self.wc >= 2) * clamp((self.s_k > 0) + self.s_klast, 0, 1) * (1 - self.L['nth'])
        self.L['nth'] = maximum(self.L['nth'], nth_hit)
        self.nth_k = where(nth_hit > 0.5, self.s_k, self.nth_k); self.nth_last = where(nth_hit > 0.5, self.s_klast * (self.s_k == 0), self.nth_last)
        h_hit = sbu * rqs * reqs * (1 - n_high) * maximum(se('ITALW'), se('HIGHW') * clamp(se('MARKDOWN') + se('ASTERISK') + self._starc_se(tr)
                                                                                          + se('HIGHOBJ'), 0, 1)) * (1 - t_high)
        hn = where(self.s_hn_first > 0, self.s_hn_first, where(self.s_hn_last > 0, self.s_hn_last, 1))
        hn = where(hn > 20, 1, hn)
        self.N_high = where(h_hit * (1 - self.L['high']) > 0.5, hn, self.N_high)
        self.L['high'] = maximum(self.L['high'], h_hit)
        commit = sbu * rqs * reqs * self.s_commit * (1 - clamp(se('TIMES') + se('FREQ3'), 0, 1)) * (1 - self.rg * self.rq) * (1 - self.s_abort)
        keep = commit > 0.5
        is_new = [(self.nbase <= k) & (self.nslot > k) for k in range(S_W)]
        self.slot_ok = [where(keep & is_new[k], 1, self.slot_ok[k]) for k in range(S_W)]
        drop = (sbu > 0.5) & (1 - keep) & ((self.rg * self.rq) < 0.5)
        self.slot_h = [where(drop & is_new[k], 0, self.slot_h[k]) for k in range(S_W)]
        self.nslot = where(drop, self.nbase, self.nslot)
        self.nbase = where(sbu > 0.5, self.nslot, self.nbase)
        sok = self.slot_ok[0]
        for k in range(1, S_W): sok = sok + self.slot_ok[k]
        self.L['forbid'] = maximum(self.L['forbid'], sok > 0)
        self.lnw_pos = where((nonws * u) > 0.5, t, self.lnw_pos)
        self.prev_end = where(sbu > 0.5, self.lnw_pos, self.prev_end)
        rw = (self.rep_wait == 1) * u * maximum(f('NL'), f('COLON')) * (1 - rep_hit)
        self.rep_s = where(rw > 0.5, t + 1, self.rep_s)
        self.rep_wait = where(rw > 0.5, 2, self.rep_wait)
        endmsg = f('IMEND') * (self.rep_wait == 2) * u
        self.rep_len = where(endmsg > 0.5, clamp(self.lnw_pos - self.rep_s + 1, lo=0), self.rep_len)
        self.L['repeat'] = maximum(self.L['repeat'], endmsg * (self.lnw_pos >= self.rep_s))
        self.rep_wait = where(endmsg > 0.5, 0, self.rep_wait)
        rs = sb > 0.5
        for nm in ['s_k', 's_klast', 's_hn_first', 's_hn_last', 's_hseen', 's_negrep', 's_commit', 's_abort', 'sc_bul', 'sc_par', 'sc_sec']:
            setattr(self, nm, where(rs, 0, getattr(self, nm)))
        self.wc = where(rs & (self.wc < 3), 0, where(rs & (self.L['nth'] < 0.5), 0, self.wc))
        # turn scope (511-527)
        is_sys_role = (self.prevtok == 151644) & (ids == 8948)
        self.in_sys = where(is_role_tok > 0.5, is_sys_role, self.in_sys)
        for k in RULES: self.Ls[k] = maximum(self.Ls[k], (self.L[k] > L_before[k]) * self.in_sys)
        self.sys_nslot = where(self.in_sys > 0.5, self.nslot, self.sys_nslot)
        user_turn = (self.prevtok == 151644) & (ids == 872)
        ut = user_turn > 0.5
        for k in RULES: self.L[k] = where(ut, self.Ls[k], self.L[k])
        self.slot_ok = [where(ut & (self.sys_nslot <= k), 0, self.slot_ok[k]) for k in range(S_W)]
        self.rep_seen = where(ut, self.L['repeat'], self.rep_seen)
        self.ec = where(ut & (self.L['end'] < 0.5), 0, self.ec)
        self.wc = where(ut & (self.L['nth'] < 0.5), 0, self.wc)
        cancel = x('CANCELW') * u
        for k in RULES: self.L[k] = where(cancel > 0.5, self.L[k] * (1 - self.Ls[k]), self.L[k])
        self.slot_ok = [where((cancel > 0.5) & (self.sys_nslot > k), 0, self.slot_ok[k]) for k in range(S_W)]
        self.cb_prev, self.sb_prev = cb, sb
        self._answer_step(ids, tr, ans_start, new_turn)
        self.prevtok = ids
        self.t = self.t + 1

    # ================================================================== _answer_step (549-642)
    def _answer_step(self, ids, tr, ans_start, new_turn):
        f = tr
        reset = new_turn > 0.5
        act = self.prog > 0
        _, slen = self.case_seq(self.prog_token())
        sub = act & (self.cj + 1 < slen)
        self.cj = where(sub, self.cj + 1, 0)
        self.pj = where(act & (1 - sub), self.pj + 1, self.pj)
        plen = self.pre_len + self.sp_len + self.suf_len
        fin = act & (self.pj >= plen)
        for kind, nm in [(1, 'qopen'), (2, 'rep'), (3, 'title'), (4, 'nth')]:
            self.done[nm] = where(fin & (self.prog == kind), 1, self.done[nm])
        self.ps_seen = where(fin & (self.prog == 10), 1, self.ps_seen)
        self.end_done = where(fin & (self.prog == 11), 1, self.end_done)
        hf_set = fin & (self.prog == 9)
        self.prog_prev = self.prog
        self.forced_brk = where(fin & ((self.prog == 6) | (self.prog == 5)), 1, self.forced_brk)
        self.prog = where(fin, 0, self.prog)
        a = self.ans * (1 - ans_start)
        nonws = f('NONWS')
        idx0 = self.n == 0
        self.lq = where(nonws * a > 0.5, f('QE'), self.lq)
        self.lx = where(nonws * a > 0.5, f('QX') * idx0, self.lx)
        self.n = self.n + a
        nn_ = f('NN') * a
        brk = nn_ * clamp(self.seg + f('PRECNN'), 0, 1)
        self.kp = self.kp + brk
        self.last_brk = brk * (1 - f('STAR3'))
        s3 = f('STAR3') * a
        self.d3 = self.d3 + s3
        self.seg = where(nn_ + s3 > 0.5, f('POSTNN') * (1 - s3), maximum(self.seg, nonws * a))
        self.forced_brk = where((f('LETTERS') * a) > 0.5, 0, self.forced_brk)
        self.k6 = self.k6 + f('STAR6X') * a
        self.wa = self.wa + f('LETTERS') * f('WSTART') * a
        s6 = f('STAR6X') * a > 0.5
        k6p = self.k6 > 0
        self.w6 = where(s6, 0, self.w6 + f('NONWS') * f('WSTART') * a * k6p)
        self.w6l = where(s6, 0, self.w6l + f('LETTERS') * a * k6p)
        self.n6 = where(s6, 0, self.n6 + a * k6p)
        self.bcount = self.bcount + a * self.at_ls * f('BULST')
        first_letters = a * f('LETTERS') * (self.cur_pf < 0)
        self.cur_pf = where(first_letters > 0.5, f('HT'), self.cur_pf)
        self.ans_pf = where((a * f('LETTERS') * (self.ans_pf < 0)) > 0.5, f('HT'), self.ans_pf)
        pbrk = clamp(nn_ + s3, 0, 1) * (1 - f('POSTNN'))
        self.prev_pf = where((pbrk * (self.cur_pf >= 0)) > 0.5, self.cur_pf, self.prev_pf)
        self.cur_pf = where(pbrk > 0.5, -1, self.cur_pf)
        self.ln_bul = where((a * self.at_ls) > 0.5, f('BULST'), self.ln_bul)
        self.ln_cont = where((a * self.at_ls) > 0.5, f('LETTERS'), maximum(self.ln_cont, a * f('LETTERS')))
        self.at_ls = where(a > 0.5, f('ENDNL'), self.at_ls)
        self.at_ls = where(ans_start > 0.5, 1, self.at_ls)
        star = f('STARC') * a
        close_h = star * self.hl_open * (self.hl_cont > 0)
        self.hcount = self.hcount + close_h
        self.hforced = where(star * self.hl_open > 0.5, 0, where(hf_set, 1, self.hforced))
        self.hl_open = where(star > 0.5, 1 - self.hl_open, self.hl_open)
        self.hl_cont = where(star > 0.5, 0, self.hl_cont + a * f('LETTERS'))
        nlr = f('NL') * a
        self.hl_open = where(nlr > 0.5, 0, self.hl_open); self.hforced = where(nlr > 0.5, 0, self.hforced)
        t_prog = fin & (self.prog_prev == 3)
        self.t_open = where(t_prog > 0.5, 1, where((f('CLOSETITLE') + f('NL')) * a > 0.5, 0, self.t_open))
        # host-written P.S. (612-617)
        w2 = clamp(self.win[3], lo=0)
        low2 = self.feat(w2)('LOW')
        P_ = (low2 == self.tk['p']) | (self.feat(low2)('NOSP') == self.tk['p'])
        dS = f('LOW') == self.tk['.s']
        dP2 = low2 == self.tk['.p']
        self.ps_seen = maximum(self.ps_seen, a * ((P_ & dS) | (dP2 & dS)))
        # end phrase suffix match (618-630)
        pos = clamp(self.end_s + self.end_m, 0, MAXMEM - 1)
        want = where(self.end_m < self.end_len, self.G.attn(pos, MAXMEM, 0, V - 1), clamp(self.end_x, lo=0))
        w0 = self.G.attn(clamp(self.end_s, 0, MAXMEM - 1), MAXMEM, 0, V - 1)
        rw_, r0 = self.feat(want), self.feat(w0)
        cw = rw_('CANON'); cg = f('CANON'); cw0 = r0('CANON')
        match = ((cg == cw) & (cg >= 0)) | (f('NOSP') == rw_('NOSP'))
        restart = ((cg == cw0) & (cg >= 0)) | (f('NOSP') == r0('NOSP'))
        self.end_m = where(a > 0.5, where(match, self.end_m + 1, where(restart, 1, 0)), self.end_m)
        nlen = self.end_len + (self.end_x >= 0)
        matched = (self.end_m >= nlen) & (nlen > 0)
        self.end_done = maximum(self.end_done, matched & (a > 0.5))
        self.end_done = where((a > 0.5) & (1 - matched) & (self.prog != 11) & (nonws > 0.5) & ((f('PRECNN') + f('LETTERS')) > 0),
                              where(match, self.end_done, 0), self.end_done)
        self.end_m = where(matched, 0, self.end_m)
        # rolling hash of the answer word (631-634)  (a)
        hn = where(f('LEAD') > 0.5, f('HT'), modp(self.hashmul(self.hcur, f('PIDX')) + f('HT')))
        hn = where((f('HASAN') < 0.5) | (f('TRAIL') > 0.5), 0, hn)
        self.hcur = where(a > 0.5, hn, where(ans_start > 0.5, 0, self.hcur))
        # new turn / answer start (635-641)
        self.ans = where(ans_start > 0.5, 1, where(reset, 0, self.ans))
        ra = reset | (ans_start > 0.5)
        for nm in ['n', 'lq', 'lx', 'seg', 'kp', 'd3', 'k6', 'w6', 'w6l', 'n6', 'wa', 'forced_brk', 'bcount', 'hl_open', 'hl_cont', 'hcount',
                   'hforced', 't_open', 'ps_seen', 'end_done', 'body', 'ln_bul', 'ln_cont']:
            setattr(self, nm, where(ra, 0, getattr(self, nm)))
        for k in self.done: self.done[k] = where(ra, 0, self.done[k])
        self.prog = where(ra, 0, self.prog); self.cj = where(ra, 0, self.cj)
        for nm in ['cur_pf', 'prev_pf', 'ans_pf']: setattr(self, nm, where(ra, -1, getattr(self, nm)))
        self._triggers(tr)

    def _triggers(self, tr):
        f = tr; g = self.gates(); a = self.ans; tk = self.tk
        idle = self.prog == 0
        nb = a * (1 - self.body) * idle
        q_go = nb * g['quote'] * (1 - self.done['qopen'])
        r_go = nb * g['repeat'] * (1 - self.done['rep']) * (1 - q_go)
        t_go = nb * g['title'] * (1 - self.done['title']) * (1 - q_go) * (1 - r_go)
        kk = where(self.nth_last > 0.5, g['npar'], self.nth_k)
        n_go = nb * g['nth'] * (kk == 1) * (1 - self.done['nth']) * (1 - q_go) * (1 - r_go) * (1 - t_go)
        self.start_prog(q_go, 1, pre=[tk['"']])
        self.start_prog(r_go, 2, span=(self.rep_s, self.rep_len), suf=[tk['\n\n']])
        self.start_prog(t_go, 3, pre=[tk['<<']])
        self.start_prog(n_go, 4, span=(self.nth_s, self.nth_len), fmap=2)
        self.body = maximum(self.body, nb * (1 - q_go) * (1 - r_go) * (1 - t_go) * (1 - n_go))
        idle = self.prog == 0; bd = self.body * idle * a
        brk = f('NN') * (1 - f('POSTNN'))
        s_go = bd * g['p3'] * brk * (self.d3 < g['npar'] - 1) * self.last_brk
        self.start_prog(s_go, 5, pre=[tk['***\n\n']])
        n2 = bd * g['nth'] * brk * (self.kp == kk - 1) * (kk >= 2) * (1 - self.done['nth']) * (1 - s_go)
        self.start_prog(n2, 4, span=(self.nth_s, self.nth_len), fmap=2)

    # ================================================================== write (698-786)
    def write(self):
        G = self.G; g = self.gates(); a = self.ans; tk = self.tk
        cnt = where(g['p3'] > 0.5, self.d3, self.kp)
        short = g['para'] * (cnt < g['npar'] - 1)
        eosb = clamp(g['two'] * (self.k6 > 0) * (self.w6 < 5) * (self.n6 < 40) + self.hforced * self.hl_open * (self.hl_cont == 0)
                     + short * (1 - self.seg) + g['bul'] * self.ln_bul * (self.ln_cont == 0), 0, 1)
        nnb = g['pn'] * (self.kp >= g['npar'] - 1)
        coef = [g['nc'],                                                                       # x COMMASET
                clamp(g['two'] * (self.k6 > 0) + g['p3'] + g['two'] * (self.k6 == 0) * (self.wa < 5), 0, 1),   # x STAR3SET
                nnb,                                                                           # x NN
                nnb * self.at_ls,                                                              # x STARTNL
                eosb,                                                                          # x EOSMASK
                g['bul'] * self.at_ls * (self.bcount >= self.N_bul),                           # x BULST
                g['bul'] * self.ln_bul * (self.ln_cont == 0) * (1 - self.at_ls)]               # x NL
        # forbidden words (717-721): targets and hcur * 131^len mod p for every distinct power (exact, see hashmul)
        live = [(self.slot_ok[k] > 0.5) & (g['forbid'] > 0.5) for k in range(S_W)]
        tgt = [where(live[k], self.slot_h[k], -5) for k in range(S_W)]
        bits = bits31(self.hcur)
        ymod = []
        for p in self.tb['POWS'].tolist():
            y = None
            for i in range(31):
                c = (pow(2, i) * int(p)) % HP
                term = bits[i].scale(c)
                y = term if y is None else y + term
            ymod.append(modp(y))
        rep_on = clamp(g['para'] + g['high'], 0, 1) * self.forced_brk * (1 - self.seg) * (self.prev_pf >= 0)
        two_on = g['two'] * (self.k6 > 0) * (self.w6l == 0) * (self.ans_pf >= 0)
        pf = where(two_on > 0.5, self.ans_pf, where(rep_on > 0.5, self.prev_pf, -5))
        hz0 = self.hcur == 0
        ins = OrderedDict()
        for i, c in enumerate(coef): ins[f'coef{i}'] = c
        for j, y in enumerate(ymod): ins[f'ymod{j}'] = y
        for k in range(S_W): ins[f'tgt{k}'] = tgt[k]
        ins['pf'] = pf; ins['hz0'] = hz0
        vo = G.vocab(ins, OrderedDict(top=(0, V - 1), blocked=(0, 1), alt=(0, V - 1)))
        top, blocked, alt = vo['top'], vo['blocked'], vo['alt']
        chosen = where(blocked > 0.5, alt, top)
        cr = self.feat(chosen)
        e_eos = cr('EOSMASK'); e_nl = cr('NLMASK')
        idle = (self.prog == 0) * a * self.body
        tc = idle * g['title'] * self.t_open * clamp(e_nl + e_eos, 0, 1)
        hc = idle * self.hforced * self.hl_open * (self.hl_cont > 0) * clamp(e_nl + e_eos, 0, 1) * (1 - tc)
        self.start_prog(tc, 13, pre=[tk['>>']])
        self.start_prog(hc, 14, pre=[tk['*']])
        e = idle * e_eos * (1 - tc) * (1 - hc)
        c1 = e * short * self.seg
        c2 = e * g['two'] * (self.k6 == 0) * (1 - c1)
        r = (1 - c1) * (1 - c2)
        c3 = e * g['bul'] * (self.bcount < self.N_bul) * r; r = r * (1 - c3)
        c4 = e * g['high'] * (self.hcount < self.N_high) * r; r = r * (1 - c4)
        c5 = e * g['ps'] * (1 - self.ps_seen) * r; r = r * (1 - c5)
        c6 = e * g['end'] * (1 - self.end_done) * r; r = r * (1 - c6)
        c7 = e * g['quote'] * (1 - self.lq * (1 - self.lx)) * r
        nn, nl = tk['\n\n'], tk['\n']
        sep = where(g['pn'] > 0.5, where(self.at_ls > 0.5, -1, nl), nn)
        endsep = where(g['pn'] > 0.5, 220, nn)
        hsep = where(g['pn'] > 0.5, 220, nn)
        qsuf = where(g['quote'] > 0.5, tk['"'], -1)
        self.start_prog(c1, 6, pre=[nn])
        self.start_prog(c2, 7, pre=[nn, tk['******'], nn])
        self.start_prog(c3, 8, pre=[where(self.at_ls > 0.5, tk['*'], nl), where(self.at_ls > 0.5, -1, tk['*'])])
        self.start_prog(c4, 9, pre=[hsep, tk['*']])
        pps = self.L['pps'] > 0.5
        self.start_prog(c5 * pps, 10, pre=[sep, tk['P'], tk['.P'], tk['.S']], suf=[tk['.']])
        self.start_prog(c5 * (1 - pps), 10, pre=[sep, tk['P'], tk['.S'], tk['.']])
        xx = where(self.end_x >= 0, self.end_x, -1)
        self.start_prog(c6, 11, pre=[endsep], span=(self.end_s, self.end_len), suf=[xx, qsuf, EOS], fmap=1)
        self.start_prog(c7, 12, pre=[tk['"'], EOS])
        free = idle * (1 - clamp(tc + hc + c1 + c2 + c3 + c4 + c5 + c6 + c7, 0, 1))
        bl = free * g['bul'] * self.at_ls * (self.bcount < self.N_bul) * (1 - cr('BULST')) \
            * (1 - cr('EOSMASK')) * (1 - cr('STARTNL')) * (1 - cr('ANGTOK'))
        self.start_prog(bl, 8, pre=[tk['-']])
        free = free * (1 - bl)
        need_case = g['caps'] * cr('LOWERSET') + g['lower'] * cr('UPPERSET')
        self.start_prog(free * clamp(blocked + need_case, 0, 1), 15, pre=[chosen])
        # (b) hole compaction of suf/pre (776-779): prefix counts + one-hot selection
        self.suf, self.suf_len = compact(self.suf)
        self.pre, self.pre_len = compact(self.pre)
        active = (self.prog > 0) & (a > 0.5)
        tokf = self.emit_token()
        widx = where(active, tokf, V)                    # row V of the force table is the zero row (silent)
        self.fires = self.fires + active
        return dict(widx=widx, active=active, tokf=tokf, top=top, blocked=blocked, alt=alt, chosen=chosen)


# ------------------------------------------------------------------------------------------------ exact helpers
def bits31(x):
    """bits of an integer wire x in [0, 2^31): b_30..b_0 by a ReLU staircase."""
    assert x.lo >= 0 and x.hi < 2 ** 31, (x.lo, x.hi)
    G = x.G
    r = G.mat(x); bits = [None] * 31
    for i in range(30, -1, -1):
        b = G.mat(ge(r, 2 ** i)); bits[i] = b
        r = G.mat(assume(r - b.scale(2 ** i), 0, 2 ** i - 1))
    return bits


def modp(y):
    """y mod p for an integer wire 0 <= y < 32 p."""
    assert y.lo >= 0 and y.hi < 32 * HP, (y.lo, y.hi)
    q = None
    for k in range(1, y.hi // HP + 1):
        s = ge(y, k * HP)
        q = s if q is None else q + s
    out = y - q.scale(HP) if q is not None else y
    return assume(out, 0, HP - 1)


def compact(regs):
    """stable compaction of the 4 registers: values >= 0 first (order kept), then -1 holes. (circuit: torch.sort of (x<0))"""
    G = regs[0].G if isinstance(regs[0], Sym) else regs[1].G
    regs = [S(G, r) for r in regs]
    valid = [r >= 0 for r in regs]
    cnt = [S(G, 0)]
    for v in valid: cnt.append(cnt[-1] + v)                    # prefix counts
    out = []
    for i in range(4):
        acc = S(G, -1)
        for j in range(i, 4):                                  # element j can only move left
            sel = valid[j] & (cnt[j] == i)
            acc = acc + sel * (regs[j] + 1)
        out.append(Sym(G, acc.lin, -1, max(r.hi for r in regs)))
    return out, cnt[4]


def build(T):
    """-> (step program, write program, register spec, tables)"""
    from sym import compile_graph
    tb = build_tables(T)
    spec = register_spec(tb['CLASSES'])
    progs = {}
    for which in ['step', 'write']:
        G = Graph()
        G.table('FEAT', tb['FEAT'][1], [(float(tb['FEAT'][0][:, i].min()), float(tb['FEAT'][0][:, i].max())) for i in range(tb['FEAT'][0].shape[1])], V)
        G.table('CODE2', ['CODE'], [(0, tb['NCODE'] - 1)], V + 2)
        G.table('ONEHOT', list(range(tb['NCODE'])), [(0, 1)] * tb['NCODE'], tb['NCODE'])
        G.table('PCOEF', [f'c{i}' for i in range(31)], [(0, HP - 1)] * 31, len(tb['POWS']))
        G.table_values = {}
        B = Build(G, tb, spec)
        if which == 'step':
            ids = G.inp('ids', 0, V - 1)
            B.step(ids)
            outs = B.outputs()
        else:
            w = B.write()
            outs = B.outputs()
            for k, v in w.items(): outs['@' + k] = S(G, v)
        prog = compile_graph(G, outs)
        prog['G'] = G; prog['assumed'] = B.assumed
        progs[which] = prog
    return progs, spec, tb
