"""PATCHED COPY of Team D's frozen circuit.py (md5 cd15769ce3f26bc5b5480e0f8864e4be) for CUDA-graph capture.
Only change: the step counter self.t is a 0-d device tensor (was a Python int) and the uses of t are written as
tensor ops. The computation is identical (verified bit-exact against the frozen file by test_circuit_g.py)."""
"""Compiled instruction keepers ("implants") for a frozen chat LM. Everything below is tensor arithmetic on 0/1 wires, small
integer registers, fixed lookup tables over the vocabulary and hard-attention reads of earlier positions; the same code runs
for every prompt (no Python branching on data, no regex, no Python rule objects at inference).

Interface (run.py): step(ids) once per token (prompt and answer); write(h) inside the host's final-norm pre-hook returns the
vector added to the host's last hidden state. Silent (no instruction latched) -> every gate is 0 -> write is exactly 0.

Building blocks and their weight-space reading (see NOTES.md):
  READER     token id -> class flags: fixed 0/1 table rows (= an embedding matrix) + an n-gram matcher over the last 5 token
             ids (= AND of position-shifted one-hot compares) for words that the tokenizer splits.
  SCOPE HEADS  'has class X since the last clause/sentence boundary' = a hard-attention head that reads the most recent position
             whose key is (X or boundary) and copies X (incremental form: c = X + (1-X)(1-B_prev)c).
  RECENCY    tokens since the last X (a hard-attention head over position, copying the distance), capped.
  LATCHES    max over earlier positions (attention to any match).
  CAPTURE    argument spans (end phrase, nth word, repeat span, forbidden words) are kept as POSITIONS (pointer registers) set
             by 0/1 transition gates; the token ids are read back later by pointer (copy) heads into the prompt memory.
  ANSWER TRACKERS  counters (uniform-attention count of a flag), 'last non-space token', line/paragraph state.
  EMITTER    'force token t' writes = the rms-normalised unembedding row of t / final-norm gain, times a large gain (a few
             hundred rare tokens use a solved direction instead, force_fix, fitted on host states from OUR dev answers);
             the token comes from a fixed table, a pointer/copy head into the prompt memory, or the readout below.
             Case rules respell every emitted token through fixed tables (UPSEQ / LOWSEQ: a token's upper/lower-case
             spelling as <= 6 tokens) - the "case expansion" of the emitter.
  READOUT (DISCLOSED)  the host's own next-token logits on its final state, read through the frozen LM head (no write yet).
             Used for: the identity of the host's top token (is it end-of-turn / a newline / in a currently blocked set /
             of the wrong case), and, when the top token is blocked, the host's best token OUTSIDE the blocked set (masked
             hard-max = hard attention over the unembedding rows). Blocked sets are 0/1 vocabulary masks = fixed tables x
             circuit gates (commas; '***'; '\n\n'; end-of-turn; bullet starts; tokens completing a forbidden word; the first
             word of the previous paragraph at a forced continuation). No solved veto/kill vectors are used (v2).
"""
import os, torch
from tables import build, CLASSES
DEV = os.environ.get('CIRC_DEV', 'cuda')

CI = {c: i for i, c in enumerate(CLASSES)}
BIGAGE = 99.0
S_W = 6                          # forbidden-word slots
HP = 2147483647
MAXMEM = 8192
EOS = 151645
GAIN = 1e4


class Circuit:
    def __init__(self, tok, model, vecs=None):
        import os; vecs = vecs or os.environ.get('VECS', 'vecs.pt')
        from common import vocab_strings
        self.tok, self.model = tok, model
        strs = vocab_strings(tok)
        self.T = build(tok, strs, device=DEV)
        import os
        dev = DEV
        d = model.lm_head.weight.shape[1]
        V = torch.load(vecs) if os.path.exists(vecs) else {'veto': {k: torch.zeros(d) for k in ['comma', 'upper', 'lower', 'star3', 'nn', 'nlstart', 'eos', 'bulst']},
                                                             'kill_M': torch.zeros(d, d), 'force_fix': {}}
        self.E = model.lm_head.weight                       # (V, d) bf16, tied with the input embeddings
        self.g = model.model.norm.weight.float(); self.eps = model.model.norm.variance_epsilon
        self.fix_idx = torch.full((self.E.shape[0],), -1, dtype=torch.long, device=dev)
        fx = V['force_fix']
        if fx:
            ks = sorted(fx); self.fix_idx[torch.tensor(ks, device=dev)] = torch.arange(len(ks), device=dev)
            self.fix_rows = torch.stack([fx[k] for k in ks]).to(dev).float()
        else: self.fix_rows = torch.zeros(1, self.E.shape[1], device=dev)
        t = self.T['TOK']; self.tk = {k: torch.tensor(v, device=dev) for k, v in t.items()}
        self.EOSMASK = torch.zeros(self.E.shape[0], device=dev); self.EOSMASK[[151645, 151643]] = 1
        self.NLMASK = self.T['NL'].clone(); self.NLMASK[self.T['CLOSETITLE'] > 0] = 0        # newline tokens that do not close a title

    # ------------------------------------------------------------------ state
    def reset(self, B):
        dev = DEV; z = lambda *s: torch.zeros(*s, device=dev); zl = lambda *s: torch.zeros(*s, dtype=torch.long, device=dev)
        C = len(CLASSES)
        self.B = B; self.t = torch.zeros((), dtype=torch.long, device=dev)
        self.mem = zl(B, MAXMEM)                                   # prompt/answer token memory (the copy heads' keys/values)
        self.win = torch.full((B, 5), -2, dtype=torch.long, device=dev)
        self.cl = z(B, C); self.se = z(B, C); self.cb_prev = z(B); self.sb_prev = z(B)
        self.asst = z(B); self.hdr = z(B)                          # role register (turn head)
        self.age_neg = z(B) + BIGAGE; self.age_num = z(B) + BIGAGE; self.age_para = z(B) + BIGAGE
        self.acc = z(B); self.prevdig = z(B); self.num_recent = z(B)
        # latches / params
        R = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'pps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul', 'star3']
        self.L = {r: z(B) for r in R}; self.Ls = {r: z(B) for r in R}; self.sys_nslot = zl(B); self.in_sys = z(B)
        self.N_par = z(B); self.N_sec = z(B); self.N_bul = z(B); self.N_high = z(B); self.nth_k = z(B); self.nth_last = z(B)
        # end phrase capture
        self.ec = z(B); self.eq_t = z(B); self.e_cf = z(B); self.end_s = zl(B); self.end_len = zl(B); self.end_x = zl(B) - 1
        # nth word capture
        self.wc = z(B); self.nth_s = zl(B); self.nth_len = zl(B); self.s_k = z(B); self.s_klast = z(B)
        self.s_hn_first = z(B); self.s_hn_last = z(B); self.s_hseen = z(B)
        # repeat
        self.lnw_pos = zl(B); self.prev_end = zl(B); self.msg_start = zl(B); self.rep_s = zl(B); self.rep_len = zl(B)
        self.rep_wait = z(B); self.rep_below = z(B); self.s_negrep = z(B); self.rep_seen = z(B)
        # forbidden words
        self.rg = z(B); self.rq = z(B); self.rqt = z(B); self.rsawq = z(B); self.rqonly = z(B); self.rq_age = z(B); self.s_abort = z(B); self.lastsep = z(B); self.rnegA = z(B); self.rfirst = z(B)
        self.slot_h = zl(B, S_W); self.slot_ok = z(B, S_W); self.nslot = zl(B); self.nbase = zl(B); self.hcur = zl(B)
        self.s_commit = z(B); self.aw = z(B) + BIGAGE; self.aw_neg = z(B); self.aw_app = z(B)
        # answer side
        self.ans = z(B); self.n = z(B); self.body = z(B)
        self.lq = z(B); self.lx = z(B); self.seg = z(B); self.kp = z(B); self.d3 = z(B); self.k6 = z(B); self.w6 = z(B); self.n6 = z(B); self.prog_prev = zl(B)
        self.at_ls = z(B); self.bcount = z(B); self.hl_open = z(B); self.hl_cont = z(B); self.hcount = z(B); self.hforced = z(B)
        self.t_open = z(B); self.ps_seen = z(B); self.end_m = zl(B); self.end_done = z(B); self.km = zl(B, S_W)
        self.done = {k: z(B) for k in ['qopen', 'rep', 'title', 'nth']}
        self.prevtok = zl(B) - 1; self.last_brk = z(B); self.ln_bul = z(B); self.ln_cont = z(B)
        self.cur_pf = zl(B) - 1; self.prev_pf = zl(B) - 1; self.ans_pf = zl(B) - 1; self.wa = z(B)
        # emitter (program registers)
        self.prog = zl(B); self.pj = zl(B); self.cj = zl(B); self.pre = zl(B, 4); self.pre_len = zl(B); self.sp_s = zl(B); self.sp_len = zl(B)
        self.suf = zl(B, 4); self.suf_len = zl(B); self.fmap = zl(B)
        self.fires = z(B); self.se_star = z(B); self.ffail = z(B); self.inq = z(B); self.insq = z(B); self.inbt = z(B); self.strun = z(B); self.strail = z(B); self.cand = {k: z(B) for k in ['nc', 'lower', 'caps', 'quote', 'ps']}; self.dm = z(B); self.se_data = z(B); self.lw_neg = z(B); self.lw_pos = z(B); self.cw_neg = z(B); self.cw_pos = z(B); self.lw_seen = z(B); self.cw_seen = z(B); self.lw_post = z(B); self.cw_post = z(B);
        self.s_impany = z(B); self.s_negcue = z(B); self.cueR = {}; self.age_you = z(B) + BIGAGE; self.w6l = z(B); self.qsent = z(B); self.qpend = z(B); self.s_sayrep = z(B); self.clause_sent = z(B) + 1; self.flip_c = z(B); self.end_sent = z(B); self.se_s3 = z(B); self.forced_brk = z(B); self.t_mine = z(B); self.se_ang = z(B); self.cpend = z(B); self.cvalid = z(B); self.capsall = z(B); self.se_six = z(B); self.numdet = z(B); self.prevdig_before = z(B)
        self.sc_bul = z(B); self.sc_par = z(B); self.sc_sec = z(B); self.imp_end = z(B); self.imp_pos = z(B) + 1; self.wlead = z(B); self.wlead_now = z(B); self.imp_any = z(B); self.negopen = z(B)

    # ------------------------------------------------------------------ helpers
    def tab(self, name, ids): return self.T[name][ids]

    def classes(self, ids):
        prev = self.win[:, -2].clamp(min=0)
        wst = torch.clamp(self.T['WSTART'][ids] + (1 - self.T['ENDALNUM'][prev]), 0, 1)     # token starts a word
        f = self.T['T'][ids] * wst[:, None]                                      # (B, C) single-token lookup (word-initial tokens only)
        P = self.T['P']                                                          # (Np, 5) right-aligned patterns, -1 = wildcard
        m = ((self.win[:, None, :] == P[None]) | (P[None] < 0)).all(-1).float()  # (B, Np)  AND of shifted compares
        return torch.clamp(f + m @ self.T['PC'], 0, 1)

    def start_prog(self, gate, kind, pre=(), span=None, suf=(), fmap=0):
        """gate (B,) 0/1: load a program into the emitter registers where gate=1 and the emitter is idle."""
        g = (gate > 0.5) & (self.prog == 0)
        dev = DEV
        def pad(lst):
            out = torch.full((self.B, 4), -1, dtype=torch.long, device=dev)
            for i, x in enumerate(lst):
                out[:, i] = x if torch.is_tensor(x) else torch.tensor(x, device=dev)
            return out
        P = pad(pre); S = pad(suf)
        pl = (P >= 0).sum(1); sl = (S >= 0).sum(1)
        ss, sn = span if span is not None else (torch.zeros_like(self.prog), torch.zeros_like(self.prog))
        self.prog = torch.where(g, torch.full_like(self.prog, kind), self.prog); self.pj = torch.where(g, 0, self.pj); self.cj = torch.where(g, 0, self.cj)
        self.pre = torch.where(g[:, None], P, self.pre); self.pre_len = torch.where(g, pl, self.pre_len)
        self.suf = torch.where(g[:, None], S, self.suf); self.suf_len = torch.where(g, sl, self.suf_len)
        self.sp_s = torch.where(g, ss, self.sp_s); self.sp_len = torch.where(g, sn, self.sp_len)
        self.fmap = torch.where(g, torch.full_like(self.fmap, fmap), self.fmap)
        return g.float()

    def prog_token(self):
        j = self.pj; pl, sl = self.pre_len, self.sp_len
        a = self.pre.gather(1, j.clamp(0, 3)[:, None])[:, 0]
        k = (j - pl).clamp(min=0)
        pos = (self.sp_s + k).clamp(0, MAXMEM - 1)
        b = self.mem.gather(1, pos[:, None])[:, 0]                              # copy head: read the prompt token at the pointer
        first = (k == 0) & (j >= pl)
        b = torch.where(first & (self.fmap == 1), self.T['NOSP'][b], b)
        b = torch.where(first & (self.fmap == 2), self.T['CAP1'][b], b)
        c = self.suf.gather(1, (j - pl - sl).clamp(0, 3)[:, None])[:, 0]
        t = torch.where(j < pl, a, torch.where(j < pl + sl, b, c))
        return t.clamp(min=0)

    def case_seq(self, t):
        """case expansion of an emitted token under an active case rule: its upper/lower-case spelling (fixed tables, <= 6 tokens)"""
        gg = self.gates()
        need_u = (gg['caps'] > 0.5) & (self.T['LOWERSET'][t] > 0.5)
        need_l = (gg['lower'] > 0.5) & (self.T['UPPERSET'][t] > 0.5)
        one = torch.full((self.B, 6), -1, dtype=torch.long, device=DEV); one[:, 0] = t
        seq = torch.where(need_u[:, None], self.T['UPSEQ'][t], torch.where(need_l[:, None], self.T['LOWSEQ'][t], one))
        seq = torch.where((seq[:, :1] < 0), one, seq)
        return seq, (seq >= 0).sum(1)

    def emit_token(self):
        seq, n = self.case_seq(self.prog_token())
        return seq.gather(1, self.cj.clamp(max=5)[:, None])[:, 0].clamp(min=0)

    # ------------------------------------------------------------------ per token
    @torch.no_grad()
    def step(self, ids):
        T = self.T; B = self.B; t = self.t
        f = lambda n: T[n][ids]
        self.mem.index_copy_(1, t.clamp(max=MAXMEM - 1).view(1), ids[:, None])
        L_before = {k: v.clone() for k, v in self.L.items()}
        self.win = torch.cat([self.win[:, 1:], ids[:, None]], 1)
        X = self.classes(ids)
        # quoted text is reported speech, not a request: cue words inside double quotes do not count (quote-parity register)
        inq_prev = self.inq
        self.inq = torch.where(f('DQOPEN') > 0.5, 1.0, torch.where(f('DQCLOSE') > 0.5, 0.0, torch.remainder(self.inq + f('DQC'), 2)))
        self.inq = torch.where((f('NL') + f('IMEND')) > 0.5, 0.0, self.inq)
        # single-quoted spans too: opened by a (space/bracket/colon +) "'" token, closed by a "'" not followed by a letter
        insq_prev = self.insq
        self.insq = torch.where(self.insq > 0.5, 1 - f('CLOSESQ'), f('OPENSQ'))
        self.insq = torch.where((f('NL') + f('IMEND')) > 0.5, 0.0, self.insq)
        inbt_prev = self.inbt
        self.inbt = torch.where((f('NL') + f('IMEND')) > 0.5, 0.0, torch.remainder(self.inbt + f('BTC'), 2))            # `code`
        # data after a colon: 'Translate this into Spanish: <text>', 'Here is the email: <text>' -> the text is data, not a request
        Xr = self.T['T'][ids]
        dm_prev = self.dm
        arm = f('COLON') * torch.clamp(self.se_data, 0, 1)
        self.dm = torch.where(arm > 0.5, 1.0, self.dm)
        self.dm = torch.where((self.dm == 1) & (f('NONWS') > 0.5) & (arm < 0.5), 2.0, self.dm)
        self.dm = torch.where(((self.dm == 2) & (f('NL') > 0.5)) | (f('NN') > 0.5) | (f('IMEND') > 0.5), 0.0, self.dm)
        mask = torch.maximum(torch.maximum(torch.maximum(inq_prev, self.inq), torch.maximum(insq_prev, self.insq)),
                             torch.maximum(torch.maximum(inbt_prev, self.inbt), (dm_prev > 0).float()))
        self.se_data = torch.maximum(torch.clamp(Xr[:, CI['DATAV']] + Xr[:, CI['DATAW']], 0, 1) * (1 - mask), (1 - self.sb_prev) * self.se_data)
        X = X * (1 - mask)[:, None]
        cb = torch.clamp(f('CLAUSEB') + f('IMEND') + X[:, CI['ANDBUT']], 0, 1)
        sb = torch.clamp(f('SENTB') + f('IMEND'), 0, 1)
        # imperative position: the token opens its clause (optionally after please/also/then/and ...)
        lead = torch.maximum(self.cb_prev, self.imp_pos)
        wst = torch.maximum(f('WSTART'), 1 - T['ENDALNUM'][self.win[:, -2].clamp(min=0)])          # token starts a word
        self.wlead = torch.where(wst > 0.5, lead, self.wlead)                                  # value at the start of the current word
        self.wlead_now = self.wlead * (1 - X[:, CI['IMPPRE']])
        self.imp_pos = torch.where((f('NONWS') * (1 - X[:, CI['IMPPRE']])) > 0.5, torch.zeros_like(lead), lead)
        # any word at imperative position that is an instruction verb (write/use/keep/... or an end-word)
        # imperative opening: the clause opens with a content word that is not a subject/determiner/question/preposition word
        # (requests open with a verb: 'Write ...', 'Highlight ...'; descriptions open with 'My friend ...', 'The tool ...', 'I ...')
        opener = self.wlead_now * X[:, CI['INSTV']]                                          # an instruction verb opens the clause
        self.clause_sent = torch.where(self.cb_prev > 0.5, self.sb_prev, self.clause_sent)  # this clause opens a sentence
        self.imp_any = torch.maximum(opener, (1 - self.cb_prev) * self.imp_any)
        auxopen = self.wlead_now * X[:, CI['AUXW']] * torch.maximum(self.clause_sent, self.sb_prev)
        nxt_word = (self.qpend > 0.5) & (wst > 0.5) & (T['ALPHA'][ids] > 0.5) & (auxopen < 0.5)
        self.qsent = torch.where(self.sb_prev > 0.5, 0.0, self.qsent)
        self.qsent = torch.where(nxt_word, 1 - X[:, CI['YOUW']], self.qsent)
        self.qpend = torch.where(auxopen > 0.5, 1.0, torch.where(nxt_word | (self.sb_prev > 0.5), 0.0, self.qpend))
        self.cl = torch.maximum(X, (1 - self.cb_prev)[:, None] * self.cl)
        self.se = torch.maximum(X, (1 - self.sb_prev)[:, None] * self.se)
        cl = lambda c: self.cl[:, CI[c]]; se = lambda c: self.se[:, CI[c]]; x = lambda c: X[:, CI[c]]
        nonws = f('NONWS')
        # a question that is not a request (no 'please', no 'can/could/would/will you') and a clause describing someone else's
        # behaviour (third-person subject / third-person verb form) are not instructions
        qm = f('QMARK') * (1 - torch.maximum(torch.maximum(inq_prev, self.inq), torch.maximum(insq_prev, self.insq)))   # a '?' inside quotes is quoted text
        nonreq_q = qm * (1 - torch.maximum(se('PLEASEW'), se('YOUW') * se('MODALQ')))
        self.se_ang = torch.maximum(f('ANGTOK'), (1 - self.sb_prev) * self.se_ang)        # sentence contains a '<<' token
        # ---- roles: <|im_start|> role ... ; in-assistant register; answer start = '</think>' '\n\n' inside the assistant turn
        is_role_tok = (self.prevtok == 151644).float()
        self.asst = torch.where(is_role_tok > 0.5, (ids == 77091).float(), self.asst)          # 77091 = 'assistant'
        user = (1 - self.asst) * (1 - f('IMSTART')) * (1 - is_role_tok)          # reading a user/system turn
        # message start pointer: position after '<|im_start|>' role '\n' of a user/system turn
        self.msg_start = torch.where((self.win[:, -3] == 151644) & (ids == 198) & (self.asst < 0.5), (t + 1).expand_as(self.msg_start), self.msg_start)
        ans_start = ((self.win[:, -2] == 151668) & (ids == 271) & (self.asst > 0.5)).float()   # '</think>' '\n\n'
        new_turn = f('IMSTART')
        # ---- numbers
        isdig, isnum = f('ISDIG'), f('ISNUM')
        acc_prev = self.acc; self.prevdig_before = self.prevdig
        self.acc = isdig * (self.prevdig * 10 * self.acc + f('NUMV')) + (1 - isdig) * 0
        numval = isdig * self.acc + (1 - isdig) * f('NUMV')
        ordv = f('ORDV') + f('ORDSUF') * self.prevdig * acc_prev
        self.prevdig = isdig
        self.num_recent = torch.where(isnum > 0.5, numval, self.num_recent)
        # recency registers are sentence-scoped (a boundary resets them); an ordinal suffix ('2'+'nd') cancels a cardinal
        self.age_num = torch.where(isnum > 0.5, torch.zeros_like(self.age_num), torch.clamp(self.age_num + 1, max=BIGAGE))
        self.age_num = torch.where((self.sb_prev + f('ORDSUF')) > 0.5, BIGAGE, self.age_num)
        # a number right after a determiner/possessive refers to something ('the 3 paragraphs below'); a unit word after the number
        # takes it ('40 words per paragraph')
        pw = torch.where(T['NONWS'][self.win[:, -2].clamp(min=0)] > 0.5, self.win[:, -2], self.win[:, -3]).clamp(min=0)   # previous non-space token
        detprev = T['T'][pw][:, CI['POSSW']]
        self.numdet = torch.where(isnum > 0.5, detprev * (1 - self.prevdig_before), self.numdet)
        self.age_num = torch.where(X[:, CI['UNITW']] > 0.5, BIGAGE, self.age_num)
        self.age_neg = torch.where(x('NEG') > 0.5, torch.zeros_like(self.age_neg), torch.where((x('CONJ') + self.sb_prev) > 0.5, BIGAGE, torch.clamp(self.age_neg + 1, max=BIGAGE)))
        self.age_neg = torch.where(x('FORGETW') > 0.5, BIGAGE, self.age_neg)                    # "don't forget ..." / "never mind ..." 
        self.age_para = torch.where(x('PARAW') > 0.5, torch.zeros_like(self.age_para), torch.clamp(self.age_para + 1, max=BIGAGE))
        u = user
        # ---- request guards (generic): a request is not an information question, not a description of someone else, not negated
        rqc = (1 - nonreq_q) * (1 - torch.maximum(cl('OTHERS'), cl('DESCV')))                  # clause level
        rqs = (1 - nonreq_q) * (1 - torch.maximum(se('OTHERS'), se('DESCV')))                  # sentence level
        negc = cl('NEG') * (1 - cl('FORGETW')); negs = se('NEG') * (1 - se('FORGETW'))        # "don't forget to ..." is not a negation
        # ---- simple clause-scoped matches
        # a comma-word used as a modifier ('no comma key', 'inverted commas') is not the object of the negation
        prev_inv = T['T'][self.win[:, -2].clamp(min=0)][:, CI['INVW']]
        valid_c = self.cpend * (1 - f('ALPHA') * (1 - x('STOPF')))
        self.cpend = x('COMMA') * (1 - prev_inv)
        self.cvalid = torch.maximum(torch.maximum(valid_c, x('COMMAP') * (1 - prev_inv)), (1 - self.cb_prev) * self.cvalid)
        m_nc = self.cvalid * torch.maximum(negc, cl('FREEW')) * (1 - cl('FREQ2')) * (1 - cl('REMOVEV'))
        # case: a case word must be about letters ('capital letters', 'uppercase', 'all caps'), not 'capital city' / 'baseball caps'
        prevw = T['T'][self.win[:, -2].clamp(min=0)]
        self.capsall = torch.maximum(x('CAPSW') * torch.maximum(prevw[:, CI['ALLQ']], prevw[:, CI['INW']]), (1 - self.cb_prev) * self.capsall)
        capletters = torch.clamp(cl('UPPERW') + cl('CAPW') * cl('LETTER') + self.capsall, 0, 1)
        nocap = torch.clamp(cl('UPPERW') + cl('CAPW') * cl('LETTER') + cl('CAPSW'), 0, 1)
        scope = torch.clamp(cl('ALLQ') + cl('INW') + cl('WHOLE2') + cl('LETTER'), 0, 1)
        # negation scope: a case word counts as negated only if the negation comes BEFORE it in the clause
        # ('write in lowercase without emojis' is a lowercase request; 'no lowercase letters' is not)
        self.lw_neg = torch.maximum(x('LOWERW') * cl('NEG'), (1 - self.cb_prev) * self.lw_neg)
        self.lw_pos = torch.maximum(x('LOWERW') * (1 - cl('NEG')), (1 - self.cb_prev) * self.lw_pos)
        capword = torch.clamp(x('UPPERW') + x('CAPW') + x('CAPSW'), 0, 1)
        self.cw_neg = torch.maximum(capword * cl('NEG'), (1 - self.cb_prev) * self.cw_neg)
        self.cw_pos = torch.maximum(capword * (1 - cl('NEG')), (1 - self.cb_prev) * self.cw_pos)
        self.lw_seen = torch.maximum(x('LOWERW'), (1 - self.cb_prev) * self.lw_seen)
        self.cw_seen = torch.maximum(capword, (1 - self.cb_prev) * self.cw_seen)
        self.lw_post = torch.maximum(x('NEG') * self.lw_seen * (1 - x('LOWERW')), (1 - self.cb_prev) * self.lw_post)   # negation AFTER the word
        self.cw_post = torch.maximum(x('NEG') * self.cw_seen * (1 - capword), (1 - self.cb_prev) * self.cw_post)
        self.flip_c = torch.maximum(x('APPEARS') * prevw[:, CI['NEG']], (1 - self.cb_prev) * self.flip_c)
        flip = self.flip_c                                                         # 'capital letters are not allowed/used' (adjacent)
        startx = cl('STARTW') * cl('FIRSTEACH')                                     # 'start each sentence with a capital letter'
        lowA = self.lw_pos * scope * (1 - self.lw_neg) * (1 - self.lw_post) * (1 - cl('TIMES')) * (1 - startx) * (1 - cl('PARTW'))
        lowB = nocap * scope * torch.maximum(self.cw_neg, self.cw_post * flip) * (1 - cl('FREQ2')) * (1 - cl('TIMES'))
        capA = capletters * self.cw_pos * scope * (1 - self.cw_neg) * (1 - self.cw_post) * (1 - cl('TIMES')) * (1 - startx) * (1 - cl('FREQ2')) * (1 - cl('PARTW'))
        capB = scope * cl('LETTER') * torch.maximum(self.lw_neg, self.lw_post * flip) * (1 - cl('FREQ2')) * (1 - cl('TIMES'))
        # quotes: quotation MARKS / double quotes (or a wrap verb) around the WHOLE response, not negated
        m_q = cl('QUOTEW') * (1 - cl('SINGLE')) * torch.maximum(cl('QMARKW'), cl('WRAPS')) * torch.maximum(cl('WRAPV'), cl('INW')) \
            * cl('WHOLE2') * (1 - cl('ENDW') * (1 - se('STARTW'))) * (1 - negc)
        # clause-level matches are read at the clause end (complete clause)
        # a clause-level request must also carry a request marker (imperative/modal/you/please/whole-response word or an opening
        # negation such as 'No commas.'): descriptions such as 'my grandfather never used commas' do not
        negopen = x('NEG') * self.wlead_now * self.clause_sent
        self.negopen = torch.maximum(negopen, (1 - self.cb_prev) * self.negopen)
        reqm = torch.clamp(cl('REQANY') + self.negopen + self.imp_any, 0, 1)
        ce = cb * rqc * reqm
        # sentence-level request marker (instruction verb at a clause opening, you/please/modal, or an opening negation)
        self.s_impany = torch.maximum(torch.maximum(self.imp_any, self.negopen), (1 - self.sb_prev) * self.s_impany)
        # 'X say(s) ...' as a report (not 'do not say', not an imperative 'Say ...', not 'to say')
        sayrep = x('SAYW') * (1 - self.wlead_now) * (self.age_neg > 2).float() * (1 - prevw[:, CI['STOPF']])
        self.s_sayrep = torch.maximum(sayrep, (1 - self.sb_prev) * self.s_sayrep)
        topic = torch.clamp(se('EXPLAINV') * se('WHQ') + se('REPORTV') + self.s_sayrep + se('CODEW'), 0, 1)
        topic = torch.clamp(topic + self.qsent, 0, 1)                                   # 'Should I ...?', 'Is it ...?' (no 'you')
        reqs = torch.clamp(se('REQANY') + self.s_impany, 0, 1) * (1 - topic)
        negx = x('NEG') * (1 - cl('FORGETW'))
        ng = lambda nm, cue: self._cue_neg(nm, cue, negx, negc)
        n_two = ng('two', torch.clamp(x('ASTER') + f('STAR6'), 0, 1)); n_title = ng('title', torch.clamp(x('TITLEW') + f('ANGTOK') + x('ANGW'), 0, 1))
        n_bul = ng('bul', torch.clamp(x('BULLETS') + x('BULLET1'), 0, 1)); n_par = ng('par', torch.clamp(x('PARAW') + x('SECTW'), 0, 1))
        n_high = ng('high', torch.clamp(x('HIGHW') + x('ITALW'), 0, 1)); n_nth = ng('nth', x('STARTV'))
        t_high = self._cue_neg('hightimes', torch.clamp(x('HIGHW') + x('ITALW'), 0, 1), x('TIMES'), cl('TIMES'))
        # two responses: the divider ('******' or six asterisks) with a separate/two-responses word in the same sentence (read at sentence end)
        run = f('LEADST') + torch.where(f('ALLST') > 0.5, self.strun, self.strail * (f('LEADST') > 0).float())
        self.strun = torch.where(f('ALLST') > 0.5, run, f('TRAILST')); self.strail = f('TRAILST')
        lit3 = (run >= 3).float() * (run < 6).float() * (1 - inq_prev) * (1 - insq_prev) * (1 - inbt_prev)
        lit6 = (run >= 6).float() * (1 - inq_prev) * (1 - insq_prev) * (1 - inbt_prev) * (1 - self.inbt)
        six = (self.num_recent == 6).float() * (self.age_num <= 3).float()
        three = (self.num_recent == 3).float() * (self.age_num <= 3).float()
        self.se_six = torch.maximum(torch.maximum(torch.maximum(f('STAR6') * (1 - inq_prev) * (1 - insq_prev) * (1 - inbt_prev) * (1 - self.inbt), lit6), x('ASTER') * six), (1 - self.sb_prev) * self.se_six)
        m_two = self.se_six * torch.maximum(se('SEPV'), se('TWOW')) * sb * rqs * reqs * (1 - n_two)
        self.se_s3 = torch.maximum(torch.maximum(torch.maximum(torch.maximum(f('STAR3') * (1 - inq_prev), lit3), x('ASTER') * three), cl('DIVW') * cl('MARKDOWN')),
                                   (1 - self.sb_prev) * self.se_s3)
        m_s3 = self.se_s3 * sb * rqs * reqs                                                    # divider mode: requested, read at sentence end
        m_title = torch.maximum(self.se_ang * se('TITLEW'), se('ANGW') * se('BRACKW') * torch.clamp(se('TITLEW') + se('YOUW') + se('WHOLE2'), 0, 1)) \
            * sb * rqs * reqs * (1 - n_title)
        # counts: a number <= 3 tokens before the item word, not a reference ('the 3 paragraphs'), no unit word in between, N <= 20;
        # kept as a sentence candidate and committed at the sentence end if the sentence is a request and not negated
        near = (self.age_num <= 3).float() * (1 - f('ISNUM')) * (1 - self.numdet) * (self.num_recent <= 20).float() * (self.num_recent >= 1).float()
        prev_b1 = prevw[:, CI['BULLET1']]
        b_hit = torch.maximum(x('BULLETS') * near, x('POINTW') * prev_b1 * (self.age_num <= 4).float() * (1 - self.numdet) * (self.num_recent <= 20).float())
        p_hit = x('PARAW') * near
        s_hit = x('SECTW') * near * (1 - torch.clamp(se('HIGHW') + se('ITALW') + se('BULLETW'), 0, 1))
        self.sc_bul = torch.where(b_hit > 0.5, self.num_recent, self.sc_bul)
        self.sc_par = torch.where(p_hit > 0.5, self.num_recent, self.sc_par)
        self.sc_sec = torch.where(s_hit > 0.5, self.num_recent, self.sc_sec)
        okc = sb * u * rqs * reqs
        okb = okc * (1 - n_bul); okp = okc * (1 - n_par)
        self.N_bul = torch.where((okb * (self.sc_bul > 0).float()) > 0.5, self.sc_bul, self.N_bul)
        self.N_par = torch.where((okp * (self.sc_par > 0).float()) > 0.5, self.sc_par, self.N_par)
        self.N_sec = torch.where((okp * (self.sc_sec > 0).float()) > 0.5, self.sc_sec, self.N_sec)
        m_bul = okb * (self.sc_bul > 0).float()
        m_para = okp * torch.maximum((self.sc_par > 0).float(), (self.sc_sec > 0).float())
        # clause-level matches are kept as sentence candidates and committed at the sentence end, unless the sentence turns out to be
        # a non-request question or a topic/reported/code sentence ('Should I include a P.S. ...?')
        for k, m in [('nc', m_nc * ce), ('lower', torch.maximum(lowA, lowB) * ce), ('caps', torch.maximum(capA, capB) * ce), ('quote', m_q * ce),
                     ('ps', cl('PSW') * cl('PSV') * (1 - negc) * (1 - cl('EXPLAINV')) * ce)]:
            self.cand[k] = torch.maximum(self.cand[k], m * u)
            self.L[k] = torch.maximum(self.L[k], self.cand[k] * torch.maximum(sb * (1 - nonreq_q), f('COLON')) * (1 - topic))
            self.cand[k] = torch.where((sb + f('COLON')) > 0.5, 0.0, self.cand[k])
        for k, m in [('pps', cl('PPS')), ('two', m_two), ('star3', m_s3), ('title', m_title), ('bul', m_bul), ('para', m_para)]:
            self.L[k] = torch.maximum(self.L[k], m * u * (1 - topic))
        # ---- end phrase capture (pointer registers end_s / end_len / end_x)
        # imperative position: the end-word opens the clause (optionally after please/also/then/and ...)
        self.imp_end = torch.maximum(x('ENDW') * self.wlead, (1 - self.cb_prev) * self.imp_end)
        endok = torch.maximum(cl('ENDV'), cl('LASTW') * torch.clamp(cl('WITHW') + cl('BEW'), 0, 1))     # 'your last answer said "X"' is not a request
        armed = endok * torch.clamp(cl('REQMOD') + self.imp_any + self.imp_end * cl('WITHW'), 0, 1) * (1 - negc) * rqc \
            * (1 - cl('EACH')) * (1 - se('PARAW')) * u * (1 - self.L['end'])
        opq = torch.maximum(f('OPENDQ'), f('OPENSQ')); qtype = f('OPENDQ') + 2 * (1 - f('OPENDQ')) * f('OPENSQ')
        idle = (self.ec == 0).float()
        go_q = idle * armed * opq + (self.ec == 1).float() * self.e_cf * opq * u           # quote opens (also right after a colon)
        colon_ok = torch.clamp(prevw[:, CI['WITHW']] + prevw[:, CI['PHRASEW']] + prevw[:, CI['ENDV']] + prevw[:, CI['BEW']], 0, 1)
        go_c = idle * armed * f('COLON') * colon_ok * (1 - go_q)
        closer = torch.where(self.eq_t == 1, f('CLOSEDQ'), f('CLOSESQ'))
        pre = T['PRE'][ids]
        in_q = (self.ec == 2).float() * u; in_c = (self.ec == 1).float() * (1 - self.e_cf) * u
        fin_q = in_q * closer
        fin_c_nl = in_c * torch.maximum(f('NL'), f('IMEND'))
        fin_c_p = in_c * f('SENTB') * (1 - f('NL'))
        L_q = t - self.end_s                                                               # tokens strictly inside
        self.end_x = torch.where(fin_q > 0.5, pre, torch.where(fin_c_nl > 0.5, torch.where(f('NL') > 0.5, pre, -1), self.end_x))
        new_len = torch.where(fin_c_p > 0.5, L_q + 1, L_q)
        fin = torch.clamp(fin_q + fin_c_nl + fin_c_p, 0, 1)
        okl = ((new_len + (self.end_x >= 0).long()) >= 1) & (new_len <= 40)
        self.end_len = torch.where(fin > 0.5, new_len, self.end_len)
        self.L['end'] = torch.maximum(self.L['end'], fin * okl.float())
        self.end_sent = torch.maximum(self.end_sent, fin * okl.float())
        # a phrase captured inside a sentence that turns out to be a non-request question is dropped ('Why do you end with "X"?')
        undo = self.end_sent * nonreq_q
        self.L['end'] = torch.where(undo > 0.5, 0.0, self.L['end']); self.ec = torch.where(undo > 0.5, 0.0, self.ec)
        self.end_sent = torch.where(sb > 0.5, 0.0, self.end_sent)
        self.ec = torch.where(fin > 0.5, torch.where(okl, 3.0, 0.0), self.ec)
        self.ec = torch.where((in_q * (L_q > 40).float()) > 0.5, 0.0, self.ec)              # unclosed quote: give up
        self.ec = torch.where(go_q > 0.5, 2.0, torch.where(go_c > 0.5, 1.0, self.ec))
        self.eq_t = torch.where(go_q > 0.5, qtype, self.eq_t)
        self.end_s = torch.where((go_q + go_c) > 0.5, (t + 1).expand_as(self.end_s), self.end_s)
        self.end_x = torch.where((go_q + go_c) > 0.5, -1, self.end_x)
        self.e_cf = go_c
        # ---- nth word capture
        armw = (self.wc == 0).float() * se('STARTV') * x('WITHW') * u
        skip = torch.clamp(x('SKIPNTH') + opq + f('COLON'), 0, 1)
        cap0 = (self.wc == 1).float() * f('ALPHA') * (1 - skip) * u
        cont = (self.wc == 2).float() * f('CONT')
        stop2 = (self.wc == 2).float() * (1 - f('CONT'))
        self.nth_s = torch.where(cap0 > 0.5, t.expand_as(self.nth_s), self.nth_s)
        self.nth_len = torch.where(cap0 > 0.5, 1, torch.where(cont > 0.5, self.nth_len + 1, self.nth_len))
        self.wc = torch.where(armw > 0.5, 1.0, torch.where(cap0 > 0.5, 2.0, torch.where(stop2 > 0.5, 3.0, self.wc)))
        self.wc = torch.where((self.wc == 1) & (skip < 0.5) & (f('ALPHA') < 0.5) & (f('NONWS') > 0.5), 0.0, self.wc)
        self.wc = torch.where((self.wc == 2) & (self.nth_s == t) & (x('ARTW') > 0.5), 0.0, self.wc)
        kval = torch.where(ordv > 0, ordv, torch.where((isnum > 0.5) & (self.age_para <= 2), numval, torch.zeros_like(ordv)))
        self.s_k = torch.where(kval * u * (1 - cap0) * (1 - cont) > 0, kval, self.s_k)     # the captured word itself is not an ordinal
        self.s_klast = torch.maximum(self.s_klast, x('LASTW') * u)
        # ---- highlight count (first number after the highlight word, else the last number before it)
        hw = torch.maximum(x('HIGHW'), x('ITALW'))
        self.s_hn_first = torch.where((isnum * self.s_hseen * (self.s_hn_first == 0).float()) > 0.5, numval, self.s_hn_first)
        self.s_hn_last = torch.where((isnum * (1 - self.s_hseen)) > 0.5, numval, self.s_hn_last)
        self.s_hseen = torch.maximum(self.s_hseen, hw)
        # ---- forbidden words: list regions (quoted or after word/keyword), slots of canonical pieces
        canon = T['CANON'][ids]
        opener = opq
        cls_tok = torch.where(self.rqt == 1, f('CLOSEDQ'), f('CLOSESQ'))
        sepish = torch.clamp(x('SEPW') + (ids == 11).float() + f('COLON') + opener, 0, 1)            # 11 = ','
        open_w = (self.rg == 0).float() * x('WORDTERM') * u
        det_before = (T['T'][self.win[:, -2].clamp(min=0)][:, CI['DETW']] > 0.5).float()      # 'the/these/following' + word(s)
        open_q = (self.rg == 0).float() * opener * (self.age_neg <= 6).float() * u * (1 - open_w)
        inr = (self.rg == 1).float() * u
        self.rq_age = torch.where(self.rq > 0.5, self.rq_age + 1, torch.zeros_like(self.rq_age))
        q_abort = inr * self.rq * (self.rq_age > 12).float()                                         # an unclosed quote: give up the region
        self.s_abort = torch.maximum(self.s_abort, q_abort)
        q_close = inr * self.rq * cls_tok
        q_open = inr * (1 - self.rq) * opener
        q_item = inr * self.rq * (1 - cls_tok) * f('LETTERS')
        u_item = inr * (1 - self.rq) * (1 - opener) * f('ALPHA') * (1 - x('STOPF')) * (1 - x('SEPW')) * (1 - self.rqonly) * (1 - self.rsawq)
        u_sep = inr * (1 - self.rq) * (1 - opener) * sepish
        firstbad = self.rfirst * inr * (1 - self.rq) * (1 - opener) * (1 - f('COLON')) * (1 - u_item)
        item = torch.clamp(q_item + u_item, 0, 1) * (1 - firstbad)
        neww = item * torch.clamp(f('WSTART') + self.lastsep, 0, 1)
        contw = item * (1 - torch.clamp(f('WSTART') + self.lastsep, 0, 1))
        # word identity = polynomial rolling hash of its lowercase letters (fixed per-token tables HT/POWT; integer registers)
        si = torch.where(neww > 0.5, self.nslot, self.nslot - 1).clamp(0, S_W - 1)
        wr = (((neww > 0.5) & (self.nslot < S_W)) | ((contw > 0.5) & (self.nslot > 0))) & (f('HASAN') > 0.5)
        b_ix = torch.arange(B, device=DEV)
        cur = self.slot_h[b_ix, si]
        nh = torch.where(neww > 0.5, T['HT'][ids], (cur * T['POWT'][ids] + T['HT'][ids]) % HP)
        self.slot_h[b_ix, si] = torch.where(wr, nh, cur)
        self.nslot = torch.where(wr & (neww > 0.5), self.nslot + 1, self.nslot)
        endreg = inr * (1 - self.rq) * (1 - opener) * (1 - u_item) * (1 - u_sep) * (1 - q_close)
        endreg = torch.maximum(endreg, firstbad)
        rq_new = torch.where(open_q + q_open > 0.5, 1.0, torch.where((q_close + q_abort) > 0.5, 0.0, self.rq))
        endreg = torch.maximum(endreg, q_abort)
        endreg = torch.maximum(endreg, inr * (1 - rq_new) * sb)
        # pattern B window after a list closed (NEG + 'appear/be/used...' or a negative adjective, within 6 tokens)
        self.aw = torch.where(endreg > 0.5, torch.zeros_like(self.aw), torch.clamp(self.aw + 1, max=BIGAGE))
        inaw = (self.aw <= 6).float()
        self.aw_neg = torch.where(endreg > 0.5, x('NEG'), torch.maximum(self.aw_neg, inaw * x('NEG')))
        self.aw_app = torch.where(endreg > 0.5, x('APPEARS'), torch.maximum(self.aw_app, inaw * x('APPEARS')))
        self.s_commit = torch.maximum(self.s_commit, self.aw_neg * self.aw_app + inaw * x('NEGADJ') * (self.aw > 0).float())
        self.s_commit = torch.maximum(self.s_commit, endreg * self.rnegA)
        st_open = torch.clamp(open_w + open_q, 0, 1)
        self.rnegA = torch.where(st_open > 0.5, (self.age_neg <= 8).float(), self.rnegA)
        self.rq = rq_new
        self.rsawq = torch.where(st_open > 0.5, open_q, torch.maximum(self.rsawq, q_open))
        self.rqonly = torch.where(open_w > 0.5, 1 - det_before, torch.where(st_open > 0.5, 0.0, self.rqonly))
        self.rqt = torch.where(open_q + q_open > 0.5, f('OPENDQ') + 2 * (1 - f('OPENDQ')), self.rqt)
        self.lastsep = torch.where(st_open + q_open + q_close + u_sep > 0.5, 1.0, torch.where(item > 0.5, 0.0, self.lastsep))
        self.rfirst = torch.where(open_w > 0.5, 1.0, torch.where(inr > 0.5, 0.0, self.rfirst))
        self.rg = torch.where(st_open > 0.5, 1.0, torch.where(endreg > 0.5, 0.0, self.rg))
        # ---- repeat request (sentence-level co-occurrence; span by pointer registers)
        self.s_negrep = torch.maximum(self.s_negrep, x('REPEATV') * (self.age_neg <= 4).float())
        self.age_you = torch.where(x('YOURW') > 0.5, torch.zeros_like(self.age_you), torch.clamp(self.age_you + 1, max=BIGAGE))
        self.s_negrep = torch.maximum(self.s_negrep, x('REQTERM') * (self.age_you <= 2).float())            # 'your previous message'
        # ---- sentence finalisation (at a sentence boundary or end of message): repeat, nth, highlight, forbidden commit
        sbu = sb * u
        qok = 1 - nonreq_q                                        # a question that is not a request does not count
        rep_hit = sbu * rqs * reqs * (1 - self.in_sys) * se('REPEATV') * se('REQTERM') * se('VERBATIM') * (1 - se('REFMT')) \
            * (1 - self.s_negrep) * (1 - self.rep_seen)
        self.rep_seen = torch.maximum(self.rep_seen, rep_hit)
        below = se('BELOWD') * (1 - se('ABOVED'))
        self.rep_s = torch.where(rep_hit * (1 - below) > 0.5, self.msg_start, self.rep_s)
        self.rep_len = torch.where(rep_hit * (1 - below) > 0.5, (self.prev_end - self.msg_start + 1).clamp(min=0), self.rep_len)
        self.rep_below = torch.where(rep_hit > 0.5, below, self.rep_below)
        self.rep_wait = torch.where(rep_hit * below > 0.5, 1.0, self.rep_wait)
        nl_now = rep_hit * below * f('NL')
        self.rep_s = torch.where(nl_now > 0.5, (t + 1).expand_as(self.rep_s), self.rep_s)
        self.rep_wait = torch.where(nl_now > 0.5, 2.0, self.rep_wait)
        self.L['repeat'] = torch.maximum(self.L['repeat'], rep_hit * (1 - below) * (self.prev_end >= self.msg_start).float())
        nth_hit = sbu * rqs * reqs * (1 - n_nth) * se('STARTV') * se('PARAW') * (self.wc >= 2).float() * ((self.s_k > 0).float() + self.s_klast).clamp(0, 1) * (1 - self.L['nth'])
        self.L['nth'] = torch.maximum(self.L['nth'], nth_hit)
        self.nth_k = torch.where(nth_hit > 0.5, self.s_k, self.nth_k); self.nth_last = torch.where(nth_hit > 0.5, self.s_klast * (self.s_k == 0).float(), self.nth_last)
        h_hit = sbu * rqs * reqs * (1 - n_high) * torch.maximum(se('ITALW'), se('HIGHW') * torch.clamp(se('MARKDOWN') + se('ASTERISK') + self._starc_se(ids)
                                                                                                    + se('HIGHOBJ'), 0, 1)) * (1 - t_high)
        hn = torch.where(self.s_hn_first > 0, self.s_hn_first, torch.where(self.s_hn_last > 0, self.s_hn_last, torch.ones_like(self.s_hn_last)))
        hn = torch.where(hn > 20, torch.ones_like(hn), hn)                                     # not a highlight count
        self.N_high = torch.where(h_hit * (1 - self.L['high']) > 0.5, hn, self.N_high)
        self.L['high'] = torch.maximum(self.L['high'], h_hit)
        commit = sbu * rqs * reqs * self.s_commit * (1 - torch.clamp(se('TIMES') + se('FREQ3'), 0, 1)) * (1 - self.rg * self.rq) * (1 - self.s_abort)
        keep = commit > 0.5
        sl_ix = torch.arange(S_W, device=DEV)[None]
        is_new = (sl_ix >= self.nbase[:, None]) & (sl_ix < self.nslot[:, None])
        self.slot_ok = torch.where(keep[:, None] & is_new, 1.0, self.slot_ok)
        drop = (sbu > 0.5) & ~keep & (self.rg * self.rq < 0.5)
        clr = drop[:, None] & is_new
        self.slot_h = torch.where(clr, 0, self.slot_h)
        self.nslot = torch.where(drop, self.nbase, self.nslot)
        self.nbase = torch.where(sbu > 0.5, self.nslot, self.nbase)
        self.L['forbid'] = torch.maximum(self.L['forbid'], (self.slot_ok.sum(1) > 0).float())
        # repeat: pointer bookkeeping (last non-space token; end of the previous sentence; 'below' span start/end)
        self.lnw_pos = torch.where((nonws * u) > 0.5, t.expand_as(self.lnw_pos), self.lnw_pos)
        self.prev_end = torch.where(sbu > 0.5, self.lnw_pos, self.prev_end)
        rw = (self.rep_wait == 1).float() * u * torch.maximum(f('NL'), f('COLON')) * (1 - rep_hit)
        self.rep_s = torch.where(rw > 0.5, (t + 1).expand_as(self.rep_s), self.rep_s)
        self.rep_wait = torch.where(rw > 0.5, 2.0, self.rep_wait)
        endmsg = f('IMEND') * (self.rep_wait == 2).float() * u
        self.rep_len = torch.where(endmsg > 0.5, (self.lnw_pos - self.rep_s + 1).clamp(min=0), self.rep_len)
        self.L['repeat'] = torch.maximum(self.L['repeat'], endmsg * (self.lnw_pos >= self.rep_s).float())
        self.rep_wait = torch.where(endmsg > 0.5, 0.0, self.rep_wait)
        # sentence-scoped registers reset after a boundary
        rs = sb > 0.5
        for nm in ['s_k', 's_klast', 's_hn_first', 's_hn_last', 's_hseen', 's_negrep', 's_commit', 's_abort', 'sc_bul', 'sc_par', 'sc_sec']:
            setattr(self, nm, torch.where(rs, 0.0, getattr(self, nm)))
        self.wc = torch.where(rs & (self.wc < 3), 0.0, torch.where(rs & (self.L['nth'] < 0.5), 0.0, self.wc))
        # turn scope (fail-safe): an instruction applies to the reply to the user turn that gives it; instructions in a SYSTEM
        # message persist for the whole conversation. At each new user turn the latches of earlier user turns are dropped.
        is_sys_role = ((self.prevtok == 151644) & (ids == 8948)).float()
        self.in_sys = torch.where(is_role_tok > 0.5, is_sys_role, self.in_sys)
        for k in self.L: self.Ls[k] = torch.maximum(self.Ls[k], (self.L[k] > L_before[k]).float() * self.in_sys)
        self.sys_nslot = torch.where(self.in_sys > 0.5, self.nslot, self.sys_nslot)
        user_turn = ((self.prevtok == 151644) & (ids == 872)).float()                          # '<|im_start|>' 'user'
        ut = user_turn > 0.5
        for k in self.L: self.L[k] = torch.where(ut, self.Ls[k], self.L[k])
        self.slot_ok = torch.where(ut[:, None] & (torch.arange(S_W, device=DEV)[None] >= self.sys_nslot[:, None]), 0.0, self.slot_ok)
        self.rep_seen = torch.where(ut, self.L['repeat'], self.rep_seen)
        self.ec = torch.where(ut & (self.L['end'] < 0.5), 0.0, self.ec)
        self.wc = torch.where(ut & (self.L['nth'] < 0.5), 0.0, self.wc)
        # a cancel cue ('back to normal', 'plain text', 'not anymore', 'no ... this time') also drops the system-message latches
        cancel = x('CANCELW') * u
        for k in self.L: self.L[k] = torch.where(cancel > 0.5, self.L[k] * (1 - self.Ls[k]), self.L[k])
        self.slot_ok = torch.where((cancel[:, None] > 0.5) & (torch.arange(S_W, device=DEV)[None] < self.sys_nslot[:, None]), 0.0, self.slot_ok)
        self.cb_prev, self.sb_prev = cb, sb
        self._answer_step(ids, ans_start, new_turn)
        self.prevtok = ids
        self.t = self.t + 1

    def _cue_neg(self, name, cue, negx, negc):
        """sentence register: a negation shares a CLAUSE with this rule's cue word (either order); negx = negation at this token"""
        R = self.cueR
        if name not in R: R[name] = [torch.zeros(self.B, device=DEV), torch.zeros(self.B, device=DEV)]
        clc, sneg = R[name]
        clc = torch.maximum(cue, (1 - self.cb_prev) * clc)                      # cue seen in the current clause
        sneg = torch.maximum(torch.maximum(cue * negc, clc * negx), (1 - self.sb_prev) * sneg)
        R[name] = [clc, sneg]
        return sneg

    def _starc_se(self, ids):
        # sentence has a '*' token (example of the markdown); kept as its own scope register
        self.se_star = torch.maximum(self.T['STARC'][ids], (1 - self.sb_prev) * self.se_star)
        return self.se_star

    # ------------------------------------------------------------------ answer-side trackers and program triggers
    def _answer_step(self, ids, ans_start, new_turn):
        T = self.T; f = lambda n: T[n][ids]
        reset = new_turn > 0.5
        # emitter: advance the program counter on the emitted token; completion flags
        act = self.prog > 0
        _, slen = self.case_seq(self.prog_token())
        sub = act & (self.cj + 1 < slen)                                      # still inside a case-expanded token
        self.cj = torch.where(sub, self.cj + 1, torch.zeros_like(self.cj))
        self.pj = torch.where(act & ~sub, self.pj + 1, self.pj)
        plen = self.pre_len + self.sp_len + self.suf_len
        fin = act & (self.pj >= plen)
        for kind, nm in [(1, 'qopen'), (2, 'rep'), (3, 'title'), (4, 'nth')]:
            self.done[nm] = torch.where(fin & (self.prog == kind), 1.0, self.done[nm])
        self.ps_seen = torch.where(fin & (self.prog == 10), 1.0, self.ps_seen)
        self.end_done = torch.where(fin & (self.prog == 11), 1.0, self.end_done)
        hf_set = fin & (self.prog == 9)
        self.prog_prev = self.prog.clone()
        # a paragraph break that the circuit forced (programs 6/9: end-of-turn -> break / appended highlight) arms the anti-repeat block
        self.forced_brk = torch.where(fin & ((self.prog == 6) | (self.prog == 5)), 1.0, self.forced_brk)
        self.prog = torch.where(fin, 0, self.prog)
        a = self.ans * (1 - ans_start)                                          # tokens of the answer (not the start marker)
        # answer trackers
        nonws = f('NONWS')
        idx0 = (self.n == 0).float()
        self.lq = torch.where(nonws * a > 0.5, f('QE'), self.lq)
        self.lx = torch.where(nonws * a > 0.5, f('QX') * idx0, self.lx)
        self.n = self.n + a
        nn_ = f('NN') * a
        brk = nn_ * torch.clamp(self.seg + f('PRECNN'), 0, 1)
        self.kp = self.kp + brk
        self.last_brk = brk * (1 - f('STAR3'))
        s3 = f('STAR3') * a
        self.d3 = self.d3 + s3
        self.seg = torch.where(nn_ + s3 > 0.5, f('POSTNN') * (1 - s3), torch.maximum(self.seg, nonws * a))
        self.forced_brk = torch.where((f('LETTERS') * a) > 0.5, 0.0, self.forced_brk)
        self.k6 = self.k6 + f('STAR6X') * a
        self.wa = self.wa + f('LETTERS') * f('WSTART') * a                                     # words in the answer
        self.w6 = torch.where(f('STAR6X') * a > 0.5, 0.0, self.w6 + f('NONWS') * f('WSTART') * a * (self.k6 > 0).float())
        self.w6l = torch.where(f('STAR6X') * a > 0.5, 0.0, self.w6l + f('LETTERS') * a * (self.k6 > 0).float())
        self.n6 = torch.where(f('STAR6X') * a > 0.5, 0.0, self.n6 + a * (self.k6 > 0).float())    # tokens since the divider
        self.bcount = self.bcount + a * self.at_ls * f('BULST')
        # first word piece of the current paragraph / of the answer (for the anti-repeat block)
        first_letters = a * f('LETTERS') * (self.cur_pf < 0).float()
        self.cur_pf = torch.where(first_letters > 0.5, T['HT'][ids], self.cur_pf)
        self.ans_pf = torch.where((a * f('LETTERS') * (self.ans_pf < 0).float()) > 0.5, T['HT'][ids], self.ans_pf)
        pbrk = torch.clamp(nn_ + s3, 0, 1) * (1 - f('POSTNN'))
        self.prev_pf = torch.where((pbrk * (self.cur_pf >= 0).float()) > 0.5, self.cur_pf, self.prev_pf)
        self.cur_pf = torch.where(pbrk > 0.5, -1, self.cur_pf)
        self.ln_bul = torch.where((a * self.at_ls) > 0.5, f('BULST'), self.ln_bul)               # current line is a bullet line
        self.ln_cont = torch.where((a * self.at_ls) > 0.5, f('LETTERS'), torch.maximum(self.ln_cont, a * f('LETTERS')))
        self.at_ls = torch.where(a > 0.5, f('ENDNL'), self.at_ls)
        self.at_ls = torch.where(ans_start > 0.5, 1.0, self.at_ls)
        # highlight tracker: star tokens toggle open/close inside a line; a close with content counts
        star = f('STARC') * a
        close_h = star * self.hl_open * (self.hl_cont > 0).float()
        self.hcount = self.hcount + close_h
        self.hforced = torch.where(star * self.hl_open > 0.5, 0.0, torch.where(hf_set, 1.0, self.hforced))
        self.hl_open = torch.where(star > 0.5, 1 - self.hl_open, self.hl_open)
        self.hl_cont = torch.where(star > 0.5, 0.0, self.hl_cont + a * f('LETTERS'))
        nlr = f('NL') * a
        self.hl_open = torch.where(nlr > 0.5, 0.0, self.hl_open); self.hforced = torch.where(nlr > 0.5, 0.0, self.hforced)
        t_prog = (fin & (self.prog_prev == 3)).float()                                     # the circuit's own '<<' was just emitted
        self.t_open = torch.where(t_prog > 0.5, 1.0, torch.where((f('CLOSETITLE') + f('NL')) * a > 0.5, 0.0, self.t_open))
        # host-written P.S. (tokens 'P' '.S' / 'P' '.P' '.S', any case) counts as the postscript
        w = self.win
        P_ = (T['LOW'][w[:, -2].clamp(min=0)] == self.tk['p']) | (T['NOSP'][T['LOW'][w[:, -2].clamp(min=0)]] == self.tk['p'])
        dS = T['LOW'][ids] == self.tk['.s']
        dP2 = T['LOW'][w[:, -2].clamp(min=0)] == self.tk['.p']
        self.ps_seen = torch.maximum(self.ps_seen, a * ((P_ & dS) | (dP2 & dS)).float())
        # end phrase: suffix match counter against the captured phrase (copy head reads phrase piece m)
        pos = (self.end_s + self.end_m).clamp(0, MAXMEM - 1)
        want = torch.where(self.end_m < self.end_len, self.mem.gather(1, pos[:, None])[:, 0], self.end_x.clamp(min=0))
        w0 = self.mem.gather(1, self.end_s.clamp(0, MAXMEM - 1)[:, None])[:, 0]
        cw = T['CANON'][want]; cg = T['CANON'][ids]; cw0 = T['CANON'][w0]
        match = ((cg == cw) & (cg >= 0)) | (T['NOSP'][ids] == T['NOSP'][want])
        restart = ((cg == cw0) & (cg >= 0)) | (T['NOSP'][ids] == T['NOSP'][w0])
        self.end_m = torch.where(a > 0.5, torch.where(match, self.end_m + 1, torch.where(restart, 1, 0)), self.end_m)
        nlen = self.end_len + (self.end_x >= 0).long()
        matched = (self.end_m >= nlen) & (nlen > 0)
        self.end_done = torch.maximum(self.end_done, (matched & (a > 0.5)).float())
        self.end_done = torch.where((a > 0.5) & ~matched & (self.prog != 11) & (nonws > 0.5) & (f('PRECNN') + f('LETTERS') > 0), torch.where(match, self.end_done, 0.0), self.end_done)
        self.end_m = torch.where(matched, 0, self.end_m)
        # rolling hash of the word being written (reset by a token that starts with / contains a non-letter)
        hn = torch.where(f('LEAD') > 0.5, T['HT'][ids], (self.hcur * T['POWT'][ids] + T['HT'][ids]) % HP)
        hn = torch.where((f('HASAN') < 0.5) | (f('TRAIL') > 0.5), torch.zeros_like(hn), hn)
        self.hcur = torch.where(a > 0.5, hn, torch.where(ans_start > 0.5, torch.zeros_like(hn), self.hcur))
        # new turn / answer start
        self.ans = torch.where(ans_start > 0.5, 1.0, torch.where(reset, 0.0, self.ans))
        for nm in ['n', 'lq', 'lx', 'seg', 'kp', 'd3', 'k6', 'w6', 'w6l', 'n6', 'wa', 'forced_brk', 'bcount', 'hl_open', 'hl_cont', 'hcount', 'hforced', 't_open', 'ps_seen', 'end_done', 'body', 'ln_bul', 'ln_cont']:
            setattr(self, nm, torch.where(reset | (ans_start > 0.5), 0.0, getattr(self, nm)))
        for k in self.done: self.done[k] = torch.where(reset | (ans_start > 0.5), 0.0, self.done[k])
        self.prog = torch.where(reset | (ans_start > 0.5), 0, self.prog); self.cj = torch.where(reset | (ans_start > 0.5), 0, self.cj)
        for nm in ['cur_pf', 'prev_pf', 'ans_pf']: setattr(self, nm, torch.where(reset | (ans_start > 0.5), -1, getattr(self, nm)))
        self._triggers(ids)

    def gates(self):
        L, a = self.L, self.ans
        g = {k: L[k] * a for k in L}
        g['caps'] = g['caps'] * (1 - g['lower'])
        npar = torch.where((L['star3'] > 0.5) & (self.N_par == 0), self.N_sec, self.N_par)
        g['para'] = g['para'] * (npar >= 2).float() * (1 - g['two'])
        g['npar'] = npar
        g['p3'] = g['para'] * L['star3']; g['pn'] = g['para'] * (1 - L['star3'])
        g['nth'] = g['nth'] * g['pn']
        g['bul'] = g['bul'] * (self.N_bul >= 1).float()
        g['high'] = g['high'] * (self.N_high >= 1).float()
        return g

    def _triggers(self, ids):
        T = self.T; f = lambda n: T[n][ids]; g = self.gates(); a = self.ans
        idle = (self.prog == 0).float()
        # start chain: quote open -> repeat copy -> title open -> first word of paragraph 1
        nb = a * (1 - self.body) * idle
        q_go = nb * g['quote'] * (1 - self.done['qopen'])
        r_go = nb * g['repeat'] * (1 - self.done['rep']) * (1 - q_go)
        t_go = nb * g['title'] * (1 - self.done['title']) * (1 - q_go) * (1 - r_go)
        kk = torch.where(self.nth_last > 0.5, g['npar'], self.nth_k)
        n_go = nb * g['nth'] * (kk == 1).float() * (1 - self.done['nth']) * (1 - q_go) * (1 - r_go) * (1 - t_go)
        self.start_prog(q_go, 1, pre=[self.tk['"']])
        self.start_prog(r_go, 2, span=(self.rep_s, self.rep_len), suf=[self.tk['\n\n']])
        self.start_prog(t_go, 3, pre=[self.tk['<<']])
        self.start_prog(n_go, 4, span=(self.nth_s, self.nth_len), fmap=2)
        self.body = torch.maximum(self.body, nb * (1 - q_go) * (1 - r_go) * (1 - t_go) * (1 - n_go))
        # in the body: after a paragraph break -> '***' divider (*** paragraphs) or the required first word (paragraph k)
        idle = (self.prog == 0).float(); bd = self.body * idle * a
        brk = f('NN') * (1 - f('POSTNN'))
        s_go = bd * g['p3'] * brk * (self.d3 < g['npar'] - 1).float() * self.last_brk
        self.start_prog(s_go, 5, pre=[self.tk['***\n\n']])
        n2 = bd * g['nth'] * brk * (self.kp == kk - 1).float() * (kk >= 2).float() * (1 - self.done['nth']) * (1 - s_go)
        self.start_prog(n2, 4, span=(self.nth_s, self.nth_len), fmap=2)

    # ------------------------------------------------------------------ write
    def force(self, t):
        d = self.E[t].float() / self.g
        d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True))
        fi = self.fix_idx[t]
        d = torch.where((fi >= 0)[:, None], self.fix_rows[fi.clamp(min=0)], d)
        return d * GAIN

    def logits(self, h):
        x = h.float(); x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g
        return x.to(self.E.dtype) @ self.E.T

    @staticmethod
    def top_in(lg, mask):
        lg = lg.float()
        mi = (lg + (1 - mask) * -1e30).max(-1).values; mo = (lg + mask * -1e30).max(-1).values
        return torch.clamp((mi - mo) * 1e6, 0, 1)

    @torch.no_grad()
    def write(self, h):
        """The write for the current position. Readouts (DISCLOSED): the host's own logits on its final state (frozen LM head);
        everything else is gates on circuit wires, fixed tables, and force writes."""
        g = self.gates(); a = self.ans; T = self.T
        lg = self.logits(h.float())                                                    # host readout (no write yet)
        V = lg.shape[-1]
        # ---- blocked token sets (0/1 over the vocabulary; fixed tables x gates)
        cnt = torch.where(g['p3'] > 0.5, self.d3, self.kp)
        short = g['para'] * (cnt < g['npar'] - 1).float()                              # more paragraphs needed
        eosb = torch.clamp(g['two'] * (self.k6 > 0).float() * (self.w6 < 5).float() * (self.n6 < 40).float() + self.hforced * self.hl_open * (self.hl_cont == 0).float()
                           + short * (1 - self.seg) + g['bul'] * self.ln_bul * (self.ln_cont == 0).float(), 0, 1)
        nnb = g['pn'] * (self.kp >= g['npar'] - 1).float()
        blk = (g['nc'][:, None] * T['COMMASET'][None]
               + torch.clamp(g['two'] * (self.k6 > 0).float() + g['p3'] + g['two'] * (self.k6 == 0).float() * (self.wa < 5).float(), 0, 1)[:, None] * T['STAR3SET'][None]
               + nnb[:, None] * T['NN'][None] + (nnb * self.at_ls)[:, None] * T['STARTNL'][None]
               + eosb[:, None] * self.EOSMASK[None]
               + (g['bul'] * self.at_ls * (self.bcount >= self.N_bul).float())[:, None] * T['BULST'][None]
               + (g['bul'] * self.ln_bul * (self.ln_cont == 0).float() * (1 - self.at_ls))[:, None] * T['NL'][None])      # no empty bullet
        # forbidden words: tokens that would COMPLETE a forbidden word given the word being written (rolling hash)
        cand = torch.where(T['LEAD'][None] > 0.5, T['HT'][None], (self.hcur[:, None] * T['POWT'][None] + T['HT'][None]) % HP)
        live = (self.slot_ok > 0.5) & (g['forbid'][:, None] > 0.5)
        tgt = torch.where(live, self.slot_h, -5)
        blk = blk + ((cand[:, None, :] == tgt[:, :, None]).any(1) & (T['HASAN'][None] > 0.5)).float()
        # anti-repeat at the start of a forced continuation: the next paragraph / second response may not open with the
        # same word as the previous paragraph / the first response
        rep_on = torch.clamp(g['para'] + g['high'], 0, 1) * self.forced_brk * (1 - self.seg) * (self.prev_pf >= 0).float()
        two_on = g['two'] * (self.k6 > 0).float() * (self.w6l == 0).float() * (self.ans_pf >= 0).float()
        pf = torch.where(two_on > 0.5, self.ans_pf, torch.where(rep_on > 0.5, self.prev_pf, -5))
        blk = blk + ((T['HT'][None] == pf[:, None]) & (T['HASAN'][None] > 0.5) & (T['LEAD'][None] + (1 - self.hcur.clamp(max=1).float())[:, None] > 0)).float()
        blk = torch.clamp(blk, 0, 1)
        top = lg.argmax(-1)
        blocked = blk.gather(1, top[:, None])[:, 0]
        alt = (lg.float() + blk * -1e30).argmax(-1)                                    # host's best token outside the blocked set
        chosen = torch.where(blocked > 0.5, alt, top)
        e_eos = self.EOSMASK[chosen]; e_nl = self.NLMASK[chosen]
        idle = (self.prog == 0).float() * a * self.body
        # swaps when the host wants a newline / to end
        tc = idle * g['title'] * self.t_open * torch.clamp(e_nl + e_eos, 0, 1)
        hc = idle * self.hforced * self.hl_open * (self.hl_cont > 0).float() * torch.clamp(e_nl + e_eos, 0, 1) * (1 - tc)
        self.start_prog(tc, 13, pre=[self.tk['>>']])
        self.start_prog(hc, 14, pre=[self.tk['*']])
        e = idle * e_eos * (1 - tc) * (1 - hc)
        c1 = e * short * self.seg
        c2 = e * g['two'] * (self.k6 == 0).float() * (1 - c1)
        r = (1 - c1) * (1 - c2)
        c3 = e * g['bul'] * (self.bcount < self.N_bul).float() * r; r = r * (1 - c3)
        c4 = e * g['high'] * (self.hcount < self.N_high).float() * r; r = r * (1 - c4)
        c5 = e * g['ps'] * (1 - self.ps_seen) * r; r = r * (1 - c5)
        c6 = e * g['end'] * (1 - self.end_done) * r; r = r * (1 - c6)
        c7 = e * g['quote'] * (1 - self.lq * (1 - self.lx)) * r
        nn, nl = self.tk['\n\n'], self.tk['\n']
        sep = torch.where(g['pn'] > 0.5, torch.where(self.at_ls > 0.5, -1, nl), nn)
        endsep = torch.where(g['pn'] > 0.5, torch.full_like(nn, 220), nn)
        hsep = torch.where(g['pn'] > 0.5, torch.full_like(nn, 220), nn)
        qsuf = torch.where(g['quote'] > 0.5, self.tk['"'], torch.full_like(nn, -1))
        eos = torch.full_like(nn, EOS)
        self.start_prog(c1, 6, pre=[nn])
        self.start_prog(c2, 7, pre=[nn, self.tk['******'], nn])
        self.start_prog(c3, 8, pre=[torch.where(self.at_ls > 0.5, self.tk['*'], nl), torch.where(self.at_ls > 0.5, -1, self.tk['*'])])
        self.start_prog(c4, 9, pre=[hsep, self.tk['*']])
        pps = (self.L['pps'] > 0.5)
        self.start_prog(c5 * pps.float(), 10, pre=[sep, self.tk['P'], self.tk['.P'], self.tk['.S']], suf=[self.tk['.']])
        self.start_prog(c5 * (~pps).float(), 10, pre=[sep, self.tk['P'], self.tk['.S'], self.tk['.']])
        x = torch.where(self.end_x >= 0, self.end_x, -1)
        self.start_prog(c6, 11, pre=[endsep], span=(self.end_s, self.end_len), suf=[x, qsuf, eos], fmap=1)
        self.start_prog(c7, 12, pre=[self.tk['"'], eos])
        # free generation: emit the host's chosen token when it differs from the host's own choice (blocked set) or when a
        # case rule needs it respelled (the emitter's case expansion does the respelling)
        free = idle * (1 - torch.clamp(tc + hc + c1 + c2 + c3 + c4 + c5 + c6 + c7, 0, 1))
        # bullets: while fewer than N bullets, a new line the host starts (after the answer has begun) becomes a bullet line
        bl = free * g['bul'] * self.at_ls * (self.bcount < self.N_bul).float() * (1 - T['BULST'][chosen]) \
            * (1 - self.EOSMASK[chosen]) * (1 - T['STARTNL'][chosen]) * (1 - T['ANGTOK'][chosen])
        self.start_prog(bl, 8, pre=[self.tk['-']])
        free = free * (1 - bl)
        need_case = g['caps'] * T['LOWERSET'][chosen] + g['lower'] * T['UPPERSET'][chosen]
        self.start_prog(free * torch.clamp(blocked + need_case, 0, 1), 15, pre=[chosen])
        # suffix/prefix lists may contain -1 holes: compact them
        sf = self.suf; order = torch.sort((sf < 0).long(), dim=1, stable=True).indices
        self.suf = sf.gather(1, order); self.suf_len = (self.suf >= 0).sum(1)
        pr = self.pre; order = torch.sort((pr < 0).long(), dim=1, stable=True).indices
        self.pre = pr.gather(1, order); self.pre_len = (self.pre >= 0).sum(1)
        active = (self.prog > 0) & (a > 0.5)
        tokf = self.emit_token()
        W = torch.where(active[:, None], self.force(tokf), torch.zeros_like(h.float()))
        self.fires = self.fires + active.float()
        # diagnostic only (does not change the write): did the forced token win?
        self.ffail = self.ffail + (active & (self.logits(h + W).argmax(-1) != tokf)).float()
        return W

    def report(self):
        g = {k: v.tolist() for k, v in self.L.items()}
        out = []
        for i in range(self.B):
            r = {k: 1 for k in g if g[k][i] > 0.5}
            r['N_par'] = int(self.N_par[i]); r['N_sec'] = int(self.N_sec[i]); r['N_bul'] = int(self.N_bul[i]); r['N_high'] = int(self.N_high[i])
            r['nth_k'] = int(self.nth_k[i]); r['nth_last'] = int(self.nth_last[i])
            r['nth_word'] = self.tok.decode(self.mem[i, self.nth_s[i]:self.nth_s[i] + self.nth_len[i]].tolist()) if r.get('nth') else ''
            r['end_phrase'] = (self.tok.decode(self.mem[i, self.end_s[i]:self.end_s[i] + self.end_len[i]].tolist()) +
                               (self.tok.decode([int(self.end_x[i])]) if self.end_x[i] >= 0 else '')) if r.get('end') else ''
            r['rep'] = self.tok.decode(self.mem[i, self.rep_s[i]:self.rep_s[i] + self.rep_len[i]].tolist()) if r.get('repeat') else ''
            r['forbid_h'] = [int(self.slot_h[i, s]) for s in range(S_W) if self.slot_ok[i, s] > 0.5]
            r['fires'] = int(self.fires[i]); r['ffail'] = int(self.ffail[i])
            out.append(r)
        return out
