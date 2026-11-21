"""M5 first-step diversity implant for the all-weights serial stack (comp_serial.SerialW), env M5DIV='beta/P/mode/tau'.

Each attempt's first line (its first P host tokens, while the line lasts) is stored in a per-slot token register R_fl.
While a later attempt writes ITS first line, a compiled logit bias -beta lowers the logit of the token that continued an
earlier attempt's first line at the same position, for every earlier attempt whose first line the current one still
reproduces token for token (prefix-match bit register s_m).  Which earlier attempts are repelled (bit register s_pm, set at the
retry trigger from the answer register) is a fixed comparator on R_ok / R_sym / R_lg:
  mode 0: all earlier attempts
  mode 1: all earlier attempts, only if the valid ones so far are unanimous (symbol-equal) -> attractor case
  mode 2: the plurality answer group (count >= 2, symbol-equal) if its mean probe logit < tau
Everything is fixed tables (canon[tok]: leading-space variant -> bare token, for position 0 only; NL[tok]; WS[tok]), registers
(R_fl, s_m, s_pm, s_l1 in-first-line bit, s_seen), counters (the existing segment host-token counter s_c = position, s_rep) and a
fixed scatter of -beta onto the logits (= a fixed one-hot output-bias table indexed by the register).  No host decision.
Pure torch (imported by comp_serial and by the CPU fuzz test)."""
import os, torch, torch.nn.functional as F

NSYM = 13


def parse(spec):
    f = (spec or '').split('/')
    beta = float(f[0]); P = int(f[1]) if len(f) > 1 else 24; mode = int(f[2]) if len(f) > 2 else 0
    tau = float(f[3]) if len(f) > 3 else 1e9
    return beta, P, mode, tau


def tables(txt, tok_ids_of):
    """canon[v] = id of t.lstrip(' ') if t starts with ' ' and that bare string is one token, else v; NL[v] = '\\n' in t;
    WS[v] = t is whitespace only (or empty)."""
    V = len(txt)
    canon = torch.arange(V)
    rev = {}
    for v, t in enumerate(txt):
        if t and t not in rev: rev[t] = v
    for v, t in enumerate(txt):
        if t.startswith(' ') and len(t) > 1 and t[1:] in rev and not t[1:].startswith(' '): canon[v] = rev[t[1:]]
    NL = torch.tensor(['\n' in t for t in txt])
    WS = torch.tensor([t.strip() == '' for t in txt])
    return canon, NL, WS


def setup(eng, spec, txt):
    dev = eng.dev; NS = eng.R_ok.shape[1]; Bp = eng.R_ok.shape[0]
    eng.m5_beta, eng.m5_P, eng.m5_mode, eng.m5_tau = parse(spec)
    c, nl, ws = tables(txt, None)
    eng.t5_canon, eng.t5_NL, eng.t5_WS = c.to(dev), nl.to(dev), ws.to(dev)
    eng.R_fl = torch.full((Bp, NS, eng.m5_P), -1, device=dev, dtype=torch.long)
    eng.s5_m = torch.zeros(Bp, NS, device=dev, dtype=torch.bool)
    eng.s5_pm = torch.zeros(Bp, NS, device=dev, dtype=torch.bool)
    eng.s5_l1 = torch.ones(Bp, device=dev, dtype=torch.bool)
    eng.s5_seen = torch.zeros(Bp, device=dev, dtype=torch.bool)
    eng.s5_bias = torch.zeros(Bp, NS, device=dev)               # per-step scratch: -beta / dup on the penalised slots
    eng.s5_ids = torch.zeros(Bp, NS, device=dev, dtype=torch.long)
    eng.m5_on = True


def regs(eng):
    return [eng.R_fl, eng.s5_m, eng.s5_pm, eng.s5_l1, eng.s5_seen]


def reset(eng):
    eng.R_fl.fill_(-1); eng.s5_m.zero_(); eng.s5_pm.zero_(); eng.s5_l1.fill_(True); eng.s5_seen.zero_()


def repel_mask(R_ok, R_sym, R_lg, a_new, mode, tau):
    """(B, NS) bool: earlier slots (< a_new) whose first line the new attempt is pushed away from."""
    NS, KS = R_sym.shape[1], R_sym.shape[2]
    ar = torch.arange(NS, device=R_ok.device)
    prev = ar[None] < a_new[:, None]
    if mode == 0: return prev
    ok = R_ok & prev
    Rh = F.one_hot(R_sym, NSYM).float()
    eq = torch.relu(torch.einsum('bikc,bjkc->bij', Rh, Rh) - (KS - 1))          # 1 iff all symbols equal
    cnt = torch.einsum('bij,bj->bi', eq, ok.float())                            # group size of slot i (valid slots)
    if mode == 1:
        nok = ok.float().sum(1)
        unan = (nok >= 1) & ((cnt * ok.float()).amax(1) >= nok)                 # one group holds every valid slot
        return prev & unan[:, None]
    # mode 2: the plurality group (count >= 2) with mean probe logit < tau
    cv = cnt * ok.float()
    top = cv.amax(1, keepdim=True)
    sl = torch.einsum('bij,bj->bi', eq, ok.float() * R_lg)
    inG = ok & (cv >= top) & (top >= 2)
    return inG & (sl < tau * cnt)


def step(eng, B, tok, rd, live, j0, do_wait, newseg, writing):
    """after comp_serial._sw's register updates: record / prefix-match / gate, and the bias of this step's logits.
    j0 = s_c before this step (position of the input token in the segment's host text)."""
    dev = tok.device; NS = eng.R_ok.shape[1]; P = eng.m5_P
    bi = torch.arange(B, device=dev)
    a = eng.s_a[:B].clamp(max=NS - 1)
    # ---- record + prefix match (the input token, if host-written, in the first line of an un-repaired attempt)
    l1 = eng.s5_l1[:B]
    recon = rd & l1 & (j0 < P) & ~newseg & (eng.s_rep[:B] == 0)
    rec = torch.where(j0 == 0, eng.t5_canon[tok], tok)
    jj = j0.clamp(0, P - 1)
    # the attempt index that wrote the input token: on a newseg step s_a may already be the new one, recon excludes newseg
    cur = eng.R_fl[bi, a, jj]
    eng.R_fl[bi, a, jj] = torch.where(recon, rec, cur)
    prevtok = eng.R_fl[:B][bi[:, None], torch.arange(NS, device=dev)[None], jj[:, None]]      # (B, NS) earlier lines at j0
    eng.s5_m[:B] = eng.s5_m[:B] & ~(recon[:, None] & (prevtok != rec[:, None]))
    seen = eng.s5_seen[:B] | (rd & ~eng.t5_WS[tok])
    l1n = l1 & ~(rd & eng.t5_NL[tok] & seen)
    # ---- retry trigger: new attempt -> first-line flag on, its row cleared, repel mask from the answer register
    pm = repel_mask(eng.R_ok[:B], eng.R_sym[:B], eng.R_lg[:B], eng.s_a[:B], eng.m5_mode, eng.m5_tau)
    eng.s5_pm[:B] = torch.where(do_wait[:, None], pm, eng.s5_pm[:B])
    eng.s5_m[:B] = torch.where(do_wait[:, None], pm, eng.s5_m[:B])
    clr = F.one_hot(eng.s_a[:B].clamp(max=NS - 1), NS).bool() & do_wait[:, None]
    eng.R_fl[:B] = torch.where(clr[:, :, None], torch.full_like(eng.R_fl[:B], -1), eng.R_fl[:B])
    eng.s5_l1[:B] = l1n | do_wait
    eng.s5_seen[:B] = seen & ~do_wait
    # ---- bias for the token sampled now (position j1 = s_c after this step)
    j1 = eng.s_c[:B]
    act_row = live & ~writing & eng.s5_l1[:B] & (eng.s_rep[:B] == 0) & (j1 < P)
    ids = eng.R_fl[:B][bi[:, None], torch.arange(NS, device=dev)[None], j1.clamp(0, P - 1)[:, None]]   # (B, NS)
    act = act_row[:, None] & eng.s5_m[:B] & eng.s5_pm[:B] & (ids >= 0)
    dup = ((ids[:, :, None] == ids[:, None, :]) & act[:, None, :]).float().sum(-1).clamp(min=1)
    eng.s5_ids[:B] = ids.clamp(min=0)
    eng.s5_bias[:B] = torch.where(act, -eng.m5_beta / dup, torch.zeros_like(dup))


def apply(eng, B, logits):
    """fixed output-bias table: logits[b, R_fl id] += bias (scatter_add of -beta / multiplicity = -beta per distinct token)."""
    logits.scatter_add_(1, eng.s5_ids[:B], eng.s5_bias[:B].to(logits.dtype))


def bias_ref(first_lines, cur_prefix, pm, beta, P):
    """python reference of the bias at the next position: first_lines = list of earlier attempts' first-line ids (canonical),
    cur_prefix = the current attempt's first-line ids so far (canonical), pm = repel flags.  -> {token: bias}"""
    j = len(cur_prefix); out = {}
    if j >= P: return out
    for fl, p in zip(first_lines, pm):
        if not p or len(fl) <= j: continue
        if list(fl[:j]) != list(cur_prefix): continue
        out[fl[j]] = -beta
    return out
