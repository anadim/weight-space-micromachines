"""Reference-side helpers shared by lockstep.py / run_c.py: read circuit.py's registers under the compiled names, and the
reference blocked set (verbatim copy of circuit.py write() lines 702-728)."""
import torch
from tables import CLASSES
CI = {c: i for i, c in enumerate(CLASSES)}
REGS = None
SAT = {}
CAPREGS = ['acc', 'num_recent', 'N_par', 'N_sec', 'N_bul', 'N_high', 'nth_k', 's_k', 's_hn_first', 's_hn_last', 'sc_bul', 'sc_par', 'sc_sec']


def ref_regs(c):
    out = {}
    B = c.B
    for n in REGS:
        if n == 't': v = torch.full((B,), float(c.t))
        elif n.startswith('win'): v = c.win[:, int(n[3:])]
        elif n.startswith('cl.'): v = c.cl[:, CI[n[3:]]]
        elif n.startswith('se.'): v = c.se[:, CI[n[3:]]]
        elif n.startswith('L.'): v = c.L[n[2:]]
        elif n.startswith('Ls.'): v = c.Ls[n[3:]]
        elif n.startswith('done.'): v = c.done[n[5:]]
        elif n.startswith('cand.'): v = c.cand[n[5:]]
        elif n.startswith('cue.'):
            _, k, w = n.split('.'); v = c.cueR[k][0 if w == 'clc' else 1] if k in c.cueR else torch.zeros(B)
        elif n.startswith('slot_h'): v = c.slot_h[:, int(n[6:])]
        elif n.startswith('slot_ok'): v = c.slot_ok[:, int(n[7:])]
        elif n.startswith('pre') and n[3:].isdigit(): v = c.pre[:, int(n[3:])]
        elif n.startswith('suf') and n[3:].isdigit(): v = c.suf[:, int(n[3:])]
        else: v = getattr(c, n)
        out[n] = v.double() if torch.is_tensor(v) else torch.full((B,), float(v), dtype=torch.float64)
    return out


def ref_blk(c):
    """verbatim copy of circuit.py write() lines 702-728 (the blocked set, before any write)."""
    g = c.gates(); a_ = c.ans; T = c.T
    cnt = torch.where(g['p3'] > 0.5, c.d3, c.kp)
    short = g['para'] * (cnt < g['npar'] - 1).float()
    eosb = torch.clamp(g['two'] * (c.k6 > 0).float() * (c.w6 < 5).float() * (c.n6 < 40).float() + c.hforced * c.hl_open * (c.hl_cont == 0).float()
                       + short * (1 - c.seg) + g['bul'] * c.ln_bul * (c.ln_cont == 0).float(), 0, 1)
    nnb = g['pn'] * (c.kp >= g['npar'] - 1).float()
    blk = (g['nc'][:, None] * T['COMMASET'][None]
           + torch.clamp(g['two'] * (c.k6 > 0).float() + g['p3'] + g['two'] * (c.k6 == 0).float() * (c.wa < 5).float(), 0, 1)[:, None] * T['STAR3SET'][None]
           + nnb[:, None] * T['NN'][None] + (nnb * c.at_ls)[:, None] * T['STARTNL'][None]
           + eosb[:, None] * c.EOSMASK[None]
           + (g['bul'] * c.at_ls * (c.bcount >= c.N_bul).float())[:, None] * T['BULST'][None]
           + (g['bul'] * c.ln_bul * (c.ln_cont == 0).float() * (1 - c.at_ls))[:, None] * T['NL'][None])
    HP = 2147483647
    cand = torch.where(T['LEAD'][None] > 0.5, T['HT'][None], (c.hcur[:, None] * T['POWT'][None] + T['HT'][None]) % HP)
    live = (c.slot_ok > 0.5) & (g['forbid'][:, None] > 0.5)
    tgt = torch.where(live, c.slot_h, -5)
    blk = blk + ((cand[:, None, :] == tgt[:, :, None]).any(1) & (T['HASAN'][None] > 0.5)).float()
    rep_on = torch.clamp(g['para'] + g['high'], 0, 1) * c.forced_brk * (1 - c.seg) * (c.prev_pf >= 0).float()
    two_on = g['two'] * (c.k6 > 0).float() * (c.w6l == 0).float() * (c.ans_pf >= 0).float()
    pf = torch.where(two_on > 0.5, c.ans_pf, torch.where(rep_on > 0.5, c.prev_pf, -5))
    blk = blk + ((T['HT'][None] == pf[:, None]) & (T['HASAN'][None] > 0.5) & (T['LEAD'][None] + (1 - c.hcur.clamp(max=1).float())[:, None] > 0)).float()
    return torch.clamp(blk, 0, 1)



def cmp_regs(c, ex):
    global REGS
    REGS = ex.regs
    R = ref_regs(c)
    # documented saturation: the compiled digit accumulator (and every register that copies a number) holds min(value, 10^6);
    # circuit.py keeps the float32 value. Compare min(ref, 10^6) and count the rows where the reference exceeded it.
    global SAT
    for n in CAPREGS:
        over = R[n] > 10 ** 6
        if bool(over.any()): SAT[n] = SAT.get(n, 0) + int(over.sum())
        R[n] = R[n].clamp(max=10 ** 6)
    ref = torch.stack([R[n].to(ex.state.device) for n in ex.regs], 1)
    if torch.equal(ref, ex.state): return []                       # one comparison (one sync) in the common case
    bad = []
    for j, n in enumerate(ex.regs):
        if not torch.equal(ex.state[:, j], ref[:, j]):
            rows = torch.nonzero(ex.state[:, j] != ref[:, j])[:, 0].tolist()
            bad.append((n, rows[:6], ref[rows[:6], j].tolist(), ex.state[rows[:6], j].tolist()))
    return bad
