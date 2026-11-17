"""Phase 2: rule-keepers compiled into weight-space circuits (no Python rule logic at run time).

  token embeddings --reader (linear map, exact 0/1 flags, checked on the whole vocabulary)--> symbol flags
  window of K flags --stack 1 (ReLU net written with wsm_hdl)--> per position: pattern matches, role-header flags, first-response flag
  hard attention head per rule ('latch'): key = the rule's pattern match; copies 1 from the most recent match anywhere before
  hard attention head ('turn'): key = a role header (<|im_start|> + role); copies [role == assistant]
  stack 2 (ReLU): fire_r = AND(latch_r, in_assistant_turn) (+ first-position / not-after-closing-quote for 'quote')
  write = sum_r fire_r * V_r   (V_r: the solved veto vectors / the '"' emit direction of Phase 1). fire = 0 -> exactly 0.

Rules compiled: no_comma, lower, caps (vetoes), quote_open (emit '"' as the first answer token).
quote_close (end-of-turn -> '"') needs the host's end-of-turn decision: it stays the Phase-1 probe (see gen.py), gated here by
the compiled fire flag.
"""
import re, torch, torch.nn as nn, torch.nn.functional as Fn
from wsm_hdl import Net, Lin, lsum, compile_net, ReLUNet

BIG, GAMMA = 1e8, 300.0          # hard attention: exact for sequences up to BIG/GAMMA ~ 3e5 tokens
K = 16                      # stack-1 window (tokens): offset 0 = current token


def core(s): return s.strip().lower()


# ---------------- symbol groups (predicates on the token's text) ----------------
GROUPS = {
    'COMMAW': lambda s: core(s) in ('comma', 'commas'),
    'NEG': lambda s: core(s) in ('no', 'not', 'without', 'avoid', 'cannot', 'never', "n't", "'t", 'refrain', 'ain', 'rain') or s in ("'t", "’t"),
    'CONTAIN': lambda s: core(s) == 'contain',
    'CAPZ': lambda s: core(s) in ('capitalized', 'capitalised'),
    'BE': lambda s: core(s) == 'be',
    'CAPSW': lambda s: core(s) == 'caps',
    'AROUND': lambda s: core(s) == 'around',
    'MARKS': lambda s: core(s) == 'marks',
    'FILL': lambda s: core(s) in ('using', 'any', 'use', 'include', 'contain', 'have', 'from', 'of', 'free', 'should', 'must', 'be'),
    'LOWERW': lambda s: core(s) in ('lowercase', 'lowercases', 'lowercased'),
    'LOWER': lambda s: core(s) == 'lower',
    'CASE': lambda s: core(s) in ('case', 'cases'),
    'CAPITAL': lambda s: core(s) in ('capital', 'uppercase'),
    'LETTERS': lambda s: core(s) in ('letters', 'letter'),
    'ALLW': lambda s: core(s) in ('all', 'only'),
    'WORDS': lambda s: core(s) in ('words', 'word'),
    'WRAP': lambda s: core(s) in ('wrap', 'wrapped', 'put', 'enclose', 'enclosed'),
    'DOUBLE': lambda s: core(s) == 'double',
    'QUOT': lambda s: core(s) in ('quotation', 'quotes', 'quote'),
    'IMSTART': lambda s: s == '<|im_start|>',
    'ASSISTANT': lambda s: s == 'assistant',
    'ROLE': lambda s: s in ('user', 'system', 'assistant'),
    'ENDTHINK': lambda s: s == '</think>',
    'NN': lambda s: s == '\n\n',
    'QCLOSE': lambda s: s.rstrip().endswith('"') and '"' in s,
}
SYMS = list(GROUPS)
RULES = ['no_comma', 'lower', 'caps', 'quote']


def build_stack1():
    """Inputs: K windows of n_sym flags, layout [offset][sym]. Outputs: match_no_comma, match_lower, match_caps, match_quote,
    is_header, header_is_assistant, first_pos, qclose."""
    n = len(SYMS); net = Net(K * n)
    f = lambda o, s: net.inp(o * n + SYMS.index(s))
    OR = lambda xs: net.clip01(lsum(xs))
    # no comma: COMMAW at 0, then j FILL tokens, then NEG (or 'fr''ain') at j+1, j = 0..3
    alts = []
    for j in range(0, 4):
        fills = [f(o, 'FILL') for o in range(1, j + 1)]
        alts.append(net.AND(f(0, 'COMMAW'), *fills, f(j + 1, 'NEG')))
    m_nc = OR(alts)
    # lowercase: 'lowercase' anywhere | 'lower' 'case' | NEG (FILL)* 'capital' 'letters'
    alts = [net.relu(f(0, 'LOWERW') - f(1, 'CONTAIN')), net.AND(f(0, 'CASE'), f(1, 'LOWER'))]
    for j in range(0, 3):
        fills = [f(o, 'FILL') for o in range(2, 2 + j)]
        alts.append(net.AND(f(0, 'LETTERS'), f(1, 'CAPITAL'), *fills, f(2 + j, 'NEG')))
    m_lo = OR(alts)
    # all caps: 'all'/'only' 'capital'/'uppercase' 'letters'  (not preceded by 'words' within the window: 'words with all capital letters')
    letters_near = OR([f(o, 'LETTERS') for o in range(3, 9)])
    m_caps_raw = OR([net.AND(f(0, 'LETTERS'), f(1, 'CAPITAL'), f(2, 'ALLW')), net.AND(f(0, 'CAPSW'), f(1, 'ALLW')),
                     net.AND(f(0, 'LOWERW'), f(1, 'CONTAIN')),                         # '... should contain lowercase letters' (negated)
                     net.AND(f(0, 'CAPZ'), f(1, 'BE'), letters_near)])                 # 'all letters in your response should be capitalized'
    words_near = OR([f(o, 'WORDS') for o in range(2, 6)])
    m_caps = net.relu(m_caps_raw - words_near)
    # quote: 'double' 'quotation'/'quotes'  (the wrap verb is not checked by the circuit)
    wrap_near = OR([f(o, 'WRAP') for o in range(2, K)])
    m_q = OR([net.AND(f(0, 'QUOT'), f(1, 'DOUBLE'), wrap_near), net.AND(f(0, 'AROUND'), f(1, 'MARKS'), f(2, 'QUOT'), f(3, 'DOUBLE'))])
    is_hdr = net.AND(f(0, 'ROLE'), f(1, 'IMSTART'))
    hdr_asst = net.AND(f(0, 'ASSISTANT'), f(1, 'IMSTART'))
    first = net.AND(f(0, 'NN'), f(1, 'ENDTHINK'))
    qclose = f(0, 'QCLOSE')
    outs = [m_nc, m_lo, m_caps, m_q, is_hdr, hdr_asst, first, qclose]
    return compile_net(net, outs)


def build_stack2():
    """Inputs: latch_r (4), in_asst, first_pos, qclose_prev -> fire_no_comma, fire_lower, fire_caps, fire_quote_open, fire_quote_close_gate."""
    net = Net(7)
    L = [net.inp(i) for i in range(4)]; ia, fp, qc = net.inp(4), net.inp(5), net.inp(6)
    outs = [net.AND(L[0], ia), net.AND(L[1], ia), net.AND(L[2], ia),
            net.AND(L[3], ia, fp),                                     # emit '"' at the first answer position
            net.relu(net.AND(L[3], ia) - fp - qc)]                    # else: end-of-turn -> '"' unless the last token closed a quote
    return compile_net(net, outs)


def make_reader(E, groups):
    """Exact linear reader for each symbol group (closed form; max-margin fallback); falls back to a 0/1 lookup row if a
    group is not linearly separable (reported)."""
    E = E.float(); W, b, table = [], [], {}
    for k, ids in groups.items():
        t = torch.tensor(ids, device=E.device)
        w = (E[t] / E[t].norm(dim=-1, keepdim=True)).mean(0); w = w / w.norm(); sc = E @ w
        own = sc[t].min(); scm = sc.clone(); scm[t] = -1e9; other = scm.max(); gap = (own - other).item()
        if gap <= 0:
            pos = torch.zeros(E.shape[0], dtype=torch.bool, device=E.device); pos[t] = True
            Ew = E / E.norm(dim=-1, keepdim=True); wv = Ew[t].mean(0).clone().requires_grad_(True); bb = torch.zeros((), device=E.device, requires_grad=True)
            with torch.enable_grad():
                opt = torch.optim.Adam([wv, bb], lr=0.01)
                for _ in range(3000):
                    s_ = Ew @ wv + bb; loss = torch.relu(1 - s_[pos]).mean() * 50 + torch.relu(1 + s_[~pos]).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            w = wv.detach() / E.norm(dim=-1).mean(); sc = E @ w
            own = sc[t].min(); scm = sc.clone(); scm[t] = -1e9; other = scm.max(); gap = (own - other).item()
        if gap <= 0:
            table[k] = ids; W.append(torch.zeros_like(w)); b.append(torch.tensor(0., device=E.device)); continue
        a = 8.0 / gap; W.append(w * a); b.append(-a * (other + gap / 4))
    return torch.stack(W), torch.stack(b), table


class RuleKeeperCircuit(nn.Module):
    def __init__(self, tok, strs, E, V):
        super().__init__()
        groups = {k: [i for i, s in enumerate(strs) if f(s)] for k, f in GROUPS.items()}
        self.groups = groups
        W, b, table = make_reader(E, groups)
        self.register_buffer('rw', W); self.register_buffer('rb', b)
        T = torch.zeros(E.shape[0], len(SYMS), device=E.device)            # lookup fallback (only for non-separable groups)
        for k, ids in table.items(): T[ids, SYMS.index(k)] = 1
        self.register_buffer('rt', T); self.table = table
        self.s1, self.s2 = ReLUNet(build_stack1()).to(E.device), ReLUNet(build_stack2()).to(E.device)
        self.register_buffer('V', V)                                       # (5, d): veto comma, veto upper, veto lower, emit '"', 0
        self.reset()

    def flags(self, emb, ids):
        return torch.clamp(Fn.linear(emb.float(), self.rw, self.rb), 0, 1) + self.rt[ids]

    def check_reader(self, E):
        ids = torch.arange(E.shape[0], device=E.device); out = []
        for i in range(0, E.shape[0], 8192):
            out.append(self.flags(E[i:i + 8192], ids[i:i + 8192]))
        F = torch.cat(out); want = torch.zeros_like(F)
        for k, g in self.groups.items(): want[g, SYMS.index(k)] = 1
        return torch.equal(F, want)

    def reset(self): self.state = None

    def step(self, emb, ids):
        """emb (B,d) / ids (B,) of the newest token(s) -> write (B,d), fire flags (B,5). Incremental, exact."""
        s = self.flags(emb, ids)                                             # (B, n)
        B = s.shape[0]; n = len(SYMS)
        if self.state is None:
            self.state = dict(win=s.new_zeros(B, K, n), latch=s.new_zeros(B, 4), asst=s.new_zeros(B), qc=s.new_zeros(B))
        st = self.state
        win = torch.cat([s[:, None], st['win'][:, :-1]], 1)                  # offset 0 = newest
        o1 = self.s1(win.reshape(B, K * n))
        m, hdr, hdr_a, first, qclose = o1[:, :4], o1[:, 4], o1[:, 5], o1[:, 6], o1[:, 7]
        # hard attention heads, incremental form: the most recent key wins (BIG score); no key -> keep previous copy
        latch = torch.maximum(st['latch'], m)                                # latch head: any match so far (value 1 at keys)
        asst = torch.where(hdr > 0.5, hdr_a, st['asst'])                     # turn head: value at the most recent role header
        o2 = self.s2(torch.cat([latch, asst[:, None], first[:, None], qclose[:, None]], -1))
        self.state = dict(win=win, latch=latch, asst=asst, qc=qclose)
        return o2 @ self.V, o2

    def run(self, emb, ids):
        """Full-sequence teacher forcing, the same circuit position by position (for checks)."""
        self.reset(); W, Fl = [], []
        for t in range(emb.shape[1]):
            w, f = self.step(emb[:, t], ids[:, t]); W.append(w); Fl.append(f)
        self.reset()
        return torch.stack(W, 1), torch.stack(Fl, 1)


def run_seq(C, emb, ids):
    """Vectorised full-sequence forward (B=1), identical math to step(): windows -> stack1 -> heads as attention -> stack2."""
    s = C.flags(emb, ids)[0]                                                    # (T, n)
    T, n = s.shape
    pad = torch.cat([s.new_zeros(K - 1, n), s])
    win = torch.stack([pad[K - 1 - o: K - 1 - o + T] for o in range(K)], 1)     # (T, K, n), offset 0 = current
    o1 = C.s1(win.reshape(T, K * n))
    m, hdr, hdr_a, first, qclose = o1[:, :4], o1[:, 4], o1[:, 5], o1[:, 6], o1[:, 7]
    i = torch.arange(T, device=s.device); dist = (i[:, None] - i[None, :]).float()
    def head(key, val):                                                         # hard attention: most recent key token (causal)
        sc = BIG * key[None, :] - GAMMA * dist
        sc = sc.masked_fill(dist < 0, float('-inf'))
        return torch.softmax(sc, -1) @ val
    latch = torch.stack([head(m[:, r], torch.cummax(m[:, r], 0).values) for r in range(4)], 1)
    asst = head(hdr, hdr_a)
    o2 = C.s2(torch.cat([latch, asst[:, None], first[:, None], qclose[:, None]], -1))
    return o2 @ C.V, o2, latch
