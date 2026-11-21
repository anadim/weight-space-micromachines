"""CPU fuzz: m5div.step/apply (compiled registers + logit bias) vs a python reference of the first-line repulsion.
Drives the registers with random serial streams that follow comp_serial._sw's timing (END -> retry/repair trigger -> written
phrase -> host tokens; s_c / s_a / s_rep / s_wr semantics), random answer registers, all 3 gate modes.
usage: STK_DIR=stk python3 test_m5.py   (exit 0 + 'M5 FUZZ PASS')"""
import os, sys, random, math, torch
sys.path.insert(0, os.environ.get('STK_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stk')))
import m5div as M5D

NS, KS, PAD = 8, 16, 12
rng = random.Random(1)
# fake vocab: words, leading-space words, newline tokens, whitespace
words = ['Tom', 'has', 'First', 'We', 'So', 'x', '5', '=', 'total', 'Let']
txt = [''] + words + [' ' + w for w in words] + ['.\n', '\n', ' ', '\n\n', ':\n', ' 3', '3', 'Wait', ' again', '.']
V = len(txt)
NLs = [i for i, t in enumerate(txt) if '\n' in t]
canon, NL, WS = M5D.tables(txt, None)
PH = [txt.index('Wait'), txt.index(' again'), txt.index('.\n')]


class Eng: pass


def mk_line(prefixes):
    """a first line: often an exact copy / prefix-copy of an earlier one (to exercise the prefix match)."""
    if prefixes and rng.random() < 0.6:
        base = rng.choice(prefixes); k = rng.randint(0, len(base))
        line = list(base[:k])
    else:
        line = []
    while len(line) < rng.randint(1, 12): line.append(rng.randint(1, 2 * len(words)))
    if rng.random() < 0.8: line.append(rng.choice(NLs))
    if rng.random() < 0.15: line = [txt.index(' ')] + line          # leading whitespace token
    return line


def make_row():
    """-> list of segments: dict(kind first|wait|rep, toks host tokens (last = END token), ans=(ok, sym, lg) committed at END)."""
    natt = rng.randint(1, NS); segs = []; lines = []
    for a in range(natt):
        line = mk_line(lines)
        if a == 0 and line and rng.random() < 0.7 and line[0] <= len(words): line[0] += len(words)   # ' Tom' on attempt 0
        lines.append(line)
        body = line + [rng.randint(1, V - 1) for _ in range(rng.randint(0, 6))]
        if rng.random() < 0.1: body = line[:rng.randint(1, max(1, len(line)))]        # attempt ends inside the first line
        segs.append(dict(kind='first' if a == 0 else 'wait', toks=body))
        if rng.random() < 0.2:
            segs.append(dict(kind='rep', toks=[rng.randint(1, V - 1) for _ in range(rng.randint(1, 8))]))
    for s in segs:
        v = rng.choice([[1, 2], [3], [1, 2], [7, 7]])
        s['ans'] = (rng.random() < 0.85, v + [PAD] * (KS - len(v)), rng.uniform(-3, 4))
    return segs


def ref_pm(slots, a_new, mode, tau):
    """python reference of the repel flags over earlier slots: slots[i] = (ok, sym tuple, lg)."""
    prev = list(range(a_new))
    if mode == 0: return [i < a_new for i in range(NS)]
    ok = [i for i in prev if slots[i][0]]
    if mode == 1:
        un = len(ok) >= 1 and len({tuple(slots[i][1]) for i in ok}) == 1
        return [i < a_new and un for i in range(NS)]
    cnt = {i: sum(tuple(slots[j][1]) == tuple(slots[i][1]) for j in ok) for i in ok}
    top = max(cnt.values()) if cnt else 0
    out = [False] * NS
    for i in ok:
        if cnt[i] == top and top >= 2:
            m = sum(slots[j][2] for j in ok if tuple(slots[j][1]) == tuple(slots[i][1])) / cnt[i]
            out[i] = m < tau
    return out


def run(B, beta, P, mode, tau):
    e = Eng(); e.dev = 'cpu'
    Bp = B + 1
    e.R_ok = torch.zeros(Bp, NS, dtype=torch.bool); e.R_sym = torch.full((Bp, NS, KS), PAD, dtype=torch.long)
    e.R_lg = torch.full((Bp, NS), -30.0)
    e.s_a = torch.zeros(Bp, dtype=torch.long); e.s_c = torch.zeros(Bp, dtype=torch.long); e.s_rep = torch.zeros(Bp, dtype=torch.long)
    M5D.setup(e, f'{beta}/{P}/{mode}/{tau}', txt)
    M5D.reset(e)
    rows = [make_row() for _ in range(B)]
    # flatten each row into steps: (input tok, input written?, event at this step: None|'wait'|'rep'|'fin', writing, seg index)
    streams = []
    for segs in rows:
        st = []
        for si, s in enumerate(segs):
            for ti, t in enumerate(s['toks']):
                last = ti == len(s['toks']) - 1
                ev = None
                if last: ev = segs[si + 1]['kind'] if si + 1 < len(segs) else 'fin'
                st.append((t, False, ev, ev is not None, si))
            if si + 1 < len(segs):
                for k, ph in enumerate(PH): st.append((ph, True, None, k < len(PH) - 1, si + 1))
        streams.append(st)
    T = max(map(len, streams))
    # reference state
    R = [dict(lines={}, slots=[(False, [PAD] * KS, -30.0)] * NS, a=0, rep=0, c=0, l1=True, seen=False, cur=[], pm=[False] * NS,
              done=False) for _ in range(B)]
    nbad = 0; nact = 0
    for t in range(T):
        tok = torch.zeros(B, dtype=torch.long); rd = torch.zeros(B, dtype=torch.bool); live = torch.zeros(B, dtype=torch.bool)
        dw = torch.zeros(B, dtype=torch.bool); ns = torch.zeros(B, dtype=torch.bool); wr = torch.zeros(B, dtype=torch.bool)
        j0 = e.s_c[:B].clone()
        for b in range(B):
            if t >= len(streams[b]) or R[b]['done']: continue
            tk, w, ev, writing, si = streams[b][t]
            tok[b] = tk; rd[b] = not w; live[b] = True; wr[b] = writing
            # comp_serial order: s_c += rd ; END commit ; triggers ; s_a / s_rep / s_c updates
            r = R[b]
            if not w: e.s_c[b] += 1
            if ev is not None:
                ok, sym, lg = rows[b][si]['ans']
                e.R_ok[b, r['a']] = ok; e.R_sym[b, r['a']] = torch.tensor(sym); e.R_lg[b, r['a']] = lg
                r['slots'] = list(r['slots']); r['slots'][r['a']] = (ok, sym, lg)
            if ev == 'wait':
                dw[b] = True; ns[b] = True; e.s_a[b] += 1; e.s_rep[b] = 0; e.s_c[b] = 0
            elif ev == 'rep':
                ns[b] = True; e.s_rep[b] += 1; e.s_c[b] = 0
        M5D.step(e, B, tok, rd, live, j0, dw, ns, wr)
        logits = torch.zeros(B, V, dtype=torch.bfloat16)
        M5D.apply(e, B, logits)
        # ---- python reference
        for b in range(B):
            if t >= len(streams[b]) or R[b]['done']: continue
            r = R[b]; tk, w, ev, writing, si = streams[b][t]
            if not w and ev not in ('wait', 'rep') and r['rep'] == 0 and r['l1'] and r['c'] < P:
                rec = int(canon[tk]) if r['c'] == 0 else tk
                r['cur'].append(rec)
            if not w:
                r['c'] += 1
                r['seen'] = r['seen'] or not bool(WS[tk])
                if bool(NL[tk]) and r['seen']: r['l1'] = False
            if ev == 'wait':
                r['lines'][r['a']] = r['cur']; r['a'] += 1; r['rep'] = 0; r['c'] = 0; r['cur'] = []; r['l1'] = True; r['seen'] = False
                r['pm'] = ref_pm(r['slots'], r['a'], mode, tau)
            elif ev == 'rep':
                r['lines'][r['a']] = r['cur'] if r['a'] not in r['lines'] else r['lines'][r['a']]
                r['rep'] += 1; r['c'] = 0
            elif ev == 'fin':
                r['done'] = True
            exp = {}
            if not writing and r['rep'] == 0 and r['l1'] and not r['done'] and ev != 'fin':
                fls = [r['lines'].get(i, []) for i in range(r['a'])]
                exp = M5D.bias_ref(fls, r['cur'], r['pm'][:r['a']], beta, P)
            got = {i: float(v) for i, v in enumerate(logits[b].float().tolist()) if v != 0}
            ok_ = set(exp) == set(got) and all(abs(got[k] - exp[k]) <= 0.02 * beta for k in exp)
            nact += bool(exp)
            if not ok_:
                nbad += 1
                if nbad <= 5: print('MISMATCH row', b, 't', t, 'a', r['a'], 'exp', exp, 'got', got, 'cur', r['cur'], 'lines', r['lines'])
    return nbad, nact


tot_bad = tot_act = 0
for it in range(60):
    mode = it % 3; beta = rng.choice([1.0, 3.0, 30.0]); P = rng.choice([1, 3, 6, 24]); tau = rng.choice([-1.0, 1.0, 2.0, 1e9])
    nb, na = run(16, beta, P, mode, tau); tot_bad += nb; tot_act += na
print('steps with a bias', tot_act, 'mismatches', tot_bad)
print('M5 FUZZ PASS' if tot_bad == 0 and tot_act > 100 else 'M5 FUZZ FAIL')
sys.exit(0 if tot_bad == 0 and tot_act > 100 else 1)
