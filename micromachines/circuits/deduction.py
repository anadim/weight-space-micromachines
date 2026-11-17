"""Natural-text ordering solver (fork B): BBH logical_deduction, read straight from the question, no tags.

stack 1 runs at ' Let' of 'A: Let's think' (the model never writes ' Let' in its own CoT). Over the current question it
  - finds every object mention (closed vocabulary of the task's 50 objects, ld_vocab_B.py), numbers the objects by first
    appearance (= the order of the options),
  - reads each sentence '<S> <predicate> [<Y>].' : relational (left of / right of / older / newer / more / less expensive /
    above / below) or absolute (leftmost, k-th from the right, k-th-newest, cheapest, finished k-th, k-th-to-last ...),
  - checks every ordering of n = 3..7 objects at once (one ReLU unit per ordering, 5166 units) -> exactly one must survive,
  - reads the asked position from the last option line ('(G) The X is the rightmost' right before 'A:').
It writes the answer letter at ' answer is (' (stack 2, via the attention hop). Silent unless exactly one ordering survives.
Positions are canonical 1..n: left / oldest / most expensive / first place = 1.
"""
import itertools
from ..hdl import Net, Lin, lsum, compile_net
from .deduction_vocab import OBJ, REL, ABS

NMAX = 7
EXTRA = ['.', '.\n', '\n', 'Q', 'A', ':', ' Let', ' (', '(', ' answer']
VOC = sorted({t for v in OBJ.values() for t in v} | {t for p, _ in REL for t in p} | {t for p, _, _ in ABS for t in p} | set(EXTRA))
SYMS = [f'T{i}' for i in range(len(VOC))]
NS = len(SYMS); SI = {t: i for i, t in enumerate(VOC)}
K = 360
LET = [chr(65 + i) for i in range(NMAX)]
OUT = LET
N_STATE = 1 + NMAX
TEXT = {**{f'T{i}': t for i, t in enumerate(VOC)}, **{c: c for c in LET}}
TYPES = list(OBJ)                       # object types
MAXJ = 8                                # constraint sentences handled


class Win:
    def __init__(self, net): self.net = net
    def t(self, o, tok): return self.net.inp(o * NS + SI[tok]) if 0 <= o < K else Lin()
    def seq(self, o_last, toks):
        """flags that toks (in text order) end at offset o_last (the last token at o_last, the first at o_last+len-1)"""
        m = len(toks); return [self.t(o_last + m - 1 - j, x) for j, x in enumerate(toks)]


def build_stack1():
    net = Net(K * NS); w = Win(net); A = net.AND
    qcum, acc = {}, Lin()
    for o in range(K):
        acc = acc + w.t(o, 'Q'); qcum[o] = acc
    inq = {o: 1 - net.clip01(qcum[o]) for o in range(K)}
    found_q = net.clip01(qcum[K - 1])
    # ---- object mentions: mt[x][v] = type v ends at offset x (inside the question) ----
    mt = {(x, v): A(*w.seq(x, OBJ[v]), inq[x + len(OBJ[v]) - 1]) for x in range(1, K) for v in TYPES if x + len(OBJ[v]) - 1 < K}
    mtg = lambda x, v: mt.get((x, v), Lin())
    present, fp = {}, {}
    for v in TYPES:
        later = Lin(); firsts = []
        for x in range(K - 1, 0, -1):                                # earlier text = larger offset
            if (x, v) in mt: firsts.append((x, A(mt[x, v], 1 - net.clip01(later)))); later = later + mt[x, v]
        present[v] = net.clip01(later)
        fp[v] = lsum(x * f for x, f in firsts)                         # offset of the first mention (0 if absent)
    npres = lsum(present.values())
    rank = {v: lsum(A(present[u], net.ge(fp[u] - fp[v], 1)) for u in TYPES if u != v) for v in TYPES}   # objects named earlier
    # an absent type has rank = npres (every present object has fp >= 1), so rank + 1 reaches npres + 1 <= len(TYPES) + 1:
    # sel needs M >= that bound, else every absent type leaks relu(rank + 1 - M) into idm (this broke all 7-object puzzles)
    idm = {x: lsum(net.sel(rank[v] + 1, mt[x, v], M=len(TYPES) + 1) for v in TYPES if (x, v) in mt) for x in range(1, K)}  # 1..n, 0 = none
    anyend = {x: lsum(mtg(x, v) for v in TYPES) for x in range(1, K)}
    yl = {(x, l): lsum(mtg(x, v) for v in TYPES if len(OBJ[v]) == l) for x in range(1, K) for l in range(1, 5)}
    # ---- constraint sentences ending at a '.' (offset e) ----
    rows = []                                                        # per e: (fire, before, after, absF, absB, S, Y, k)
    for e in range(1, K - 12):
        dot = A(w.t(e, '.') + w.t(e, '.\n'), inq[e])           # the last sentence before 'Options:' ends in the merged token '.\n'
        dets_b, dets_a, S_parts, Y_parts, kF, kB, aF, aB = [], [], [], [], [], [], [], []
        for p, d in REL:
            for l in range(1, 5):
                s_end = e + l + len(p) + 1
                if s_end + 3 >= K: continue
                f = A(dot, yl.get((e + 1, l), Lin()), *w.seq(e + l + 1, p), anyend.get(s_end, Lin()))
                (dets_b if d > 0 else dets_a).append(f)
                S_parts.append(net.sel(idm[s_end], f, M=NMAX)); Y_parts.append(net.sel(idm[e + 1], f, M=NMAX))
        for p, k, back in ABS:
            s_end = e + len(p) + 1
            if s_end + 3 >= K: continue
            f = A(dot, *w.seq(e + 1, p), anyend.get(s_end, Lin()))
            (aB if back else aF).append(f); (kB if back else kF).append(k * f)
            S_parts.append(net.sel(idm[s_end], f, M=NMAX))
        before, after, absF, absB = lsum(dets_b), lsum(dets_a), lsum(aF), lsum(aB)
        rows.append((e, before + after + absF + absB, before, after, absF, absB, lsum(S_parts), lsum(Y_parts), lsum(kF) + lsum(kB)))
    # rank the constraint sentences (text order = decreasing offset) and gather them into MAXJ slots
    cnt, run = {}, Lin()
    for e, fire, *_ in reversed(rows):
        cnt[e] = run; run = run + fire
    ncons = run
    slots = []; slotv = []
    for j in range(MAXJ):
        g = {e: A(fire, net.eq(cnt[e], j)) for e, fire, *_ in rows}
        pick = lambda idx, M: lsum(net.sel(r[idx], g[r[0]], M=M) for r in rows)
        bj, aj, fj, kbj = (lsum(A(g[r[0]], r[i]) for r in rows) for i in (2, 3, 4, 5))
        Sj, Yj, Kj = pick(6, NMAX), pick(7, NMAX), pick(8, NMAX)
        Sh = [net.eq(Sj, a + 1) for a in range(NMAX)]; Yh = [net.eq(Yj, b + 1) for b in range(NMAX)]; Kh = [net.eq(Kj, k) for k in range(1, NMAX + 1)]
        slots.append((bj, aj, fj, kbj, Sh, Yh, Kh)); slotv += [bj, aj, fj, kbj, Sj, Yj, Kj]
    R = {(a, b): lsum(A(bj, Sh[a], Yh[b]) + A(aj, Yh[a], Sh[b]) for bj, aj, fj, kbj, Sh, Yh, Kh in slots) for a in range(NMAX) for b in range(NMAX) if a != b}
    AF = {(a, k): lsum(A(fj, Sh[a], Kh[k - 1]) for bj, aj, fj, kbj, Sh, Yh, Kh in slots) for a in range(NMAX) for k in range(1, NMAX + 1)}
    AB = {(a, k): lsum(A(kbj, Sh[a], Kh[k - 1]) for bj, aj, fj, kbj, Sh, Yh, Kh in slots) for a in range(NMAX) for k in range(1, NMAX + 1)}
    # ---- every ordering ----
    X = {}                                                           # X[n][p][a]: object a at position p (0-based)
    alive = []
    for n in range(3, NMAX + 1):
        eqn = net.eq(npres, n)
        X[n] = [[[] for _ in range(NMAX)] for _ in range(NMAX)]
        for perm in itertools.permutations(range(n)):              # perm[p] = object at position p
            pos = {a: p for p, a in enumerate(perm)}
            viol = lsum(R[a, b] for a in range(n) for b in range(n) if a != b and pos[a] > pos[b])
            viol = viol + lsum(AF[a, k] for a in range(n) for k in range(1, NMAX + 1) if pos[a] != k - 1)
            viol = viol + lsum(AB[a, k] for a in range(n) for k in range(1, NMAX + 1) if pos[a] != n - k)
            viol = viol + lsum(R[a, b] for a in range(NMAX) for b in range(NMAX) if a != b and (a >= n or b >= n))
            ok = net.relu(eqn - viol); alive.append(ok)
            for p, a in enumerate(perm): X[n][p][a].append(ok)
    nalive = lsum(alive)
    # ---- the asked position: the last option line '... <predicate>' '\n' 'A' ':' ' Let' ----
    head = [w.t(0, ' Let'), w.t(1, ':'), w.t(2, 'A'), w.t(3, '\n')]
    askF = {k: [] for k in range(1, NMAX + 1)}; askB = {k: [] for k in range(1, NMAX + 1)}
    for p, k, back in ABS:
        f = A(*head, *w.seq(4, p), anyend.get(4 + len(p), Lin()))
        (askB if back else askF)[k].append(f)
    askF = {k: lsum(v) for k, v in askF.items()}; askB = {k: lsum(v) for k, v in askB.items()}
    has_ask = lsum(askF.values()) + lsum(askB.values())
    ans = []
    for a in range(NMAX):
        parts = []
        for n in range(3, NMAX + 1):
            for k in range(1, n + 1):
                parts.append(A(askF[k], lsum(X[n][k - 1][a])))
                parts.append(A(askB[k], lsum(X[n][n - k][a])))
        ans.append(lsum(parts))
    valid = A(w.t(0, ' Let'), found_q, net.eq(nalive, 1), has_ask, net.ge(ncons, 1))
    St = [valid] + [A(valid, net.clip01(x)) for x in ans]          # ans[a] counts orderings (can be > 1 when not valid): clip before the AND
    probes = {'npres': [npres], 'ncons': [ncons], 'nalive': [nalive], 'has_ask': [has_ask], 'ans': ans, 'valid': [valid], 'slots': slotv, 'fp': [fp[v] for v in TYPES], 'rank': [rank[v] for v in TYPES], 'present': [present[v] for v in TYPES]}
    return net, St, probes


def build_stack2():
    net = Net(K * NS + N_STATE); w = Win(net); A = net.AND; b = K * NS
    valid = net.inp(b); L = [net.inp(b + 1 + i) for i in range(NMAX)]
    trig = A(w.t(0, ' (') + w.t(0, '('), w.t(1, ' is'), w.t(2, ' answer'))
    return net, [A(trig, valid, L[i]) for i in range(NMAX)], {'trig': [trig]}


def build():
    n1, S, _ = build_stack1(); n2, O, _ = build_stack2()
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


# ---------------- Python reference ----------------
def ld_ref(q):
    import re
    body, opts = q.split('Options:')
    m = re.search(r'(?:there (?:are|were)|sells) \w+ [\w ]+?: (.*?)\.(?:\s|$)', body)
    if not m: return ''
    objs = [re.sub(r'^(a|an|the)\s+', '', x.strip()) for x in re.split(r',\s*(?:and\s+)?|\s+and\s+', m.group(1))]
    n = len(objs); idx = {o.lower(): i for i, o in enumerate(objs)}
    ob = '|'.join(re.escape(o) for o in sorted(objs, key=len, reverse=True))
    relp = {re.sub(r" the$", "", p.strip()): d for p, d in [(' is to the left of the', 1), (' is to the right of the', -1), (' is older than the', 1), (' is newer than the', -1),
                                      (' are more expensive than the', 1), (' are less expensive than the', -1), (' finished above', 1), (' finished below', -1)]}
    ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
    absp = {'is the leftmost': (1, 0), 'is the rightmost': (1, 1), 'is the oldest': (1, 0), 'is the newest': (1, 1), 'are the most expensive': (1, 0),
            'are the cheapest': (1, 1), 'finished first': (1, 0), 'finished last': (1, 1)}
    for k in range(2, 8):
        o = ORD[k - 1]
        absp.update({f'is the {o} from the left': (k, 0), f'is the {o} from the right': (k, 1), f'is the {o}-oldest': (k, 0), f'is the {o}-newest': (k, 1),
                     f'are the {o}-most expensive': (k, 0), f'are the {o}-cheapest': (k, 1), f'finished {o}': (k, 0), f'finished {o}-to-last': (k, 1)})
    cons = []
    for s in re.split(r'(?<=\.)\s+', body[m.end():].strip()):
        s = s.strip().rstrip('.')
        mm = re.fullmatch(rf'(?:The )?({ob}) (.+?) (?:the )?({ob})', s)
        if mm and mm.group(2) in relp: cons.append(('r', idx[mm.group(1).lower()], idx[mm.group(3).lower()], relp[mm.group(2)])); continue
        mm = re.fullmatch(rf'(?:The )?({ob}) (.+)', s)
        if mm and mm.group(2) in absp: cons.append(('a', idx[mm.group(1).lower()], *absp[mm.group(2)])); continue
        if s: return ''
    ol = opts.strip().split('\n')[-1]
    mm = re.fullmatch(rf'\([A-G]\) (?:The )?({ob}) (.+)', ol.strip())
    if not mm or mm.group(2) not in absp: return ''
    k, back = absp[mm.group(2)]; P = (n - k) if back else (k - 1)
    sols = []
    for perm in itertools.permutations(range(n)):
        pos = {a: p for p, a in enumerate(perm)}; good = True
        for c in cons:
            if c[0] == 'r': good &= (pos[c[1]] < pos[c[2]]) if c[3] > 0 else (pos[c[1]] > pos[c[2]])
            else: good &= pos[c[1]] == ((n - c[2]) if c[3] else (c[2] - 1))
        if good: sols.append(perm)
    if len(sols) != 1: return ''
    return '(' + LET[sols[0][P]] + ')'
