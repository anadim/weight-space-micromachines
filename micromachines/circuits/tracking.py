"""Natural-text swap tracker (fork B): BBH tracking_shuffled_objects, read straight from the question, no tags.

stack 1 runs at the 'Options' token. It looks back over the current question (back to its 'Q'), finds every
'NAME and NAME' pair (the swaps, in text order) and the last name mentioned (the person asked about), and applies
the swaps to the identity permutation. In BBH the options list the items in the order of the initial holders
(Alice, Bob, Claire, ...), so the answer letter = initial-holder index of the item the asked person ends with.
It is valid only if #swaps == #people named in the question (BBH structure) and an asked person exists.
stack 2 writes that letter when the model has written ' answer is (' (the copy comes by the attention hop).
Names: ' Alice' ' Bob' ' Claire' ' Dave' ' Eve' ' Fred' single tokens; Gertrude = ' Ger' 'tr' 'ude'.
"""
from ..hdl import Net, Lin, lsum, compile_net

PEOPLE = ['Alice', 'Bob', 'Claire', 'Dave', 'Eve', 'Fred', 'Gertrude']
NP = 7
SYMS = [f'N{i}' for i in range(6)] + ['GER', 'TR', 'UDE', 'AND', 'Q', 'OPT', 'LP', 'IS', 'ANS']
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 400                       # window for the question (7-object questions are < 300 tokens)
LET = [chr(65 + i) for i in range(NP)]
OUT = LET
N_STATE = 1 + NP
TEXT = {**{f'N{i}': ' ' + PEOPLE[i] for i in range(6)}, 'GER': ' Ger', 'TR': 'tr', 'UDE': 'ude', 'AND': ' and', 'Q': 'Q',
        'OPT': 'Options', 'LP': (' (', '('), 'IS': ' is', 'ANS': ' answer', **{c: c for c in LET}}


class Win:
    def __init__(self, net): self.net = net
    def s(self, o, sym): return self.net.inp(o * NS + SI[sym]) if 0 <= o < K else Lin()


def build_stack1():
    net = Net(K * NS); w = Win(net); A = net.AND
    # inside the current question: no 'Q' at offsets 0..o
    qseen = {}
    acc = Lin()
    for o in range(K):
        acc = acc + w.s(o, 'Q'); qseen[o] = acc
    inq = {o: 1 - net.clip01(qseen[o]) for o in range(K)}
    found_q = net.clip01(qseen[K - 1])
    # a name token ENDING at offset o (Gertrude ends at 'ude'), a name token STARTING at offset o (Gertrude starts at ' Ger')
    gend = {o: A(w.s(o + 2, 'GER'), w.s(o + 1, 'TR'), w.s(o, 'UDE')) for o in range(K)}
    nend = {o: [w.s(o, f'N{i}') for i in range(6)] + [gend[o]] for o in range(K)}
    nbeg = {o: [w.s(o, f'N{i}') for i in range(6)] + [w.s(o, 'GER')] for o in range(K)}
    # swap pair: name ends at o+2, ' and' at o+1, name starts at o (text order: left name, and, right name)
    isP = {o: A(lsum(nend[o + 2]), w.s(o + 1, 'AND'), lsum(nbeg[o]), inq[o + 2]) for o in range(1, K - 2)}
    cnt, run = {}, Lin()
    for o in range(K - 3, 0, -1):                      # pairs earlier in the text have larger offsets
        cnt[o] = run; run = run + isP[o]
    nsw = run
    rk = {(o, r): A(isP[o], net.eq(cnt[o], r)) for o in isP for r in range(NP)}
    lv = {o: lsum((p + 1) * nend[o + 2][p] for p in range(NP)) for o in isP}          # left name index + 1
    rv = {o: lsum((p + 1) * nbeg[o][p] for p in range(NP)) for o in isP}              # right name index + 1
    Ai = {r: lsum(net.sel(lv[o], rk[o, r], M=NP) for o in isP) for r in range(NP)}
    Bi = {r: lsum(net.sel(rv[o], rk[o, r], M=NP) for o in isP) for r in range(NP)}
    Aw = {(r, p): net.eq(Ai[r], p + 1) for r in range(NP) for p in range(NP)}
    Bw = {(r, p): net.eq(Bi[r], p + 1) for r in range(NP) for p in range(NP)}
    # people named in the question, and the asked person = the most recent name before 'Options'
    present = [net.clip01(lsum(A(nend[o][p], inq[o]) for o in range(1, K))) for p in range(NP)]
    npeople = lsum(present)
    anyname = {o: lsum(nend[o]) for o in range(K)}
    seen, run = {}, Lin()
    for o in range(1, K):
        seen[o] = run; run = run + anyname[o]
    last = {o: A(anyname[o], 1 - net.clip01(seen[o]), inq[o]) for o in range(1, K)}        # the most recent name
    av = lsum(net.sel(lsum((p + 1) * nend[o][p] for p in range(NP)), last[o], M=NP) for o in range(1, K))
    asked = [net.eq(av, p + 1) for p in range(NP)]
    has_ask = lsum(asked)
    # apply the swaps: S[p][i] = person p holds the item first held by person i
    S = [[Lin(c=1) if p == i else Lin() for i in range(NP)] for p in range(NP)]
    for r in range(NP):
        T = []
        for p in range(NP):
            row = []
            for i in range(NP):
                keep = net.relu(S[p][i] - Aw[r, p] - Bw[r, p])
                take = lsum(A(Aw[r, p], Bw[r, q], S[q][i]) + A(Bw[r, p], Aw[r, q], S[q][i]) for q in range(NP) if q != p)
                row.append(keep + take)
            T.append(row)
        S = T
    ans = [lsum(A(asked[p], S[p][i]) for p in range(NP)) for i in range(NP)]
    valid = A(w.s(0, 'OPT'), found_q, net.eq(nsw - npeople, 0), net.ge(npeople, 2), has_ask)
    St = [valid] + [A(valid, a) for a in ans]
    probes = {'nswaps': [nsw], 'npeople': [npeople], 'asked': asked, 'ans': ans, 'valid': [valid]}
    return net, St, probes


def build_stack2():
    net = Net(K * NS + N_STATE); w = Win(net); A = net.AND; b = K * NS
    valid = net.inp(b); L = [net.inp(b + 1 + i) for i in range(NP)]
    trig = A(w.s(0, 'LP'), w.s(1, 'IS'), w.s(2, 'ANS'))
    out = [A(trig, valid, L[i]) for i in range(NP)]
    return net, out, {'trig': [trig]}


def build():
    n1, S, _ = build_stack1(); n2, O, _ = build_stack2()
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


def tracking_ref(q):
    """Python reference on a BBH question: '(X)' or ''."""
    import re
    body = q.split('Options:')[0]
    nm = '|'.join(PEOPLE)
    sw = re.findall(rf'({nm}) and ({nm})', body); asked = re.findall(rf'({nm})', body)
    names = {x for x in PEOPLE if x in body}
    if len(sw) != len(names) or not asked: return ''
    perm = list(range(NP))
    for a, b in sw:
        ia, ib = PEOPLE.index(a), PEOPLE.index(b); perm[ia], perm[ib] = perm[ib], perm[ia]
    return '(' + LET[perm[PEOPLE.index(asked[-1])]] + ')'
