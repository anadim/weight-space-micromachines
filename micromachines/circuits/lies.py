"""Fork A natural-format BBH machines (zero training, hand-written ReLU circuits).

liesnat (web_of_lies, official CoT):  '... that Y <tells the truth|lies>, if X says Y <tells the truth|lies>, then X'
    -> writes ' tells' when the two values agree, else ' lies' (X's name after 'then' has as many tokens as after 'if');
    '... that X <value>. So the answer is' -> ' Yes' / ' No'.
boolnat (boolean_expressions): MM3's exact evaluator (circuits4 stack 1) re-keyed so that an expression between two '='
    signs (the model's own 'A = not not True =') is evaluated; the machine writes ' True' / ' False' right after the '='.
"""
from ..hdl import Net, Lin, lsum, compile_net
from . import boolean as B4

# ---------------- web of lies ----------------
LS = ['THAT', 'IF', 'SAYS', 'THEN', 'TELLS', 'THE', 'TRUTH', 'LIES', 'COMMA', 'DOT', 'SO', 'ANSWER', 'IS', 'SINCE', 'WE', 'KNOW', 'FROM',
      'OPAR', 'RP', 'LP', 'NL']
LSI = {s: i for i, s in enumerate(LS)}; LNS = len(LS)
LK = 40; NMAX = 4
LOUT = ['W_T', 'W_L', 'W_Y', 'W_N']
LTEXT = {'THAT': ' that', 'IF': ' if', 'SAYS': ' says', 'THEN': ' then', 'TELLS': ' tells', 'THE': ' the', 'TRUTH': ' truth',
         'LIES': ' lies', 'COMMA': ',', 'DOT': '.', 'SO': ' So', 'ANSWER': ' answer', 'IS': ' is', 'SINCE': ' Since', 'WE': ' we', 'KNOW': ' know',
         'FROM': ' from', 'OPAR': ' (', 'RP': ')', 'LP': '(', 'NL': '.\n',
         'W_T': ' tells', 'W_L': ' lies', 'W_Y': ' Yes', 'W_N': ' No'}


def _lies_stack1():
    net = Net(LK * LNS)
    return net, [net.relu(net.inp(LSI['THEN']))]          # no call state is needed: everything is read from the window


def _lies_stack2():
    net = Net(LK * LNS + 1); A = net.AND
    s = lambda o, x: net.inp(o * LNS + LSI[x]) if 0 <= o < LK else None
    other = lambda o: 1 - lsum(net.inp(o * LNS + i) for i in range(LNS))
    VALS = ((0, ['LIES']), (1, ['TRUTH', 'THE', 'TELLS']))          # reversed token order (offsets grow to the left)
    agree, differ = [], []
    for n in range(1, NMAX + 1):
        for said, sv in VALS:
            for m in range(1, NMAX + 1):
                for known, kv in VALS:
                    for m2 in range(1, NMAX + 1):
                        parts = [other(o) for o in range(n)] + [s(n, 'THEN'), s(n + 1, 'COMMA')]
                        p = n + 2; parts += [s(p + j, x) for j, x in enumerate(sv)]; p += len(sv)
                        parts += [other(p + j) for j in range(m)]; p += m
                        parts += [s(p, 'SAYS')]; p += 1
                        parts += [other(p + j) for j in range(n)]; p += n
                        parts += [s(p, 'IF'), s(p + 1, 'COMMA')]; p += 2
                        parts += [s(p + j, x) for j, x in enumerate(kv)]; p += len(kv)
                        parts += [other(p + j) for j in range(m2)]; p += m2
                        parts += [s(p, 'THAT')]
                        if p >= LK: continue
                        (agree if said == known else differ).append(A(*parts))
    # v2: '... then|that Y <value>.\n(k) X says Y <said>. Since we know from (j) that Y' -> copy Y's value from the line before
    for n in range(1, NMAX + 1):
        for said, sv in VALS:
            for m in range(1, NMAX + 1):
                for pv, pvv in VALS:
                    parts = [other(o) for o in range(n)] + [s(n, 'THAT'), s(n + 1, 'RP'), other(n + 2), s(n + 3, 'OPAR'), s(n + 4, 'FROM'),
                             s(n + 5, 'KNOW'), s(n + 6, 'WE'), s(n + 7, 'SINCE'), s(n + 8, 'DOT')]
                    p = n + 9; parts += [s(p + j, x) for j, x in enumerate(sv)]; p += len(sv)
                    parts += [other(p + j) for j in range(n)]; p += n
                    parts += [s(p, 'SAYS')]; p += 1
                    parts += [other(p + j) for j in range(m)]; p += m
                    parts += [s(p, 'RP'), other(p + 1), s(p + 2, 'LP'), s(p + 3, 'NL')]; p += 4
                    parts += [s(p + j, x) for j, x in enumerate(pvv)]; p += len(pvv)
                    parts += [other(p + j) for j in range(n)]; p += n
                    if p >= LK: continue
                    parts += [s(p, 'THEN') + s(p, 'THAT')]
                    (agree if pv else differ).append(A(*parts))
    # '(1) X <value>. So, we know that X' -> copy the value
    for n in range(1, NMAX + 1):
        for v, vv in VALS:
            parts = [other(o) for o in range(n)] + [s(n, 'THAT'), s(n + 1, 'KNOW'), s(n + 2, 'WE'), s(n + 3, 'COMMA'), s(n + 4, 'SO'), s(n + 5, 'DOT')]
            p = n + 6; parts += [s(p + j, x) for j, x in enumerate(vv)]; p += len(vv)
            parts += [other(p + j) for j in range(n)]; p += n
            parts += [s(p, 'RP'), other(p + 1), s(p + 2, 'OPAR') + s(p + 2, 'LP')]
            (agree if v else differ).append(A(*parts))
    yes, no = [], []
    for v, vv in VALS:                                              # '... that X <value>. So the answer is'
        for m in range(1, NMAX + 1):
            parts = [s(0, 'IS'), s(1, 'ANSWER'), s(2, 'THE'), s(3, 'SO'), s(4, 'DOT')] + [s(5 + j, x) for j, x in enumerate(vv)]
            p = 5 + len(vv); parts += [other(p + j) for j in range(m)] + [s(p + m, 'THAT')]
            (yes if v else no).append(A(*parts))
    out = {'W_T': net.clip01(lsum(agree)), 'W_L': net.clip01(lsum(differ)), 'W_Y': net.clip01(lsum(yes)), 'W_N': net.clip01(lsum(no))}
    return net, [out[t] for t in LOUT]


def build_lies():
    n1, S = _lies_stack1(); n2, O = _lies_stack2()
    return compile_net(n1, S), compile_net(n2, O)


def lies_ref(known, said): return ' tells' if known == said else ' lies'


# ---------------- boolean expressions ----------------
BOUT = ['VT', 'VF', 'VD']


def _bool_stack2():
    net = Net(B4.K * B4.NS + B4.N_STATE)
    valid, value = net.inp(B4.K * B4.NS), net.inp(B4.K * B4.NS + 1)
    ce = net.inp(0 * B4.NS + B4.SI['CE'])
    s = lambda o, x: net.inp(o * B4.NS + B4.SI[x])
    dot = net.AND(s(1, 'CE'), s(0, 'T1') + s(0, 'F1'), valid)               # right after its value: '.' ends the expression
    return net, [net.AND(ce, value), net.AND(ce, valid, 1 - value), dot]     # value already includes valid


def build_bool():
    n1, S = B4.build_stack1(); n2, O = _bool_stack2()
    return compile_net(n1, S), compile_net(n2, O)


BTEXT = {'T0': 'True', 'T1': ' True', 'F0': 'False', 'F1': ' False', 'N0': 'not', 'N1': ' not', 'A0': 'and', 'A1': ' and',
         'R0': 'or', 'R1': ' or', 'B0': '(', 'B1': ' (', 'C0': ')', 'C1': ' )', 'CS': ('=', ' ='), 'CE': ('=', ' ='),
         'OS': ' cat', 'OE': ' dog', 'VT': ' True', 'VF': ' False', 'VD': '.'}
