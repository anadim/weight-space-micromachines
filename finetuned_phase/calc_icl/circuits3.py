"""Truth-tracker micromachine (MM3): one bit of state, kept or flipped by each call."""
from hdl import Net, Lin, lsum, compile_net

SYMS = list('0123') + ['CS', 'CE', 'OS', 'OE']
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
K = 5
OUT = ['0', '1', 'OS', 'OE']
N_STATE = 2                       # has-state flag, bit


def build_stack1():
    net = Net(K * NS); s = lambda o, x: net.inp(o * NS + SI[x])
    has = net.AND(s(0, 'OE'), s(1, '0') + s(1, '1'), s(2, 'OS'))
    return net, [has, net.AND(s(0, 'OE'), s(1, '1'), s(2, 'OS'))]


def build_stack2():
    net = Net(K * NS + N_STATE); s = lambda o, x: net.inp(o * NS + SI[x]) if o < K else Lin()
    has, bit = net.inp(K * NS), net.inp(K * NS + 1)
    out0 = out1 = Lin(); oe = Lin()
    for k, t in ((1, 0), (2, 1)):                       # call end at offset k, box start at k-1, t digits emitted
        here = net.AND(s(k, 'CE'), s(k - 1, 'OS'), *([s(0, '0') + s(0, '1')] if t else []))
        if t: oe = oe + here; continue
        call = {d: net.AND(here, s(k + 1, d), s(k + 2, 'CS')) for d in '0123'}
        keep1 = net.AND(call['2'], has, bit); keep0 = net.AND(call['2'], has, 1 - bit)
        flip1 = net.AND(call['3'], has, 1 - bit); flip0 = net.AND(call['3'], has, bit)
        out1 = call['1'] + keep1 + flip1; out0 = call['0'] + keep0 + flip0
        oe = oe + here - out0 - out1                    # invalid call: empty box
    return net, [out0, out1, net.AND(s(0, 'CE'), net.clip01(lsum(s(o, 'CS') for o in range(1, K)))), oe]   # v3: own call only


def build():
    n1, S = build_stack1(); n2, O = build_stack2()
    return compile_net(n1, S), compile_net(n2, O)
