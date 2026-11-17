"""Shared parts for 'one-shot' micromachines: stack 1 computes the whole reply at the call end (</call>) as Lmax slots,
each a one-hot over the reply symbols (all zero after the last one); one attention head copies that state forward;
stack 2 (built here) writes the reply one token per step: <box> slot0 slot1 ... </box>."""
from hdl import Net, Lin, lsum


class Win:
    def __init__(self, net, SI, NS, K, base=0):
        self.net, self.SI, self.NS, self.K, self.base = net, SI, NS, K, base

    def s(self, o, sym):
        return self.net.inp(self.base + o * self.NS + self.SI[sym]) if 0 <= o < self.K else Lin()

    def any(self, o, syms): return lsum(self.s(o, x) for x in syms)


def inside_flags(net, w, K, cs='CS'):
    """inside[o] = offset o lies inside the most recent call (after its <call> token, before the current token)."""
    cs_at = {o: w.s(o, cs) for o in range(1, K)}
    inside = {}
    for o in range(1, K - 1):
        no_cs = 1 - net.clip01(lsum(cs_at[u] for u in range(1, o + 1)))
        after = net.clip01(lsum(cs_at[j] for j in range(o + 1, K)))
        inside[o] = net.AND(no_cs, after)
    return inside, cs_at


def build_stack2_seq(SYMS, SI, NS, K, content, Lmax):
    """content: reply symbols (each in SYMS). State = Lmax * len(content) slot flags. -> (net, outputs over content+['OS','OE'])."""
    net = Net(K * NS + Lmax * len(content)); w = Win(net, SI, NS, K)
    slot = lambda t, v: net.inp(K * NS + t * len(content) + v)
    isout = lambda o: w.any(o, content)
    e = {k: net.AND(w.s(k, 'CE'), w.s(k - 1, 'OS'), *[isout(o) for o in range(k - 1)]) for k in range(1, min(Lmax + 2, K))}
    act = lsum(e.values())
    out = {}
    for v, sym in enumerate(content):
        out[sym] = lsum(net.AND(e[k], slot(k - 1, v)) for k in e if k - 1 < Lmax)
    out['OE'] = act - lsum(out.values())
    out['OS'] = w.s(0, 'CE')
    return net, [out[x] for x in content] + [out['OS'], out['OE']]
