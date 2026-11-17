"""Shared circuit pieces for the Team-A call-format tools (written ReLU logic; compiled by micromachines/hdl.py).

Layout (same as the repo's other call-format machines):
  stack 1  runs at </tool_call> (offset 0 of its window): parses the call and computes the whole reply as LMAX slots,
           each a one-hot over the reply symbols (all zero after the last one; all zero everywhere = empty reply = error)
  (one hard attention head copies that state from the most recent </tool_call>)
  stack 2  writes <tool_response> right after </tool_call>, then slot 0, slot 1, ..., then </tool_response>.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from micromachines.hdl import Net, Lin, lsum, compile_net  # noqa: E402

DIGITS = list('0123456789')


class Win:
    """Symbol window: s(o, sym) = flag of symbol sym at offset o (0 = current token). Classes: s(o, (a, b)) = sum."""

    def __init__(self, net, SI, NS, K):
        self.net, self.SI, self.NS, self.K = net, SI, NS, K

    def s(self, o, sym):
        if isinstance(sym, (tuple, list)): return lsum(self.s(o, x) for x in sym)
        return self.net.inp(o * self.NS + self.SI[sym]) if 0 <= o < self.K else Lin()

    def dig(self, o): return self.s(o, DIGITS)
    def dval(self, o): return lsum(self.s(o, str(v)) * v for v in range(1, 10))


def call_frame(net, w, name):
    """-> csl[c]: the most recent <tool_call> is at offset c (one-hot over c), inside[o]: offset o (>=1) lies inside it,
    found: a <tool_call> is in the window, nameok: it is followed by NAME and a space token."""
    K = w.K
    cs = {c: w.s(c, 'CS') for c in range(1, K)}
    csl ={c: (net.AND(cs[c], 1 - net.clip01(lsum(cs[u] for u in range(1, c)))) if c > 1 else cs[c]) for c in range(1, K)}
    inside = {o: lsum(csl[c] for c in range(o + 1, K)) for o in range(1, K)}
    found = lsum(csl.values())
    nameok = lsum(net.AND(csl[c], w.s(c - 1, name), w.s(c - 2, 'SP')) for c in range(3, K))
    return csl, inside, found, nameok


def build_stack2_seq(SI, NS, K, content, Lmax):
    """State = Lmax * len(content) slot flags + 'mine' (the call names this tool). -> (net, outputs over content + ['OS', 'OE']).
    A tool writes nothing at all for a call that names another tool, so several tools can be attached at once."""
    net = Net(K * NS + Lmax * len(content) + 1); w = Win(net, SI, NS, K)
    slot = lambda t, v: net.inp(K * NS + t * len(content) + v)
    mine = net.inp(K * NS + Lmax * len(content))
    isout = lambda o: w.s(o, content)
    e = {k: net.AND(mine, w.s(k, 'CE'), w.s(k - 1, 'OS'), *[isout(o) for o in range(k - 1)]) for k in range(1, min(Lmax + 2, K))}
    act = lsum(e.values())
    out = {}
    for v, sym in enumerate(content):
        out[sym] = lsum(net.AND(e[k], slot(k - 1, v)) for k in e if k - 1 < Lmax)
    out['OE'] = act - lsum(out.values())
    out['OS'] = net.AND(mine, w.s(0, 'CE'))
    return net, [out[x] for x in content] + [out['OS'], out['OE']]
