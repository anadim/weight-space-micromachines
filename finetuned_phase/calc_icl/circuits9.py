"""Sorting micromachine (MM9) for BBH word_sorting: <sort>w1 w2 ... wn</sort> -> box: ' X ...' the words' letters
(A = first word, B = second, ...) in alphabetical order.

Unlike the other machines, this one reads a fixed per-token table (an embedding-style lookup over the same vocabulary):
for every token, whether it starts with a space, its number of characters, and the codes of its first LC characters.
Words are compared on their first LC characters (ties keep the original order)."""
from hdl import Net, Lin, lsum, compile_net
from seqmachine import build_stack2_seq

LC = 10                     # characters compared
NW = 20                     # words at most
TPW = 6                     # tokens per word at most
K = 64
LET = [chr(65 + i) for i in range(NW)]
CONTENT = [' ' + l for l in LET]
FEAT = ['lead', 'nch'] + [f'c{k}' for k in range(LC)] + ['CS', 'CE', 'OS', 'OE'] + CONTENT
NS = len(FEAT); SI = {s: i for i, s in enumerate(FEAT)}
SYMS = FEAT                 # for the shared stack 2: content letters and CS/CE/OS/OE are 0/1 features
OUT = CONTENT + ['OS', 'OE']
LMAX = NW
N_STATE = LMAX * len(CONTENT)


def sort_ref(words):
    """words: list of strings -> list of letters in sorted order (stable), or None."""
    if not 1 <= len(words) <= NW: return None
    key = lambda w: [min(ord(ch), 255) for ch in w[:LC]]              # characters past code 255 compare as equal (ASCII is exact)
    order = sorted(range(len(words)), key=lambda i: (key(words[i]), i))
    return [LET[i] for i in order]


class F:
    def __init__(self, net, base=0):
        self.net, self.base = net, base

    def s(self, o, f):
        return self.net.inp(self.base + o * NS + SI[f]) if 0 <= o < K else Lin()


def build_stack1():
    net = Net(K * NS); w = F(net)
    cs = {o: w.s(o, 'CS') for o in range(1, K)}
    O = list(range(1, K - 1))
    inside = {}
    for o in O:
        no_cs = 1 - net.clip01(lsum(cs[u] for u in range(1, o + 1)))
        inside[o] = net.AND(no_cs, net.clip01(lsum(cs[j] for j in range(o + 1, K))))
    start = {o: net.AND(inside[o], net.clip01(w.s(o, 'lead') + w.s(o + 1, 'CS'))) for o in O}
    nw = lsum(start.values())
    ordn = {o: lsum(start[u] for u in O if u > o) for o in O}                    # 0-based word index of a start at o
    # characters of the word starting at s: tokens o = s, s-1, ... until the next start
    key = {}
    for s in O:
        for c in range(LC):
            parts = []
            for d in range(TPW):
                o = s - d
                if o < 1: break
                same = net.AND(start[s], *[1 - start[u] for u in range(o, s)], inside[o]) if d else start[s]
                off = lsum(w.s(u, 'nch') for u in range(o + 1, s + 1))         # characters before token o in this word
                for k in range(c + 1):
                    if c - k >= LC: continue
                    parts.append(net.sel(w.s(o, f'c{c - k}'), net.AND(same, net.eq(off, k)), M=255))
            key[s, c] = lsum(parts)
    W = {}
    for i in range(NW):
        for c in range(LC):
            W[i, c] = lsum(net.sel(key[s, c], net.AND(start[s], net.eq(ordn[s], i)), M=255) for s in O)
    exists = {i: net.ge(nw, i + 1) for i in range(NW)}
    less = {}
    for i in range(NW):
        for j in range(NW):
            if i == j: continue
            # key_j < key_i, or equal and j < i  (lexicographic, first differing character decides)
            sg = [net.ge(W[i, c] - W[j, c], 1) - net.ge(W[j, c] - W[i, c], 1) for c in range(LC)]
            score = lsum(sgn * (1 << (LC - 1 - c)) for c, sgn in enumerate(sg))
            less[j, i] = net.ge(score, 1) + (net.eq(score, 0) if j < i else Lin())
    rank = {i: lsum(net.AND(exists[j], less[j, i]) for j in range(NW) if j != i) for i in range(NW)}
    bad = lsum(net.AND(inside[o], 1 - w.s(o, 'lead') - w.s(o + 1, 'CS'), 1 - net.ge(w.s(o, 'nch'), 1)) for o in O)   # empty tokens inside
    ok = net.AND(w.s(0, 'CE'), net.ge(nw, 1), 1 - net.ge(nw, NW + 1), 1 - net.clip01(bad))
    slots = []
    for t in range(LMAX):
        for i in range(NW):
            slots.append(net.AND(ok, exists[i], net.eq(rank[i], t)))
    return net, slots


def build():
    n1, S = build_stack1(); n2, O = build_stack2_seq(SYMS, SI, NS, K, CONTENT, LMAX)
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


def token_table(tok, V, specials):
    """(V, NS) feature table from the tokenizer. specials: dict CS/CE/OS/OE -> token id."""
    import torch
    T = torch.zeros(V, NS)
    for i in range(min(V, len(tok))):
        s = tok.decode([i])
        if i in specials.values(): continue
        lead = s.startswith(' ')
        body = s[1:] if lead else s
        T[i, SI['lead']] = float(lead); T[i, SI['nch']] = float(min(len(body), 255))
        for k, ch in enumerate(body[:LC]): T[i, SI[f'c{k}']] = float(min(ord(ch), 255))
        if s in CONTENT: T[i, SI[s]] = 1.0
    for name, i in specials.items(): T[i, :] = 0; T[i, SI[name]] = 1.0
    return T


if __name__ == '__main__':
    import time
    t = time.time(); c1, c2 = build()
    for name, c in (('stack1', c1), ('stack2', c2)):
        print(name, 'units', c['n_units'], 'layers', len(c['layers']), 'nonzeros', sum(int((W != 0).sum()) for _, W, _ in c['layers']) + int((c['out'][1] != 0).sum()))
    print(f'{time.time()-t:.1f}s')
