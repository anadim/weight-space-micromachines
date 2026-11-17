"""Bracket micromachine (MM6) for Dyck languages: <dyck>SEQ</dyck> -> box: the closing brackets that complete SEQ,
innermost first. SEQ is a run of ( [ { < ) ] } > tokens (with or without a leading space).
An open bracket is still open iff no later stretch of the sequence closes more brackets than it opens.
Kinds are not checked: the input is assumed to be well nested (as in BBH)."""
from hdl import Net, Lin, lsum, compile_net
from seqmachine import Win, inside_flags, build_stack2_seq

OPEN, CLOSE = '([{<', ')]}>'
SYMS = [c + v for c in OPEN + CLOSE for v in '01'] + ['CS', 'CE', 'OS', 'OE']      # v=1: with a leading space
NS = len(SYMS); SI = {s: i for i, s in enumerate(SYMS)}
LMAX = 12
K = 112                                       # call content up to ~108 brackets
CONTENT = [c + '1' for c in CLOSE]
OUT = CONTENT + ['OS', 'OE']
N_STATE = LMAX * len(CONTENT)


def dyck_ref(seq):
    """seq: string of bracket characters (spaces ignored). -> the closers for the brackets still open, innermost first.
    A closer closes the most recent open bracket whatever its kind (the input is assumed well-nested); a closer with
    nothing to close is an error."""
    st = []
    for ch in seq.replace(' ', ''):
        if ch in OPEN: st.append(ch)
        elif ch in CLOSE:
            if not st: return None
            st.pop()
        else: return None
    if len(st) > LMAX or len(seq.replace(' ', '')) > K - 4: return None
    return ''.join(CLOSE[OPEN.index(c)] for c in reversed(st))


def build_stack1():
    net = Net(K * NS); w = Win(net, SI, NS, K)
    inside, cs_at = inside_flags(net, w, K)
    O = [o for o in range(1, K - 1)]
    op = {o: w.any(o, [c + v for c in OPEN for v in '01']) for o in O}
    cl = {o: w.any(o, [c + v for c in CLOSE for v in '01']) for o in O}
    typ = {(o, i): w.any(o, [OPEN[i] + v for v in '01']) for o in O for i in range(4)}
    ctyp = {(o, i): w.any(o, [CLOSE[i] + v for v in '01']) for o in O for i in range(4)}
    opI = {o: net.AND(op[o], inside[o]) for o in O}
    clI = {o: net.AND(cl[o], inside[o]) for o in O}
    # every token inside the call must be a bracket
    bad = [net.AND(inside[o], 1 - op[o] - cl[o]) for o in O]
    # the open bracket at o is still open iff no stretch (o-1 .. o') closes more than it opens
    unm = {}
    for o in O:
        ge = [net.ge(lsum(clI[u] - opI[u] for u in range(o2, o)), 1) for o2 in range(1, o)]
        unm[o] = net.relu(opI[o] - lsum(ge))
    # a closing bracket with nothing to close (seen from the start of the call), or of the wrong kind
    for o in O:
        exc = net.ge(lsum(clI[u] - opI[u] for u in range(o, K - 1)), 1)
        bad.append(net.AND(clI[o], exc))
    count = lsum(unm.values())
    ok = net.AND(w.s(0, 'CE'), net.clip01(lsum(cs_at.values())), 1 - net.clip01(lsum(bad)), 1 - net.ge(count, LMAX + 1))
    rank = {o: lsum(unm[u] for u in range(1, o)) for o in O}
    slots = []
    for t in range(LMAX):
        for i in range(4):
            slots.append(lsum(net.AND(ok, unm[o], net.eq(rank[o], t), typ[o, i]) for o in O))
    return net, slots


def build():
    n1, S = build_stack1(); n2, O = build_stack2_seq(SYMS, SI, NS, K, CONTENT, LMAX)
    assert len(S) == N_STATE
    return compile_net(n1, S), compile_net(n2, O)


if __name__ == '__main__':
    import time
    t = time.time(); c1, c2 = build()
    for name, c in (('stack1', c1), ('stack2', c2)):
        print(name, 'units', c['n_units'], 'layers', len(c['layers']), 'nonzeros', sum(int((W != 0).sum()) for _, W, _ in c['layers']) + int((c['out'][1] != 0).sum()))
    print(f'{time.time()-t:.1f}s')
