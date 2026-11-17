"""Symbol-level test of the calculator circuit (circuits/calc.py, default build) against the exact Python reference.
No model needed (CPU or GPU). An invalid call (e.g. division by zero, malformed) must give silence = an empty output.
python tests/test_calc_circuit.py [cpu|cuda] [N_EXHAUSTIVE=60] [N_RANDOM=3000]"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import random, time, torch
from micromachines.circuits.calc import build, SYMS, SI, NS, K, OUT
from micromachines.hdl import ReLUNet
from micromachines.calcfmt import calc_ref, NI, F

dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
c1, c2 = build()
n1, n2 = ReLUNet(c1).to(dev), ReLUNet(c2).to(dev)
print('stack1 units', n1.n_units, 'layers', n1.L, 'nnz', n1.nnz, '(dense would be', n1.n_dense, ') | stack2 units', n2.n_units, 'layers', n2.L, 'nnz', n2.nnz, '(dense', n2.n_dense, ')')
OTHER = NS
PMAX = 60
OUT_SYM = torch.tensor([SI[t] for t in OUT], device=dev)
CH = {**{c: SI[c] for c in '0123456789+-*/.'}}


def to_syms(e):                        # mimic the tokenizer: "+-", "--", "*-", "/-" are single tokens
    out, i = [], 0
    while i < len(e):
        if e[i] in '+-*/' and i + 1 < len(e) and e[i + 1] == '-' and i > 0:
            out.append(SI['N' + e[i]]); i += 2
        else:
            out.append(CH.get(e[i], OTHER)); i += 1
    return out


def window(seq):                       # seq (B,T) symbol ids, right-aligned -> one-hot window (B, K*NS), offset 0 = last
    last = seq[:, -K:].flip(1)
    oh = torch.nn.functional.one_hot(last, NS + 1)[..., :NS].float()
    return oh.reshape(seq.shape[0], -1)


@torch.no_grad()
def run(exprs, raw=False):
    """exprs: strings placed between <tool_call> and </tool_call>. Returns the emitted strings."""
    B = len(exprs); T = K + PMAX
    seq = torch.full((B, T), OTHER, dtype=torch.long)
    for i, e in enumerate(exprs):
        ids = ([] if raw else [SI['CS']]) + to_syms(e) + [SI['CE']]
        seq[i, T - len(ids):] = torch.tensor(ids)
    seq = seq.to(dev)
    S = n1(window(seq))
    outs = [''] * B; alive = torch.ones(B, dtype=torch.bool, device=dev); bad = 0
    for step in range(PMAX):
        o = n2(torch.cat([window(seq), S], -1))
        if not bool(((o == 0) | (o == 1)).all()): bad += 1
        hot = o.sum(-1)
        silent = alive & (hot == 0) & (step == 0)            # the current circuit is silent on invalid calls (empty output)
        alive = alive & ~silent
        if not bool((hot[alive] == 1).all()): bad += 1
        if not bool((hot[~alive] == 0).all()): bad += 1
        nxt = OUT_SYM[o.argmax(-1)]
        nxt = torch.where(alive, nxt, torch.full_like(nxt, OTHER))
        seq = torch.cat([seq[:, 1:], nxt[:, None]], 1)
        for i in torch.nonzero(alive).flatten().tolist():
            t = SYMS[nxt[i]]
            if t == 'OE': alive[i] = False
            elif t != 'OS': outs[i] += t
        if not alive.any(): break
    return outs, bad + int(alive.any())


def check(exprs, name, bs=4096):
    t = time.time(); nbad = 0; nerr = 0; shown = 0
    for b in range(0, len(exprs), bs):
        chunk = exprs[b:b + bs]
        outs, bad = run(chunk); nbad += bad
        for e, o in zip(chunk, outs):
            if o != calc_ref(e):
                nerr += 1
                if shown < 8: print('   MISMATCH', e, 'net:', repr(o), 'ref:', repr(calc_ref(e))); shown += 1
    print(f'{name}: {len(exprs)} cases, {nerr} wrong, {nbad} non-one-hot steps, {time.time()-t:.1f}s', flush=True)
    return nerr + nbad


def rnd_num(r, maxi=NI, maxf=F, pfrac=0.3):
    s = str(r.randrange(10 ** r.randint(1, maxi)))
    if r.random() < 0.1: s = '0' * r.randint(1, max(1, maxi - len(s))) + s if len(s) < maxi else s
    if r.random() < pfrac: s += '.' + ''.join(r.choice('0123456789') for _ in range(r.randint(1, maxf)))
    return ('-' if r.random() < 0.25 else '') + s


if __name__ == '__main__':
    r = random.Random(0); tot = 0
    tot += check(['48/2', '48+24', '3.75*2', '5-12', '0.5+0.5', '100/3', '7/2', '1/8', '2/3', '0.1/0.0003', '999999999999*999999999999',
                  '100001/100000', '1/0', '0/5', '0-0', '0*0', '12/0.0001', '999999999999.9999+999999999999.9999', '0.0001*0.0001', '1/3000', '10/4', '1000/10',
                  '-5+3', '3*-5', '-1/3', '-0*5', '5--3', '1/-8', '-5-3', '-5--5', '3+-5', '-3*-5', '-7/-2', '0/-5', '-1/3000000', '1/1000000', '-1/1000001',
                  '999999999999999999*999999999999999999', '999999999999999999.999999+0.000001', '-999999999999999999.999999-999999999999999999.999999', '1/0.000001', '999999999999999999/0.000001', '10000000/3'], 'smoke')
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    tot += check([f'{a}{op}{b}' for op in '+-*/' for a in range(N) for b in range(N)], f'exhaustive 0..{N-1}')
    M = min(N, 40)
    tot += check([f'{a}{op}{b}' for op in '+-*/' for a in range(-M, M) for b in range(-M, M)], f'exhaustive signed -{M}..{M-1}')
    for op in '+-*/':
        tot += check([f'{rnd_num(r)}{op}{rnd_num(r)}' for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 3000)], f'random {op}')
    bad = ['--5+3', '5---3', '5+', '-+5', '-', '5-', '1234567890123456789+1', '1.1234567+1', '-.5+1', '5+-', '', '5', '5+', '+5', '5++5', '1.2.3+4', '.5+1', '5.+1', '1+5.', '1+.5', '1234567890123+1', '1.23456+1', 'a+b', '1 + 2', '1+2+3', '-1+2', '1/0.0', '1+2.', '12+*3', '0.0000/0']
    tot += check(bad, 'malformed')
    outs, _ = run(['5+5' + '</x>'], raw=True)
    print('TOTAL FAILURES', tot)
    assert tot == 0
    print('PASS')
