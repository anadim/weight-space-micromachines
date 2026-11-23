"""Exhaustive 0..199 sweep of the calculator circuit (micromachines/circuits/calc.py, default build).
Same harness and reference as repo/tests/test_calc_circuit.py (it is imported, so run()/calc_ref/tokenisation are
identical to `python tests/test_calc_circuit.py cpu 200`), but split per operation and with the full failure list.
Cases: f'{a}{op}{b}' for op in + - * / and a, b in 0..N-1 (N=200) -> 4 x 40,000 = 160,000.
Pass = net output string == calc_ref(expr) (x/0 must give silence = '', non-terminating quotients follow calc_ref's
truncation convention) and every step one-hot.
usage: python calc_sweep_0_199.py REPO_DIR [cpu|cuda] [N=200] [BS=4096]"""
import hashlib, os, platform, sys, time, datetime

repo = os.path.abspath(sys.argv[1]); dev = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200; BS = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
t0 = time.time()


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


print('date (UTC):', datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
print('command:', 'python', ' '.join(sys.argv))
print('host:', platform.node(), '| python', platform.python_version())
for f in [os.path.abspath(__file__), *(os.path.join(repo, p) for p in
          ['tests/test_calc_circuit.py', 'micromachines/circuits/calc.py', 'micromachines/hdl.py', 'micromachines/calcfmt.py'])]:
    print(f'sha256 {sha(f)}  {f}')

sys.path.insert(0, repo); sys.path.insert(0, os.path.join(repo, 'tests'))
sys.argv = [sys.argv[0], dev]                       # test module reads the device from argv at import
import torch
import test_calc_circuit as T                       # builds the circuit (stack1, stack2) on import
print('torch', torch.__version__, '| threads', torch.get_num_threads(), '| device', dev)

h = hashlib.sha256()                                # fingerprint of the compiled weights actually evaluated
for tag, net in (('stack1', T.n1), ('stack2', T.n2)):
    for k, v in sorted(net.state_dict().items()):
        v = v.detach().cpu().contiguous()
        h.update(f'{tag}.{k}:{v.dtype}:{tuple(v.shape)}'.encode()); h.update(v.numpy().tobytes())
print('sha256 compiled weights (stack1+stack2 buffers):', h.hexdigest())
print(f'range: a, b in 0..{N-1}, ops + - * /, {4*N*N} cases; batch {BS}', flush=True)

fails = []; rows = []
for op in '+-*/':
    t = time.time(); exprs = [f'{a}{op}{b}' for a in range(N) for b in range(N)]; wrong = 0; nonone = 0
    for i in range(0, len(exprs), BS):
        chunk = exprs[i:i + BS]
        outs, bad = T.run(chunk); nonone += bad
        for e, o in zip(chunk, outs):
            ref = T.calc_ref(e)
            if o != ref: wrong += 1; fails.append((e, o, ref))
    rows.append((op, len(exprs), len(exprs) - wrong, wrong, nonone))
    print(f'op {op}: {len(exprs)} cases, {len(exprs)-wrong} correct, {wrong} wrong, {nonone} non-one-hot/unterminated batch-steps, {time.time()-t:.1f}s', flush=True)

print('\nsummary\n op | cases | correct | wrong | non-one-hot')
for r in rows: print(' %s | %d | %d | %d | %d' % r)
tc, tok, tw, tn = (sum(r[i] for r in rows) for i in (1, 2, 3, 4))
print(f' all | {tc} | {tok} | {tw} | {tn}')
print(f'\nfailures: {len(fails)}' + (' (first 200 listed)' if len(fails) > 200 else ''))
for e, o, ref in fails[:200]: print(f'   MISMATCH {e} net: {o!r} ref: {ref!r}')
ex = ['199*199', '198/7', '5/0', '0/0', '0-199', '199+199', '1/3']
print('examples net (ref):', ', '.join(f'{e} -> {o!r} ({T.calc_ref(e)!r})' for e, o in zip(ex, T.run(ex)[0])))
print(f'total runtime {time.time()-t0:.1f}s')
print('RESULT', 'PASS' if tw == 0 and tn == 0 else 'FAIL')
