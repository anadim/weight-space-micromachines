# fuzz: (1) reader registers vs the scorer's regex on random '#### number' texts; (2) voter circuit vs the reference vote.
import os as _o, sys as _s, random as _r, re as _re, json as _j
_s.path.insert(0, _o.environ['STK_DIR']); _o.environ.setdefault('SAN_DIR', _o.environ['STK_DIR'])
import comp_serial as CSR
eng.__class__ = CSR.SerialWD
eng.c2_setup(); eng.set_c2('off')
eng.sw_setup(N=8, acap=a.max_new, probe=f"{_o.environ['STK_DIR']}/verifier_probe_L18.pt", cache=f"{_o.environ['STK_DIR']}/dfa_tables.pt")
rng = _r.Random(0); T_ = lambda s: tok(s, add_special_tokens=False)['input_ids']
NUM = r'####\s*(-?[\d,]*\.?\d+)'
def rnum():
    k = rng.random()
    iv = str(rng.randint(0, 10 ** rng.randint(1, 9)))
    if k < 0.2: iv = f"{int(iv):,}"
    s = ('-' if rng.random() < 0.15 else '') + iv
    if rng.random() < 0.25: s += '.' + str(rng.randint(0, 10 ** rng.randint(1, 4))).rjust(rng.randint(1, 4), '0')
    return s
bad = wrong_flag = missed = 0; N1 = 20000
for i in range(N1):
    pre = rng.choice(['So the total is 5 + 3 = 8.\n', 'The answer is clear.\n', '', 'Total: $12.\n'])
    tail = rng.choice(['\n', '\n\n', ' dollars\n', '.\n', '<|endoftext|>', '\nQuestion'])
    txt = pre + '#### ' + rng.choice(['', '$']) + rnum() + tail
    ids = T_(txt)
    closed, syms, fl = eng.sw_read_sim(ids)
    m = _re.search(NUM, txt)
    ref = m.group(1).replace(',', '') if m else None
    got = CSR.SerialW._val(syms) if closed else None
    if (ref is None) != (got is None): missed += 1; continue
    if ref is not None and abs(float(ref) - got) > 1e-9 * max(1, abs(float(ref))):
        bad += 1
        if bad < 5: print('MISMATCH', repr(txt), ref, got, flush=True)
    import san_engine as SE
    if ref is not None and SE.py_flag(txt) != fl: wrong_flag += 1
print('FUZZ reader', _j.dumps(dict(n=N1, wrong_value=bad, missed=missed, wrong_flag=wrong_flag)), flush=True)
# voter
import torch as _t
B = 512; trials = 40; dis = 0; tot = 0
for tr in range(trials):
    vals = [[rng.choice([None, '12', '7', '7.5', '-3', '100', '12']) for _ in range(8)] for _ in range(B)]
    sym = _t.full((B, 8, 16), CSR.PAD, dtype=_t.long); ok = _t.zeros(B, 8, dtype=_t.bool); bd = _t.zeros(B, 8, dtype=_t.bool)
    for j in range(B):
        for i, v in enumerate(vals[j]):
            if v is None: continue
            ok[j, i] = True; bd[j, i] = v.startswith('-') or '.' in v
            for k, ch in enumerate(v): sym[j, i, k] = '0123456789-.'.index(ch)
    lg = _t.randn(B, 8) * 2
    if tr % 2: lg = _t.round(lg)                      # exact ties in the probe weights
    for N in (4, 8):
        for vm in (0, 1):
            eng.set_sw(N=N, vmode=vm, san=True)
            has, widx = eng._vote(ok.cuda(), bd.cuda(), lg.cuda(), sym.cuda())
            ref = eng.sw_vote(sym, ok, bd, lg, N, vmode=vm, san=True)
            for j in range(B):
                got = int(widx[j]) if bool(has[j]) else None
                tot += 1
                if got != ref[j]:
                    rv = lambda x: None if x is None else vals[j][x]
                    if rv(got) != rv(ref[j]): dis += 1          # a different ANSWER (not only a different slot of the same answer)
print('FUZZ voter', _j.dumps(dict(decisions=tot, different_answer=dis)), flush=True)
print('STKDONE', flush=True)
