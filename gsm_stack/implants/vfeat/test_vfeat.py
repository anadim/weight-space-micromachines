"""CPU tests for the vfeat implant (job/stk/numreg.py + serial_stack.py SER_VF=1).

python3.11 implants/vfeat/test_vfeat.py [--runs]
 1. numreg fuzz: random class/digit token streams -> batched torch automaton vs the offline regex reference (distinct sets).
 2. numreg on real attempts: body tokens of every dev2 attempt (Qwen tokenizer classes) -> nnum / qcov_d vs implants/vfeat/cache.
 3. voter fuzz: random registers -> compiled _act cluster voter vs the offline ClusterLin pick (voter/learn.py) + '#### X' tokens.
 4. --runs: in-stream registers (job/out/*_vf.jsonl.gz) vs offline features of the same attempts; same-stream pick check.
"""
import os, sys, types, random, argparse, gzip, json, pickle, math
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants')
ap = argparse.ArgumentParser(); ap.add_argument('--runs', action='store_true'); ap.add_argument('--trials', type=int, default=4096)
A = ap.parse_args()
sys.path.insert(0, os.path.join(HERE, 'job', 'stk')); sys.path.insert(0, f'{H}/voter')
from numreg import NumReg, ref_numbers, token_tables
FAIL = []


def t_fuzz():
    rng = random.Random(0); B = 512; T = 40; bad = 0; n = 0
    for trial in range(16):
        C = torch.tensor([[rng.choice([3, 3, 3, 3, 4, 5, 0, 0]) for _ in range(T)] for _ in range(B)])
        D = torch.tensor([[rng.randrange(10) for _ in range(T)] for _ in range(B)], dtype=torch.float64) * (C == 3)
        L = torch.tensor([rng.randrange(T + 1) for _ in range(B)])
        nr = NumReg(B, 64, 'cpu'); nr.run(torch.arange(B), C, D, L)
        for b in range(B):
            ref = {float(x) for x in ref_numbers(C[b, :L[b]].tolist(), [int(v) for v in D[b, :L[b]].tolist()])}
            got = nr.S[b, :nr.n[b]].tolist()
            n += 1
            if ref != set(got) or len(got) != len(ref):
                bad += 1
                if bad < 4: print('  mismatch', C[b, :L[b]].tolist(), D[b, :L[b]].tolist(), sorted(ref), sorted(got))
    print(f'[1] numreg fuzz: {n} random streams, {bad} mismatches', 'PASS' if bad == 0 else 'FAIL'); bad and FAIL.append(1)


def body_ids(a, NH):
    """host-read part of the final segment: tokens before the token completing '####' (incl. that token, class other)"""
    ho = a['hash_off']
    return a['seg_tokens'][:ho + 1] if ho is not None else a['seg_tokens']


def t_real(S='dev2'):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(f'{H}/atlas/data/qwen3_tokenizer.json')
    V = tk.get_vocab_size(); txt = [tk.decode([i]) for i in range(V)]
    cls, dig = token_tables(txt); cls = torch.tensor(cls); dig = torch.tensor(dig, dtype=torch.float64)
    vf = pickle.load(open(f'{HERE}/cache/vf_{S}.pkl', 'rb'))
    rows = [json.loads(l) for l in gzip.open(f'{H}/atlas/data/attempts_{S}.jsonl.gz')]
    rows = [r for r in rows if r['reg_ok']]                          # committed, valid answer (the slots the voter keeps)
    for r in rows:
        for k in ('seg_tokens',):
            if isinstance(r[k], str): r[k] = json.loads(r[k])
        if isinstance(r['hash_off'], str): r['hash_off'] = int(r['hash_off'])
    bad_n = bad_q = 0; B = 256
    for i in range(0, len(rows), B):
        R = rows[i:i + B]; b = len(R)
        bt = [body_ids(a, None) for a in R]; qt = [tk.encode(f"Question: {a['q']}\nAnswer:", add_special_tokens=False).ids for a in R]
        def run(seqs):
            T = max(map(len, seqs)); I = torch.zeros(b, T, dtype=torch.long)
            for j, s in enumerate(seqs): I[j, :len(s)] = torch.tensor(s)
            nr = NumReg(b, 32, 'cpu'); nr.run(torch.arange(b), cls[I], dig[I], torch.tensor([len(s) for s in seqs])); return nr
        nb, nq = run(bt), run(qt)
        for j, a in enumerate(R):
            f = vf[a['gid']]
            Sb = set(nb.S[j, :nb.n[j]].tolist()); Sq = set(nq.S[j, :nq.n[j]].tolist())
            qc = sum(x in Sb for x in Sq) / len(Sq) if Sq else 1.0
            bad_n += abs(int(nb.n[j]) / 10 - f['nnum']) > 1e-9; bad_q += abs(qc - f['qcov_d']) > 1e-9
    print(f'[2] numreg on real {S} attempts (valid, n={len(rows)}): nnum mismatches {bad_n}, qcov_d mismatches {bad_q}',
          'PASS' if bad_n + bad_q <= len(rows) * 0.002 else 'FAIL')
    if bad_n + bad_q > len(rows) * 0.002: FAIL.append(2)


# ---------------------------------------------------------------- engine stubs (as implants/voter2/test_voter2.py)
def stubs():
    tri = types.ModuleType('triton'); tri.jit = lambda f: f; tri.cdiv = lambda a, b: -(-a // b)
    tl = types.ModuleType('triton.language'); tl.constexpr = int; tri.language = tl
    fq = types.ModuleType('fastqwen'); fq.D = 32
    hk = types.ModuleType('hostkern'); en = types.ModuleType('engine')
    class _Engine:
        def _mut(self): return []
    en.Engine = _Engine
    sys.modules.update({'triton': tri, 'triton.language': tl, 'fastqwen': fq, 'hostkern': hk, 'engine': en})
    return fq


VOC = ['<eos>', '\n\n'] + sorted(set("Wait, let me solve it again.\n We need to find####0123456789-.,"))
VOC = list(dict.fromkeys(VOC + ['####'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:+=')))


class Tok:
    def __len__(self): return len(VOC)
    def decode(self, ids): return ''.join(VOC[i] for i in ids)
    def __call__(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            if s.startswith('####', i): out.append(VOC.index('####')); i += 4
            else: out.append(VOC.index(s[i])); i += 1
        return {'input_ids': out}


def make_engine(B, fq, SS, vf):
    os.environ.pop('SER_V2', None)
    if vf: os.environ['SER_VF'] = '1'; os.environ['SER_VF_W'] = f'{HERE}/job/stk/vf_w.pt'
    else: os.environ.pop('SER_VF', None)
    e = SS.SerialEngine.__new__(SS.SerialEngine)
    e.tok = Tok(); e.dev = 'cpu'; e.B_max = B; e.S_max = 64; e.eos = 0
    e.W = {'E': torch.zeros(len(VOC), fq.D)}
    e.has_nn = torch.tensor(['\n\n' in t for t in VOC]); e.ends_nl = torch.tensor([t.endswith('\n') for t in VOC])
    e.starts_nl = torch.tensor([t.startswith('\n') for t in VOC])
    e.done = torch.zeros(B + 1, dtype=torch.bool); e.glen = torch.full((B + 1,), 100, dtype=torch.long)
    e.slot = torch.full((B + 1,), 120, dtype=torch.long); e.temp = 1.0
    e.serial_setup(acap=320)
    e.set_serial(N=8, vmode=1, forget=True, posfix=True, R=0, san=True)
    return e


VALS = [3.0, 7.0, 12.0, 0.5, 120.0, 7.0000001, -4.0, 1e6]


def base_ref(val, ok, bad, lg, n):
    sane = [ok[i] and not bad[i] for i in range(n)]; keep = sane if any(sane) else ok[:n]
    best, bi = None, -1
    for i in range(n):
        if not keep[i]: continue
        sc = sum(1 / (1 + math.exp(-lg[j])) for j in range(n) if keep[j] and abs(val[i] - val[j]) <= 1e-6 * max(1.0, abs(val[j]))) - 1e-9 * i
        if best is None or sc > best: best, bi = sc, i
    return bi


def t_voter(trials, B=64):
    """random registers + reader state of the committing attempt -> compiled _act; vf pick vs voter/learn.ClusterLin (numpy),
    pick0 / vf-off pick vs the base sigmoid(lg) voter, '#### X' queue tokens of the picked slot."""
    import math as _m; globals()['math'] = _m
    fq = stubs(); import serial_stack as SS
    sys.path.insert(0, f'{H}/voter'); from learn import ClusterLin
    W = torch.load(f'{HERE}/job/stk/vf_w.pt'); m = ClusterLin(); m.th = np.array(W['th']); m.c0 = float(W['c0'])
    bad_vf = bad_p0 = bad_off = bad_q = bad_reg = n_diff = n_chk = 0
    for vf in (True, False):
        rng = random.Random(3); e = make_engine(B, fq, SS, vf)
        tok = torch.full((B,), VOC.index(' '), dtype=torch.long); nxt = torch.zeros(B, dtype=torch.long); r = torch.arange(B)
        for t in range(trials // B):
            n = rng.randint(1, 8); e.k_N.fill_(n)
            for x in (e.qs, e.ql, e.qp, e.rep, e.seg, e.R_n): x.zero_()
            e.qfin.zero_(); e.pick.fill_(-1); e.pick0.fill_(-1); e.astart.fill_(90); e.att[:B] = n - 1
            if vf:
                for x in e.vf_b.tensors() + e.vf_q.tensors(): x.zero_()
            exp = []
            for i in range(B):
                val = [rng.choice(VALS) for _ in range(8)]; ok = [rng.random() < 0.8 for _ in range(8)]; bad = [rng.random() < 0.2 for _ in range(8)]
                lg = [rng.choice([-30.0, rng.gauss(0, 4)]) for _ in range(8)]
                nn = [float(rng.randint(0, 18)) for _ in range(8)]; qc = [rng.choice([1.0, 0.5, 0.75, 0.0, 2 / 3]) for _ in range(8)]
                nt = [float(rng.choice([rng.randint(40, 330), 120])) for _ in range(8)]
                for k in range(8):
                    e.R_val[i, k] = val[k]; e.R_ok[i, k] = ok[k]; e.R_bad[i, k] = bad[k]; e.R_lg[i, k] = lg[k]; lg[k] = float(e.R_lg[i, k])
                    if vf: e.R_nn[i, k] = nn[k]; e.R_qc[i, k] = qc[k]; e.R_nt[i, k] = nt[k]
                    ntk = rng.randint(1, 4); e.R_n[i, k] = ntk; e.R_tok[i, k, :ntk] = torch.tensor([VOC.index(str(rng.randint(0, 9))) for _ in range(ntk)])
                ph = rng.choice([0, 3, 5, 6, 6]); hasd = rng.random() < 0.9; neg = rng.random() < 0.1; fnz = rng.random() < 0.1
                cv = rng.choice(VALS); lgset = rng.random() < 0.9; clg = rng.gauss(0, 4)
                e.ph[i] = ph; e.hasd[i] = hasd; e.neg[i] = neg; e.fnz[i] = fnz; e.val[i] = cv; e.lgset[i] = lgset; e.lg[i] = clg
                na = rng.randint(1, 4); e.an[i] = na; e.atok[i, :na] = torch.tensor([VOC.index(str(rng.randint(0, 9))) for _ in range(na)])
                k = n - 1; valid = 3 <= ph <= 6 and hasd
                val[k] = -cv if neg else cv; ok[k] = valid; bad[k] = (not valid) or neg or fnz
                lg[k] = float(np.float32(clg)) if lgset else -30.0
                if vf:      # reader state of the committing attempt: body set, question set, segment start (glen = 100)
                    nb = rng.randint(0, 12); Sb = rng.sample([1.0, 2.0, 3.0, 5.0, 50.0, 60.0, 0.5, 12.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0], nb)
                    nq = rng.randint(0, 5); Sq = rng.sample([1.0, 2.0, 3.0, 50.0, 60.0, 0.25, 100.0], nq)
                    e.vf_b.S[i, :nb] = torch.tensor(Sb, dtype=torch.float64); e.vf_b.n[i] = nb
                    e.vf_q.S[i, :nq] = torch.tensor(Sq, dtype=torch.float64); e.vf_q.n[i] = nq
                    ss = rng.randint(0, 99); e.vf_ss[i] = ss
                    nn[k] = float(nb); qc[k] = sum(x in Sb for x in Sq) / nq if nq else 1.0; nt[k] = float(100 - ss + 1)
                # numpy reference (offline ClusterLin over the n committed slots)
                st = types.SimpleNamespace(S=list(range(n)))
                sane = np.array([ok[j] and not bad[j] for j in range(n)]); okn = np.array(ok[:n])
                st.keep = sane if sane.any() else okn
                v = np.array(val[:n]); st.same = np.abs(v[:, None] - v[None]) <= 1e-6 * np.maximum(np.abs(v[None]), 1.0)
                ntn = np.array(nt[:n]); zt = (ntn - ntn.mean()) / (ntn.std() + 1e-3) if n > 1 else np.zeros(n)
                st.F = np.stack([np.maximum(np.array(lg[:n]), -12), sane.astype(float), np.array(nn[:n]) / 10, zt, np.array(qc[:n])], 1)
                if st.keep.any():
                    sc = np.where(st.keep, m.score(st) - 1e-9 * np.arange(n), -1e18); pv = int(np.argmax(sc))
                else: pv = -1
                p0 = base_ref(val, ok, bad, lg, n)
                exp.append((pv, p0, e.atok[i, :na].tolist(), (nn[k], qc[k], nt[k])))
            e._act(B, tok, nxt, torch.zeros(B), torch.zeros(B, len(VOC)), r)
            if n == 1: continue
            for i, (pv, p0, atk, regk) in enumerate(exp):
                n_chk += 1; want = pv if vf else p0
                if vf:
                    bad_vf += int(e.pick[i]) != pv; bad_p0 += int(e.pick0[i]) != p0; n_diff += pv != p0
                    got = (float(e.R_nn[i, n - 1]), float(e.R_qc[i, n - 1]), float(e.R_nt[i, n - 1]))
                    bad_reg += any(abs(x - y) > 1e-12 for x, y in zip(got, regk))
                else: bad_off += int(e.pick[i]) != p0
                ql = int(e.ql[i]); q = e.qs[i, :ql].tolist(); s_ = VOC.index('####')
                if want >= 0:
                    hp = q.index(s_); w_ = atk if want == n - 1 else e.R_tok[i, want, :int(e.R_n[i, want])].tolist()
                    bad_q += q[hp + 1:-1] != w_ or q[-1] != 0
                else: bad_q += q != [0]
    nb_ = bad_vf + bad_p0 + bad_off + bad_q + bad_reg
    print(f'[3] voter fuzz: {n_chk} rows (vf on + off): vf-pick mismatches {bad_vf}, pick0 {bad_p0}, vf-off pick {bad_off}, '
          f'committed registers {bad_reg}, queue (#### X) {bad_q}; rows where vf and base differ: {n_diff}', 'PASS' if nb_ == 0 else 'FAIL')
    nb_ and FAIL.append(3)


def t_stream():
    """token-level drive of _read/_act (one row, two attempts): committed nn / qc / nt vs the regex reference"""
    fq = stubs(); import serial_stack as SS
    e = make_engine(1, fq, SS, True); e.k_N.fill_(2); T = Tok()
    q = T("Question: Tom has 3,000 apples and 2.5 kg, 12 pears.\nAnswer:")['input_ids']
    r = torch.arange(1)
    for x in e.vf_b.tensors() + e.vf_q.tensors() + [e.vf_ss, e.R_nn, e.R_qc, e.R_nt]: x.zero_()
    e.vf_q.run(r, e.r_cls[torch.tensor([q])], e.r_dig[torch.tensor([q])], torch.tensor([len(q)]))
    e.glen.zero_(); e.att.zero_(); e.astart.zero_()
    bodies = [" 3,000 + 12 = 3012, so 7.5.\n#### 3012\n\n", " 3000 and 2.5, 1,2 then 12.\n#### 15\n\n"]
    seq = T(bodies[0])['input_ids']; outs = []; i = 0; k = 0; nt = 0
    tok = torch.tensor([seq[0]])
    while k < 2 and i < 400:
        e._read(1, tok, r)
        nx = torch.tensor([seq[i + 1] if i + 1 < len(seq) else 0])
        new, *_ = e._act(1, tok, nx, torch.zeros(1), torch.zeros(1, len(VOC)), r)
        e.glen += 1; i += 1
        if int(e.att[0]) > k or (k == 1 and int(e.qfin[0])):
            k += 1
            if k == 1: seq = [int(x) for x in e.qs[0, :int(e.ql[0])]] + T(bodies[1])['input_ids']; i = 0; tok = torch.tensor([seq[0]]); continue
        tok = new
    from fractions import Fraction
    Q = {float(x) for x in ref_numbers(*zip(*[(int(e.r_cls[t]), int(e.r_dig[t])) for t in q]))}
    ok = True
    for k, b in enumerate(bodies):
        body = b.split('####')[0]; ids = T(body)['input_ids']
        Sb = {float(x) for x in ref_numbers(*zip(*[(int(e.r_cls[t]), int(e.r_dig[t])) for t in ids]))}
        qc = sum(x in Sb for x in Q) / len(Q)
        got = (float(e.R_nn[0, k]), float(e.R_qc[0, k]))
        ok &= got == (float(len(Sb)), qc)
        print(f'  attempt {k}: body nums {sorted(Sb)} q nums {sorted(Q)} -> want nn={len(Sb)} qc={qc:.3f}, got nn={got[0]:.0f} qc={got[1]:.3f} nt={float(e.R_nt[0, k]):.0f} end={int(e.R_end[0, k])}')
    nt_ok = float(e.R_nt[0, 0]) == int(e.R_end[0, 0]) + 1 and float(e.R_nt[0, 1]) == int(e.R_end[0, 1]) - int(e.R_end[0, 0])
    print('[4] token-level stream:', 'PASS' if ok and nt_ok else 'FAIL'); (ok and nt_ok) or FAIL.append(4)


def runs():
    """in-stream registers (job/out/*_vf.jsonl.gz) vs offline features (cache/vf_*.pkl, voter/cache att ntok) on rows whose
    tokens are identical to the base rollout (gen/out) up to the final vote; same-stream pick vs pick0 accuracy"""
    import vote as V  # noqa
    vf_ok = True
    for S in ('dev2', 'dev'):
        vfc = pickle.load(open(f'{HERE}/cache/vf_{S}.pkl', 'rb'))
        by, _ = V.load_set(S); nt_off = {f"{S}/{a['idx']}/s{a['seed']}/k{a['k']}": a['ntok'] for L in by.values() for a in L}
        for sd in (7, 11, 13, 17):
            f = f'{HERE}/job/out/{S}_N8_R1_v1_s1_m0_T1.0_seed{sd}_vf.jsonl.gz'
            if not os.path.exists(f): print('  missing', f); continue
            R = [json.loads(l) for l in gzip.open(f)]
            B0 = [json.loads(l) for l in gzip.open(f'{H}/gen/out/{S}_N8_R1_v1_s1_m0_T1.0_seed{sd}.jsonl.gz')]
            same = n_att = bn = bq = bt = 0; nrows = len(R)
            for r, b in zip(R, B0):
                assert r['idx'] == b['idx'] and r['q'] == b['q']
                e = r['end'][r['natt'] - 1]
                if r['natt'] != b['natt'] or r['tokens'][:e + 1] != b['tokens'][:e + 1]: continue
                same += 1
                for k in range(r['natt']):
                    g = f"{S}/{r['idx']}/s{sd}/k{k}"
                    if g not in vfc: continue
                    n_att += 1; c = vfc[g]
                    bn += abs(r['nn'][k] / 10 - c['nnum']) > 1e-9; bq += abs(r['qc'][k] - c['qcov_d']) > 1e-9
                    if g in nt_off: bt += abs(r['nt'][k] - nt_off[g]) > 0
            print(f'  {S} s{sd}: rows {nrows}, token-identical to base {same}; attempts {n_att}: nnum mismatches {bn}, qcov_d {bq}, ntok {bt}')
            vf_ok &= (bn + bq + bt) <= 0.005 * max(n_att, 1)
    print('[runs] in-stream registers vs offline:', 'PASS' if vf_ok else 'FAIL'); vf_ok or FAIL.append('runs')


if __name__ == '__main__':
    if A.runs: runs()
    else: t_fuzz(); t_real('dev2'); t_voter(A.trials); t_stream()
    print('ALL PASS' if not FAIL else f'FAILED {FAIL}')
