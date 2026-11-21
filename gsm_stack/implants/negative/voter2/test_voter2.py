"""CPU tests for the v2 verifier + voter implant (job/stk/serial_stack.py, SER_V2=1).

python3.11 implants/voter2/test_voter2.py [--trials 4000] [--runs]
 1. fuzz: random register contents (8 slots: value, valid, bad, L18 logit, v2 logit; plus the reader state of the attempt
    being committed) -> the compiled _act (register commit + exp-weight voter + final-queue writer), checked against an
    independent python reference voter: chosen slot, base pick (sigmoid(lg) voter), and the written '#### X' tokens.
 2. --runs: in-stream v2 register (job/out/*_v2.jsonl.gz) vs the offline probe on teacher-forced states
    (probe/out/pred_trainC/hash_{set}.npz, same trainC weights) per attempt (joined on idx, seed, k): Pearson corr, MAE.
"""
import os, sys, types, math, random, argparse, gzip, json
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants')
ap = argparse.ArgumentParser(); ap.add_argument('--trials', type=int, default=4000); ap.add_argument('--runs', action='store_true')
A = ap.parse_args()

# ---------------------------------------------------------------- stubs: triton / fastqwen / hostkern / engine (CPU import)
tri = types.ModuleType('triton'); tri.jit = lambda f: f; tri.cdiv = lambda a, b: -(-a // b)
tl = types.ModuleType('triton.language'); tl.constexpr = int; tri.language = tl
fq = types.ModuleType('fastqwen'); fq.D = 32
hk = types.ModuleType('hostkern')
en = types.ModuleType('engine')
class _Engine:
    def _mut(self): return []
en.Engine = _Engine
sys.modules.update({'triton': tri, 'triton.language': tl, 'fastqwen': fq, 'hostkern': hk, 'engine': en})
sys.path.insert(0, os.path.join(HERE, 'job', 'stk'))
import serial_stack as SS

VOC = ['<eos>', '\n\n'] + sorted(set("Wait, let me solve it again.\n We need to find####0123456789-.,"))
VOC = list(dict.fromkeys(VOC + ['####']))


class Tok:
    def __len__(self): return len(VOC)
    def decode(self, ids): return ''.join(VOC[i] for i in ids)
    def __call__(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            if s.startswith('####', i): out.append(VOC.index('####')); i += 4
            else: out.append(VOC.index(s[i])); i += 1
        return {'input_ids': out}


def make_engine(B):
    e = SS.SerialEngine.__new__(SS.SerialEngine)
    e.tok = Tok(); e.dev = 'cpu'; e.B_max = B; e.S_max = 64; e.eos = 0
    e.W = {'E': torch.zeros(len(VOC), fq.D)}
    txt = VOC
    e.has_nn = torch.tensor(['\n\n' in t for t in txt]); e.ends_nl = torch.tensor([t.endswith('\n') for t in txt])
    e.starts_nl = torch.tensor([t.startswith('\n') for t in txt])
    e.done = torch.zeros(B + 1, dtype=torch.bool); e.glen = torch.full((B + 1,), 100, dtype=torch.long)
    e.slot = torch.full((B + 1,), 120, dtype=torch.long); e.temp = 1.0
    e.serial_setup(acap=320)
    e.v2 = True
    e.v2_th.copy_(torch.tensor([1.3, 0.4, 0.8], dtype=torch.float64)); e.v2_b0.fill_(-0.7); e.v2_b.fill_(0.25)
    e.set_serial(N=8, vmode=1, forget=True, posfix=True, R=0, san=True)
    return e


VALS = [3.0, 7.0, 12.0, 0.5, 120.0, 7.0000001, -4.0, 1e6]


def ref_pick(val, ok, bad, w, n):
    used = list(range(n)); okk = [ok[i] for i in used]; sane = [ok[i] and not bad[i] for i in used]
    keep = sane if any(sane) else okk
    best, bi = None, -1                    # spec (as the stack): argmax of cluster score - 1e-9 * slot, earliest on ties
    for i in used:
        if not keep[i]: continue
        sc = sum(w[j] for j in used if keep[j] and abs(val[i] - val[j]) <= 1e-6 * max(1.0, abs(val[j]))) - 1e-9 * i
        if best is None or sc > best: best, bi = sc, i
    return bi


def fuzz(trials, B=64, seed=0, sig=False):
    global SIG; SIG = sig
    rng = random.Random(seed); e = make_engine(B); e.v2_sig.fill_(sig)
    th = e.v2_th.tolist(); b0 = float(e.v2_b0); b = float(e.v2_b)
    tok = torch.full((B,), VOC.index(' '), dtype=torch.long); nxt = torch.zeros(B, dtype=torch.long)
    r = torch.arange(B); n_bad = n_bad0 = n_q = 0; n_chk = 0; n_diff = 0
    for t in range(trials // B):
        n = rng.randint(1, 8); e.k_N.fill_(n)
        for x in (e.qs, e.ql, e.qp, e.rep, e.seg, e.R_n): x.zero_()
        e.qfin.zero_(); e.pick.fill_(-1); e.pick0.fill_(-1); e.astart.fill_(90)
        e.att[:B] = n - 1
        exp = []
        for i in range(B):
            val = [rng.choice(VALS) for _ in range(8)]; ok = [rng.random() < 0.8 for _ in range(8)]; bad = [rng.random() < 0.2 for _ in range(8)]
            lg = [rng.choice([-30.0, rng.gauss(0, 4)]) for _ in range(8)]; v2 = [rng.choice([-30.0, rng.gauss(0, 4), rng.gauss(0, 15)]) for _ in range(8)]
            # earlier slots: registers; the current slot (n-1) is committed by _act from the reader state
            for k in range(8):
                e.R_val[i, k] = val[k]; e.R_ok[i, k] = ok[k]; e.R_bad[i, k] = bad[k]; e.R_lg[i, k] = lg[k]; e.R_v2[i, k] = v2[k]
                lg[k] = float(e.R_lg[i, k]); v2[k] = float(e.R_v2[i, k])
                nt = rng.randint(1, 4); e.R_n[i, k] = nt; e.R_tok[i, k, :nt] = torch.tensor([VOC.index(str(rng.randint(0, 9))) for _ in range(nt)])
            ph = rng.choice([0, 3, 5, 6, 6]); hasd = rng.random() < 0.9; neg = rng.random() < 0.1; fnz = rng.random() < 0.1
            cv = rng.choice(VALS); lgset = rng.random() < 0.9
            clg = rng.gauss(0, 4); hz = rng.gauss(0, 3); ms = rng.gauss(0, 20); mc = float(rng.choice([0, 1, 3, 9]))
            e.ph[i] = ph; e.hasd[i] = hasd; e.neg[i] = neg; e.fnz[i] = fnz; e.val[i] = cv; e.lgset[i] = lgset; e.lg[i] = clg
            e.v2_hz[i] = hz; e.v2_ms[i] = ms; e.v2_mc[i] = mc
            na = rng.randint(1, 4); e.an[i] = na; e.atok[i, :na] = torch.tensor([VOC.index(str(rng.randint(0, 9))) for _ in range(na)])
            k = n - 1
            valid = 3 <= ph <= 6 and hasd
            val[k] = -cv if neg else cv; ok[k] = valid; bad[k] = (not valid) or neg or fnz
            lg[k] = float(np.float32(clg)) if lgset else -30.0
            v2[k] = float(np.float32(np.float32(hz) + np.float32(ms) / np.float32(max(mc, 1.0)) + np.float32(b))) if lgset else -30.0
            sane = [ok[j] and not bad[j] for j in range(8)]
            zz = [th[0] * max(v2[j], -12) + th[1] * max(lg[j], -10) + th[2] * sane[j] + b0 for j in range(8)]
            w2 = [1 / (1 + math.exp(-z_)) if SIG else math.exp(max(-30, min(30, z_))) for z_ in zz]
            w0 = [1 / (1 + math.exp(-lg[j])) for j in range(8)]
            p2 = ref_pick(val, ok, bad, w2, n); p0 = ref_pick(val, ok, bad, w0, n)
            atk = e.atok[i, :na].tolist()
            exp.append((p2, p0, n, atk, v2[k]))
        lp = torch.zeros(B); logits = torch.zeros(B, len(VOC))
        new, lp2, wbit, writing = e._act(B, tok, nxt, lp, logits, r)
        for i, (p2, p0, n_, atk, v2k) in enumerate(exp):
            n_chk += 1
            if n > 1:
                n_bad += int(e.pick[i]) != p2; n_bad0 += int(e.pick0[i]) != p0
                n_diff += p2 != p0
                ql = int(e.ql[i]); q = e.qs[i, :ql].tolist(); s = VOC.index('####')
                if p2 >= 0:
                    hp = q.index(s); want = (atk if p2 == n - 1 else e.R_tok[i, p2, :int(e.R_n[i, p2])].tolist())
                    n_q += q[hp + 1:-1] != want or q[-1] != 0
                else:
                    n_q += q != [0]
            if abs(float(e.R_v2[i, n - 1]) - v2k) > 1e-4 * max(1, abs(v2k)): n_bad += 1
    print(f'fuzz ({"sigmoid" if sig else "exp"} weights): {n_chk} rows ({trials // B} batches): v2-pick mismatches {n_bad}, base-pick mismatches {n_bad0}, '
          f'queue (#### X) mismatches {n_q}; rows where v2 and base picks differ: {n_diff}')
    return n_bad + n_bad0 + n_q


def qstart(wf):
    """start of the final queue line ('#### X' written by the voter): last token with the queue-start bit 8"""
    return max(i for i, w in enumerate(wf) if w & 8)


def runs():
    def gids(S):
        out = []
        for l in gzip.open(f'{H}/probe/tables/attempts_{S}.jsonl.gz', 'rt'):
            i = l.index('"gid": "') + 8; out.append(l[i:l.index('"', i)])
        return out
    od = os.path.join(HERE, 'job', 'out')
    for S in ('dev', 'dev2'):
        z = dict(np.load(f'{H}/probe/out/pred_trainC/hash_{S}.npz')); c = list(z['cols']).index('hash_mnl_c4'); G = gids(S)
        off = {G[a]: float(z['S'][r, c]) for r, a in enumerate(z['att'])}
        for f in sorted(os.listdir(od)):
            if not (f.startswith(S + '_N8') and f.endswith('_v2.jsonl.gz')): continue
            sd = int(f.split('_seed')[1].split('_')[0]); x, y = [], []
            base = {r['idx']: r for r in map(json.loads, gzip.open(f'{H}/gen/out/{f.replace("_v2.jsonl.gz", ".jsonl.gz")}', 'rt'))}
            nskip = 0
            for l in gzip.open(os.path.join(od, f), 'rt'):
                d = json.loads(l); bb = base[d['idx']]
                if d['tokens'][:qstart(d['wf'])] != bb['tokens'][:qstart(bb['wf'])]: nskip += 1; continue   # offline states = base tokens
                for k in range(d['natt']):
                    g = f'{S}/{d["idx"]}/s{sd}/k{k}'
                    if g in off and d['v2'][k] > -29: x.append(d['v2'][k]); y.append(off[g])
            x, y = np.array(x), np.array(y)
            print(f'{S} seed {sd}: in-stream v2 vs offline (teacher-forced) n={len(x)} corr={np.corrcoef(x, y)[0, 1]:.5f} '
                  f'MAE={np.abs(x - y).mean():.4f} max|d|={np.abs(x - y).max():.3f} sign agree={(np.sign(x) == np.sign(y)).mean():.4f} (rows with prefix != base skipped: {nskip})')


if __name__ == '__main__':
    bad = fuzz(A.trials // 2, sig=False) + fuzz(A.trials // 2, seed=1, sig=True)
    if A.runs: runs()
    print('PASS' if bad == 0 else 'FAIL')
    sys.exit(0 if bad == 0 else 1)
