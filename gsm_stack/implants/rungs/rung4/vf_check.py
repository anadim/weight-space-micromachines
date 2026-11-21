"""ladnext offline reference for AW_VF rows (node, needs the tokenizer): python3 vf_check.py ROWS.jsonl.gz [...]
Per committed attempt slot k < natt (end[k] >= 0): recompute from the row's tokens / wf / question text
  nt = attempt's final-segment length: end[k] - (last queue start <= end[k], wf bit 8; 0 if none) + 1
  nn = |distinct text_numbers(body)|, body = decode(host tokens of that segment (wf bit 2 clear) while '####' has not appeared)
  qc = fraction of distinct text_numbers(question) found in the body numbers (1 if the question has none)
and compare to the in-graph registers (nn, qc, nt). Then the picks from the row registers: vf pick vs offline ClusterLin
(vf_learn_ref.py, fixed vf_w.pt), pick0 vs the base sigmoid(lg) group voter. Prints mismatch counts (only rows with a final pick)."""
import ast, gzip, json, os, re, sys, types
from fractions import Fraction
import numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
NUMRE = re.compile(r'\d+(?:\.\d+)?')
_ts = {}
def TS(x):
    if x not in _ts: _ts[x] = tok.decode([x])
    return _ts[x]


def text_numbers(t):                     # = voter/feats.text_numbers
    t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', t)
    return [Fraction(m.group(0)) for m in NUMRE.finditer(t)]


_src = open(os.path.join(H, 'vf_learn_ref.py')).read(); _ns = {'np': np}
exec(compile(ast.Module(body=[n for n in ast.parse(_src).body if isinstance(n, ast.ClassDef) and n.name == 'ClusterLin'],
                        type_ignores=[]), 'vf_learn_ref.py', 'exec'), _ns)
W = torch.load(os.path.join(H, 'stk', 'vf_w.pt'))
M = _ns['ClusterLin'](); M.th = np.array(W['th'], dtype=np.float64); M.c0 = float(W['c0'])
SAN = os.environ.get('AW_SAN', '1') == '1'


def seg_feats(r, k, qn):
    t, w, e = r['tokens'], r['wf'], r['end'][k]
    s = max([i for i in range(e + 1) if w[i] & 8], default=0)
    host = [t[i] for i in range(s, e + 1) if not (w[i] & 2)]
    body = []; acc = ''
    for x in host:
        if '####' in acc: break
        body.append(x); acc += TS(x)
    txt = tok.decode(body)
    bn = set(text_numbers(txt))
    qc = (sum(x in bn for x in qn) / len(qn)) if qn else 1.0
    return float(len(bn)), qc, float(e - s + 1)


def check(path):
    rows = [json.loads(l) for l in gzip.open(path, 'rt')]
    nreg = [0, 0, 0]; nslot = 0; bvf = bp0 = npk = ndiff = 0; vf_ok = p0_ok = 0
    for r in rows:
        n = r['natt']; qn = set(text_numbers(r['q']))
        for k in range(min(n, len(r['end']))):
            if r['end'][k] < 0: continue
            nslot += 1
            nn, qc, nt = seg_feats(r, k, qn)
            nreg[0] += abs(nn - r['nn'][k]) > 1e-9; nreg[1] += abs(qc - r['qc'][k]) > 1e-9; nreg[2] += abs(nt - r['nt'][k]) > 1e-9
        if r['pick0'] < 0 and r['pick'] < 0: continue
        npk += 1
        sy = [tuple(x) for x in r['sym'][:n]]
        ok = np.array(r['rok'][:n], bool); sane = ok & ~np.array(r['bad'][:n], bool)
        st = types.SimpleNamespace(S=list(range(n)))
        st.keep = sane if (SAN and sane.any()) else ok
        st.same = np.array([[sy[i] == sy[j] for j in range(n)] for i in range(n)])
        ntn = np.array(r['nt'][:n]); zt = (ntn - ntn.mean()) / (ntn.std() + 1e-3) if n > 1 else np.zeros(n)
        lg = np.array(r['lg'][:n], dtype=np.float64)
        st.F = np.stack([np.maximum(lg, -12), sane.astype(float), np.array(r['nn'][:n]) / 10, zt, np.array(r['qc'][:n])], 1)
        if st.keep.any():
            sc = np.where(st.keep, M.score(st) - 1e-9 * np.arange(n), -1e18); pv = int(np.argmax(sc))
            bs = {}
            for i in range(n):
                if st.keep[i]: bs[sy[i]] = bs.get(sy[i], 0) + 1 / (1 + np.exp(-lg[i]))
            mx = max(bs.values()); p0 = min(i for i in range(n) if st.keep[i] and bs[sy[i]] >= mx - 1e-6)
        else: pv = p0 = -1
        bvf += pv != r['pick']; bp0 += p0 != r['pick0']; ndiff += r['pick'] != r['pick0']
        g = r['gold']
        val = lambda i: r['val'][i] if i >= 0 else None
        cor = lambda i: i >= 0 and abs(val(i) - float(g)) <= 1e-6 * max(1.0, abs(float(g)))
        vf_ok += cor(r['pick']); p0_ok += cor(r['pick0'])
    N = len(rows)
    print(f"VFCHECK {os.path.basename(path)} rows {N} slots {nslot}: register mismatches nn {nreg[0]} qc {nreg[1]} nt {nreg[2]}; "
          f"rows with a final pick {npk}: vf-pick vs ClusterLin {bvf}, pick0 vs base voter {bp0}; vf != pick0 {ndiff}; "
          f"same-stream acc(register value) vf {100 * vf_ok / N:.2f} pick0 {100 * p0_ok / N:.2f}", flush=True)


if __name__ == '__main__':
    for p in sys.argv[1:]: check(p)
