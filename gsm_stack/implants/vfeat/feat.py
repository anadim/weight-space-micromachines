"""Cheap per-attempt voter features (lane voter-feat).  -> implants/vfeat/cache/vf_{set}.pkl  {gid: {feat: value}}

python3.11 implants/vfeat/feat.py [sets...]
Features (all from the attempt text / sampled logp, i.e. computable in the stream):
  qcov_d   fraction of question digit-numbers that appear in the attempt body (before '####')   [= voter/feats qcov]
  qcov_w   same, question numbers incl. number words (twice, half, dozen, one..twenty, ...)
  nmiss_d / nmiss_w   count of missing question quantities
  fk{t}    per 100 host tokens: count of sampled tokens with p < t (close-fork proxy; needs no top-2)
  tfk3/tfk5 (dev/dev2 mixed-item attempts only) true close-fork rate p2/p1 >= 0.3 / 0.5 per 100 host tokens (probe/branch pos)
  ans_last_eq  answer value == last '= v' value of the body
  nnum     number of distinct numbers in the body / 10
"""
import os, re, sys, gzip, json, math, pickle
from fractions import Fraction
import numpy as np

H = os.path.expanduser('gsm_stack/implants'); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{H}/voter')
from feats import text_numbers, body_text, norm_num

WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
         'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
         'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
         'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'twice': 2, 'double': 2, 'doubled': 2, 'triple': 3,
         'tripled': 3, 'thrice': 3, 'quadruple': 4, 'half': Fraction(1, 2), 'halves': Fraction(1, 2), 'third': Fraction(1, 3),
         'thirds': Fraction(1, 3), 'quarter': Fraction(1, 4), 'quarters': Fraction(1, 4), 'fourth': Fraction(1, 4),
         'dozen': 12, 'dozens': 12, 'fifth': Fraction(1, 5), 'tenth': Fraction(1, 10)}
WRE = re.compile(r'\b(' + '|'.join(sorted(WORDS, key=len, reverse=True)) + r')\b', re.I)


def word_nums(t):
    return [Fraction(WORDS[m.group(1).lower()]) for m in WRE.finditer(t)]


def covered(x, nums):
    if x in nums: return True
    if x.numerator == 1 and x.denominator > 1:              # 'half' -> '/2', '0.5', '50%'
        return Fraction(x.denominator) in nums or x * 100 in nums
    return False


def feats(a, tfk):
    body, _ = body_text(a)
    dq = set(text_numbers(a['q'])); wq = set(word_nums(a['q'])) | dq
    bn = set(text_numbers(body)); bw = bn | set(word_nums(body))
    md = [x for x in dq if x not in bn]; mw = [x for x in wq if not covered(x, bw)]
    lp, wf = a['seg_logp'], a['seg_wf']
    ho = a.get('hash_off'); ho = len(lp) if ho is None else int(ho)
    host = [l for l, w in list(zip(lp, wf))[:ho] if w == 0]
    nh = max(1, len(host))
    r = dict(qcov_d=1 - len(md) / len(dq) if dq else 1.0, qcov_w=1 - len(mw) / len(wq) if wq else 1.0,
             nmiss_d=float(len(md)), nmiss_w=float(len(mw)), nq_w=float(len(wq)),
             nnum=len(bn) / 10.0)
    for t in (0.3, 0.5, 0.77, 0.9):
        r[f'fk{int(t * 100)}'] = 100 * sum(1 for l in host if l < math.log(t)) / nh
    eqv = [norm_num(m.group(1)) for m in re.finditer(r'=\s*\$?\s*(\d[\d,]*(?:\.\d+)?)', body)]
    eqv = [x for x in eqv if x is not None]
    v = a['reg_val'] if a['reg_ok'] else None
    r['ans_last_eq'] = float(v is not None and bool(eqv) and abs(float(eqv[-1]) - v) <= 1e-6 * max(1, abs(v)))
    if tfk is not None:
        r.update(tfk)
    return r


def true_forks(S):
    """gid -> {tfk3, tfk5} from probe/branch/data/{S}_pos.npz (mixed-item attempts, host tokens o >= plen)"""
    p = f'{H}/probe/branch/data/{S}_pos.npz'
    if not os.path.exists(p): return {}
    J = [json.loads(l) for l in gzip.open(f'{H}/probe/branch/job/tables/jobs_{S}.jsonl.gz')]
    z = np.load(p)
    att, wf = z['att'], z['wf']; r = np.exp(z['lp2'].astype(np.float32) - z['lp1'].astype(np.float32))
    out = {}
    m = wf == 0
    n = np.bincount(att[m], minlength=len(J)); c3 = np.bincount(att[m], weights=(r[m] >= 0.3), minlength=len(J))
    c5 = np.bincount(att[m], weights=(r[m] >= 0.5), minlength=len(J))
    for i, j in enumerate(J):
        if n[i]: out[j['gid']] = dict(tfk3=100 * c3[i] / n[i], tfk5=100 * c5[i] / n[i])
    return out


def run(S):
    T = true_forks(S)
    out = {}
    for line in gzip.open(f'{H}/atlas/data/attempts_{S}.jsonl.gz'):
        a = json.loads(line)
        for k in ('seg_tokens', 'seg_wf', 'seg_logp'):
            if isinstance(a[k], str): a[k] = json.loads(a[k])
        out[a['gid']] = feats(a, T.get(a['gid']))
    os.makedirs(f'{HERE}/cache', exist_ok=True)
    pickle.dump(out, open(f'{HERE}/cache/vf_{S}.pkl', 'wb'))
    print(S, len(out), 'with true forks', len(T), flush=True)


if __name__ == '__main__':
    for S in (sys.argv[1:] or ['trainC', 'dev', 'dev2']): run(S)
