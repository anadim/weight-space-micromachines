"""Offline trigger precision on existing dumps (split-A train, RL+calc host): which cheap exact checks on the
finished answer flag wrong answers and not right ones? Run on greedy and on the 8 T=1 samples."""
import json, re, sys, collections
NUM = re.compile(r'(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)')
WORDS = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,
         'fifteen':15,'twenty':20,'thirty':30,'forty':40,'fifty':50,'hundred':100,'dozen':12,'half':0.5,'twice':2,'double':2,'triple':3,'thrice':3}
def nums(s, words=False):
    out = []
    for m in NUM.finditer(s):
        try: out.append(float(m.group(1).replace(',', '')))
        except: pass
    if words:
        for w in re.findall(r'[a-z]+', s.lower()):
            if w in WORDS: out.append(float(WORDS[w]))
    return out
def body(txt):
    t = txt.split('\n\n')[0].replace('⟦', '').replace('⟧', '')
    return t
def ans(t):
    m = re.search(r'####\s*(-?[\d,]*\.?\d+)', t)
    return float(m.group(1).replace(',', '')) if m else None
def feats(q, txt):
    t = body(txt); a = ans(t)
    pre = t.split('####')[0]
    qn = [x for x in nums(q) ]
    tn = nums(pre)
    eqres = [float(x.replace(',', '')) for x in re.findall(r'=\s*\$?(-?\d[\d,]*(?:\.\d+)?)', pre)]
    f = {}
    if a is None: return {'noans': 1}
    qs = set(qn)
    f['ans_is_qnum'] = a in qs
    f['ans_is_qnum_noeq'] = a in qs and a not in eqres
    unused = [x for x in qs if x not in tn and not (x in (1, 2) )]
    f['unused_qnum'] = len(unused) > 0
    f['unused_qnum_big'] = any(x not in tn for x in qs if x >= 3)
    f['ans_not_last_eq'] = bool(eqres) and a != eqres[-1]
    f['ans_not_any_eq'] = bool(eqres) and a not in eqres
    f['no_eq'] = not eqres
    f['few_eq'] = len(eqres) <= 1
    f['ans_lt1'] = 0 < a < 1
    return f
def run(path):
    C = collections.defaultdict(lambda: [0, 0, 0, 0])   # flagged_wrong, flagged_right, wrong, right
    for l in open(path):
        r = json.loads(l)
        if r['kind'] != 'gsm': continue
        for s in [r['greedy']] + r.get('samples', []):
            f = feats(r['q'], s['txt']); ok = s['ok'] > 0
            for k in ['ans_is_qnum','ans_is_qnum_noeq','unused_qnum','unused_qnum_big','ans_not_last_eq','ans_not_any_eq','no_eq','few_eq','ans_lt1','noans']:
                v = f.get(k, False)
                c = C[k]; c[2 + ok] += 1
                if v: c[ok] += 1
    for k, (fw, fr, w, rr) in C.items():
        print(f'{k:20s} flags wrong {fw}/{w} ({100*fw/max(1,w):.1f}%)  right {fr}/{rr} ({100*fr/max(1,rr):.2f}%)  precision {100*fw/max(1,fw+fr):.1f}%')
run(sys.argv[1])
