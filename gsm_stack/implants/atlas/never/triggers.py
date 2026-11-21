"""Offline trigger fire rates for candidate implants, on ALL dev/dev2 attempts (S3 stack, 32/item).
Fire rate on right attempts = what an implant would touch on already-right attempts (break exposure)."""
import gzip, json, re, sys, collections
sys.path.insert(0, 'atlas/never'); from labels import L
NUMRE = re.compile(r'(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s*/\s*(\d+))?(\s*%)?')
WORDS = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,
         'twice':2,'double':2,'doubles':2,'triple':3,'triples':3,'half':0.5,'thrice':3}
def nums(t):
    out = []
    for a, b, p in NUMRE.findall(t):
        x = float(a.replace(',', ''))
        if b and float(b) != 0: out.append(('frac', x, float(b)))
        elif p: out.append(('pct', x))
        else: out.append(('n', x))
    return out
def numset(t):
    s = set()
    for k in nums(t):
        if k[0] == 'frac': s |= {k[1], k[2], round(k[1]/k[2], 6)}
        elif k[0] == 'pct': s |= {k[1], round(k[1]/100, 6)}
        else: s.add(k[1])
    return s
def used(k, S):
    if k[0] == 'frac': return (k[1] in S and k[2] in S) or round(k[1]/k[2], 6) in S or round(k[1]/k[2], 2) in S
    if k[0] == 'pct': return k[1] in S or round(k[1]/100, 6) in S
    return k[1] in S
def qwords(q):
    return [w for w in re.findall(r"[a-z]+", q.lower()) if w in WORDS]
LEX = {  # unit / world-fact trigger lexicon (question side) -> fact an exact table would write
 'pair': r'\bpairs?\b', 'couple': r'\bcouples?\b', 'dozen': r'\bdozens?\b', 'horseshoe': r'horseshoe',
 'tire': r'\btires?\b', 'year/annual': r'\b(annual|yearly|per year|a year|each year|every year)\b', 'week': r'\bweeks?\b',
 'BC/born': r'\b(BC|born)\b', 'every $': r'for every \$', 'ft/in': r'\b(feet|foot)\b.*\binch|\binch.*\b(feet|foot)\b',
 'oz/qt/gal/l': r'\b(ounces?|quarts?|pints?|liters?|litres?)\b', 'cents/coins': r'\b(cents?|nickels?|dimes?|quarters)\b',
}
RT = r"\b(one way|and back|to and from|round trip|both ways|back home|there and back|morning and (the )?evening)\b"
ASK = re.compile(r'how (?:many|much) ([a-z]+)', re.I)

def run():
    R = collections.defaultdict(lambda: [0, 0])  # key -> [fires, n]
    per_item = {}
    lexq = collections.defaultdict(list)
    for s in ['dev', 'dev2']:
        items = {d['idx']: d for d in map(json.loads, gzip.open(f'atlas/data/items_{s}.jsonl.gz'))}
        for i, it in items.items():
            for k, rx in LEX.items():
                if re.search(rx, it['q'], re.I): lexq[k].append((s, i, it['n_right'], it.get('n_att', 32)))
            if re.search(RT, it['q'], re.I): lexq['ROUNDTRIP'].append((s, i, it['n_right'], it.get('n_att', 32)))
        for l in gzip.open(f'atlas/data/attempts_{s}.jsonl.gz'):
            a = json.loads(l)
            if a['ans'] is None: continue
            body = a['text'].split('####')[0]
            S = numset(body)
            qn = nums(a['q'])
            unused = [k for k in qn if not used(k, S)]
            ww = [w for w in qwords(a['q']) if w not in body.lower() and WORDS[w] not in S]
            ok = 'R' if a['ok'] else 'W'
            f1 = bool(unused); f2 = f1 or bool(ww)
            R[('unused_digit', ok)][0] += f1; R[('unused_digit', ok)][1] += 1
            R[('unused_digit+word', ok)][0] += f2; R[('unused_digit+word', ok)][1] += 1
            # ceil: non-integer final answer
            try: v = float(a['ans'].replace(',', ''))
            except Exception: v = None
            ni = v is not None and abs(v - round(v)) > 1e-9
            R[('nonint_answer', ok)][0] += ni; R[('nonint_answer', ok)][1] += 1
            # target noun register: ask noun absent from the last line before ####
            m = ASK.search(a['q'])
            if m:
                noun = m.group(1).lower().rstrip('s')[:5]
                lines = [x for x in body.strip().split('\n') if x.strip()]
                last = lines[-1].lower() if lines else ''
                fire = noun not in last and noun not in ('more', 'money', 'much', 'time', 'long', 'far', 'old', 'total')
                R[('target_noun_absent', ok)][0] += fire; R[('target_noun_absent', ok)][1] += 1
            # round trip: question has the phrase, attempt has no doubling
            if re.search(RT, a['q'], re.I):
                dbl = re.search(r'(\b2 ?[x*×] |[x*×] ?2\b|twice|round trip|both ways|back)', body)
                R[('roundtrip_no_x2', ok)][0] += not dbl; R[('roundtrip_no_x2', ok)][1] += 1
            key = (s, a['idx'])
            if key in L:
                d = per_item.setdefault(key, collections.Counter())
                d['W'] += not a['ok']; d['W_unused'] += (not a['ok']) and f1; d['W_unused_w'] += (not a['ok']) and f2
                d['R'] += a['ok']; d['R_unused_w'] += a['ok'] and f2
    return R, per_item, lexq

if __name__ == '__main__':
    R, per_item, lexq = run()
    out = {'fire': {f'{k[0]}|{k[1]}': [v[0], v[1], round(100 * v[0] / max(1, v[1]), 2)] for k, v in sorted(R.items())}}
    for k, v in sorted(R.items()): print(k, v, f'{100*v[0]/max(1,v[1]):.1f}%')
    print('\nlexicon (questions dev+dev2 = 900): n_q, attempt acc on those q, slice items (n_right<=4)')
    lx = {}
    for k, L_ in lexq.items():
        n = len(L_); acc = sum(x[2] for x in L_) / sum(x[3] for x in L_)
        sl = [(s, i) for s, i, r, _ in L_ if r <= 4]
        lx[k] = dict(n_q=n, att_acc=round(100 * acc, 1), always=sum(r == 32 for _, _, r, _ in L_), slice=sl)
        print(f'{k:12s} n_q={n:3d} att_acc={100*acc:.1f}% always={lx[k]["always"]} slice={sl}')
    out['lexicon'] = lx
    print('\nper-item unused-number fire on wrong attempts, by category')
    bycat = collections.defaultdict(lambda: collections.Counter())
    pi = {}
    for key, d in per_item.items():
        c = L[key][0]; bycat[c]['W'] += d['W']; bycat[c]['Wu'] += d['W_unused']; bycat[c]['Wuw'] += d['W_unused_w']
        bycat[c]['items'] += 1; bycat[c]['items_fire50'] += d['W_unused_w'] >= 0.5 * max(1, d['W'])
        pi[f'{key[0]}:{key[1]}'] = dict(W=d['W'], W_unused=d['W_unused'], W_unused_w=d['W_unused_w'], R=d['R'], R_unused_w=d['R_unused_w'])
    for c, d in bycat.items(): print(c, dict(d))
    out['per_item'] = pi; out['bycat'] = {c: dict(d) for c, d in bycat.items()}
    json.dump(out, open('atlas/never/triggers.json', 'w'), indent=1)
