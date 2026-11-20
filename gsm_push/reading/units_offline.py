"""Offline: how often does the unit the question asks for ('how many minutes ...') differ from the unit word attached to
the answer's last computed value (same dimension)? Precision/recall on split-A dumps."""
import json, re, sys, collections
DIM = {'time': ['second','minute','hour','day','week','month','year'], 'money': ['cent','dollar'],
       'len': ['inch','foot','feet','yard','mile','meter','centimeter','kilometer'], 'mass': ['ounce','pound','gram','kilogram','ton'],
       'vol': ['cup','pint','quart','gallon','liter','milliliter'], 'pct': ['percent','%']}
U = {u: d for d, us in DIM.items() for u in us}
def norm(w):
    w = w.lower().rstrip('s')
    if w == 'feet': return 'foot'
    if w in ('%',): return '%'
    return w if w in U or w == 'foot' else None
def asked(q):
    m = re.findall(r'(?:how many|how much|in)\s+([a-z%]+)', q.lower())
    for w in reversed(m):
        n = norm(w)
        if n: return n
    if '$' in q and re.search(r'how much (money|did|does|will|would|is|are|was)', q.lower()): return 'dollar'
    return None
def lastunit(txt):
    t = txt.split('\n\n')[0].split('####')[0]
    m = re.findall(r'=⟦?\s*\$?-?[\d,.]+\s*([a-zA-Z%]+)', t)
    for w in reversed(m):
        n = norm(w)
        if n: return n
    return None
C = collections.Counter(); ex = []
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r['kind'] != 'gsm': continue
    a = asked(r['q'])
    for j, s in enumerate([r['greedy']] + r['samples']):
        u = lastunit(s['txt']); ok = s['ok'] > 0
        C['n', ok] += 1
        if a: C['asked', ok] += 1
        if a and u and u != a and U.get(u) == U.get(a):
            C['mismatch', ok] += 1
            if j == 0: ex.append((ok, a, u, r['q'][-90:]))
print(dict(C)); [print(e) for e in ex[:15]]
