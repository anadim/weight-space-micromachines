"""How often does the host state a calendar/unit fact ('There are N X in a Y') and get it wrong?"""
import json, re, collections
FACT = {('day','week'):7, ('weekday','week'):5, ('hour','day'):24, ('minute','hour'):60, ('second','minute'):60, ('month','year'):12,
        ('week','year'):52, ('day','year'):365, ('inch','foot'):12, ('feet','yard'):3, ('foot','yard'):3, ('ounce','pound'):16,
        ('cent','dollar'):100, ('quarter','dollar'):4, ('nickel','dollar'):20, ('dime','dollar'):10, ('egg','dozen'):12, ('item','dozen'):12,
        ('monday','week'):1, ('tuesday','week'):1, ('wednesday','week'):1, ('thursday','week'):1, ('friday','week'):1, ('saturday','week'):1, ('sunday','week'):1,
        ('weekend','week'):1, ('minute','day'):1440, ('hour','week'):168, ('day','fortnight'):14, ('week','fortnight'):2, ('year','decade'):10, ('year','century'):100,
        ('millimeter','centimeter'):10, ('centimeter','meter'):100, ('meter','kilometer'):1000, ('gram','kilogram'):1000, ('milliliter','liter'):1000, ('quart','gallon'):4, ('pint','quart'):2, ('cup','pint'):2}
R = re.compile(r'(\d[\d,]*)\s+([A-Za-z]+?)s?\s+in\s+(?:a|one|1|each|every)\s+([A-Za-z]+)', re.I)
def norm(w): w = w.lower(); return w[:-1] if w.endswith('s') and w not in ('hours',) else w
C = collections.Counter(); ex = []
def scan(txt, ok):
    for m in R.finditer(txt.replace('⟦','').replace('⟧','')):
        k = (norm(m.group(2)), norm(m.group(3)))
        if k in FACT:
            v = float(m.group(1).replace(',', ''))
            C['stated', ok] += 1
            if v != FACT[k]:
                C['wrong_fact', ok] += 1
                if len(ex) < 12: ex.append((ok, m.group(0)))
for l in open('dig/fail_data/a2s2_u300/train_full.jsonl'):
    r = json.loads(l)
    if r['kind'] != 'gsm': continue
    for s in [r['greedy']] + r['samples']: scan(s['txt'], s['ok'] > 0)
for d in ['c_s2', 'c_s2d3']:
    for x in map(json.loads, open(f'gsm_push/reading/out/{d}/off.jsonl')): scan(x['txt'], x['ok'] > 0)
print(dict(C)); [print(e) for e in ex]
