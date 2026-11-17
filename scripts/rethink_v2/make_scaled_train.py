"""Train-side analogue of GSM-Hard (GSM-Hard = GSM8K test with one number replaced by a large one, answer recomputed).
For each GSM8K TRAIN problem: pick one number of the question that is an operand of the solution's <<expr=res>>
annotations and occurs exactly once (by value) in the question; replace it with a random 5-7 digit integer; re-execute the
annotation chain, propagating changed intermediate results; the new gold = the re-executed final answer.
Skips problems whose annotations do not re-evaluate to their stated results, whose value roles are ambiguous (an
intermediate result equal to a question number or to another changed value), or whose answer does not change.
Never touches GSM8K test / GSM-Hard.  python scripts/rethink_v2/make_scaled_train.py data/gsm8k/train.jsonl OUT.jsonl"""
import json, re, sys, random
NUM = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])')
ANN = re.compile(r'<<([^<>=]+)=([^<>]+)>>')
rng = random.Random(1234)


def fnum(s):
    return float(s.replace(',', ''))


def ev(expr):
    e = expr.replace('x', '*').replace('×', '*').replace('÷', '/').replace(',', '')
    if not re.fullmatch(r'[\d\.\s\+\-\*/\(\)]+', e): return None
    try: return float(eval(e))
    except Exception: return None


def close(a, b): return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def build(r):
    q, sol = r['question'], r['answer']
    gold = fnum(sol.split('####')[1].strip())
    anns = ANN.findall(sol)
    if not anns: return None
    for e, res in anns:
        v = ev(e)
        try: rv = fnum(res.strip().rstrip('.'))
        except ValueError: return None
        if v is None or not close(v, rv): return None
    qnums = [(m.start(), m.end(), m.group()) for m in NUM.finditer(q)]
    vals = [fnum(s) for _, _, s in qnums]
    operands = set()
    for e, _ in anns:
        for s in NUM.findall(e.replace('x', ' x ')): operands.add(fnum(s))
    cands = [k for k, v in enumerate(vals) if vals.count(v) == 1 and v in operands and v >= 2]
    rng.shuffle(cands)
    for k in cands:
        old = vals[k]; new = rng.randint(10 ** rng.randint(4, 6), 10 ** 7 - 1) * 1.0
        mp = {old: new}; ok = True
        for e, res in anns:
            def sub(m):
                v = fnum(m.group())
                return repr(mp[v]) if v in mp else m.group().replace(',', '')
            e2 = NUM.sub(sub, e.replace('x', ' * ').replace('×', ' * '))
            v2 = ev(e2); rv = fnum(res.strip().rstrip('.'))
            if v2 is None: ok = False; break
            if not close(v2, rv):
                if (rv in mp and not close(mp[rv], v2)) or (rv in vals and rv != old): ok = False; break
                mp[rv] = v2
        if not ok or gold not in mp: continue
        ng = mp[gold]
        if close(ng, gold): continue
        s, t, _ = qnums[k]
        q2 = q[:s] + str(int(new)) + q[t:]
        ng = round(ng, 6)
        return {'src': r['_src'], 'question': q2, 'answer': f'#### {ng}', 'orig_answer': sol, 'scaled': [old, new]}
    return None


rows = [dict(json.loads(l), _src=k) for k, l in enumerate(open(sys.argv[1]))]
out = [b for b in map(build, rows) if b is not None]
with open(sys.argv[2], 'w') as f:
    for o in out: f.write(json.dumps(o) + '\n')
print(f'{len(out)}/{len(rows)} scaled problems')
