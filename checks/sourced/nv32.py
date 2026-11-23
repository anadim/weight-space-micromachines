"""(g) §4.4 'only 2-3% (on dev) are never solved in 32 tries; about a quarter of those have a wrong gold'.
Never solved = 0 correct among all independent single-attempt samples (SC rows: att_ok per sample) from the u0 host
(a2_u300_s2) B-stack, T=1. Items: dev600 (GSM8K train split B, shuffled Random(0), first 600) / dev2 (next 300).
Gold labels for the never-solved items: nv_gold_labels.json (criterion inside).
python3 checks/sourced/nv32.py SET FILE [FILE ...]   (FILEs = SC rows of fe_hook KIND SC, pooled per item; shipped:
  checks/sourced/res/fe/dev_sc{8,4}_s{7,11,13}.items.jsonl). Needs data/gsm8k/{train.jsonl,shots_v1.json}
  (python data/prepare_gsm.py) to rebuild the dev order and check the gold of every item."""
import json, sys, os, collections, random, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get('WSM_ROOT') or os.path.join(HERE, '..', '..')
SET, FILES = sys.argv[1], sys.argv[2:]
R = [[json.loads(l) for l in open(f)] for f in FILES]; n = len(R[0]); assert all(len(r) == n for r in R)
W = f'{ROOT}/data/gsm8k/'
part = lambda q: (lambda h: 'A' if h < 6 else ('B' if h < 8 else 'C'))(int(hashlib.md5(q.encode()).hexdigest(), 16) % 10)
sq = {s['question'] for s in json.load(open(W + 'shots_v1.json'))}
it = [r for r in (json.loads(l) for l in open(W + 'train.jsonl')) if part(r['question']) == 'B' and r['question'] not in sq]
random.Random(0).shuffle(it); it = it[:600] if SET == 'dev' else it[600:900] if SET == 'dev2' else None
if it: assert all(it[j]['answer'].split('####')[1].strip().replace(',', '') == R[0][j]['gold'] for j in range(n))
k = [sum(sum(r[j]['att_ok']) for r in R) for j in range(n)]; T = sum(len(r[0]['att_ok']) for r in R)
z = [j for j in range(n) if k[j] == 0]
print(f'{SET}: {n} items, {T} independent tries per item ({", ".join(os.path.basename(f) for f in FILES)})')
print(f'never solved: {len(z)} ({100*len(z)/n:.2f}%); solved in exactly 1: {sum(x == 1 for x in k)}; in <= 2: {sum(x <= 2 for x in k)}; '
      f'mean per-try acc {100*sum(k)/(n*T):.2f}%; majority acc {100*sum(r["ok"] for r in R[0])/n:.2f}% (first file)')
# fewer tries: never solved within the first t tries (pass@t complement), from the same samples
for t in sorted({1, 4, 8, 16, 24, 32, T} & set(range(1, T + 1))):
    zz = sum(1 for j in range(n) if not any(a for r in R for a in r[j]['att_ok'][:max(0, t - sum(len(q[0]['att_ok']) for q in R[:R.index(r)]))]))
    print(f'  never solved in the first {t:2d} tries: {zz} ({100*zz/n:.2f}%)')
LB = json.load(open(f'{HERE}/nv_gold_labels.json')).get(SET, {})
c = collections.Counter(LB.get(str(j), ['UNLABELED'])[0] for j in z)
print(f'gold labels of the {len(z)} never-solved: {dict(c)}; WRONG+AMBIG {c["WRONG"] + c["AMBIG"]} '
      f'({100*(c["WRONG"] + c["AMBIG"])/max(1, len(z)):.0f}%), +ILLPOSED {c["WRONG"] + c["AMBIG"] + c["ILLPOSED"]} '
      f'({100*(c["WRONG"] + c["AMBIG"] + c["ILLPOSED"])/max(1, len(z)):.0f}%)')
for j in z:
    a = collections.Counter(x for r in R for x in r[j]['answers'])
    lab = LB.get(str(j), ['UNLABELED', ''])
    print(f'- {SET} {j} gold {R[0][j]["gold"]} | {lab[0]} | top answers {a.most_common(3)}' + ('' if lab[0] != 'UNLABELED' else
          f'\n    Q: {it[j]["question"] if it else ""}\n    GOLD: {it[j]["answer"] if it else ""}'.replace('\n', '\n    ')))
