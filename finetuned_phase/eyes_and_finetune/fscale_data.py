"""Focused scaling study: how many fine-tuning examples does each arm need?
Tool arm = learn the call format (machine computes); no-tool arm = same sequences, box in the loss (model computes).
5 tasks where the machine should matter: word_sorting, logical_deduction_seven_objects, dyck_languages (BBH prompts),
bignum arithmetic (3-6 digit operands, calculator), eyes (MNIST digits as pixel tokens + calculator).
Nested training subsets data/fs_{n}.jsonl (n per task, n in 8..8192) and dev data/fs_dev.jsonl (200 per task).
BBH test inputs are excluded by bbhfmt.make; eyes dev uses MNIST test images (train uses MNIST train images)."""
import json, random, ast, torch
from multiprocessing import Pool
import bbhfmt as B, eyes as EY
from calcfmt import calc_ref, CS, CE, OS, OE

SIZES = (8, 32, 128, 512, 2048, 8192)
BBH_T = ['word_sorting', 'logical_deduction_seven_objects', 'dyck_languages']
EXPR = ['a*b', 'a*b+c', 'a+b*c', 'a*b-c', '(a+b)*c', 'a-b*c', 'a*b+c*d']          # results stay within the calculator's 18 digits


def bignum(r):
    expr = r.choice(EXPR)
    def num():
        d = r.randint(3, 6); return r.randint(10 ** (d - 1), 10 ** d - 1)          # 3- to 6-digit operands
    v = {k: num() for k in 'abcd'}
    calls = []
    def ev(n):
        if isinstance(n, ast.Name): return str(v[n.id])
        l, rr = ev(n.left), ev(n.right); op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*'}[type(n.op)]
        e = f'{l}{op}{rr}'; o = calc_ref(e); calls.append((e, o)); return o
    val = ev(ast.parse(expr, mode='eval').body)
    txt = expr
    for k in 'abcd': txt = txt.replace(k, str(v[k]))
    q = f'What is {txt.replace("*", " * ").replace("+", " + ").replace("-", " - ")}?'
    body = ' '.join(f'{e} = {CS}{e}{CE}{OS}{o}{OE} {o}.' for e, o in calls)
    return {'task': 'bignum', 'prompt': f"Q: {q}\nA: Let's think step by step.", 'completion': f' {body} So the answer is {val}.', 'gold': val, 'question': q}


def gen(args):
    t, n, seed = args
    if t in BBH_T: return t, B.make(t, n, seed)
    r = random.Random(seed); out, seen = [], set()
    while len(out) < n:
        x = bignum(r)
        if x['question'] not in seen: seen.add(x['question']); out.append(x)
    return t, out


if __name__ == '__main__':
    N = max(SIZES)
    with Pool(8) as p:
        tr = dict(p.map(gen, [(t, N, 300 + i) for i, t in enumerate(BBH_T + ['bignum'])]))
        dv = dict(p.map(gen, [(t, 400, 400 + i) for i, t in enumerate(BBH_T + ['bignum'])]))
    M = torch.load('data/mnist14q.pt')
    r = random.Random(77); tr['eyes'] = [EY.question(r, M['Xtr'], M['Ytr']) for _ in range(N)]
    r = random.Random(78); dv['eyes'] = [EY.question(r, M['Xte'], M['Yte']) for _ in range(200)]
    tasks = list(tr)
    for t in tasks:                                         # dev never overlaps train
        trq = {x['question'] for x in tr[t]}; dv[t] = [dict(x, task='fs_' + t) for x in dv[t] if x['question'] not in trq][:200]
    rng = random.Random(0)
    for n in SIZES:
        rows = [dict(x, task='fs_' + t) for t in tasks for x in tr[t][:n]]; rng.shuffle(rows)
        with open(f'data/fs_{n}.jsonl', 'w') as f:
            for x in rows: f.write(json.dumps(x) + '\n')
    with open('data/fs_dev.jsonl', 'w') as f:
        for t in tasks:
            for x in dv[t]: f.write(json.dumps(x) + '\n')
    print({t: len(dv[t]) for t in tasks}, 'dev'); print(tr['bignum'][0]['completion'])
