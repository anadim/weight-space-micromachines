"""Extra fine-tuning data so the call habit is 'arithmetic -> call', not 'GSM8K word problem -> call'.
  A. bare arithmetic in several framings (single ops and short chains; large, decimal and negative numbers)
  B. GSM8K train problems under other prompt templates
  C. error recovery: a malformed call + empty box (both kept out of the loss), then the model writes the result itself
Output rows: {"prompt", "completion", "kind"}.  Also writes a held-out bare-arithmetic test with unseen phrasings."""
import json, random, re, sys
from calcfmt import calc_ref, convert_solution, CS, CE, OS, OE, ANN

r = random.Random(7)
call = lambda e: f'{CS}{e}{CE}{OS}{calc_ref(e)}{OE}'


def rint(maxd=9):
    d = r.choices(range(1, maxd + 1), weights=[4, 6, 6, 5, 4, 3, 2, 1.5, 1][:maxd])[0]
    return str(r.randrange(10 ** (d - 1) if d > 1 else 0, 10 ** d))


def rnum(neg=0.08, dec=0.2):
    s = rint()
    if r.random() < dec: s = rint(5) + '.' + str(r.randrange(1, 10 ** r.randint(1, 3))).rstrip('0').zfill(1)
    if s.endswith('.'): s += '5'
    return ('-' if r.random() < neg else '') + s


def pair(op):
    a, b = rnum(), rnum()
    if op == '/':
        if r.random() < 0.6:                                   # exact division
            b = rint(4).lstrip('0') or '7'; a = str(int(b) * int(rint(5)))
        if float(b) == 0: b = '4'
    return a, b


NL = {'+': ['{a} + {b}', '{a} plus {b}', 'the sum of {a} and {b}'], '-': ['{a} - {b}', '{a} minus {b}', 'the difference between {a} and {b}'],
      '*': ['{a} * {b}', '{a} times {b}', '{a} x {b}', 'the product of {a} and {b}', '{a} multiplied by {b}'], '/': ['{a} / {b}', '{a} divided by {b}', '{a} ÷ {b}']}
TRAIN_T = [('Question: What is {x}?\nAnswer:', ' '), ('Question: Compute {x}.\nAnswer:', ' '), ('Q: {x} = ?\nA:', ' '), ('Problem: Calculate {x}.\nSolution:', ' '),
           ('User: what is {x}?\nAssistant:', ' '), ('Calculate {x}.\n', ''), ('{x} =', ' '), ('Question: Find {x}.\nAnswer:', ' ')]
TEST_T = [('How much is {x}?\n', ''), ('Evaluate {x}.\nResult:', ' '), ('Human: can you work out {x} for me?\nAI:', ' '), ('Math: {x}\nAnswer:', ' ')]


def single(templates):
    op = r.choice('+-*/'); a, b = pair(op)
    e = f'{a}{op}{b}'; res = calc_ref(e)
    if res == '': return None
    t, sp = r.choice(templates)
    return {'prompt': t.format(x=r.choice(NL[op]).format(a=a, b=b)), 'completion': f'{sp}{e} = {call(e)}{res}\n#### {res}', 'kind': 'bare1', 'gold': res}


def chain(templates):
    """a op1 b op2 c with ordinary precedence, written with symbols."""
    ops = [r.choice('+-*') for _ in range(2)]; nums = [rint(6) for _ in range(3)]
    x = f'{nums[0]} {ops[0]} {nums[1]} {ops[1]} {nums[2]}'
    lines = []
    if ops[1] == '*' and ops[0] in '+-':
        e1 = f'{nums[1]}*{nums[2]}'; m = calc_ref(e1); e2 = f'{nums[0]}{ops[0]}{m}'
    else:
        e1 = f'{nums[0]}{ops[0]}{nums[1]}'; m = calc_ref(e1); e2 = f'{m}{ops[1]}{nums[2]}'
    res = calc_ref(e2)
    if m == '' or res == '': return None
    t, sp = r.choice(templates)
    return {'prompt': t.format(x=x), 'completion': f'{sp}{e1} = {call(e1)}{m}\n{e2} = {call(e2)}{res}\n#### {res}', 'kind': 'bare2', 'gold': res}


def gsm_templates(rows, n):
    T = [('Problem: {q}\nSolution:', ' '), ('Q: {q}\nA:', ' '), ('User: {q}\nAssistant:', ' '), ('{q}\n', ''), ('Solve step by step.\n{q}\nAnswer:', ' ')]
    out = []
    for ex in r.sample(rows, n):
        body, fin = convert_solution(ex['answer']); t, sp = r.choice(T)
        out.append({'prompt': t.format(q=ex['question']), 'completion': f'{sp}{body}\n#### {fin}', 'kind': 'template'})
    return out


def corrupt(e):
    m = re.fullmatch(r'(-?[\d.]+)([+\-*/])(-?[\d.]+)', e)
    if not m: return None
    a, op, b = m.groups(); c = []
    c.append(f'{a} {op} {b}')
    c.append(f'{a}{op}{b}=')
    c.append(f'{a}{op}{b}.' + ''.join(r.choice('0123456789') for _ in range(16)) if '.' not in b else f'{a}{op}{b}' + ''.join(r.choice('0123456789') for _ in range(14)))
    if len(a) > 3 and '.' not in a and a[0] != '-': c.append(f'{a[:-3]},{a[-3:]}{op}{b}')
    c.append(f'{a}{op}{b}{op}{r.randint(2, 9)}')
    c.append(f'({a}{op}{b})')
    c = [x for x in c if calc_ref(x) == '']
    return r.choice(c) if c else None


def recovery(rows, n):
    """One call per example is replaced by a malformed call and an empty box; the text after it is unchanged."""
    pat = re.compile(re.escape(CS) + '(.*?)' + re.escape(CE) + re.escape(OS) + '(.*?)' + re.escape(OE))
    out = []
    for ex in r.sample(rows, 4 * n):
        body, fin = convert_solution(ex['answer']); ms = list(pat.finditer(body))
        if not ms: continue
        m = r.choice(ms); bad = corrupt(m.group(1))
        if bad is None: continue
        body = body[:m.start()] + f'{CS}{bad}{CE}{OS}{OE}' + body[m.end():]
        out.append({'prompt': f"Question: {ex['question']}\nAnswer:", 'completion': f' {body}\n#### {fin}', 'kind': 'recovery'})
        if len(out) == n: break
    return out


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('data/train.jsonl')]
    random.Random(1234).shuffle(rows); train = rows[500:]                    # never touch the dev split
    aug = []
    while sum(x['kind'] == 'bare1' for x in aug) < 1200:
        x = single(TRAIN_T); aug += [x] if x else []
    while sum(x['kind'] == 'bare2' for x in aug) < 500:
        x = chain(TRAIN_T); aug += [x] if x else []
    aug += gsm_templates(train, 1000) + recovery(train, 400)
    r.shuffle(aug)
    with open('data/aug.jsonl', 'w') as f:
        for x in aug: f.write(json.dumps(x) + '\n')
    tst = []
    while len(tst) < 400:
        x = (single if len(tst) % 4 else chain)(TEST_T)
        if x: tst.append({'question': x['prompt'], 'answer': '#### ' + x['gold'], 'raw_prompt': True})
    with open('data/bare_test.jsonl', 'w') as f:
        for x in tst: f.write(json.dumps(x) + '\n')
    import collections
    print(collections.Counter(x['kind'] for x in aug), 'bare test', len(tst))
    for k in ('bare1', 'bare2', 'recovery'):
        x = next(v for v in aug if v['kind'] == k); print(f'--- {k}\n{x["prompt"]}{x["completion"][-330:]}')
