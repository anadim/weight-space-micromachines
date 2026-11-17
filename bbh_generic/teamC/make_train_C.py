"""Training rows for the shared call-format fine-tune (protocol rule 14), Team C tools: expr, bool, calc.
Seed 2 only (never dev seed 0 / test seed 1 / out-of-style seed 3). No BBH data. Team C's reported numbers do NOT use a
fine-tuned host (the frozen-host pattern tools work out of the box); these rows are for the integration step.

Row: {"prompt": "<optional own few-shot prefix>Q: ...\nA: Let's think step by step.", "completion": "\n... <tool_call>NAME ARGS</tool_call><tool_response>R</tool_response> ... So the answer is X."}
The tool's reply span is included (as in teamA/ft.py rows); the reference is callfmt_C.run_ref.

python make_train_C.py OUT.jsonl [N_PER_KIND=400]
"""
import json, random, sys, re
import gen_tasks as GT, gen_ood as GO, callfmt_C as CF

CALL = lambda a: f'{CF.CS}{a}{CF.CE}{CF.OS}{CF.run_ref(a)}{CF.OE}'


def arith(r):
    q = GT.gen_arith(r, 'dev'); e = q[:-2]
    v = CF.run_ref('expr ' + e)
    if not v: return None
    if r.random() < 0.5:
        body = f'I evaluate the whole expression with the expression tool.\n{CALL("expr " + e)}\nSo the answer is {v}.'
    else:                                                   # group by group, then combine
        m = re.fullmatch(r'\((\(.*?\)) ([+\-*]) (\(.*?\))\)', e)
        if not m: return None
        A, op, B = m.groups(); a, b = CF.run_ref('expr ' + A + ' + 0'), CF.run_ref('expr ' + B + ' + 0')
        if not a or not b: return None
        body = (f'This equation can be written as "A {op} B", where A = {A} and B = {B}.\n'
                f'A: {CALL("expr " + A + " + 0")}\nB: {CALL("expr " + B + " + 0")}\n'
                f'Then A {op} B: {CALL(f"expr {a} {op} {b}")}\nSo the answer is {CF.run_ref(f"expr {a} {op} {b}")}.')
    return q, body


def boolean(r):
    q, t = GT.gen_bool(r, 'dev'); e = q[:-3]
    v = CF.run_ref('bool ' + e)
    if not v: return None
    return q, f'I evaluate the expression with the boolean tool.\n{CALL("bool " + e)}\nSo the answer is {v}.'


def counting(r):
    q, t = GT.gen_count(r, 'dev')
    cat = re.search(r'(fruits|vegetables|animals|musical instruments|tools|vehicles|items|things|objects)', q.split('.')[-1] if '?' in q else q).group(1)
    items = re.split(r', and |, | and ', re.sub(r'^(I have|I own|My collection contains|There are|Last week I got|I bought|In my garage I keep) ', '', q.split('.')[0]))
    members = {s for c, v in GT.CATS.items() if (c == cat or cat in ('items', 'things', 'objects')) for p in v for s in p}
    if cat in ('items', 'things', 'objects'): members |= {s for p in GT.FILLER for s in p}
    lines, qs = [], []
    for it in items:
        m = re.match(r'(a|an|two|three|four|five) (.+)', it.strip())
        if not m: continue
        n = {'a': 1, 'an': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}[m.group(1)]; name = m.group(2)
        if name in members: lines.append(f'- {name} ({n})'); qs.append(str(n))
    if not qs or ' '.join(qs) and str(sum(map(int, qs))) != t: return None
    s = ' + '.join(qs) + (' + 0' if len(qs) == 1 else '')
    return q, (f'We first identify the {cat} on the list and include their quantity in parentheses:\n' + '\n'.join(lines) +
               f'\nNow we add the numbers in parentheses: {CALL("expr " + s)}\nSo the answer is {t}.')


def shop(r):
    q, t = GO.shop(r)
    m = re.findall(r'(\d+) [a-z ]+? at (\d+) dollars each', q)
    cost = ' + '.join(f'{a} * {b}' for a, b in m)
    c = CF.run_ref('expr ' + cost)
    pay = re.search(r'pays with (\d+) dollars', q)
    body = f'The cost is {CALL("expr " + cost)}.'
    if pay: body += f' The change is {CALL(f"expr {pay.group(1)} - {c}")}.'
    return q, body + f' So the answer is {t}.'


def decimal_shop(r):
    name = r.choice(GO.NAMES); it = r.choice(GO.ITEMS); n = r.randint(2, 9); p = f'{r.randint(1, 30)}.{r.choice(["25", "5", "75", "99", "1"])}'
    v = CF.run_ref(f'calc {n} * {p}')
    q = f'{name} buys {n} {it[1]} at {p} dollars each. How much does {name} spend?'
    return q, f'The cost is {CALL(f"calc {n} * {p}")} dollars. So the answer is {v}.'


def bcode(r):
    q, t = GO.bcode(r)
    vals = dict(re.findall(r'(\w+) = (True|False)', q)); e = re.search(r'`(.*)`', q).group(1)
    sub = re.sub(r'\b(\w+)\b', lambda m: vals.get(m.group(1), m.group(1)), e)
    return q, f'Substituting the values: {CALL("bool " + sub)}. So the answer is {t}.'


KINDS = {'arith': arith, 'bool': boolean, 'count': counting, 'shop': shop, 'dshop': decimal_shop, 'bcode': bcode}

if __name__ == '__main__':
    out, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400
    r = random.Random('teamC-train-2'); rows = []
    for k, f in KINDS.items():
        got = 0
        while got < N:
            x = f(r)
            if not x: continue
            q, body = x
            assert CF.OS + CF.OE not in body, (k, body)                     # every call in the training rows is answerable
            rows.append({'prompt': f"Q: {q}\nA: Let's think step by step.", 'completion': '\n' + body, 'kind': k}); got += 1
    r.shuffle(rows)
    with open(out, 'w') as fh:
        for x in rows: fh.write(json.dumps(x) + '\n')
    print(len(rows), 'rows ->', out)
    for k in KINDS: print(next(x for x in rows if x['kind'] == k))
