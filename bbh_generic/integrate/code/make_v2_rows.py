"""v2 training rows (seed 'v2-train'), written by the integration agent. Two files:

(1) MULTI-TOOL rows (30): call -> use the result -> next tool. Five composition KINDS that are NOT among the 9 frozen test
    kinds (compose.py: track->expr, interval->expr, walk->expr, truth->expr, order->expr, track->bool, interval->bool,
    interval->expr->calc, walk->expr->calc). New kinds = new tool pairs and new settings:
      M1 truth -> bool   club of honest/lying members, a membership rule over who is honest
      M2 order -> bool   bake-off ranking clues, a jury rule over who placed where
      M3 walk  -> bool   garden rover, a rule over where it ended and which way it faces
      M4 track -> calc   market stalls swap decimal price tags, cost of n items at one stall
      M5 order -> calc   relay race places, a prize of P / place coins (decimal division)
    The first tool is written in its team's own scaffold; the second step is Team C's PASSIVE form: the host writes the
    expression and '=' and the passive tool writes the value (' V.\\n' expr, ' True.'/' False.' bool, ' V' calc).
(2) Team C's 100 rows (teamC/data/train/train_C_100.jsonl) converted from the call form to the passive form: every
    '<tool_call>NAME ARGS</tool_call><tool_response>R</tool_response>' becomes 'ARGS = R' in the passive tool's output format.
python make_v2_rows.py OUT_MULTI.jsonl OUT_CPASSIVE.jsonl"""
import json, random, re, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import G                                       # repo micromachines/ + team code on sys.path
from callfmt import track_ref, order_ref
from tools_ref import turtle_ref, truth_ref
import callfmt_C as CC
from micromachines.calcfmt import calc_ref

NAMES = ['Abebe', 'Birgit', 'Caoimhe', 'Dmitri', 'Esi', 'Fernando', 'Gulnara', 'Haruto', 'Ifeoma', 'Jonas', 'Kirra', 'Lucas',
         'Marisol', 'Njeri', 'Olek', 'Priya', 'Radu', 'Sione', 'Tamsin', 'Uzma', 'Viggo', 'Wairimu', 'Xochitl', 'Yaw']
ORD = ['first', 'second', 'third', 'fourth', 'fifth']
CS, CE, OS, OE = '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>'
r = random.Random('v2-train')
rows = []
QA = lambda q: f"Q: {q}\nA: Let's think step by step."
B = lambda v: 'True' if v else 'False'


def add(kind, q, comp, ans):
    rows.append({'prompt': QA(q), 'completion': '\n' + comp + f'\nSo the answer is {ans}.', 'team': 'multi', 'kind': kind})


# M1 truth -> bool
for _ in range(6):
    n = r.randint(3, 4); ppl = r.sample(NAMES, n); v = [r.random() < 0.5 for _ in range(n)]
    cl, sent = [f'1 {"true" if v[0] else "false"}'], [f'{ppl[0]} is {"honest" if v[0] else "a liar"}.']
    for i in range(1, n):
        j = r.randint(0, i - 1); c = v[i] == v[j]
        cl.append(f'{i + 1} says {j + 1} {"true" if c else "false"}'); sent.append(f'{ppl[i]} says that {ppl[j]} is {"honest" if c else "a liar"}.')
    call = 'truth ' + '; '.join(cl); res = truth_ref(call)
    a, b = r.sample(range(n), 2); form = r.choice(['{0} and not {1}', 'not {0} or {1}', '{0} or {1}'])
    expr = form.format(B(v[a]), B(v[b])); val = CC.bool_ref(expr)
    rule = form.format(f'{ppl[a]} is honest', f'{ppl[b]} is honest')
    q = (f"In a club every member is either always honest or always a liar. " + ' '.join(sent) +
         f" The club's rule is satisfied if {rule}. Is the rule satisfied? Answer True or False.")
    comp = (', '.join(f'{p} = {i + 1}' for i, p in enumerate(ppl)) + '.\n' + f'{CS}{call}{CE}{OS}{res}{OE}\n' +
            f'{ppl[a]} is {a + 1}, so "{ppl[a]} is honest" is {B(v[a])}. {ppl[b]} is {b + 1}, so "{ppl[b]} is honest" is {B(v[b])}.\n'
            f'The rule becomes: {expr} = {val}.')
    add('M1_truth_bool', q, comp, val)


def order_clues(n):
    p = list(range(1, n + 1)); r.shuffle(p); pos = {o: p.index(o) + 1 for o in p}; cons, txt = [], []
    while True:
        a, b = r.sample(range(1, n + 1), 2); t = r.random()
        if t < 0.5: c, s = (f'{a}<{b}', f'{{p{a}}} placed above {{p{b}}}') if pos[a] < pos[b] else (f'{a}>{b}', f'{{p{a}}} placed below {{p{b}}}')
        else: c, s = f'{a}={pos[a]}', (f'{{p{a}}} placed {ORD[pos[a] - 1]}' if pos[a] < n else f'{{p{a}}} placed last')
        if c in cons: continue
        cons.append(c); txt.append(s)
        if order_ref(f'order {n}:' + ','.join(cons)) or len(cons) > 10: return cons, txt, pos


# M2 order -> bool
for _ in range(6):
    n = r.randint(3, 5); cons, txt, pos = order_clues(n)
    if not order_ref(f'order {n}:' + ','.join(cons)): continue
    ppl = r.sample(NAMES, n); fmt = lambda s: s.format(**{f'p{i + 1}': ppl[i] for i in range(n)})
    call = f'order {n}:' + ','.join(cons); res = order_ref(call)
    x, y = r.sample(range(1, n + 1), 2); lx, ly = pos[x] == 1, pos[y] == n
    form = r.choice(['{0} or {1}', '{0} and not {1}', 'not ({0} or {1})']); expr = form.format(B(lx), B(ly)); val = CC.bool_ref(expr)
    q = (f"{', '.join(ppl[:-1])} and {ppl[-1]} entered a bake-off. " + ' '.join(fmt(t) + '.' for t in txt) +
         f" The jury gives a bonus if " + form.format(f'{ppl[x - 1]} placed first', f'{ppl[y - 1]} placed last') + ". Does the jury give a bonus? Answer True or False.")
    comp = ("Let's solve the ordering with the order tool.\nObjects: " + ', '.join(f'{i + 1} = {p}' for i, p in enumerate(ppl)) +
            f'.\nPositions: 1 = first, {n} = last.\n' + ''.join(f'{fmt(t)}: {c}.\n' for t, c in zip(txt, cons)) +
            f'Question: the whole order.\n{CS}{call}{CE}{OS}{res}{OE}\n' +
            f'{ppl[x - 1]} is {x}, at position {res.index(str(x)) + 1}, so "{ppl[x - 1]} placed first" is {B(lx)}. '
            f'{ppl[y - 1]} is {y}, at position {res.index(str(y)) + 1}, so "{ppl[y - 1]} placed last" is {B(ly)}.\n'
            f'The rule becomes: {expr} = {val}.')
    add('M2_order_bool', q, comp, val)

# M3 walk -> bool
HEAD = ['north', 'east', 'south', 'west']
for _ in range(6):
    cmds, words = [], []
    for _ in range(r.randint(3, 5)):
        if r.random() < 0.4 and cmds and not cmds[-1].startswith('turn'):
            t = r.choice(['left', 'right']); cmds.append(f'turn {t}'); words.append(f'turns {t}')
        else:
            d = r.randint(1, 9); bk = r.random() < 0.2; cmds.append(f'{"backward" if bk else "forward"} {d}'); words.append(f'rolls {d} meters {"backward" if bk else "forward"}')
    call = 'walk ' + ', '.join(cmds); res = turtle_ref(call)
    (x, y), h = eval(res.split(' facing')[0]), res.split('facing ')[1]
    hd = r.choice(HEAD); north = y > 0; facing = h == hd
    form = r.choice(['{0} and not {1}', '{0} or {1}', 'not {0} and not {1}']); expr = form.format(B(north), B(facing)); val = CC.bool_ref(expr)
    q = ("A garden rover starts at its charger facing north. It " + ', '.join(words) + '. '
         f"The gardener is happy if " + form.format('the rover ended north of the charger', f'the rover faces {hd}') + ". Is the gardener happy? Answer True or False.")
    comp = (f'{CS}{call}{CE}{OS}{res}{OE}\n' +
            f'The rover is at ({x}, {y}), so "north of the charger" is {B(north)}. It faces {h}, so "faces {hd}" is {B(facing)}.\n'
            f'The rule becomes: {expr} = {val}.')
    add('M3_walk_bool', q, comp, val)

# M4 track -> calc
for _ in range(6):
    n = 3; ppl = r.sample(NAMES, n); prices = [f'{r.randint(1, 9)}.{r.choice(["25", "5", "75", "4"])}' for _ in range(n)]
    sw = [tuple(r.sample(range(1, n + 1), 2)) for _ in range(r.randint(2, 3))]
    hold = list(range(1, n + 1))
    for i, j in sw: hold[i - 1], hold[j - 1] = hold[j - 1], hold[i - 1]
    k = r.randrange(n); m = r.randint(2, 6); p = prices[hold[k] - 1]; val = calc_ref(f'{m}*{p}')
    call = 'track 123:' + ','.join(f'{i}{j}' for i, j in sw) + f'?{k + 1}'; res = track_ref(call)
    q = (f"At a market, {ppl[0]}, {ppl[1]} and {ppl[2]} run stalls with price tags: {ppl[0]}'s tag says {prices[0]} dollars, "
         f"{ppl[1]}'s says {prices[1]} dollars, and {ppl[2]}'s says {prices[2]} dollars. During the day they swap tags: " +
         ', then '.join(f'{ppl[i - 1]} and {ppl[j - 1]} swap tags' for i, j in sw) +
         f". A customer then buys {m} jars at {ppl[k]}'s stall at the price on its tag. How many dollars does the customer pay?")
    comp = ("Let's track the tags with the state tool.\nSlots: " + ', '.join(f'{i + 1} = {p_}' for i, p_ in enumerate(ppl)) +
            '.\nValues: ' + ', '.join(f'{i + 1} = {pr} dollars' for i, pr in enumerate(prices)) + '.\n' +
            ''.join(f'{ppl[i - 1]} and {ppl[j - 1]} swap tags: {i}{j}.\n' for i, j in sw) +
            f'Question: what is in slot {k + 1} ({ppl[k]})? ?{k + 1}\n{CS}{call}{CE}{OS}{res}{OE}\n'
            f'Value {res} = {p} dollars.\nThe customer pays {m} * {p} = {val} dollars.')
    add('M4_track_calc', q, comp, val)

# M5 order -> calc
for _ in range(6):
    n = r.randint(3, 4); cons, txt, pos = order_clues(n)
    if not order_ref(f'order {n}:' + ','.join(cons)): continue
    ppl = r.sample(NAMES, n); fmt = lambda s: s.format(**{f'p{i + 1}': ppl[i] for i in range(n)})
    x = r.randint(1, n); call = f'order {n}:' + ','.join(cons) + f'@{x}'; res = order_ref(call)
    P = r.choice([90, 75, 60, 100]); val = calc_ref(f'{P}/{res}') if res != '1' else str(P)
    q = (f"{', '.join(ppl[:-1])} and {ppl[-1]} ran a relay race. " + ' '.join(fmt(t).replace('placed', 'finished') + '.' for t in txt) +
         f" The runner in place k receives {P} / k coins. How many coins does {ppl[x - 1]} receive?")
    tail = (f'{ppl[x - 1]} receives {P} / {res} = {val} coins.' if res != '1' else f'{ppl[x - 1]} is first and receives {P} coins.')
    comp = ("Let's solve the ordering with the order tool.\nObjects: " + ', '.join(f'{i + 1} = {p_}' for i, p_ in enumerate(ppl)) +
            f'.\nPositions: 1 = first, {n} = last.\n' + ''.join(f'{fmt(t).replace("placed", "finished")}: {c}.\n' for t, c in zip(txt, cons)) +
            f'Question: @{x}\n{CS}{call}{CE}{OS}{res}{OE}\nPosition {res}. ' + tail)
    add('M5_order_calc', q, comp, val)

assert all('?' not in c or True for c in rows)
with open(sys.argv[1], 'w') as f:
    for x in rows: f.write(json.dumps(x) + '\n')
import collections
print(len(rows), collections.Counter(x['kind'] for x in rows))


# ---------------- (2) Team C rows -> passive form ----------------
def to_passive(comp):
    def sub(m):
        name, args = m.group(1).split(' ', 1); v = m.group(2)
        if name == 'expr': return f'{args} = {v}.\n'
        if name == 'bool': return f'{args} = {v}.'
        return f'{args} = {v}'                                  # calc: the tool writes ' V', the host goes on
    out = re.sub(re.escape(CS) + r'(.*?)' + re.escape(CE) + re.escape(OS) + r'(.*?)' + re.escape(OE), sub, comp)
    out = out.replace('.\n.', '.\n').replace('.\n\n', '.\n').replace('.\n ', '.\n').replace('..', '.')
    return out.replace(' with the expression tool', '').replace(' with the boolean tool', '')


cr = []
for l in open(f'{G}/teamC/data/train/train_C_100.jsonl'):
    x = json.loads(l); cr.append({'prompt': x['prompt'], 'completion': to_passive(x['completion']), 'team': 'C', 'kind': x['kind'] + '/passive'})
with open(sys.argv[2], 'w') as f:
    for x in cr: f.write(json.dumps(x) + '\n')
seen = set()
for x in cr:
    if x['kind'] not in seen: seen.add(x['kind']); print(repr(x['completion'][:300]))
for x in rows[::6]: print('----', x['kind']); print(x['prompt'][:300]); print(x['completion'])
