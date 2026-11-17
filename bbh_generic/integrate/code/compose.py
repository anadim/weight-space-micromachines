"""COMPOSITION TEST (protocol rule 14): NEW multi-tool questions, of kinds never seen in any team's dev/test/train.
Each question needs 2-3 DIFFERENT tools. Gold answers and gold intermediate results come from the teams' Python references
(callfmt.track_ref/order_ref, tools_ref.interval_ref/turtle_ref/truth_ref, callfmt_C.expr_ref/bool_ref/calc_ref).
Written by the integration agent (own names, own settings, own wording); no BBH data, no team generator is used.

Families (5 questions each, seed 'compose-0'):
  F1 track -> expr       raffle tickets traded, then arithmetic on the final ticket numbers
  F2 interval -> expr    total free hours of a studio
  F3 walk -> expr        mowing robot, shortest street distance back (|x| + |y|)
  F4 truth -> expr       quiz show: truthful/lying contestants -> total points
  F5 order -> expr       cycling race places -> prize coins
  F6 track -> bool       card passing -> does the team win (a logical rule over who holds what)
  F7 interval -> bool    room free at all? -> can the club meet (logical rule)
  F8 interval -> expr -> calc   free hours x a decimal hourly price
  F9 walk -> expr -> calc       drone: street distance home x decimal fuel per block
Per-tool acceptance ('accept'): the replies that are TRUE under the canonical numbering (people / objects / values numbered
in order of first mention, as every team's scaffold does). A valid but differently numbered call counts as incorrect.

python compose.py OUT.jsonl
"""
import json, random, sys, os, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _paths  # noqa: F401  (repo + team code on sys.path)
from callfmt import track_ref, order_ref
from tools_ref import interval_ref, turtle_ref, truth_ref
import callfmt_C as CC

NAMES = ['Aiyana', 'Bjorn', 'Chidi', 'Dalia', 'Emeka', 'Freya', 'Goran', 'Hana', 'Ines', 'Jiro', 'Kalani', 'Leif', 'Maren',
         'Nnamdi', 'Oona', 'Pita', 'Quilla', 'Rasmus', 'Saoirse', 'Tavita', 'Ulla', 'Vikram', 'Wanjiru', 'Ximena', 'Yusuf', 'Zeynep']
ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
TIMES = lambda h: f'{h % 12 or 12}{"am" if h < 12 else "pm"}'
r = random.Random('compose-0')
Q = []
cnt = lambda fam: sum(1 for x in Q if x['family'] == fam)


def add(fam, tools, q, ans, atype, accept, gold_calls):
    Q.append({'id': f'{fam}_{sum(1 for x in Q if x["family"] == fam)}', 'family': fam, 'tools': tools, 'question': q,
              'answer': str(ans), 'atype': atype, 'accept': accept, 'gold_calls': gold_calls})


def track_accept(n, ops, slots):
    st = ''.join(str(i) for i in range(1, n + 1)); base = f'track {st}:' + ','.join(ops)
    acc = {track_ref(base)} | {track_ref(base + f'?{k}') for k in slots}
    return base, sorted(a for a in acc if a)


def order_accept(n, cons, x):
    base = f'order {n}:' + ','.join(cons)
    acc = {order_ref(base), order_ref(base + f'@{x}')}
    return base, sorted(a for a in acc if a)


def swaps(n, k):
    out = []
    for _ in range(k):
        i, j = r.sample(range(1, n + 1), 2); out.append((i, j))
    return out


def apply(n, sw):
    hold = list(range(1, n + 1))                     # hold[slot-1] = value
    for i, j in sw: hold[i - 1], hold[j - 1] = hold[j - 1], hold[i - 1]
    return hold


SW = ['{a} and {b} trade tickets', '{a} swaps tickets with {b}', '{b} and {a} exchange their tickets']
# ---------------- F1 track -> expr ----------------
for i in range(200):
    if cnt('F1_track_expr') >= 5: break
    n = r.choice([3, 4]); ppl = r.sample(NAMES, n); nums = r.sample(range(10, 60), n); sw = swaps(n, r.randint(2, 4))
    hold = apply(n, sw); x, y = r.sample(range(n), 2); k = r.randint(2, 4)
    vx, vy = nums[hold[x] - 1], nums[hold[y] - 1]; ans = vx + k * vy
    st = ', '.join(f'{ppl[i]} has ticket number {nums[i]}' for i in range(n - 1)) + f', and {ppl[-1]} has ticket number {nums[-1]}'
    steps = [r.choice(SW).format(a=ppl[i - 1], b=ppl[j - 1]) for i, j in sw]
    story = 'First ' + steps[0] + ''.join(f', then {s}' for s in steps[1:-1]) + (f', and finally {steps[-1]}' if len(steps) > 1 else '')
    q = (f"At a raffle, {', '.join(ppl[:-1])} and {ppl[-1]} each hold one ticket: {st}. During the evening they trade tickets. "
         f"{story}. At the end, what is the number on {ppl[x]}'s ticket plus {k} times the number on {ppl[y]}'s ticket?")
    base, acc = track_accept(n, [f'{i}{j}' for i, j in sw], [x + 1, y + 1])
    add('F1_track_expr', ['track', 'expr'], q, ans, 'num', {'track': acc, 'expr': [str(ans)]}, [base, f'expr {vx} + {k} * {vy}'])


def busy_set(lo, hi, m):
    out = []
    for _ in range(m):
        s = r.randint(lo, hi - 1); e = min(hi, s + r.randint(1, 3)); out.append((s, e))
    return out


def free_of(lo, hi, busy):
    fr = [h for h in range(lo, hi) if not any(s <= h < e for s, e in busy)]
    return len(fr)


# ---------------- F2 interval -> expr ----------------
PLACES2 = ['pottery studio', 'climbing gym', 'community kitchen', 'rehearsal room', 'print shop']
for i in range(200):
    if cnt('F2_interval_expr') >= 5: break
    lo = r.randint(7, 10); hi = lo + r.randint(7, 10); busy = busy_set(lo, hi, r.randint(2, 4))
    call = f'interval {TIMES(lo)}-{TIMES(hi)} busy ' + ', '.join(f'{TIMES(s)}-{TIMES(e)}' for s, e in busy)
    res = interval_ref(call); tot = free_of(lo, hi, busy)
    if res in ('none', '') or tot == 0: continue
    q = (f"The {PLACES2[cnt('F2_interval_expr')]} opens at {TIMES(lo)} and closes at {TIMES(hi)}. Today it is reserved " +
         ', '.join(f'from {TIMES(s)} to {TIMES(e)}' for s, e in busy[:-1]) + f' and from {TIMES(busy[-1][0])} to {TIMES(busy[-1][1])}. '
         f"For how many hours in total is the {PLACES2[cnt('F2_interval_expr')]} not reserved today?")
    pieces = [p.split('-') for p in res.split(', ')]
    add('F2_interval_expr', ['interval', 'expr'], q, tot, 'num', {'interval': [res], 'expr': [str(tot)]}, [call, 'expr (hours of each free piece summed)'])


def walk_story(r, unit, verb):
    cmds, words = [], []
    for _ in range(r.randint(3, 6)):
        if r.random() < 0.35 and cmds and not cmds[-1].startswith('turn'):
            t = r.choice(['left', 'right']); cmds.append(f'turn {t}'); words.append(f'turns {t}')
        else:
            d = r.randint(1, 12); bk = r.random() < 0.25
            cmds.append(f'{"backward" if bk else "forward"} {d}'); words.append(f'{verb} {d} {unit} {"backward" if bk else "forward"}')
    if cmds[-1].startswith('turn'): cmds.pop(); words.pop()
    if not any(c.startswith('turn') for c in cmds): return walk_story(r, unit, verb)
    return cmds, words


# ---------------- F3 walk -> expr ----------------
for i in range(200):
    if cnt('F3_walk_expr') >= 5: break
    cmds, words = walk_story(r, 'meters', 'moves')
    call = 'walk ' + ', '.join(cmds); res = turtle_ref(call)
    x, y = eval(res.split(' facing')[0]); d = abs(x) + abs(y)
    if d < 5: continue
    q = ("A lawn-mowing robot starts at its shed facing north. It " + ', '.join(words[:-1]) + f', and then {words[-1]}. '
         "Driving only north-south and east-west, how many meters is the shortest way from the robot back to the shed?")
    add('F3_walk_expr', ['walk', 'expr'], q, d, 'num', {'walk': [res.split(' facing')[0]], 'expr': [str(d)]}, [call, f'expr {abs(x)} + {abs(y)}'])

# ---------------- F4 truth -> expr ----------------
for i in range(200):
    if cnt('F4_truth_expr') >= 5: break
    n = r.randint(3, 5); ppl = r.sample(NAMES, n); v = [r.random() < 0.5 for _ in range(n)]
    cl, sent = [f'1 {"true" if v[0] else "false"}'], [f'{ppl[0]} {"tells the truth" if v[0] else "lies"}.']
    for i in range(1, n):
        j = r.randint(0, i - 1); c = v[i] == v[j]                     # what i claims about j, consistent with v
        cl.append(f'{i + 1} says {j + 1} {"true" if c else "false"}')
        sent.append(f'{ppl[i]} says that {ppl[j]} {"tells the truth" if c else "lies"}.')
    call = 'truth ' + '; '.join(cl); res = truth_ref(call)
    pt, pl = r.choice([(4, 1), (5, 2), (3, 1), (6, 2)]); ans = sum(pt if x else pl for x in v)
    q = (f"In a quiz show, each of the contestants {', '.join(ppl[:-1])} and {ppl[-1]} either always tells the truth or always lies. "
         + ' '.join(sent) + f" Every truthful contestant wins {pt} points and every lying contestant wins {pl} point{'s' if pl > 1 else ''}. "
         "How many points do the contestants win in total?")
    add('F4_truth_expr', ['truth', 'expr'], q, ans, 'num', {'truth': [res], 'expr': [str(ans)]}, [call, 'expr ' + ' + '.join(str(pt if x else pl) for x in v)])


# ---------------- F5 order -> expr ----------------
def order_puzzle(n):
    p = list(range(1, n + 1)); r.shuffle(p)                  # p[k] = object at place k+1
    pos = {o: p.index(o) + 1 for o in p}
    cons, txt = [], []
    while True:
        a, b = r.sample(range(1, n + 1), 2); t = r.random()
        if t < 0.45:
            c = f'{a}<{b}' if pos[a] < pos[b] else f'{a}>{b}'; w = 'ahead of' if pos[a] < pos[b] else 'behind'
            s = f'{{p{a}}} finished {w} {{p{b}}}.'
        elif t < 0.65 and pos[a] + 1 == pos[b] or (t < 0.65 and pos[b] + 1 == pos[a]):
            if pos[b] + 1 == pos[a]: a, b = b, a
            c = f'{a}+{b}'; s = f'{{p{a}}} finished right before {{p{b}}}.'
        elif t < 0.85:
            c = f'{a}={pos[a]}'; s = f'{{p{a}}} finished {ORD[pos[a] - 1]}.' if pos[a] < n else f'{{p{a}}} finished last.'
        else:
            c = f'{a}!={(pos[a] % n) + 1}'; s = f'{{p{a}}} did not finish {ORD[pos[a] % n]}.'
        if c in cons: continue
        cons.append(c); txt.append(s)
        if order_ref(f'order {n}:' + ','.join(cons)): return cons, txt, pos
        if len(cons) > 12: return None


for i in range(200):
    if cnt('F5_order_expr') >= 5: break
    n = r.randint(4, 5)
    while True:
        z = order_puzzle(n)
        if z: break
    cons, txt, pos = z; ppl = r.sample(NAMES, n); x = r.randint(1, n)
    top, step = r.choice([(50, 10), (60, 12), (40, 8), (100, 15)]); ans = top - step * (pos[x] - 1)
    q = (f"{', '.join(ppl[:-1])} and {ppl[-1]} rode in a cycling race, and there were no ties. " +
         ' '.join(t.format(**{f'p{i + 1}': ppl[i] for i in range(n)}) for t in txt) +
         f" The winner receives {top} coins, and each later place receives {step} coins less than the place before it. "
         f"How many coins does {ppl[x - 1]} receive?")
    base, acc = order_accept(n, cons, x)
    add('F5_order_expr', ['order', 'expr'], q, ans, 'num', {'order': acc, 'expr': [str(ans)]}, [base + f'@{x}', f'expr {top} - {step} * {pos[x] - 1}'])

# ---------------- F6 track -> bool ----------------
COL = ['red', 'blue', 'green', 'yellow', 'white']
for i in range(200):
    if cnt('F6_track_bool') >= 5: break
    n = r.choice([3, 4]); ppl = r.sample(NAMES, n); cols = r.sample(COL, n); sw = swaps(n, r.randint(2, 4)); hold = apply(n, sw)
    lits, sl = [], []
    for _ in range(3):
        i = r.randrange(n); c = r.randrange(n); lits.append((f'{ppl[i]} holds the {cols[c]} card', hold[i] == c + 1)); sl.append(i + 1)
    form = r.choice(['({0} or {1}) and not ({2})', '{0} and not ({1} or {2})', 'not ({0}) or ({1} and {2})', '({0} and {1}) or {2}'])
    btxt = form.format(*[l for l, _ in lits]); bexpr = form.format(*[str(v) for _, v in lits]); ans = CC.bool_ref(bexpr)
    st = ', '.join(f'{ppl[i]} the {cols[i]} card' for i in range(n - 1)) + f', and {ppl[-1]} the {cols[-1]} card'
    steps = [f'{ppl[i - 1]} and {ppl[j - 1]} swap cards' for i, j in sw]
    q = (f"{', '.join(ppl[:-1])} and {ppl[-1]} each hold one card: {st}. Then " + ', then '.join(steps) + '. '
         f"The team wins if {btxt}. Does the team win? Answer True or False.")
    base, acc = track_accept(n, [f'{i}{j}' for i, j in sw], sl)
    add('F6_track_bool', ['track', 'bool'], q, ans, 'bool', {'track': acc, 'bool': [ans]}, [base, 'bool ' + bexpr])

# ---------------- F7 interval -> bool ----------------
PLACES7 = ['seminar room', 'school gym', 'church hall', 'meeting pod', 'roof terrace']
for i in range(200):
    if cnt('F7_interval_bool') >= 5: break
    lo = r.randint(8, 10); hi = lo + r.randint(5, 8)
    full = cnt('F7_interval_bool') in (0, 3)
    if full: busy = [(lo, lo + 3), (lo + 2, hi)]                 # fully booked
    else: busy = busy_set(lo, hi, r.randint(2, 3))
    call = f'interval {TIMES(lo)}-{TIMES(hi)} busy ' + ', '.join(f'{TIMES(s)}-{TIMES(e)}' for s, e in busy)
    res = interval_ref(call); free = res != 'none'
    if not res or (free == full): continue
    key, heat = r.random() < 0.5, r.random() < 0.5
    bexpr = f'{free} and ({key} or not {heat})'; ans = CC.bool_ref(bexpr)
    q = (f"The {PLACES7[cnt('F7_interval_bool')]} is open from {TIMES(lo)} to {TIMES(hi)} and is booked " +
         ' and '.join(f'from {TIMES(s)} to {TIMES(e)}' for s, e in busy) +
         ". The book club can meet only if the room has some free time today and (the coordinator has the key or not the heating is broken). "
         f"The coordinator {'has' if key else 'does not have'} the key, and the heating {'is' if heat else 'is not'} broken. "
         "Can the book club meet? Answer True or False.")
    add('F7_interval_bool', ['interval', 'bool'], q, ans, 'bool', {'interval': [res], 'bool': [ans]}, [call, 'bool ' + bexpr])

# ---------------- F8 interval -> expr -> calc ----------------
PLACES8 = ['recording booth', 'sauna', 'dance studio', 'tennis court', 'photo studio']
for i in range(200):
    if cnt('F8_interval_expr_calc') >= 5: break
    lo = r.randint(8, 11); hi = lo + r.randint(6, 10); busy = busy_set(lo, hi, r.randint(2, 3))
    call = f'interval {TIMES(lo)}-{TIMES(hi)} busy ' + ', '.join(f'{TIMES(s)}-{TIMES(e)}' for s, e in busy)
    res = interval_ref(call); tot = free_of(lo, hi, busy)
    if res in ('none', '') or tot == 0: continue
    price = r.choice(['12.5', '7.25', '9.5', '15.75', '4.5']); ans = CC.calc_ref(f'{tot} * {price}')
    q = (f"The {PLACES8[cnt('F8_interval_expr_calc')]} is open from {TIMES(lo)} to {TIMES(hi)} and is already booked " +
         ', '.join(f'from {TIMES(s)} to {TIMES(e)}' for s, e in busy[:-1]) + f' and from {TIMES(busy[-1][0])} to {TIMES(busy[-1][1])}. '
         f"A group rents every hour that is still free, at {price} dollars per hour. How many dollars do they pay?")
    add('F8_interval_expr_calc', ['interval', 'expr', 'calc'], q, ans, 'num', {'interval': [res], 'expr': [str(tot)], 'calc': [ans]},
        [call, 'expr (free hours summed)', f'calc {tot} * {price}'])

# ---------------- F9 walk -> expr -> calc ----------------
for i in range(200):
    if cnt('F9_walk_expr_calc') >= 5: break
    cmds, words = walk_story(r, 'blocks', 'flies')
    call = 'walk ' + ', '.join(cmds); res = turtle_ref(call)
    x, y = eval(res.split(' facing')[0]); d = abs(x) + abs(y)
    if d < 5: continue
    fuel = r.choice(['0.75', '1.5', '0.25', '2.5']); ans = CC.calc_ref(f'{d} * {fuel}')
    q = ("A delivery drone starts at the depot facing north. It " + ', '.join(words[:-1]) + f', and then {words[-1]}. '
         f"To fly home it may only fly north-south and east-west and takes the shortest such route; it uses {fuel} liters of fuel "
         "per block. How many liters of fuel does it need to fly home?")
    add('F9_walk_expr_calc', ['walk', 'expr', 'calc'], q, ans, 'num', {'walk': [res.split(' facing')[0]], 'expr': [str(d)], 'calc': [ans]},
        [call, f'expr {abs(x)} + {abs(y)}', f'calc {d} * {fuel}'])

with open(sys.argv[1], 'w') as f:
    for x in Q: f.write(json.dumps(x) + '\n')
import collections
print(len(Q), collections.Counter(x['family'] for x in Q))
for x in Q[::5]: print(json.dumps(x)[:700])
