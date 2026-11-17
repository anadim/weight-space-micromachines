"""Generated train/dev data for BBH tasks, in micromachine call format.

Every row = {task, prompt, completion, gold}: prompt = the task's OFFICIAL 3-shot CoT prefix + 'Q: <question>\\nA: Let's think
step by step.' (the same prompt used at test time), completion = call-format reasoning + ' So the answer is X.'
Questions are generated from each task's definition with our own names/contents; BBH test inputs are never used, and
any generated question that equals a test input is dropped.
"""
import json, os, random, re, itertools
from calcfmt import calc_ref, CS, CE, OS, OE
from swapfmt import make_story, swap_ref, CS2, CE2, OS2, OE2
from boolfmt import gen_expr, bool_ref, CS4, CE4, OS4, OE4
from liefmt import truth_ref, solve, NAMES as LIE_NAMES, CS3, CE3, OS3, OE3
import circuits5 as C5, circuits6 as C6, circuits7 as C7, circuits8 as C8, circuits9 as C9, circuits10 as C10, circuits11 as C11

BBH = os.environ.get('BBH', 'data/bbh')
LG = ('<|logic|>', '<|/logic|>', '<|logic_out|>', '<|/logic_out|>')
DY = ('<|dyck|>', '<|/dyck|>', '<|dyck_out|>', '<|/dyck_out|>')
WK = ('<|walk|>', '<|/walk|>', '<|walk_out|>', '<|/walk_out|>')
TM = ('<|time|>', '<|/time|>', '<|time_out|>', '<|/time_out|>')
SO = ('<|sort|>', '<|/sort|>', '<|sort_out|>', '<|/sort_out|>')
DT = ('<|date|>', '<|/date|>', '<|date_out|>', '<|/date_out|>')
SH = ('<|shape|>', '<|/shape|>', '<|shape_out|>', '<|/shape_out|>')
NEW_TOKENS = list(LG) + list(DY) + list(WK) + list(TM) + list(SO) + list(DT) + list(SH)
EYE_TOKENS = ['<|img|>', '<|/img|>', '<|see|>', '<|/see|>', '<|see_out|>', '<|/see_out|>']   # MM12 eyes (appended: earlier ids unchanged)
NEW_TOKENS = NEW_TOKENS + EYE_TOKENS


def prefix(task):
    return open(f'{BBH}/cot/{task}.txt').read().split('-----', 1)[1].strip() + '\n\n'


def test_inputs(task):
    return {e['input'] for e in json.load(open(f'{BBH}/data/{task}.json'))['examples']}


def row(task, q, body, ans):
    return {'task': task, 'prompt': prefix(task) + f"Q: {q}\nA: Let's think step by step.", 'completion': f' {body} So the answer is {ans}.', 'gold': ans, 'question': q}


calc = lambda e: (f'{CS}{e}{CE}{OS}{calc_ref(e)}{OE}', calc_ref(e))


# ---------------------------------------------------------------- tracking_shuffled_objects (MM2)
def tracking(r, n_people):
    while True:
        story, people, items, swaps, ask = make_story(r)
        if len(people) == n_people: break
    box = lambda c, st: (lambda out: (f'{CS2}{c}{CE2}{OS2}{out}{OE2}', out))(swap_ref(c, st))
    b, st = box(str(n_people), None)
    lines = ['People: ' + ', '.join(f'{i + 1} {p}' for i, p in enumerate(people)) + '.', 'Start: ' + ', '.join(f'{i + 1} {it}' for i, it in enumerate(items)) + '. ' + b]
    for i, j in swaps:
        b, st = box(f'{i}{j}', st); lines.append(f'{people[i - 1]} and {people[j - 1]}: {b}')
    b, res = box(f'0{ask}', st); lines.append(f'{people[ask - 1]} is {ask}: {b}')
    letter = chr(64 + int(res)); lines.append(f'{res} is {items[int(res) - 1]}, option ({letter}).')
    name = {3: 'three', 5: 'five', 7: 'seven'}[n_people]
    return row(f'tracking_shuffled_objects_{name}_objects', story, '\n'.join(lines), f'({letter})')


# ---------------------------------------------------------------- boolean_expressions (MM3)
def boolean(r):
    e = ' '.join(gen_expr(r, r.choice([3, 4, 5, 6, 7, 8, 8, 8, 8, 9, 10, 11, 12])))
    v = bool_ref(e)
    return row('boolean_expressions', f'{e} is', f'{CS4}{e}{CE4}{OS4}{v}{OE4}', v)


# ---------------------------------------------------------------- web_of_lies (MM4)
def lies(r):
    k = r.randint(3, 7); ppl = r.sample(LIE_NAMES, k)
    sents = [f'{ppl[0]} {r.choice(["tells the truth", "lies"])}.'] + [f'{ppl[i]} says {ppl[i - 1]} {r.choice(["tells the truth", "lies"])}.' for i in range(1, k)]
    q = 'Question: ' + ' '.join(sents) + f' Does {ppl[-1]} tell the truth?'
    people, calls, st = solve(q); lines, cur = [], None
    for p, c in zip(people, calls):
        cur = truth_ref(c, cur); lines.append(f'{p}: {CS3}{c}{CE3}{OS3}{cur}{OE3}')
    ans = 'Yes' if st == '1' else 'No'
    return row('web_of_lies', q, '\n'.join(lines) + f'\n{people[-1]}: {st}, {ans}.', ans)


# ---------------------------------------------------------------- multistep_arithmetic_two (MM1 chains)
def _group(r):
    n = r.choice([3, 4, 4, 4]); nums = [r.randint(-9, 9) for _ in range(n)]; ops = [r.choice('+-*') for _ in range(n - 1)]
    return nums, ops


def _gtext(nums, ops):
    return '(' + ' '.join([str(nums[0])] + [f'{o} {x}' for o, x in zip(ops, nums[1:])]) + ')'


def _eval_group(nums, ops):
    """Multiplications first (left to right), then + and - left to right. -> (calls, value)"""
    terms, signs, calls = [[nums[0]]], ['+'], []
    for o, x in zip(ops, nums[1:]):
        if o == '*': terms[-1].append(x)
        else: terms.append([x]); signs.append(o)
    vals = []
    for t in terms:
        v = t[0]
        for x in t[1:]:
            e = f'{v}*{x}'; s, out = calc(e); calls.append(s); v = int(out)
        vals.append(v)
    acc = vals[0]
    for sg, v in zip(signs[1:], vals[1:]):
        e = f'{acc}{sg}{v}'; s, out = calc(e); calls.append(s); acc = int(out)
    return calls, acc


def multistep(r):
    a, b = _group(r), _group(r); op = r.choice('+-*')
    q = f'({_gtext(*a)} {op} {_gtext(*b)}) ='
    ca, va = _eval_group(*a); cb, vb = _eval_group(*b)
    s, out = calc(f'{va}{op}{vb}')
    body = (f'A = {_gtext(*a)}: ' + ' '.join(ca) + f' A = {va}.\nB = {_gtext(*b)}: ' + ' '.join(cb) + f' B = {vb}.\nA {op} B = {s}')
    return row('multistep_arithmetic_two', q, body, out)


# ---------------------------------------------------------------- object_counting (MM1 chains)
CATS = {
    'musical instruments': ['piano', 'flute', 'trombone', 'violin', 'accordion', 'clarinet', 'drum', 'trumpet', 'cello', 'harp', 'banjo', 'tuba', 'saxophone', 'guitar'],
    'fruits': ['apple', 'banana', 'strawberry', 'peach', 'orange', 'plum', 'raspberry', 'grape', 'nectarine', 'blackberry', 'mango', 'pear', 'kiwi', 'cherry'],
    'vegetables': ['cabbage', 'carrot', 'cauliflower', 'stalk of celery', 'garlic', 'lettuce head', 'onion', 'potato', 'yam', 'head of broccoli', 'zucchini', 'cucumber', 'eggplant', 'pepper'],
    'animals': ['bear', 'cat', 'chicken', 'cow', 'dog', 'donkey', 'duck', 'fish', 'frog', 'goat', 'mouse', 'pig', 'rabbit', 'snail', 'snake', 'horse', 'sheep', 'owl'],
    'objects': ['bed', 'car', 'chair', 'couch', 'fridge', 'lamp', 'microwave', 'oven', 'stove', 'table', 'toaster', 'bookshelf', 'mirror', 'sofa'],
}
PLURAL = {'stalk of celery': 'stalks of celery', 'lettuce head': 'lettuce heads', 'head of broccoli': 'heads of broccoli', 'mouse': 'mice', 'fish': 'fish',
          'sheep': 'sheep', 'peach': 'peaches', 'potato': 'potatoes', 'couch': 'couches', 'raspberry': 'raspberries', 'strawberry': 'strawberries',
          'blackberry': 'blackberries', 'cherry': 'cherries', 'mango': 'mangoes', 'bookshelf': 'bookshelves'}
WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}


def objcount(r):
    cat = r.choice(list(CATS)); others = [c for c in CATS if c != cat and c != 'objects']
    items = []
    pool = CATS[cat] if cat != 'objects' else CATS['objects']
    for it in r.sample(pool, r.randint(3, 8)): items.append((it, True))
    if cat != 'objects':
        oc = r.choice(others)
        for it in r.sample(CATS[oc], r.randint(0, 4)): items.append((it, False))
    r.shuffle(items)
    counted, phr = [], []
    for it, rel in items:
        n = r.choice([1, 1, 1, 1, 2, 2, 3, 4, 5])
        art = ('an ' if it[0] in 'aeiou' else 'a ') if n == 1 else WORDS[n] + ' '
        phr.append(art + (it if n == 1 else PLURAL.get(it, it + 's')))
        if rel: counted.append((it, n))
    q = 'I have ' + ', '.join(phr[:-1]) + ', and ' + phr[-1] + f'. How many {cat} do I have?'
    lines = [f'The {cat} are: ' + ', '.join(f'{it} ({n})' for it, n in counted) + '.']
    acc = counted[0][1]; calls = []
    for it, n in counted[1:]:
        s, out = calc(f'{acc}+{n}'); calls.append(s); acc = int(out)
    return row('object_counting', q, lines[0] + (' ' + ' '.join(calls) if calls else '') + f' Total {acc}.', str(acc))


# ---------------------------------------------------------------- logical_deduction (MM5)
ORD = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh'}
DOMAINS = [
    dict(intro='In a {ev}, there were {n} {kind}: {objs}.', ev=['chess tournament', 'spelling bee', 'cooking contest', 'marathon', 'dance competition'],
         kind='players', objs=['Mia', 'Leo', 'Ivy', 'Kai', 'Zoe', 'Max', 'Uma', 'Tom', 'Ada', 'Ben', 'Eli', 'Fay'], art='', name=lambda o: o,
         lt='{a} finished above {b}', gt='{a} finished below {b}', pos=lambda k, n: ORD[k], neg=lambda k, n: 'last' if k == 1 else f'{ORD[k]}-to-last',
         isat='{a} finished {p}', axis='Positions count from the top (first place = 1).'),
    dict(intro='On a shelf, there are {n} {kind}: {objs}.', kind='mugs', objs=['white', 'black', 'teal', 'pink', 'tan', 'navy', 'lime', 'rust', 'plum', 'gold'],
         art='a ', name=lambda o: f'{o} mug', lt='The {a} is to the left of the {b}', gt='The {a} is to the right of the {b}',
         pos=lambda k, n: 'leftmost' if k == 1 else f'{ORD[k]} from the left', neg=lambda k, n: 'rightmost' if k == 1 else f'{ORD[k]} from the right',
         isat='The {a} is the {p}', axis='Positions count from the left.'),
    dict(intro='On a fence, there are {n} {kind}: {objs}.', kind='birds', objs=['pigeon', 'sparrow', 'heron', 'finch', 'magpie', 'wren', 'swallow', 'parrot', 'eagle', 'gull'],
         art='a ', name=lambda o: o, lt='The {a} is to the left of the {b}', gt='The {a} is to the right of the {b}',
         pos=lambda k, n: 'leftmost' if k == 1 else f'{ORD[k]} from the left', neg=lambda k, n: 'rightmost' if k == 1 else f'{ORD[k]} from the right',
         isat='The {a} is the {p}', axis='Positions count from the left.'),
    dict(intro='In a parking lot, there are {n} {kind}: {objs}.', kind='cars', objs=['van', 'jeep', 'coupe', 'sedan', 'pickup', 'roadster', 'taxi', 'camper', 'wagon', 'cab'],
         art='a ', name=lambda o: o, lt='The {a} is older than the {b}', gt='The {a} is newer than the {b}',
         pos=lambda k, n: 'oldest' if k == 1 else f'{ORD[k]}-oldest', neg=lambda k, n: 'newest' if k == 1 else f'{ORD[k]}-newest',
         isat='The {a} is the {p}', axis='Positions count from the oldest.'),
    dict(intro='A market sells {n} {kind}: {objs}.', kind='vegetables', objs=['leeks', 'beets', 'radishes', 'turnips', 'squash', 'okra', 'kale', 'peas', 'corn', 'chard'],
         art='', name=lambda o: o, lt=('The {a} are cheaper than the {b}', 'The {a} are less expensive than the {b}'), gt='The {a} are more expensive than the {b}',
         pos=lambda k, n: 'cheapest' if k == 1 else f'{ORD[k]}-cheapest', neg=lambda k, n: 'most expensive' if k == 1 else f'{ORD[k]}-most expensive',
         isat='The {a} are the {p}', axis='Positions count from the cheapest.'),
]


def logical(r, n):
    D = r.choice(DOMAINS); objs = r.sample(D['objs'], n); names = [D['name'](o) for o in objs]
    perm = list(range(1, n + 1)); r.shuffle(perm); pos = {o: k + 1 for k, o in enumerate(perm)}
    cands = []
    for a in range(1, n + 1):
        cands.append(('=', a, pos[a])); cands.append(('=-', a, n + 1 - pos[a]))
        for b in range(1, n + 1):
            if a != b: cands.append(('<' if pos[a] < pos[b] else '>', a, b))
    alive = list(itertools.permutations(range(1, n + 1)))         # orderings still consistent with the statements so far
    holds = lambda p, c: (lambda P: (P[c[1]] < P[c[2]]) if c[0] == '<' else (P[c[1]] > P[c[2]]) if c[0] == '>' else
                          (P[c[1]] == c[2]) if c[0] == '=' else (P[c[1]] == n + 1 - c[2]))({o: i + 1 for i, o in enumerate(p)})
    cands = [c for c in cands if not (c[0] in ('=', '=-') and c[2] > 3 and r.random() < 0.6)]   # mostly near-the-end positions, like BBH
    cons = []
    while len(alive) > 1:                                          # pick a fairly informative statement each time
        pool = r.sample(cands, min(6, len(cands)))
        scored = sorted(pool, key=lambda c: sum(holds(p, c) for p in alive))
        c = scored[0] if r.random() < 0.7 else r.choice(scored[:3])
        nxt = [p for p in alive if holds(p, c)]
        if len(nxt) < len(alive): cons.append(c); alive = nxt
        cands.remove(c)
    k = r.randint(1, n); use_neg = r.random() < 0.4
    q_phr = D['neg'](n + 1 - k, n) if use_neg else D['pos'](k, n)
    qcall = f'?-{n + 1 - k}' if use_neg else f'?{k}'

    def sent(c):
        op, a, b = c
        if op in '<>':
            tpl = D['lt'] if op == '<' else D['gt']
            if isinstance(tpl, tuple): tpl = tpl[(a * 7 + b) % len(tpl)]
            return tpl.format(a=names[a - 1], b=names[b - 1])
        p = D['pos'](b, n) if op == '=' else D['neg'](b, n)
        return D['isat'].format(a=names[a - 1], p=p)
    ev = r.choice(D['ev']) if 'ev' in D else ''
    objtxt = ', '.join((D['art'] + names[i] if D['art'] else names[i]) for i in range(n - 1)) + ', and ' + (D['art'] + names[-1] if D['art'] else names[-1])
    intro = D['intro'].format(ev=ev, n={3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven'}[n], kind=D['kind'], objs=objtxt)
    stm = ' '.join(sent(c) + '.' for c in cons)
    opts = '\n'.join(f'({chr(65 + i)}) ' + D['isat'].format(a=names[i], p=q_phr) for i in range(n))
    q = ('The following paragraphs each describe a set of {} objects arranged in a fixed order. The statements are logically consistent within each paragraph. '
         .format({3: 'three', 5: 'five', 7: 'seven'}.get(n, str(n))) + intro + ' ' + stm + '\nOptions:\n' + opts)
    call = f'{n}:' + ''.join(f'{a}{op}{b};' for op, a, b in cons) + qcall
    out = C5.logic_ref(call); assert out
    lines = ['Objects: ' + ', '.join(f'{i + 1} {nm}' for i, nm in enumerate(names)) + '. ' + D['axis']]
    for c in cons: lines.append(f'{sent(c)}: {c[1]}{c[0]}{c[2]}')
    lines.append(f'Asked: the {q_phr}: {qcall}')
    lines.append(f'{LG[0]}{call}{LG[1]}{LG[2]}{out}{LG[3]}')
    letter = chr(64 + int(out)); disp = names[int(out) - 1] if D['kind'] == 'players' else 'the ' + names[int(out) - 1]
    lines.append(f'{out} is {disp}, option ({letter}).')
    name = {3: 'three', 5: 'five', 7: 'seven'}[n]
    return row(f'logical_deduction_{name}_objects', q, '\n'.join(lines), f'({letter})')


# ---------------------------------------------------------------- dyck_languages (MM6)
def dyck(r):
    """Random walk with a chosen number u of unclosed brackets at the end (u = answer length, 1..10)."""
    while True:
        u = r.choice([1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10])
        L = r.choice([r.randint(u, 14), r.randint(10, 40), r.randint(30, 90)])
        if (L - u) % 2: L += 1
        st, out = [], []
        for k in range(L):
            rem = L - k - 1
            can_open = len(st) + 1 - u <= rem
            can_close = bool(st) and u - (len(st) - 1) <= rem
            if can_close and (not can_open or r.random() < 0.45):
                out.append(C6.CLOSE[C6.OPEN.index(st.pop())])
            else:
                c = r.choice(C6.OPEN); st.append(c); out.append(c)
        if len(st) == u and len(out) <= 100: break
    seq = ' '.join(out); clo = C6.dyck_ref(seq)
    assert len(clo) == u
    q = f'Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: {seq}'
    ans = ' '.join(clo)
    return row('dyck_languages', q, f'{DY[0]}{seq}{DY[1]}{DY[2]} {ans}{DY[3]}', ans)


# ---------------------------------------------------------------- navigate (MM7)
def navigate(r):
    face_fwd = r.random() < 0.5; its, sents = [], []
    n = r.randint(3, 9)
    for _ in range(n):
        if not face_fwd and r.random() < 0.35:
            t = r.choice(['TL', 'TR', 'TA']); its.append(t); sents.append({'TL': 'Turn left.', 'TR': 'Turn right.', 'TA': 'Turn around.'}[t]); continue
        k = r.randint(1, 10)
        if face_fwd: d = r.choice('FBLR')
        else: d = r.choice('FFFFBLR')
        its.append(f'{d}{k}')
        word = {'F': 'forward', 'B': 'backward', 'L': 'left', 'R': 'right'}[d]
        sents.append(f'Take {k} step{"s" if k > 1 else ""}' + ('' if (not face_fwd and d == 'F' and r.random() < 0.8) else f' {word}') + '.')
    if r.random() < 0.5 and C7.walk_ref(';'.join(its)) == 'No':        # make about half return to the start
        h, x, y = 0, 0, 0
        for it in its:
            if it[0] == 'T': h = (h + {'TR': 1, 'TL': 3, 'TA': 2}[it]) % 4; continue
            d = (C7.REL[it[0]] + h) % 4; k = int(it[1:]); x += k if d == 1 else -k if d == 3 else 0; y += k if d == 0 else -k if d == 2 else 0
        for need, ax in ((-y, 0), (-x, 1)):
            if need:
                absd = ax if need > 0 else (ax + 2) % 4; rel = (absd - h) % 4; d = {0: 'F', 1: 'R', 2: 'B', 3: 'L'}[rel]
                if face_fwd or d == 'F':
                    k = abs(need); its.append(f'{d}{k}'); word = {'F': 'forward', 'B': 'backward', 'L': 'left', 'R': 'right'}[d]
                    sents.append(f'Take {k} step{"s" if k > 1 else ""}' + ('' if (not face_fwd and d == 'F') else f' {word}') + '.')
                else:                                                      # turn to face it, then walk forward
                    t = {1: 'TR', 2: 'TA', 3: 'TL'}[rel]; its.append(t); sents.append({'TL': 'Turn left.', 'TR': 'Turn right.', 'TA': 'Turn around.'}[t]); h = (h + {'TR': 1, 'TL': 3, 'TA': 2}[t]) % 4
                    k = abs(need); its.append(f'F{k}'); sents.append(f'Take {k} step{"s" if k > 1 else ""}.')
    if any(int(it[1:]) > 99 for it in its if it[0] != 'T'): return navigate(r)
    call = ';'.join(its); out = C7.walk_ref(call)
    q = 'If you follow these instructions, do you return to the starting point? ' + ('Always face forward. ' if face_fwd else '') + ' '.join(sents) + '\nOptions:\n- Yes\n- No'
    lines = [f'{s}: {it}' for s, it in zip(sents, its)]
    return row('navigate', q, ('Always face forward.\n' if face_fwd else '') + '\n'.join(lines) + f'\n{WK[0]}{call}{WK[1]}{WK[2]}{out}{WK[3]}', out)


# ---------------------------------------------------------------- temporal_sequences (MM8)
PEOPLE = ['Nadia', 'Omar', 'Priya', 'Felix', 'Greta', 'Hugo', 'Ines', 'Jonas', 'Kira', 'Lars', 'Marta', 'Nico', 'Olga', 'Pablo', 'Rhea', 'Soren', 'Tara', 'Ugo']
PLACES = ['museum', 'bakery', 'swimming pool', 'bookstore', 'art gallery', 'zoo', 'post office', 'farmers market', 'bowling alley', 'aquarium', 'hair salon', 'skate park']
ACTS = ['jogging in the park', 'buying groceries at the supermarket', 'painting at the studio', 'repairing a bicycle at the garage', 'studying at the cafe',
        'waiting at the bus stop', 'playing chess at the club', 'shopping for shoes at the mall', 'visiting a friend downtown', 'feeding ducks at the pond',
        'eating lunch at the diner', 'practicing guitar at home', 'washing the car at the car wash', 'standing in line at the bank']


def hr(h):
    return f'{12 if h % 12 == 0 else h % 12}{"am" if h < 12 else "pm"}'


def temporal(r):
    W = r.randint(5, 10); C = r.randint(max(W + 5, 16), 22)
    cuts = sorted(r.sample(range(W + 1, C), r.randint(2, min(6, C - W - 1))))
    segs = list(zip([W] + cuts, cuts + [C])); gap = r.randrange(len(segs))
    busy = [sg for i, sg in enumerate(segs) if i != gap]
    opts = [segs[gap]] + r.sample(busy, min(3, len(busy)))
    extra = [(a, b) for a in range(W, C) for b in range(a + 1, C + 1) if (a, b) not in opts and (a, b) != segs[gap]
             and any(x < b and a < y for x, y in busy)]                    # overlaps a time they were seen elsewhere
    while len(opts) < 4: opts.append(extra.pop(r.randrange(len(extra))))
    r.shuffle(opts)
    name = r.choice(PEOPLE); place = r.choice(PLACES); wit = r.sample([p for p in PEOPLE if p != name], len(busy)); acts = r.sample(ACTS, len(busy))
    q = (f'Today, {name} went to the {place}. Between what times could they have gone?\nWe know that:\n{name} woke up at {hr(W)}.\n'
         + ''.join(f'{w} saw {name} {a} from {hr(x)} to {hr(y)}.\n' for w, a, (x, y) in zip(wit, acts, busy))
         + f'The {place} was closed after {hr(C)}.\nBetween what times could {name} have gone to the {place}?\nOptions:\n'
         + '\n'.join(f'({L}) {hr(x)} to {hr(y)}' for L, (x, y) in zip('ABCD', opts)))
    call = f'{W}-{C};' + ';'.join(f'{x}-{y}' for x, y in busy) + '|' + ';'.join(f'{x}-{y}' for x, y in opts)
    out = C8.time_ref(call); assert out
    body = (f'Window: woke up at {hr(W)} = {W}, closed after {hr(C)} = {C}: {W}-{C}.\nSeen: ' + ', '.join(f'{hr(x)} to {hr(y)} = {x}-{y}' for x, y in busy)
            + '.\nOptions: ' + ', '.join(f'({L}) {hr(x)} to {hr(y)} = {x}-{y}' for L, (x, y) in zip('ABCD', opts)) + f'.\n{TM[0]}{call}{TM[1]}{TM[2]}{out}{TM[3]}')
    return row('temporal_sequences', q, body, f'({out})')


# ---------------------------------------------------------------- word_sorting (MM9)
_WORDS = None


def _words():
    global _WORDS
    if _WORDS is None:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
        _WORDS = sorted(w[1:] for w in tk.get_vocab() if w.startswith('Ġ') and w[1:].isascii() and w[1:].isalpha() and w[1:].islower() and 3 <= len(w) - 1 <= 9)
    return _WORDS


def wordsort(r):
    V = _words()
    def word():
        k = r.random()
        if k < 0.55: return r.choice(V)
        if k < 0.9: return r.choice(V) + r.choice(V)[:r.randint(2, 5)]
        return r.choice(V) + r.choice(["'s", "'", '&t'])
    while True:
        ws = []
        while len(ws) < r.randint(2, 20):
            w = word()
            if w not in ws: ws.append(w)
        if len(_tok().encode(' '.join(ws))) <= C9.K - 8 and all(len(_tok().encode(' ' + w)) <= C9.TPW for w in ws): break
    q = f'Sort the following words alphabetically: List: {" ".join(ws)}'
    order = C9.sort_ref(ws); srt = [ws[ord(l) - 65] for l in order]
    body = ('Words: ' + ', '.join(f'{C9.LET[i]} {w}' for i, w in enumerate(ws)) + '.\n'
            + f'{SO[0]}{" ".join(ws)}{SO[1]}{SO[2]}{"".join(" " + l for l in order)}{SO[3]}\nIn order: ' + ', '.join(f'{l} {ws[ord(l) - 65]}' for l in order) + '.')
    return row('word_sorting', q, body, ' '.join(srt))


_TK = None


def _tok():
    global _TK
    if _TK is None:
        from transformers import AutoTokenizer
        _TK = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    return _TK


# ---------------------------------------------------------------- date_understanding (MM10)
import datetime as _dt, calendar as _cal
MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONL = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
WD = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
NUMW = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
HOL = [('Christmas Eve', 12, 24), ('Christmas Day', 12, 25), ("New Year's Eve", 12, 31), ("New Year's Day", 1, 1), ('Halloween', 10, 31),
       ("Valentine's Day", 2, 14), ('Independence Day in the US', 7, 4), ('the last day of June', 6, 30), ('the first day of September', 9, 1)]
mdy = lambda d: f'{d.month:02d}/{d.day:02d}/{d.year:04d}'


def dcall(d, n, u):
    c = f'{mdy(d)}{"+" if n >= 0 else "-"}{abs(n)}{u}'; out = C10.date_ref(c)
    return f'{DT[0]}{c}{DT[1]}{DT[2]}{out}{DT[3]}', out


def _asked(r):
    k = r.choice(['tomorrow', 'yesterday', 'today', 'days_ago', 'days_later', 'week_ago', 'weeks_later', 'month_ago', 'year_ago', 'years_later'])
    if k == 'tomorrow': return 'the date tomorrow', 1, 'D'
    if k == 'yesterday': return 'the date yesterday', -1, 'D'
    if k == 'today': return 'the date today', 0, 'D'
    if k == 'days_ago': n = r.choice([2, 3, 4, 5, 7, 10, 24, 30]); return f'the date {n} days ago', -n, 'D'
    if k == 'days_later': n = r.choice([2, 3, 5, 10, 24, 30]); return f'the date {n} days from today', n, 'D'
    if k == 'week_ago': n = r.choice([1, 1, 2, 3]); return f'the date {"one week" if n == 1 else NUMW[n] + " weeks"} ago from today', -n, 'W'
    if k == 'weeks_later': n = r.choice([1, 2]); return f'the date {"one week" if n == 1 else NUMW[n] + " weeks"} from today', n, 'W'
    if k == 'month_ago': return 'the date one month ago', -1, 'M'
    if k == 'year_ago': return 'the date one year ago from today', -1, 'Y'
    n = r.choice([1, 2, 10]); return f'the date {"one year" if n == 1 else NUMW[n] + " years"} from today', n, 'Y'


def dateq(r):
    y = r.randint(1900, 2030)
    kind = r.choice(['holiday', 'today', 'tomorrow', 'yesterday', 'anniv', 'uk', 'ago_was', 'weekday', 'later_is',
                     'nopad', 'nopad', 'passed', 'delayed', 'fullname'])
    steps = []
    if kind == 'holiday':
        name, m, d = r.choice(HOL); today = _dt.date(y, m, d); ctx = f'Today is {name} of {y}.'; why = f'Today is {name} of {y}, which is {mdy(today)}.'
    elif kind == 'today':
        today = _dt.date(y, 1, 1) + _dt.timedelta(days=r.randint(0, 364)); ctx = r.choice([f'Today is {MON[today.month - 1]} {today.day}, {y}.', f'It is {mdy(today)} today.', f'Today is {MONL[today.month - 1]} {today.day}, {y}.'])
        why = f'Today is {mdy(today)}.'
    elif kind in ('tomorrow', 'yesterday'):
        today = _dt.date(y, 1, 2) + _dt.timedelta(days=r.randint(0, 362)); other = today + _dt.timedelta(days=1 if kind == 'tomorrow' else -1)
        ctx = f'{kind.capitalize()} {"is" if kind == "tomorrow" else "was"} {mdy(other)}.'
        c, out = dcall(other, -1 if kind == 'tomorrow' else 1, 'D'); why = f'{kind.capitalize()} is {mdy(other)}, so today is {c} {out}.'
    elif kind == 'anniv':
        y0 = r.randint(1900, 2010); k = r.choice([1, 2, 3, 5, 10, 20, 25]); d0 = _dt.date(y0, r.randint(1, 12), r.randint(1, 28))
        today = _dt.date(y0 + k, d0.month, d0.day)
        ctx = f'Jane and John married on {MON[d0.month - 1]} {d0.day}, {y0}. It is their {k}-year anniversary today.'
        c, out = dcall(d0, k, 'Y'); why = f'They married on {mdy(d0)}; {k} years later is {c} {out}, which is today.'
    elif kind == 'uk':
        today = _dt.date(y, r.randint(1, 12), r.randint(1, 28))
        ctx = f'In the UK, people usually put the day before the month when formatting the date. Therefore, today is {today.day:02d}/{today.month:02d}/{y} to them.'
        why = f'In the UK format {today.day:02d}/{today.month:02d}/{y} means day {today.day}, month {today.month}, so today is {mdy(today)}.'
    elif kind == 'ago_was':
        today = _dt.date(y, 1, 1) + _dt.timedelta(days=r.randint(10, 360)); n = r.choice([1, 2, 3, 7, 10]); past = today - _dt.timedelta(days=n)
        ctx = f'It was {MON[past.month - 1]} {past.day}, {y} {n} day{"s" if n > 1 else ""} ago.' if r.random() < 0.5 else f'{NUMW.get(n, n)} days ago it was {mdy(past)}.'.capitalize()
        c, out = dcall(past, n, 'D'); why = f'{n} day{"s" if n > 1 else ""} ago was {mdy(past)}, so today is {c} {out}.'
    elif kind == 'weekday':
        first = _dt.date(y, 1, 1); wd = first.weekday(); tw = r.randrange(7); off = (tw - wd) % 7
        today = first + _dt.timedelta(days=off)
        ctx = f'The first day of {y} is a {WD[wd]}, and today is the first {WD[tw]} of {y}.'
        c, out = dcall(first, off, 'D'); why = f'Jan 1, {y} is a {WD[wd]}; the first {WD[tw]} is {off} day{"s" if off != 1 else ""} later: {c} {out}, which is today.'
    elif kind == 'nopad':                                   # dates written without zero padding: the call needs MM/DD/YYYY
        today = _dt.date(y, 1, 1) + _dt.timedelta(days=r.randint(0, 364)); np_ = f'{today.month}/{today.day}/{y}'
        ctx = r.choice([f'It is {np_} today.', f'Today, {np_}, is a day that we will never forget.', f'Today is {np_}.', f'The current date is {np_}.'])
        why = f'Today is {np_}, that is {mdy(today)}.'
    elif kind == 'passed':                                  # large day offsets
        d0 = _dt.date(y, 1, 1) + _dt.timedelta(days=r.randint(0, 364)); n = r.randint(11, 400); today = d0 + _dt.timedelta(days=n)
        ctx = r.choice([f'Jane started her new job on {MON[d0.month - 1]} {d0.day}, {y}. {n} days have passed since then.',
                        f'The project began on {MONL[d0.month - 1]} {d0.day}, {y}, and today is day {n + 1} of the project.' if False else f'The project began on {MONL[d0.month - 1]} {d0.day}, {y}. Today is exactly {n} days later.'])
        c, out = dcall(d0, n, 'D'); why = f'The start was {mdy(d0)}; {n} days later is {c} {out}, which is today.'
    elif kind == 'delayed':
        d0 = _dt.date(y, 1, 1) + _dt.timedelta(days=r.randint(0, 360)); n = r.choice([1, 1, 2, 3, 7]); today = d0 + _dt.timedelta(days=n)
        ev = r.choice(['The concert', 'The meeting', 'The flight', 'The exam'])
        ctx = f'{ev} was scheduled for {mdy(d0) if r.random() < 0.5 else f"{d0.month}/{d0.day}/{y}"}, but was delayed by {NUMW[n] if n in NUMW else n} day{"s" if n > 1 else ""} to today.'
        c, out = dcall(d0, n, 'D'); why = f'It was scheduled for {mdy(d0)} and delayed by {n} day{"s" if n > 1 else ""}: {c} {out}, which is today.'
    elif kind == 'fullname':
        today = _dt.date(y, 1, 2) + _dt.timedelta(days=r.randint(0, 362)); w = r.choice(['tomorrow', 'yesterday'])
        other = today + _dt.timedelta(days=1 if w == 'tomorrow' else -1)
        ctx = f'{w.capitalize()} {"is" if w == "tomorrow" else "was"} {MONL[other.month - 1]} {other.day}, {other.year}.'
        c, out = dcall(other, -1 if w == 'tomorrow' else 1, 'D'); why = f'{w.capitalize()} is {MONL[other.month - 1]} {other.day}, {other.year}, that is {mdy(other)}, so today is {c} {out}.'
    else:
        today = _dt.date(y, 1, 1) + _dt.timedelta(days=r.randint(0, 340)); n = r.choice([1, 2, 3, 5, 10]); fut = today + _dt.timedelta(days=n)
        ctx = f'The deadline is {MON[fut.month - 1]} {fut.day}, {y}, which is {n} day{"s" if n > 1 else ""} away from now.'
        c, out = dcall(fut, -n, 'D'); why = f'The deadline {mdy(fut)} is {n} day{"s" if n > 1 else ""} away, so today is {c} {out}.'
    phr, n, u = _asked(r)
    if n == 0: tgt_s, tgt = 'It is today.', today
    else:
        c, out = dcall(today, n, u); tgt_s, tgt = f'{phr[0].upper() + phr[1:]}: {c}', _dt.datetime.strptime(out, '%m/%d/%Y').date()
    ans = mdy(tgt)
    opts = {ans}
    while len(opts) < r.choice([5, 6]):
        k = r.random()
        if k < 0.3: o = tgt + _dt.timedelta(days=r.choice([-30, -7, -2, -1, 1, 2, 7, 30, 365, -365]))
        elif k < 0.5: o = _dt.date(tgt.year + r.choice([-50, -20, 1, 20, 40, 69]), tgt.month, min(tgt.day, 28))
        elif k < 0.7: o = _dt.date(tgt.year, r.randint(1, 12), min(tgt.day, 28))
        else: o = _dt.date(r.randint(1900, 2030), r.randint(1, 12), r.randint(1, 28))
        if 1000 <= o.year <= 2999: opts.add(mdy(o))
    opts = list(opts); r.shuffle(opts); L = chr(65 + opts.index(ans))
    q = f'{ctx} What is {phr} in MM/DD/YYYY?\nOptions:\n' + '\n'.join(f'({chr(65 + i)}) {o}' for i, o in enumerate(opts))
    return row('date_understanding', q, f'{why}\n{tgt_s}\nThat is {ans}, option ({L}).', f'({L})')


# ---------------------------------------------------------------- geometric_shapes (MM11)
import math as _m
SHAPE_NAME = {'2': 'line', '3': 'triangle', '5': 'pentagon', '6': 'hexagon', '7': 'heptagon', '8': 'octagon', 'K': 'kite', 'T': 'trapezoid', 'S': 'sector', 'E': 'ellipse'}
BASE_OPTS = ['circle', 'heptagon', 'hexagon', 'kite', 'line', 'octagon', 'pentagon', 'rectangle', 'sector', 'triangle']


def _path(r, code):
    f = lambda v: f'{v:.2f}'
    cx, cy, R = r.uniform(25, 75), r.uniform(25, 75), r.uniform(8, 24)
    rot = lambda raw, th: [(cx + x * _m.cos(th) - y * _m.sin(th), cy + x * _m.sin(th) + y * _m.cos(th)) for x, y in raw]
    if code in 'SE':
        a1, a2 = r.uniform(0, 3), r.uniform(3, 6); p1 = (cx + R * _m.cos(a1), cy + R * _m.sin(a1)); p2 = (cx + R * _m.cos(a2), cy + R * _m.sin(a2)); rr = f'{f(R)},{f(R)}'
        if code == 'S': return f'M {f(cx)},{f(cy)} L {f(p1[0])},{f(p1[1])} A {rr} 0.00 0,1 {f(p2[0])},{f(p2[1])} L {f(cx)},{f(cy)}'
        return f'M {f(p1[0])},{f(p1[1])} A {rr} {f(r.uniform(0, 360))} 1,0 {f(p2[0])},{f(p2[1])} A {rr} {f(r.uniform(0, 360))} 1,0 {f(p1[0])},{f(p1[1])}'
    if code == 'K':
        a, b, c = r.uniform(5, 20), r.uniform(5, 20), r.uniform(5, 25); pts = rot([(0, a), (b, 0), (0, -c), (-b, 0)], r.uniform(0, 6.28))
    elif code == 'T':
        w1, w2, h = r.uniform(5, 30), r.uniform(5, 30), r.uniform(3, 25); sh = r.uniform(-10, 10); pts = rot([(-w1 / 2, 0), (w1 / 2, 0), (w2 / 2 + sh, h), (-w2 / 2 + sh, h)], r.uniform(0, 6.28))
    else:
        n = int(code); angs = sorted(r.uniform(0, 6.28) for _ in range(n)); pts = [(cx + R * _m.cos(a), cy + R * _m.sin(a)) for a in angs]
    pts = [(min(max(x, 0.5), 99.4), min(max(y, 0.5), 99.4)) for x, y in pts]
    seq = pts + [pts[0]] if len(pts) > 2 else pts
    out = 'M ' + f(seq[0][0]) + ',' + f(seq[0][1])
    for i, (x, y) in enumerate(seq[1:]):
        if r.random() < 0.3: out += f' M {f(seq[i][0])},{f(seq[i][1])}'
        out += f' L {f(x)},{f(y)}'
    return out


def geometric(r):
    while True:
        code = r.choice(list(SHAPE_NAME)); d = _path(r, code)
        if C11.shape_ref(d) == code and len(_tok().encode(d)) <= C11.K - 6: break
    opts = list(BASE_OPTS)
    if code in 'TE' or r.random() < 0.2: opts.append(SHAPE_NAME[code] if code in 'TE' else r.choice(['trapezoid', 'ellipse']))
    L = chr(65 + opts.index(SHAPE_NAME[code]))
    q = f'This SVG path element <path d="{d}"/> draws a\nOptions:\n' + '\n'.join(f'({chr(65 + i)}) {o}' for i, o in enumerate(opts))
    body = f'{SH[0]}{d}{SH[1]}{SH[2]}{code}{SH[3]}\nCode {code}: {SHAPE_NAME[code]}, option ({L}).'
    return row('geometric_shapes', q, body, f'({L})')


GEN = {
    'tracking_shuffled_objects_three_objects': lambda r: tracking(r, 3), 'tracking_shuffled_objects_five_objects': lambda r: tracking(r, 5),
    'tracking_shuffled_objects_seven_objects': lambda r: tracking(r, 7), 'boolean_expressions': boolean, 'web_of_lies': lies,
    'multistep_arithmetic_two': multistep, 'object_counting': objcount,
    'logical_deduction_three_objects': lambda r: logical(r, 3), 'logical_deduction_five_objects': lambda r: logical(r, 5),
    'logical_deduction_seven_objects': lambda r: logical(r, 7), 'dyck_languages': dyck, 'navigate': navigate, 'temporal_sequences': temporal, 'word_sorting': wordsort, 'date_understanding': dateq, 'geometric_shapes': geometric,
}


def make(task, n, seed):
    r = random.Random(seed); bad = test_inputs(task); rows, seen = [], set()
    while len(rows) < n:
        x = GEN[task](r)
        if x['question'] in bad or x['question'] in seen: continue
        seen.add(x['question']); rows.append(x)
    return rows


if __name__ == '__main__':
    import sys
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    tag = sys.argv[2] if len(sys.argv) > 2 else 'bbh'
    tr, dv = [], []
    for i, t in enumerate(GEN):
        tr += make(t, n_train, 100 + i); dv += make(t, 100, 900 + i)
    random.Random(0).shuffle(tr)
    with open(f'data/{tag}_train.jsonl', 'w') as f:
        for x in tr: f.write(json.dumps(x) + '\n')
    with open(f'data/{tag}_dev.jsonl', 'w') as f:
        for x in dv: f.write(json.dumps(x) + '\n')
    print(len(tr), 'train', len(dv), 'dev')
    for t in GEN:
        x = next(v for v in dv if v['task'] == t)
        print('=====', t, '\nQ:', x['question'][-400:], '\nA:', x['completion'][:700])
