"""Team C task generators (own data; no BBH test data was read to write them).

python gen_tasks.py OUTDIR            -> OUTDIR/{dev,test,train}/{task}.json  in BBH's json shape {"examples": [{"input", "target"}]}
  dev   = seed 0, 30 per task  (the only split used for development)
  test  = seed 1, 100 per task (fresh; extra paraphrase / structure variety that dev does not have)
  train = seed 2 (not used: no fine-tuning in Team C)

Surface form follows the PUBLIC task descriptions and the 3 official CoT exemplars (answer format, option style):
  multistep_arithmetic_two : '((a op b op c op d) op (e op f ...)) ='       answer: integer
  boolean_expressions      : 'not ( True ) and False is'                    answer: True / False
  object_counting          : 'I have a ..., two ..., and a .... How many fruits do I have?'   answer: integer
All vocabularies (item names, categories) are our own lists, chosen to avoid the words of the 3 CoT exemplars.
"""
import json, os, random, sys

# ------------------------------------------------------------------ multistep arithmetic
def arith_group(r, n, lo=-9, hi=9, two_digit=0.0):
    nums = []
    for _ in range(n):
        if r.random() < two_digit: v = r.choice([-1, 1]) * r.randint(10, 30)
        else: v = r.randint(lo, hi)
        nums.append(v)
    ops = [r.choice(['+', '-', '*']) for _ in range(n - 1)]
    s = str(nums[0])
    for o, v in zip(ops, nums[1:]): s += f' {o} {v}'
    return s


def gen_arith(r, split):
    if split == 'dev':
        n1, n2 = r.choice([3, 4]), r.choice([3, 4]); g1, g2 = arith_group(r, n1), arith_group(r, n2)
        return f"(({g1}) {r.choice('+-*')} ({g2})) ="
    # fresh test: wider spread (2..5 numbers per group, a few two-digit numbers, sometimes three groups)
    k = r.choice([2, 2, 2, 3])
    gs = [f'({arith_group(r, r.choice([2, 3, 4, 5]), two_digit=0.1)})' for _ in range(k)]
    if k == 2: return f"({gs[0]} {r.choice('+-*')} {gs[1]}) ="
    return f"({gs[0]} {r.choice('+-*')} {gs[1]} {r.choice('+-')} {gs[2]}) ="


# ------------------------------------------------------------------ boolean expressions
def bool_expr(r, nlit, depth=0, maxdepth=2):
    """random expression with nlit literals, BBH spacing: '( x )' with spaces, 'not' prefixes."""
    def atom(d):
        if d < maxdepth and r.random() < 0.25:
            return '( ' + bool_expr(r, r.choice([1, 2]), d + 1, maxdepth) + ' )'
        return r.choice(['True', 'False'])
    def neg(x):
        k = r.choice([0, 0, 0, 1, 1, 2])
        return ' '.join(['not'] * k + [x])
    parts = [neg(atom(depth))]
    for _ in range(nlit - 1):
        parts += [r.choice(['and', 'or']), neg(atom(depth))]
    return ' '.join(parts)


def bool_eval(s):
    return str(eval(s))


def gen_bool(r, split):
    while True:
        e = bool_expr(r, r.choice([2, 3]) if split == 'dev' else r.choice([2, 3, 4]), maxdepth=1 if split == 'dev' else 2)
        n = len(e.split())
        if (split == 'dev' and 4 <= n <= 9) or (split != 'dev' and 3 <= n <= 13): return e + ' is', bool_eval(e)


# ------------------------------------------------------------------ object counting
CATS = {
    'fruits': [('mango', 'mangoes'), ('kiwi', 'kiwis'), ('papaya', 'papayas'), ('lychee', 'lychees'), ('fig', 'figs'), ('apricot', 'apricots'),
               ('guava', 'guavas'), ('pomegranate', 'pomegranates'), ('lime', 'limes'), ('lemon', 'lemons'), ('cherry', 'cherries'),
               ('pear', 'pears'), ('coconut', 'coconuts'), ('date', 'dates'), ('tangerine', 'tangerines'), ('persimmon', 'persimmons'),
               ('watermelon', 'watermelons'), ('cantaloupe', 'cantaloupes'), ('pineapple', 'pineapples'), ('quince', 'quinces')],
    'vegetables': [('potato', 'potatoes'), ('cucumber', 'cucumbers'), ('zucchini', 'zucchinis'), ('eggplant', 'eggplants'), ('radish', 'radishes'),
                   ('beet', 'beets'), ('turnip', 'turnips'), ('leek', 'leeks'), ('pumpkin', 'pumpkins'), ('cauliflower', 'cauliflowers'),
                   ('artichoke', 'artichokes'), ('parsnip', 'parsnips'), ('bell pepper', 'bell peppers'), ('cabbage', 'cabbages'),
                   ('shallot', 'shallots'), ('sweet potato', 'sweet potatoes'), ('asparagus spear', 'asparagus spears'), ('kale bunch', 'kale bunches')],
    'animals': [('giraffe', 'giraffes'), ('hamster', 'hamsters'), ('parrot', 'parrots'), ('turtle', 'turtles'), ('ferret', 'ferrets'),
                ('lizard', 'lizards'), ('rabbit', 'rabbits'), ('goat', 'goats'), ('owl', 'owls'), ('horse', 'horses'), ('duck', 'ducks'),
                ('sheep', 'sheep'), ('chinchilla', 'chinchillas'), ('gecko', 'geckos'), ('llama', 'llamas'), ('penguin', 'penguins'),
                ('hedgehog', 'hedgehogs'), ('canary', 'canaries')],
    'musical instruments': [('cello', 'cellos'), ('harp', 'harps'), ('banjo', 'banjos'), ('ukulele', 'ukuleles'), ('oboe', 'oboes'),
                            ('saxophone', 'saxophones'), ('drum', 'drums'), ('tuba', 'tubas'), ('harmonica', 'harmonicas'), ('mandolin', 'mandolins'),
                            ('xylophone', 'xylophones'), ('bassoon', 'bassoons'), ('accordion', 'accordions'), ('tambourine', 'tambourines'),
                            ('cornet', 'cornets'), ('sitar', 'sitars')],
    'tools': [('hammer', 'hammers'), ('screwdriver', 'screwdrivers'), ('wrench', 'wrenches'), ('saw', 'saws'), ('chisel', 'chisels'),
              ('drill', 'drills'), ('pliers set', 'pliers sets'), ('tape measure', 'tape measures'), ('shovel', 'shovels'), ('rake', 'rakes')],
    'vehicles': [('bicycle', 'bicycles'), ('scooter', 'scooters'), ('motorcycle', 'motorcycles'), ('van', 'vans'), ('truck', 'trucks'),
                 ('canoe', 'canoes'), ('tractor', 'tractors'), ('sailboat', 'sailboats'), ('skateboard', 'skateboards'), ('minibus', 'minibuses')],
}
FILLER = [('lamp', 'lamps'), ('chair', 'chairs'), ('notebook', 'notebooks'), ('umbrella', 'umbrellas'), ('mug', 'mugs'), ('pillow', 'pillows'),
          ('backpack', 'backpacks'), ('candle', 'candles'), ('stapler', 'staplers'), ('kettle', 'kettles')]
NUMW = {1: 'a', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
AN = lambda w: 'an' if w[0] in 'aeiou' else 'a'

Q_DEV = ['How many {c} do I have?']
Q_TEST = ['How many {c} do I have?', 'What is the total number of {c} I have?', 'Count the {c}: how many are there?',
          'How many {c} are there in total?', 'In total, how many {c} do I own?']
OPEN_DEV = ['I have {L}.']
OPEN_TEST = ['I have {L}.', 'I own {L}.', 'My collection contains {L}.', 'There are {L} in my house.', 'Last week I got {L}.',
             'I bought {L}, and my neighbor has a dog.', 'In my garage I keep {L}.']


def gen_count(r, split):
    cat = r.choice(['fruits', 'vegetables', 'animals', 'musical instruments'] + (['tools', 'vehicles'] if split != 'dev' else []))
    n_target = r.randint(2, 6) if split == 'dev' else r.randint(1, 8)
    mode = r.random()
    n_other = 0 if mode < 1 / 3 else (r.randint(1, 5) if split == 'dev' else r.randint(1, 6))      # 1/3: every item is in the category
    tgt = r.sample(CATS[cat], n_target)
    others_pool = [x for c, v in CATS.items() if c != cat for x in v] + FILLER
    oth = r.sample(others_pool, n_other)
    if mode > 5 / 6:                                     # 1/6: 'how many items' -- everything counts
        cat = r.choice(['items', 'things'] if split == 'dev' else ['items', 'things', 'objects']); tgt, oth = tgt + oth, []
    items, total = [], 0
    for (s, p), is_t in [(x, True) for x in tgt] + [(x, False) for x in oth]:
        q = r.choice([1, 1, 1, 1, 2, 3]) if split == 'dev' else r.choice([1, 1, 1, 2, 2, 3, 4, 5])
        if is_t: total += q
        items.append(f'{AN(s)} {s}' if q == 1 else f'{NUMW[q]} {p}')
    r.shuffle(items)
    L = items[0] if len(items) == 1 else f'{items[0]} and {items[1]}' if len(items) == 2 else ', '.join(items[:-1]) + ', and ' + items[-1]
    o = r.choice(OPEN_DEV if split == 'dev' else OPEN_TEST).format(L=L)
    q = r.choice(Q_DEV if split == 'dev' else Q_TEST).format(c=cat)
    return f'{o} {q}', str(total)


TASKS = {'multistep_arithmetic_two': gen_arith, 'boolean_expressions': gen_bool, 'object_counting': gen_count}
SPLITS = {'dev': (0, 30), 'test': (1, 100), 'train': (2, 0)}

if __name__ == '__main__':
    out = sys.argv[1]
    for split, (seed, n) in SPLITS.items():
        if n == 0: continue
        os.makedirs(f'{out}/{split}', exist_ok=True)
        for t, g in TASKS.items():
            r = random.Random(f'{t}-{seed}')
            ex = []
            seen = set()
            while len(ex) < n:
                res = g(r, split)
                if t == 'multistep_arithmetic_two': inp, tgt = res, str(eval(res[:-2]))
                else: inp, tgt = res
                if inp in seen: continue
                seen.add(inp); ex.append({'input': inp, 'target': tgt})
            json.dump({'examples': ex}, open(f'{out}/{split}/{t}.json', 'w'), indent=1)
            print(split, t, len(ex), ex[0])
