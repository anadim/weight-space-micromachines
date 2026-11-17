"""Our own generator for 'objects in a fixed order + clues' questions (logical_deduction-like surface form), plus other styles:
  style 'bbh' : BBH-like ("The following paragraphs each describe a set of N objects arranged in a fixed order. ..." + clues +
                options "(A) The X is the <position phrase>"), relation families: left/right, top/bottom, age, price,
                finishing order, height, weight, arrival time (generic comparative / ordinal English, written by us)
  style 'alt' : TRAIN-ONLY other style: people queuing at a counter, with adjacency ("directly in front of", "next to") and
                negative ("is not at the front") clues, and both "who is k-th?" and "what position is X?" questions
  style 'oos' : EVAL-ONLY settings never trained on: theater seating (left/right seats, "immediately to the left", "beside",
                "not on an aisle seat") and a contest ranking ("scored higher", "ranked right above", "was not ranked last")
Splits: 'train' (seed 2), 'dev' (seed 0), 'test' (seed 1; adds EXTRA domains/objects/paraphrases never used in train/dev).
"""
import random, itertools
from vocab import NAMES_BASE, NAMES_EXTRA, COLORS_BASE, COLORS_EXTRA, ORDINALS, NUMWORDS, pick
from callfmt import CS, CE, OS, OE, order_ref, order_violates

LET = 'ABCDEFGHIJ'
EXCLUDE = set()          # families left out (ablation of the age/price scales: make_data.py train_ablate)
O = lambda k: ORDINALS[k - 1]

# relation families. pos 1 = the side named first in 'ends'. before = closer to pos 1.
FAM = {
    'lr': dict(ends=('leftmost', 'rightmost'),
               before=['is to the left of', 'is left of', 'sits to the left of'], before_x=['is positioned somewhere left of', 'is further left than'],
               after=['is to the right of', 'is right of', 'sits to the right of'], after_x=['is positioned somewhere right of', 'is further right than'],
               s1=['is the leftmost'], sk=['is the {o} from the left'], e1=['is the rightmost'], ek=['is the {o} from the right'],
               s1_x=['is the farthest to the left'], sk_x=['is in the {o} spot from the left'], e1_x=['is the farthest to the right'], ek_x=['is in the {o} spot from the right']),
    'tb': dict(ends=('top', 'bottom'),
               before=['is above', 'is higher than'], before_x=['sits somewhere above'], after=['is below', 'is lower than'], after_x=['sits somewhere below'],
               s1=['is at the top'], sk=['is the {o} from the top'], e1=['is at the bottom'], ek=['is the {o} from the bottom'],
               s1_x=['is the highest'], sk_x=['is {o} counting from the top'], e1_x=['is the lowest'], ek_x=['is {o} counting from the bottom']),
    'finish': dict(ends=('first', 'last'),
                   before=['finished above', 'finished ahead of', 'finished before'], before_x=['came in ahead of'],
                   after=['finished below', 'finished behind', 'finished after'], after_x=['came in behind'],
                   s1=['finished first'], sk=['finished {o}'], e1=['finished last'], ek=['finished {o}-to-last'],
                   s1_x=['came in first'], sk_x=['came in {o}'], e1_x=['came in last'], ek_x=['came in {o}-to-last']),
    'arrive': dict(ends=('first', 'last'),
                   before=['arrived before', 'arrived earlier than'], before_x=['showed up before'], after=['arrived after', 'arrived later than'], after_x=['showed up after'],
                   s1=['arrived first'], sk=['arrived {o}'], e1=['arrived last'], ek=['arrived {o}-to-last'],
                   s1_x=['was the first to arrive'], sk_x=['was the {o} to arrive'], e1_x=['was the last to arrive'], ek_x=['was the {o}-to-last to arrive']),
}

# domains: setting sentence, family, objects (base / test-only), article style
DOM = [
    dict(fam='lr', intro='On a windowsill, there are {N} potted plants: {L}.', objs=['cactus', 'fern', 'orchid', 'bonsai', 'begonia', 'tulip', 'violet', 'aloe', 'ivy', 'geranium']),
    dict(fam='lr', intro='On a mantelpiece, there are {N} candles: {L}.', objs=[f'{c} candle' for c in COLORS_BASE], objs_x=[f'{c} candle' for c in COLORS_EXTRA]),
    dict(fam='lr', intro='On a kitchen counter, there are {N} jars: {L}.', objs=['jar of honey', 'jar of pickles', 'jar of jam', 'jar of olives', 'jar of mustard', 'jar of capers', 'jar of salsa', 'jar of pesto']),
    dict(fam='tb', intro='In a stack, there are {N} boxes: {L}.', objs=[f'{c} box' for c in COLORS_BASE], objs_x=[f'{c} box' for c in COLORS_EXTRA]),
    dict(fam='tb', intro='On a tiered stand, there are {N} cakes: {L}.', objs=['lemon cake', 'carrot cake', 'sponge cake', 'cheesecake', 'fruitcake', 'pound cake', 'honey cake', 'coffee cake']),
    dict(fam='finish', intro='In a swimming race, there were {N} swimmers: {L}.', people=True),
    dict(fam='finish', intro='In a chess tournament, there were {N} players: {L}.', people=True),
    dict(fam='arrive', intro='At a dinner party, {N} guests arrived at different times: {L}.', people=True),
]
DOM_X = [                                   # test-only domains
    dict(fam='lr', intro='On a coat rack, there are {N} umbrellas: {L}.', objs=[f'{c} umbrella' for c in COLORS_BASE + COLORS_EXTRA]),
    dict(fam='finish', intro='In a baking contest, there were {N} bakers: {L}.', people=True),
]


# ---- generic comparative scales (v2). A written-down list of common English gradable adjectives with their comparative and
# superlative forms, all rendered with the SAME templates ("is X-er than", "is the X-est", "is the k-th X-est" / "k-th-X-est").
# Not mined from BBH: any everyday comparison (size, weight, age, price, speed, temperature, loudness, ...) is in scope.
# Each entry: comparative, opposite comparative, superlative, opposite superlative; *_x = test-only synonyms.
SCALES = {
    'height': ('taller', 'shorter', 'tallest', 'shortest', 'loftier', 'lower'),
    'weight': ('heavier', 'lighter', 'heaviest', 'lightest', 'weightier', 'less weighty'),
    'age_obj': ('older', 'newer', 'oldest', 'newest', 'more antique', 'more modern'),
    'age_live': ('older', 'younger', 'oldest', 'youngest', 'more senior', 'more junior'),
    'price': ('more expensive', 'cheaper', 'most expensive', 'cheapest', 'costlier', 'lower-priced'),
    'speed': ('faster', 'slower', 'fastest', 'slowest', 'quicker', 'less quick'),
    'size': ('bigger', 'smaller', 'biggest', 'smallest', 'larger', 'less large'),
    'length': ('longer', 'shorter', 'longest', 'shortest', 'lengthier', 'less long'),
    'temp': ('warmer', 'colder', 'warmest', 'coldest', 'hotter', 'cooler'),
    'loud': ('louder', 'quieter', 'loudest', 'quietest', 'noisier', 'softer'),
    'bright': ('brighter', 'dimmer', 'brightest', 'dimmest', 'more luminous', 'less luminous'),
    'width': ('wider', 'narrower', 'widest', 'narrowest', 'broader', 'less broad'),
}
SCALES_X = {                                                    # test-only scales
    'spice': ('spicier', 'milder', 'spiciest', 'mildest', 'hotter', 'less hot'),
    'strength': ('stronger', 'weaker', 'strongest', 'weakest', 'more powerful', 'less powerful'),
}


def scale_family(c1, c2, s1, s2, x1, x2):
    return dict(ends=(s1, s2), copula=True, before=[f'{c1} than'], after=[f'{c2} than'], before_x=[f'{x1} than'], after_x=[f'{x2} than'],
                s1=[f'the {s1}'], sk=['the {o}-' + s1, 'the {o} ' + s1], e1=[f'the {s2}'], ek=['the {o}-' + s2, 'the {o} ' + s2],
                s1_x=[f'the {s1} one'], sk_x=['the {o} ' + s1 + ' one'], e1_x=[f'the {s2} one'], ek_x=['the {o} ' + s2 + ' one'])


for _k, _v in {**SCALES, **SCALES_X}.items(): FAM[_k] = scale_family(*_v)

DOM += [
    dict(fam='height', intro='In a park, there are {N} trees: {L}.', objs=['oak', 'maple', 'birch', 'willow', 'cedar', 'elm', 'poplar', 'spruce', 'linden', 'ash']),
    dict(fam='height', intro='In a greenhouse, there are {N} sunflowers: {L}.', objs=[f'{c} sunflower' for c in COLORS_BASE]),
    dict(fam='weight', intro='At a post office, there are {N} parcels: {L}.', objs=[f'{c} parcel' for c in COLORS_BASE], objs_x=[f'{c} parcel' for c in COLORS_EXTRA]),
    dict(fam='weight', intro='At an airport, there are {N} suitcases: {L}.', objs=['leather suitcase', 'canvas duffel', 'hard-shell case', 'garment bag', 'backpack', 'trunk', 'tote bag', 'rolling case']),
    dict(fam='age_obj', intro='In a museum, there are {N} clocks: {L}.', objs=['cuckoo clock', 'grandfather clock', 'wall clock', 'alarm clock', 'pocket watch', 'mantel clock', 'sundial', 'water clock']),
    dict(fam='age_obj', intro='A family owns {N} heirlooms: {L}.', objs=['teapot', 'quilt', 'mirror', 'music box', 'rocking chair', 'brooch', 'lantern', 'violin', 'sewing machine']),
    dict(fam='age_live', intro='In a family, there are {N} siblings: {L}.', people=True),
    dict(fam='age_live', intro='At a sanctuary, there are {N} tortoises: {L}.', objs=['tortoise named ' + n for n in ['Pip', 'Moss', 'Clover', 'Basil', 'Pebble', 'Juniper', 'Sage', 'Thistle']], named=True),
    dict(fam='price', intro='A bakery sells {N} pastries: {L}.', objs=['croissants', 'scones', 'muffins', 'eclairs', 'donuts', 'pretzels', 'bagels', 'strudels', 'macarons'], plural=True),
    dict(fam='price', intro='A cheese shop sells {N} cheeses: {L}.', objs=['brie', 'gouda', 'feta', 'cheddar', 'gruyere', 'halloumi', 'manchego', 'ricotta', 'stilton']),
    dict(fam='speed', intro='In a nature film, {N} animals are compared: {L}.', objs=['cheetah', 'gazelle', 'ostrich', 'greyhound', 'hare', 'tortoise', 'sloth', 'elk', 'coyote']),
    dict(fam='speed', intro='On a railway, there are {N} trains: {L}.', objs=['freight train', 'night train', 'commuter train', 'steam train', 'bullet train', 'mail train', 'tourist train']),
    dict(fam='size', intro='In a pet shop, there are {N} aquariums: {L}.', objs=[f'{c} aquarium' for c in COLORS_BASE]),
    dict(fam='length', intro='In a hardware store, there are {N} ropes: {L}.', objs=[f'{c} rope' for c in COLORS_BASE]),
    dict(fam='temp', intro='At a cafe, there are {N} drinks on the counter: {L}.', objs=['cocoa', 'latte', 'iced tea', 'cider', 'lemonade', 'espresso', 'smoothie', 'chai']),
    dict(fam='loud', intro='In a workshop, there are {N} machines: {L}.', objs=['drill press', 'lathe', 'band saw', 'air compressor', 'grinder', 'sander', 'jigsaw', 'planer']),
    dict(fam='bright', intro='In a living room, there are {N} lamps: {L}.', objs=['desk lamp', 'floor lamp', 'lava lamp', 'reading lamp', 'paper lantern', 'chandelier', 'nightlight', 'salt lamp']),
    dict(fam='width', intro='Along a hallway, there are {N} doors: {L}.', objs=[f'{c} door' for c in COLORS_BASE]),
]
DOM_X += [
    dict(fam='age_obj', intro='In a garage, there are {N} bicycles: {L}.', objs=['tandem', 'unicycle', 'mountain bike', 'tricycle', 'penny-farthing', 'road bike', 'cargo bike', 'folding bike']),
    dict(fam='height', intro='In a city, there are {N} towers: {L}.', objs=['clock tower', 'water tower', 'radio tower', 'bell tower', 'lighthouse', 'watchtower', 'minaret', 'obelisk']),
    dict(fam='spice', intro='A food truck offers {N} sauces: {L}.', objs=['salsa verde', 'harissa', 'sriracha', 'chimichurri', 'mole', 'peri-peri', 'tzatziki', 'gochujang']),
    dict(fam='strength', intro='In a lab, there are {N} magnets: {L}.', objs=[f'{c} magnet' for c in COLORS_BASE + COLORS_EXTRA]),
]


def art(x): return ('an ' if x[0].lower() in 'aeiou' else 'a ') + x


def join(xs): return xs[0] if len(xs) == 1 else (', '.join(xs[:-1]) + (',' if len(xs) > 2 else '') + ' and ' + xs[-1])


def solutions(n, cons):
    return [p for p in itertools.permutations(range(1, n + 1)) if not any(order_violates(p, op, a, b) for op, a, b in cons)]


def true_constraints(perm, kinds, r):
    """All true constraints of the given kinds for the ordering perm (perm[k-1] = object at position k), shuffled."""
    n = len(perm); pos = {o: k + 1 for k, o in enumerate(perm)}; out = []
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if a == b: continue
            if '<>' in kinds: out.append(('<' if pos[a] < pos[b] else '>', a, b))
            if '+' in kinds and pos[b] == pos[a] + 1: out.append(('+', a, b))
            if '~' in kinds and abs(pos[a] - pos[b]) == 1 and a < b: out.append(('~', a, b))
        if '=' in kinds:
            out.append(('=', a, pos[a]) if r.random() < 0.5 else ('=-', a, n + 1 - pos[a]))
        if '!' in kinds:
            for k in range(1, n + 1):
                if k != pos[a]: out.append(('!=', a, k) if r.random() < 0.5 else ('!=-', a, n + 1 - k))
    r.shuffle(out); return out


def make_clues(r, n, kinds, abs_frac):
    """A random ordering and a (near-)minimal list of true clues that pins it down uniquely. Clue groups are drawn with
    weights: absolute positions abs_frac, the rest split over the relational groups present in kinds."""
    perm = list(range(1, n + 1)); r.shuffle(perm)
    pool = true_constraints(perm, kinds, r)
    groups = {'abs': [c for c in pool if c[0] in ('=', '=-')], 'rel': [c for c in pool if c[0] in ('<', '>')],
              'adj': [c for c in pool if c[0] in ('+', '~')], 'neg': [c for c in pool if c[0] in ('!=', '!=-')]}
    wts = {'abs': abs_frac, 'rel': 1.0, 'adj': 0.8, 'neg': 0.6}
    cons, sols = [], solutions(n, [])
    while len(sols) > 1:
        live = [g for g in groups if groups[g]]
        g = r.choices(live, [wts[x] if x != 'abs' else abs_frac / (1 - abs_frac) * sum(wts[y] for y in live if y != 'abs') for x in live])[0] if len(live) > 1 else live[0]
        c = groups[g].pop()
        s2 = [p for p in sols if not order_violates(p, *c)]
        if len(s2) < len(sols): cons.append(c); sols = s2
    for c in list(cons):                                   # drop redundant clues
        rest = [x for x in cons if x != c]
        if len(solutions(n, rest)) == 1 and r.random() < 0.8: cons = rest
    r.shuffle(cons)
    return tuple(sols[0]), cons


def make_bbh(r, n, split):
    test = split == 'test'
    d = r.choice([x for x in DOM + (DOM_X if test else []) * (2 if test else 0) if x['fam'] not in EXCLUDE])
    F = FAM[d['fam']]; X = lambda key: F[key] + (F.get(key + '_x', []) if test else [])
    if d.get('people'): objs = pick(r, NAMES_BASE, NAMES_EXTRA, n, split)
    else: objs = pick(r, d['objs'], d.get('objs_x', []), n, split)
    people, plural = d.get('people'), d.get('plural')
    The = (lambda x: x) if people else (lambda x: 'The ' + x)
    the = (lambda x: x) if people else (lambda x: 'the ' + x)
    be = '' if not F.get('copula') else ('are ' if plural else 'is ')
    perm, cons = make_clues(r, n, ('<>', '='), 0.4)
    def phrase(op, b):
        if op in ('=', '=-'):
            k = b; start = op == '='
            t = r.choice(X('s1' if start else 'e1')) if k == 1 else r.choice(X('sk' if start else 'ek')).format(o=O(k))
            return be + t
        return be + r.choice(X('before' if op == '<' else 'after'))
    clues = []
    for op, a, b in cons:
        p = phrase(op, b)
        clues.append(f'{The(objs[a - 1])} {p}.' if op in ('=', '=-') else f'{The(objs[a - 1])} {p} {the(objs[b - 1])}.')
    # question: one position, asked from either end
    k = r.randint(1, n); start = r.random() < 0.5
    if k == 1: pred = be + r.choice(F['s1'] if start else F['e1'])
    else: pred = be + r.choice(F['sk'] if start else F['ek']).format(o=O(k))
    ans_obj = perm[k - 1] if start else perm[n - k]
    order = list(range(1, n + 1))
    if r.random() < 0.25: r.shuffle(order)
    listing = join([o if (people or plural) else art(o) for o in objs])
    q = (f'The following paragraphs each describe a set of {NUMWORDS[n]} objects arranged in a fixed order. The statements are logically '
         f'consistent within each paragraph. ' + d['intro'].format(N=NUMWORDS[n], L=listing) + ' ' + ' '.join(clues)
         + '\nOptions:\n' + '\n'.join(f'({LET[t]}) {The(objs[o - 1])} {pred}' for t, o in enumerate(order)))
    letter = LET[order.index(ans_obj)]
    sym = {'<': '<', '>': '>', '=': '=', '=-': '=-'}
    call = f'order {n}:' + ','.join(f'{a}{sym[op]}{b}' for op, a, b in cons) + (f'?{k}' if start else f'?-{k}')
    res = order_ref(call); assert res == str(ans_obj), (call, res, ans_obj)
    sol = (" Let's solve the ordering with the order tool.\n"
           + 'Objects: ' + ', '.join(f'{t + 1} = {o}' for t, o in enumerate(objs)) + '.\n'
           + f"Positions: 1 = {F['ends'][0]}, {n} = {F['ends'][1]}.\n"
           + ''.join(f'{c[:-1]}: {a}{sym[op]}{b}.\n' for c, (op, a, b) in zip(clues, cons))
           + f"Question: the one that {pred}: {'?' if start else '?-'}{k}\n"
           + f'{CS}{call}{CE}{OS}{res}{OE}\n'
           + f'Object {res} = {objs[int(res) - 1]}. So the answer is ({letter}).')
    lab = {'lr': ('left', 'right'), 'tb': ('top', 'bottom')}.get(d['fam'], F['ends'])
    cot = (" Let's think step by step.\n" + ''.join(f'({t + 1}) {c}\n' for t, c in enumerate(clues))
           + f'({len(clues) + 1}) Combining these, we get the following ordering: "({lab[0]}) ' + ' '.join(objs[o - 1] for o in perm) + f' ({lab[1]})".\n'
           + f'According to this ordering, {the(objs[ans_obj - 1])} {pred}.\n{The(objs[ans_obj - 1])} {pred}. So the answer is ({letter}).')
    return dict(input=q, target=f'({letter})', gold_call=call, gold_result=res, solution=sol, cot=cot, n=n, style='bbh')


# ---------------------------------------------------------------- alt (train only): a queue, with adjacency and negation
def make_alt(r, n, split):
    names = pick(r, NAMES_BASE, NAMES_EXTRA, n, split)
    perm, cons = make_clues(r, n, ('<>', '+', '~', '=', '!'), 0.35)
    def say(op, a, b):
        A, B = names[a - 1], names[b - 1] if op in '<>+~' else None
        if op == '<': return f'{A} is somewhere ahead of {B}.'
        if op == '>': return f'{A} is somewhere behind {B}.'
        if op == '+': return r.choice([f'{A} is directly in front of {B}.', f'{B} is directly behind {A}.'])
        if op == '~': return r.choice([f'{A} is next to {B}.', f'{A} and {B} stand next to each other.'])
        if op == '=': return f'{A} is at the front.' if b == 1 else f'{A} is {O(b)} in line.'
        if op == '=-': return f'{A} is at the back.' if b == 1 else f'{A} is {O(b)} from the back.'
        if op == '!=': return f'{A} is not at the front.' if b == 1 else f'{A} is not {O(b)} in line.'
        return f'{A} is not at the back.' if b == 1 else f'{A} is not {O(b)} from the back.'
    clues = [say(*c) for c in cons]
    q = f'{NUMWORDS[n].capitalize()} people are waiting in a single line at a ticket counter: {join(names)}. ' + ' '.join(clues) + ' '
    if r.random() < 0.6:
        k = r.randint(1, n); start = r.random() < 0.6
        q += f'Who is {O(k)} in line?' if start else (f'Who is at the back?' if k == 1 else f'Who is {O(k)} from the back?')
        qs = f'?{k}' if start else f'?-{k}'; opts = list(names); ans = names[(perm[k - 1] if start else perm[n - k]) - 1]
        call_ans = lambda res: f'Person {res} = {names[int(res) - 1]}.'
    else:
        a = r.randint(1, n); q += f'In which position, counting from the front, is {names[a - 1]}?'; qs = f'@{a}'
        opts = [O(k) for k in range(1, n + 1)]; ans = O(perm.index(a) + 1)
        call_ans = lambda res: f'Position {res} = {O(int(res))}.'
    r.shuffle(opts) if qs[0] != '@' else None
    letter = LET[opts.index(ans)]
    q += '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    call = f'order {n}:' + ','.join(f'{a}{op}{b}' for op, a, b in cons) + qs
    res = order_ref(call); assert res
    sol = (" Let's solve the ordering with the order tool.\n" + 'Objects: ' + ', '.join(f'{t + 1} = {o}' for t, o in enumerate(names)) + '.\n'
           + f'Positions: 1 = front, {n} = back.\n' + ''.join(f'{c[:-1]}: {a}{op}{b}.\n' for c, (op, a, b) in zip(clues, cons))
           + f'Question: {qs}\n{CS}{call}{CE}{OS}{res}{OE}\n{call_ans(res)} So the answer is ({letter}).')
    return dict(input=q, target=f'({letter})', gold_call=call, gold_result=res, solution=sol, cot=None, n=n, style='alt')


# ---------------------------------------------------------------- oos (eval only): theater seating, contest ranking
def make_oos(r, n, split):
    names = pick(r, NAMES_BASE, NAMES_EXTRA, n, split)
    seating = r.random() < 0.5
    perm, cons = make_clues(r, n, ('<>', '+', '~', '=', '!'), 0.35)
    def say(op, a, b):
        A = names[a - 1]; B = names[b - 1] if op in '<>+~' else None
        if seating:
            if op == '<': return f'{A} sits somewhere to the left of {B}.'
            if op == '>': return f'{A} sits somewhere to the right of {B}.'
            if op == '+': return f'{A} sits immediately to the left of {B}.'
            if op == '~': return f'{A} sits beside {B}.'
            if op == '=': return f'{A} has the leftmost seat.' if b == 1 else f'{A} sits in the {O(b)} seat from the left.'
            if op == '=-': return f'{A} has the rightmost seat.' if b == 1 else f'{A} sits in the {O(b)} seat from the right.'
            if op == '!=': return f'{A} does not have the leftmost seat.' if b == 1 else f'{A} is not in the {O(b)} seat from the left.'
            return f'{A} does not have the rightmost seat.' if b == 1 else f'{A} is not in the {O(b)} seat from the right.'
        if op == '<': return f'{A} scored higher than {B}.'
        if op == '>': return f'{A} scored lower than {B}.'
        if op == '+': return f'{A} was ranked right above {B}.'
        if op == '~': return f'{A} and {B} were ranked next to each other.'
        if op == '=': return f'{A} won the contest.' if b == 1 else f'{A} was ranked {O(b)}.'
        if op == '=-': return f'{A} was ranked last.' if b == 1 else f'{A} was ranked {O(b)} from the bottom.'
        if op == '!=': return f'{A} did not win.' if b == 1 else f'{A} was not ranked {O(b)}.'
        return f'{A} was not ranked last.' if b == 1 else f'{A} was not ranked {O(b)} from the bottom.'
    clues = [say(*c) for c in cons]
    if seating: q = f'{join(names)} bought tickets for {NUMWORDS[n]} adjacent seats in one row of a theater. ' + ' '.join(clues) + ' '
    else: q = f'At a cooking contest the judges ranked {NUMWORDS[n]} contestants with no ties: {join(names)}. ' + ' '.join(clues) + ' '
    if r.random() < 0.6:
        k = r.randint(1, n); start = r.random() < 0.6
        if seating: q += f'Who sits in the {O(k)} seat from the {"left" if start else "right"}?'
        else: q += f'Who was ranked {O(k)}?' if start else f'Who was ranked {O(k)} from the bottom?'
        opts = list(names); r.shuffle(opts); ans = names[(perm[k - 1] if start else perm[n - k]) - 1]; kind = '?'
    else:
        a = r.randint(1, n)
        q += (f'Counting from the left, which seat does {names[a - 1]} sit in?' if seating else f'What rank did {names[a - 1]} get?')
        opts = [O(k) for k in range(1, n + 1)]; ans = O(perm.index(a) + 1); kind = '@'
    letter = LET[opts.index(ans)]
    q += '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    qs = (f'?{k}' if start else f'?-{k}') if kind == '?' else f'@{a}'
    call = f'order {n}:' + ','.join(f'{x}{op}{y}' for op, x, y in cons) + qs
    assert order_ref(call) and (names[int(order_ref(call)) - 1] if kind == '?' else O(int(order_ref(call)))) == ans
    return dict(input=q, target=f'({letter})', gold_call=call, gold_result=order_ref(call), solution=None, cot=None, n=n, style='oos', kind=kind, setting='seating' if seating else 'ranking')


def make(r, n, split, style='bbh'):
    return {'bbh': make_bbh, 'alt': make_alt, 'oos': make_oos}[style](r, n, split)


if __name__ == '__main__':
    r = random.Random(0)
    for st in ('bbh', 'alt', 'oos'):
        for n in (3, 5, 7):
            x = make(r, n, 'test', st); print('=' * 80); print(x['input']); print('->', x['target']); print(x['solution']); print(x['cot'])
