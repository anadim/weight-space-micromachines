"""Programmatic OWN puzzle-format no-tool negatives (generic genres, own word lists, own phrasing; no BBH item or template
copied). None is solvable by an attached tool: no attribute ordering from clues, no swaps/tracking, no time intervals,
no walking directions, no truth-tellers, no arithmetic. Each generator returns (question, short_answer, one_line_reason).
"""
import random
FRUIT = ['apple', 'banana', 'cherry', 'mango', 'pear', 'plum', 'grape', 'peach', 'lemon', 'kiwi']
ANIMAL = ['otter', 'falcon', 'badger', 'heron', 'lynx', 'beaver', 'gecko', 'walrus', 'moose', 'koala']
TOOL = ['hammer', 'wrench', 'chisel', 'saw', 'pliers', 'drill', 'trowel', 'shovel', 'rake', 'clamp']
COLOR = ['crimson', 'teal', 'amber', 'violet', 'olive', 'ivory', 'coral', 'indigo', 'maroon', 'navy']
INSTR = ['violin', 'flute', 'cello', 'drum', 'harp', 'oboe', 'banjo', 'tuba', 'piano', 'lute']
CATS = {'fruit': FRUIT, 'animal': ANIMAL, 'tool': TOOL, 'color': COLOR, 'musical instrument': INSTR}
WORDS = sorted(set(FRUIT + ANIMAL + TOOL + COLOR + INSTR + ['lantern', 'meadow', 'quarry', 'saddle', 'thimble', 'velvet',
        'whistle', 'zephyr', 'basket', 'cobble', 'dagger', 'ember', 'fable', 'glacier', 'hollow', 'jasmine']))
CAPITALS = {'France': 'Paris', 'Japan': 'Tokyo', 'Italy': 'Rome', 'Egypt': 'Cairo', 'Canada': 'Ottawa', 'Kenya': 'Nairobi',
            'Peru': 'Lima', 'Norway': 'Oslo', 'Spain': 'Madrid', 'Greece': 'Athens'}
L = 'ABCDE'


def mc(opts):
    return 'Options:\n' + '\n'.join(f'({L[i]}) {o}' for i, o in enumerate(opts))


def g_alpha(r):
    ws = r.sample(WORDS, r.randint(4, 7))
    return (f"Put these words in dictionary order: {' '.join(ws)}", ' '.join(sorted(ws)), 'Compare the words letter by letter')


def g_longest(r):
    ws = r.sample(WORDS, 5); ans = max(ws, key=len)
    if sum(len(w) == len(ans) for w in ws) > 1: return g_longest(r)
    return (f"Which word in this list is the longest? {', '.join(ws)}", ans, 'Look at how many letters each word has')


def g_oddone(r):
    c1, c2 = r.sample(list(CATS), 2); ws = r.sample(CATS[c1], 3) + [r.choice(CATS[c2])]; r.shuffle(ws)
    odd = [w for w in ws if w in CATS[c2]][0]
    return (f"Which item does not belong with the others?\n{mc(ws)}", f'({L[ws.index(odd)]})', f'Three are a {c1}; {odd} is a {c2}')


def g_member(r):
    c = r.choice(list(CATS)); others = [w for k, v in CATS.items() if k != c for w in v]
    opts = r.sample(others, 3) + [r.choice(CATS[c])]; r.shuffle(opts); a = [o for o in opts if o in CATS[c]][0]
    return (f"Which of the following is a {c}?\n{mc(opts)}", f'({L[opts.index(a)]})', f'{a} is a {c}')


def g_capital(r):
    k = r.choice(list(CAPITALS)); opts = r.sample([v for x, v in CAPITALS.items() if x != k], 3) + [CAPITALS[k]]; r.shuffle(opts)
    return (f"What is the capital city of {k}?\n{mc(opts)}", f'({L[opts.index(CAPITALS[k])]})', f'The capital of {k} is {CAPITALS[k]}')


def g_syllog(r):
    a, b, c = r.sample(['glorps', 'wibbles', 'zants', 'morks', 'tavs', 'quibs', 'flens', 'drats'], 3)
    valid = r.random() < 0.5
    if valid: q = f"All {a} are {b}. All {b} are {c}. Does it follow that all {a} are {c}? Answer yes or no."
    else: q = f"All {a} are {b}. Some {b} are {c}. Does it follow that all {a} are {c}? Answer yes or no."
    return (q, 'yes' if valid else 'no', 'Check whether the chain of categories is complete')


def g_reverse(r):
    w = r.choice(WORDS)
    return (f"Spell the word '{w}' backwards.", w[::-1], 'Read the letters from the last to the first')


def g_story(r):
    n = r.choice(['Ama', 'Rui', 'Lena', 'Omar', 'Ines', 'Tariq']); item = r.choice(FRUIT + TOOL); place = r.choice(['market', 'workshop', 'garden', 'fair'])
    col = r.choice(COLOR)
    opts = r.sample([x for x in FRUIT + TOOL if x != item], 2) + [item]; r.shuffle(opts)
    q = (f"{n} walked to the {place} on a {r.choice(['sunny', 'windy', 'cold'])} morning, admired a {col} banner, chatted with a friend, "
         f"and finally bought a {item} before going home. What did {n} buy?\n{mc(opts)}")
    return (q, f'({L[opts.index(item)]})', f'The story says {n} bought a {item}')


def g_rhyme(r):
    pairs = [('cat', 'hat', ['dog', 'cup']), ('moon', 'spoon', ['star', 'fork']), ('tree', 'bee', ['leaf', 'ant']),
             ('cake', 'lake', ['pie', 'river']), ('ring', 'king', ['crown', 'bell']), ('boat', 'coat', ['ship', 'scarf'])]
    w, a, d = r.choice(pairs); opts = d + [a]; r.shuffle(opts)
    return (f"Which word rhymes with '{w}'?\n{mc(opts)}", f'({L[opts.index(a)]})', f'{a} ends with the same sound as {w}')


def g_antonym(r):
    pairs = [('hot', 'cold'), ('early', 'late'), ('full', 'empty'), ('loud', 'quiet'), ('rough', 'smooth'), ('brave', 'timid'),
             ('ancient', 'modern'), ('generous', 'stingy')]
    w, a = r.choice(pairs); d = r.sample([p[1] for p in pairs if p[0] != w], 2); opts = d + [a]; r.shuffle(opts)
    return (f"Which word means the opposite of '{w}'?\n{mc(opts)}", f'({L[opts.index(a)]})', f'The opposite of {w} is {a}')


GENS = [g_alpha, g_longest, g_oddone, g_member, g_capital, g_syllog, g_reverse, g_story, g_rhyme, g_antonym]
