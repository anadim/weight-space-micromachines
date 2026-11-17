"""Swap-tracker micromachine (MM2): reference semantics, BBH parser, and a generator of training stories.

Call format (all single, otherwise unused tokens in the Qwen2.5 vocab):
    <|quad_start|>N<|quad_end|>   -> state box  <|object_ref_start|>12..N<|object_ref_end|>      (new arrangement)
    <|quad_start|>IJ<|quad_end|>  -> state box with positions I and J swapped                      (swap)
    <|quad_start|>0I<|quad_end|>  -> box with the single digit at position I                       (read)
The state is a digit string; digit at position i = whose original item person i holds now.
"""
import re, json, random

CS2, CE2, OS2, OE2 = '<|quad_start|>', '<|quad_end|>', '<|object_ref_start|>', '<|object_ref_end|>'
NMAX = 7


def swap_ref(call: str, state):
    """call: the text inside the call; state: content of the most recent box (None if there is none).
    Returns the new box content ('' = error)."""
    if not re.fullmatch(r'\d{1,2}', call): return ''
    if len(call) == 1:
        n = int(call)
        return ''.join(str(i) for i in range(1, n + 1)) if 1 <= n <= NMAX else ''
    if state is None or not re.fullmatch(r'\d{1,%d}' % NMAX, state): return ''
    a, b = int(call[0]), int(call[1]); n = len(state)
    if a == 0:
        return state[b - 1] if 1 <= b <= n else ''
    if not (1 <= a <= n and 1 <= b <= n and a != b): return ''
    s = list(state); s[a - 1], s[b - 1] = s[b - 1], s[a - 1]
    return ''.join(s)


def last_state(text: str):
    """Content of the most recent box in a transcript (None if no box yet)."""
    m = re.findall(re.escape(OS2) + '(.*?)' + re.escape(OE2), text, flags=re.S)
    return m[-1] if m else None


# ---------------------------------------------------------------- BBH tracking_shuffled_objects
BBH_NAMES = ['Alice', 'Bob', 'Claire', 'Dave', 'Eve', 'Fred', 'Gertrude']


def parse_bbh(inp: str):
    """-> (people, swaps [(i,j)], asked person index, n options). Indices are 1-based story order."""
    first = inp.split('. ')[0]
    people = [n for n in BBH_NAMES if re.search(r'\b%s\b' % n, first)]
    people.sort(key=lambda n: first.index(n))
    idx = {n: i + 1 for i, n in enumerate(people)}
    swaps = [(idx[a], idx[b]) for a, b in re.findall(r'(?:First|Then|Finally), (\w+) and (\w+) ', inp)]
    q = re.search(r'At the end of the \w+, (\w+) ', inp).group(1)
    return people, swaps, idx[q], len(re.findall(r'\n\([A-G]\)', inp))


def simulate(n, swaps, ask):
    st = swap_ref(str(n), None)
    for i, j in swaps: st = swap_ref(f'{i}{j}', st)
    return int(swap_ref(f'0{ask}', st))


# ---------------------------------------------------------------- training stories (names and settings differ from BBH)
NAMES = ['Liam', 'Noah', 'Olivia', 'Emma', 'Ava', 'Mia', 'Lucas', 'Ethan', 'Mason', 'Logan', 'Sofia', 'Isla', 'Zoe', 'Nora', 'Leo', 'Owen', 'Ruby', 'Ivy',
         'Hugo', 'Felix', 'Maya', 'Nina', 'Omar', 'Priya', 'Quinn', 'Rosa', 'Tariq', 'Uma', 'Victor', 'Wendy', 'Xavier', 'Yara', 'Zane', 'Hana', 'Kenji',
         'Lars', 'Ines', 'Joel', 'Kira', 'Mateo']
PARTNERS = ['Tom', 'Greta', 'Pablo', 'Anika', 'Dmitri', 'Selma', 'Rafael', 'Bianca', 'Stefan', 'Lucia', 'Andre', 'Marta']
SETTINGS = [
    dict(intro='{P} are neighbors who each look after a pet.', start='At the start of the week, {A}.', has='{n} is looking after the {i}',
         mid='During the week, they keep trading pets.', verbs=['swap pets', 'trade pets', 'exchange pets'], end='At the end of the week, {n} is looking after the',
         items=['parrot', 'hamster', 'turtle', 'rabbit', 'ferret', 'gecko', 'canary', 'goldfish', 'kitten', 'puppy']),
    dict(intro='{P} are musicians in a band.', start='At the start of rehearsal, they each pick up an instrument: {A}.', has='{n} has the {i}',
         mid='As rehearsal goes on, pairs of musicians exchange instruments.', verbs=['swap instruments', 'exchange instruments', 'trade instruments'],
         end='At the end of rehearsal, {n} has the', items=['violin', 'cello', 'flute', 'trumpet', 'drum', 'banjo', 'harp', 'oboe', 'tuba', 'guitar']),
    dict(intro='{P} are guests at a costume party.', start='When the party begins, each of them is wearing a hat: {A}.', has='{n} has the {i}',
         mid='During the party, guests keep switching hats.', verbs=['swap hats', 'switch hats', 'trade hats'], end='At the end of the party, {n} has the',
         items=['red hat', 'blue hat', 'green hat', 'yellow hat', 'purple hat', 'orange hat', 'white hat', 'black hat', 'pink hat', 'brown hat']),
    dict(intro='{P} work at the same company.', start='At the start of the quarter, each of them has one project: {A}.', has='{n} is working on {i}',
         mid='Over the quarter, the projects get reassigned several times.', verbs=['trade projects', 'swap projects', 'switch projects'],
         end='At the end of the quarter, {n} is working on', items=['Project Falcon', 'Project Otter', 'Project Maple', 'Project Comet', 'Project Ember', 'Project Tundra', 'Project Lotus', 'Project Quartz']),
    dict(intro='{P} are cooks in a busy kitchen.', start='At the start of the shift, each of them is making one dish: {A}.', has='{n} is making the {i}',
         mid='As orders pile up, the cooks keep handing dishes over to each other.', verbs=['switch dishes', 'swap dishes', 'trade dishes'],
         end='At the end of the shift, {n} is making the', items=['soup', 'risotto', 'salad', 'curry', 'pasta', 'omelette', 'stew', 'dumplings', 'pie']),
    dict(intro='{P} are valet drivers at a hotel.', start='At the start of the evening, each of them holds the key to one car: {A}.', has='{n} has the key to the {i}',
         mid='Throughout the evening, the drivers pass keys around.', verbs=['swap keys', 'trade keys', 'exchange keys'], end='At the end of the evening, {n} has the key to the',
         items=['Volvo', 'Fiat', 'Honda', 'Jeep', 'Tesla', 'Mazda', 'Skoda', 'Lexus', 'Saab']),
    dict(intro='{P} are the crew of a small ship.', start='At the start of the voyage, each of them has a role: {A}.', has='{n} is the {i}',
         mid='As the voyage goes on, crew members trade roles from time to time.', verbs=['trade roles', 'switch roles', 'swap roles'], end='At the end of the voyage, {n} is the',
         items=['navigator', 'cook', 'captain', 'engineer', 'lookout', 'medic', 'pilot', 'purser', 'deckhand']),
    dict(intro='{P} are students in a lab class.', start='At the start of the class, each of them has a lab partner: {A}.', has='{n} is working with {i}',
         mid='During the class, the students change partners several times.', verbs=['switch partners', 'swap partners', 'trade partners'],
         end='At the end of the class, {n} is working with', items=PARTNERS),
]


def join(xs): return ', '.join(xs[:-1]) + (',' if len(xs) > 2 else '') + ' and ' + xs[-1]


def make_story(r: random.Random):
    n = r.randint(3, NMAX); s = r.choice(SETTINGS)
    people = r.sample(NAMES, n); items = r.sample(s['items'], n)
    k = max(2, n + r.choice([0, 0, 0, -1, 1, 2]))
    swaps = [tuple(r.sample(range(1, n + 1), 2)) for _ in range(k)]
    ask = r.randint(1, n); verb = r.choice(s['verbs'])
    lines = []
    for t, (i, j) in enumerate(swaps):
        w = 'First' if t == 0 else ('Finally' if t == k - 1 else 'Then')
        lines.append(f'{w}, {people[i - 1]} and {people[j - 1]} {verb}.')
    story = (s['intro'].format(P=join(people)) + ' ' + s['start'].format(A=join([s['has'].format(n=p, i=it) for p, it in zip(people, items)])) + '\n'
             + s['mid'] + ' ' + ' '.join(lines) + ' ' + s['end'].format(n=people[ask - 1]) + '\nOptions:\n'
             + '\n'.join(f'({chr(65 + i)}) {it}' for i, it in enumerate(items)))
    return story, people, items, swaps, ask


def render(people, items, swaps, ask):
    """The call-format worked solution (boxes are filled with the reference)."""
    box = lambda c, st: (lambda out: (f'{CS2}{c}{CE2}{OS2}{out}{OE2}', out))(swap_ref(c, st))
    n = len(people); b, st = box(str(n), None)
    out = ['People: ' + ', '.join(f'{i + 1} {p}' for i, p in enumerate(people)) + '.',
           'Start: ' + ', '.join(f'{i + 1} {it}' for i, it in enumerate(items)) + '. ' + b]
    for i, j in swaps:
        b, st = box(f'{i}{j}', st); out.append(f'{people[i - 1]} and {people[j - 1]}: {b}')
    b, res = box(f'0{ask}', st)
    out.append(f'{people[ask - 1]} is {ask}: {b}')
    out.append(f'{res} is {items[int(res) - 1]}, option ({chr(64 + int(res))}).')
    return '\n'.join(out), f'({chr(64 + int(res))})'


def make_rows(n, seed):
    r = random.Random(seed); rows = []
    for _ in range(n):
        story, people, items, swaps, ask = make_story(r)
        body, ans = render(people, items, swaps, ask)
        rows.append({'prompt': f'Question: {story}\nAnswer:', 'completion': f' {body}\n#### {ans}', 'kind': 'swap', 'question': story, 'answer': f'#### {ans}'})
    return rows


if __name__ == '__main__':
    import sys
    rows = make_rows(4000, 11); dev = make_rows(300, 12)
    with open('data/swap_train.jsonl', 'w') as f:
        for x in rows: f.write(json.dumps(x) + '\n')
    with open('data/swap_dev.jsonl', 'w') as f:
        for x in dev: f.write(json.dumps(x) + '\n')
    print(rows[0]['prompt'] + rows[0]['completion'])
    for path in sys.argv[1:]:                                   # BBH files: check the parser + simulator against the gold labels
        ex = json.load(open(path))['examples']; ok = 0; order = 0
        with open('data/bbh_' + path.split('objects_')[1].split('_')[0] + '.jsonl', 'w') as f:
            for e in ex:
                people, swaps, ask, nopt = parse_bbh(e['input'])
                res = simulate(len(people), swaps, ask)
                ok += f'({chr(64 + res)})' == e['target'] and nopt == len(people)
                f.write(json.dumps({'question': e['input'], 'answer': '#### ' + e['target'], 'gold_calls': [str(len(people))] + [f'{i}{j}' for i, j in swaps] + [f'0{ask}']}) + '\n')
        print(path.split('/')[-1], f'parser+simulator matches gold on {ok}/{len(ex)}')
