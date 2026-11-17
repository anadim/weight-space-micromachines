"""Our own generator for 'who holds what after a sequence of swaps' questions (tracking_shuffled_objects-like surface form),
plus two other styles for the generality checks (rule 12):
  style 'bbh'   : BBH-like multiple choice ("... At the end of the X, NAME has the\nOptions:\n(A) ...")
  style 'alt'   : TRAIN-ONLY other style: lockers/desks inventory with swaps AND replacements, and "who has the X?" questions
                  (uses the tool's set op and @ query)
  style 'oos'   : EVAL-ONLY out-of-style setting never trained on: a card game (trades, discards + draws), different wording,
                  questions both "which card does X hold" and "who holds card Y"
Splits: 'train' (seed 2), 'dev' (seed 0), 'test' (seed 1; adds EXTRA names/items/paraphrases never used in train/dev).
Each item: {'input', 'target', 'gold_call', 'solution' (tool-format completion, train), 'cot' (plain CoT, for few-shot prefixes)}.
"""
import random
from vocab import NAMES_BASE, NAMES_EXTRA, COLORS_BASE, COLORS_EXTRA, pick
from callfmt import CS, CE, OS, OE, track_ref

LET = 'ABCDEFGHIJ'


def art(x): return ('an ' if x[0].lower() in 'aeiou' else 'a ') + x


def join(xs): return xs[0] if len(xs) == 1 else (', '.join(xs[:-1]) + (',' if len(xs) > 2 else '') + ' and ' + xs[-1])


# ---- scenarios: who the people are, what they hold (base items / test-only items), the noun used in swap sentences
SCEN = [
    dict(group='hikers on a mountain trail', event='hike', each='each carrying a water bottle', has='{n} has {i}', noun='bottle',
         items=[f'{c} bottle' for c in COLORS_BASE], extra=[f'{c} bottle' for c in COLORS_EXTRA], end='has the', prog='Along the way, pairs of hikers swap bottles.'),
    dict(group='coworkers sharing an office', event='week', each='each with a plant on the desk', has='{n} has {i}', noun='plant',
         items=['cactus', 'fern', 'bonsai', 'orchid', 'succulent', 'peace lily', 'snake plant', 'rubber plant', 'jade plant', 'ivy', 'palm', 'basil pot'],
         extra=['monstera', 'begonia', 'aloe', 'bamboo stalk'], end='has the', prog='During the week, the coworkers keep trading plants.'),
    dict(group='neighbors on the same street', event='summer', each='each having borrowed one tool', has='{n} has {i}', noun='tool',
         items=['hammer', 'ladder', 'drill', 'rake', 'wheelbarrow', 'saw', 'shovel', 'hedge trimmer', 'sander', 'wrench', 'stapler gun', 'level'],
         extra=['chisel', 'crowbar', 'leaf blower', 'pickaxe'], end='has the', prog='Over the summer, the neighbors pass tools around.'),
    dict(group='children at a summer camp', event='camp', each='each with a toy', has='{n} has {i}', noun='toy',
         items=['kite', 'yo-yo', 'puzzle', 'spinning top', 'jump rope', 'frisbee', 'toy robot', 'rubber duck', 'kaleidoscope', 'harmonica', 'marble bag', 'slinky'],
         extra=['boomerang', 'pinwheel', 'rag doll', 'toy drum'], end='has the', prog='As camp goes on, the children trade toys.'),
    dict(group='cooks in a restaurant kitchen', event='shift', each='each in charge of one dish', has='{n} is making {i}', noun='dish',
         items=['risotto', 'curry', 'paella', 'goulash', 'ramen', 'lasagna', 'falafel', 'ceviche', 'pierogi', 'tagine', 'chowder', 'gumbo'],
         extra=['moussaka', 'bibimbap', 'jollof rice', 'borscht'], end='is making the', prog='As orders pile up, the cooks swap dishes.', noart=True),
    dict(group='musicians in a band', event='rehearsal', each='each playing an instrument', has='{n} is playing {i}', noun='instrument',
         items=['violin', 'cello', 'flute', 'trumpet', 'banjo', 'harp', 'oboe', 'tuba', 'ukulele', 'clarinet', 'mandolin', 'bassoon'],
         extra=['sitar', 'accordion', 'xylophone', 'bagpipe'], end='is playing the', prog='During rehearsal, the musicians exchange instruments.'),
    dict(group='students in a chemistry lab', event='lab session', each='each paired with a lab partner', has='{n} is working with {i}', noun='partner',
         items=None, extra=None, end='is working with', prog='During the session, the students change partners.', people=True),
    dict(group='travelers at a train station', event='morning', each='each holding a ticket', has='{n} has a ticket to {i}', noun='ticket',
         items=['Lisbon', 'Krakow', 'Osaka', 'Nairobi', 'Lima', 'Hanoi', 'Tbilisi', 'Oslo', 'Cusco', 'Accra', 'Porto', 'Riga'],
         extra=['Valparaiso', 'Samarkand', 'Windhoek', 'Ljubljana'], end='has the ticket to', prog='While waiting, pairs of travelers swap tickets.', city=True),
    dict(group='actors in a school play', event='play', each='each cast in a role', has='{n} plays {i}', noun='role',
         items=['the narrator', 'the villain', 'the jester', 'the queen', 'the sailor', 'the ghost', 'the baker', 'the knight', 'the wizard', 'the gardener', 'the detective', 'the pirate'],
         extra=['the oracle', 'the shepherd', 'the inventor', 'the dragon'], end='plays', prog='During rehearsals, the director has actors trade roles.', role=True),
    dict(group='friends at a costume party', event='party', each='each wearing a hat', has='{n} is wearing {i}', noun='hat',
         items=['beret', 'fedora', 'sombrero', 'top hat', 'beanie', 'cowboy hat', 'bowler hat', 'turban', 'panama hat', 'sun hat', 'bucket hat', 'fez'],
         extra=['tricorn', 'deerstalker', 'boater', 'ushanka'], end='is wearing the', prog='Throughout the party, guests keep switching hats.'),
]

SWAP_BASE = ['{a} and {b} swap {N}s.', '{a} and {b} trade {N}s.', '{a} and {b} switch {N}s.', '{a} and {b} exchange {N}s.',
             '{a} swaps {N}s with {b}.', '{a} trades {N}s with {b}.']
SWAP_EXTRA = ['{a} exchanges {N}s with {b}.', '{a} and {b} decide to swap their {N}s.', '{a} hands {b} their {N} and takes {b}\'s {N} in return.',
              '{a} and {b} give each other their {N}s.']
FIRST_BASE, MID_BASE, LAST_BASE = ['First'], ['Then', 'Then', 'Next', 'After that'], ['Finally']
FIRST_EXTRA, MID_EXTRA, LAST_EXTRA = ['To begin'], ['Later', 'Subsequently'], ['Lastly', 'At last']
END_BASE = ['At the end of the {E}, {x} {V}']
END_EXTRA = ['When the {E} is over, {x} {V}', 'By the end of the {E}, {x} {V}']


def value_text(s, it):
    if s.get('people') or s.get('city') or s.get('role') or s.get('noart'): return it
    return art(it)


def option_text(s, it):
    if s.get('role'): return it[4:] if it.startswith('the ') else it
    return it


def make_bbh(r, n, split):
    test = split == 'test'
    s = r.choice(SCEN)
    names = pick(r, NAMES_BASE, NAMES_EXTRA, 2 * n, split)
    people, rest = names[:n], names[n:]
    items = rest if s.get('people') else pick(r, s['items'], s['extra'], n, split)
    k = n if r.random() < 0.6 else max(2, n + r.choice([-1, 1, 1, 2]))
    swaps = []
    for _ in range(k):
        i, j = r.sample(range(1, n + 1), 2); swaps.append((i, j))
    ask = r.randint(1, n)
    sw_bank = SWAP_BASE + (SWAP_EXTRA if test else [])
    verb = r.choice(sw_bank)
    sents = []
    for t, (i, j) in enumerate(swaps):
        if t == 0: con = r.choice(FIRST_BASE + (FIRST_EXTRA if test else []))
        elif t == k - 1: con = r.choice(LAST_BASE + (LAST_EXTRA if test else []))
        else: con = r.choice(MID_BASE + (MID_EXTRA if test else []))
        v = verb if r.random() < 0.8 else r.choice(sw_bank)
        sents.append(f'{con}, ' + v.format(a=people[i - 1], b=people[j - 1], N=s['noun']))
    end = r.choice(END_BASE + (END_EXTRA if test else [])).format(E=s['event'], x=people[ask - 1], V=s['end'])
    q = (f"{join(people)} are {s['group']}. At the start of the {s['event']}, they are {s['each']}: "
         + join([s['has'].format(n=p, i=value_text(s, it)) for p, it in zip(people, items)]) + '.\n'
         + s['prog'] + ' ' + ' '.join(sents) + ' ' + end)
    order = list(range(n))
    if r.random() < 0.25: r.shuffle(order)                     # options usually in start order (BBH-like), sometimes shuffled
    opts = [option_text(s, items[o]) for o in order]
    q += '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    call = f"track {''.join(str(i) for i in range(1, n + 1))}:" + ','.join(f'{i}{j}' for i, j in swaps) + f'?{ask}'
    res = int(track_ref(call)); final = items[res - 1]
    letter = LET[order.index(res - 1)]
    sol = (" Let's track the " + s['noun'] + "s with the state tool.\n"
           + 'Slots: ' + ', '.join(f'{t + 1} = {p}' for t, p in enumerate(people)) + '.\n'
           + 'Start values: ' + ', '.join(f'{t + 1} = {option_text(s, it)}' for t, it in enumerate(items)) + '.\n'
           + ''.join(f'{people[i - 1]} and {people[j - 1]} swap: {i}{j}.\n' for i, j in swaps)
           + f'Question: what does {people[ask - 1]} (slot {ask}) have at the end? ?{ask}\n'
           + f'{CS}{call}{CE}{OS}{res}{OE}\n'
           + f'Value {res} = {option_text(s, final)}. So the answer is ({letter}).')
    # plain CoT in the style of the official prompt (used only to build few-shot prefixes for training)
    st = list(items); lines = ['(0) At the start: ' + ', '.join(f'{p}: {option_text(s, x)}' for p, x in zip(people, st)) + '.']
    for t, (i, j) in enumerate(swaps):
        st[i - 1], st[j - 1] = st[j - 1], st[i - 1]
        lines.append(f'({t + 1}) {people[i - 1]} and {people[j - 1]} swap {s["noun"]}s: ' + ', '.join(f'{p}: {option_text(s, x)}' for p, x in zip(people, st)) + '.')
    cot = ' Let\'s think step by step.\n' + '\n'.join(lines) + f'\n{end} {option_text(s, final)}. So the answer is ({letter}).'
    return dict(input=q, target=f'({letter})', gold_call=call, gold_result=str(res), solution=sol, cot=cot, n=n, style='bbh')


# ---------------------------------------------------------------- alt (train only): lockers with swaps + replacements, "who has X?"
ALT_ITEMS = ['scarf', 'umbrella', 'camera', 'thermos', 'notebook', 'radio', 'compass', 'flashlight', 'sketchbook', 'pocket knife', 'binoculars',
             'tennis racket', 'skateboard', 'lunchbox', 'snow globe', 'chess set']


def make_alt(r, n, split):
    names = pick(r, NAMES_BASE, NAMES_EXTRA, n, split)
    items = r.sample(ALT_ITEMS, min(len(ALT_ITEMS), n + 3)); start, spare = items[:n], items[n:]
    vals = list(range(1, n + 1)); nextv = n + 1; vname = {i + 1: start[i] for i in range(n)}
    ops, sents = [], []
    for t in range(r.randint(n - 1, n + 2)):
        if r.random() < 0.3 and spare and nextv <= 9:
            i = r.randint(1, n); new = spare.pop(); vname[nextv] = new
            ops.append(f'{i}={nextv}')
            sents.append(r.choice(['{a} puts the {o} away and takes {x} instead.', '{a} replaces the {o} in the locker with {x}.']).format(
                a=names[i - 1], o=vname[vals[i - 1]], x=art(new)))
            vals[i - 1] = nextv; nextv += 1
        else:
            i, j = r.sample(range(1, n + 1), 2); ops.append(f'{i}{j}')
            sents.append(r.choice(['{a} and {b} swap the contents of their lockers.', '{a} trades lockers\' contents with {b}.']).format(a=names[i - 1], b=names[j - 1]))
            vals[i - 1], vals[j - 1] = vals[j - 1], vals[i - 1]
    q = (f'{join(names)} each have a locker at the gym. ' + ' '.join(f"{p}'s locker holds {art(x)}." for p, x in zip(names, start)) + ' '
         + 'Over the month the following happens. ' + ' '.join(sents) + ' ')
    if r.random() < 0.5:                                             # who has item X?
        v = r.choice(vals); q += f'Whose locker holds the {vname[v]} at the end of the month?'
        call = f"track {''.join(str(i) for i in range(1, n + 1))}:" + ','.join(ops) + f'@{v}'
        res = int(track_ref(call)); ans = names[res - 1]; opts = list(names); qline = f'Question: who has value {v} ({vname[v]}) at the end? @{v}\n'
        tail = f'Slot {res} = {ans}.'
    else:
        a = r.randint(1, n); q += f"What is in {names[a - 1]}'s locker at the end of the month?"
        call = f"track {''.join(str(i) for i in range(1, n + 1))}:" + ','.join(ops) + f'?{a}'
        res = int(track_ref(call)); ans = vname[res]; opts = sorted({vname[v] for v in vname}, key=lambda x: r.random())[:max(n, 3)]
        if ans not in opts: opts[0] = ans
        r.shuffle(opts); qline = f"Question: what is in {names[a - 1]}'s locker (slot {a}) at the end? ?{a}\n"; tail = f'Value {res} = {ans}.'
    letter = LET[opts.index(ans)]
    q += '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    sol = (" Let's track the lockers with the state tool.\n" + 'Slots: ' + ', '.join(f'{t + 1} = {p}' for t, p in enumerate(names)) + '.\n'
           + 'Values: ' + ', '.join(f'{v} = {vname[v]}' for v in sorted(vname) if v < nextv) + '.\n'
           + ''.join(f'{s} {o}.\n' for s, o in zip(sents, ops)) + qline + f'{CS}{call}{CE}{OS}{res}{OE}\n' + f'{tail} So the answer is ({letter}).')
    return dict(input=q, target=f'({letter})', gold_call=call, gold_result=str(res), solution=sol, cot=None, n=n, style='alt')


# ---------------------------------------------------------------- alt2 (train only): variables in a short program (digit values, copy op)
VARS = list('abcdefghkmnpqrstuvwxyz')


def make_code(r, n, split):
    names = r.sample(VARS, n); st = [r.randint(0, 9) for _ in range(n)]
    ops, lines = [], []
    for _ in range(r.randint(n - 1, n + 3)):
        t = r.random(); i, j = r.sample(range(1, n + 1), 2)
        if t < 0.5: ops.append(f'{i}{j}'); lines.append(r.choice([f'{names[i - 1]}, {names[j - 1]} = {names[j - 1]}, {names[i - 1]}', f'swap({names[i - 1]}, {names[j - 1]})']))
        elif t < 0.75: v = r.randint(0, 9); ops.append(f'{i}={v}'); lines.append(f'{names[i - 1]} = {v}')
        else: ops.append(f'{i}<{j}'); lines.append(f'{names[i - 1]} = {names[j - 1]}')
    init = '; '.join(f'{v} = {x}' for v, x in zip(names, st))
    q = (r.choice(['A short program first sets', 'A script starts with']) + f' {init}. Then it runs these statements in order: ' + '; '.join(lines) + '. ')
    vals = list(st)
    for op in ops:
        if '=' in op: vals[int(op[0]) - 1] = int(op[2])
        elif '<' in op: vals[int(op[0]) - 1] = vals[int(op[2]) - 1]
        else: i, j = int(op[0]) - 1, int(op[1]) - 1; vals[i], vals[j] = vals[j], vals[i]
    once = [v for v in set(vals) if vals.count(v) == 1]
    base = f"track {''.join(map(str, st))}:" + ','.join(ops)
    if once and r.random() < 0.4:
        v = r.choice(once); q += f'Which variable holds the value {v} at the end?'; call = base + f'@{v}'
        res = track_ref(call); ans = names[int(res) - 1]; opts = sorted(names); tail = f'Slot {res} = {ans}.'; qline = f'Question: which variable holds {v}? @{v}\n'
    else:
        a = r.randint(1, n); q += f'What is the value of {names[a - 1]} at the end?'; call = base + f'?{a}'
        res = track_ref(call); ans = res; opts = sorted({str(x) for x in vals} | {str(x) for x in r.sample(range(10), 3)})[:max(n, 3)]
        if ans not in opts: opts[-1] = ans
        r.shuffle(opts); tail = f'Value = {res}.'; qline = f'Question: value of {names[a - 1]} (slot {a})? ?{a}\n'
    letter = LET[opts.index(ans)]
    q += '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    sol = (" Let's track the variables with the state tool.\n" + 'Slots: ' + ', '.join(f'{t + 1} = {v}' for t, v in enumerate(names)) + '.\n'
           + 'Start values (the digits themselves): ' + ''.join(map(str, st)) + '.\n'
           + ''.join(f'{l}: {o}.\n' for l, o in zip(lines, ops)) + qline + f'{CS}{call}{CE}{OS}{res}{OE}\n{tail} So the answer is ({letter}).')
    return dict(input=q, target=f'({letter})', gold_call=call, gold_result=res, solution=sol, cot=None, n=n, style='code')


# ---------------------------------------------------------------- oos (eval only): card game
RANKS = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king']
SUITS = ['hearts', 'spades', 'clubs', 'diamonds']


def make_oos(r, n, split):
    names = pick(r, NAMES_BASE, NAMES_EXTRA, n, split)
    cards = r.sample([f'{a} of {b}' for a in RANKS for b in SUITS], n + 3); start, spare = cards[:n], cards[n:]
    vals = list(range(1, n + 1)); nextv = n + 1; vname = {i + 1: start[i] for i in range(n)}
    sents, ops = [], []
    for t in range(r.randint(n - 1, n + 2)):
        if r.random() < 0.3 and spare and nextv <= 9:
            i = r.randint(1, n); new = spare.pop(); vname[nextv] = new
            sents.append(r.choice(['{a} throws away the {o} and draws the {x} from the deck.', '{a} discards their card and picks up the {x}.']).format(a=names[i - 1], o=vname[vals[i - 1]], x=new))
            ops.append(f'{i}={nextv}'); vals[i - 1] = nextv; nextv += 1
        else:
            i, j = r.sample(range(1, n + 1), 2)
            sents.append(r.choice(['{a} and {b} trade cards.', '{b} gives {a} a card and gets {a}\'s card back.', '{a} swaps hands with {b}.']).format(a=names[i - 1], b=names[j - 1]))
            ops.append(f'{i}{j}'); vals[i - 1], vals[j - 1] = vals[j - 1], vals[i - 1]
    q = (f'In a card game, every player holds exactly one card. {join(names)} are playing. They are dealt: '
         + join([f'{p} the {c}' for p, c in zip(names, start)]) + '. Then the game goes like this: ' + ' '.join(sents) + ' ')
    if r.random() < 0.5:
        v = r.choice(vals); q += f'Who ends the game holding the {vname[v]}?'; ans = names[vals.index(v)]; opts = list(names); r.shuffle(opts)
        call = f"track {''.join(str(i) for i in range(1, n + 1))}:" + ','.join(ops) + f'@{v}'; kind = '@'
    else:
        a = r.randint(1, n); q += f'Which card does {names[a - 1]} hold when the game ends?'; ans = vname[vals[a - 1]]
        opts = [vname[v] for v in sorted(vname)]; r.shuffle(opts); kind = '?'
        call = f"track {''.join(str(i) for i in range(1, n + 1))}:" + ','.join(ops) + f'?{a}'
    letter = LET[opts.index(ans)]
    q += '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    return dict(input=q, target=f'({letter})', gold_call=call, gold_result=track_ref(call), solution=None, cot=None, n=n, style='oos', kind=kind)


def make(r, n, split, style='bbh'):
    return {'bbh': make_bbh, 'alt': make_alt, 'code': make_code, 'oos': make_oos}[style](r, n, split)


if __name__ == '__main__':
    r = random.Random(0)
    for st in ('bbh', 'alt', 'oos'):
        for n in (3, 5):
            x = make(r, n, 'test', st); print('=' * 80); print(x['input']); print('->', x['target']); print(x['solution']); print(x['cot'])
