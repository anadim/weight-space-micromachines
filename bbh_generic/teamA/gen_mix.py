"""v3 (train-only) 'mix' styles: generic stories that exercise every operation and query kind of both tools with several
wordings each, in several kinds of settings, with optional irrelevant sentences and question-first layout. Written after
looking at FT-v1 failures on our OWN out-of-style set oos1 (replacements written as swaps; "who has X" asked as "what does
X have"; "immediately left of" written as '<'; "which seat" asked as ?k). The fixes are generic (more operations, wordings
and settings), not copies of oos1 templates; oos1 is therefore no longer held out, and a NEW held-out set oos2 (gen_oos2.py)
was written before training on this data.
"""
import random
from vocab import NAMES_BASE, NAMES_EXTRA, COLORS_BASE, ORDINALS, NUMWORDS, pick
from callfmt import CS, CE, OS, OE, track_ref, order_ref
import gen_order

LET = 'ABCDEFGHIJ'
O = lambda k: ORDINALS[k - 1]
FILLER = ['It was a sunny afternoon.', 'Nobody was in a hurry.', 'A radio was playing in the background.', 'Everyone agreed it had been a long day.',
          'The room smelled of fresh coffee.', 'There were 12 chairs in the room, but only some were used.', 'It had rained the night before.',
          'Someone mentioned that the event started at 9 a.m.']


def art(x): return ('an ' if x[0].lower() in 'aeiou' else 'a ') + x


def join(xs): return xs[0] if len(xs) == 1 else (', '.join(xs[:-1]) + (',' if len(xs) > 2 else '') + ' and ' + xs[-1])


def assemble(r, setup, events, question):
    """story sentences + optional filler + optional question-first layout."""
    ev = list(events)
    for _ in range(r.choice([0, 0, 1, 2])): ev.insert(r.randint(0, len(ev)), r.choice(FILLER))
    if r.random() < 0.2: return f'{question} {setup} ' + ' '.join(ev)
    return f'{setup} ' + ' '.join(ev) + f' {question}'


# ---------------------------------------------------------------- state tracker
HOLD_ITEMS = ['lantern', 'map', 'kettle', 'blanket', 'whistle', 'rope', 'basket', 'teapot', 'globe', 'clock', 'vase', 'sand timer', 'drum', 'mask',
              'helmet', 'mug', 'pillow', 'trophy', 'stopwatch', 'puzzle box']
CONTAINERS = [('box', [f'{c} box' for c in COLORS_BASE]), ('drawer', ['top drawer', 'middle drawer', 'bottom drawer', 'left drawer', 'right drawer', 'corner drawer', 'desk drawer', 'deep drawer']),
              ('jar', [f'jar number {i}' for i in range(1, 10)]), ('locker', [f'locker {c}' for c in 'ABCDEFGHJ'] + [f'locker {k}' for k in (12, 17, 23, 31, 46, 58)]),
              ('room', [f'room {k}' for k in (101, 102, 104, 205, 207, 210, 301, 315, 322)]), ('bin', [f'bin {k}' for k in (3, 8, 11, 14, 19, 27, 36, 40)]), ('shelf', ['first shelf', 'second shelf', 'third shelf', 'fourth shelf', 'fifth shelf', 'sixth shelf', 'seventh shelf'])]
CONTENTS = ['marbles', 'buttons', 'keys', 'coins', 'shells', 'stamps', 'beads', 'pencils', 'candles', 'batteries', 'seeds', 'ribbons', 'dice', 'corks']


def track_people(r, n):
    names = pick(r, NAMES_BASE, NAMES_EXTRA, n, 'train'); items = r.sample(HOLD_ITEMS, n + 4)
    start, spare = items[:n], items[n:]
    slots = names; label = lambda v: v
    swap_t = ['{a} and {b} swap items.', '{a} and {b} trade what they are holding.', '{a} gives the {x} to {b} and gets the {y} in return.',
              '{b} hands {a} the {y} and takes the {x}.', '{a} exchanges items with {b}.', 'The items of {a} and {b} are switched.']
    set_t = ['{a} puts down the {x} and picks up {z}.', '{a} trades in the {x} for {z} from the shelf.', '{a} loses the {x} and finds {z}.',
             '{a} replaces the {x} with {z}.', '{a} drops the {x}; {a} now carries {z}.']
    setup = f'{join(names)} are each holding one thing: ' + join([f'{p} has {art(x)}' for p, x in zip(names, start)]) + '.'
    qv = ['What is {a} holding at the end?', 'At the end, what does {a} have?', 'Which item does {a} end up with?']
    qs = ['Who is holding the {z} at the end?', 'Who ends up with the {z}?', 'At the end, who has the {z}?']
    return slots, start, spare, swap_t, set_t, setup, qv, qs, 'item', art


def track_containers(r, n):
    noun, labels = r.choice(CONTAINERS); slots = r.sample(labels, n); stuff = r.sample(CONTENTS, n + 4)
    start, spare = stuff[:n], stuff[n:]
    swap_t = ['The contents of the {a} and the {b} are swapped.', 'Someone swaps what is in the {a} with what is in the {b}.',
              'The {x} in the {a} and the {y} in the {b} change places.', 'The {a} and the {b} trade contents.']
    set_t = ['The {a} is emptied and refilled with {z}.', 'The {x} in the {a} are thrown out and replaced with {z}.', 'Someone puts {z} into the {a} after removing the {x}.']
    setup = f'There are {NUMWORDS.get(n, str(n))} {noun}s: ' + join([f'the {s} holds {x}' for s, x in zip(slots, start)]) + '.'
    qv = ['What is in the {a} at the end?', 'At the end, what does the {a} hold?']
    qs = ['Which {noun} holds the {z} at the end?', 'Where are the {z} at the end?']
    return slots, start, spare, swap_t, set_t, setup, [q.replace('{noun}', noun) for q in qv], [q.replace('{noun}', noun) for q in qs], noun, lambda x: x


SETTINGS = [('phone', 'ringtone', ['Chimes', 'Marimba', 'Old Bell', 'Birdsong', 'Harp', 'Radar', 'Ripples', 'Beacon', 'Silk', 'Night Owl', 'Pulse', 'Breeze']),
            ('radio', 'station', ['Jazz FM', 'News 24', 'Classic Rock', 'City Talk', 'Folk Hour', 'Sports Live', 'Weather Line', 'Soul Radio', 'Kids Club', 'Late Show']),
            ('laptop', 'wallpaper', ['the beach photo', 'the forest photo', 'the galaxy image', 'the city skyline', 'the cat picture', 'the mountain photo', 'the abstract pattern', 'the sunset photo', 'the map image'])]


def track_settings(r, n):
    dev, what, pool = r.choice(SETTINGS); names = pick(r, NAMES_BASE, NAMES_EXTRA, n, 'train'); vals = r.sample(pool, min(len(pool), n + 3))
    start, spare = vals[:n], vals[n:]
    swap_t = ['{a} and {b} swap ' + what + 's.', '{a} trades ' + what + 's with {b}.']
    set_t = ['{a} changes the ' + what + ' to {z}.', '{a} switches to {z}.', '{a} picks a new ' + what + ': {z}.']
    copy_t = ['{a} copies the ' + what + ' of {b}.', '{a} changes their ' + what + ' to match {b}\'s.', '{a} sets the same ' + what + ' as {b}.',
              '{a} likes the ' + what + ' of {b} and uses it too.']
    setup = f'{join(names)} each have a {dev}, and each {dev} has one {what}: ' + join([f"{p}'s is {x}" for p, x in zip(names, start)]) + '.'
    qv = ['What ' + what + ' does {a} have at the end?', 'At the end, which ' + what + ' is {a} using?']
    qs = ['Who has the ' + what + ' {z} at the end?', 'At the end, whose ' + what + ' is {z}?']
    return names, start, spare, swap_t, set_t, copy_t, setup, qv, qs, (lambda x: x)


def make_track_mix(r, n):
    u = r.random()
    if u < 0.4: slots, start, spare, swap_t, set_t, setup, qv, qs, unit, A = track_people(r, n); copy_t = []
    elif u < 0.7: slots, start, spare, swap_t, set_t, setup, qv, qs, unit, A = track_containers(r, n); copy_t = ['The {a} is refilled with the same thing as the {b}.']
    else: slots, start, spare, swap_t, set_t, copy_t, setup, qv, qs, A = track_settings(r, n)
    vals = list(range(1, n + 1)); vname = {i + 1: start[i] for i in range(n)}; nextv = n + 1; ops, ev, lines = [], [], []
    for _ in range(r.randint(max(2, n - 1), n + 2)):
        t = r.random()
        if t < 0.3 and spare and nextv <= 9:
            i = r.randint(1, n); z = spare.pop(); vname[nextv] = z
            ev.append(r.choice(set_t).format(a=slots[i - 1], x=vname[vals[i - 1]], z=A(z)))
            ops.append(f'{i}={nextv}'); lines.append(f'new value {nextv} = {z}; {i}={nextv}'); vals[i - 1] = nextv; nextv += 1
        elif t < 0.45 and copy_t:
            i, j = r.sample(range(1, n + 1), 2)
            ev.append(r.choice(copy_t).format(a=slots[i - 1], b=slots[j - 1])); ops.append(f'{i}<{j}'); lines.append(f'copy: {i}<{j}'); vals[i - 1] = vals[j - 1]
        elif t < 0.55 and n >= 3 and not copy_t:
            i, j, k = r.sample(range(1, n + 1), 3)                  # i -> j -> k -> i: each passes to the next
            ev.append(r.choice([f'{slots[i - 1]} passes their item to {slots[j - 1]}, {slots[j - 1]} passes theirs to {slots[k - 1]}, and {slots[k - 1]} passes theirs to {slots[i - 1]}, all at once.',
                                f'At the same moment, {slots[i - 1]} hands over to {slots[j - 1]}, {slots[j - 1]} to {slots[k - 1]}, and {slots[k - 1]} to {slots[i - 1]}.']))
            ops += [f'{i}{k}', f'{j}{k}']; lines.append(f'rotation {i}->{j}->{k}->{i} = two swaps: {i}{k},{j}{k}')
            vi, vj, vk = vals[i - 1], vals[j - 1], vals[k - 1]; vals[j - 1], vals[k - 1], vals[i - 1] = vi, vj, vk
        else:
            i, j = r.sample(range(1, n + 1), 2)
            ev.append(r.choice(swap_t).format(a=slots[i - 1], b=slots[j - 1], x=vname[vals[i - 1]], y=vname[vals[j - 1]]))
            ops.append(f'{i}{j}'); lines.append(f'{i}{j}'); vals[i - 1], vals[j - 1] = vals[j - 1], vals[i - 1]
    ev = [e[0].upper() + e[1:] for e in ev]
    base = f"track {''.join(str(i) for i in range(1, n + 1))}:" + ','.join(ops)
    if r.random() < 0.5:
        v = r.choice(vals); q = r.choice(qs).format(z=vname[v])
        if vals.count(v) != 1: v = None
    else: v = None
    if v is not None:
        call = base + f'@{v}'; res = track_ref(call)
        ans = slots[int(res) - 1]; opts = list(slots); qline = f'Question: which slot holds value {v} ({vname[v]})? @{v}\n'; tail = f'Slot {res} = {ans}.'
    else:
        a = r.randint(1, n); q = r.choice(qv).format(a=slots[a - 1]); call = base + f'?{a}'; res = track_ref(call)
        ans = vname[int(res)]; pool = [vname[v] for v in vname]; r.shuffle(pool); opts = pool[:max(3, min(len(pool), n))]
        if ans not in opts: opts[0] = ans
        qline = f'Question: what is in slot {a} ({slots[a - 1]})? ?{a}\n'; tail = f'Value {res} = {ans}.'
    assert res
    r.shuffle(opts); letter = LET[opts.index(ans)]
    text = assemble(r, setup, ev, q) + '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    sol = (" Let's track this with the state tool.\n" + 'Slots: ' + ', '.join(f'{t + 1} = {s}' for t, s in enumerate(slots)) + '.\n'
           + 'Values: ' + ', '.join(f'{v} = {vname[v]}' for v in range(1, n + 1)) + '.\n'
           + ''.join(f'{e} -> {o}\n' for e, o in zip(ev, lines)) + qline + f'{CS}{call}{CE}{OS}{res}{OE}\n{tail} So the answer is ({letter}).')
    return dict(input=text, target=f'({letter})', gold_call=call, gold_result=res, solution=sol, cot=None, n=n, style='mix')


# ---------------------------------------------------------------- ordering solver
# axis kinds: words for "earlier" (closer to position 1) / "later", "immediately", position phrases from each end
AXES = [
    dict(setup='{N} people stand in a queue: {L}.', unit='person', ends=('front', 'back'), bef=['is ahead of', 'stands in front of'], aft=['is behind', 'stands behind'],
         imm_b=['is directly in front of', 'is right in front of'], imm_a=['is directly behind', 'is right behind'], adj=['is next to', 'stands beside'],
         s=lambda k: 'is at the front' if k == 1 else f'is {O(k)} in the queue', e=lambda k: 'is at the back' if k == 1 else f'is {O(k)} from the back',
         qs=lambda k: 'Who is at the front?' if k == 1 else f'Who is {O(k)} in the queue?', qe=lambda k: 'Who is at the back?' if k == 1 else f'Who is {O(k)} from the back?',
         qa='In which position, counting from the front, is {a}?'),
    dict(setup='{N} paintings hang in a row on a wall: {L}.', unit='painting', obj=True, ends=('leftmost', 'rightmost'), bef=['hangs left of', 'is to the left of'], aft=['hangs right of', 'is to the right of'],
         imm_b=['hangs immediately to the left of', 'is just left of'], imm_a=['hangs immediately to the right of', 'is just right of'], adj=['hangs next to', 'is adjacent to'],
         s=lambda k: 'is the leftmost' if k == 1 else f'is {O(k)} from the left', e=lambda k: 'is the rightmost' if k == 1 else f'is {O(k)} from the right',
         qs=lambda k: 'Which painting is the leftmost?' if k == 1 else f'Which painting is {O(k)} from the left?', qe=lambda k: 'Which painting is the rightmost?' if k == 1 else f'Which painting is {O(k)} from the right?',
         qa='Counting from the left, what position does {a} hang in?'),
    dict(setup='In a quiz league, {N} teams finished the season with no ties: {L}.', unit='team', ends=('first', 'last'), bef=['finished above', 'ranked higher than'], aft=['finished below', 'ranked lower than'],
         imm_b=['finished one place above', 'ranked directly above'], imm_a=['finished one place below', 'ranked directly below'], adj=['finished next to', 'was in a neighboring place to'],
         s=lambda k: 'finished first' if k == 1 else f'finished {O(k)}', e=lambda k: 'finished last' if k == 1 else f'finished {O(k)} from the bottom',
         qs=lambda k: 'Which team finished first?' if k == 1 else f'Which team finished {O(k)}?', qe=lambda k: 'Which team finished last?' if k == 1 else f'Which team finished {O(k)} from the bottom?',
         qa='In which place did {a} finish?'),
    dict(setup='A workshop has {N} sessions, one after another: {L}.', unit='session', obj=True, ends=('first', 'last'), bef=['comes before', 'is earlier than'], aft=['comes after', 'is later than'],
         imm_b=['comes right before', 'is immediately before'], imm_a=['comes right after', 'is immediately after'], adj=['is next to', 'is back-to-back with'],
         s=lambda k: 'is first' if k == 1 else f'is the {O(k)} session', e=lambda k: 'is last' if k == 1 else f'is {O(k)} from the end',
         qs=lambda k: 'Which session is first?' if k == 1 else f'Which session is the {O(k)}?', qe=lambda k: 'Which session is last?' if k == 1 else f'Which session is {O(k)} from the end?',
         qa='In which position is {a} scheduled?'),
    dict(setup='{N} families live in one building, one family per floor: {L}.', unit='family', fam=True, ends=('top floor', 'ground floor'), bef=['lives above', 'lives higher than'], aft=['lives below', 'lives lower than'],
         imm_b=['lives directly above', 'lives one floor above'], imm_a=['lives directly below', 'lives one floor below'], adj=['lives on a floor next to', 'is a direct neighbor (one floor apart) of'],
         s=lambda k: 'lives on the top floor' if k == 1 else f'lives on the {O(k)} floor from the top', e=lambda k: 'lives on the ground floor' if k == 1 else f'lives on the {O(k)} floor from the bottom',
         qs=lambda k: 'Who lives on the top floor?' if k == 1 else f'Who lives on the {O(k)} floor from the top?', qe=lambda k: 'Who lives on the ground floor?' if k == 1 else f'Who lives on the {O(k)} floor from the bottom?',
         qa='Counting from the top, which floor do {a} live on?'),
]
PAINTINGS = ['seascape', 'portrait', 'still life', 'city view', 'mountain scene', 'abstract', 'forest scene', 'night sky', 'harbor view']
SESSIONS = ['keynote', 'panel', 'demo', 'lunch talk', 'poster session', 'workshop', 'tutorial', 'closing talk', 'lightning talks']
TEAMS = ['the Owls', 'the Comets', 'the Pumas', 'the Anchors', 'the Ravens', 'the Sparks', 'the Mavericks', 'the Tides', 'the Lynxes']



# v4: property rankings with comparatives NOT in the BBH-style scale list (generic English adjectives), both ends worded
PROPS = [('trails', ['the ridge trail', 'the river trail', 'the pine trail', 'the canyon trail', 'the lake loop', 'the summit path', 'the meadow path', 'the old mule track'], 'steeper', 'gentler', 'steepest', 'gentlest'),
         ('knives', ['the bread knife', 'the paring knife', 'the cleaver', 'the boning knife', 'the carving knife', 'the utility knife', 'the fillet knife'], 'sharper', 'duller', 'sharpest', 'dullest'),
         ('ropes', ['the red rope', 'the hemp rope', 'the nylon rope', 'the climbing rope', 'the jute rope', 'the cotton rope', 'the wire rope'], 'thicker', 'thinner', 'thickest', 'thinnest'),
         ('streets', ['Elm Street', 'Harbor Road', 'Mill Lane', 'Station Avenue', 'Church Row', 'Market Street', 'Bridge Road', 'Park Lane'], 'busier', 'quieter', 'busiest', 'quietest'),
         ('mattresses', ['the foam mattress', 'the spring mattress', 'the latex mattress', 'the futon', 'the air mattress', 'the wool mattress', 'the hybrid mattress'], 'firmer', 'softer', 'firmest', 'softest'),
         ('breads', ['the rye loaf', 'the baguette', 'the sourdough', 'the brioche', 'the bagel', 'the pita', 'the focaccia', 'the ciabatta'], 'fresher', 'staler', 'freshest', 'stalest'),
         ('wells', ['the north well', 'the south well', 'the old well', 'the stone well', 'the village well', 'the farm well', 'the garden well'], 'deeper', 'shallower', 'deepest', 'shallowest')]


def prop_axis(r):
    noun, objs, c1, c2, s1, s2 = r.choice(PROPS)
    k1 = lambda k: f'is the {s1}' if k == 1 else r.choice([f'is the {O(k)} {s1}', f'is the {O(k)}-{s1}', f'is {O(k)} in order of {s1[:-3] if s1.endswith("est") else s1}ness'][:2])
    k2 = lambda k: f'is the {s2}' if k == 1 else r.choice([f'is the {O(k)} {s2}', f'is the {O(k)}-{s2}'])
    return dict(setup=f'{{N}} {noun} are compared: {{L}}.', unit=noun, obj=True, objs=objs, ends=(s1, s2), bef=[f'is {c1} than'], aft=[f'is {c2} than'],
                imm_b=[f'is the next {c1[:-2] if False else c1} one after'] if False else [f'is just one step {c1} than'], imm_a=[f'is just one step {c2} than'], adj=['is next in the ranking to'],
                s=k1, e=k2, qs=lambda k: f'Which one is the {s1}?' if k == 1 else f'Which one is the {O(k)} {s1}?', qe=lambda k: f'Which one is the {s2}?' if k == 1 else f'Which one is the {O(k)} {s2}?',
                qa=f'Ranked from {s1} to {s2}, in which position is {{a}}?')


def make_order_mix(r, n):
    X = r.choice(AXES) if r.random() < 0.7 else prop_axis(r)
    if X.get('objs'): objs = r.sample(X['objs'], n); show = lambda o: o
    elif X.get('obj'): objs = r.sample(PAINTINGS if X['unit'] == 'painting' else SESSIONS, n); show = lambda o: 'the ' + o
    elif X['unit'] == 'team': objs = r.sample(TEAMS, n); show = lambda o: o
    elif X.get('fam'): objs = [f'the {nm}s' for nm in pick(r, NAMES_BASE, NAMES_EXTRA, n, 'train')]; show = lambda o: o
    else: objs = pick(r, NAMES_BASE, NAMES_EXTRA, n, 'train'); show = lambda o: o
    perm, cons = gen_order.make_clues(r, n, ('<>', '+', '~', '=', '!'), 0.35)
    def say(op, a, b):
        A = show(objs[a - 1]); B = show(objs[b - 1]) if op in ('<', '>', '+', '~') else None
        if op == '<': return f'{A} {r.choice(X["bef"])} {B}.'
        if op == '>': return f'{A} {r.choice(X["aft"])} {B}.'
        if op == '+': return f'{A} {r.choice(X["imm_b"])} {B}.' if r.random() < 0.5 else f'{B} {r.choice(X["imm_a"])} {A}.'
        if op == '~': return f'{A} {r.choice(X["adj"])} {B}.'
        if op == '=': return f'{A} {X["s"](b)}.'
        if op == '=-': return f'{A} {X["e"](b)}.' if (b == 1 or r.random() < 0.7) else f'{A} is {O(b)}-to-last.'
        neg = lambda ph: ph.replace('is ', 'is not ', 1) if ph.startswith('is ') else ('did not ' + ph.replace('finished', 'finish', 1) if ph.startswith('finished') else 'does not ' + ph.replace('lives', 'live', 1))
        return f'{A} {neg(X["s"](b) if op == "!=" else X["e"](b))}.'
    clues = [say(*c) for c in cons]
    clues = [c[0].upper() + c[1:] for c in clues]
    setup = X['setup'].format(N=NUMWORDS[n], L=join([show(o) for o in objs]))
    setup = setup[0].upper() + setup[1:]
    if r.random() < 0.65:
        k = r.randint(1, n); start = r.random() < 0.6
        q = X['qs'](k) if start else X['qe'](k); qs = f'?{k}' if start else f'?-{k}'
        ans_i = perm[k - 1] if start else perm[n - k]; opts = [show(o) for o in objs]; ans = show(objs[ans_i - 1]); r.shuffle(opts)
        tail = lambda res: f'Object {res} = {show(objs[int(res) - 1])}.'
    else:
        a = r.randint(1, n); q = X['qa'].format(a=show(objs[a - 1])); qs = f'@{a}'; opts = [O(k) for k in range(1, n + 1)]; ans = O(perm.index(a) + 1)
        tail = lambda res: f'Position {res} = {O(int(res))}.'
    letter = LET[opts.index(ans)]
    text = assemble(r, setup, clues, q) + '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    call = f'order {n}:' + ','.join(f'{a}{op}{b}' for op, a, b in cons) + qs; res = order_ref(call); assert res
    sol = (" Let's solve the ordering with the order tool.\n" + 'Objects: ' + ', '.join(f'{t + 1} = {o}' for t, o in enumerate(objs)) + '.\n'
           + f"Positions: 1 = {X['ends'][0]}, {n} = {X['ends'][1]}.\n" + ''.join(f'{c[:-1]}: {a}{op}{b}.\n' for c, (op, a, b) in zip(clues, cons))
           + f'Question: {qs}\n{CS}{call}{CE}{OS}{res}{OE}\n{tail(res)} So the answer is ({letter}).')
    return dict(input=text, target=f'({letter})', gold_call=call, gold_result=res, solution=sol, cot=None, n=n, style='mix')


if __name__ == '__main__':
    r = random.Random(1)
    for f in (make_track_mix, make_order_mix):
        for n in (3, 5, 7):
            x = f(r, n); print('=' * 70); print(x['input']); print(x['target']); print(x['solution'])
