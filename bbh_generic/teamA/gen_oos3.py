"""oos3: a DEV-ONLY out-of-style set (seed 5) used to analyze FT-v3 failures before v4 (so oos2 and the judges' new rounds
stay held out). Tracker: buses at numbered bays (bay labels are not slot numbers; a bus is swapped out for a new one; who
is where), a band's setlist cards copied between music stands (copy op). Ordering: climbers on a rope (above/below,
directly above, not k-th), a scale ranking with comparatives not in training (deeper/shallower lakes; sweeter/sourer)."""
import random
from vocab import NAMES_BASE, NAMES_EXTRA, ORDINALS, NUMWORDS, pick
from callfmt import track_ref, order_ref
import gen_order

LET = 'ABCDEFGHIJ'; O = lambda k: ORDINALS[k - 1]
BUSES = ['the airport shuttle', 'the night bus', 'the school bus', 'the express', 'the tour coach', 'the minibus', 'the double-decker', 'the hospital shuttle', 'the park bus', 'the trolley']
SONGS = ['"Blue Morning"', '"Paper Boats"', '"Late Train"', '"Copper Sky"', '"Salt and Honey"', '"Northern Lights"', '"Small Hours"', '"River Song"', '"Glass House"']
LAKES = ['Lake Ardo', 'Lake Belvin', 'Lake Corra', 'Lake Dunmere', 'Lake Esk', 'Lake Fallow', 'Lake Grane', 'Lake Hollin']
FRUITS = ['the quince', 'the pomelo', 'the gooseberry', 'the persimmon', 'the tamarind', 'the lychee', 'the kumquat', 'the medlar']


def join(xs): return xs[0] if len(xs) == 1 else (', '.join(xs[:-1]) + (',' if len(xs) > 2 else '') + ' and ' + xs[-1])


def track3(r, n):
    if r.random() < 0.5:
        bays = [f'bay {b}' for b in sorted(r.sample(range(10, 40), n))]; buses = r.sample(BUSES, n + 3); start, spare = buses[:n], buses[n:]
        text = f'A bus station has {NUMWORDS.get(n, n)} bays in use: ' + join([f'{b} has {x}' for b, x in zip(bays, start)]) + '.'
        vals = list(range(1, n + 1)); vname = {i + 1: start[i] for i in range(n)}; nv = n + 1; ops = []; ev = []
        for _ in range(r.randint(n - 1, n + 2)):
            if spare and r.random() < 0.3 and nv <= 9:
                i = r.randint(1, n); z = spare.pop(); vname[nv] = z; ops.append(f'{i}={nv}'); ev.append(f'{vname[vals[i - 1]].capitalize()} leaves {bays[i - 1]} and {z} pulls in there.'); vals[i - 1] = nv; nv += 1
            else:
                i, j = r.sample(range(1, n + 1), 2); ops.append(f'{i}{j}'); ev.append(f'The buses at {bays[i - 1]} and {bays[j - 1]} switch bays.'); vals[i - 1], vals[j - 1] = vals[j - 1], vals[i - 1]
        base = f"track {''.join(str(i) for i in range(1, n + 1))}:" + ','.join(ops)
        if r.random() < 0.5:
            v = r.choice(vals); call = base + f'@{v}'; ans = bays[int(track_ref(call)) - 1]; q = f'At which bay is {vname[v]} now?'; opts = list(bays)
        else:
            a = r.randint(1, n); call = base + f'?{a}'; ans = vname[int(track_ref(call))]; q = f'Which bus is at {bays[a - 1]} now?'; opts = r.sample(list(vname.values()), min(len(vname), max(3, n)))
            if ans not in opts: opts[0] = ans
    else:
        names = pick(r, NAMES_BASE, NAMES_EXTRA, n, 'dev'); songs = r.sample(SONGS, n); vals = list(range(1, n + 1)); vname = {i + 1: songs[i] for i in range(n)}
        text = f'In a band, each musician has one song card on their stand: ' + join([f'{p} has {s}' for p, s in zip(names, songs)]) + '.'
        ops = []; ev = []
        for _ in range(r.randint(n - 1, n + 1)):
            i, j = r.sample(range(1, n + 1), 2)
            if r.random() < 0.35: ops.append(f'{i}<{j}'); ev.append(f'{names[i - 1]} throws away their card and copies the song on {names[j - 1]}\'s card.'); vals[i - 1] = vals[j - 1]
            else: ops.append(f'{i}{j}'); ev.append(f'{names[i - 1]} and {names[j - 1]} trade cards.'); vals[i - 1], vals[j - 1] = vals[j - 1], vals[i - 1]
        a = r.randint(1, n); call = f"track {''.join(str(i) for i in range(1, n + 1))}:" + ','.join(ops) + f'?{a}'; ans = vname[int(track_ref(call))]
        q = f'Which song is on {names[a - 1]}\'s card at the end?'; opts = list(songs)
    r.shuffle(opts); letter = LET[opts.index(ans)]
    text += ' ' + ' '.join(ev) + ' ' + q + '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    return dict(input=text, target=f'({letter})', gold_call=call, gold_result=track_ref(call), n=n, style='oos3')


def order3(r, n):
    kind = r.choice(['rope', 'lake', 'fruit'])
    objs = pick(r, NAMES_BASE, NAMES_EXTRA, n, 'dev') if kind == 'rope' else r.sample(LAKES if kind == 'lake' else FRUITS, n)
    perm, cons = gen_order.make_clues(r, n, ('<>', '+', '~', '=', '!') if kind == 'rope' else ('<>', '='), 0.35)
    W = {'rope': ('is higher on the rope than', 'is lower on the rope than', 'is directly above', 'is right next to', 'is the {k} climber from the top', 'is the {k} climber from the bottom'),
         'lake': ('is deeper than', 'is shallower than', None, None, 'is the {k} deepest', 'is the {k} shallowest'),
         'fruit': ('is sweeter than', 'is sourer than', None, None, 'is the {k} sweetest', 'is the {k} sourest')}[kind]
    cap = lambda s: s[0].upper() + s[1:]
    def say(op, a, b):
        A = cap(objs[a - 1]); B = objs[b - 1] if op in ('<', '>', '+', '~') else None
        if op == '<': return f'{A} {W[0]} {B}.'
        if op == '>': return f'{A} {W[1]} {B}.'
        if op == '+': return f'{A} {W[2]} {B}.'
        if op == '~': return f'{A} {W[3]} {B}.'
        k = O(b)
        if op == '=': return f'{A} {W[4].format(k=k)}.'
        if op == '=-': return f'{A} {W[5].format(k=k)}.'
        if op == '!=': return f'{A} ' + W[4].format(k=k).replace('is ', 'is not ', 1) + '.'
        return f'{A} ' + W[5].format(k=k).replace('is ', 'is not ', 1) + '.'
    intro = {'rope': f'{join(objs)} are climbing a cliff on one rope, one above another.', 'lake': f'A map shows {NUMWORDS[n]} lakes: {join(objs)}.',
             'fruit': f'A taster ranks {NUMWORDS[n]} fruits from sweetest to sourest: {join(objs)}.'}[kind]
    k = r.randint(1, n); start = r.random() < 0.5
    q = W[4 if start else 5].format(k=O(k)); q = f'Which one {q}?' if kind != 'rope' else f'Who {q}?'
    ans = objs[(perm[k - 1] if start else perm[n - k]) - 1]; opts = list(objs); r.shuffle(opts); letter = LET[opts.index(ans)]
    call = f'order {n}:' + ','.join(f'{a}{op}{b}' for op, a, b in cons) + (f'?{k}' if start else f'?-{k}')
    text = intro + ' ' + ' '.join(say(*c) for c in cons) + ' ' + q + '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    return dict(input=text, target=f'({letter})', gold_call=call, gold_result=order_ref(call), n=n, style='oos3', setting=kind)


if __name__ == '__main__':
    import json
    r = random.Random(5); rows = []
    for tool, f in (('track', track3), ('order', order3)):
        for n in (3, 5, 7):
            for i in range(20):
                x = f(r, n); x.update(task=f'{tool}{n}', tool=tool, id=f'{tool}{n}_oos3_{i}'); rows.append(x)
    with open('data/oos3.jsonl', 'w') as fh:
        for x in rows: fh.write(json.dumps(x) + '\n')
    print(len(rows)); print(rows[3]['input'], rows[3]['target']); print(rows[70]['input'], rows[70]['target'])
