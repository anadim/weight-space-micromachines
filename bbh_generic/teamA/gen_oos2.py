"""oos2: a second OUT-OF-STYLE eval set (rule 12), written before fine-tune v3 and never trained on or used for development.
Tracker: (a) a cook's numbered pots (slots are numbered positions, digits in the text, swaps of contents + refills),
         (b) office desks (people as slots, desks as values, 'moves to' wording).
Ordering: (a) birthdays through the year (earlier/later, 'the day after', 'not in the first ...'),
          (b) cars in a row of parking spaces counted from the entrance (closer/farther, 'right next to').
"""
import random
from vocab import NAMES_BASE, NAMES_EXTRA, ORDINALS, NUMWORDS, pick
from callfmt import track_ref, order_ref
import gen_order

LET = 'ABCDEFGHIJ'; O = lambda k: ORDINALS[k - 1]
FOODS = ['soup', 'rice', 'beans', 'pasta', 'stew', 'porridge', 'lentils', 'noodles', 'chili', 'polenta', 'broth', 'curry', 'oatmeal']
DESKS = ['the desk by the window', 'the desk near the door', 'the corner desk', 'the standing desk', 'the desk by the plant', 'the desk under the clock',
         'the desk next to the printer', 'the round desk', 'the desk by the stairs']


def join(xs): return xs[0] if len(xs) == 1 else (', '.join(xs[:-1]) + (',' if len(xs) > 2 else '') + ' and ' + xs[-1])


def track_oos2(r, n):
    pots = r.random() < 0.5
    if pots:
        foods = r.sample(FOODS, n + 4); start, spare = foods[:n], foods[n:]; slots = [f'pot {i}' for i in range(1, n + 1)]
        text = f'A cook lines up {NUMWORDS.get(n, n)} pots on a long stove, numbered 1 to {n} from left to right. ' + ' '.join(f'Pot {i + 1} holds {x}.' for i, x in enumerate(start))
    else:
        names = pick(r, NAMES_BASE, NAMES_EXTRA, n, 'dev'); desks = r.sample(DESKS, n); start, spare = desks, []; slots = names
        text = f'In a small office, {join(names)} each have their own desk: ' + join([f'{p} sits at {d}' for p, d in zip(names, desks)]) + '.'
    vals = list(range(1, n + 1)); vname = {i + 1: start[i] for i in range(n)}; nv = n + 1; ops = []; ev = []
    for _ in range(r.randint(n - 1, n + 2)):
        if pots and spare and r.random() < 0.3 and nv <= 9:
            i = r.randint(1, n); z = spare.pop(); vname[nv] = z; ops.append(f'{i}={nv}')
            ev.append(f'The cook scrapes out pot {i} and starts {z} in it.'); vals[i - 1] = nv; nv += 1
        else:
            i, j = r.sample(range(1, n + 1), 2); ops.append(f'{i}{j}')
            ev.append(f'The cook pours pot {i} and pot {j} into each other\'s places.' if pots else r.choice([f'{slots[i - 1]} and {slots[j - 1]} move to each other\'s desks.', f'{slots[i - 1]} takes over the desk of {slots[j - 1]}, who moves to the desk {slots[i - 1]} left.']))
            vals[i - 1], vals[j - 1] = vals[j - 1], vals[i - 1]
    text += ' ' + ' '.join(ev) + ' '
    base = f"track {''.join(str(i) for i in range(1, n + 1))}:" + ','.join(ops)
    if r.random() < 0.5:
        a = r.randint(1, n); call = base + f'?{a}'; ans = vname[int(track_ref(call))]
        text += f'What is cooking in pot {a} now?' if pots else f'Where does {slots[a - 1]} sit now?'
        opts = sorted(set(vname.values()), key=lambda x: r.random())[:max(3, n)]
        if ans not in opts: opts[0] = ans
    else:
        v = r.choice(vals); call = base + f'@{v}'; ans = slots[int(track_ref(call)) - 1]
        text += f'Which pot has the {vname[v]} now?' if pots else f'Who sits at {vname[v]} now?'
        opts = list(slots)
    r.shuffle(opts); letter = LET[opts.index(ans)]
    text += '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    return dict(input=text, target=f'({letter})', gold_call=call, gold_result=track_ref(call), n=n, style='oos2', kind=call[-2], setting='pots' if pots else 'desks')


def order_oos2(r, n):
    bday = r.random() < 0.5
    names = pick(r, NAMES_BASE, NAMES_EXTRA, n, 'dev')
    perm, cons = gen_order.make_clues(r, n, ('<>', '+', '~', '=', '!'), 0.35)
    def say(op, a, b):
        A = names[a - 1]; B = names[b - 1] if op in ('<', '>', '+', '~') else None
        if bday:
            return {'<': f"{A}'s birthday is earlier in the year than {B}'s.", '>': f"{A}'s birthday falls later in the year than {B}'s.",
                    '+': f"{B}'s birthday is the very next one after {A}'s.", '~': f"{A}'s and {B}'s birthdays are consecutive in the list.",
                    '=': f"{A}'s birthday is the {O(b)} of the group's birthdays.", '=-': f"{A}'s birthday is the {O(b)}-to-last of the group's birthdays." if b > 1 else f"{A} has the last birthday of the year.",
                    '!=': f"{A}'s birthday is not the {O(b)} one.", '!=-': f"{A}'s birthday is not the {O(b)}-to-last one." if b > 1 else f"{A} does not have the last birthday."}[op]
        return {'<': f'{A} parked closer to the entrance than {B}.', '>': f'{A} parked farther from the entrance than {B}.',
                '+': f'{A} parked one space closer to the entrance than {B}, right next to them.', '~': f'{A} and {B} parked in neighboring spaces.',
                '=': f'{A} parked in space {b}, counting from the entrance.', '=-': f'{A} parked in the {O(b)} space counting from the far end.',
                '!=': f'{A} did not park in space {b}.', '!=-': f'{A} did not park in the {O(b)} space from the far end.'}[op]
    clues = [say(*c) for c in cons]
    if bday: text = f'{join(names)} have birthdays on {NUMWORDS[n]} different days of the year. List their birthdays in calendar order. ' + ' '.join(clues) + ' '
    else: text = f'A parking row has {NUMWORDS[n]} spaces, numbered 1 to {n} starting at the entrance. {join(names)} each parked in a different space. ' + ' '.join(clues) + ' '
    if r.random() < 0.6:
        k = r.randint(1, n); start = r.random() < 0.6; q = f'?{k}' if start else f'?-{k}'
        ans = names[(perm[k - 1] if start else perm[n - k]) - 1]; opts = list(names)
        text += (f'Whose birthday is the {O(k)} in calendar order?' if start else f'Whose birthday is the {O(k)}-to-last in calendar order?' if k > 1 else 'Whose birthday comes last?') if bday else (f'Who parked in space {k}?' if start else f'Who parked in the {O(k)} space from the far end?')
    else:
        a = r.randint(1, n); q = f'@{a}'; ans = O(perm.index(a) + 1); opts = [O(k) for k in range(1, n + 1)]
        text += f"Where does {names[a - 1]}'s birthday fall in calendar order?" if bday else f'Counting from the entrance, which space did {names[a - 1]} park in?'
    if q[0] != '@': r.shuffle(opts)
    letter = LET[opts.index(ans)]
    text += '\nOptions:\n' + '\n'.join(f'({LET[t]}) {o}' for t, o in enumerate(opts))
    call = f'order {n}:' + ','.join(f'{a}{op}{b}' for op, a, b in cons) + q
    return dict(input=text, target=f'({letter})', gold_call=call, gold_result=order_ref(call), n=n, style='oos2', kind=q[0], setting='birthdays' if bday else 'parking')


if __name__ == '__main__':
    import json
    r = random.Random(4); rows = []
    for tool, f in (('track', track_oos2), ('order', order_oos2)):
        for n in (3, 5, 7):
            for i in range(30):
                x = f(r, n); x.update(task=f'{tool}{n}', tool=tool, id=f'{tool}{n}_oos2_{i}'); rows.append(x)
    with open('data/oos2.jsonl', 'w') as fh:
        for x in rows: fh.write(json.dumps(x) + '\n')
    print(len(rows)); print(rows[0]['input'], rows[0]['target'], rows[0]['gold_call']); print(rows[100]['input'], rows[100]['target'], rows[100]['gold_call'])
