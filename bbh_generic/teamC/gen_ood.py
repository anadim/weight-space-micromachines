"""Out-of-style settings (protocol rule 12): the same tools used on text that does not look like the BBH tasks.
Own 3-shot prompts (written here) + generated questions (seed 3), evaluated with run_gen.py --prompt_file.

  num_shop   word problems about buying things (prices, quantities, change)          tool: expr_gen in prose
  num_list   a shopping/inventory list with multi-digit quantities; total count     tool: expr_gen (sum)
  bool_code  a code-like condition with variable values given; does it hold?        tool: bool_gen

python gen_ood.py OUTDIR   -> OUTDIR/ood/{task}.json, OUTDIR/ood/{task}.prompt.txt
"""
import json, os, random, sys

ITEMS = [('notebook', 'notebooks'), ('pen', 'pens'), ('lamp', 'lamps'), ('ticket', 'tickets'), ('towel', 'towels'), ('plant', 'plants'),
         ('cable', 'cables'), ('mug', 'mugs'), ('shirt', 'shirts'), ('bag of rice', 'bags of rice'), ('battery pack', 'battery packs'),
         ('chair', 'chairs'), ('candle', 'candles'), ('jar of honey', 'jars of honey'), ('board game', 'board games')]
NAMES = ['Amara', 'Bilal', 'Chen', 'Dagny', 'Esteban', 'Farida', 'Goran', 'Hana', 'Ikaika', 'Jun', 'Kwame', 'Lucia', 'Mihail', 'Noor',
         'Oskar', 'Priya', 'Quentin', 'Rasa', 'Sione', 'Tamar']

SHOP_PROMPT = """Solve the shopping problem. Write each calculation, then give the answer.

Q: Lena buys 4 scarves at 13 dollars each and 3 hats at 8 dollars each. She pays with a 100-dollar bill. How much change does she get?
A: Let's think step by step.
The scarves and hats cost 4 * 13 + 3 * 8 = 76. The change is 100 - 76 = 24. So the answer is 24.

Q: A school orders 12 boxes of chalk with 25 sticks in each box, and 30 loose sticks. How many sticks does it get?
A: Let's think step by step.
The number of sticks is 12 * 25 + 30 = 330. So the answer is 330.

Q: Tomas had 250 dollars. He bought 6 books at 17 dollars each and then earned 45 dollars. How much money does he have now?
A: Let's think step by step.
He has 250 - 6 * 17 + 45 = 193. So the answer is 193.

"""

LIST_PROMPT = """Count the total number of items.

Q: Inventory: 14 forks, 9 spoons, 23 plates, and 6 bowls. How many items are there in total?
A: Let's think step by step.
We add the quantities: 14 + 9 + 23 + 6 = 52. So the answer is 52.

Q: My bag holds 3 apples, 2 bananas, and 11 grapes. How many pieces of fruit are in it?
A: Let's think step by step.
We add the quantities: 3 + 2 + 11 = 16. So the answer is 16.

Q: Packed for the trip: 7 shirts, 18 socks, 4 jackets, 2 hats, and 9 books. How many things were packed?
A: Let's think step by step.
We add the quantities: 7 + 18 + 4 + 2 + 9 = 40. So the answer is 40.

"""

BOOL_PROMPT = """Decide whether the condition in the code holds.

Q: Given a = True, b = False. Does `a and not b` evaluate to True?
A: Let's think step by step.
Substituting the values: True and not False = True. So the answer is True.

Q: Given ready = False, admin = True, beta = True. Does `ready or (admin and not beta)` evaluate to True?
A: Let's think step by step.
Substituting the values: False or (True and not True) = False. So the answer is False.

Q: Given x = True, y = True. Does `not (x and y) or not x` evaluate to True?
A: Let's think step by step.
Substituting the values: not (True and True) or not True = False. So the answer is False.

"""

VARS = ['enabled', 'debug', 'is_admin', 'paid', 'logged_in', 'ready', 'cached', 'verbose', 'locked', 'online', 'p', 'q', 'r', 'flag']


def shop(r):
    n = r.choice([2, 2, 3]); its = r.sample(ITEMS, n)
    qs = [r.randint(2, 19) for _ in its]; ps = [r.randint(3, 98) for _ in its]
    total = sum(q * p for q, p in zip(qs, ps))
    name = r.choice(NAMES)
    parts = [f'{q} {pl} at {p} dollars each' for (s, pl), q, p in zip(its, qs, ps)]
    buy = ', '.join(parts[:-1]) + ' and ' + parts[-1]
    if r.random() < 0.5:
        pay = (total // 100 + 1 + r.randint(0, 3)) * 100
        return f'{name} buys {buy}. {name} pays with {pay} dollars. How much change does {name} get?', str(pay - total)
    return f'{name} buys {buy}. How much does {name} spend in total?', str(total)


def lst(r):
    n = r.randint(3, 8); its = r.sample(ITEMS, n); qs = [r.randint(2, 95) for _ in its]
    L = ', '.join(f'{q} {pl}' for (s, pl), q in zip(its[:-1], qs[:-1])) + f', and {qs[-1]} {its[-1][1]}'
    head = r.choice(['Inventory:', 'My shopping list:', 'In the storeroom there are', 'The order contains'])
    return f'{head} {L}. How many items are there in total?', str(sum(qs))


def bexpr(r, vs, d=0):
    n = r.randint(1, 3); parts = []
    for i in range(n):
        a = '(' + bexpr(r, vs, d + 1) + ')' if d < 2 and r.random() < 0.3 else r.choice(vs)
        parts.append(' '.join(['not'] * r.choice([0, 0, 1]) + [a]))
        if i < n - 1: parts.append(r.choice(['and', 'or']))
    return ' '.join(parts)


def bcode(r):
    k = r.randint(2, 4); vs = r.sample(VARS, k); vals = {v: r.random() < 0.5 for v in vs}
    while True:
        e = bexpr(r, vs)
        if len(e.split()) >= 4: break
    given = ', '.join(f'{v} = {vals[v]}' for v in vs)
    return f'Given {given}. Does `{e}` evaluate to True?', str(eval(e, {}, vals))


if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(f'{out}/ood', exist_ok=True)
    for t, g, pr in (('num_shop', shop, SHOP_PROMPT), ('num_list', lst, LIST_PROMPT), ('bool_code', bcode, BOOL_PROMPT)):
        r = random.Random(f'{t}-3'); ex = []
        while len(ex) < 60:
            q, a = g(r)
            if q not in {e['input'] for e in ex}: ex.append({'input': q, 'target': a})
        json.dump({'examples': ex}, open(f'{out}/ood/{t}.json', 'w'), indent=1)
        open(f'{out}/ood/{t}.prompt.txt', 'w').write(pr)
        print(t, ex[0], ex[1])
