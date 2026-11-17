"""Exactness of the natural ordering solver on ALL BBH logical_deduction questions + synthetic puzzles + off-task silence.
python test_natB2.py [N_SYNTH] [SEED]"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import json, os, random, sys, itertools, torch
from micromachines.circuits import deduction as L
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import SPECS, Attached
tok = AutoTokenizer.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base')); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base'), torch_dtype=torch.float32).to(dev).eval()
import time; t0 = time.time()
Att = Attached(model, tok, machines=('logic_nat',)); M = Att.nets['logic_nat']; E = model.get_input_embeddings(); outs = SPECS['logic_nat']['out']
print('attached', f'{time.time() - t0:.0f}s', flush=True)
def letter(ctx):
    ids = tok(ctx, add_special_tokens=False)['input_ids']
    with torch.no_grad(): e = M.prime_sym(M.symbols(E(torch.tensor([ids], device=dev))))[:, -1]
    return ('' if e.max() < 0.5 else '(' + outs[int(e.argmax())] + ')'), len(ids)
BBH = os.environ.get('BBH', 'data/bbh'); C = {'exact': 0, 'silent': 0, 'WRONG': 0, 'ref_silent': 0}; maxq = 0
for n in ('three', 'five', 'seven'):
    for e in json.load(open(f'{BBH}/data/logical_deduction_{n}_objects.json'))['examples']:
        ref = L.ld_ref(e['input'])
        if ref != e['target']: C['ref_silent'] += 1
        got, Lq = letter(f"\n\nQ: {e['input']}\nA: Let's think step by step.\n(1) ok.\nSo the answer is (")
        maxq = max(maxq, len(tok(e['input'])['input_ids']))
        if got == '': C['silent'] += 1
        elif got == e['target']: C['exact'] += 1
        else: C['WRONG'] += 1; print('WRONG', n, got, e['target'], ref, flush=True)
    print(n, C, f'{time.time() - t0:.0f}s', flush=True)
off = 0
for t in ('tracking_shuffled_objects_five_objects', 'date_understanding', 'temporal_sequences'):
    for e in json.load(open(f'{BBH}/data/{t}.json'))['examples'][:40]:
        got, _ = letter(f"\n\nQ: {e['input']}\nA: Let's think step by step.\nSo the answer is ("); off += got != ''
print('BBH logical_deduction', C, '| off-task answers (want 0):', off, '| longest question tokens', maxq, flush=True)
# synthetic puzzles from the same templates (random objects, random hidden order, constraints added until unique)
r = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0); N = int(sys.argv[1]) if len(sys.argv) > 1 else 100; bad = sil = 0
DOM = {'b': ('On a branch, there are {n} birds: ', ['blue jay', 'quail', 'falcon', 'owl', 'raven', 'robin', 'hawk', 'crow', 'hummingbird', 'cardinal'], 'is to the left of the', 'is to the right of the', [('is the leftmost', 1, 0), ('is the rightmost', 1, 1)] + [(f'is the {o} from the left', k, 0) for k, o in enumerate(['second', 'third', 'fourth'], 2)] + [(f'is the {o} from the right', k, 1) for k, o in enumerate(['second', 'third', 'fourth'], 2)], True),
       'g': ('In a golf tournament, there were {n} golfers: ', ['Ana', 'Eve', 'Ada', 'Dan', 'Rob', 'Amy', 'Joe', 'Mel', 'Mya', 'Eli'], 'finished above', 'finished below', [('finished first', 1, 0), ('finished last', 1, 1), ('finished second', 2, 0), ('finished third', 3, 0), ('finished second-to-last', 2, 1), ('finished third-to-last', 3, 1)], False),
       'f': ('A fruit stand sells {n} fruits: ', ['apples', 'peaches', 'mangoes', 'loquats', 'watermelons', 'kiwis', 'plums', 'oranges', 'pears', 'cantaloupes'], 'are more expensive than the', 'are less expensive than the', [('are the most expensive', 1, 0), ('are the cheapest', 1, 1), ('are the second-most expensive', 2, 0), ('are the second-cheapest', 2, 1), ('are the third-most expensive', 3, 0), ('are the third-cheapest', 3, 1)], True)}
NW = {3: 'three', 5: 'five', 7: 'seven'}
for it in range(N):
    dk = r.choice(list(DOM)); intro, pool, before, after, absl, the = DOM[dk]; n = r.choice([3, 5, 7])
    objs = r.sample(pool, n); order = r.sample(range(n), n); pos = {a: order.index(a) for a in range(n)}
    art = (lambda o: ('an ' if o[0] in 'aeiou' else 'a ') + o) if dk == 'b' else (lambda o: o)
    cap = (lambda o: 'The ' + o) if the else (lambda o: o)
    cons = []
    def nsol(cs):
        c = 0
        for perm in itertools.permutations(range(n)):
            ps = {a: p for p, a in enumerate(perm)}; ok = True
            for x in cs:
                if x[0] == 'r': ok &= ps[x[1]] < ps[x[2]]
                else: ok &= ps[x[1]] == ((n - x[2]) if x[3] else x[2] - 1)
            c += ok
        return c
    while nsol(cons) > 1:
        if r.random() < 0.5:
            a, b = r.sample(range(n), 2); cons.append(('r', a, b) if pos[a] < pos[b] else ('r', b, a))
        else:
            a = r.randrange(n); ph = [x for x in absl if (x[1] - 1 if not x[2] else n - x[1]) == pos[a]]
            if ph: cons.append(('a', a) + r.choice(ph)[1:])
    sents = []
    for x in cons:
        if x[0] == 'r':
            if r.random() < 0.5: sents.append(f'{cap(objs[x[1]])} {before} {objs[x[2]]}.')
            else: sents.append(f'{cap(objs[x[2]])} {after} {objs[x[1]]}.')
        else:
            ph = [p for p, k, b in absl if (k, b) == x[2:]][0]; sents.append(f'{cap(objs[x[1]])} {ph}.')
    ap, ak, ab = r.choice([x for x in absl if x[1] <= n])
    q = (f'The following paragraphs each describe a set of {NW[n]} objects arranged in a fixed order. The statements are logically consistent within each paragraph. '
         + intro.format(n=NW[n]) + ', '.join(art(o) for o in objs[:-1]) + ', and ' + art(objs[-1]) + '. ' + ' '.join(sents)
         + '\nOptions:\n' + '\n'.join(f'({L.LET[i]}) {cap(o)} {ap}' for i, o in enumerate(objs)))
    ref = L.ld_ref(q); got, _ = letter(f"\n\nQ: {q}\nA: Let's think step by step.\nSo the answer is (")
    if got == '': sil += 1
    elif got != ref: bad += 1; print('SYN-WRONG', got, ref, q[-300:])
print('synthetic', N, 'wrong', bad, 'silent', sil, f'{time.time() - t0:.0f}s')
assert C['WRONG'] == 0 and off == 0 and bad == 0, (C, off, bad)
print('PASS (silences are allowed: out of capacity)')
