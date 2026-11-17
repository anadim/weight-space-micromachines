"""Exactness of the natural tracking machine on ALL BBH tracking questions (and random synthetic ones):
teacher-forced prompt '...Q: <question>\\nA: ... So the answer is (' -> the machine's letter vs Python (tracking_ref) vs gold.
python test_natB.py [N_SYNTH]"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import json, os, random, sys, torch
from micromachines.circuits import tracking as T
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import SPECS, Attached
tok = AutoTokenizer.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base'))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base'), torch_dtype=torch.float32).to(dev).eval()
Att = Attached(model, tok, machines=('track_nat',)); M = Att.nets['track_nat']; I = Att.all_ids['track_nat']
E = model.get_input_embeddings(); outs = SPECS['track_nat']['out']
def letter(ctx):
    ids = tok(ctx, add_special_tokens=False)['input_ids']
    with torch.no_grad():
        e = M.prime_sym(M.symbols(E(torch.tensor([ids], device=dev))))[:, -1]
    return '' if e.max() < 0.5 else '(' + outs[int(e.argmax())] + ')', len(ids)
BBH = os.environ.get('BBH', 'data/bbh'); C = {'exact': 0, 'silent': 0, 'WRONG': 0}; maxlen = 0
pre = open(f'{BBH}/cot/tracking_shuffled_objects_three_objects.txt').read().split('-----', 1)[1].strip()
for n in ('three', 'five', 'seven'):
    for e in json.load(open(f'{BBH}/data/tracking_shuffled_objects_{n}_objects.json'))['examples']:
        ref = T.tracking_ref(e['input']); assert ref == e['target']
        got, L = letter(pre[-1500:] + f"\n\nQ: {e['input']}\nA: Let's think step by step.\n(1) blah.\nSo the answer is (")
        maxlen = max(maxlen, L)
        if got == '': C['silent'] += 1
        elif got == ref: C['exact'] += 1
        else: C['WRONG'] += 1; print('WRONG', n, got, ref, flush=True)
    print(n, C, flush=True)
# off-task: the machine must stay silent on other tasks' answer slots
off = 0
for t in ('logical_deduction_five_objects', 'date_understanding', 'web_of_lies'):
    for e in json.load(open(f'{BBH}/data/{t}.json'))['examples'][:40]:
        got, _ = letter(f"Q: {e['input']}\nA: Let's think step by step.\nSo the answer is (")
        off += got != ''
print('BBH tracking', C, '| off-task answers written (want 0):', off)
# synthetic: random people counts / swap orders, all three phrasings
r = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0); bad = 0; N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
for k in range(N):
    n = r.choice([3, 4, 5, 6, 7]); ppl = T.PEOPLE[:n]; items = [f'{c} ball' for c in r.sample(['red', 'blue', 'green', 'pink', 'white', 'black', 'orange', 'purple', 'brown'], n)]
    sw = [tuple(r.sample(ppl, 2)) for _ in range(n)]
    q = f"{', '.join(ppl[:-1])}, and {ppl[-1]} are playing a game. At the start of the game, they are each holding a ball: " + \
        ', '.join(f'{p} has a {it}' for p, it in zip(ppl[:-1], items[:-1])) + f", and {ppl[-1]} has a {items[-1]}.\nAs the game progresses, pairs of players trade balls. " + \
        ' '.join(f'{w}, {a} and {b} swap balls.' for w, (a, b) in zip(['First'] + ['Then'] * (n - 2) + ['Finally'], sw)) + \
        f" At the end of the game, {r.choice(ppl)} has the\nOptions:\n" + '\n'.join(f'({T.LET[i]}) {it}' for i, it in enumerate(items))
    got, _ = letter(f"Q: {q}\nA: Let's think step by step.\nSo the answer is ("); ref = T.tracking_ref(q)
    bad += got != ref
    if got != ref: print('SYN-MISMATCH', got, ref)
print('synthetic', N, 'mismatches', bad, '| longest prompt tokens', maxlen)
assert C['WRONG'] == 0 and off == 0 and bad == 0, (C, off, bad)
print('PASS')
