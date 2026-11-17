"""Zero-training install: FROZEN base Qwen2.5-0.5B + the calculator micromachine, used only through in-context examples.

Call format = ordinary tokens the base model already has (GSM8K-annotation style):
    ... <<4193*679= 2847047>> The answer is 2847047.
The model writes ' <<' A op B '='; the machine writes ' ', the digits, '>>'. No Qwen weight, embedding or vocabulary entry
changes; the only addition is the machine's hand-written weights.

python calc_icl.py N        -> accuracy with / without the machine on two sets, call statistics, off-call bit-identity
"""
import sys, os, re, json, random, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import calcnet
from calcnet import attach_calcnet, SPECS
from calcfmt import calc_ref

# the same circuit, keyed to plain-text tokens
SPECS['calcpt'] = dict(SPECS['calc'], text={**SPECS['calc']['text'], 'CS': ' <<', 'CE': '=', 'OS': ' ', 'OE': '>>'})

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
torch.manual_seed(0)
MODEL = os.environ.get('MODEL', 'Qwen/Qwen2.5-0.5B'); NOMM = os.environ.get('NOMM') == '1'   # NOMM: a bigger model, no machine
dev = os.environ.get('DEV', 'cpu')
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev).eval()
base_params = None if NOMM else {k: v.clone() for k, v in model.state_dict().items()}

SHOTS = [("What is 23 * 47?", "23*47", "1081"),
         ("A box holds 18 pens. How many pens are in 7 boxes?", "18*7", "126"),
         ("What is 3150 + 862?", "3150+862", "4012"),
         ("Tom had 905 stickers and gave away 278. How many are left?", "905-278", "627"),
         ("What is 615 * 342?", "615*342", "210330"),
         ("A farm has 46 rows of 129 trees. How many trees are there?", "46*129", "5934")]
PREFIX = ''.join(f"Q: {q}\nA: <<{e}= {r}>> The answer is {r}.\n\n" for q, e, r in SHOTS)


def num(r, lo, hi):
    d = r.randint(lo, hi); return r.randint(10 ** (d - 1), 10 ** d - 1)


def sets():
    r = random.Random(7)
    bare = []
    for _ in range(N):
        a, b = num(r, 3, 6), num(r, 2, 4)
        bare.append((f"What is {a} * {b}?", a * b))
    T = [("A crate holds {a} apples. A store has {b} crates. How many apples does the store have?", lambda a, b: a * b),
         ("A school has {a} students. Each student buys {b} pencils. How many pencils are bought?", lambda a, b: a * b),
         ("A tank holds {a} liters. {b} liters leak out. How many liters are left?", lambda a, b: a - b),
         ("A library had {a} books and received {b} more. How many books does it have now?", lambda a, b: a + b)]
    word = []
    for _ in range(N):
        t, f = r.choice(T); a, b = num(r, 3, 5), num(r, 2, 4)
        if f(a, b) < 0: a, b = b, a
        word.append((t.format(a=a, b=b), f(a, b)))
    return {'bare_mult': bare, 'word': word}


@torch.no_grad()
def generate(q, use, max_new=48):
 
    ids = tok(PREFIX + f"Q: {q}\nA:", return_tensors='pt').input_ids.to(dev)
    if A is not None: A.pending = None                                    # detached: the hook adds nothing
    if use: A.prime(ids)
    o = model(input_ids=ids, use_cache=True); past = o.past_key_values; lg = o.logits[:, -1]
    gen = []
    for _ in range(max_new):
        t = int(lg.argmax(-1)); gen.append(t)
        txt = tok.decode(gen)
        if '\n' in txt: break
        if use: A.step(torch.tensor([t], device=dev))
        o = model(input_ids=torch.tensor([[t]], device=dev), past_key_values=past, use_cache=True); past = o.past_key_values; lg = o.logits[:, -1]
    return tok.decode(gen)


def score(out, gold):
    m = re.search(r'The answer is\s*(-?\d+)', out)
    call = re.search(r'<<\s*(-?\d+)\s*([*+\-])\s*(-?\d+)\s*=\s*(-?\d*)\s*>>', out)
    return dict(ok=bool(m) and int(m.group(1)) == gold, called=bool(call),
                box_right=bool(call) and call.group(4) != '' and calc_ref(f'{call.group(1)}{call.group(2)}{call.group(3)}') == call.group(4))


A = None
if NOMM:
    for name, data in sets().items():
        t0 = time.time(); rows = [dict(q=q, gold=g, out=(o := generate(q, False)), **score(o, g)) for q, g in data]
        print(f'{MODEL} {name:10s} no_machine acc {100*sum(r["ok"] for r in rows)/len(rows):5.1f}% | calls {100*sum(r["called"] for r in rows)/len(rows):5.1f}% | {time.time()-t0:.0f}s', flush=True)
    sys.exit()
A = attach_calcnet(model, tok, machines=('calcpt',))
print('reader gaps (min)', round(min(A.gaps.values()), 3), '| head margins (min)', round(min(v[0] for v in A.margins.values()), 2), flush=True)
assert all(torch.equal(v, base_params[k]) for k, v in model.state_dict().items()), 'a Qwen weight changed'
print('Qwen weights: all identical to the downloaded base model', flush=True)

res = {}
for name, data in sets().items():
    for cond in ('no_machine', 'machine'):
        t0 = time.time(); rows = []
        for q, gold in data:
            out = generate(q, cond == 'machine')
            rows.append(dict(q=q, gold=gold, out=out, **score(out, gold)))
        acc = 100 * sum(r['ok'] for r in rows) / len(rows); called = 100 * sum(r['called'] for r in rows) / len(rows)
        boxr = sum(r['box_right'] for r in rows); nc = sum(r['called'] for r in rows)
        res[f'{name}/{cond}'] = dict(acc=acc, called=called, box_right=f'{boxr}/{nc}', rows=rows)
        print(f'{name:10s} {cond:10s} acc {acc:5.1f}% | wrote a well-formed call {called:5.1f}% | box correct {boxr}/{nc} | {time.time()-t0:.0f}s', flush=True)
        print('   e.g.', repr(rows[0]['out'][:110]), flush=True)

# off-call bit-identity: teacher-forced logits on ordinary text, with the machine attached vs. detached
texts = [json.loads(l)['question'] for l in open('test.jsonl')][:20]
same = 0
with torch.no_grad():
    for t in texts:
        ids = tok(t, return_tensors='pt').input_ids
        A.pending = None; l0 = model(input_ids=ids).logits
        A.prime(ids, full=True); l1 = model(input_ids=ids).logits; A.pending = None
        same += torch.equal(l0, l1)
print(f'OFF-CALL: {same}/{len(texts)} GSM8K questions give bit-identical logits at every position with the machine attached', flush=True)
json.dump(res, open(f'calc_icl_{N}.json', 'w'), indent=1)
