"""Step 1 of the integration: ALL eight tools attached to ONE host (integ.attach_all). Three checks.

(1) OWN CALLS + NO CROSS-FIRING (machine level, real readers from the host's embeddings): for every call, every one of the
    eight machines is driven incrementally on  CONTEXT + <tool_call>CALL</tool_call>  (its own writes fed back). Pass iff
    the owner writes exactly <tool_response>REF</tool_response> (REF = the team's Python reference; for REF == '' the empty
    response or silence is accepted = fail-safe) and the seven other machines write NOTHING. Calls: each team's own
    machine-alone test pools (A: test_tools.rand_track/rand_order + MALFORMED; B: test_tools.fuzz_call/corrupt (copied);
    C: callfmt_C.random_call + EDGE_CASES) + gold calls of each team's fresh-test / out-of-style data; three contexts, the
    third holding earlier answered calls of OTHER tools.
(2) END-TO-END through the host's own norm + LM head: greedy generation after  CONTEXT<tool_call>CALL</tool_call>  with all
    eight machines attached must start with <tool_response>REF</tool_response>.
(3) BIT-IDENTITY with no call: logits of the host with all tools attached (hook active, every prompt position primed)
    vs. detached, on texts with no tool call (the 27 official BBH 3-shot CoT prompts + team questions). Must be equal bit
    for bit, and every machine's write must be exactly 0.

python test_integ.py OUT.json [--model M] [--n_rand 150] [--n_gold 80] [--n_gen 40]
"""
import os, sys, json, random, time, argparse, importlib.util, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _paths import G, COT
import integ
from integ import attach_all, prep_call_tokens, TOOLS, CALL_NAME, ref
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.specs import SPECS
from micromachines.machine import first
import callfmt_C as CC, tools_ref as RB
from tools_ref import fmt_time

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--n_rand', type=int, default=150); ap.add_argument('--n_gold', type=int, default=80); ap.add_argument('--n_gen', type=int, default=40)
ap.add_argument('--bit_texts', type=int, default=60)
a = ap.parse_args()
CS, CE, OS, OE = '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
if a.model == 'Qwen/Qwen3-0.6B-Base': prep_call_tokens(model, tok)      # the four envelope tokens need distinct rows
t0 = time.time(); NET = attach_all(model, tok); print(f'attached {TOOLS} ({time.time() - t0:.0f}s)', flush=True)
E = model.get_input_embeddings()
NAME2M = {v: k for k, v in CALL_NAME.items()}

# ---------------- call pools ----------------
spec = importlib.util.spec_from_file_location('A_test_tools', f'{G}/teamA/test_tools.py')
AT = importlib.util.module_from_spec(spec); spec.loader.exec_module(AT)


def fuzz_B(tool, r):            # copied from teamB/test_tools.py fuzz_call
    if tool == 'interval':
        ap_ = r.random() < 0.6; hi = 24 if ap_ else 99
        a_ = r.randint(0, hi - 1); b_ = r.randint(a_ + 1, hi)
        f = (lambda h: fmt_time(h, True)) if ap_ else (lambda h: str(h))
        busy = []
        for _ in range(r.randint(1, 9)):
            x = r.randint(max(0, a_ - 3), hi - 1); y = r.randint(x + 1, min(hi, x + 8)); busy.append(f'{f(x)}-{f(y)}')
        return f'interval {f(a_)}-{f(b_)} busy ' + ', '.join(busy)
    if tool == 'walk':
        c = []
        for _ in range(r.randint(1, 20)):
            c.append(r.choice(['turn left', 'turn right', 'turn around']) if r.random() < 0.4 else f"{r.choice(['forward', 'backward', 'left', 'right'])} {r.randint(1, 99)}")
        return 'walk ' + ', '.join(c)
    n = r.randint(1, 12); ks = list(range(1, n + 1)); r.shuffle(ks); cl = []
    for k in ks:
        if r.random() < 0.3: cl.append(f"{k} {r.choice(['true', 'false'])}")
        else: cl.append(f"{k} says {r.randint(1, 12)} {r.choice(['true', 'false'])}")
    return 'truth ' + '; '.join(cl)


def corrupt_B(call, r):         # copied from teamB/test_tools.py corrupt
    ops = [lambda c: c.replace(',', ' and', 1), lambda c: c.replace(';', ' and', 1), lambda c: c + ' please',
           lambda c: c.replace('-', ' to ', 1), lambda c: c.replace(' 1', ' 123', 1), lambda c: c.replace('turn', 'spin', 1),
           lambda c: c.replace(', ', ' ', 1), lambda c: c.replace('; ', ' ', 1), lambda c: c.replace(' forward', ' forward forward', 1),
           lambda c: c.replace(' says', ' says says', 1), lambda c: c + ' 3pm', lambda c: c + ', 7', lambda c: c.replace(' busy', '', 1),
           lambda c: c.replace(' busy', ' busy busy', 1), lambda c: c + '; 2', lambda c: c.replace(' true', '', 1)]
    for _ in range(20):
        c2 = r.choice(ops)(call)
        if c2 != call: return c2
    return None


r = random.Random(2026)
pool = []                                                    # (label, call)
for _ in range(a.n_rand): pool.append(('A.rand_track', AT.rand_track(r)))
for _ in range(a.n_rand): pool.append(('A.rand_order', AT.rand_order(r)))
for k, v in AT.MALFORMED.items(): pool += [(f'A.malformed_{k}', c) for c in v]
for t in ('interval', 'walk', 'truth'):
    for _ in range(a.n_rand):
        c = fuzz_B(t, r)
        if t == 'interval' and RB.run_ref(c) and len(RB.run_ref(c).split(', ')) > 4: continue
        pool.append((f'B.fuzz_{t}', c))
        if r.random() < 0.3:
            c2 = corrupt_B(c, r)
            if c2: pool.append((f'B.corrupt_{t}', c2))
pool += [('B.other', c) for c in ['interval of 3 numbers', 'walking 3', 'truthful']]
for _ in range(2 * a.n_rand): pool.append(('C.random', CC.random_call(r)))
pool += [('C.edge', c) for c in CC.EDGE_CASES + ['calc $4.50 * 3', 'calc (2 + 3) * 1.5', 'calc x * 2', 'calc 2.5 * 4 apples']]
gold = [json.loads(l)['gold_call'] for f in ('test', 'oos2') for l in open(f'{G}/teamA/data/{f}.jsonl')]
pool += [('A.gold', c) for c in r.sample(gold, a.n_gold)]
gold = [json.loads(l)['call'] for f in ('temporal_test', 'navigate_test', 'lies_test', 'meeting_oos', 'robot_oos', 'knights_oos')
        for l in open(f'{G}/teamB/data/{f}.jsonl')]
pool += [('B.gold', c) for c in r.sample(gold, a.n_gold)]
gold = [m for l in open(f'{G}/teamC/data/train/train_C_100.jsonl') for m in CC.CALL_RE.findall(json.loads(l)['completion'])]
pool += [('C.gold', c) for c in r.sample(gold, min(a.n_gold, len(gold)))]
CTX = ['', 'Some text before. 12, 3: ',
       f'{CS}track 12:12{CE}{OS}21{OE} then {CS}interval 9am-5pm busy 10am-12pm{CE}{OS}9am-10am, 12pm-5pm{OE} and '
       f'{CS}expr 3 * (4 + 5){CE}{OS}27{OE}, so {CS}walk forward 3, turn left{CE}{OS}(0, 3) facing west{OE} next ']


@torch.no_grad()
def drive(name, ctx, maxn=120):
    M = NET.nets[name]; I = NET.all_ids[name]; outs = SPECS[name]['out']; ot = [first(I[s]) for s in outs]
    ids = tok(ctx, add_special_tokens=False)['input_ids']; out = []
    e = M.prime_sym(M.symbols(E(torch.tensor([ids], device='cuda'))))
    early = float(e[0, :-1].abs().sum())                     # writes before </tool_call> (inside the context)
    e = e[:, -1]
    for _ in range(maxn):
        if e.abs().max() < 0.5: break
        j = int(e.argmax()); out.append(ot[j])
        if ot[j] == tok.convert_tokens_to_ids(OE): break
        e = M.step_sym(M.symbols(E(torch.tensor([out[-1]], device='cuda'))))[:, 0]
    return tok.decode(out), early


res = {'check1': {}, 'fail1': [], 'check2': {}, 'fail2': [], 'check3': {}}
t0 = time.time()
ncall = len(tok(CTX[2], add_special_tokens=False)['input_ids'])


@torch.no_grad()
def silent_others(owner, text, start):
    """teacher forcing over the whole transcript: every non-owner machine must write exactly 0 at every position >= start
    (start = first token of this call; context 2's earlier calls are answered by their own owners)."""
    ids = torch.tensor([tok(text, add_special_tokens=False)['input_ids']], device='cuda'); emb = E(ids); bad = {}
    for m in TOOLS:
        if m == owner: continue
        e, _ = NET.nets[m].emit(emb)
        nz = (e[0, start:].abs().sum(-1) > 0).nonzero().flatten().tolist()
        if nz: bad[m] = nz[:5]
    return bad


for label, call in pool:
    nm = call.split(' ', 1)[0]
    owner = NAME2M.get(nm)
    want = ref(call) if owner else None
    if nm == 'walk' or call.startswith('walk'): want = RB.run_ref(call); owner = 'turtle_gen' if want is not None else None
    for ci, ctx in enumerate(CTX):
        text = ctx + CS + call + CE
        start = len(tok(ctx, add_special_tokens=False)['input_ids']) if ctx else 0
        got = ''
        if owner:                                              # owner driven incrementally on its own writes
            got, early = drive(owner, text)
            good = ((got == OS + want + OE) or (want == '' and got in ('', OS + OE))) and (early == 0 or ci == 2)
        else: good = True
        full = text + (got if owner else '') + ' and the text goes on.'
        bad = silent_others(owner, full, max(start - 1, 0))
        ok = good and not bad
        s = res['check1'].setdefault(label, [0, 0]); s[0] += ok; s[1] += 1
        if not ok and len(res['fail1']) < 60:
            res['fail1'].append({'label': label, 'call': call, 'ctx': ci, 'want': want, 'owner': owner, 'got': got, 'others_wrote_at': bad})
print(f'check 1 done ({time.time() - t0:.0f}s)', flush=True)
for k, (g, n) in res['check1'].items(): print(f'  {k:22s} {g}/{n}')
for f in res['fail1'][:15]: print('  FAIL', f)

# ---------------- (2) end-to-end greedy generation with the host ----------------
eos = tok.convert_tokens_to_ids('<|endoftext|>')


@torch.no_grad()
def gen(prompts, max_new=110):
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device='cuda'); att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device='cuda')
    NET.prime(x)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values; lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    g = []
    for _ in range(max_new):
        nxt = lg.argmax(-1); g.append(nxt); att = torch.cat([att, att.new_ones(B_, 1)], 1)
        NET.step(nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    g = torch.stack(g, 1).tolist()
    return [tok.decode(s) for s in g]


by = {}
for label, call in pool:
    nm = call.split(' ', 1)[0]
    w = ref(call) if nm in NAME2M else None
    if w: by.setdefault(NAME2M[nm], []).append((call, w))
for m, lst in by.items():
    lst = lst[:a.n_gen]; okc = 0
    for b in range(0, len(lst), 20):
        ch = lst[b:b + 20]
        outs = gen(["Q: Let's use the tool.\nA: Let's think step by step.\n" + CS + c + CE for c, _ in ch])
        for (c, w), o in zip(ch, outs):
            good = o.startswith(OS + w + OE); okc += good
            if not good and len(res['fail2']) < 30: res['fail2'].append({'tool': m, 'call': c, 'want': w, 'got': o[:120]})
    res['check2'][m] = [okc, len(lst)]
    print(f'  e2e {m:14s} {okc}/{len(lst)}', flush=True)

# ---------------- (3) bit-identity with no call ----------------
texts = []
bd = COT
if os.path.isdir(bd):
    for f in sorted(os.listdir(bd)): texts.append(open(f'{bd}/{f}').read().split('-----', 1)[1].strip() + "\n\nQ: x\nA: Let's think step by step.")
for l in list(open(f'{G}/teamA/data/test.jsonl'))[:10]: texts.append('Q: ' + json.loads(l)['input'] + "\nA: Let's think step by step." + json.loads(l)['cot'])
for f in ('temporal_test', 'robot_oos', 'knights_oos'):
    for l in list(open(f'{G}/teamB/data/{f}.jsonl'))[:5]: texts.append('Q: ' + json.loads(l)['question'] + "\nA: Let's think step by step. 3 + 4 = 7 and True and not False = True.")
for t in ('multistep_arithmetic_two', 'boolean_expressions', 'object_counting'):
    for e in json.load(open(f'{G}/teamC/data/test/{t}.json'))['examples'][:5]: texts.append('Q: ' + e['input'] + "\nA: Let's think step by step. (3 + 4) * 2 = 14.")
texts = texts[:a.bit_texts]
n_eq = n_zero = 0
with torch.no_grad():
    for t in texts:
        ids = torch.tensor([tok(t, add_special_tokens=False)['input_ids']], device='cuda')
        NET.prime(ids, full=True)
        zero = bool((NET.pending == 0).all()); n_zero += zero
        l1 = model(input_ids=ids).logits
        hook = NET.hook; hook.remove(); pend = NET.pending; NET.pending = None
        l0 = model(input_ids=ids).logits
        NET.hook = model.model.norm.register_forward_pre_hook(NET._add); NET.pending = pend
        n_eq += bool(torch.equal(l0, l1))
res['check3'] = {'texts': len(texts), 'all_writes_zero': n_zero, 'logits_bit_identical': n_eq, 'max_len': None}
print(f'check 3: {n_zero}/{len(texts)} all-zero writes, {n_eq}/{len(texts)} bit-identical logits', flush=True)
json.dump(res, open(a.out, 'w'), indent=1)
