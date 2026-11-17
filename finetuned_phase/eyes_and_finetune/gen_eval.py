"""Batched greedy generation + GSM8K scoring.
--calc self    : the model writes everything (no calculator)
--calc oracle  : Python calculator fills <|box_start|>...<|box_end|> after every </tool_call>
--calc net     : CalcNet (in-weights calculator) is attached; plain greedy decoding, no logic in the loop
"""
import json, re, argparse, random, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from calcfmt import calc_ref, CS, CE, OS, OE
from regfmt import reg_ref
from swapfmt import swap_ref, last_state, CS2, CE2, OS2, OE2
from liefmt import truth_ref, last_state3, CS3, CE3, OS3, OE3
from boolfmt import bool_ref, bool_ref_toks, CS4, CE4, OS4, OE4

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--calc', default='self', choices=['self', 'oracle', 'net', 'random'])   # random: boxes get random content of the right shape
ap.add_argument('--split', default='test')      # test | dev | path to jsonl
ap.add_argument('--out', required=True)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--max_new', type=int, default=512)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--dtype', default='bf16')
ap.add_argument('--machines', default='calc')      # micromachines to attach in --calc net: calc, swap, or calc,swap
ap.add_argument('--answer', default='number', choices=['number', 'letter', 'yesno', 'bool'])
ap.add_argument('--link_from', default='')         # compute readers / write directions from this older checkpoint (stale link)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = 'left'
dt = torch.bfloat16 if a.dtype == 'bf16' else torch.float32
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt).cuda().eval()
eos = tok.convert_tokens_to_ids('<|endoftext|>')
cs_id, ce_id, os_id, oe_id = (tok.convert_tokens_to_ids(t) for t in (CS, CE, OS, OE))
char_id = {c: tok(c)['input_ids'][0] for c in '0123456789.-'}
rng = random.Random(0)
FILL = a.calc in ('oracle', 'random')


def scramble(kind, ref, call=''):
    """Random box content with the shape of the true one (used by --calc random)."""
    if a.calc != 'random' or ref == '': return ref
    if kind == 'calc':
        return ''.join(c if c in '.-' else rng.choice('0123456789' if i else '123456789') for i, c in enumerate(ref))
    if kind == 'swap':
        if len(call) == 2 and call[0] == '0': return str(rng.randint(1, 7))
        s = list(ref); rng.shuffle(s); return ''.join(s)
    if kind == 'bool': return rng.choice(['True', 'False'])
    return rng.choice('01')
cs2_id, ce2_id, os2_id, oe2_id = (tok.convert_tokens_to_ids(t) for t in (CS2, CE2, OS2, OE2))
cs3_id, ce3_id, os3_id, oe3_id = (tok.convert_tokens_to_ids(t) for t in (CS3, CE3, OS3, OE3))
cs4_id, ce4_id, os4_id, oe4_id = (tok.convert_tokens_to_ids(t) for t in (CS4, CE4, OS4, OE4))
word_id = {w: tok(w)['input_ids'][0] for w in ('True', 'False')}

if a.split == 'test': rows = [json.loads(l) for l in open('data/test.jsonl')]
elif a.split == 'dev':
    rows = [json.loads(l) for l in open('data/train.jsonl')]; random.Random(1234).shuffle(rows); rows = rows[:500]
else: rows = [json.loads(l) for l in open(a.split)]
if a.limit: rows = rows[:a.limit]

net = None
if a.calc == 'net':
    from calcnet import attach_calcnet
    link = AutoModelForCausalLM.from_pretrained(a.link_from, torch_dtype=dt).cuda().eval() if a.link_from else None
    net = attach_calcnet(model, tok, machines=tuple(a.machines.split(',')), link=link)
    if link is not None: print('STALE LINK from', a.link_from, '| old reader still exact on the new embeddings:', net.stale_reader_exact, flush=True); del link   # the machines now write into the residual stream before the final norm
    print('reader relative gaps (min):', min(net.gaps.values()), '| head margins (min logit gap, max leaked prob):', min(v[0] for v in net.margins.values()), max(v[1] for v in net.margins.values()), flush=True)


@torch.no_grad()
def generate(prompts):
    enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')
    B = len(prompts)
    ids, att = enc.input_ids, enc.attention_mask
    if net is not None: net.prime(ids)
    def fwd(x, att, past=None):                  # LM head on the last position only (the full-prompt logits are huge)
        o = model.model(input_ids=x, attention_mask=att, past_key_values=past, use_cache=True)
        return model.lm_head(o.last_hidden_state[:, -1]).float(), o.past_key_values
    logits, past = fwd(ids, att)
    gen = [[] for _ in range(B)]; queue = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device='cuda'); dl = [False] * B
    for _ in range(a.max_new):
        nxt = logits.argmax(-1)
        if FILL:
            for i in range(B):
                if queue[i]: nxt[i] = queue[i].pop(0)
        nxt[done] = eos
        nl = nxt.tolist()
        for i in range(B):
            if dl[i]: continue
            t = nl[i]; gen[i].append(t)
            if t == eos: done[i] = True; dl[i] = True
            elif FILL and t == ce_id and not queue[i]:
                g = gen[i]; j = len(g) - 1
                while j >= 0 and g[j] != cs_id: j -= 1
                expr = tok.decode(g[j + 1:-1]) if j >= 0 else ''
                ref = reg_ref(expr, tok.decode(g[:max(j, 0)], skip_special_tokens=False))      # vK = K-th box (register format)
                queue[i] = [os_id] + [char_id[c] for c in scramble('calc', ref)] + [oe_id]
            elif FILL and t == ce2_id and not queue[i]:
                g = gen[i]; j = len(g) - 1
                while j >= 0 and g[j] != cs2_id: j -= 1
                call = tok.decode(g[j + 1:-1]) if j >= 0 else ''
                queue[i] = [os2_id] + [char_id[c] for c in scramble('swap', swap_ref(call, last_state(tok.decode(g[:-1]))), call)] + [oe2_id]
            elif FILL and t == ce3_id and not queue[i]:
                g = gen[i]; j = len(g) - 1
                while j >= 0 and g[j] != cs3_id: j -= 1
                call = tok.decode(g[j + 1:-1]) if j >= 0 else ''
                queue[i] = [os3_id] + [char_id[c] for c in scramble('truth', truth_ref(call, last_state3(tok.decode(g[:-1]))))] + [oe3_id]
            elif FILL and t == ce4_id and not queue[i]:
                g = gen[i]; j = len(g) - 1
                while j >= 0 and g[j] != cs4_id: j -= 1
                v = scramble('bool', bool_ref_toks(tok.convert_ids_to_tokens(g[j + 1:-1])) if j >= 0 else '')
                queue[i] = [os4_id] + ([word_id[v]] if v else []) + [oe4_id]
        if done.all(): break
        att = torch.cat([att, torch.ones(B, 1, dtype=att.dtype, device='cuda')], 1)
        if net is not None: net.step(nxt)
        logits, past = fwd(nxt[:, None], att, past)
    return gen


def extract(txt):
    if a.answer == 'letter':
        m = re.search(r'####\s*\(?([A-G])\)?', txt)
        return m.group(1) if m else None
    if a.answer == 'yesno':
        m = re.search(r'####\s*(Yes|No)', txt)
        return m.group(1) if m else None
    if a.answer == 'bool':
        m = re.search(r'####\s*(True|False)', txt)
        return m.group(1) if m else None
    m = re.search(r'####\s*(-?[\d,]*\.?\d+)', txt)
    return m.group(1).replace(',', '') if m else None


t0 = time.time(); res = []
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))
for b in range(0, len(order), a.bs):
    idx = order[b:b + a.bs]
    gens = generate([rows[i]['question'] if rows[i].get('raw_prompt') else f"Question: {rows[i]['question']}\nAnswer:" for i in idx])
    for i, g in zip(idx, gens):
        res.append((i, g))
res.sort()
correct = ncall = nerr = ntok = 0
with open(a.out, 'w') as f:
    for i, g in res:
        txt = tok.decode(g, skip_special_tokens=False).replace('<|endoftext|>', '')
        gold = rows[i]['answer'].split('####')[1].strip().replace(',', '')
        pred = extract(txt)
        if a.answer in ('letter', 'yesno', 'bool'): ok = pred is not None and pred == gold.strip('()')
        else:
            try: ok = pred is not None and abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
            except ValueError: ok = False
        calls = [(m.group(1), m.group(2), txt[:m.start()]) for m in re.finditer(re.escape(CS) + '(.*?)' + re.escape(CE) + re.escape(OS) + '(.*?)' + re.escape(OE), txt)]
        wrong = sum(1 for e, o, before in calls if reg_ref(e, before) != o); calls = [c[:2] for c in calls]
        st = None                                         # swap-tracker calls: replay the transcript with the reference
        for m in re.finditer(re.escape(CS2) + '(.*?)' + re.escape(CE2) + '(?:' + re.escape(OS2) + '(.*?)' + re.escape(OE2) + ')?', txt, flags=re.S):
            calls.append((m.group(1), m.group(2))); wrong += m.group(2) is not None and m.group(2) != swap_ref(m.group(1), st)
            if m.group(2) is not None: st = m.group(2)
        for m in re.finditer(re.escape(CS4) + '(.*?)' + re.escape(CE4) + '(?:' + re.escape(OS4) + '(.*?)' + re.escape(OE4) + ')?', txt, flags=re.S):
            calls.append((m.group(1), m.group(2))); wrong += m.group(2) is not None and m.group(2) != bool_ref(m.group(1))
        st = None
        for m in re.finditer(re.escape(CS3) + '(.*?)' + re.escape(CE3) + '(?:' + re.escape(OS3) + '(.*?)' + re.escape(OE3) + ')?', txt, flags=re.S):
            calls.append((m.group(1), m.group(2))); wrong += m.group(2) is not None and m.group(2) != truth_ref(m.group(1), st)
            if m.group(2) is not None: st = m.group(2)
        correct += ok; ncall += len(calls); nerr += wrong; ntok += len(g)
        f.write(json.dumps({'i': i, 'gold': gold, 'pred': pred, 'ok': ok, 'ncalls': len(calls), 'wrong_outputs': wrong, 'ids': g, 'out': txt}) + '\n')
n = len(rows)
print(f'RESULT model={a.model} calc={a.calc} split={a.split}: acc {correct}/{n} = {100*correct/n:.2f}% | calls {ncall} ({ncall/n:.2f}/problem), outputs differing from exact calculator: {nerr} | {ntok/n:.0f} tok/problem | {time.time()-t0:.0f}s')
