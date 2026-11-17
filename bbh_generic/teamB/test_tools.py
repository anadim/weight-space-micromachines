"""Exactness test of the compiled tools (circuit + Qwen tokenizer; symbol flags from token ids, the reader itself is
checked exact over the whole vocabulary at attach time).

For every text: run the machine over the token sequence (teacher forcing). Pass iff the machine writes exactly the
reference tokens '<tool_response>...</tool_response>' right after each '</tool_call>' and is silent at every other position.
python test_tools.py interval_gen data/temporal_dev.jsonl,data/lab_train.jsonl:200 [--calls 'free 9-17 busy 10-12' ...]
"""
import sys, json, time, argparse, torch
sys.path.insert(0, '.')
import circuits_B  # registers the specs
from micromachines.specs import SPECS
from micromachines.machine import Micromachine, symbol_token_ids
from transformers import AutoTokenizer
from tools_ref import interval_ref, turtle_ref, truth_ref, run_ref, CS, CE, OS, OE

REF = {'interval_gen': interval_ref, 'turtle_gen': turtle_ref, 'truth_gen': truth_ref}

ap = argparse.ArgumentParser()
ap.add_argument('tool'); ap.add_argument('files', nargs='?', default='')
ap.add_argument('--calls', nargs='*', default=[])
ap.add_argument('--show', type=int, default=3)
ap.add_argument('--fuzz', type=int, default=0)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
t0 = time.time()
M = Micromachine(a.tool); print(f'built {a.tool}: stack1 {M.stack1.n_units} units / {M.stack1.nnz} nnz, stack2 {M.stack2.n_units} units ({time.time()-t0:.0f}s)', flush=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; M = M.to(dev)
sp = SPECS[a.tool]; ids = symbol_token_ids(tok, a.tool)
sym_of = {}
for i, s in enumerate(sp['syms']):
    for t in (ids[s] if isinstance(ids[s], list) else [ids[s]]): sym_of.setdefault(t, []).append(i)
out_tok = [(ids[o] if not isinstance(ids[o], list) else ids[o][0]) for o in sp['out']]

texts = []
for spec in [f for f in a.files.split(',') if f]:
    path, _, lim = spec.partition(':')
    for k, l in enumerate(open(path)):
        if lim and k >= int(lim): break
        texts.append(json.loads(l)['completion'])
import random
from tools_ref import fmt_time


def fuzz_call(tool, r):
    if tool == 'interval_gen':
        ap_ = r.random() < 0.6; hi = 24 if ap_ else 99
        a_ = r.randint(0, hi - 1); b_ = r.randint(a_ + 1, hi)
        f = (lambda h, e=False: fmt_time(h, True)) if ap_ else (lambda h, e=False: str(h))
        busy = []
        for _ in range(r.randint(1, 9)):
            x = r.randint(max(0, a_ - 3), hi - 1); y = r.randint(x + 1, min(hi, x + 8)); busy.append(f'{f(x)}-{f(y)}')
        return f'interval {f(a_)}-{f(b_)} busy ' + ', '.join(busy)
    if tool == 'turtle_gen':
        c = []
        for _ in range(r.randint(1, 20)):
            c.append(r.choice(['turn left', 'turn right', 'turn around']) if r.random() < 0.4 else f"{r.choice(['forward', 'backward', 'left', 'right'])} {r.randint(1, 99)}")
        return 'walk ' + ', '.join(c)
    n = r.randint(1, 12); ks = list(range(1, n + 1)); r.shuffle(ks); cl = []
    for k in ks:
        if r.random() < 0.3: cl.append(f"{k} {r.choice(['true', 'false'])}")
        else: cl.append(f"{k} says {r.randint(1, 12)} {r.choice(['true', 'false'])}")
    return 'truth ' + '; '.join(cl)


def corrupt(call, r):
    """malformed variants: the tool must stay silent"""
    ops = [lambda c: c.replace(',', ' and', 1), lambda c: c.replace(';', ' and', 1), lambda c: c + ' please',
           lambda c: c.replace('-', ' to ', 1), lambda c: c.replace(' 1', ' 123', 1), lambda c: c.replace('turn', 'spin', 1),
           lambda c: c.replace(', ', ' ', 1), lambda c: c.replace('; ', ' ', 1), lambda c: c.replace(' forward', ' forward forward', 1),
           lambda c: c.replace(' says', ' says says', 1), lambda c: c + ' 3pm', lambda c: c + ', 7', lambda c: c.replace(' busy', '', 1),
           lambda c: c.replace(' busy', ' busy busy', 1), lambda c: c + '; 2', lambda c: c.replace(' true', '', 1)]
    for _ in range(20):
        c2 = r.choice(ops)(call)
        if c2 != call: return c2
    return None


r = random.Random(7); silent = []; errs = []
OTHER = ['track 123:12,23?1', 'order 4:1+2,3=1,4>2', 'calc 12 * 7', '3 + 4', 'interval of 3 numbers', 'walking 3', 'truthful']
for _ in range(a.fuzz):
    c = fuzz_call(a.tool, r)
    if a.tool == 'interval_gen' and len(REF[a.tool](c).split(', ')) > 4: errs.append(c); continue   # > 4 gaps: error by design
    if r.random() < 0.1: silent.append(fuzz_call(r.choice([x for x in REF if x != a.tool]), r)); continue  # another tool's call
    a.calls.append(c)
    if r.random() < 0.3:
        c2 = corrupt(c, r)
        if c2: errs.append(c2)
silent += OTHER if a.fuzz else []
for c in a.calls:
    texts.append(f'some text before.\n{CS}{c}{CE}{OS}{REF[a.tool](c)}{OE}\nmore text after.')
ERR = []
for c in errs:                       # own NAME, malformed ARGS: the envelope's empty response (or silence, see below)
    if run_ref(c) == '': ERR.append(c)

ok = 0; bad = []
for txt in texts:
    seq = tok("A: Let's think step by step." + txt, add_special_tokens=False)['input_ids']
    sym = torch.zeros(1, len(seq), len(sp['syms']), device=dev)
    for p, t in enumerate(seq):
        for i in sym_of.get(t, []): sym[0, p, i] = 1
    with torch.no_grad():
        e, _ = M.emit_sym(sym)
    e = e[0].round()
    # expected writes: inside response spans (position p predicts token p+1)
    exp = [None] * len(seq)
    inresp = False
    for p in range(len(seq) - 1):
        if seq[p] == ids['CE']: inresp = True
        if inresp: exp[p] = seq[p + 1]
        if seq[p + 1] == ids['OE']: inresp = False
    good = True; got = []
    for p in range(len(seq)):
        row = e[p]; nz = (row != 0).nonzero().flatten().tolist()
        if exp[p] is None:
            if nz: good = False; got.append(('extra', p, nz))
        else:
            if len(nz) != 1 or row[nz[0]] != 1 or out_tok[nz[0]] != exp[p]: good = False; got.append(('miss', p, tok.decode([exp[p]]), [tok.decode([out_tok[i]]) for i in nz]))
    ok += good
    if not good: bad.append((txt, got))
print(f'{a.tool}: {ok}/{len(texts)} exact ({time.time()-t0:.0f}s)')
ne = ns_ = 0
for c in ERR:                        # error: '<tool_response></tool_response>' exactly, or silent (NAME not followed by its grammar)
    seq = tok(f'x.\n{CS}{c}{CE}{OS}{OE}', add_special_tokens=False)['input_ids'] + tok(' and on.', add_special_tokens=False)['input_ids']
    sym = torch.zeros(1, len(seq), len(sp['syms']), device=dev)
    for p, t in enumerate(seq):
        for i in sym_of.get(t, []): sym[0, p, i] = 1
    with torch.no_grad(): e, _ = M.emit_sym(sym)
    e = e[0].round(); nzp = [p for p in range(len(seq)) if (e[p] != 0).any()]
    pce = seq.index(ids['CE'])
    if not nzp: ns_ += 1
    elif nzp == [pce, pce + 1] and out_tok[int(e[pce].argmax())] == ids['OS'] and out_tok[int(e[pce + 1].argmax())] == ids['OE']: ne += 1
    else: print('WRONG on malformed call:', c, [(p, tok.decode([out_tok[int(e[p].argmax())]])) for p in nzp][:6])
print(f'{a.tool}: malformed own calls: {ne} empty response, {ns_} silent, {len(ERR) - ne - ns_} WRONG (of {len(ERR)})')
nsil = 0
for c in silent:                     # malformed / out-of-range calls: no write anywhere
    seq = tok(f'x.\n{CS}{c}{CE}', add_special_tokens=False)['input_ids'] + tok(' the model goes on.', add_special_tokens=False)['input_ids']
    sym = torch.zeros(1, len(seq), len(sp['syms']), device=dev)
    for p, t in enumerate(seq):
        for i in sym_of.get(t, []): sym[0, p, i] = 1
    with torch.no_grad(): e, _ = M.emit_sym(sym)
    if (e.round() != 0).any(): print('NOT SILENT on malformed:', c)
    else: nsil += 1
print(f'{a.tool}: silent on {nsil}/{len(silent)} calls to other tools')
for txt, got in bad[:a.show]:
    print('---'); print(txt[:600]); print(got[:12])
