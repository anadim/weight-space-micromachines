"""Exactness + silence test of the COMPILED chain2 micromachine through the real decode path (CalcImplant.prime_rows /
step, the same calls the RL engine makes), with calc_s4 attached beside it.
For each form: 10k random expressions, text '... is <expr> =' + the host's lead (' ' or ' $'); then the machine's own
writes are fed back token by token until its end-of-number veto. Pass = written text == exact value (calc2.fmt).
Silence: on non-chain text the machine's output is zero at EVERY position (teacher-forced emit), and wherever chain2
writes, calc_s4's output is zero (no double write)."""
import os, sys, random, json, torch
_R = os.environ.get('WSM_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # repo root
sys.path.insert(0, os.path.join(_R, 'rl', 'wiring')); sys.path.insert(0, os.path.join(_R, 'rl', 'gsm_grpo')); sys.path.insert(0, _R)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from transformers import AutoTokenizer, AutoModelForCausalLM
import calc2, chain2
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
chain2.register(tok)
from implant import CalcImplant
dev = 'cuda'
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to(dev).eval()
E = hf.get_input_embeddings().weight.detach()
BM = 1024
calc = CalcImplant(hf, tok, BM, machine='calc_s4', fused=False)
ch = CalcImplant(hf, tok, BM, machine='chain2', fused=False)
OUTTOK = []
for s in chain2.OUT:
    t = chain2.TEXT.get(s, s); t = t[0] if isinstance(t, tuple) else t
    if s == 'DOT': t = '.'
    if s == 'MN': t = '-'
    OUTTOK.append(tok(t, add_special_tokens=False)['input_ids'][0])
OE = chain2.OUT.index('OE')
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
FILL = T('Question: a shop sells things.\nAnswer: We add the amounts step by step and keep the units.\n' * 10)
R = random.Random(int(os.environ.get('SEED', 1)))


def num(dec=0.0, comma=False, big=0.1, dollar=0.2, maxint=9):
    d = R.randint(1, maxint if R.random() < big else 4)
    v = R.randint(10 ** (d - 1) if d > 1 else 0, 10 ** d - 1)
    s = f'{v:,}' if comma and v >= 1000 else str(v)
    if R.random() < dec: s += '.' + str(R.randint(0, 10 ** R.randint(1, 4) - 1)).rjust(1, '0')
    if R.random() < dollar: s = '$' + s
    return s


def join(parts):
    out = parts[0]
    for p in parts[1:]: out += f" {R.choice(['+', '-'])} {p}"
    return out


def gen(form):
    if form == 'chain': return join([num() for _ in range(R.randint(4, 9))])
    if form == 'comma':
        k = R.randint(2, 6); parts = [num(comma=True, big=0.6) for _ in range(k)]
        if not any(',' in p for p in parts): parts[0] = f'{R.randint(1000, 999999):,}'
        return join(parts)
    if form == 'sign': return '-' + join([num(dollar=0) for _ in range(R.randint(3, 7))])
    if form == 'decimal': return join([num(dec=0.7) for _ in range(R.randint(4, 8))])
    if form == 'negative':
        k = R.randint(4, 8); parts = [num(dollar=0) for _ in range(k)]
        return parts[0] + ''.join(f' - {p}' for p in parts[1:])


def ref(e):
    s = e.replace('$', '').replace(',', '')
    toks = s.split(' ')
    v = Fraction(toks[0]); i = 1
    while i < len(toks):
        x = Fraction(toks[i + 1]); v = v + x if toks[i] == '+' else v - x; i += 2
    return calc2.fmt(v)


def expected_fire(e):
    s = e.replace('$', ''); nops = s.count(' + ') + s.count(' - ')
    return nops >= 3 or ',' in s or (s.startswith('-') and nops >= 2)


@torch.no_grad()
def run_batch(texts, leads):
    seqs = [FILL + T(t) + T(l) for t, l in zip(texts, leads)]
    KW = ch.K + ch.W
    tail = torch.tensor([s[-KW:] for s in seqs], device=dev)
    n = len(seqs); rows = torch.arange(n, device=dev)
    e2 = ch.prime_rows(E[tail], rows); e1 = calc.prime_rows(E[tail], rows)
    started = torch.zeros(n, dtype=torch.bool, device=dev); outs = [[] for _ in range(n)]; done = torch.zeros(n, dtype=torch.bool, device=dev); dbl = 0; silent0 = (e2.abs().sum(-1) == 0)
    for step in range(40):
        nz = e2.abs().sum(-1) > 0
        dd = ((e1.abs().sum(-1) > 0) & nz & ~done)
        CS = list('0123456789') + ['.', '-', 'OS', 'OE']; CH = [{'DOT': '.', 'MN': '-'}.get(x, x) for x in chain2.OUT]
        for j in torch.nonzero(dd)[:, 0].tolist():
            if CS[int(e1[j].argmax())] != CH[int(e2[j].argmax())]: DIS[0] += 1
        dbl += int(dd.sum())
        for j in torch.nonzero(dd)[:, 0].tolist():
            if len(DBL) < 8: DBL.append((texts[j], leads[j], step, tok.decode(outs[j]), chain2.OUT[int(e2[j].argmax())], int(e1[j].argmax())))
        sym = e2.argmax(-1)
        fin = (~nz) | (sym == OE)
        after = (e1.abs().sum(-1) > 0) & ~nz & started & ~done   # calc_s4 writes once chain2 has gone quiet
        CAFTER[0] += int(after.sum())
        started |= nz
        nxt = torch.tensor(OUTTOK, device=dev)[sym]
        for j in torch.nonzero(~done & ~fin)[:, 0].tolist(): outs[j].append(int(nxt[j]))
        done |= fin
        if bool(done.all()): break
        nxt = torch.where(done, torch.full_like(nxt, T(' the')[0]), nxt)
        e2 = ch.step(E[nxt], n, nxt); e1 = calc.step(E[nxt], n, nxt)
    return [tok.decode(o) for o in outs], silent0.tolist(), dbl


DBL = []; DIS = [0]; CAFTER = [0]
res = []
N = int(os.environ.get('N', 10000))
for form in os.environ.get('FORMS', 'chain,comma,sign,decimal,negative').split(','):
    wrong = nofire = badfire = dbl = 0; ex = []
    for c0 in range(0, N, BM):
        es = [gen(form) for _ in range(min(BM, N - c0))]
        leads = [R.choice([' ', ' $']) for _ in es]
        texts = [f'So the total amount is {e} =' for e in es]
        got, sil, d = run_batch(texts, leads); dbl += d
        for e, g, s0 in zip(es, got, sil):
            want = ref(e); ef = expected_fire(e)
            if s0 and g == '':
                if ef: nofire += 1; ex.append(('nofire', e)) if len(ex) < 6 else None
                continue
            if not ef: badfire += 1
            if g != want:
                wrong += 1
                if len(ex) < 6: ex.append((e, g, want))
    DBL.clear()
    r = dict(form=form, n=N, wrong=wrong, nofire=nofire, fired_unexpected=badfire, calc_s4_double_writes=dbl, double_writes_disagreeing=DIS[0], calc_writes_after_chain=CAFTER[0], ex=ex); DIS[0] = 0; CAFTER[0] = 0
    print(json.dumps(r), flush=True); res.append(r)


# ---- silence: teacher-forced emit over whole sequences; chain2 output must be zero at every position ----
def silent_texts(n):
    out = []
    for _ in range(n):
        k = R.random()
        a, b, c = R.randint(1, 9999), R.randint(1, 9999), R.randint(1, 999)
        if k < 0.2: out.append(f'So he has {a} {R.choice(["+", "-", "x", "/"])} {b} = {a + b} apples.')   # calc_s4 forms
        elif k < 0.35: out.append(f'The total is {a} + {b} + {c} = {a + b + c}.')                       # 3 operands: calc_s4
        elif k < 0.5: out.append(f'The cost is {a} x {b} + {c} + {a} + {b} = 5.')                      # mixed operators
        elif k < 0.6: out.append(f'So it is ({a} + {b}) + {c} + {a} = 7.')                             # brackets
        elif k < 0.7: out.append(f'Let x be the number. Then 3x + {c} + {a} + {b} = {a}.')            # algebra
        elif k < 0.8: out.append(f'He walks {a} miles + {b} miles + {c} miles + 4 miles = 9 miles.')  # words inside
        elif k < 0.9: out.append(f'She bought {a}, {b}, {c} and 4 apples, so 4 + 5 + 6 + 7 is the plan.')   # no '='
        else: out.append(f'The answer is {R.randint(1, 99)}.{R.randint(0, 99)} dollars, i.e. {a}.')
    return out


acted = 0; ex = []
for c0 in range(0, N, 256):
    ts = silent_texts(min(256, N - c0))
    ids = [FILL + T(t) for t in ts]
    L = max(map(len, ids)); pad = T(' the')[0]
    x = torch.tensor([[pad] * (L - len(s)) + s for s in ids], device=dev)
    with torch.no_grad():
        e = ch.M.emit(E[x])[0]
    a = (e.abs().sum(-1) > 0).any(-1)
    acted += int(a.sum())
    for j in torch.nonzero(a)[:, 0].tolist()[:3]:
        if len(ex) < 6: ex.append(ts[j])
r = dict(form='silent', n=N, acted=acted, ex=ex); print(json.dumps(r), flush=True); res.append(r)
r = dict(chain2_units=ch.M.stack1.n_units + ch.M.stack2.n_units, nnz=ch.M.stack1.nnz + ch.M.stack2.nnz, symbols=len(chain2.SYMS))
print(json.dumps(r)); res.append(r)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_chain2_result.json'), 'w'), indent=1)
