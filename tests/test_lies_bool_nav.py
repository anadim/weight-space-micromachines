"""Teacher-forced exactness of fork A machines on generated official-format text, every position checked against Python:
the machine must write exactly the reference token at each decision point and nothing anywhere else.
python test_natA.py lies|bool N SEED"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import random, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import SPECS, first, Attached
which, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
import os
MP = os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base')
tok = AutoTokenizer.from_pretrained(MP)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.float32).to(DEV).eval()
name = {'lies': 'liesnat', 'bool': 'boolnat', 'nav': 'navnat'}[which]
A = Attached(model, tok, machines=(name,)); M = A.nets[name]; I = A.all_ids[name]
outs = SPECS[name]['out']; out_tok = [first(I[s]) for s in outs]
E = model.get_input_embeddings(); r = random.Random(seed)
NAMES = ['Fidel', 'Jerry', 'Vina', 'Millicent', 'Raymond', 'Sherrie', 'Vernell', 'Alexis', 'Michaela', 'Elanor', 'Tamika', 'Willian',
         'Shaunda', 'Fletcher', 'Conception', 'Bernita', 'Jamey', 'Teressa', 'Kristian', 'Delbert', 'Helene', 'Gwenn', 'Ka', 'Leda']
V = lambda b: 'tells the truth' if b else 'lies'


def machine_writes(text):
    ids = tok(text, add_special_tokens=False)['input_ids']
    with torch.no_grad():
        e, _ = M.emit(E(torch.tensor([ids], device=DEV)))
    e = e[0]; w = {}
    for p in range(len(ids)):
        if e[p].max() > 0.5: w[p] = out_tok[int(e[p].argmax())]
    return ids, w


def check(text, decisions):
    """decisions: list of (prefix_text, ref_token_text): the machine must write ref right after prefix, nowhere else."""
    ids, w = machine_writes(text)
    want = {}
    for pre, ref in decisions:
        if isinstance(pre, list): want[len(pre) - 1] = ref; continue           # token-level decision (prefix ids, next id)
        p = len(tok(pre, add_special_tokens=False)['input_ids']) - 1
        want[p] = tok(ref, add_special_tokens=False)['input_ids'][0]
    bad = [(p, tok.decode(ids[max(0, p - 8):p + 1]), tok.decode([w.get(p, -1)]) if p in w else None, tok.decode([want[p]]) if p in want else None)
           for p in sorted(set(w) | set(want)) if w.get(p) != want.get(p)]
    return bad


C = {'items': 0, 'decisions': 0, 'bad_items': 0}
for it in range(N):
    if which == 'lies':
        k = r.randint(3, 6); ppl = r.sample(NAMES, k); vals = [r.random() < 0.5]
        p1 = ' (1) %s %s. So, we know that %s' % (ppl[0], V(vals[0]), ppl[0]); dec = [(p1, ' tells' if vals[0] else ' lies')]
        text = p1 + ' %s.\n' % V(vals[0])
        for j in range(1, k):
            said = r.random() < 0.5; v = vals[-1] == said; vals.append(v)
            kp = text + '(%d) %s says %s %s. Since we know from (%d) that %s' % (j + 1, ppl[j], ppl[j - 1], V(said), j, ppl[j - 1])
            dec.append((kp, ' tells' if vals[-2] else ' lies'))
            pre = kp + ' %s, if %s says %s %s, then %s' % (V(vals[-2]), ppl[j], ppl[j - 1], V(said), ppl[j])
            dec.append((pre, ' tells' if v else ' lies')); text = pre + ' %s.\n' % V(v)
        pre = text + 'Now, the question asks: Does %s tell the truth? We know from (%d) that %s %s. So the answer is' % (ppl[-1], k, ppl[-1], V(vals[-1]))
        dec.append((pre, ' Yes' if vals[-1] else ' No')); text = pre + (' Yes.' if vals[-1] else ' No.')
    elif which == 'nav':
        aff = r.random() < 0.3; x = y = 0; f = 1                       # facing: 0 +x, 1 +y, 2 -x, 3 -y
        VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]; FN = ['positive x', 'positive y', 'negative x', 'negative y']
        steps = (['Always face forward'] if aff else []) + [None] * r.randint(2, 8)
        text = ' We start at the origin (0, 0), facing the positive y-axis.\n'; dec = []
        for k, st in enumerate(steps):
            if st is None:
                if aff or r.random() < 0.6:
                    n = r.choice([r.randint(1, 10), r.randint(1, 99)]) if r.random() < 0.2 else r.randint(1, 10)
                    d = r.choice(['', ' forward', ' backward', ' left', ' right']) if aff else r.choice(['', '', ' forward', ' backward'])
                    st = 'Take %d step%s%s' % (n, '' if n == 1 else 's', d)
                    rr = {'': 0, ' forward': 0, ' left': 1, ' backward': 2, ' right': 3}[d]; vx, vy = VEC[(f + rr) % 4]; x += n * vx; y += n * vy
                else:
                    t = r.choice(['left', 'right', 'around']); st = 'Turn ' + t; f = (f + {'left': 1, 'around': 2, 'right': 3}[t]) % 4
            pre = text + '(%d) %s:' % (k + 1, st)
            coord = ' (%d, %d), facing the %s' % (x, y, FN[f])
            ids_pre = tok(pre, add_special_tokens=False)['input_ids']; ids_full = tok(pre + coord + '-axis.\n', add_special_tokens=False)['input_ids']
            assert ids_full[:len(ids_pre)] == ids_pre, (pre, coord)
            ncoord = len(tok(pre + coord, add_special_tokens=False)['input_ids']) - len(ids_pre)
            for j in range(ncoord): dec.append((ids_full[:len(ids_pre) + j], ids_full[len(ids_pre) + j]))
            text = pre + coord + '-axis.\n'
        home = (x, y) == (0, 0)
        ids_pre = tok(text + 'Since', add_special_tokens=False)['input_ids']; sc = ' (%d, %d)' % (x, y)       # v2: machine writes ' (X, Y)'
        ids_full = tok(text + 'Since' + sc + ' is', add_special_tokens=False)['input_ids']; nsc = len(tok(text + 'Since' + sc, add_special_tokens=False)['input_ids']) - len(ids_pre)
        assert ids_full[:len(ids_pre)] == ids_pre
        for j in range(nsc): dec.append((ids_full[:len(ids_pre) + j], ids_full[len(ids_pre) + j]))
        pre = text + 'Since (%d, %d) is' % (x, y); dec.append((pre, ' (' if home else ' not'))
        mid = pre + (' (0, 0), we are indeed where we started.' if home else ' not (0, 0), we are not where we started.') + ' So the answer is'
        dec.append((mid, ' Yes' if home else ' No')); text = mid + (' Yes.' if home else ' No.')
    else:
        def gen(d=0):
            k = r.random()
            if k < 0.4 or d >= 2: return r.choice(['True', 'False']) if r.random() < 0.7 else 'not ' + r.choice(['True', 'False'])
            if k < 0.6: return 'not ' + gen(d + 1)
            if k < 0.8: return '( ' + gen(d + 1) + ' )'
            return gen(d + 1) + r.choice([' and ', ' or ']) + gen(d + 1)
        ex = gen()
        while len(ex.split()) > 16: ex = gen()
        val = eval(ex.replace('( ', '(').replace(' )', ')'))
        pre = 'Let\'s evaluate A: A = %s =' % ex
        dec = [(pre, ' True' if val else ' False'), (pre + (' True' if val else ' False'), '.')]
        text = pre + (' True' if val else ' False') + '.'
        # a sub-expression with a variable must stay silent
        text += ' So Z = A and ( %s ) =' % r.choice(['True', 'False']) + ' something.'
    bad = check(text, dec)
    C['items'] += 1; C['decisions'] += len(dec); C['bad_items'] += bool(bad)
    if bad and C['bad_items'] <= 5: print('BAD', bad[:3])
print(which, C)
assert C['bad_items'] == 0, C
print('PASS')
