"""IFEval-reader silence on GSM prompts (CPU, token-level circuit only; no model forward): step Team D's frozen circuit
over [8-shot prefix + 'Question: ...\\nAnswer:'] for every dev-B (800) and GSM8K-test question, exactly the token stream
the engine feeds it, and report which rule latches fire. The circuit's gates also need the assistant-role register
(chat template), which the base-format prompt never sets -> also report it."""
import os, sys, json, torch
os.environ['CIRC_DEV'] = 'cpu'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wiring'))
import common as C
from transformers import AutoTokenizer, AutoModelForCausalLM
tokb = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
pre, shot_q = C.prefix_ids(tokb)
dev = C.load_train('B', shot_q=shot_q)
import random
rng = random.Random(123); dg = [r for r in dev if r['kind'] == 'gsm']; ds = [r for r in dev if r['kind'] == 'scaled']
rng.shuffle(dg); rng.shuffle(ds); dev = dg[:400] + ds[:400]
test = C.load_test('test', shot_q=shot_q)
sys.modules.pop('common')
sys.path.insert(0, os.path.join(HERE, 'ifeval_frozen'))
import importlib.util
spec = importlib.util.spec_from_file_location('circuit_g', os.path.join(HERE, 'circuit_g.py'))
CG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CG)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16)
circ = CG.Circuit(tok, m, vecs=os.path.join(HERE, 'ifeval_frozen', 'vecs2.pt'))
from torch.utils._pytree import tree_flatten, tree_unflatten
def gather_tree(x, idx, B):
    if torch.is_tensor(x): return x.index_select(0, idx) if (x.dim() > 0 and x.shape[0] == B) else x
    if isinstance(x, tuple): return tuple(gather_tree(t, idx, B) for t in x)
    if isinstance(x, list): return [gather_tree(t, idx, B) for t in x]
    if isinstance(x, dict): return {k: gather_tree(v, idx, B) for k, v in x.items()}
    return x
before = set(vars(circ)); circ.reset(1); SN = [k for k in vars(circ) if k not in before and k != 'B']
pt = torch.tensor(pre)
for t in range(len(pre)): circ.step(pt[t:t + 1])          # shared 8-shot prefix once
PST = {a_: tree_flatten(getattr(circ, a_)) for a_ in SN}
print('prefix latched:', [k for k in circ.report()[0] if circ.report()[0][k] == 1], flush=True)
RULES = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul']
out = {}
for name, items in (('dev', dev), ('test', test)):
    per = {r: 0 for r in RULES}; anyl = 0; asst = 0; ex = []
    BS = 32
    for b0 in range(0, len(items), BS):
        bt = items[b0:b0 + BS]
        qs = [C.q_ids(tokb, r['q']) for r in bt]
        L = max(map(len, qs))
        x = torch.tensor([[151643] * (L - len(q)) + q for q in qs])
        for a_, (leaves, spec_) in PST.items():
            setattr(circ, a_, tree_unflatten([l.clone() if torch.is_tensor(l) else l for l in leaves], spec_))
        circ.B = 1
        idx = torch.zeros(len(qs), dtype=torch.long)
        for a_ in SN: setattr(circ, a_, gather_tree(getattr(circ, a_), idx, 1))
        circ.B = len(qs)
        for t in range(L): circ.step(x[:, t])
        rep = circ.report()
        for r_, it in zip(rep, bt):
            lat = [k for k in RULES if k in r_]
            for k in lat: per[k] += 1
            if lat: anyl += 1; ex.append((it['q'][:120], lat))
        asst += int((circ.asst > 0.5).sum())
    out[name] = dict(n=len(items), latched_any=anyl, rate=anyl / len(items), per_rule={k: v for k, v in per.items() if v},
                     assistant_register_on=asst, examples=ex[:8])
    print(name, json.dumps(out[name], indent=1), flush=True)
json.dump(out, open(os.path.join(HERE, 'if_silence.json'), 'w'), indent=1)
