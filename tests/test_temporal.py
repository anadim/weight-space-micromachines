"""Exactness of the natural schedule machine (fork C) on every BBH temporal_sequences question (machine alone, fp32)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import SPECS, Attached
from micromachines.circuits import temporal as T
tok = AutoTokenizer.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base'))
model = AutoModelForCausalLM.from_pretrained(os.environ.get('MODEL', 'Qwen/Qwen3-0.6B-Base'), torch_dtype=torch.float32).eval().to('cuda' if torch.cuda.is_available() else 'cpu'); import time; _t0 = time.time()
A = Attached(model, tok, machines=('time_n',)); M = A.nets['time_n']; E = model.get_input_embeddings(); print('attached', f'{time.time() - _t0:.0f}s', flush=True); outs = SPECS['time_n']['out']
C = dict(exact=0, silent=0, WRONG=0, gold=0)
for e in json.load(open(os.environ.get('BBH', 'data/bbh') + '/data/temporal_sequences.json'))['examples']:
    ctx = f"Q: {e['input']}\nA: Let's think step by step.\nSo the answer is ("
    ids = tok(ctx, add_special_tokens=False)['input_ids']
    with torch.no_grad(): em = M.prime_sym(M.symbols(E(torch.tensor([ids], device=model.device))))[0, -1]
    got = outs[int(em.argmax())] if em.max() > 0.5 else None; want = T.time_ref(e['input'])
    if got is None: C['silent'] += 1
    elif got == want: C['exact'] += 1
    else: C['WRONG'] += 1; print('WRONG', got, want)
    C['gold'] += f'({got})' == e['target']
print('time_n on 250 BBH questions:', C)
assert C['WRONG'] == 0, C
print('PASS')
