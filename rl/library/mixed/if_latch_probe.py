"""CPU probe: which IFEval rules latch on our chat prompts, per task type, for several prompt variants (token-level
circuit only, no model forward). Batched: prompts left-padded with <|endoftext|>."""
import os, sys, json, torch
os.environ['CIRC_DEV'] = 'cpu'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mixed_task as MT
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
sys.path.insert(0, os.path.join(HERE, '..', 'ifeval_frozen'))
import importlib.util
spec = importlib.util.spec_from_file_location('circuit_g', os.path.join(HERE, '..', 'circuit_g.py')); CG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CG)
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16)
circ = CG.Circuit(tok, m, vecs=os.path.join(HERE, '..', 'ifeval_frozen', 'vecs2.pt'))
RULES = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul', 'pps', 'star3']
FIN = "Think step by step, then finish with a line of the form 'The answer is X.'"
VARIANTS = {
    'A_current': (MT.SYS, FIN),
    'B_noquote_fin': (MT.SYS, 'Think step by step. State the final result as: The answer is X.'),
    'C_fin_in_system': ("You are a careful assistant. For math and reasoning questions, think step by step; your last sentence states the result as: The answer is X.", ''),
    'D_C_plain_system': ("You are a helpful assistant who reasons step by step and states the final result as: The answer is X.", ''),
}
rows = MT.load('test'); by = {}
for r in rows:
    if len(by.setdefault(r['type'], [])) < 40: by[r['type']].append(r)
for vn, (SYSV, FINV) in VARIANTS.items():
    out = {}
    for t, rr in by.items():
        ids = []
        for r in rr:
            u = r['user'].replace('\n' + FIN, ('\n' + FINV) if FINV else '')
            s = tok.apply_chat_template([{'role': 'system', 'content': SYSV}, {'role': 'user', 'content': u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids.append(tok(s, add_special_tokens=False)['input_ids'])
        L = max(map(len, ids)); x = torch.tensor([[151643] * (L - len(q)) + q for q in ids])
        circ.reset(len(ids))
        for i in range(L): circ.step(x[:, i])
        rep = circ.report(); cnt = {}
        for rp in rep:
            for k in RULES:
                if rp.get(k): cnt[k] = cnt.get(k, 0) + 1
        out[t] = dict(n=len(rr), any=sum(1 for rp in rep if any(rp.get(k) for k in RULES)), rules=cnt)
    print(vn, json.dumps(out), flush=True)
