"""single-prompt trace of the emitter / readouts (debug only)"""
import sys, json, torch
from common import load, EOS_IDS
from circuit import Circuit
tok, model = load(); C = Circuit(tok, model)
d = [json.loads(l) for l in open(sys.argv[1]) if json.loads(l)['key'] == int(sys.argv[2])][0]
maxn = int(sys.argv[3]) if len(sys.argv) > 3 else 300
m = ([{'role': 'system', 'content': d['system']}] if d.get('system') else []) + (d.get('history') or []) + [{'role': 'user', 'content': d['prompt']}]
ids = tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False).input_ids
st = {'on': False}; log = []
def hook(mod, x):
    if not st['on']: return None
    h = x[0]; W = C.write(h[:, -1])
    log.append(dict(prog=int(C.prog[0]), pj=int(C.pj[0]), body=int(C.body[0]), kp=int(C.kp[0]), d3=int(C.d3[0]), k6=int(C.k6[0]), bc=int(C.bcount[0]),
                    hc=int(C.hcount[0]), seg=int(C.seg[0]), atls=int(C.at_ls[0]), Wn=float(W.norm())))
    return (h + W[:, None].to(h.dtype),)
model.model.norm.register_forward_pre_hook(hook)
C.reset(1)
x = torch.tensor([ids], device='cuda')
with torch.no_grad():
    for t in range(x.shape[1]): C.step(x[:, t])
    st['on'] = True
    o = model.model(input_ids=x, use_cache=True); past = o.past_key_values
    out = []
    for s in range(maxn):
        nxt = model.lm_head(o.last_hidden_state[:, -1]).argmax(-1)
        out.append(nxt.item()); print(repr(tok.decode([nxt.item()])), log[-1])
        if nxt.item() in EOS_IDS: break
        C.step(nxt)
        o = model.model(input_ids=nxt[:, None], past_key_values=past, use_cache=True); past = o.past_key_values
print(C.report())
