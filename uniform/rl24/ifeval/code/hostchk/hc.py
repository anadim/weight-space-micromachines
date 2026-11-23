"""HOST CHECK (task [26]): plain / + 14-rule implant on an IFEval-like DEV set, for Base vs Instruct hosts.
Same decode loop as f14/run_c.py (left pad 151643 stepped through the implant, write hook on the final-norm input during
generation only, greedy). Prompt formats: chat (Qwen3 template, enable_thinking=False; the Qwen3-0.6B tokenizer is used for every
host, same vocab) or plain ("Instruction: ...\n\nResponse:\n", reply truncated at a new "Instruction:" line).
python hc.py DATA OUT --model Qwen/Qwen3-0.6B-Base --fmt chat [--sd X.pt] [--bs 52] [--max_new 1024]"""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'f14'))
from execu import Executor
from transformers import AutoTokenizer, AutoModelForCausalLM
EOS = [151645, 151643]
ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('out'); ap.add_argument('--model', required=True)
ap.add_argument('--fmt', default='chat'); ap.add_argument('--sd', default=''); ap.add_argument('--bs', type=int, default=52)
ap.add_argument('--max_new', type=int, default=1024); ap.add_argument('--every', type=int, default=2)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
docs = [json.loads(l) for l in open(a.data)][::a.every]
def fmt(d):
    if a.fmt == 'chat':
        return tok(tok.apply_chat_template([{'role': 'user', 'content': d['prompt']}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False), add_special_tokens=False).input_ids
    return tok(f"Instruction: {d['prompt']}\n\nResponse:\n", add_special_tokens=False).input_ids
P_all = [fmt(d) for d in docs]
order = sorted(range(len(docs)), key=lambda i: (len(P_all[i]), docs[i]['key']))
batches = [order[b:b + a.bs] for b in range(0, len(order), a.bs)]
EX = Executor(torch.load(a.sd), device='cuda', host_E=model.lm_head.weight) if a.sd else None
state = {'on': False}
def _hook(mod, x):
    if not state['on'] or EX is None: return None
    h = x[0]; W = EX.write(h[:, -1]); return (h + W[:, None].to(h.dtype),)
model.model.norm.register_forward_pre_hook(_hook)
@torch.no_grad()
def run(idx):
    P = [P_all[i] for i in idx]; B = len(P); L = max(map(len, P))
    x = torch.tensor([[EOS[1]] * (L - len(p)) + p for p in P], device='cuda')
    att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in P], device='cuda')
    out = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device='cuda'); nf = torch.zeros(B, device='cuda')
    if EX is not None:
        EX.reset(B)
        for t in range(L): EX.step(x[:, t])
    state['on'] = True
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    for step in range(a.max_new):
        if EX is not None: nf += (EX.wout['active'] > 0.5).float()
        nxt = lg.argmax(-1)
        fin = (nxt == EOS[0]) | (nxt == EOS[1])
        for i in torch.nonzero(~done & ~fin)[:, 0].tolist(): out[i].append(nxt[i].item())
        done = done | fin
        if bool(done.all()): break
        if a.fmt == 'plain' and step % 32 == 31:
            for i in torch.nonzero(~done)[:, 0].tolist():
                if '\nInstruction:' in tok.decode(out[i][-40:]): done[i] = True
            if bool(done.all()): break
        nxt = torch.where(done, torch.full_like(nxt, EOS[1]), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        if EX is not None: EX.step(nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    state['on'] = False
    res = []
    for j, i in enumerate(idx):
        r = tok.decode(out[j]); fin = bool(done[j])
        if a.fmt == 'plain' and '\nInstruction:' in r: r = r.split('\nInstruction:')[0]; fin = True
        circ = {}
        if EX is not None:
            circ = {'fires': int(EX.reg('fires')[j]), 'nwrite': int(nf[j])}
            for k in ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'pps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul', 'star3']:
                if EX.reg('L.' + k)[j] > 0: circ['L.' + k] = 1
        res.append({'key': docs[i]['key'], 'response': r.strip(), 'finish': 'stop' if fin else 'length', 'ntok': len(out[j]), 'circ': circ})
    return res
t0 = time.time(); rows = []
for bi, b in enumerate(batches):
    rows += run(b); print(f'{bi + 1}/{len(batches)} {time.time() - t0:.0f}s', flush=True)
rows.sort(key=lambda r: r['key'])
with open(a.out, 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print('done', a.out, time.time() - t0, flush=True)
