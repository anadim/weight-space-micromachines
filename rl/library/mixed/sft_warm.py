"""Short SFT warm-start (format only) before the mixed-task RL. Plain host fine-tune, NO implants attached, no wiring
(implants and wires untouched; wires are created fresh at RL start). Data (held out from the 5k RL pool and the 1k test):
  - BBH call/passive format: bbh_integrate train_union_v2 (432 rows: Team A/B call transcripts with <tool_call>NAME ARGS
    </tool_call><tool_response>..</tool_response>, Team C passive '=' / 'is' transcripts, 100 no-call negatives), the
    question re-wrapped in our chat format; any row whose question appears in the mixed test set is dropped.
  - GSM calculator format: GSM8K-train originals of split A (the RL pool holds only their SCALED copies), gold solution
    with annotations stripped ('48/2 = 24'), ending 'The answer is N.' (n = --n_gsm).
  - IFEval: none (the chat format alone sets the circuit's assistant-role register).
Loss on the assistant completion only. Usage: python sft_warm.py --out DIR [--steps 80 --lr 1e-5]"""
import os, sys, json, re, random, argparse, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.environ.get('RLW_DIR', 'SET_VIA_ENV'))
import mixed_task as MT, common as C
from micromachines.calcfmt import strip_solution
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--host', default='Qwen/Qwen3-0.6B')
ap.add_argument('--steps', type=int, default=80); ap.add_argument('--bs', type=int, default=16); ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--n_gsm', type=int, default=200); ap.add_argument('--noisy', type=float, default=0.0); ap.add_argument('--union', default=os.environ.get('UNION', 'SET_VIA_ENV'))
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.host)
test_q = {r.get('q', '') for r in MT.load('test')}
rows = []
for l in open(a.union):
    r = json.loads(l)
    q = r['prompt'].split('Q: ')[-1]; q = re.sub(r"\nA: Let's think step by step\.?\s*$", '', q).strip()
    if q in test_q: continue
    comp = r['completion'].strip()
    comp = re.sub(r'So the answer is (.*?)\.?\s*$', r'The answer is \1.', comp)
    rows.append(('bbh:' + r['team'], q, comp))
tr = [json.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')]
g = [r for r in tr if C.part(r['question']) == 'A']; random.Random(5).shuffle(g)
for r in g[:a.n_gsm]:
    body, fin = strip_solution(r['answer'])
    rows.append(('gsm', r['question'], body.strip() + f'\nThe answer is {fin}.'))
if a.noisy > 0:                                     # noisy-routing SFT: misplaced tool calls in a fraction of rows
    import re as _re
    calls = []
    for k, u, c in rows:
        calls += _re.findall(r'<tool_call>[^<]*</tool_call><tool_response>[^<]*</tool_response>', c)
    rn = random.Random(11); new = []; nmis = {}
    for k, u, c in rows:
        if rn.random() < a.noisy:
            lines = c.split('\n')
            if k == 'gsm' or k == 'bbh:nocall':        # a BBH call (track/order/interval/walk/truth) where none belongs
                ins = 'Let me check with a tool: ' + rn.choice(calls); tag = 'call_in_' + k
            else:                                      # a calculator equation in a BBH row that needs none
                x, y = rn.randint(2, 99), rn.randint(2, 99); op = rn.choice(['+', '*', '-'])
                ins = f'Side note: {x} {op} {y} = {eval(f"{x}{op}{y}")}.'; tag = 'calc_in_' + k
            lines.insert(max(1, len(lines) - 1), ins); c = '\n'.join(lines); nmis[tag] = nmis.get(tag, 0) + 1
        new.append((k, u, c))
    rows = new
    print('noisy-routing insertions', nmis, flush=True)
print('SFT rows', len(rows), {k: sum(1 for x in rows if x[0] == k) for k in sorted({x[0] for x in rows})}, flush=True)
ex = []
for kind, u, c in rows:
    p = tok.apply_chat_template([{'role': 'system', 'content': MT.SYS}, {'role': 'user', 'content': u}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    pi = tok(p, add_special_tokens=False)['input_ids']; ci = tok(c + '<|im_end|>', add_special_tokens=False)['input_ids']
    if len(pi) + len(ci) <= 1400: ex.append((pi, ci))
model = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.float32).cuda()
model.gradient_checkpointing_enable()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
rng = random.Random(0)
for step in range(1, a.steps + 1):
    lr = a.lr * min(1.0, step / 5) * (0.5 * (1 + __import__('math').cos(3.14159 * step / a.steps)))
    for gr in opt.param_groups: gr['lr'] = lr
    batch = rng.sample(ex, a.bs); tot = 0.0; ntok = sum(len(c) for _, c in batch)
    for pi, ci in batch:
        ids = torch.tensor([pi + ci]).cuda(); lab = torch.tensor([[-100] * len(pi) + ci]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(ids, labels=lab)
        loss = out.loss * len(ci) / ntok; loss.backward(); tot += float(loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step % 10 == 0 or step == 1: print('step', step, 'loss', round(tot, 4), flush=True)
model.to(torch.bfloat16).save_pretrained(a.out); tok.save_pretrained(a.out)
json.dump(vars(a) | {'n_rows': len(ex)}, open(os.path.join(a.out, 'sft_config.json'), 'w'), indent=1)
print('saved', a.out, flush=True)
