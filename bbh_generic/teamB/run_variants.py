"""Run judge-written variants (protocol rule 11).
variants JSON: [{"style": "temporal|navigate|lies|<other>", "question": ..., "target": ..., "call": "NAME ARGS" (optional),
                 "expect": "RESULT" (optional, judge's own hand-computed tool result)}]
* host + tools: the fine-tuned host with all three tools attached, official 3-shot prompt for temporal/navigate/lies styles,
  zero-shot 'Q: ...\\nA: Let's think step by step.' otherwise; reports the generation, whether a call was made, the answer.
* tool alone: each given call is run through the compiled tool (teacher forced, as in test_tools.py) and compared with
  the judge's expected result and with tools_ref.run_ref.
python run_variants.py judge/variants_interval_1.json judge/results_interval_1.json --model runs/ft2
"""
import sys, json, re, argparse, torch
sys.path.insert(0, '.')
import circuits_B
from circuits_B import TOOLS_B
from micromachines.machine import attach, Micromachine, symbol_token_ids
from micromachines.specs import SPECS
from transformers import AutoTokenizer, AutoModelForCausalLM
from run_eval import prompt_for, extract, norm
from tools_ref import run_ref, CS, CE, OS, OE

ap = argparse.ArgumentParser()
ap.add_argument('variants'); ap.add_argument('out'); ap.add_argument('--model', default='runs/ft2')
a = ap.parse_args()
V = json.load(open(a.variants))
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
net = attach(model, tok, machines=TOOLS_B)
eos = tok.convert_tokens_to_ids('<|endoftext|>')


@torch.no_grad()
def gen(prompt, max_new=400):
    x = torch.tensor([tok(prompt, add_special_tokens=False)['input_ids']]).cuda()
    net.prime(x); o = model(input_ids=x, use_cache=True); past = o.past_key_values; lg = o.logits[:, -1].float(); out = []
    for step in range(max_new):
        if step < 4: lg[:, eos] = -1e9
        t = lg.argmax(-1); out.append(t.item())
        if t.item() == eos or '\n\nQ' in tok.decode(out[-12:]): break
        net.step(t); o = model(input_ids=t[:, None], past_key_values=past, use_cache=True); past = o.past_key_values; lg = o.logits[:, -1].float()
    return tok.decode(out).replace('<|endoftext|>', '')


def tool_alone(call):
    """write what the attached tools write after '<tool_call>CALL</tool_call>' (forced decoding of the tool's own tokens only)"""
    seq = tok(f'x.\n{CS}{call}{CE}', add_special_tokens=False)['input_ids']
    x = torch.tensor([seq]).cuda(); net.prime(x, full=True)
    d = model.config.hidden_size
    last = lambda: net.pending.reshape(-1, d)[-1].float()               # the write at the newest position
    w = last(); written = []
    for _ in range(80):
        if w.abs().max() == 0: break
        h = model.model.norm(w.view(1, 1, d).to(model.dtype)); t = int(model.lm_head(h).float().argmax(-1).flatten()[0])  # the write alone decides the token
        written.append(t); net.step(torch.tensor([t], device='cuda')); w = last()
        if t == tok.convert_tokens_to_ids(OE): break
    return tok.decode(written) if written else '(silent)'


res = []
for v in V:
    task = v.get('style') if v.get('style') in ('temporal', 'navigate', 'lies') else 'other'
    r = dict(v)
    if v.get('question'):
        txt = gen(prompt_for(task, v['question']))
        p = extract(txt); tk = 'temporal' if task == 'temporal' else 'x'
        golds = [v['target']] + list(v.get('accept') or [])
        calls = re.findall(r'<tool_call>(.*?)</tool_call><tool_response>(.*?)</tool_response>', txt)
        r.update(gen=txt, pred=p, called='<tool_call>' in txt, ok=any(norm(p, tk) == norm(g_, tk) for g_ in golds),
                 host_calls=[{'call': c, 'tool_wrote': w, 'strict_ref': run_ref(c), 'tool_correct': (w == run_ref(c))} for c, w in calls])
    if v.get('call'):
        w = tool_alone(v['call']); ref = run_ref(v['call'])
        r.update(tool_wrote=w, tool_ref=ref, tool_matches_ref=(w == f'{OS}{ref}{OE}') if ref is not None else (w == '(silent)'),
                 tool_matches_expect=(w == ('(silent)' if v.get('expect') in (None, 'SILENT') else f'{OS}{v["expect"]}{OE}')) if 'expect' in v else None)
    res.append(r)
    print(json.dumps({k: r.get(k) for k in ('id', 'style', 'target', 'pred', 'ok', 'host_calls', 'call', 'tool_wrote', 'tool_matches_ref', 'tool_matches_expect')}), flush=True)
json.dump(res, open(a.out, 'w'), indent=1)
q = [r for r in res if 'ok' in r]; c = [r for r in res if 'tool_wrote' in r]
print(f'host+tools correct {sum(r["ok"] for r in q)}/{len(q)}, called {sum(r["called"] for r in q)}/{len(q)}; '
      f'tool alone matches ref {sum(r["tool_matches_ref"] for r in c)}/{len(c)}')
