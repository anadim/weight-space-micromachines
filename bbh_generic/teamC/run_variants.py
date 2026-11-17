"""Run a judge's reworded variants (protocol rule 11).
python run_variants.py variants.json out.json
variants.json: list of
  {"id": .., "kind": "host",  "task": "<bbh task or ood task>", "question": "...", "target": "..."}   host + tools vs plain host
  {"id": .., "kind": "alone", "tool": "expr_gen|bool_gen|calc_gen|expr_call|bool_call|calc_call", "text": "...", "expect": "..."}
     machine alone on the text (it ends where the tool should fire); expect = what it should write ('' = silent)
"""
import os, sys, json, re, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.environ.get('WSM_REPO', os.path.join(HERE, '..', '..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
import teamc_tools as TT
from micromachines import SPECS, first
V = json.load(open(sys.argv[1])); out = []
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', torch_dtype=torch.bfloat16).cuda().eval()
E = model.get_input_embeddings(); eos = tok.convert_tokens_to_ids('<|endoftext|>')
NAT = ('expr_gen', 'calc_gen', 'bool_gen')
OOD = {t: f'{HERE}/data/ood/{t}.prompt.txt' for t in ('num_shop', 'num_list', 'bool_code')}


def prompt(task):
    if task in OOD: return open(OOD[task]).read()
    return open(f'{HERE}/cot/{task}.txt').read().split('-----', 1)[1].strip() + '\n\n'


@torch.no_grad()
def gen(A, text, max_new=400):
    ids = tok(text, add_special_tokens=False)['input_ids']; x = torch.tensor([ids], device='cuda')
    if A is not None: A.prime(x)
    o = model.model(input_ids=x, use_cache=True); past = o.past_key_values; lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    g, s = [], ''
    for step in range(max_new):
        if step < 4: lg[:, eos] = -1e9
        w = False
        if A is not None:
            pd = A.pending[:, -1] if A.pending.dim() == 3 else A.pending; w = bool(pd.abs().max() > 0)
        n = int(lg.argmax(-1)); g.append(n); s += f'[{tok.decode([n])}]' if w else tok.decode([n])
        if n == eos or '\n\nQ' in tok.decode(g[-12:]): break
        nx = torch.tensor([[n]], device='cuda')
        if A is not None: A.step(nx[:, 0])
        o = model.model(input_ids=nx, past_key_values=past, use_cache=True); past = o.past_key_values; lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    return tok.decode(g).replace('<|endoftext|>', ''), s.replace('<|endoftext|>', '')


def extract(t):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', t.split('\n\nQ:')[0]); return m[-1].strip() if m else ''


def same(p, t):
    p = p.strip().strip('.').strip(); m = re.search(r'-?\d+', p.replace(',', ''))
    return (m.group() == t) if re.fullmatch(r'-?\d+', t) and m else p.lower() == t.lower()


host = [v for v in V if v['kind'] == 'host']; alone = [v for v in V if v['kind'] == 'alone']
if host:
    plain = {v['id']: gen(None, prompt(v['task']) + f"Q: {v['question']}\nA: Let's think step by step.") for v in host}
    A = TT.attach_gen(model, tok, NAT, veto=f'{HERE}/runs/veto_qwen3.pt')      # the deployed configuration (run_gen.py)
    for v in host:
        txt, marked = gen(A, prompt(v['task']) + f"Q: {v['question']}\nA: Let's think step by step.")
        pp, pt = extract(plain[v['id']][0]), extract(txt)
        out.append({**v, 'plain_pred': pp, 'plain_ok': same(pp, v['target']), 'tool_pred': pt, 'tool_ok': same(pt, v['target']),
                    'tool_output_marked': marked.split('\n\nQ:')[0]})
    A.detach()
if alone:
    calls = [v for v in alone if v['tool'].endswith('_call')]
    if calls: TT.prep_call_tokens(model, tok)
    A = TT.attach_gen(model, tok, tuple(sorted({v['tool'] for v in alone})), veto=f'{HERE}/runs/veto_qwen3.pt')
    for v in alone:
        M = A.nets[v['tool']]; I = A.all_ids[v['tool']]; outs = SPECS[v['tool']]['out']; ot = [first(I[s]) for s in outs]
        ids = tok(v['text'], add_special_tokens=False)['input_ids']; w = []
        with torch.no_grad():
            e = M.prime_sym(M.symbols(E(torch.tensor([ids], device='cuda')).float()))[:, -1]
            for _ in range(20):
                if e.max() < 0.5: break
                j = int(e.argmax())
                if outs[j] in ('OE', 'VD') and not v['tool'].endswith('_call'): w.append(-1); break   # end marker (veto / '.\n' / '.')
                w.append(ot[j])
                if outs[j] == 'OE' or (v['tool'] == 'bool_gen' and outs[j] in ('VT', 'VF') and 'VD' not in outs): break
                e = M.step_sym(M.symbols(E(torch.tensor([w[-1]], device='cuda')).float()))[:, 0]
        got = tok.decode([t for t in w if t >= 0]) + ('<END>' if w and w[-1] == -1 else '')
        g = got.replace(TT.CALL_OS, '').replace(TT.CALL_OE, '').replace('<END>', '')
        if g.endswith('.\n'): g = g[:-2]
        if v['tool'] == 'bool_gen' and g.endswith('.'): g = g[:-1]
        silent_ok = got in ('', '<END>', TT.CALL_OS + TT.CALL_OE)
        out.append({**v, 'wrote': got, 'ok': (g.strip() == v['expect'].strip()) if v['expect'] else silent_ok})
json.dump(out, open(sys.argv[2], 'w'), indent=1)
for r in out:
    print(json.dumps({k: r[k] for k in r if k not in ('tool_output_marked',)}))
    if 'tool_output_marked' in r: print('   ', r['tool_output_marked'][-400:].replace('\n', ' | '))
