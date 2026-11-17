"""BBH PEEK copy of teamB/run_eval.py (data source = official BBH json; everything else unchanged).
Evaluate a host (frozen base or our fine-tune) on Team B's generated sets, with or without the tools attached.
Same decoding as scripts/run_bbh.py: greedy, left-padded batches, 4-token minimum, stop at '\\n\\nQ' or <|endoftext|>.
BBH-style sets use the OFFICIAL 3-shot CoT prompt of the task, unchanged; out-of-style sets use a zero-shot 'Q: ...\\nA: ...'.

python run_eval.py OUT.json --model Qwen/Qwen3-0.6B-Base|runs/ft --sets temporal_dev,navigate_dev --tools none|all|interval_gen,...
"""
import os, sys, re, json, time, argparse, torch
REPO = os.environ.get('WSM_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'bbh_generic', 'teamB')); sys.path.insert(0, REPO)
import circuits_B
from circuits_B import TOOLS_B
from micromachines.machine import attach
from transformers import AutoTokenizer, AutoModelForCausalLM

BBH_NAME = {'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies'}
HERE = os.path.join(REPO, 'bbh_generic', 'teamB'); BBH_DATA = os.environ.get('BBH_DIR', os.path.join(REPO, 'data', 'bbh', 'data'))


def prompt_for(task, q, zero_shot=False):
    if task in BBH_NAME and not zero_shot:
        pre = open(f'{HERE}/cot/{BBH_NAME[task]}.txt').read().split('-----', 1)[1].strip() + '\n\n'
        return pre + f"Q: {q}\nA: Let's think step by step."
    return f"Q: {q}\nA: Let's think step by step."


def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''


def norm(x, task):
    x = x.strip().strip('.').strip()
    if task == 'temporal':
        m = re.match(r'\(?([A-D])\)?(\s.*)?$', x); return '(' + m.group(1) + ')' if m else x
    x = re.sub(r'\s+', ' ', x.lower())
    return re.sub(r'(\d(?:am|pm)?)\s*(?:-|to)\s*(\d)', r'\1-\2', x)      # a time range: 'X to Y' == 'X-Y'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--sets', required=True); ap.add_argument('--tools', default='none')
    ap.add_argument('--bs', type=int, default=32); ap.add_argument('--max_new', type=int, default=400)
    ap.add_argument('--min_tokens', type=int, default=4); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--zero_shot', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    tools = TOOLS_B if a.tools == 'all' else () if a.tools == 'none' else tuple(a.tools.split(','))
    net = None
    if tools:
        t0 = time.time(); net = attach(model, tok, machines=tools)
        print(f'attached {tools} ({time.time() - t0:.0f}s)', flush=True)
    eos = tok.convert_tokens_to_ids('<|endoftext|>')

    @torch.no_grad()
    def generate(prompts, max_new):
        seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B_ = len(seqs)
        x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev); att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
        if net is not None: net.prime(x)
        o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values; lg = model.lm_head(o.last_hidden_state[:, -1]).float()
        gen = [[] for _ in range(B_)]; done = [False] * B_
        for step in range(max_new):
            if step < a.min_tokens: lg[:, eos] = -1e9
            nxt = lg.argmax(-1); nl = nxt.tolist()
            for i in range(B_):
                if done[i]: continue
                gen[i].append(nl[i])
                if nl[i] == eos or (len(gen[i]) % 6 == 0 and '\n\nQ' in tok.decode(gen[i][-12:])): done[i] = True
            if all(done): break
            nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt); att = torch.cat([att, att.new_ones(B_, 1)], 1)
            if net is not None: net.step(nxt)
            o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
            lg = model.lm_head(o.last_hidden_state[:, -1]).float()
        return gen

    t0 = time.time(); rows, per = [], {}
    for sname in a.sets.split(','):
        task = sname.split('_')[0]
        ex = [{'id': i, 'question': e['input'], 'target': e['target']} for i, e in enumerate(json.load(open(f'{BBH_DATA}/{BBH_NAME[sname]}.json'))['examples'])][:a.limit or None]
        for b in range(0, len(ex), a.bs):
            ch = ex[b:b + a.bs]
            for e, g in zip(ch, generate([prompt_for(task, e['question'], a.zero_shot) for e in ch], a.max_new)):
                txt = tok.decode(g).replace('<|endoftext|>', ''); p = extract(txt); ok = norm(p, task) == norm(e['target'], task)
                called = '<tool_call>' in txt
                per.setdefault(sname, []).append(ok)
                rows.append({'set': sname, 'id': e['id'], 'gold': e['target'], 'pred': p, 'ok': ok, 'called': called, 'out': txt,
                             'ref_call': e.get('call')})
        n = len(per[sname]); print(f'{sname:20s} {100 * sum(per[sname]) / n:5.1f} (n={n}) | {time.time() - t0:.0f}s', flush=True)
    acc = {t: 100 * sum(v) / len(v) for t, v in per.items()}
    json.dump({'model': a.model, 'tools': list(tools), 'per_set': acc, 'rows': rows}, open(a.out, 'w'), indent=1)
