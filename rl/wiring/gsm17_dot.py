"""Qwen3-1.7B GSM8K / GSM-Hard transfer eval, SAME protocol as scripts/run_gsm.py (8 worked GSM8K shots, annotations
stripped, greedy, stop at a blank line or <|endoftext|>, max 320 new tokens, problems sorted by length, batches of --bs,
first '#### n' scored), with the calculator machine selectable (calc_s3 = the transfer runs; calc_s4 = has the dot veto)
and per-token implant write flags recorded.

Dot-tail misfire = an implant write at a generated token whose previous token was written by the HOST, ends with '.',
and follows a number (e.g. host writes '= 72.' to end a sentence, the implant then appends a fraction digit).
python gsm17_dot.py OUT.json --machine calc_s4 --split test --veto veto_q17.pt
"""
import os, sys, re, json, time, argparse, torch
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WSM_ROOT', os.path.dirname(os.path.dirname(HERE)))   # repo root
sys.path.insert(0, REPO)
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach
from micromachines.calcfmt import strip_solution

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--model', default='Qwen/Qwen3-1.7B-Base')
ap.add_argument('--split', default='test', choices=['test', 'hard'])
ap.add_argument('--mode', default='calc', choices=['plain', 'calc'])
ap.add_argument('--machine', default='calc_s3', choices=['calc_s3', 'calc_s4'])
ap.add_argument('--veto', required=True)
ap.add_argument('--data', default=os.path.join(REPO, 'data'))
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--max_batches', type=int, default=0)
a = ap.parse_args()
shots = json.load(open(f'{a.data}/gsm8k/shots_v1.json'))
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
net = attach(model, tok, machines=(a.machine,), veto=a.veto, veto_machines=(a.machine,),
             zero_os=(a.machine != 'calc_s3')) if a.mode == 'calc' else None
eos = tok.convert_tokens_to_ids('<|endoftext|>')
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
rows = [json.loads(l) for l in open(f'{a.data}/gsm8k/test.jsonl' if a.split == 'test' else f'{a.data}/gsmhard/test.jsonl')]
shot_q = {s['question'] for s in shots}; rows = [r for r in rows if r['question'] not in shot_q]
prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")


def wflag():
    if net is None or net.pending is None: return None
    p = net.pending
    return (p.reshape(p.shape[0], -1, p.shape[-1])[:, -1].abs().sum(-1) > 0).tolist()


@torch.no_grad()
def generate(seqs, max_new=320):
    L = max(len(s) for s in seqs); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    if net is not None: net.prime(x)
    wf = wflag()                                      # write applied to the prediction of the FIRST generated token
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    gen = [[] for _ in range(B_)]; fl = [[] for _ in range(B_)]; done = [False] * B_
    for _ in range(max_new):
        nxt = lg.argmax(-1); nl = nxt.tolist()
        for i in range(B_):
            if done[i]: continue
            gen[i].append(nl[i]); fl[i].append(bool(wf[i]) if wf is not None else False)
            if nl[i] == eos or (len(gen[i]) % 4 == 0 and '\n\n' in tok.decode(gen[i][-8:])): done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B_, 1)], 1)
        if net is not None: net.step(nxt)
        wf = wflag()
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    return gen, fl


def correct(pred, gold):
    try: return pred is not None and abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    except ValueError: return False


def dot_tail(g, f):
    """implant writes right after a HOST-written token that ends with '.' and follows a number (sentence-end dot)."""
    n = 0; ex = []
    for t in range(2, len(g)):
        if not f[t] or f[t - 1]: continue
        prev = tok.decode([g[t - 1]])
        if prev.endswith('.') and re.search(r'\d\s*\.?$', tok.decode(g[max(0, t - 4):t - 1]) + prev.rstrip('.')):
            n += 1
            if len(ex) < 2: ex.append(tok.decode(g[max(0, t - 6):t + 2]))
    return n, ex


order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))
if a.max_batches: order = order[:a.max_batches * a.bs]
NUM = r'####\s*(-?[\d,]*\.?\d+)'
t0 = time.time(); res = []
for b in range(0, len(order), a.bs):
    idx = order[b:b + a.bs]
    gens, fls = generate([prompt + T(f"Question: {rows[i]['question']}\nAnswer:") for i in idx])
    for i, g, f in zip(idx, gens, fls):
        txt = tok.decode(g).replace('<|endoftext|>', '').split('\n\n')[0]
        cut = len(T(txt)) if txt else 0                       # approx. scored span (tokens up to the stop)
        gold = rows[i]['answer'].split('####')[1].strip().replace(',', '')
        m = re.search(NUM, txt); pred = m.group(1).replace(',', '') if m else None
        nd, ex = dot_tail(g, f)
        calls = sum(1 for t in range(len(f)) if f[t] and (t == 0 or not f[t - 1]))
        res.append({'i': i, 'gold': gold, 'pred': pred, 'ok': correct(pred, gold), 'writes': sum(f), 'calls': calls,
                    'dot_tail': nd, 'dot_ex': ex, 'out': txt})
n = len(res); acc = 100 * sum(r['ok'] for r in res) / n
summ = dict(model=a.model, mode=a.mode, machine=a.machine, split=a.split, n=n, acc=acc,
            dot_tail_per_q=sum(r['dot_tail'] for r in res) / n, q_with_dot_tail=sum(r['dot_tail'] > 0 for r in res),
            acc_q_with_dot_tail=(100 * sum(r['ok'] for r in res if r['dot_tail'] > 0) / max(1, sum(r['dot_tail'] > 0 for r in res))),
            calls_per_q=sum(r['calls'] for r in res) / n, writes_per_q=sum(r['writes'] for r in res) / n, secs=time.time() - t0)
json.dump({**summ, 'rows': res}, open(a.out, 'w'), indent=1)
print('RESULT', json.dumps(summ), flush=True)
