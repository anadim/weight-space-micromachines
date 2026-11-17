"""GSM8K / GSM-Hard with a FROZEN base model and zero training: 8 worked GSM8K training solutions as the prompt
(annotations removed, i.e. plain chain of thought), greedy decoding; optionally the calculator implant reading the model's
own 'a op b =' steps, and optionally a second "rethink" pass. This is the harness behind results/gsm/.

python scripts/run_gsm.py OUT.json [--split test|hard] [--mode plain|calc] [--rethink] [--veto runs/veto_qwen3.pt]
       [--model Qwen/Qwen3-0.6B-Base] [--bs 32] [--max_batches N]

plain     : no implant
calc      : calculator implant calc_s3 attached; it ends each number with the solved veto write (scripts/solve_veto.py)
--rethink : after the first answer, the EVAL CODE appends "Hmm, wait. Let me rethink this.\n" and lets the model go on
            (the implant stays attached); the last '####' answer counts. This nudge is not (yet) written by an implant.
Problems are sorted by question length and run in batches of --bs (left-padded); --max_batches N runs only the first N
batches of that order (for a quick check that matches the full run token for token).
"""
import os, sys, re, json, time, argparse, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach
from micromachines.calcfmt import strip_solution

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--split', default='test', choices=['test', 'hard'])
ap.add_argument('--mode', default='calc', choices=['plain', 'calc'])
ap.add_argument('--rethink', action='store_true')
ap.add_argument('--veto', default='runs/veto_qwen3.pt')
ap.add_argument('--shots', default='data/gsm8k/shots_v1.json')
ap.add_argument('--data', default='data')
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--max_batches', type=int, default=0)
a = ap.parse_args()
shots = json.load(open(a.shots))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev).eval()
net = attach(model, tok, machines=('calc_s3',), veto=a.veto, veto_machines=('calc_s3',), zero_os=False) if a.mode == 'calc' else None
eos = tok.convert_tokens_to_ids('<|endoftext|>')
T = lambda s: tok(s, add_special_tokens=False)['input_ids']

rows = [json.loads(l) for l in open(f'{a.data}/gsm8k/test.jsonl' if a.split == 'test' else f'{a.data}/gsmhard/test.jsonl')]
shot_q = {s['question'] for s in shots}; rows = [r for r in rows if r['question'] not in shot_q]
prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")


@torch.no_grad()
def generate(seqs, max_new=320):
    L = max(len(s) for s in seqs); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    if net is not None: net.prime(x)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    gen = [[] for _ in range(B_)]; done = [False] * B_
    for _ in range(max_new):
        nxt = lg.argmax(-1); nl = nxt.tolist()
        for i in range(B_):
            if done[i]: continue
            gen[i].append(nl[i])
            if nl[i] == eos or (len(gen[i]) % 4 == 0 and '\n\n' in tok.decode(gen[i][-8:])): done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B_, 1)], 1)
        if net is not None: net.step(nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    return gen


def correct(pred, gold):
    try: return pred is not None and abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    except ValueError: return False


t0 = time.time(); res = []
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))
if a.max_batches: order = order[:a.max_batches * a.bs]
for b in range(0, len(order), a.bs):
    idx = order[b:b + a.bs]
    for i, g in zip(idx, generate([prompt + T(f"Question: {rows[i]['question']}\nAnswer:") for i in idx])):
        txt = tok.decode(g).replace('<|endoftext|>', '').split('\n\n')[0]
        gold = rows[i]['answer'].split('####')[1].strip().replace(',', '')
        m = re.search(r'####\s*(-?[\d,]*\.?\d+)', txt); pred = m.group(1).replace(',', '') if m else None
        res.append({'i': i, 'q': rows[i]['question'], 'gold': gold, 'pred': pred, 'ok': correct(pred, gold), 'out': txt})
PH = "Hmm, wait. Let me rethink this.\n"
if a.rethink:                                         # second pass: the rethink phrase after the first answer, implant still attached
    for b in range(0, len(res), a.bs):
        ch = res[b:b + a.bs]
        for r, g in zip(ch, generate([prompt + T(f"Question: {rows[r['i']]['question']}\nAnswer:" + r['out'].rstrip() + '\n' + PH) for r in ch], max_new=360)):
            t2 = tok.decode(g).replace('<|endoftext|>', '').split('\n\nQuestion')[0].split('\n\n\n')[0]
            m2 = re.findall(r'####\s*(-?[\d,]*\.?\d+)', t2); r['first_ok'] = r['ok']; r['rethink_tokens'] = len(g)
            if m2:
                r['out2'] = t2; r['ok'] = correct(m2[-1].replace(',', ''), r['gold'])
n = len(res); acc = 100 * sum(r['ok'] for r in res) / n
if a.rethink: print(f"first pass {100 * sum(r['first_ok'] for r in res) / n:.2f}% -> after rethink {acc:.2f}%")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({'model': a.model, 'mode': a.mode, 'split': a.split, 'rethink': a.rethink, 'n': n, 'acc': acc, 'shots': len(shots), 'rows': res},
          open(a.out, 'w'), indent=1)
print(f"RESULT {a.model} mode={a.mode} split={a.split}: {sum(r['ok'] for r in res)}/{n} = {acc:.2f}% | "
      f"{time.time() - t0:.0f}s")
