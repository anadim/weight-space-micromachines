"""GSM8K / GSM-Hard with the frozen host and the GENERIC generation loop of scripts/run_gsm.py (the plain/calc loop),
optionally with the compiled rethink implant (micromachines/rethink_circuit.py) attached. The harness knows nothing
about the implant: it attaches it (hooks) before generating, then only generates, stops and scores, exactly as for the
plain and calc arms. Stop rule (unchanged): <|endoftext|>, or a blank line in the last 8 tokens checked every 4 tokens,
or --max_new tokens. Score (same for every arm): the text up to the first blank line; the LAST '####' number in it.

v2: the implant is micromachines/rethink_circuit_v2.py (mistake detector + rethink). Same loop, same stop rule, same
score, same --max_new for every arm.
python scripts/rethink_v2/run_gsm_v2.py OUT.json [--split test|hard|train_sel] [--mode plain|calc] [--implant none|always|gated]
       [--intervene hash|end] [--cap N] [--detector FILE] [--max_new 700] [--bs 32] [--max_batches N] [--start_batch N]
"""
import os, sys, re, json, time, argparse, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach
from micromachines.calcfmt import strip_solution

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--split', default='test', choices=['test', 'hard', 'train_sel', 'scaled_sel', 'train_rep', 'scaled_rep'])
ap.add_argument('--mode', default='plain', choices=['plain', 'calc'])
ap.add_argument('--implant', default='always', choices=['none', 'always', 'gated'])
ap.add_argument('--detector', default='results/rethink_v2/detector.pt')
ap.add_argument('--intervene', default='hash', choices=['hash', 'end', 'hybrid'])
ap.add_argument('--cap', type=int, default=0)
ap.add_argument('--dirs', default='results/rethink_v2/write_dirs_all.pt')
ap.add_argument('--veto', default='runs/veto_qwen3.pt')
ap.add_argument('--shots', default='data/gsm8k/shots_v1.json')
ap.add_argument('--data', default='data')
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--max_new', type=int, default=700)
ap.add_argument('--max_batches', type=int, default=0)
ap.add_argument('--start_batch', type=int, default=0)
a = ap.parse_args()
shots = json.load(open(a.shots))
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
net = attach(model, tok, machines=('calc_s3',), veto=a.veto, veto_machines=('calc_s3',), zero_os=False) if a.mode == 'calc' else None
if a.implant != 'none':                     # attached after the calculator: its hook sees the calculator's write
    from micromachines.rethink_circuit_v2 import RethinkCircuitV2
    rc = RethinkCircuitV2(model, tok, mode=a.implant, detector=a.detector if a.implant == 'gated' else None,
                          intervene=a.intervene, cap=a.cap, dirs=a.dirs)
    print(f'rethink v2 attached ({a.implant}, intervene={a.intervene}, cap={a.cap}): {rc.R} sequences, '
          f'{len(rc.out_ids)} write dirs, {rc.n_params():,} fixed numbers, smallest margin {rc.margins:.1f}', flush=True)
eos = tok.convert_tokens_to_ids('<|endoftext|>')
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
SRC = {'test': 'gsm8k/test.jsonl', 'hard': 'gsmhard/test.jsonl', 'train_sel': 'gsm8k/train_sel.jsonl', 'scaled_sel': 'gsm8k/train_scaled_sel.jsonl', 'train_rep': 'gsm8k/train_rep.jsonl', 'scaled_rep': 'gsm8k/train_scaled_rep.jsonl'}
rows = [json.loads(l) for l in open(f'{a.data}/{SRC[a.split]}')]
shot_q = {s['question'] for s in shots}; rows = [r for r in rows if r['question'] not in shot_q]
prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")


@torch.no_grad()
def generate(seqs, max_new):                # = scripts/run_gsm.py generate(), max_new configurable
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


NUM = r'####\s*(-?[\d,]*\.?\d+)'
t0 = time.time(); res = []
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))
order = order[a.start_batch * a.bs:]
if a.max_batches: order = order[:a.max_batches * a.bs]
for b in range(0, len(order), a.bs):
    idx = order[b:b + a.bs]
    for i, g in zip(idx, generate([prompt + T(f"Question: {rows[i]['question']}\nAnswer:") for i in idx], a.max_new)):
        txt = tok.decode(g).replace('<|endoftext|>', '').split('\n\n')[0]
        gold = rows[i]['answer'].split('####')[1].strip().replace(',', '')
        m = re.findall(NUM, txt); pred = m[-1].replace(',', '') if m else None
        res.append({'i': i, 'gold': gold, 'pred': pred, 'ok': correct(pred, gold), 'out': txt, 'ids': g, 'n_tok': len(g)})
    print(f'batch {b // a.bs}: {sum(r["ok"] for r in res)}/{len(res)} | {time.time() - t0:.0f}s', flush=True)
n = len(res); acc = 100 * sum(r['ok'] for r in res) / n
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({'model': a.model, 'intervene': a.intervene, 'cap': a.cap, 'detector': a.detector if a.implant == 'gated' else None, 'mode': a.mode, 'split': a.split, 'implant': a.implant, 'max_new': a.max_new, 'bs': a.bs,
           'n': n, 'acc': acc, 'rows': res}, open(a.out, 'w'))
print(f"RESULT mode={a.mode} implant={a.implant} split={a.split}: {sum(r['ok'] for r in res)}/{n} = {acc:.2f}% | {time.time() - t0:.0f}s")
