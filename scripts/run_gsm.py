"""GSM8K / GSM-Hard with a FROZEN base model and zero training: 8 worked GSM8K training solutions as the prompt
(annotations removed, i.e. plain chain of thought), greedy decoding; optionally the calculator implant reading the model's
own 'a op b =' steps, and optionally a second "rethink" pass. This is the harness behind results/gsm/.

python scripts/run_gsm.py OUT.json [--split test|hard] [--mode plain|calc] [--rethink | --rethink_implant always|gated]
       [--probe results/rethink/rethink_probe_L28.pt] [--veto runs/veto_qwen3.pt] [--model Qwen/Qwen3-0.6B-Base] [--bs 32] [--max_batches N]

plain     : no implant
calc      : calculator implant calc_s3 attached; it ends each number with the solved veto write (scripts/solve_veto.py)
--rethink : REFERENCE path (evaluation code): after the first answer, this script appends "Hmm, wait. Let me rethink this.\n"
            to the text, re-encodes it and lets the model go on (calculator still attached); the last '####' answer counts.
--rethink_implant always|gated : the same nudge written by the `rethink` IMPLANT (micromachines/rethink.py) in ONE
            generation: where the model would end its answer, the implant writes "\nHmm, wait. Let me rethink this.\n"
            token by token instead, then goes silent. 'gated': only if the fitted layer-28 probe fires at the model's '####'
            (--probe, from scripts/gate/export_probe.py). The script only generates, stops (same stop rule as the reference:
            a blank line or <|endoftext|>, at most 320 tokens before and 360 after the phrase) and scores.
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
ap.add_argument('--rethink_implant', default='none', choices=['none', 'always', 'gated'])
ap.add_argument('--probe', default='results/rethink/rethink_probe_L28.pt')
ap.add_argument('--veto', default='runs/veto_qwen3.pt')
ap.add_argument('--shots', default='data/gsm8k/shots_v1.json')
ap.add_argument('--data', default='data')
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--max_batches', type=int, default=0)
ap.add_argument('--start_batch', type=int, default=0, help='skip the first N batches of the order (to split a run into chunks)')
ap.add_argument('--fp32', action='store_true', help='float32 host on GPU (numerics check; the reported runs are bf16)')
a = ap.parse_args()
shots = json.load(open(a.shots))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev == 'cuda' and not a.fp32 else torch.float32).to(dev).eval()
net = attach(model, tok, machines=('calc_s3',), veto=a.veto, veto_machines=('calc_s3',), zero_os=False) if a.mode == 'calc' else None
eos = tok.convert_tokens_to_ids('<|endoftext|>')
assert not (a.rethink and a.rethink_implant != 'none'), 'choose the reference (--rethink) or the implant, not both'
rt = None
if a.rethink_implant != 'none':
    from micromachines.rethink import RethinkImplant, PHRASE
    rt = RethinkImplant(model, tok, mode=a.rethink_implant, probe=a.probe if a.rethink_implant == 'gated' else None)
    print(f'rethink implant attached ({a.rethink_implant}): {rt.n_seqs} forced sequences, {len(rt.out_ids)} write directions, '
          f'smallest margin {rt.margins:.1f}', flush=True)
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


@torch.no_grad()
def generate_implant(seqs, max0=320, max1=360):
    """One generation with the rethink implant attached. Stop rule = the reference's: <|endoftext|>, or a blank line in the
    last 8 tokens checked every 4 tokens, counted from the start (or from the end of the implant's write); at most max0
    tokens before the implant writes and max1 after. Returns tokens and where the implant's write starts / ends."""
    L = max(len(s) for s in seqs); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    if net is not None: net.prime(x)
    rt.prime(x)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    gen = [[] for _ in range(B_)]; done = [False] * B_; seg = [0] * B_; w0 = [None] * B_; w1 = [None] * B_; bad = 0
    for _ in range(max0 + rt.L + max1):
        nxt = lg.argmax(-1); nl = nxt.tolist(); wr = rt.wrote.tolist(); ft = rt.forced_token().tolist()
        for i in range(B_):
            if done[i]: continue
            if wr[i]:                                     # an implant-written token
                bad += nl[i] != ft[i]
                if w0[i] is None: w0[i] = len(gen[i])
                gen[i].append(nl[i]); seg[i] = w1[i] = len(gen[i]); continue
            if seg[i] == 0 and len(gen[i]) >= max0: done[i] = True; continue
            gen[i].append(nl[i]); k = len(gen[i]) - seg[i]
            if nl[i] == eos or (k % 4 == 0 and '\n\n' in tok.decode(gen[i][max(seg[i], len(gen[i]) - 8):])) or (seg[i] and k >= max1):
                done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B_, 1)], 1)
        if net is not None: net.step(nxt)
        rt.step(nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    sc = rt.state['score'].tolist()
    return gen, w0, w1, sc, bad


def correct(pred, gold):
    try: return pred is not None and abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    except ValueError: return False


t0 = time.time(); res = []
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['question']))
order = order[a.start_batch * a.bs:]
if a.max_batches: order = order[:a.max_batches * a.bs]
NUM = r'####\s*(-?[\d,]*\.?\d+)'
for b in range(0, len(order) if rt is not None else 0, a.bs):      # the rethink implant: one generation per problem
    idx = order[b:b + a.bs]
    gens, W0, W1, SC, bad = generate_implant([prompt + T(f"Question: {rows[i]['question']}\nAnswer:") for i in idx])
    if bad: print(f'WARNING: {bad} implant writes did not produce their token', flush=True)
    for i, g, w0, w1, sc in zip(idx, gens, W0, W1, SC):
        gold = rows[i]['answer'].split('####')[1].strip().replace(',', '')
        r = {'i': i, 'q': rows[i]['question'], 'gold': gold, 'probe_score': None if sc != sc else sc, 'implant_wrote': w0 is not None,
             'ids': g, 'write_span': [w0, w1]}
        if w0 is None:                                    # implant silent (gate off): the answer stands as the model ended it
            txt = tok.decode(g).replace('<|endoftext|>', '').split('\n\n')[0]
            m = re.search(NUM, txt); pred = m.group(1).replace(',', '') if m else None
            r.update(out=txt, pred=pred, ok=correct(pred, gold), first_ok=correct(pred, gold), rethink_tokens=0)
        else:
            txt = tok.decode(g[:w0]); forced = tok.decode(g[w0:w1])
            t2 = tok.decode(g[w1:]).replace('<|endoftext|>', '').split('\n\nQuestion')[0].split('\n\n\n')[0]
            m = re.search(NUM, txt); pred = m.group(1).replace(',', '') if m else None; m2 = re.findall(NUM, t2)
            fin = m2[-1].replace(',', '') if m2 else pred
            r.update(out=txt, forced=forced, rt_text=t2, pred=pred, first_ok=correct(pred, gold), ok=correct(fin, gold),
                     rethink_tokens=len(g) - w1)
            if m2: r['out2'] = t2
        res.append(r)
for b in range(0, len(order) if rt is None else 0, a.bs):
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
            m2 = re.findall(r'####\s*(-?[\d,]*\.?\d+)', t2); r['first_ok'] = r['ok']; r['rethink_tokens'] = len(g); r['rt_text'] = t2
            if m2:
                r['out2'] = t2; r['ok'] = correct(m2[-1].replace(',', ''), r['gold'])
n = len(res); acc = 100 * sum(r['ok'] for r in res) / n
if a.rethink or rt is not None: print(f"first pass {100 * sum(r['first_ok'] for r in res) / n:.2f}% -> after rethink {acc:.2f}%")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({'model': a.model, 'mode': a.mode, 'split': a.split, 'rethink': a.rethink, 'rethink_implant': a.rethink_implant, 'n': n, 'acc': acc, 'shots': len(shots), 'rows': res},
          open(a.out, 'w'), indent=1)
print(f"RESULT {a.model} mode={a.mode} split={a.split}: {sum(r['ok'] for r in res)}/{n} = {acc:.2f}% | "
      f"{time.time() - t0:.0f}s")
