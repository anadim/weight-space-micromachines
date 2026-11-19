"""xfamily GSM harness (derived from scripts/run_gsm.py; coordinator fixes 2026-11-17):
 - calculator = calc_s4 (not calc_s3), attached as in BBH (veto ends the number, model writes the space after '=' itself)
 - the spec is sanitized generically for the host vocabulary: every symbol keeps only its single-token alternatives;
   a symbol with NO single-token form -> 'CANNOT_ATTACH' and exit code 3 (reported as a negative result)
 - stop rule: '\\n\\nQuestion:' appears (not a blank line), or <eos>, or max_new tokens, or the first '#### <number>' line is
   complete (the score only uses the first one, so stopping there is equivalent and cheaper)
 - answer = first '#### <number>' in the text before '\\n\\nQuestion'; rows without one are counted as unextracted
Same 8-shot plain-CoT prompt, greedy, left-padded batches sorted by question length.
python run_gsm_x.py OUT.json --model M --split test|hard --mode plain|calc [--veto V] [--bs 32] [--max_new 400]
"""
import os, sys, re, json, time, argparse, torch
sys.path.insert(0, os.environ.get('WSM_REPO', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach
from micromachines.specs import SPECS
from micromachines.calcfmt import strip_solution

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--model', required=True)
ap.add_argument('--split', default='test', choices=['test', 'hard']); ap.add_argument('--mode', default='calc', choices=['plain', 'calc'])
ap.add_argument('--calc_machine', default='calc_s4'); ap.add_argument('--veto', default='')
ap.add_argument('--shots', default='data/gsm8k/shots_v1.json'); ap.add_argument('--data', default='data')
ap.add_argument('--bs', type=int, default=32); ap.add_argument('--max_new', type=int, default=400); ap.add_argument('--max_batches', type=int, default=0)
a = ap.parse_args()
dev = 'cuda'
tok = AutoTokenizer.from_pretrained(a.model)
T = lambda s: tok(s, add_special_tokens=False)['input_ids']


def sanitize(name):
    sp = SPECS[name]; txt = dict(sp['text']); changes = []
    for s in sp['syms'] + [o for o in sp['out'] if o not in sp['syms']]:
        v = txt[s]; alts = list(v) if isinstance(v, (list, tuple)) else [v]
        keep = [x for x in alts if len(T(x)) == 1]
        if not keep:
            print(f'CANNOT_ATTACH: symbol {s!r} of {name} has no single-token form in {a.model} (alternatives {alts!r})', flush=True)
            sys.exit(3)
        if len(keep) != len(alts): changes.append(f'{s}: {alts} -> {keep}')
        txt[s] = keep[0] if len(keep) == 1 else tuple(keep)
    SPECS[name + '_v'] = dict(sp, text=txt)
    return name + '_v', changes


net = None
if a.mode == 'calc':
    cm, ch = sanitize(a.calc_machine)
    print('sanitized', a.calc_machine, '->', cm, '|', '; '.join(ch) or 'no change', flush=True)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
if a.mode == 'calc':
    net = attach(model, tok, machines=(cm,), veto=a.veto, veto_machines=(cm,), zero_os=True)   # reader asserted exact on the vocab
    print('attached; min write margin', min(m for m, _ in net.margins.values()), flush=True)
eos = tok.eos_token_id
shots = json.load(open(a.shots))
rows = [json.loads(l) for l in open(f'{a.data}/gsm8k/test.jsonl' if a.split == 'test' else f'{a.data}/gsmhard/test.jsonl')]
shot_q = {s['question'] for s in shots}
idx_all = [i for i, r in enumerate(rows) if r['question'] not in shot_q]
prompt = []
for s in shots:
    body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}"); prompt += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
STOP = '\n\nQuestion:'
DONE_RE = re.compile(r'####\s*-?[\d,]*\.?\d+[^\S\n]*\n')   # first answer line complete -> later text cannot change the score


@torch.no_grad()
def generate(seqs):
    L = max(len(s) for s in seqs); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    if net is not None: net.prime(x)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    gen = [[] for _ in range(B_)]; done = [False] * B_
    for _ in range(a.max_new):
        nxt = lg.argmax(-1); nl = nxt.tolist()
        for i in range(B_):
            if done[i]: continue
            gen[i].append(nl[i])
            if nl[i] == eos or (len(gen[i]) % 4 == 0 and (STOP in tok.decode(gen[i][-12:]) or DONE_RE.search(tok.decode(gen[i]).split('\n\nQuestion')[0]))): done[i] = True
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
order = sorted(idx_all, key=lambda i: len(rows[i]['question']))
if a.max_batches: order = order[:a.max_batches * a.bs]
for b in range(0, len(order), a.bs):
    idx = order[b:b + a.bs]
    for i, g in zip(idx, generate([prompt + T(f"Question: {rows[i]['question']}\nAnswer:") for i in idx])):
        txt = tok.decode(g, skip_special_tokens=True).split('\n\nQuestion')[0]
        gold = rows[i]['answer'].split('####')[1].strip().replace(',', '')
        m = re.search(NUM, txt); pred = m.group(1).replace(',', '') if m else None
        res.append({'i': i, 'gold': gold, 'pred': pred, 'ok': correct(pred, gold), 'unextracted': pred is None, 'ntok': len(g), 'out': txt})
n = len(res); acc = 100 * sum(r['ok'] for r in res) / n; unx = 100 * sum(r['unextracted'] for r in res) / n
sub = [r for r in res if r['i'] >= 200]; acc_sub = 100 * sum(r['ok'] for r in sub) / max(1, len(sub))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({'model': a.model, 'mode': a.mode, 'split': a.split, 'calc_machine': a.calc_machine if net else None, 'n': n, 'acc': acc,
           'unextracted_pct': unx, 'n_200plus': len(sub), 'acc_200plus': acc_sub, 'stop': STOP, 'max_new': a.max_new, 'rows': res}, open(a.out, 'w'), indent=1)
print(f"RESULT {a.model} mode={a.mode} split={a.split}: {sum(r['ok'] for r in res)}/{n} = {acc:.2f}% | unextracted {unx:.2f}% | "
      f"idx>=200: {acc_sub:.2f}% (n={len(sub)}) | {time.time() - t0:.0f}s", flush=True)
