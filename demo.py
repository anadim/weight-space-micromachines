"""Demo: a frozen Qwen3-0.6B-Base with one implant, the calculator (calc_s3), attached.

The model is prompted with two worked multiplications and writes its own 'a * b ='; the implant reads those tokens and
writes the exact digits; the model then goes on by itself. Implant-written tokens are shown in [brackets].
Then two checks:
  1. until the implant's first write, the generation's logits are bit-identical to the plain model's;
  2. on a text with no trigger, the full-sequence logits are bit-identical with and without the implant.

python demo.py [--device cpu|cuda] [--a 59283 --b 914]
Downloads Qwen/Qwen3-0.6B-Base (~1.2 GB) from Hugging Face on first use. CPU, fp32: about 1-3 minutes.
No veto vector is needed here: without it the implant ends its number with a space.
"""
import os
# Intel MKL (the CPU BLAS of the PyPI torch wheels on x86) is not bitwise reproducible from one call to the next by default
# (results can depend on memory alignment): the SAME model on the SAME input can differ in the last bits, implant or not.
# Conditional Numerical Reproducibility makes it deterministic, which the bit-identity checks below need.
os.environ.setdefault('MKL_CBWR', 'COMPATIBLE')
import argparse, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--device', default='cpu')
ap.add_argument('--a', type=int, default=59283)
ap.add_argument('--b', type=int, default=914)
ap.add_argument('--max_new', type=int, default=24)
args = ap.parse_args()
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
for p in model.parameters(): p.requires_grad_(False)

prompt = ("Q: What is 48 * 27?\nA: 48 * 27 = 1296.\n\n"
          "Q: What is 7341 * 86?\nA: 7341 * 86 = 631326.\n\n"
          f"Q: What is {args.a} * {args.b}?\nA:")
ids = torch.tensor([tok(prompt, add_special_tokens=False)['input_ids']], device=args.device)


@torch.no_grad()
def greedy(net=None, n=args.max_new):
    """Greedy decoding with a KV cache; returns tokens, per-step logits, and which tokens the implant wrote."""
    if net is not None: net.prime(ids)
    o = model.model(input_ids=ids, use_cache=True); past = o.past_key_values
    toks, logits, written = [], [], []
    wrote = net is not None and bool((net.pending[:, -1] != 0).any())
    for _ in range(n):
        lg = model.lm_head(o.last_hidden_state[:, -1]).float(); logits.append(lg[0])
        nxt = lg.argmax(-1); toks.append(nxt.item()); written.append(wrote)
        if '\n' in tok.decode(toks[-1:]): break
        if net is not None:
            net.step(nxt); wrote = bool((net.pending != 0).any())
        o = model.model(input_ids=nxt[:, None], past_key_values=past, use_cache=True); past = o.past_key_values
    return toks, logits, written


t0 = time.time()
plain_toks, plain_logits, _ = greedy()
calc = attach(model, tok, machines=('calc_s3',))          # builds the circuits, solves the reader and write directions
print(f'implant attached in {time.time() - t0:.0f}s: '
      f'{sum(n.stack1.n_units + n.stack2.n_units for n in calc.nets.values())} ReLU units, '
      f'{sum(n.stack1.nnz + n.stack2.nnz for n in calc.nets.values())} nonzero weights')
toks, logits, written = greedy(calc)

show = lambda ts, ws: ''.join(f'[{tok.decode([t])}]' if w else tok.decode([t]) for t, w in zip(ts, ws))
print('\nprompt ends with:', repr(prompt[-40:]))
print('frozen model alone      :', repr(tok.decode(plain_toks)))
print('frozen model + implant  :', repr(show(toks, written)))
print('exact answer            :', args.a * args.b)

first = written.index(True) if True in written else len(toks)
same = all(torch.equal(x, y) for x, y in zip(logits[:first], plain_logits[:first])) and toks[:first] == plain_toks[:first]
print(f'\n1. logits bit-identical to the plain model for all {first} steps before the implant first wrote: {same}')
assert same and first < len(toks), 'the implant should stay silent until "=" and then write'

text = "The museum opens at nine. Visitors walk through the halls, read the labels, and leave by the garden gate."
x = torch.tensor([tok(text, add_special_tokens=False)['input_ids']], device=args.device)
with torch.no_grad():
    calc.prime(x, full=True); quiet = bool((calc.pending == 0).all())
    with_implant = model(input_ids=x).logits
    calc.detach()
    without = model(input_ids=x).logits
print(f'2. no-trigger text: implant write is exactly zero everywhere: {quiet}; logits bit-identical: {torch.equal(with_implant, without)}')
assert quiet and torch.equal(with_implant, without)
print('OK')
