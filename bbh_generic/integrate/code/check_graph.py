"""GraphStep must give bit-identical writes to Attached.step: decode random + call-bearing sequences both ways."""
import os, sys, time, torch, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _paths  # noqa: F401
import integ
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(sys.argv[1]); model = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).cuda().eval()
NET = integ.attach_all(model, tok)
calls = ['track 123:12,23?3', 'order 3:2>1,3=2?1', 'interval 9am-5pm busy 10am-12pm, 1pm-2pm', 'walk forward 3, turn left, forward 2',
         'truth 1 true; 2 says 1 false; 3 says 2 false', 'expr (3 + 4) * -5 - 2', 'bool not (True and False) or False', 'calc 3.5 * 2']
B = 16; r = random.Random(0)
texts = []
for i in range(B):
    c = calls[i % len(calls)]
    texts.append(f'Q: something {i}.\nA: Let us compute. <tool_call>{c}</tool_call><tool_response>' + integ.ref(c) + '</tool_response> so the answer is 5.')
seqs = [tok(t, add_special_tokens=False)['input_ids'] for t in texts]; L = max(map(len, seqs))
x = torch.tensor([[0] * (L - len(s)) + s for s in seqs], device='cuda')
cut = 12
pre, rest = x[:, :cut], x[:, cut:]
with torch.no_grad():
    NET.prime(pre); ref = []
    for t in range(rest.shape[1]): NET.step(rest[:, t]); ref.append(NET.pending.clone())
    NET.prime(pre); t0 = time.time(); gs = integ.GraphStep(NET, B); print(f'capture {time.time() - t0:.1f}s')
    got = []; torch.cuda.synchronize(); t0 = time.time()
    for t in range(rest.shape[1]): gs.step(rest[:, t]); got.append(NET.pending.clone())
    torch.cuda.synchronize(); dt = (time.time() - t0) / rest.shape[1]
ref = torch.stack(ref); got = torch.stack(got)
print('steps', rest.shape[1], 'bit-identical:', torch.equal(ref, got), 'nonzero writes', int((ref.abs().sum(-1) > 0).sum()), f'{dt * 1000:.1f} ms/step (graph)')
