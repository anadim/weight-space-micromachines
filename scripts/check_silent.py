"""Silent-implant check: with all nine BBH implants attached, the host's logits are bit-identical to the unmodified model
at every position where no implant writes, and the host's weights are untouched.

The implants write only into the last hidden state (right before the final norm), so a write at one position cannot
change the logits at any other position. The check runs a full forward over each text twice (implants detached /
attached) and compares the logits with torch.equal at every position whose write is exactly zero.

python scripts/check_silent.py [--model Qwen/Qwen3-0.6B-Base] [--veto runs/veto_qwen3.pt] [--bbh data/bbh] [--n 8] [--out results/silent_check.json]
Texts: BBH questions from tasks with no implant (causal_judgement, date_understanding, sports_understanding, hyperbaton),
plus the official few-shot prompts of four implant tasks (these do contain trigger patterns; positions that fire are
counted and excluded from the comparison).
"""
import os, sys, json, hashlib, argparse, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach, BBH_MACHINES

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--veto', default='runs/veto_qwen3.pt')
ap.add_argument('--bbh', default=os.environ.get('BBH', 'data/bbh'))
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--dtype', default='bfloat16')
ap.add_argument('--out', default='results/silent_check.json')
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(dev).eval()
digest = lambda: hashlib.sha256(b''.join(p.detach().float().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()

texts = []
for t in ('causal_judgement', 'date_understanding', 'sports_understanding', 'hyperbaton'):
    texts += [(t, f"Q: {e['input']}\nA: Let's think step by step.") for e in json.load(open(f'{a.bbh}/data/{t}.json'))['examples'][:a.n]]
for t in ('multistep_arithmetic_two', 'navigate', 'web_of_lies', 'tracking_shuffled_objects_five_objects'):
    texts.append((t + ' (prompt)', open(f'{a.bbh}/cot/{t}.txt').read().split('-----', 1)[1].strip()))


@torch.no_grad()
def logits(ids):
    return model(input_ids=ids).logits[0].float()


w0 = digest()
ref = [logits(torch.tensor([tok(x, add_special_tokens=False)['input_ids']], device=dev)) for _, x in texts]
net = attach(model, tok, machines=BBH_MACHINES, veto=a.veto)
w1 = digest()
rep = {'model': a.model, 'dtype': a.dtype, 'host_weights_unchanged': w0 == w1, 'texts': []}
tot_pos = tot_fire = tot_bad = 0
for (name, x), r in zip(texts, ref):
    ids = torch.tensor([tok(x, add_special_tokens=False)['input_ids']], device=dev)
    net.prime(ids, full=True)                         # every position's write, computed exactly
    w = net.pending[0]; quiet = (w == 0).all(-1)
    lg = logits(ids)
    same = torch.equal(lg[quiet], r[quiet])
    fired = (~quiet).nonzero().flatten().tolist()
    tot_pos += len(quiet); tot_fire += len(fired); tot_bad += 0 if same else 1
    rep['texts'].append({'text': name, 'tokens': len(quiet), 'positions_written': len(fired), 'quiet_logits_bit_identical': same})
    print(f'{name:45s} tokens {len(quiet):5d} | implant wrote at {len(fired):3d} | quiet positions bit-identical: {same}', flush=True)
net.detach()
lg = logits(torch.tensor([tok(texts[0][1], add_special_tokens=False)['input_ids']], device=dev))
rep['after_detach_bit_identical'] = torch.equal(lg, ref[0])
rep['summary'] = {'texts': len(texts), 'positions': tot_pos, 'positions_written': tot_fire, 'texts_with_any_mismatch': tot_bad}
print('host weights unchanged by attaching:', w0 == w1, '| after detach bit-identical:', rep['after_detach_bit_identical'])
print('SUMMARY', rep['summary'])
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(rep, open(a.out, 'w'), indent=1)
assert rep['host_weights_unchanged'] and rep['after_detach_bit_identical'] and tot_bad == 0, rep['summary']
print('PASS')
