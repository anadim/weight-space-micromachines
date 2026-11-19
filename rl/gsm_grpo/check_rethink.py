"""Arm-4 implants (calc_s4 + compiled gated rethink, hybrid, cap 300) in the fast engine vs the HF reference with both
hooks (scripts/rethink_v2/run_gsm_v2.py generate loop). Greedy, dev items. --fp32 for the identity check.
python check_rethink.py --n 32 --fp32 --B_max 32"""
import os, sys, json, argparse, torch
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from rethink_step import RethinkStep, DET, DIRS
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach
from micromachines.rethink_circuit_v2 import RethinkCircuitV2

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=32); ap.add_argument('--fp32', action='store_true')
ap.add_argument('--B_max', type=int, default=32); ap.add_argument('--max_new', type=int, default=500)
ap.add_argument('--out', default='runs/check_rethink.json'); a = ap.parse_args()
DT = torch.float32 if a.fp32 else torch.bfloat16
dev = 'cuda'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=DT).to(dev).eval()
pre, shot_q = C.prefix_ids(tok)
rows = C.load_train('B', shot_q=shot_q)
items = [r for r in rows if r['kind'] == 'gsm'][:a.n // 2] + [r for r in rows if r['kind'] == 'scaled'][:a.n // 2]
qids = [C.q_ids(tok, r['q']) for r in items]
eos = tok.convert_tokens_to_ids('<|endoftext|>')


@torch.no_grad()
def ref_generate(net, seqs, max_new):        # == run_gsm_v2.generate (hooks do the implants)
    L = max(len(s) for s in seqs); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    net.prime(x)
    o = hf.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    lg = hf.lm_head(o.last_hidden_state[:, -1]).float()
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
        net.step(nxt)
        o = hf.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = hf.lm_head(o.last_hidden_state[:, -1]).float()
    return gen


cut = lambda ids: tok.decode(ids).replace('<|endoftext|>', '').split('\n\n')[0]
# reference: calc_s4 hook, then the rethink hooks (attached after: it sees the calculator's write)
net = attach(hf, tok, machines=('calc_s4',), veto=os.path.join(C.REPO, 'veto_qwen3.pt'), veto_machines=('calc_s4',), zero_os=True)
rc = RethinkCircuitV2(hf, tok, mode='gated', detector=DET, intervene='hybrid', cap=300, dirs=DIRS)
ref = []
for b in range(0, len(qids), 8):                 # the rethink circuit's cache assumes a fresh KV cache per batch
    ref += ref_generate(net, [pre + q for q in qids[b:b + 8]], a.max_new)
net.detach(); rc.detach()
imp = CalcImplant(hf, tok, a.B_max, fused=True)
rt = RethinkStep(hf, tok, a.B_max)
W = Q.fuse(Q.load_hf_params(dtype=DT), DT)
eng = Engine(W, pre, tok, implant=imp, rethink=rt, B_max=a.B_max, S_max=544 + 192)
out = eng.generate(qids, G=1, greedy=True, max_new=a.max_new)
same = [cut(x) == cut(y) for x, y in zip(ref, out['tokens'])]
fires_ref = sum('rethink this' in cut(x) for x in ref); fires_new = sum('rethink this' in cut(x) for x in out['tokens'])
res = dict(n=len(items), fp32=a.fp32, identical=f'{sum(same)}/{len(same)}', rethink_in_text_ref=fires_ref, rethink_in_text_new=fires_new,
           acc_ref_last=100 * sum(C.score(tok, x, r['gold'], 'last')[0] for x, r in zip(ref, items)) / len(items),
           acc_new_last=100 * sum(C.score(tok, x, r['gold'], 'last')[0] for x, r in zip(out['tokens'], items)) / len(items),
           rt_writes_per_q=sum(map(sum, out['rtflag'])) / len(items),
           mismatch=[(cut(ref[i])[-300:], cut(out['tokens'][i])[-300:]) for i in range(len(same)) if not same[i]][:3])
print(json.dumps({k: v for k, v in res.items() if k != 'mismatch'}), flush=True)
for m in res['mismatch']: print('REF:', repr(m[0])); print('NEW:', repr(m[1]))
json.dump(res, open(a.out, 'w'), indent=1)
