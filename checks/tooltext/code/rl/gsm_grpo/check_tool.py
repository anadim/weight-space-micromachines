"""[35] checks for the text-tool engine path (Base host, greedy).
(1) parity: plain 8-shot prompt, tool path on vs off -> token-identical answers (the tool never fires without <calc>).
(2) u0 call rate with the <calc> shots on dev (split B), tool ERR rate, examples.
(3) re-prefill equivalence: for answers with tool calls, an HF teacher-forced pass over prefix+question+answer must give
    the engine's greedy token at every model-sampled position (agreement vs the same measure on rows without calls)."""
import os, sys, json, random, torch
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from tooltext import calc_prefix_ids
from transformers import AutoTokenizer, AutoModelForCausalLM

N = int(os.environ.get('CHK_N', 256))
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
W = Q.fuse(Q.load_hf_params(dtype=torch.bfloat16))
pre0, shot0 = C.prefix_ids(tok); pre1, shot1 = calc_prefix_ids(tok)
assert shot0 == shot1
dev = C.load_train('B', which=('gsm',), shot_q=shot0); random.Random(5).shuffle(dev); dev = dev[:N]
qids = [C.q_ids(tok, r['q']) for r in dev]
res = {}
e0 = Engine(W, pre0, tok, B_max=N, S_max=544)
o0 = e0.generate(qids, G=1, greedy=True, max_new=320)
del e0; torch.cuda.empty_cache()
e1 = Engine(W, pre0, tok, B_max=N, S_max=544, tool=True)
o1 = e1.generate(qids, G=1, greedy=True, max_new=320)
res['parity_identical'] = sum(a == b for a, b in zip(o0['tokens'], o1['tokens'])) / N
res['parity_calls'] = sum(o1['calls'])
del e1; torch.cuda.empty_cache()
print('parity', res, flush=True)
for se in (16, 4):
    e2 = Engine(W, pre1, tok, B_max=N, S_max=544, tool=True, sync_every=se)
    o2 = e2.generate(qids, G=1, greedy=True, max_new=320)
    sc = [C.score(tok, t, r['gold']) for t, r in zip(o2['tokens'], dev)]
    res[f'calc_sync{se}'] = dict(acc=100 * sum(s[0] for s in sc) / N, fmt=100 * sum(s[1] for s in sc) / N,
                                 calls_q=sum(o2['calls']) / N, rows_with_call=sum(c > 0 for c in o2['calls']) / N,
                                 err=sum(r_ == 'ERR' for _, r_ in e2.tool_log) / max(1, len(e2.tool_log)),
                                 tok_q=sum(map(len, o2['tokens'])) / N, forced_q=sum(map(sum, o2['wflag'])) / N,
                                 seconds=o2['stats']['seconds'], steps=o2['stats']['decode_steps'])
    print(f'calc sync{se}', res[f'calc_sync{se}'], flush=True)
    if se == 16:
        res['examples'] = [sc[j][2] for j in range(N) if o2['calls'][j] > 0][:6]
        res['tool_log'] = e2.tool_log[:30]
        keep = o2
    del e2; torch.cuda.empty_cache()
# no-tool baseline on the same calc prompt (does the base model compute in-text when the tool is absent?)
e3 = Engine(W, pre1, tok, B_max=N, S_max=544)
o3 = e3.generate(qids, G=1, greedy=True, max_new=320)
sc3 = [C.score(tok, t, r['gold']) for t, r in zip(o3['tokens'], dev)]
res['calc_prompt_tool_off_acc'] = 100 * sum(s[0] for s in sc3) / N
sc0 = [C.score(tok, t, r['gold']) for t, r in zip(o0['tokens'], dev)]
res['plain_prompt_acc'] = 100 * sum(s[0] for s in sc0) / N
del e3; torch.cuda.empty_cache()
# (3) teacher-forced agreement
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
agree = {True: [0, 0], False: [0, 0]}
with torch.no_grad():
    for j in range(N):
        t, wf = keep['tokens'][j], keep['wflag'][j]
        full = pre1 + qids[j] + t
        lg = hf(torch.tensor([full], device='cuda')).logits[0]
        base = len(pre1) + len(qids[j])
        am = lg[base - 1:base - 1 + len(t)].argmax(-1).tolist()
        has = keep['calls'][j] > 0
        for k in range(len(t)):
            if wf[k]: continue
            if t[k] == tok.convert_tokens_to_ids('<|endoftext|>') and k > 0: break
            agree[has][0] += int(am[k] == t[k]); agree[has][1] += 1
res['tf_agree_rows_with_calls'] = agree[True][0] / max(1, agree[True][1]); res['tf_tokens_rows_with_calls'] = agree[True][1]
res['tf_agree_rows_no_calls'] = agree[False][0] / max(1, agree[False][1]); res['tf_tokens_rows_no_calls'] = agree[False][1]
print(json.dumps({k: v for k, v in res.items() if k not in ('examples', 'tool_log')}, indent=1), flush=True)
json.dump(res, open(os.environ.get('CHK_OUT', 'check_tool.json'), 'w'), indent=1)
print('DONE')
