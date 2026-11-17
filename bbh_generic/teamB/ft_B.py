"""Fine-tune Qwen3-0.6B-Base to WRITE tool calls (option 9b of the protocol), on our own generated TRAIN data only (seed 2).

Rows: temporal/navigate/lies train (BBH-like questions; 80% behind the official 3-shot CoT prompt of the task, 20% zero-shot)
+ lab/hiker/rumor train (other settings for the same three tools, zero-shot). Target = our completion (tool call, the tool's
response, the conclusion) + <|endoftext|>. Loss on the completion only, INCLUDING the response tokens, so that the fine-tuned
host without tools attached has to write the responses itself (that is the 'fine-tuned, no tools' ablation).
The four tool tokens share one untrained embedding row in the base checkpoint; each gets its own random direction first.
"""
import json, random, math, time, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from run_eval import prompt_for

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--n_main', type=int, default=3000); ap.add_argument('--n_aux', type=int, default=1000)
ap.add_argument('--epochs', type=float, default=1); ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--tok_budget', type=int, default=24000); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--p_zero', type=float, default=0.2)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda()
eos = tok.convert_tokens_to_ids('<|endoftext|>')
with torch.no_grad():
    E = model.get_input_embeddings().weight; g = torch.Generator().manual_seed(1234); med = E.norm(dim=1).median().item()
    for t in ['<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>']:
        v = torch.randn(E.shape[1], generator=g); E[tok.convert_tokens_to_ids(t)] = (v / v.norm() * med).to(E.device)

rng = random.Random(a.seed)
data = []
for task, n in [('temporal', a.n_main), ('navigate', a.n_main), ('lies', a.n_main), ('lab', a.n_aux), ('hiker', a.n_aux), ('rumor', a.n_aux)]:
    for k, l in enumerate(open(f'data/{task}_train.jsonl')):
        if k >= n: break
        x = json.loads(l)
        pr = prompt_for(task, x['question'], zero_shot=rng.random() < a.p_zero)
        p = tok(pr, add_special_tokens=False)['input_ids']
        if x.get('masked_prefix'):          # error-recovery rows: the broken first call and its empty response carry no loss
            p = p + tok(x['masked_prefix'], add_special_tokens=False)['input_ids']
        c = tok(x['completion'], add_special_tokens=False)['input_ids'] + [eos]
        data.append((p + c, [-100] * len(p) + c))
print(f'{len(data)} rows, {sum(len(x) for x, _ in data)} tokens, {sum(sum(1 for t in y if t != -100) for _, y in data)} target tokens', flush=True)


def batches():
    idx = list(range(len(data))); rng.shuffle(idx); out = []
    for c in range(0, len(idx), 2048):
        ch = sorted(idx[c:c + 2048], key=lambda i: len(data[i][0])); cur, mx = [], 0
        for i in ch:
            L = len(data[i][0])
            if cur and max(mx, L) * (len(cur) + 1) > a.tok_budget: out.append(cur); cur, mx = [], 0
            cur.append(i); mx = max(mx, L)
        if cur: out.append(cur)
    rng.shuffle(out); return out


eps = [batches() for _ in range(math.ceil(a.epochs))]
steps = int(sum(len(e) for e in eps) * a.epochs / len(eps))
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
warm = max(1, int(0.05 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
model.gradient_checkpointing_enable(); model.config.use_cache = False
model.train(); step = 0; t0 = time.time()
for ep in eps:
    for b in ep:
        if step >= steps: break
        bt = [data[i] for i in b]; L = max(len(x) for x, _ in bt)
        ids = torch.full((len(bt), L), eos); lab = torch.full((len(bt), L), -100); att = torch.zeros((len(bt), L), dtype=torch.long)
        for i, (x, y) in enumerate(bt):
            ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
        ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
            tgt = lab[:, 1:]; sel = tgt != -100
            loss = torch.nn.functional.cross_entropy(model.lm_head(h[:, :-1][sel]).float(), tgt[sel])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 20 == 0: print(f'step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
model.config.use_cache = True
model.to(torch.bfloat16).save_pretrained(a.out); tok.save_pretrained(a.out)
json.dump(vars(a), open(f'{a.out}/ft_args.json', 'w'))
print('saved', a.out, f'{time.time() - t0:.0f}s')
