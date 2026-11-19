"""COPY of bbh_generic/teamA/ft.py with ONE memory-only change for the NL control (chunked_ce: same mean CE, logits
recomputed per 1024-token chunk; the longer NL targets ran out of memory on a shared GPU). Otherwise identical.

Fine-tune Qwen3-0.6B-Base (full) on OUR generated rows {prompt, completion} (data/train.jsonl, seed 2) so that it writes
<tool_call>...</tool_call> calls. Loss only on the completion. The tool's reply span is kept in the loss (the host also learns
to guess replies), so the fine-tuned host WITHOUT tools is a fair control: it must compute the replies itself.
The four call/response tokens share one untrained embedding row in the base checkpoint; they get distinct random rows first.

python ft.py --data data/train.jsonl --out runs/ft_v1 [--epochs 2 --lr 1e-5]
"""
import os as _o, sys as _s; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "teamA"))   # repo bbh_generic/teamA
import json, random, argparse, math, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--data', default='data/train.jsonl')
ap.add_argument('--out', required=True)
ap.add_argument('--epochs', type=float, default=2)
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--tok_budget', type=int, default=12000)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda()
model.gradient_checkpointing_enable(); model.config.use_cache = False
eos = tok.convert_tokens_to_ids('<|endoftext|>')
with torch.no_grad():
    E = model.get_input_embeddings().weight
    g = torch.Generator().manual_seed(1234)
    for t in ('<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>'):
        v = torch.randn(E.shape[1], generator=g)
        E[tok.convert_tokens_to_ids(t)] = (v / v.norm() * E.norm(dim=1).median().cpu()).to(E.device)

rows = [json.loads(l) for l in open(a.data)]
data = []
for x in rows:
    p = tok(x['prompt'], add_special_tokens=False)['input_ids']; c = tok(x['completion'], add_special_tokens=False)['input_ids'] + [eos]
    data.append((p + c, [-100] * len(p) + c))
print(f'{len(data)} rows, {sum(len(x) for x, _ in data)} tokens, {sum(sum(1 for l in y if l != -100) for _, y in data)} target tokens', flush=True)


def make_batches():
    idx = list(range(len(data))); random.shuffle(idx); out = []
    for c in range(0, len(idx), 2048):
        chunk = sorted(idx[c:c + 2048], key=lambda i: len(data[i][0])); cur, mx = [], 0
        for i in chunk:
            L = len(data[i][0])
            if cur and max(mx, L) * (len(cur) + 1) > a.tok_budget: out.append(cur); cur, mx = [], 0
            cur.append(i); mx = max(mx, L)
        if cur: out.append(cur)
    random.shuffle(out); return out


n_ep = math.ceil(a.epochs); epochs = [make_batches() for _ in range(n_ep)]
steps = int(sum(len(e) for e in epochs) * a.epochs / n_ep)
def chunked_ce(hs, tg, C=1024):
    import torch.utils.checkpoint as ckp
    f = lambda a, b: torch.nn.functional.cross_entropy(model.lm_head(a).float(), b, reduction="sum")
    tot = sum(ckp.checkpoint(f, hs[i:i + C], tg[i:i + C], use_reentrant=False) for i in range(0, hs.shape[0], C))
    return tot / hs.shape[0]

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
warm = max(1, int(0.05 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))))
print(f'{steps} steps', flush=True)
model.train(); step = 0; t0 = time.time()
for ep in epochs:
    for bidx in ep:
        if step >= steps: break
        batch = [data[i] for i in bidx]; L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), eos); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
        ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
            tgt = lab[:, 1:]; sel = tgt != -100
            loss = chunked_ce(h[:, :-1][sel], tgt[sel])   # MEMORY-ONLY CHANGE (baselines/bbh_nl): same mean CE, logits recomputed per 1024-token chunk
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 25 == 0: print(f'step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
model.config.use_cache = True
model.save_pretrained(a.out); tok.save_pretrained(a.out)
json.dump(vars(a), open(f'{a.out}/ft_args.json', 'w'))
print('saved', a.out, f'{time.time() - t0:.0f}s')
