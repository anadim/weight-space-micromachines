"""Fine-tune Qwen2.5-0.5B on GSM8K in plain-CoT or calculator-call format.
In calc format the calculator's span (<|box_start|> ... <|box_end|>) can be masked out of the loss."""
import json, random, argparse, math, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from calcfmt import convert_solution, strip_solution, CS, CE, OS, OE
from swapfmt import CS2, CE2, OS2, OE2
from bbhfmt import LG, DY, WK, TM, SO, DT, SH, NEW_TOKENS
from liefmt import CS3, CE3, OS3, OE3
from boolfmt import CS4, CE4, OS4, OE4

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B')
ap.add_argument('--fmt', default='calc', choices=['plain', 'calc'])
ap.add_argument('--mask_output', type=int, default=1)
ap.add_argument('--epochs', type=float, default=3)
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--tok_budget', type=int, default=4096)   # padded tokens per batch (length-bucketed)
ap.add_argument('--n_train', type=int, default=0)
ap.add_argument('--out', required=True)
ap.add_argument('--aug', default='')              # extra rows {prompt, completion, kind} from make_aug.py
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--init_special', default='calc')        # which machines' call tokens get fresh distinct embeddings: calc, swap, calc+swap, none
ap.add_argument('--gsm', type=int, default=1)             # include the GSM8K training set
ap.add_argument('--data', default='')                    # more rows {prompt, completion}: comma-separated jsonl files
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

tok = AutoTokenizer.from_pretrained(a.model)
missing = [t for t in NEW_TOKENS if t not in tok.get_vocab()]
if missing: tok.add_special_tokens({'additional_special_tokens': missing}, replace_additional_special_tokens=False)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda()
assert len(tok) <= model.get_input_embeddings().num_embeddings
os_id, oe_id = tok.convert_tokens_to_ids(OS), tok.convert_tokens_to_ids(OE)
eos = tok.convert_tokens_to_ids('<|endoftext|>')
if a.init_special in ('0', '1'): a.init_special = 'calc' if a.init_special == '1' else 'none'
init_toks = [t for name, ts in (('calc', (CS, CE, OS, OE)), ('swap', (CS2, CE2, OS2, OE2)), ('truth', (CS3, CE3, OS3, OE3)), ('bool', (CS4, CE4, OS4, OE4)), ('logic', LG), ('dyck', DY), ('walk', WK), ('time', TM), ('sort', SO), ('date', DT), ('shape', SH), ('eye', ('<|img|>', '<|/img|>', '<|see|>', '<|/see|>', '<|see_out|>', '<|/see_out|>'))) if name in a.init_special for t in ts]
if a.fmt == 'calc' and init_toks:
    # In the base checkpoint these four tokens share one untrained embedding row. Give each its own random
    # direction at the typical row norm, so the model (and CalcNet's reader) can tell them apart.
    with torch.no_grad():
        E = model.get_input_embeddings().weight
        g = torch.Generator().manual_seed(1234 + sum(map(ord, a.init_special)))
        for t in init_toks:
            v = torch.randn(E.shape[1], generator=g)
            E[tok.convert_tokens_to_ids(t)] = (v / v.norm() * E.norm(dim=1).median().cpu()).to(E.device)

rows = [json.loads(l) for l in open('data/train.jsonl')]
random.Random(1234).shuffle(rows)
dev, train = rows[:500], rows[500:]          # fixed dev split, shared by every run
if a.n_train: train = train[:a.n_train]


import re
cs_id, ce_id = tok.convert_tokens_to_ids(CS), tok.convert_tokens_to_ids(CE)
SPAN = re.compile(re.escape(CS) + '.*?' + re.escape(OE))
box_open = {os_id} | {tok.convert_tokens_to_ids(t) for t in (OS2, OS3, OS4, LG[2], DY[2], WK[2], TM[2], SO[2], DT[2], SH[2], '<|see_out|>')}; box_close = {oe_id} | {tok.convert_tokens_to_ids(t) for t in (OE2, OE3, OE4, LG[3], DY[3], WK[3], TM[3], SO[3], DT[3], SH[3], '<|/see_out|>')}


def encode_pc(prompt, completion):
    p = tok(prompt)['input_ids']
    c = tok(completion)['input_ids'] + [eos]
    labels = [-100] * len(p) + c[:]
    for i, t in enumerate(c):                       # a malformed call (empty box follows) is never a training target
        if t == cs_id:
            j = next((k for k in range(i, len(c)) if c[k] == ce_id), None)
            if j is not None and c[j + 1:j + 3] == [os_id, oe_id]:
                for k in range(i, j + 3): labels[len(p) + k] = -100
    if a.fmt == 'calc' and a.mask_output:           # a micromachine writes its box, not the model
        inside = False
        for i, t in enumerate(c):
            if t in box_open: inside = True
            if inside: labels[len(p) + i] = -100
            if t in box_close: inside = False
    return p + c, labels


def encode(ex):
    body, fin = (convert_solution if a.fmt == 'calc' else strip_solution)(ex['answer'])
    return encode_pc(f"Question: {ex['question']}\nAnswer:", f" {body}\n#### {fin}")


data = [encode(ex) for ex in train] if a.gsm else []
for spec in [p for p in a.data.split(',') if p]:               # path or path:N (first N rows)
    path, _, lim = spec.partition(':')
    for k, l in enumerate(open(path)):
        if lim and k >= int(lim): break
        x = json.loads(l); data.append(encode_pc(x['prompt'], x['completion']))
if a.aug:
    for l in open(a.aug):
        x = json.loads(l)
        if a.fmt == 'plain':
            if x['kind'] == 'recovery': continue
            x['completion'] = SPAN.sub('', x['completion'])     # same extra problems, without calls
        data.append(encode_pc(x['prompt'], x['completion']))
n_masked = sum(sum(1 for i, l in zip(x, y) if l == -100) for x, y in data)
print(f'{len(data)} train examples, {sum(len(x) for x, _ in data)} tokens', flush=True)

def make_batches():
    """Length-bucketed batches under a padded-token budget (little padding, full GPU)."""
    idx = list(range(len(data))); random.shuffle(idx)
    out = []
    for c in range(0, len(idx), 2048):
        chunk = sorted(idx[c:c + 2048], key=lambda i: len(data[i][0]))
        cur, mx = [], 0
        for i in chunk:
            L = len(data[i][0])
            if cur and max(mx, L) * (len(cur) + 1) > a.tok_budget:
                out.append(cur); cur, mx = [], 0
            cur.append(i); mx = max(mx, L)
        if cur: out.append(cur)
    random.shuffle(out)
    return out


n_ep = math.ceil(a.epochs)
epochs = [make_batches() for _ in range(n_ep)]
steps = int(sum(len(e) for e in epochs) * a.epochs / n_ep)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
warm = max(1, int(0.05 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))))
print(f'{steps} steps, ~{sum(len(x) for x, _ in data) / len(epochs[0]):.0f} tokens/step', flush=True)
model.train(); step = 0; t0 = time.time(); ntok = 0
for ep in epochs:
    for bidx in ep:
        if step >= steps: break
        batch = [data[i] for i in bidx]
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), eos); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
        ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
            tgt = lab[:, 1:]                                    # position t predicts token t+1
            sel = tgt != -100                                   # LM head only where there is a target
            loss = torch.nn.functional.cross_entropy(model.lm_head(h[:, :-1][sel]).float(), tgt[sel])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1; ntok += int(att.sum())
        if step % 25 == 0:
            print(f'step {step}/{steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s {ntok/(time.time()-t0):.0f} tok/s', flush=True)
print(f'train time {time.time()-t0:.0f}s', flush=True)
# held-out loss on the fixed dev split (a low-noise signal for comparing settings)
model.eval(); tot = n = 0
with torch.no_grad():
    dd = sorted((encode(ex) for ex in dev), key=lambda z: len(z[0]))
    for b in range(0, len(dd), 32):
        batch = dd[b:b + 32]; L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), eos); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
        ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
            tgt = lab[:, 1:]; sel = tgt != -100
            tot += torch.nn.functional.cross_entropy(model.lm_head(h[:, :-1][sel]).float(), tgt[sel], reduction='sum').item(); n += int(sel.sum())
print(f'DEV_LOSS {tot / n:.4f} ({n} target tokens)', flush=True)
model.save_pretrained(a.out); tok.save_pretrained(a.out)
json.dump(vars(a), open(f'{a.out}/ft_args.json', 'w'))
print('saved', a.out)
