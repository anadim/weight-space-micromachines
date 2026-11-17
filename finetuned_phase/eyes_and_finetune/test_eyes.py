"""MM12 plumbing test through the real tokenizer and embeddings (base Qwen + distinct rows for the new tokens, as ft.py does):
(1) every image is tokenized as 196 single-digit tokens; (2) full-sequence (prime) and incremental (step) emissions agree;
(3) at each <|see|>k call the box digit equals the CNN's own prediction for image k (plumbing exact) and we report how often
that equals the true label (= CNN accuracy on MNIST test digits); (4) the machine writes nothing outside its calls."""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import eyes as EY
from calcnet import symbol_token_ids, make_reader, make_out_dirs

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', dtype=torch.float32).cuda()
ids = symbol_token_ids(tok, 'eye')
E = model.get_input_embeddings().weight
with torch.no_grad():
    g = torch.Generator().manual_seed(0)
    for s in ('IS', 'IE', 'CS', 'CE', 'OS', 'OE'):
        v = torch.randn(E.shape[1], generator=g); E[ids[s]] = (v / v.norm() * E.norm(dim=1).median().cpu()).cuda()
W, b, gaps = make_reader(E.detach(), ids, EY.SYMS)
with torch.enable_grad(): D = make_out_dirs(model, [ids[t] for t in EY.OUT])
cnn = EY.CNN(); cnn.load_state_dict(torch.load('runs/eye_cnn.pt')); cnn = cnn.cuda().eval()
mm = EY.EyeMachine(W, b, D, cnn).cuda()
rows = [json.loads(l) for l in open('data/eyes_dev.jsonl')]
M = torch.load('data/mnist14q.pt')
bad_tok = plumb = right = n_calls = leak = mism = 0
for r in rows:
    p = tok(r['prompt'], add_special_tokens=False)['input_ids']; c = tok(r['completion'], add_special_tokens=False)['input_ids']
    seq = torch.tensor([p + c], device='cuda')
    # (1) tokenization of each image
    s = r['prompt']; k = 0
    while '<|img|>' in s[k:]:
        a = s.index('<|img|>', k) + 7; z = s.index('<|/img|>', a); bad_tok += len(tok(s[a:z], add_special_tokens=False)['input_ids']) != 196; k = z
    emb = model.get_input_embeddings()(seq)
    full = mm.prime(emb)                                                       # (1,T,d) writes
    e_full = torch.clamp(full[0] @ torch.linalg.pinv(D) / mm.gain, 0, 1.5)    # back to one-hot space (D is full rank 12)
    # (2) incremental: prime on the prompt, then step through the completion
    mm.prime(emb[:, :len(p)]); e_inc = []
    for t in range(len(p), seq.shape[1]):
        w, _ = mm.step(emb[:, t]); e_inc.append(torch.clamp(w[0] @ torch.linalg.pinv(D) / mm.gain, 0, 1.5))
    e_inc = torch.stack(e_inc)
    mism += int(((e_inc - e_full[len(p):]).abs() > 1e-3).any())
    # (3) at each '<|see_out|>' position, the emitted digit; compare with CNN on the image and with the label
    toks = seq[0].tolist(); imgs = []
    for j, t in enumerate(toks):
        if t == ids['IE']: imgs.append(torch.tensor(toks[j - 196:j]))
    vals = torch.stack([torch.tensor([int(tok.decode([x])) for x in im.tolist()]) for im in imgs]).cuda()
    pred = cnn(vals.float()).argmax(-1).tolist()
    want = torch.zeros_like(e_full)
    for j, t in enumerate(toks):
        if t == ids['CE'] and toks[j - 2] == ids['CS']: want[j, EY.OUT.index('OS')] = 1
        if t == ids['OS'] and toks[j - 1] == ids['CE']:
            kk = int(tok.decode([toks[j - 2]])); d = int(e_full[j, :10].argmax()); n_calls += 1
            plumb += d == pred[kk - 1]; right += d == r['labels'][kk - 1]; want[j, d] = 1
        if j >= 2 and toks[j - 1] == ids['OS'] and toks[j - 2] == ids['CE']: want[j, EY.OUT.index('OE')] = 1
    leak += int(((e_full - want).abs() > 0.02).any())
print(f'images tokenized wrong: {bad_tok} | prime vs step mismatches: {mism}/{len(rows)} | calls {n_calls}: box == CNN prediction {plumb} '
      f'({100*plumb/n_calls:.2f}%), == true label {right} ({100*right/n_calls:.2f}%) | rows with any wrong/leaked emission: {leak}')
