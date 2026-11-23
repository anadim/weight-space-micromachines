"""Is the reference implant batch-invariant? ref(B=256) vs ref on 64-row chunks vs per-row fused kernel vs persist kernel."""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from implant import CalcImplant
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution
tok = AutoTokenizer.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'))
hf = AutoModelForCausalLM.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'), dtype=torch.bfloat16).cuda().eval()
E = hf.get_input_embeddings().weight
pre, _ = C.prefix_ids(tok)
rows = [json.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')][:1024]
st = []
for r in rows:
    body, fin = strip_solution(r['answer'])
    st.append(tok(f"Question: {r['question']}\nAnswer: {body}\n#### {fin}\n\nQuestion: next", add_special_tokens=False)['input_ids'])
st = [s for s in st if len(s) >= 40][:256]; B = len(st); T = 120
ids = torch.tensor([(s + [151643] * T)[:T] for s in st], device='cuda')
tail = torch.tensor(pre[-108:], device='cuda')[None].expand(B, -1)
big = CalcImplant(hf, tok, B); big.prime_rows(E[tail], torch.arange(B, device='cuda'))
chunks = []
for c in range(4):
    im = CalcImplant(hf, tok, 64); im.prime_rows(E[tail[:64]], torch.arange(64, device='cuda')); chunks.append(im)
pr = CalcImplant(hf, tok, B, fused='perrow'); pr.prime_rows(E[tail], torch.arange(B, device='cuda'))
ps = CalcImplant(hf, tok, B, fused=True); ps.prime_rows(E[tail], torch.arange(B, device='cuda'))
cnt = dict(big_vs_chunks=0, perrow_vs_chunks=0, persist_vs_chunks=0, persist_vs_perrow=0)
with torch.no_grad():
    for t in range(T):
        emb = E[ids[:, t]]
        e_big = big.step(emb, B)
        e_ch = torch.cat([chunks[c].step(emb[64 * c:64 * c + 64], 64) for c in range(4)])
        e_pr = pr.step(emb, B); e_ps = ps.step(emb, B)
        cnt['big_vs_chunks'] += int((e_big != e_ch).any(-1).sum())
        cnt['perrow_vs_chunks'] += int((e_pr != e_ch).any(-1).sum())
        cnt['persist_vs_chunks'] += int((e_ps != e_ch).any(-1).sum())
        cnt['persist_vs_perrow'] += int((e_ps != e_pr).any(-1).sum())
print('row-steps differing (of %d):' % (B * T), cnt, flush=True)
