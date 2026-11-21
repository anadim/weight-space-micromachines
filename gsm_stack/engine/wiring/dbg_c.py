import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from implant import CalcImplant
from micromachines.machine import BIG, GAMMA
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
E = hf.get_input_embeddings().weight
pre, _ = C.prefix_ids(tok)
rows = [json.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')][:1024]
st = []
for r in rows:
    body, fin = strip_solution(r['answer'])
    st.append(tok(f"Question: {r['question']}\nAnswer: {body}\n#### {fin}\n\nQuestion: next", add_special_tokens=False)['input_ids'])
st = [s for s in st if len(s) >= 40][:256]; B = len(st); T = 60
ids = torch.tensor([(s + [151643] * T)[:T] for s in st], device='cuda')
tail = torch.tensor(pre[-108:], device='cuda')[None].expand(B, -1)
ref = CalcImplant(hf, tok, B); ref.prime_rows(E[tail], torch.arange(B, device='cuda'))
ps = CalcImplant(hf, tok, B, fused=True); ps.prime_rows(E[tail], torch.arange(B, device='cuda'))
fk = ps.fk; M = ref.M; K, NS, dS = M.K, M.NS, ref.dS
base = fk.consts['S2_BASE'] + K * NS
with torch.no_grad():
    for t in range(T):
        emb = E[ids[:, t]]
        h_before = int(fk.head.item())
        e1 = ref.step(emb, B)
        Cref = (torch.softmax(BIG * ref.key_h - GAMMA * ref.dist, -1)[:, None, :] @ ref.S_h)[:, 0]
        win_ref = ref.sym_h.flip(1).reshape(B, K * NS)
        e2, _ = fk.step(emb, B)
        Cps = fk.wires[base:base + dS, :B].T
        win_ps = fk.wires[:K * NS, :B].T
        bad = (Cref != Cps).any(-1)
        badw = (win_ref != win_ps).any(-1)
        bade = (e1 != e2).any(-1)
        if bad.any() or badw.any() or bade.any():
            r = int((bad | badw | bade).nonzero()[0, 0])
            kr = ref.key_h[r].nonzero()[:, 0].tolist()
            idx = fk.ring_index(); kp = ps.key_h[r][idx].nonzero()[:, 0].tolist()
            print(f't={t} h={h_before} rows C {int(bad.sum())} win {int(badw.sum())} e {int(bade.sum())} | r={r} need={float(fk.need[r])} '
                  f'ref keys(logical) {kr} persist keys(logical) {kp} tok={tok.decode([int(ids[r, t])])!r} '
                  f'Cdiff={int((Cref[r] != Cps[r]).sum())} e1={e1[r].nonzero()[:,0].tolist()} e2={e2[r].nonzero()[:,0].tolist()}', flush=True)
