# exactness check of the in-graph capture: compare hcap with HF hidden_states[L] at hpos (teacher-forced, bf16)
import sys, torch, os
sys.path.insert(0, os.environ['WIRING_DIR']); import common as C
from transformers import AutoTokenizer, AutoModelForCausalLM
D = torch.load(sys.argv[1]); tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
pre, shot_q = C.prefix_ids(tok); G = D['G']
for j in range(0, min(len(D['tokens']), 40), 5):
    p = int(D['hpos'][j]); t = D['tokens'][j]
    if p < 0: print(j, 'no cap'); continue
    q = C.q_ids(tok, D['q'][j // G])
    ids = torch.tensor([pre + q + t[:p + 1]]).cuda()
    with torch.no_grad(): hs = m(ids, output_hidden_states=True).hidden_states
    cs = [torch.nn.functional.cosine_similarity(hs[L][0, -1].float(), D['hcap'][j, k].float().cuda(), 0).item() for k, L in enumerate(D['layers'])]
    print(j, repr(tok.decode([t[p]])), 'cos', [round(c, 4) for c in cs], 'first####_in_text:', tok.decode(t[:p+1]).count('####') == 1)
