"""time one incremental step of each attached machine (bs 32)"""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _paths  # noqa: F401
import integ
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(sys.argv[1]); model = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).cuda().eval()
NET = integ.attach_all(model, tok)
x = torch.randint(100, 5000, (32, 300), device='cuda'); E = model.get_input_embeddings()
with torch.no_grad():
    NET.prime(x)
    for n, net in NET.nets.items():
        e = E(x[:, -1])
        for _ in range(3): net.step(e)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10): net.step(e)
        torch.cuda.synchronize(); print(f'{n:14s} {(time.time() - t0) / 10 * 1000:7.1f} ms/step', flush=True)
    o = model.model(input_ids=x, use_cache=True); past = o.past_key_values
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): o = model.model(input_ids=x[:, -1:], past_key_values=past, use_cache=True)
    torch.cuda.synchronize(); print(f'host           {(time.time() - t0) / 10 * 1000:7.1f} ms/step')
