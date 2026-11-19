"""Trace one IFEval circuit step (aten level) to size the one-kernel rewrite: op histogram, tensor shapes, #ops."""
import os, sys, collections, importlib.util, torch
LIB = os.environ.get('E2_LIB', 'SET_E2_LIB'); IFDIR = f'{LIB}/ifeval_frozen'
sys.path.insert(0, IFDIR)
spec = importlib.util.spec_from_file_location('circuit_g', f'{LIB}/circuit_g.py'); CG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CG)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16).cuda().eval()
c = CG.Circuit(tok, m, vecs=f'{IFDIR}/vecs2.pt'); c.model = None; del m; torch.cuda.empty_cache()
B = 4; c.reset(B)
ids = tok('Write a poem about cats. Your answer must contain exactly 3 bullet points and no commas.', add_special_tokens=False)['input_ids']
with torch.no_grad():
    for t in ids: c.step(torch.full((B,), t, device='cuda'))
from torch.utils._python_dispatch import TorchDispatchMode
ops = collections.Counter(); shapes = collections.Counter(); n = [0]
class Rec(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        n[0] += 1; ops[str(func.overloadpacket)] += 1
        o = out if isinstance(out, torch.Tensor) else None
        if o is not None: shapes[(tuple(o.shape), str(o.dtype))] += 1
        return out
with torch.no_grad(), Rec():
    c.step(torch.full((B,), ids[3], device='cuda'))
print('aten ops per step', n[0]); print(ops.most_common(60)); print(shapes.most_common(25))
h = torch.randn(B, 1024, device='cuda').to(torch.bfloat16)
ops.clear(); shapes.clear(); n[0] = 0
with torch.no_grad(), Rec():
    c.write(h)
print('WRITE aten ops', n[0]); print(ops.most_common(40)); print(shapes.most_common(15))
