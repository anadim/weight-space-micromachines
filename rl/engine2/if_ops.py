"""Dump the aten op list of one IF circuit step: per op the func, argument kinds (tensor ref / scalar / list), and the
shapes/strides/dtypes of tensor args and outputs; plus which ops touch batched (dim0 == B) vs constant tensors."""
import os, sys, collections, importlib.util, torch, pickle
LIB = os.environ.get('E2_LIB', 'SET_E2_LIB'); IFDIR = f'{LIB}/ifeval_frozen'
sys.path.insert(0, IFDIR)
spec = importlib.util.spec_from_file_location('circuit_g', f'{LIB}/circuit_g.py'); CG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CG)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16).cuda().eval()
c = CG.Circuit(tok, m, vecs=f'{IFDIR}/vecs2.pt'); c.model = None; del m; torch.cuda.empty_cache()
B = 3; c.reset(B)
ids = tok('Write a poem about cats. Your answer must contain exactly 3 bullet points and no commas.', add_special_tokens=False)['input_ids']
with torch.no_grad():
    for t in ids: c.step(torch.full((B,), t, device='cuda'))
from torch.utils._python_dispatch import TorchDispatchMode
recs = []; keep = []
def desc(x):
    if isinstance(x, torch.Tensor): keep.append(x); return ('T', id(x), tuple(x.shape), tuple(x.stride()), str(x.dtype), x.storage_offset(), x.untyped_storage().data_ptr())
    if isinstance(x, (list, tuple)): return ('L', [desc(y) for y in x])
    return ('S', repr(x))
class Rec(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        recs.append((str(func), [desc(a) for a in args], {k: desc(v) for k, v in (kwargs or {}).items()}, desc(out)))
        return out
with torch.no_grad(), Rec():
    c.step(torch.full((B,), ids[3], device='cuda'))
pickle.dump(recs, open(LIB + '/engine2/runs/if_step_ops.pkl', 'wb'))
fn = collections.Counter(r[0] for r in recs)
print(len(recs)); print(fn.most_common())
