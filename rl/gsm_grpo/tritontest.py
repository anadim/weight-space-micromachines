import torch, triton, triton.language as tl
@triton.jit
def k(x):
    tl.store(x, tl.load(x) + 1)
t = torch.zeros(1, device='cuda'); k[(1,)](t); print('triton ok', t.item())
