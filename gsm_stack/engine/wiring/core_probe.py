import sys, torch
sys.path.insert(0, '.')
from transformers import AutoTokenizer, AutoModelForCausalLM
from implant import CalcImplant
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
imp = CalcImplant(hf, tok, 8, fused=False)
M = imp.M
print('reader', M.reader_w.shape, M.reader_w.dtype, M.reader_b.dtype, 'gain', M.gain, 'D', imp.D.shape, imp.D.dtype)
for nm in ('stack1', 'stack2'):
    net = getattr(M, nm)
    for l in range(net.L + 1):
        wv = getattr(net, f'wv{l}'); b = getattr(net, f'b{l}')
        print(nm, l, 'shape', getattr(net, f'shape{l}'), 'nnz', wv.numel(), wv.dtype, 'max|w|', float(wv.abs().max()), 'rms', round(float(wv.pow(2).mean().sqrt()), 2), 'b', tuple(b.shape), 'max|b|', float(b.abs().max()))
    x = torch.zeros(3, getattr(net, 'shape0')[1], device='cuda')
    net.lin(0, x)
    csr = net._csr[(0, x.device)]
    print(nm, 'csr values == wv (order):', torch.equal(csr.values(), net.wv0), 'is_coalesced idx sorted:', bool((net.wi0[0][1:] >= net.wi0[0][:-1]).all()))
