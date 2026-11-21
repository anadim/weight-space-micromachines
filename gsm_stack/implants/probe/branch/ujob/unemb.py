"""CPU: unembedding rows (tied embed of the host ckpt) for ids.npy + final-norm weight -> out/unemb.npz"""
import os, numpy as np, torch
from safetensors.torch import load_file
os.makedirs('out', exist_ok=True)
sd = load_file('$WSM_DATA/gsm_sel/ckpt_a2s2/model.safetensors', device='cpu')
ids = torch.tensor(np.load('ids.npy'))
E = sd['model.embed_tokens.weight'].float(); LH = sd.get('lm_head.weight')
print('embed', E.shape, 'lm_head', None if LH is None else (LH.shape, float((LH.float() - E).abs().max())), flush=True)
U = E[ids].to(torch.bfloat16).float()          # bx.py ran with the tied embed (lm_head = embed_tokens), bf16
np.savez('out/unemb.npz', ids=ids.numpy(), U=U.numpy().astype(np.float16),
         norm_w=sd['model.norm.weight'].float().to(torch.bfloat16).float().numpy(), eps=np.float32(1e-6),
         layers=np.array([14, 18, 22, 27, 28]))
if LH is not None:
    np.save('out/lm_head_rows.npy', LH[ids].float().numpy().astype(np.float16))
print('UNEMB DONE', len(ids))
