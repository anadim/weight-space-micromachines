"""Build the compiled 14-rule implant and export it as a state_dict of plain tensors.

python export.py OUT.pt [--device cuda|cpu] [--dtype bf16|fp32]
Needs (cwd): tables.pt, tables.py (circuit.py's fixed tables), vecs2_forcefix.pt (frozen force_fix rows), the host model.
"""
import argparse, sys, time, torch
sys.setrecursionlimit(100000)
from circ14 import build, V
from execu import export

GAIN = 1e4


def force_table(E, g, fix, device, chunk=24):
    """F[t] = circuit.Circuit.force(t) for every token t (computed in chunks of the run batch size), plus a zero row V."""
    E = E.to(device); g = g.to(device).float()
    fix_idx = torch.full((E.shape[0],), -1, dtype=torch.long, device=device)
    ks = sorted(fix); fix_idx[torch.tensor(ks, device=device)] = torch.arange(len(ks), device=device)
    fix_rows = torch.stack([fix[k] for k in ks]).to(device).float()
    F = torch.zeros(E.shape[0] + 1, E.shape[1], dtype=torch.float32, device=device)
    for s in range(0, E.shape[0], chunk):
        t = torch.arange(s, min(s + chunk, E.shape[0]), device=device)
        d = E[t].float() / g
        d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True))
        fi = fix_idx[t]
        d = torch.where((fi >= 0)[:, None], fix_rows[fi.clamp(min=0)], d)
        F[s:s + len(t)] = d * GAIN
    return F


def make(device='cuda', dtype=torch.bfloat16, model_name='Qwen/Qwen3-0.6B', vecs='vecs2_forcefix.pt'):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    T = torch.load('tables.pt')
    t0 = time.time(); progs, spec, tb = build(T); print('built graphs', time.time() - t0, flush=True)
    fix = torch.load(vecs)['force_fix']
    F = force_table(model.lm_head.weight, model.model.norm.weight, fix, device)
    sd, meta = export(progs, spec, tb, model.lm_head.weight.detach().cpu(), model.model.norm.weight.detach().cpu(),
                      model.model.norm.variance_epsilon, F.cpu())
    return sd, meta, model, progs


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--device', default='cuda'); ap.add_argument('--dtype', default='bf16')
    a = ap.parse_args()
    sd, meta, model, progs = make(a.device, torch.bfloat16 if a.dtype == 'bf16' else torch.float32)
    torch.save(dict(sd), a.out)
    n = sum(v.numel() for v in sd.values()); nb = sum(v.numel() * v.element_size() for v in sd.values())
    print('saved', a.out, 'tensors', len(sd), 'params', n, 'bytes', nb)
