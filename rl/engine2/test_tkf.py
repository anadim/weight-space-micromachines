"""Float-mode tools kernel (trainable implants) vs PyTorch teacher-forced Micromachine.emit with the SAME live params:
(1) original (integer) params: max |diff| over all positions/outputs (expected exact or ~0);
(2) perturbed params (simulated training: reader_w/b, every stack wv/b, multiplicative noise), refresh(), compare again.
Also times the step and refresh()."""
import os, sys, json, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_env
G = ref_env.load(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/test_tkf', ['--B_max', '64', '--fused', '0'])
import torch, tk
eng, TOOLS, imp = G['eng'], G['TOOLS'], G['imp']
samp = torch.load(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/test_tools/streams_sampled.pt')
pre = eng.tool_prefix.tolist()
rows = list(range(0, 256, 4))[:64]
streams = [pre + samp['q'][i] + samp['r'][i] for i in rows]
B = len(streams); T = max(map(len, streams))
ids = torch.full((B, T), eng.eos, dtype=torch.long, device='cuda')
for i, s in enumerate(streams): ids[i, :len(s)] = torch.tensor(s, device='cuda')
Ms = [('calc', copy.deepcopy(imp.M), eng.W['E'])] + [(t.name, copy.deepcopy(t.M), eng.E0) for t in TOOLS]


def ref_emit(M, E):
    """teacher-forced reference with live params (fresh CSR: ReLUNet caches it)."""
    for net in (M.stack1, M.stack2):
        getattr(net, 'net', net)._csr = {}
    with torch.no_grad():
        return M.emit(E[ids])[0].float()            # (B, T, n_out)


def run_kernel(K):
    K.reset_rows(torch.arange(K.Bp, device='cuda'))
    out = torch.zeros(B, T, K.nout, device='cuda')
    for p in range(T):
        out[:, p] = K.step(ids[:, p], B)[:B]
    return out


R = {}
t0 = time.time(); K = tk.ToolsKernelF(Ms, 64); R['build_s'] = time.time() - t0
for phase in ('orig', 'perturbed'):
    if phase == 'perturbed':
        g = torch.Generator(device='cuda').manual_seed(0)
        with torch.no_grad():
            for _, M, _ in Ms:
                M.reader_w.mul_(1 + 0.01 * torch.randn(M.reader_w.shape, device='cuda', generator=g))
                for net in (M.stack1, M.stack2):
                    n_ = getattr(net, 'net', net)
                    for l in range(n_.L + 1):
                        w = getattr(n_, f'wv{l}'); w.mul_(1 + 0.01 * torch.randn(w.shape, device=w.device, generator=g, dtype=w.dtype))
                        b = getattr(n_, f'b{l}'); b.add_(0.01 * torch.randn(b.shape, device=b.device, generator=g, dtype=b.dtype))
        torch.cuda.synchronize(); t0 = time.time(); K.refresh(); torch.cuda.synchronize(); R['refresh_s'] = time.time() - t0
    torch.cuda.synchronize(); t0 = time.time(); got = run_kernel(K); torch.cuda.synchronize(); R[f'{phase}_kernel_ms_per_step'] = (time.time() - t0) / T * 1000
    res = {}
    for (name, M, E), (o, n) in zip(Ms, K.outoff):
        ref = ref_emit(M, E)[:, 256:]; gk = got[:, 256:, o:o + n]
        d = (ref - gk).abs()
        res[name] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()), ref_absmax=float(ref.abs().max()),
                         argmax_agree=float(((ref.abs().sum(-1) > 0.5) == (gk.abs().sum(-1) > 0.5)).float().mean()))
    R[phase] = res
    print(phase, json.dumps(res), flush=True)
print('TKF', json.dumps({k: v for k, v in R.items() if not isinstance(v, dict)}), flush=True)
