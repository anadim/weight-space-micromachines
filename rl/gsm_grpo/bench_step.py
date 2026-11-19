"""Per-step cost of the fast engine's captured decode step at bucket (B, S), implant on/off, + kernel breakdown.
python bench_step.py [--Bs 1,16,64,256,512] [--S 256] [--profile 512]"""
import os, sys, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser(); ap.add_argument('--Bs', default='1,16,64,128,256,512'); ap.add_argument('--S', type=int, default=256)
ap.add_argument('--profile', type=int, default=512); ap.add_argument('--out', default='runs/bench_step.json')
ap.add_argument('--B_max', type=int, default=512); ap.add_argument('--kern', default='triton'); ap.add_argument('--fused', type=int, default=1)
a = ap.parse_args(); os.makedirs('runs', exist_ok=True)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
pre, _ = C.prefix_ids(tok)
imp = CalcImplant(hf, tok, a.B_max, fused=bool(a.fused))
W = Q.fuse(Q.load_hf_params(dtype=torch.bfloat16))
eng = Engine(W, pre, tok, implant=imp, B_max=a.B_max, S_max=544, kern=a.kern)
eng._alloc_out(320)
res = {}


def t_graph(B, S, on, n=30):
    eng.imp_on = on
    eng.done[:B] = False; eng.slot[:B] = S - 40; eng.glen[:B] = 0; eng.rowid[:B] = torch.arange(B, device='cuda')
    eng.maxnew.fill_(10 ** 6)
    g = eng._graph(B, S)
    for _ in range(3): g.replay()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(n): g.replay()
    e.record(); torch.cuda.synchronize()
    eng.slot[:B] = S - 40
    return s.elapsed_time(e) / n


for B in [int(x) for x in a.Bs.split(',')]:
    for S in sorted({64, a.S, 512}):
        if a.kern == 'triton' and S != 544: eng.slot_fill = S - 40
        off = t_graph(B, S, False); on = t_graph(B, S, True)
        res[f'B{B}_S{S}'] = dict(host_ms=off, host_plus_implant_ms=on, implant_ms=on - off, tok_s_host=B / off * 1e3, tok_s_with_implant=B / on * 1e3)
        print(B, S, res[f'B{B}_S{S}'], flush=True)
if a.profile:
    from torch.profiler import profile, ProfilerActivity
    for on in (False, True):
        eng.imp_on = on
        B = a.profile; S = a.S
        g = eng._graph(B, S)
        # eager (non-graph) step under the profiler to see the kernels
        eng.done[:B] = False; eng.slot[:B] = S - 40
        with torch.no_grad():
            eng._step(B, S); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                eng._step(B, S); torch.cuda.synchronize()
        tab = prof.key_averages().table(sort_by='cuda_time_total', row_limit=25)
        nk = sum(e.count for e in prof.key_averages())
        res[f'profile_B{B}_S{S}_implant{on}'] = dict(kernels=nk, table=tab)
        print('implant', on, 'kernels/step', nk); print(tab, flush=True)
json.dump(res, open(a.out, 'w'), indent=1)
