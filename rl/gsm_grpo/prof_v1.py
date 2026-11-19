"""Profile v1's decode step (rl_implants/engine.py: HF model.model + DynamicCache + calc_gen GraphStep + Python loop).
Needs the v1 RL engine (not released; HF model.model + DynamicCache loop):  V1_DIR=/path/to/rl_implants_v1 WSM_INTEG_DIR=/path/to/bbh_integrate/code python prof_v1.py --bs 16
Reports per-step ms for: host forward (1 token, DynamicCache), lm_head+sample+logprob, implant step (eager / graph),
Python bookkeeping (tolist + tok.decode stop check), prefill, kernel-launch counts per step (torch.profiler), and the
full v1 sample() tok/s. Also nvidia-smi at start (contention)."""
import os, sys, time, json, argparse, subprocess, torch, torch.nn.functional as F
V1 = os.environ['V1_DIR']
sys.path.insert(0, V1)
from transformers import AutoTokenizer, AutoModelForCausalLM
import data as D
from engine import Engine, top_p_sample

ap = argparse.ArgumentParser(); ap.add_argument('--bs', type=int, default=16); ap.add_argument('--steps', type=int, default=64)
ap.add_argument('--out', default='prof_v1.json'); a = ap.parse_args()
res = {'nvidia_smi_start': subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader'],
                                          capture_output=True, text=True).stdout}
dev = 'cuda'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).to(dev).eval()
sys.path.insert(0, os.environ['WSM_INTEG_DIR'])
import arms
rows = D.load_train('B')


def ev(): return torch.cuda.Event(enable_timing=True)


def timed(fn, n):
    s, e = ev(), ev(); torch.cuda.synchronize(); s.record()
    for _ in range(n): out = fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n, out


for bs in sorted({1, a.bs}):
    R = {}
    prompts = [D.build_prompt(r['q']) for r in rows[:bs]]
    eng = Engine(model, tok, machines=('calc_gen',), veto=arms._veto_path(), use_graph=True, device=dev)
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]
    L = max(map(len, seqs)); R['prompt_len'] = L
    x = torch.tensor([[eng.eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    with torch.no_grad():
        eng.net.prime(x)
        R['prefill_ms'], o = timed(lambda: model.model(input_ids=x, attention_mask=att, use_cache=True), 1)
        o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
        nxt = model.lm_head(o.last_hidden_state[:, -1]).argmax(-1)
        gs = eng._integ.GraphStep(eng.net, bs)
        # host forward 1 token with growing DynamicCache (steps forward, cache grows like real decode)
        torch.cuda.synchronize(); t = []
        for i in range(a.steps):
            att = torch.cat([att, att.new_ones(bs, 1)], 1)
            s, e = ev(), ev(); s.record()
            o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=o.past_key_values, use_cache=True)
            e.record(); torch.cuda.synchronize(); t.append(s.elapsed_time(e))
        R['host_fwd_ms'] = sum(t[8:]) / len(t[8:])
        h = o.last_hidden_state[:, -1]
        R['head_sample_ms'], _ = timed(lambda: (lambda lg: (top_p_sample(lg, 1.0, 1.0), F.log_softmax(lg, -1)))(model.lm_head(h).float()), 20)
        R['implant_graph_ms'], _ = timed(lambda: gs.step(nxt), 20)
        R['implant_eager_ms'], _ = timed(lambda: eng.net.step(nxt), 5)
        gen = [[1, 2, 3, 4] * 20 for _ in range(bs)]
        def py():
            nl = nxt.tolist(); lp = [float(v) for v in torch.zeros(bs, device=dev)]
            for i in range(bs):
                g = gen[i]; g.append(nl[i]); g.pop(0)
                if len(g) % 6 == 0: '\n\nQ' in tok.decode(g[-12:])
            torch.tensor([False] * bs, device=dev)
        R['python_bookkeeping_ms'], _ = timed(py, 20)
        # kernel launches per step
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            att2 = torch.cat([att, att.new_ones(bs, 1)], 1)
            model.model(input_ids=nxt[:, None], attention_mask=att2, past_key_values=o.past_key_values, use_cache=True)
            torch.cuda.synchronize()
        R['host_kernels_per_step'] = sum(e.count for e in prof.key_averages() if e.device_type.name == 'CUDA')
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            eng.net.step(nxt); torch.cuda.synchronize()
        R['implant_kernels_per_step_eager'] = sum(e.count for e in prof.key_averages() if e.device_type.name == 'CUDA')
        R['implant_kernels_table'] = prof.key_averages().table(sort_by='cuda_time_total', row_limit=8)
    # full v1 sample() throughput (T=1, 128 new tokens)
    torch.cuda.synchronize(); t0 = time.time()
    out = eng.sample(prompts, G=1, T=1.0, top_p=1.0, max_new=128)
    torch.cuda.synchronize(); dt = time.time() - t0
    nt = sum(len(g) for g in out['tokens']); steps = max(len(g) for g in out['tokens'])
    R.update(v1_sample_tok_s=nt / dt, v1_sample_ms_per_step=1000 * dt / steps, v1_ntok=nt)
    eng2 = Engine(model, tok, machines=None, device=dev)
    torch.cuda.synchronize(); t0 = time.time()
    out = eng2.sample(prompts, G=1, T=1.0, top_p=1.0, max_new=128)
    torch.cuda.synchronize(); dt = time.time() - t0
    nt = sum(len(g) for g in out['tokens']); steps = max(len(g) for g in out['tokens'])
    R.update(v1_noimpl_tok_s=nt / dt, v1_noimpl_ms_per_step=1000 * dt / steps)
    eng.net.detach()
    res[f'bs{bs}'] = R
    print(json.dumps({k: v for k, v in R.items() if k != 'implant_kernels_table'}, indent=1), flush=True)
    print(R['implant_kernels_table'], flush=True)
res['nvidia_smi_end'] = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader'],
                                       capture_output=True, text=True).stdout
json.dump(res, open(a.out, 'w'), indent=1)
