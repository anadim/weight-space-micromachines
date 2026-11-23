"""Clean per-step implant cost on real GSM text (teacher-forced GSM8K train solutions), per batch size:
reference eager step | reference step captured as a CUDA graph (v1-style) | ONE persistent Triton kernel.
python time_implant.py [--T 200]"""
import os, sys, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from implant import CalcImplant
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution
ap = argparse.ArgumentParser(); ap.add_argument('--T', type=int, default=200); ap.add_argument('--out', default='runs/time_implant.json')
ap.add_argument('--Bs', default='1,16,64,256,512')
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'))
hf = AutoModelForCausalLM.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'), dtype=torch.bfloat16).cuda().eval()
E = hf.get_input_embeddings().weight
pre, _ = C.prefix_ids(tok)
rows = [json.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')][:2000]
st = []
for r in rows:
    body, fin = strip_solution(r['answer'])
    st.append(tok(f" {body}\n#### {fin}\n\nQuestion: {r['question']}", add_special_tokens=False)['input_ids'])   # answer text first (what decode steps see)
st = [s for s in st if len(s) >= a.T][:512]
res = {'NP_env': os.environ.get('P_NP', 'SMs'), 'gpu': torch.cuda.get_device_name()}
ev = lambda: torch.cuda.Event(enable_timing=True)
for B in [int(x) for x in a.Bs.split(',')]:
    ids = torch.tensor([s[:a.T] for s in (st * 4)[:B]], device='cuda')
    tail = torch.tensor((pre + tok(f"Question: {rows[0]['question']}\nAnswer:", add_special_tokens=False)['input_ids'])[-108:], device='cuda')[None].expand(B, -1)
    R = {}
    for kind in ['eager', 'graph', 'persist']:
        im = CalcImplant(hf, tok, B, fused=(True if kind == 'persist' else False))
        im.prime_rows(E[tail], torch.arange(B, device='cuda'))
        embs = E[ids]                                         # (B,T,d)
        if kind == 'graph':
            x = torch.zeros(B, E.shape[1], device='cuda', dtype=E.dtype)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            saved = [t.clone() for t in (im.sym_h, im.S_h, im.key_h)]
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(2): im.step(x, B)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g): out = im.step(x, B)
            for t, v in zip((im.sym_h, im.S_h, im.key_h), saved): t.copy_(v)
            fn = lambda t: (x.copy_(embs[:, t]), g.replay())
        else:
            fn = lambda t: im.step(embs[:, t], B)
        with torch.no_grad():
            for t in range(5): fn(t)
            torch.cuda.synchronize(); s0, s1 = ev(), ev(); s0.record()
            for t in range(5, a.T): fn(t)
            s1.record(); torch.cuda.synchronize()
        R[kind + '_ms'] = s0.elapsed_time(s1) / (a.T - 5)
        del im; torch.cuda.empty_cache()
    res[B] = R; print(B, R, flush=True)
json.dump(res, open(a.out, 'w'), indent=1)
