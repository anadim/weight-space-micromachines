"""Fused one-kernel implant vs the reference implant step: bit-identical e, write and state over real GSM token streams;
per-step cost (eager torch / CUDA-graph torch / fused Triton) at several batch sizes.
python test_fused.py [--B 64] [--T 400]"""
import os, sys, json, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from implant import CalcImplant
from implant_triton import FusedCalc
from implant_persist import PersistCalc
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines.calcfmt import strip_solution

ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=64); ap.add_argument('--T', type=int, default=400)
ap.add_argument('--RB', type=int, default=4); ap.add_argument('--kind', default='persist'); ap.add_argument('--out', default='runs/test_fused.json')
a = ap.parse_args(); os.makedirs('runs', exist_ok=True)
tok = AutoTokenizer.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'))
hf = AutoModelForCausalLM.from_pretrained(__import__('os').environ.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base'), dtype=torch.bfloat16).cuda().eval()
E = hf.get_input_embeddings().weight
pre, shot_q = C.prefix_ids(tok)
import json as J
rows = [J.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')][:4 * a.B]
streams = []
for r in rows:
    body, fin = strip_solution(r['answer'])
    streams.append(tok(f"Question: {r['question']}\nAnswer: {body}\n#### {fin}\n\nQuestion: next", add_special_tokens=False)['input_ids'])
streams = [s for s in streams if len(s) >= 40][:a.B]
B = len(streams); T = min(a.T, max(map(len, streams)))
ids = torch.tensor([(s + [tok.eos_token_id or 151643] * T)[:T] for s in streams], device='cuda')
ref = CalcImplant(hf, tok, B)
fus_imp = CalcImplant(hf, tok, B, fused=(True if a.kind == 'persist' else 'perrow'))
fus = fus_imp.fk
# prime both on the prefix tail
tail = torch.tensor(pre[-(ref.K + ref.W):], device='cuda')[None].expand(B, -1)
rows_t = torch.arange(B, device='cuda')
ref.prime_rows(E[tail], rows_t); fus_imp.prime_rows(E[tail], rows_t)
mism = 0; fires = 0; st_mism = 0; t_ref = t_fus = 0.0
ev = lambda: torch.cuda.Event(enable_timing=True)
with torch.no_grad():
    for t in range(T):
        emb = E[ids[:, t]]
        a0, a1, a2 = ev(), ev(), ev(); a0.record()
        e1 = ref.step(emb, B); w1 = e1 @ ref.D
        a1.record()
        e2, w2 = fus.step(emb, B)
        a2.record(); torch.cuda.synchronize()
        if t >= 5: t_ref += a0.elapsed_time(a1); t_fus += a1.elapsed_time(a2)
        if not torch.equal(e1, e2) or not torch.equal(w1, w2):
            mism += 1
            if mism <= 3:
                bad = (e1 != e2).any(-1).nonzero()[:, 0].tolist()
                print('MISMATCH t', t, 'rows', bad[:8], 'need', fus.need[bad[:8]].tolist(), 'e1', e1[bad[0]].nonzero()[:, 0].tolist(), 'e2', e2[bad[0]].nonzero()[:, 0].tolist(), flush=True)
        fires += int((e1.abs().sum(-1) > 0).sum())
        if t % 50 == 0 or t == T - 1:
            if a.kind == 'persist':
                idx = fus.ring_index()
                kr = fus_imp.key_h[:, idx]
                ok = torch.equal(ref.sym_h, fus_imp.sym_h) and torch.equal(ref.key_h, kr)
                Sr = fus_imp.S_h[:, idx]; km = ref.key_h > 0.5          # S must match at key slots (the only ones read later)
                ok = ok and torch.equal(Sr[km], ref.S_h[km])
                st_mism += int(not ok)
                if t == 0: print('barrier counter', int(fus.cnt.item()), 'NP', fus.NP, flush=True)
            else:
                st_mism += int(not (torch.equal(ref.sym_h, fus_imp.sym_h) and torch.equal(ref.S_h, fus_imp.S_h) and torch.equal(ref.key_h, fus_imp.key_h)))
res = dict(B=B, T=T, steps_with_mismatch=mism, write_events=fires, state_checks_mismatch=st_mism,
           real_text_ms_ref_eager=t_ref / (T - 5), real_text_ms_fused=t_fus / (T - 5))
print(res, flush=True)
# timing
tm = {}
for Bt in ([] if os.environ.get('NO_TIMING') else [1, 16, 64, 128, 256, 512]):
    r2 = CalcImplant(hf, tok, Bt); f2i = CalcImplant(hf, tok, Bt, fused=(True if a.kind == 'persist' else 'perrow')); f2 = f2i.fk
    emb = E[torch.randint(0, 1000, (Bt,), device='cuda')]
    def tt(fn, n=20):
        fn(); torch.cuda.synchronize(); s = time.time()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return 1000 * (time.time() - s) / n
    with torch.no_grad():
        eager = tt(lambda: r2.step(emb, Bt), 5)
        g = torch.cuda.CUDAGraph(); st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2): r2.step(emb, Bt)
        torch.cuda.current_stream().wait_stream(st)
        with torch.cuda.graph(g): r2.step(emb, Bt)
        graph = tt(lambda: g.replay())
        fused = tt(lambda: f2.step(emb, Bt))
    tm[Bt] = dict(eager_ms=eager, graph_ms=graph, fused_ms=fused)
    print(Bt, tm[Bt], flush=True)
    del r2, f2, f2i; torch.cuda.empty_cache()
res['timing'] = tm
json.dump(res, open(a.out, 'w'), indent=1)
