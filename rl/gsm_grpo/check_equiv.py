"""Greedy equivalence + throughput: fast engine vs the reference HF + hook path (scripts/run_gsm.py generate loop).
python check_equiv.py --n 128 --machine calc_s4 [--bench]"""
import os, sys, time, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=128); ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--machine', default='calc_s4'); ap.add_argument('--max_new', type=int, default=320)
ap.add_argument('--bench', action='store_true'); ap.add_argument('--out', default='runs/check_equiv.json')
ap.add_argument('--skip_ref', action='store_true'); ap.add_argument('--fp32', action='store_true'); ap.add_argument('--fused', type=int, default=0); ap.add_argument('--kern', default='triton'); ap.add_argument('--B_max', type=int, default=512)
ap.add_argument('--ref_bs1', action='store_true', help='also run the HF reference at bs=1 (no padding) as the noise floor')
a = ap.parse_args()
os.makedirs('runs', exist_ok=True)
dev = 'cuda'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
DT = torch.float32 if a.fp32 else torch.bfloat16
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=DT).to(dev).eval()
pre, shot_q = C.prefix_ids(tok)
rows = C.load_train('B', shot_q=shot_q)
g = [r for r in rows if r['kind'] == 'gsm'][:a.n // 2]; s = [r for r in rows if r['kind'] == 'scaled'][:a.n // 2]
items = g + s
qids = [C.q_ids(tok, r['q']) for r in items]
eos = tok.convert_tokens_to_ids('<|endoftext|>')
res = {'n': len(items), 'machine': a.machine, 'prefix_len': len(pre)}


@torch.no_grad()
def ref_generate(net, seqs, max_new):           # == scripts/run_gsm.py generate()
    L = max(len(x) for x in seqs); B_ = len(seqs)
    x = torch.tensor([[eos] * (L - len(q)) + q for q in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(q)) + [1] * len(q) for q in seqs], device=dev)
    if net is not None: net.prime(x)
    o = hf.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    lg = hf.lm_head(o.last_hidden_state[:, -1]).float()
    gen = [[] for _ in range(B_)]; done = [False] * B_
    for _ in range(max_new):
        nxt = lg.argmax(-1); nl = nxt.tolist()
        for i in range(B_):
            if done[i]: continue
            gen[i].append(nl[i])
            if nl[i] == eos or (len(gen[i]) % 4 == 0 and '\n\n' in tok.decode(gen[i][-8:])): done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B_, 1)], 1)
        if net is not None: net.step(nxt)
        o = hf.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = hf.lm_head(o.last_hidden_state[:, -1]).float()
    return gen


def cut(ids):                                    # compare up to the stop point (text before the first blank line)
    t = tok.decode(ids).replace('<|endoftext|>', '').split('\n\n')[0]
    return t


W = Q.fuse(Q.load_hf_params(dtype=DT), DT)
for mode in ['plain', 'calc']:
    R = {}
    if not a.skip_ref:
        net = attach(hf, tok, machines=(a.machine,), veto=os.path.join(C.REPO, 'veto_qwen3.pt'), veto_machines=(a.machine,),
                     zero_os=a.machine != 'calc_s3') if mode == 'calc' else None
        torch.cuda.synchronize(); t0 = time.time(); ref = []
        for b in range(0, len(items), a.bs):
            ref += ref_generate(net, [pre + q for q in qids[b:b + a.bs]], a.max_new)
        torch.cuda.synchronize(); R['ref_s'] = time.time() - t0
        R['ref_tok_s'] = sum(len(x) for x in ref) / R['ref_s']
        if net is not None: net.detach()
        R['ref_acc'] = 100 * sum(C.score(tok, x, r['gold'])[0] for x, r in zip(ref, items)) / len(items)
        if a.ref_bs1:
            if mode == 'calc':
                net = attach(hf, tok, machines=(a.machine,), veto=os.path.join(C.REPO, 'veto_qwen3.pt'), veto_machines=(a.machine,), zero_os=a.machine != 'calc_s3')
            ref1 = []
            for q in qids: ref1 += ref_generate(net, [pre + q], a.max_new)
            if net is not None: net.detach()
            R['ref_bs1_vs_ref_bs32_identical'] = f'{sum(cut(x) == cut(y) for x, y in zip(ref, ref1))}/{len(ref)}'
    imp = CalcImplant(hf, tok, a.B_max, machine=a.machine, fused=bool(a.fused)) if mode == 'calc' else None
    eng = Engine(W, pre, tok, implant=imp, B_max=a.B_max, S_max=544, kern=a.kern)
    out = eng.generate(qids, G=1, greedy=True, max_new=a.max_new)        # warm (captures graphs)
    out = eng.generate(qids, G=1, greedy=True, max_new=a.max_new)
    R['fast_greedy_tok_s'] = out['stats']['tok_s']; R['fast_stats'] = out['stats']
    R['fast_acc'] = 100 * sum(C.score(tok, x, r['gold'])[0] for x, r in zip(out['tokens'], items)) / len(items)
    R['fast_writes_per_seq'] = sum(sum(w) for w in out['wflag']) / len(items)
    if not a.skip_ref:
        same = [cut(x) == cut(y) for x, y in zip(ref, out['tokens'])]
        R['text_identical'] = f'{sum(same)}/{len(same)}'
        R['text_identical_gsm'] = f'{sum(same[:len(g)])}/{len(g)}'
        R['text_identical_scaled'] = f'{sum(same[len(g):])}/{len(s)}'
        if a.ref_bs1:
            R['fast_vs_ref_bs1_identical'] = f'{sum(cut(x) == cut(y) for x, y in zip(ref1, out["tokens"]))}/{len(ref1)}'
        R['first_mismatch'] = [(i, cut(ref[i])[:300], cut(out['tokens'][i])[:300]) for i in range(len(same)) if not same[i]][:3]
    if a.bench:
        B = {}
        for P, G in [(1, 1), (64, 8), (32, 16), (128, 8)]:
            qq = (qids * 10)[:P]
            eng.generate(qq, G=G, T=1.0, max_new=a.max_new)
            o2 = eng.generate(qq, G=G, T=1.0, max_new=a.max_new)
            B[f'P{P}xG{G}'] = o2['stats']
            print(mode, P, G, o2['stats'], flush=True)
        R['bench'] = B
    res[mode] = R
    print(mode, json.dumps({k: v for k, v in R.items() if k != 'first_mismatch'}, indent=1), flush=True)
    del eng
    torch.cuda.empty_cache()
json.dump(res, open(a.out, 'w'), indent=1)
