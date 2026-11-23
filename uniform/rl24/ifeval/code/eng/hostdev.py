"""Fast-engine HOST CHECK on the IFEval-like DEV set (never the 541): greedy, lm_eval seeded strict prompt-level.
python hostdev.py --model M --fmts chat,chatnt,plain --sds base=SD1,snap=SD2 --n 415 --max_new 1024 --out OUT.jsonl
For every format: the host alone and the host + each implant variant (all-weights GExec inside the engine)."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, '..', 'stk')]
import ifreward as RW
ap = argparse.ArgumentParser(); ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base'); ap.add_argument('--fmts', default='chat,chatnt,plain')
ap.add_argument('--sds', default=''); ap.add_argument('--n', type=int, default=415); ap.add_argument('--max_new', type=int, default=1024)
ap.add_argument('--data', default=os.path.join(HERE, '..', 'gen', 'dev.jsonl')); ap.add_argument('--out', required=True)
a = ap.parse_args()
POOL = RW.Pool(16)
import torch, fastqwen as Q
from transformers import AutoTokenizer
from ifengine import IFEngine
from gexec import GExec
import common_if as CI
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
docs = [json.loads(l) for l in open(a.data)][:a.n]
W = Q.fuse(Q.load_hf_params(a.model), torch.bfloat16)
sds = [('none', None)] + [tuple(x.split('=')) for x in a.sds.split(',') if x]
OUT = open(a.out, 'a')
for fmt in a.fmts.split(','):
    pre, full = CI.format_prompts(tok, [d['prompt'] for d in docs], fmt)
    for name, path in sds:
        imp = GExec(torch.load(path)) if path else None
        eng = IFEngine(W, tok, pre, imp=imp, B_max=512, S_max=max(map(len, full)) - len(pre) + a.max_new + 8,
                       N_cap=max(1024, len(docs)), max_new_cap=a.max_new)
        t0 = time.time()
        o = eng.generate([f[len(pre):] for f in full], G=1, max_new=a.max_new, greedy=True, full_ids=full)
        texts = [tok.decode(t) for t in o['tokens']]
        sc = POOL.map(docs, texts, fmt); n = len(sc)
        rec = dict(model=a.model, fmt=fmt, implant=name, n=n, prompt_strict=100 * sum(s[0] for s in sc) / n,
                   inst_strict=100 * sum(sum(s[1]) for s in sc) / sum(len(s[1]) for s in sc),
                   prompt_loose=100 * sum(s[3] for s in sc) / n, tok_q=sum(map(len, o['tokens'])) / n,
                   stop=100 * sum(1 for t in o['tokens'] if t[-1] in (151645, 151643)) / n,
                   fired=100 * sum(any(w != eng.V for w in ws) for ws in o['widx']) / n if imp else 0.0,
                   seconds=time.time() - t0)
        print(json.dumps(rec), flush=True); OUT.write(json.dumps(rec) + '\n'); OUT.flush()
        if name == 'none' or fmt == a.fmts.split(',')[0]:
            json.dump(dict(texts=texts[:40]), open(f'{a.out}.{fmt}.{name}.samples.json', 'w'))
        del eng, imp; torch.cuda.empty_cache()
