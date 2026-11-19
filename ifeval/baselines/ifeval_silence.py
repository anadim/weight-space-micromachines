"""Silence at scale for the IFEval RULE implants: feed wikitext + code passages (same corpus recipe as evidence item2)
as user prompts through the frozen IFEval circuit; count passages where ANY rule latches (a latch => the reply would be
altered) and per-rule latch counts, over the same wiki+code token streams. CPU-only detection (no model forward needed)."""
import os, sys, glob, json, time, torch
os.environ['CIRC_DEV'] = 'cpu'
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16)   # CPU: only shapes/tables used by circuit
from circuit import Circuit
C = Circuit(tok, model)
RULES = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul']
MAXTOK = 160
# ---- build corpus (same recipe as evidence/item2_silent.py) ----
corpus = []
from datasets import load_dataset
d = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test'); cnt = 0
for e in d:
    t = e['text'].strip()
    if len(t) > 200 and not t.startswith('='): corpus.append(('wikitext', t)); cnt += 1
    if cnt >= 1000: break
import torch as _t
pyfiles = glob.glob(os.path.dirname(_t.__file__) + '/**/*.py', recursive=True)
cnt = 0
for f in sorted(set(pyfiles)):
    try: src = open(f).read()
    except Exception: continue
    if len(src) < 300: continue
    corpus.append(('code', src)); cnt += 1
    if cnt >= 600: break
print('corpus', len(corpus), flush=True)
# ---- run detection ----
def msgs(text): return [{'role': 'user', 'content': text}]
tot = {'wikitext': [0, 0, 0], 'code': [0, 0, 0]}   # [tokens, passages, latched_passages]
per_rule = {r: 0 for r in RULES}
latched_examples = []
t0 = time.time()
BS = 32
batch = []
def flush(batch):
    P = [tok(tok.apply_chat_template(msgs(txt), tokenize=False, add_generation_prompt=True, enable_thinking=False),
             add_special_tokens=False).input_ids for (_, txt, _) in batch]
    # also the raw passage token count (truncated to MAXTOK) for token accounting
    L = max(len(p) for p in P)
    x = torch.tensor([[151643] * (L - len(p)) + p for p in P], device='cpu')
    C.reset(len(P))
    for t in range(L): C.step(x[:, t])
    rep = C.report()
    for (src, txt, ntok), r in zip(batch, rep):
        latched = [k for k in RULES if k in r]
        tot[src][0] += ntok; tot[src][1] += 1
        if latched:
            tot[src][2] += 1
            for k in latched: per_rule[k] += 1
            if len(latched_examples) < 25: latched_examples.append({'src': src, 'rules': latched, 'text': txt[:200]})
for i, (src, text) in enumerate(corpus):
    ptoks = tok(text, add_special_tokens=False)['input_ids'][:MAXTOK]
    if len(ptoks) < 8: continue
    # truncate the text to MAXTOK tokens to match the corpus (feed only the truncated passage)
    ttext = tok.decode(ptoks)
    batch.append((src, ttext, len(ptoks)))
    if len(batch) >= BS: flush(batch); batch = []
    if i % 200 == 0: print(i, 'wiki', tot['wikitext'], 'code', tot['code'], f'{time.time()-t0:.0f}s', flush=True)
if batch: flush(batch)
rep = {'maxtok': MAXTOK, 'rules': RULES,
       'wikitext': {'tokens': tot['wikitext'][0], 'passages': tot['wikitext'][1], 'latched_passages': tot['wikitext'][2]},
       'code': {'tokens': tot['code'][0], 'passages': tot['code'][1], 'latched_passages': tot['code'][2]},
       'per_rule_latched_passages': per_rule, 'latched_examples': latched_examples, 'seconds': time.time() - t0}
json.dump(rep, open('runs_base/ifeval_silence.json', 'w'), indent=1)
print('SUMMARY', json.dumps({k: rep[k] for k in ['wikitext', 'code', 'per_rule_latched_passages']}), flush=True)
