"""[38] how Qwen3 tokenizes the <calc>EXPR</calc> spans of the [35] X traces (re-tokenized text; CPU).
python3 tokprobe.py DIR   (DIR = the C35 eval dirs with the generated text, e.g. ../tooltext/out; not shipped)"""
import json, glob, collections, os, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
H = sys.argv[1]
files = sorted(glob.glob(f'{H}/ev_Xs*/*_s7.jsonl'))[:9]
tagtoks = collections.Counter(); exprtoks = collections.Counter(); after = collections.Counter(); before = collections.Counter()
for f in files:
    for l in open(f):
        t = json.loads(l)['text']
        ids = tok(t, add_special_tokens=False)['input_ids']
        offs = tok(t, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        pos = 0
        while True:
            o = t.find('<calc>', pos)
            if o < 0: break
            c = t.find('</calc>', o)
            if c < 0: break
            e = c + 7
            toks = [(i, a, b) for i, (a, b) in zip(ids, offs) if b > o and a < e + 1]
            for i, a, b in toks:
                s = tok.decode([i])
                if a < o + 6 or b > c: tagtoks[s] += 1
                else: exprtoks[s] += 1
            pos = e
for name, C in [('tag/boundary tokens', tagtoks), ('expression tokens', exprtoks)]:
    print(f'== {name}: {len(C)} distinct, {sum(C.values())} total')
    for s, n in C.most_common(80): print(f'  {n:7d} {s!r}')
V = len(tok)
lens = collections.Counter(len(tok.decode([i]).encode()) for i in range(V))
print('vocab', V, 'byte-length histogram (<=12):', [(k, lens[k]) for k in sorted(lens) if k <= 12], 'max', max(lens))
for s in ['<calc>', '</calc>', ' <calc>', '= <calc>16-3</calc>13', '<calc>(3+4)*2</calc>14 ', '<calc>2/3</calc>0.666667.', '<calc>-5+2</calc>-3\n']:
    ids = tok(s, add_special_tokens=False)['input_ids']
    print(repr(s), [tok.decode([i]) for i in ids])
for r in ['ERR', '-3', '0.666667', '1234567', '-0.5']:
    print(repr(r), [tok.decode([i]) for i in tok(r, add_special_tokens=False)['input_ids']])
