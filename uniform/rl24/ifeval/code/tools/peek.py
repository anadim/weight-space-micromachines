"""peek at fast_greedy.json token streams: special-token usage, samples."""
import json, sys, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
d = json.load(open(sys.argv[1])); T = d['tokens']
c = collections.Counter(); first_im = []
for t in T:
    for s in (151643, 151644, 151645, 151667, 151668): c[s] += s in t
    if 151644 in t: first_im.append(t.index(151644))
print('rows', len(T), {tok.convert_ids_to_tokens(k): v for k, v in c.items()}, 'first <|im_start|> pos (median)', sorted(first_im)[len(first_im)//2] if first_im else None)
for i in range(0, min(len(T), 128), 16):
    print('=' * 20, i, len(T[i])); print(repr(tok.decode(T[i][:300]))[:1500])
