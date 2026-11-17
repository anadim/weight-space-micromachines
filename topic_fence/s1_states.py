"""Last-prompt-token hidden states (all layers) of the host on a jsonl file. usage: s1_states.py MODEL in.jsonl out.pt"""
import sys, torch
from fc import *
name, inp, outp = sys.argv[1:4]
rows = load_jsonl(inp)
m, tok = load(name)
X = last_states(m, tok, [r['text'] for r in rows], bs=16)
torch.save(X, outp); print(X.shape)
