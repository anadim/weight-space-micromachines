import sys, traceback
from transformers import AutoTokenizer
CALC_SYMS = list('0123456789+-*/.=') + [' ', ' $', '$', '\n', '.\n', ':\n', ',\n', ' \n', ' +', ' -', ' *', ' /', ' x', ' ×', '>>', '--']
cands = [
 "HuggingFaceTB/SmolLM2-1.7B",
 "HuggingFaceTB/SmolLM2-360M",
 "HuggingFaceTB/SmolLM3-3B-Base",
 "allenai/OLMo-2-0425-1B",
 "allenai/tmax-4b",
 "ibm-granite/granite-3.3-2b-base",
 "LiquidAI/LFM2-1.2B",
 "meta-llama/Llama-3.2-1B",
 "google/gemma-3-1b-pt",
 "microsoft/Phi-4-mini-instruct",
]
for m in cands:
    try:
        tk = AutoTokenizer.from_pretrained(m)
        def enc(s): return tk(s, add_special_tokens=False)['input_ids']
        n123 = enc("12345")
        singledigit = all(len(enc(d))==1 for d in "0123456789") and len(n123)==5
        # symbols that are NOT exactly one token:
        bad=[]
        for s in CALC_SYMS:
            t=enc(s)
            if len(t)!=1: bad.append((repr(s),len(t)))
        print(f"OK {m} | 12345->{n123} singledigit={singledigit} | vocab={tk.vocab_size} | bad_syms={bad[:12]}")
    except Exception as e:
        print(f"FAIL {m} | {type(e).__name__}: {str(e)[:120]}")
