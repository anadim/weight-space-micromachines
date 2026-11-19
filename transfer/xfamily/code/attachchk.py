import sys; sys.path.insert(0,__import__('os').environ.get('WSM_REPO','.'))
from micromachines.specs import SPECS
from transformers import AutoTokenizer
for m in sys.argv[1:]:
    tk=AutoTokenizer.from_pretrained(m, trust_remote_code=True); T=lambda s: tk(s,add_special_tokens=False)['input_ids']
    sp=SPECS['calc_s4']; miss=[]
    for s in sp['syms']:
        v=sp['text'][s]; alts=list(v) if isinstance(v,(list,tuple)) else [v]
        if not any(len(T(a))==1 for a in alts): miss.append((s,alts))
    print(f"{m}: 12345->{[tk.decode([i]) for i in T('12345')]} | calc_s4 attach: {'CANNOT_ATTACH '+str(miss) if miss else 'builds (all symbols single-token)'}")
