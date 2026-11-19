import sys
sys.path.insert(0,__import__('os').environ.get('WSM_REPO','.'))
from micromachines.specs import SPECS
from transformers import AutoTokenizer
spec='calc_s3'
text=SPECS[spec]['text']; syms=SPECS[spec]['syms']; out=SPECS[spec]['out']
print("SYMS:",syms); print("OUT:",out)
models=sys.argv[1:]
for m in models:
    tk=AutoTokenizer.from_pretrained(m)
    enc=lambda s: tk(s,add_special_tokens=False)['input_ids']
    bad={}
    for s in syms:
        t=text[s]; alts=list(t) if isinstance(t,(list,tuple)) else [t]
        res=[(a,len(enc(a))) for a in alts]
        multi=[a for a,n in res if n!=1]
        single=[a for a,n in res if n==1]
        if multi: bad[s]=(multi,single)
    print(f"\n== {m} ==")
    for s,(mu,si) in bad.items():
        print(f"  sym {s!r}: multitoken={[repr(x) for x in mu]} single_ok={[repr(x) for x in si]}")
    if not bad: print("  ALL SINGLE-TOKEN")
