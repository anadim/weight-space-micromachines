"""Silence check for the transferred calculator: attach calc to a frozen host; on text with no 'a op b =' trigger the
implant's write is exactly 0 at every position, so the host's logits must be bit-identical to the unattached model, and
the host weights are untouched. Also reports how many positions fired on GSM-style text (should be only at equations)."""
import os, sys, json, hashlib, argparse, torch
sys.path.insert(0, os.environ.get('WSM_REPO', '.'))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach
from micromachines.specs import SPECS

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--veto', required=True)
ap.add_argument('--calc_machine', default='calc_s4'); ap.add_argument('--out', default='/tmp/silent.json')
a = ap.parse_args()
dev='cuda'
tok=AutoTokenizer.from_pretrained(a.model)
model=AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
dig=lambda: hashlib.sha256(b''.join(p.detach().float().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
PROSE=[
 "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
 "In 1999 the committee approved the proposal after a long and heated debate about funding.",
 "She walked into the room, looked around, and quietly closed the door behind her.",
 "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.",
 "def add(a, b):\n    total = a + b\n    return total\n\nprint(add(3, 4))",
 "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
 "The treaty was signed by seventeen nations, ending nearly a decade of conflict.",
 "Chapter one begins with a description of the village and its oldest inhabitants.",
]
GSM=["Natalia sold clips to 48 friends. In total she sold 48 + 24 = 72 clips. She earned 72 / 2 = 36 dollars."]
def _san(name):
    sp=SPECS[name]; txt=dict(sp['text'])
    for k,v in list(txt.items()):
        alts=list(v) if isinstance(v,(list,tuple)) else [v]
        keep=[x for x in alts if len(tok(x,add_special_tokens=False)['input_ids'])==1]
        txt[k]=keep[0] if len(keep)==1 else tuple(keep)
    SPECS[name+'_v']=dict(sp,text=txt); return name+'_v'
CM=_san(a.calc_machine)
w0=dig()
@torch.no_grad()
def lg(ids): return model(input_ids=ids).logits[0].float()
ref=[lg(torch.tensor([tok(x,add_special_tokens=False)['input_ids']],device=dev)) for x in PROSE+GSM]
net=attach(model, tok, machines=(CM,), veto=a.veto, veto_machines=(CM,), zero_os=True)
w1=dig()
rep={'model':a.model,'calc_machine':a.calc_machine,'host_weights_unchanged':w0==w1,'texts':[]}
allgood=True; prose_fires=0
for x,r in zip(PROSE+GSM, ref):
    ids=torch.tensor([tok(x,add_special_tokens=False)['input_ids']],device=dev)
    net.prime(ids, full=True); w=net.pending[0]; quiet=(w==0).all(-1)
    out=lg(ids); same=torch.equal(out[quiet], r[quiet]); fired=int((~quiet).sum())
    is_prose = x in PROSE
    if is_prose: prose_fires+=fired
    allgood = allgood and same
    rep['texts'].append({'prose':is_prose,'tokens':int(quiet.numel()),'fired':fired,'quiet_bit_identical':bool(same)})
rep['all_quiet_bit_identical']=bool(allgood); rep['prose_fires']=prose_fires
json.dump(rep, open(a.out,'w'), indent=1)
print(f"host_weights_unchanged={rep['host_weights_unchanged']} all_quiet_bit_identical={allgood} prose_fires={prose_fires} (GSM text fired at equation positions)")
