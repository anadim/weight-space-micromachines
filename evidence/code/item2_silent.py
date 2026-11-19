"""Item 2: silent property at scale. All nine BBH implants attached to frozen Qwen3-0.6B-Base.
Over a large natural corpus (wikitext prose + real code + GSM8K solutions): count every position where ANY
implant writes, confirm logits bit-identical everywhere else, report false-fire rate per 1M tokens,
dump context of every fire. One prime per machine; incremental save every 100 passages."""
import os, sys, json, glob, time, argparse, torch
os.environ.setdefault('MKL_CBWR','COMPATIBLE')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach, BBH_MACHINES
ap=argparse.ArgumentParser(); ap.add_argument('--out',default='evidence/item2')
ap.add_argument('--n_wiki',type=int,default=1000); ap.add_argument('--n_code',type=int,default=600)
ap.add_argument('--n_gsm',type=int,default=400); ap.add_argument('--maxtok',type=int,default=160)
ap.add_argument('--limit',type=int,default=0); ap.add_argument('--machines',default=''); a=ap.parse_args()
os.makedirs(a.out,exist_ok=True)
REPO=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
M='Qwen/Qwen3-0.6B-Base'; tok=AutoTokenizer.from_pretrained(M)
dev='cuda' if torch.cuda.is_available() else 'cpu'
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
# corpus
corpus=[]
from datasets import load_dataset
d=load_dataset('wikitext','wikitext-103-raw-v1',split='test'); cnt=0
for e in d:
    t=e['text'].strip()
    if len(t)>200 and not t.startswith('='): corpus.append(('wikitext',t)); cnt+=1
    if cnt>=a.n_wiki: break
pyfiles=[]
for root in [os.path.dirname(torch.__file__), REPO]: pyfiles+=glob.glob(root+'/**/*.py',recursive=True)
cnt=0
for f in sorted(set(pyfiles)):
    try: src=open(f).read()
    except Exception: continue
    if len(src)<300: continue
    corpus.append(('code',src)); cnt+=1
    if cnt>=a.n_code: break
rows=[json.loads(l) for l in open(f'{REPO}/data/gsm8k/train.jsonl')]
corpus+=[('gsm8k', r['answer']) for r in rows[:a.n_gsm]]
if a.limit: corpus=corpus[:a.limit]
print(f'corpus: {len(corpus)} passages',flush=True)
MACH=tuple(a.machines.split(',')) if a.machines else BBH_MACHINES
t0=time.time(); net=attach(model,tok,machines=MACH,veto=None)
print(f'attached {len(MACH)} machines in {time.time()-t0:.0f}s',flush=True)
emb_layer=model.get_input_embeddings(); names=list(net.nets.keys())
@torch.no_grad()
def logits(ids): return model(input_ids=ids).logits[0].float()
tot_tok=0; tot_fire=0; bad=0; fires=[]; per_machine={n:0 for n in names}; by_source={}
def save():
    rep={'model':M,'passages':len(corpus),'tokens':tot_tok,'positions_fired':tot_fire,
     'texts_with_quiet_mismatch':bad,'fires_per_1M_tokens':tot_fire/max(tot_tok,1)*1e6,
     'per_machine_fires':per_machine,'by_source':{k:{'tokens':v[0],'fires':v[1]} for k,v in by_source.items()},
     'maxtok':a.maxtok,'seconds':time.time()-t0}
    json.dump(rep,open(f'{a.out}/silent_scale.json','w'),indent=1)
    json.dump(fires,open(f'{a.out}/fires.json','w'),indent=1)
    return rep
for pi,(src,text) in enumerate(corpus):
    ids_list=tok(text,add_special_tokens=False)['input_ids'][:a.maxtok]
    if len(ids_list)<8: continue
    ids=torch.tensor([ids_list],device=dev)
    net.pending=None; ref=logits(ids); emb=emb_layer(ids)
    permach={}; pend=None; fired_any=torch.zeros(ids.shape[1],dtype=torch.bool,device=dev)
    for n in names:
        w=net.nets[n].prime(emb, full=True)
        pend = w if pend is None else pend+w
        f=(w!=0).any(-1)[0]; permach[n]=f; fired_any|=f
    net.pending=pend
    lg=logits(ids); quiet=~fired_any
    if not torch.equal(lg[quiet], ref[quiet]): bad+=1
    tot_tok+=len(ids_list); by_source.setdefault(src,[0,0]); by_source[src][0]+=len(ids_list)
    fpos=fired_any.nonzero().flatten().tolist(); tot_fire+=len(fpos); by_source[src][1]+=len(fpos)
    for n in names: per_machine[n]+=int(permach[n].sum())
    for p in fpos:
        which=[n for n in names if bool(permach[n][p])]
        fires.append({'passage':pi,'source':src,'pos':p,'machines':which,
                      'context':tok.decode(ids_list[max(0,p-12):p+2]),'token':tok.decode([ids_list[p]])})
    net.pending=None
    if pi%100==0:
        print(f'{pi}/{len(corpus)} tok={tot_tok} fires={tot_fire} bad={bad} {time.time()-t0:.0f}s',flush=True)
        save(); torch.cuda.empty_cache()
rep=save()
print('SUMMARY',json.dumps(rep),flush=True)
print(f'quiet-position bit-identical on ALL passages: {bad==0}',flush=True)
