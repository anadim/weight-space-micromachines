"""Item 1: calculator accuracy vs operand size, frozen Qwen3-0.6B-Base, few-shot, no training.
Multiplication heatmap (d_a x d_b, 1..9) + addition/subtraction vs digit length. Base vs +calc_s4 implant."""
import os, sys, json, time, re, random, argparse
os.environ.setdefault('MKL_CBWR','COMPATIBLE')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach
ap=argparse.ArgumentParser(); ap.add_argument('--out',default='evidence/item1'); ap.add_argument('--n',type=int,default=100)
ap.add_argument('--bs',type=int,default=120); ap.add_argument('--seed',type=int,default=0); a=ap.parse_args()
os.makedirs(a.out,exist_ok=True)
M='Qwen/Qwen3-0.6B-Base'; tok=AutoTokenizer.from_pretrained(M)
dev='cuda' if torch.cuda.is_available() else 'cpu'
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16 if dev=='cuda' else torch.float32).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
net=attach(model,tok,machines=('calc_s4',)); print('attached calc_s4',flush=True)
eos=tok.convert_tokens_to_ids('<|endoftext|>')
FS={'*':("Q: What is 48 * 27?\nA: 48 * 27 = 1296.\n\n""Q: What is 7341 * 86?\nA: 7341 * 86 = 631326.\n\n"),
    '+':("Q: What is 48 + 27?\nA: 48 + 27 = 75.\n\n""Q: What is 7341 + 86?\nA: 7341 + 86 = 7427.\n\n"),
    '-':("Q: What is 48 - 27?\nA: 48 - 27 = 21.\n\n""Q: What is 50 - 134?\nA: 50 - 134 = -84.\n\n")}
NUM=re.compile(r'-?\d+')
def prompt(a_,b_,op): return FS[op]+f"Q: What is {a_} {op} {b_}?\nA: {a_} {op} {b_} ="
@torch.no_grad()
def batch_gen(prompts, use, max_new=42):
    seqs=[tok(p,add_special_tokens=False)['input_ids'] for p in prompts]
    L=max(len(s) for s in seqs); B=len(seqs)
    x=torch.tensor([[eos]*(L-len(s))+s for s in seqs],device=dev)
    att=torch.tensor([[0]*(L-len(s))+[1]*len(s) for s in seqs],device=dev)
    if use: net.prime(x)
    else: net.pending=None
    o=model.model(input_ids=x,attention_mask=att,use_cache=True); past=o.past_key_values
    lg=model.lm_head(o.last_hidden_state[:,-1]).float()
    gen=[[] for _ in range(B)]; done=[False]*B
    for _ in range(max_new):
        nxt=lg.argmax(-1); nl=nxt.tolist()
        for i in range(B):
            if done[i]: continue
            gen[i].append(nl[i])
            if nl[i]==eos or '\n' in tok.decode(gen[i][-1:]): done[i]=True
        if all(done): break
        nxt=torch.where(torch.tensor(done,device=dev),torch.full_like(nxt,eos),nxt)
        att=torch.cat([att,att.new_ones(B,1)],1)
        if use: net.step(nxt)
        o=model.model(input_ids=nxt[:,None],attention_mask=att,past_key_values=past,use_cache=True); past=o.past_key_values
        lg=model.lm_head(o.last_hidden_state[:,-1]).float()
    return [tok.decode(g) for g in gen]
def parse(txt):
    m=NUM.search(txt); return m.group(0) if m else None
def run_cells(cells, op):
    """cells: list of (da,db). returns dict (da,db)->(base_acc,impl_acc,n)."""
    probs=[]  # (idx, a,b,ans)
    rng=random.Random(a.seed)
    meta=[]
    for (da,db) in cells:
        for _ in range(a.n):
            A_=rng.randint(10**(da-1) if da>1 else 1,10**da-1)
            B_=rng.randint(10**(db-1) if db>1 else 1,10**db-1)
            ans={'*':A_*B_,'+':A_+B_,'-':A_-B_}[op]
            meta.append((da,db,A_,B_,ans))
    res={}
    for use in (False,True):
        correct={}
        for i in range(0,len(meta),a.bs):
            chunk=meta[i:i+a.bs]
            outs=batch_gen([prompt(A_,B_,op) for _,_,A_,B_,_ in chunk],use)
            for (da,db,A_,B_,ans),txt in zip(chunk,outs):
                g=parse(txt); ok=(g is not None and int(g)==ans)
                correct.setdefault((da,db),[0,0]); correct[(da,db)][0]+=ok; correct[(da,db)][1]+=1
            print(f'{op} use={use} {i+len(chunk)}/{len(meta)}',flush=True)
        for k,(c,t) in correct.items():
            res.setdefault(k,{}); res[k]['impl' if use else 'base']=c/t; res[k]['n']=t
    return res
t0=time.time()
# multiplication heatmap
mult_cells=[(da,db) for da in range(1,10) for db in range(1,10)]
mult=run_cells(mult_cells,'*')
json.dump({f'{k[0]}x{k[1]}':v for k,v in mult.items()},open(f'{a.out}/mult_heatmap.json','w'),indent=1)
print('mult done',time.time()-t0,flush=True)
# add/sub vs digit length (both operands d digits)
for op,name in (('+','add'),('-','sub')):
    cells=[(d,d) for d in range(1,10)]
    r=run_cells(cells,op)
    json.dump({f'd{k[0]}':v for k,v in r.items()},open(f'{a.out}/{name}_bylen.json','w'),indent=1)
    print(name,'done',time.time()-t0,flush=True)
print('ALL DONE',time.time()-t0,flush=True)
