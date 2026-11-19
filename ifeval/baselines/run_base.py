"""Variant of run.py: adds --think (enable_thinking) and --sys (system prompt for all rows). Same batching/decoding."""
import argparse, json, os, pickle, time, torch
from common import load, EOS_IDS
ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--mode", default="plain")
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--max_new", type=int, default=2048)
ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--keys", default="")
ap.add_argument("--think", type=int, default=0); ap.add_argument("--sys", default="")
a = ap.parse_args()
tok, model = load()
docs = [json.loads(l) for l in open(a.data)]
if a.keys: ks=set(map(int,a.keys.split(","))); docs=[d for d in docs if d["key"] in ks]
if a.limit: docs=docs[:a.limit]
SYS = a.sys if a.sys else ""
def msgs(d):
    m=[]
    s = SYS or d.get("system")
    if s: m.append({"role":"system","content":s})
    m += d.get("history") or []
    m.append({"role":"user","content":d["prompt"]})
    return m
TH = bool(a.think)
P_all=[tok(tok.apply_chat_template(msgs(d),tokenize=False,add_generation_prompt=True,enable_thinking=TH),add_special_tokens=False).input_ids for d in docs]
order=sorted(range(len(docs)),key=lambda i:(len(P_all[i]),docs[i]["key"]))
batches=[order[b:b+a.bs] for b in range(0,len(order),a.bs)]
si,sn=map(int,a.shard.split("/")); mine=[bi for bi in range(len(batches)) if bi%sn==si]
CIRC=None
if a.mode=="implant":
    from circuit import Circuit
    CIRC=Circuit(tok,model); state={"on":False}
    def _hook(mod,x):
        if not state["on"]: return None
        h=x[0]; return (h+CIRC.write(h[:,-1])[:,None].to(h.dtype),)
    model.model.norm.register_forward_pre_hook(_hook)
@torch.no_grad()
def run(idx):
    P=[P_all[i] for i in idx]; B=len(P); L=max(map(len,P))
    x=torch.tensor([[EOS_IDS[1]]*(L-len(p))+p for p in P],device="cuda")
    att=torch.tensor([[0]*(L-len(p))+[1]*len(p) for p in P],device="cuda")
    out=[[] for _ in range(B)]; done=torch.zeros(B,dtype=torch.bool,device="cuda")
    if CIRC is not None:
        CIRC.reset(B)
        for t in range(L): CIRC.step(x[:,t])
        state["on"]=True
    o=model.model(input_ids=x,attention_mask=att,use_cache=True); past=o.past_key_values
    lg=model.lm_head(o.last_hidden_state[:,-1]).float()
    for step in range(a.max_new):
        nxt=lg.argmax(-1); fin=(nxt==EOS_IDS[0])|(nxt==EOS_IDS[1])
        for i in torch.nonzero(~done & ~fin)[:,0].tolist(): out[i].append(nxt[i].item())
        done=done|fin
        if bool(done.all()): break
        nxt=torch.where(done,torch.full_like(nxt,EOS_IDS[1]),nxt)
        att=torch.cat([att,att.new_ones(B,1)],1)
        if CIRC is not None: CIRC.step(nxt)
        o=model.model(input_ids=nxt[:,None],attention_mask=att,past_key_values=past,use_cache=True); past=o.past_key_values
        lg=model.lm_head(o.last_hidden_state[:,-1]).float()
    if CIRC is not None: state["on"]=False
    info=CIRC.report() if CIRC is not None else [None]*B
    return [(docs[i],tok.decode(out[j]),bool(done[j]),len(out[j]),info[j]) for j,i in enumerate(idx)]
part=a.out+".partial"; res={}
if os.path.exists(part): res=pickle.load(open(part,"rb")); print("resumed",len(res),flush=True)
t0=time.time()
for bi in mine:
    if bi in res: continue
    while True:
        try: res[bi]=run(batches[bi]); break
        except torch.OutOfMemoryError:
            if CIRC is not None: state["on"]=False
            torch.cuda.empty_cache(); print("OOM batch",bi,flush=True); time.sleep(20)
    pickle.dump(res,open(part,"wb")); print(f"{len(res)}/{len(mine)} {time.time()-t0:.0f}s",flush=True)
rows=sorted([r for v in res.values() for r in v],key=lambda r:r[0]["key"])
with open(a.out,"w") as f:
    for d,resp,fin,n,info in rows:
        f.write(json.dumps({"key":d["key"],"response":resp,"finish":"stop" if fin else "length","ntok":n,"circ":info})+"\n")
os.remove(part); print("done",time.time()-t0)
