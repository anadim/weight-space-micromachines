"""Item 4: token-level authorship visualization. For one BBH question per task, run frozen Qwen3-0.6B-Base
with all nine implants and record which tokens the model wrote vs which an implant forced. Emit an HTML page."""
import os, sys, re, json, time, html, argparse, torch
os.environ.setdefault('MKL_CBWR','COMPATIBLE')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
from micromachines import attach, BBH_MACHINES, VETO_MACHINES
ap=argparse.ArgumentParser(); ap.add_argument('--out',default='evidence/item4')
ap.add_argument('--bbh',default='data/bbh')
ap.add_argument('--veto',default='runs/veto_qwen3.pt')
ap.add_argument('--max_new',type=int,default=230); a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
TASKS=['multistep_arithmetic_two','navigate','web_of_lies','boolean_expressions','temporal_sequences',
       'tracking_shuffled_objects_three_objects','logical_deduction_three_objects']
M='Qwen/Qwen3-0.6B-Base'; dev='cuda' if torch.cuda.is_available() else 'cpu'
tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16 if dev=='cuda' else torch.float32).to(dev).eval()
veto=a.veto if os.path.exists(a.veto) else None
t0=time.time(); net=attach(model,tok,machines=BBH_MACHINES,veto=veto if any(m in VETO_MACHINES for m in BBH_MACHINES) else None)
print(f'attached nine in {time.time()-t0:.0f}s',flush=True)
eos=tok.convert_tokens_to_ids('<|endoftext|>')
pre=lambda t: open(f'{a.bbh}/cot/{t}.txt').read().split('-----',1)[1].strip()+'\n\n'
@torch.no_grad()
def gen_one(prompt,max_new):
    ids=torch.tensor([tok(prompt,add_special_tokens=False)['input_ids']],device=dev)
    net.prime(ids)
    o=model.model(input_ids=ids,use_cache=True); past=o.past_key_values
    lg=model.lm_head(o.last_hidden_state[:,-1]).float()
    wrote_now=bool((net.pending[:,-1].abs().sum()>0).item())
    toks=[]; auth=[]
    for step in range(max_new):
        if step<4: lg[:,eos]=-1e9
        nx=lg.argmax(-1); t=nx.item(); toks.append(t); auth.append(wrote_now)
        if t==eos or (len(toks)%6==0 and '\n\nQ' in tok.decode(toks[-12:])): break
        net.step(nx); wrote_now=bool((net.pending.abs().sum()>0).item())
        o=model.model(input_ids=nx[:,None],past_key_values=past,use_cache=True); past=o.past_key_values
        lg=model.lm_head(o.last_hidden_state[:,-1]).float()
    return toks,auth
def spans(toks,auth):
    out=[]; i=0
    while i<len(toks):
        j=i
        while j<len(toks) and auth[j]==auth[i]: j+=1
        out.append((auth[i],tok.decode(toks[i:j]))); i=j
    return out
cards=[]
for t in TASKS:
    ex=json.load(open(f'{a.bbh}/data/{t}.json'))['examples'][0]
    prompt=pre(t)+f"Q: {ex['input']}\nA: Let's think step by step."
    toks,auth=gen_one(prompt,a.max_new)
    nw=sum(auth); txt=tok.decode(toks).replace('<|endoftext|>','')
    print(f'{t:45s} tokens={len(toks)} implant_written={nw}',flush=True)
    body=''.join((f'<span class="imp">{html.escape(s)}</span>' if w else html.escape(s)) for w,s in spans(toks,auth))
    cards.append((t,ex['input'],ex['target'],nw,len(toks),body))
    json.dump({'task':t,'question':ex['input'],'target':ex['target'],'implant_tokens':nw,'total_tokens':len(toks),
               'spans':[{'implant':w,'text':s} for w,s in spans(toks,auth)]},open(f'{a.out}/{t}.json','w'),indent=1)
H=['<!doctype html><meta charset=utf-8><title>Where each implant fires</title>',
'<style>body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:860px;margin:40px auto;color:#222;padding:0 16px}',
'h1{font-size:24px}h2{font-size:16px;margin:28px 0 6px;color:#111}.q{color:#6b7280;font-size:13px;margin:2px 0 8px}',
'pre{white-space:pre-wrap;word-wrap:break-word;background:#fafafa;border:1px solid #eee;border-radius:8px;padding:14px;font-size:13px;line-height:1.5}',
'.imp{background:#118659;color:#fff;border-radius:3px;padding:0 2px}.legend span{padding:1px 6px;border-radius:3px}</style>',
'<h1>Where each implant fires</h1>',
'<p class="legend">Frozen Qwen3-0.6B-Base, all nine implants attached. '
'<span style="background:#f3f4f6">model-written tokens</span> &nbsp; <span class="imp">implant-written tokens</span>. '
'One question per task, greedy decoding.</p>']
for t,q,tgt,nw,tot,body in cards:
    H.append(f'<h2>{t}</h2><div class="q">gold: {html.escape(str(tgt))} &middot; implant wrote {nw}/{tot} generated tokens</div><pre>{body}</pre>')
open(f'{a.out}/fire_viz.html','w').write('\n'.join(H))
print('wrote',f'{a.out}/fire_viz.html',flush=True)
