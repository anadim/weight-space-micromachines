import sys, json, random, torch; sys.path.insert(0,'.')
import fc
from fc import load, chat, Fence, generate, load_jsonl
from blocker_circuit import BlockerCircuit
F = sys.argv[1] if len(sys.argv)>1 else 'math'
NMODE = sys.argv[2] if len(sys.argv)>2 else 'both'
NP = int(sys.argv[3]) if len(sys.argv)>3 else 120
m, tok = load('Qwen/Qwen3-1.7B')
probe = json.load(open(f'res/probe_{F}_q17.json'))['probe']
mon = json.load(open(f'res/mon_{F}_q17.json'))
PHRASE = fc.PHRASE[F]
# dev prompts: mix of this-fence positives and negatives/other
dev = load_jsonl('data/dev_all.jsonl')
rng = random.Random(0); rng.shuffle(dev)
pos = [r['text'] for r in dev if r.get('fence')==F and r.get('label')=='pos'][:NP]
neg = [r['text'] for r in dev if not (r.get('fence')==F and r.get('label')=='pos')][:NP]
prompts = pos + neg
print(f'fence {F}: {len(pos)} pos + {len(neg)} neg = {len(prompts)} prompts')

def run_emulated(kind):
    if kind=='text_stop':
        fz = Fence(m, tok, 'text', phrase=PHRASE, probe=probe, stop=True)
    else:
        fz = Fence(m, tok, 'monitor', phrase=PHRASE, probe=dict(layer=mon['layer'],w=mon['w'],b=mon['b'],thr=mon['thr']), k=mon['k'])
    G = generate(m, tok, prompts, fence=fz, max_new=120, bs=32)
    fz.detach()
    return [(g['text'], g.get('fired',False), g.get('firestep',-1), g.get('score',None)) for g in G]

@torch.no_grad()
def run_compiled(kind):
    bc = BlockerCircuit(m, tok, kind, probe if kind=='text_stop' else dict(layer=mon['layer'],w=mon['w'],b=mon['b'],thr=mon['thr']),
                        PHRASE, k=mon.get('k',3), trace=True)
    eos=[tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id]
    out=[]
    for i in range(0,len(prompts),32):
        P=prompts[i:i+32]
        enc=tok(chat(tok,P),return_tensors='pt',padding=True,truncation=True,max_length=1024).to('cuda')
        gen=m.generate(**enc,max_new_tokens=120,do_sample=False,temperature=None,top_p=None,top_k=None,eos_token_id=eos,pad_token_id=tok.pad_token_id)
        g=gen[:,enc['input_ids'].shape[1]:]
        for b in range(len(P)):
            ids=g[b].tolist(); cut=[j for j,t in enumerate(ids) if t in eos]
            ids=ids[:cut[0]] if cut else ids
            out.append(tok.decode(ids,skip_special_tokens=True))
    bc.detach()
    return out

for kind in (['text_stop','monitor'] if NMODE=='both' else [NMODE]):
    em = run_emulated(kind)
    co = run_compiled(kind)
    agree = sum(e[0]==c for e,c in zip(em,co))
    firem = sum(e[1] for e in em)
    print(f'--- {F} {kind}: text agreement {agree}/{len(prompts)}  (emulated fired {firem})')
    # show first few disagreements
    nd=0
    for i,(e,c) in enumerate(zip(em,co)):
        if e[0]!=c and nd<4:
            nd+=1
            print(f'  DISAGREE idx{i} pos={i<len(pos)} emu_fired={e[1]} firestep={e[2]}')
            print('    EMU:', repr(e[0][:120]))
            print('    CMP:', repr(c[:120]))
