"""How does the host end its answer after '#### N'? (dev items, host + calc, greedy, no rethink attached)
Counts the token sequence the host writes right after the answer number, and whether it hits the rethink circuit's
end rows (token contains '\\n\\n' or is <|endoftext|>, or '\\n' after a token ending in '\\n').
python end_check.py --ckpt DIR|base --n 300"""
import os, sys, re, json, random, argparse, collections, torch
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fastqwen as Q
from engine import Engine
from implant import CalcImplant
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='base'); ap.add_argument('--n', type=int, default=300)
ap.add_argument('--B_max', type=int, default=128); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
hf = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16).cuda().eval()
pre, shot_q = C.prefix_ids(tok)
dev = C.load_train('B', shot_q=shot_q); random.Random(5).shuffle(dev); items = dev[:a.n]
sd = Q.load_hf_params(dtype=torch.bfloat16)
if a.ckpt != 'base':
    from safetensors.torch import load_file
    ck = load_file(f'{a.ckpt}/model.safetensors', device='cuda')
    for k in sd: sd[k].copy_(ck[k])
W = Q.fuse(sd)
imp = CalcImplant(hf, tok, a.B_max, fused=True); del hf
eng = Engine(W, pre, tok, implant=imp, B_max=a.B_max, S_max=544)
out = eng.generate([C.q_ids(tok, r['q']) for r in items], G=1, greedy=True, max_new=320)
ends = collections.Counter(); after = collections.Counter()
for ids in out['tokens']:
    txt = [tok.decode([t]) for t in ids]
    k = next((j for j, t in enumerate(txt) if '####' in t), None)
    if k is None: ends['<no ####>'] += 1; continue
    j = k + 1
    while j < len(txt) and re.fullmatch(r'[ \d,\.\-\$]*', txt[j]) and txt[j] != '': j += 1     # ' ' + digits of the number
    tail = ''.join(txt[j:j + 3])
    after[repr(tail[:12])] += 1
    nxt = txt[j] if j < len(txt) else '<cut>'
    endtok = ('\n\n' in nxt) or nxt == '<|endoftext|>' or (nxt.startswith('\n') and j > 0 and txt[j - 1].endswith('\n'))
    ends['end-token right after the number' if endtok else 'other: ' + repr(nxt)] += 1
print(a.ckpt, json.dumps(ends.most_common(8)), flush=True)
print('  next text after number:', after.most_common(8), flush=True)
