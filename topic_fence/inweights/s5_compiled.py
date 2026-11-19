"""FINAL TEST eval with the COMPILED blocker (blocker_circuit.BlockerCircuit), conditions probe_text_stop + monitor
(+ plain), same test sets/judge/metrics as the emulated run. usage: s5_compiled.py MODEL TAG  (FENCES env)."""
import sys, os, json, re, torch, numpy as np
import fc
from fc import load, chat, load_jsonl
from blocker_circuit import BlockerCircuit
name, TAG = sys.argv[1:3]
CONDS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['plain', 'probe_text_stop', 'monitor']
rows = load_jsonl('data/test.jsonl'); mm = load_jsonl('data/mmlu_ret.jsonl')
texts = [r['text'] for r in rows]; mtexts = [r['text'] for r in mm]
m, tok = load(name)
EOS = [tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id]
def letter(t):
    x = re.search(r'\b([ABCD])\b', t[:40]); return x.group(1) if x else None

@torch.no_grad()
def gen_compiled(prompts, bc, max_new, bs=32):
    res = []
    for i in range(0, len(prompts), bs):
        P = prompts[i:i + bs]
        enc = tok(chat(tok, P), return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')
        out = m.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                         eos_token_id=EOS, pad_token_id=tok.pad_token_id)
        g = out[:, enc['input_ids'].shape[1]:]
        fired = bc.fired.tolist() if bc is not None else [False] * len(P)
        fstep = bc.firestep.tolist() if bc is not None else [-1] * len(P)
        for b in range(len(P)):
            ids = g[b].tolist(); cut = [j for j, t in enumerate(ids) if t in EOS]; ids = ids[:cut[0]] if cut else ids
            res.append(dict(text=tok.decode(ids, skip_special_tokens=True), n=len(ids), fired=bool(fired[b]), firestep=int(fstep[b])))
    return res

@torch.no_grad()
def silent_check(bc, prompts, max_new=40, bs=32):
    n = same = 0; mx = 0.0
    for i in range(0, len(prompts), bs):
        P = prompts[i:i + bs]
        enc = tok(chat(tok, P), return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')
        kw = dict(max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None, eos_token_id=EOS,
                  output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        bc.active = False; a = m.generate(**enc, **kw)
        bc.active = True; b = m.generate(**enc, **kw)
        fired = bc.fired.clone()
        Tn = min(len(a.scores), len(b.scores))
        for r in range(len(P)):
            if fired[r]: continue
            d = max(float((a.scores[t][r] - b.scores[t][r]).abs().max()) for t in range(Tn))
            same_len = a.sequences[r].shape == b.sequences[r].shape and bool((a.sequences[r] == b.sequences[r]).all())
            n += 1; same += int(d == 0.0 and same_len); mx = max(mx, d)
    return n, same, mx

OUT = os.environ.get('OUT', f'res/test_{TAG}_gen.json')
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for F in os.environ.get('FENCES', 'math,ml').split(','):
    probe = json.load(open(f'res/probe_{F}_q17.json'))['probe']
    mon = json.load(open(f'res/mon_{F}_q17.json'))
    PHRASE = fc.PHRASE[F]
    for c in CONDS:
        if f'{F}/{c}' in out: continue
        bc = None
        if c == 'probe_text_stop':
            bc = BlockerCircuit(m, tok, 'text_stop', probe, PHRASE)
        elif c == 'monitor':
            bc = BlockerCircuit(m, tok, 'monitor', dict(layer=mon['layer'], w=mon['w'], b=mon['b'], thr=mon['thr']), PHRASE, k=mon['k'])
        G = gen_compiled(texts, bc, 120)
        GM = gen_compiled(mtexts, bc, 8)
        rec = dict(gen=G, mmlu=[dict(text=g['text'], fired=g['fired'], pred=letter(g['text'])) for g in GM])
        if bc is not None:
            rec['silent'] = silent_check(bc, texts[:640])
            print(F, c, 'silent', rec['silent'], flush=True)
            bc.detach()
        out[f'{F}/{c}'] = rec
        print(F, c, 'done; fire', round(np.mean([g['fired'] for g in G]), 3), flush=True)
        json.dump(out, open(OUT, 'w'), ensure_ascii=False)
