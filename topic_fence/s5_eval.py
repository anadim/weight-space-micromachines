"""FINAL evaluation on the untouched test set. usage: s5_eval.py MODEL TAG   (needs res/probe_{F}_{TAG}.json, res/mon_{F}_{TAG}.json,
res/latent_cfg_{F}_{TAG}.json). Generates replies for every condition, then judges. Writes res/test_{TAG}_gen.json."""
import sys, json, re, torch, numpy as np
from fc import *
name, TAG = sys.argv[1:3]
CONDS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['plain', 'sys', 'sys_strong', 'kw', 'kw_stop', 'probe_text', 'probe_text_stop', 'probe_latent', 'monitor']
rows = load_jsonl('data/test.jsonl'); mm = load_jsonl('data/mmlu_ret.jsonl')
texts = [r['text'] for r in rows]; mtexts = [r['text'] for r in mm]
m, tok = load(name)
import os
OUT = os.environ.get('OUT', f'res/test_{TAG}_gen.json')
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
def letter(t):
    x = re.search(r'\b([ABCD])\b', t[:40]); return x.group(1) if x else None

@torch.no_grad()
def silent_check(fz, prompts, gates=None, max_new=40, bs=32):
    """per-step logits of non-fired rows vs the plain host on identical batches; returns (#rows checked, #identical, max|diff|)."""
    n = same = 0; mx = 0.0
    eos = [tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id]
    for i in range(0, len(prompts), bs):
        P = prompts[i:i + bs]; enc = tok(chat(tok, P), return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')
        kw = dict(max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None, eos_token_id=eos, output_scores=True,
                  return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        a = m.generate(**enc, **kw)
        fz.reset(len(P), None if gates is None else torch.tensor(gates[i:i + bs])); b = m.generate(**enc, **kw)
        fired = (fz.wrote > 0); fz.state = None
        T = min(len(a.scores), len(b.scores))
        for r in range(len(P)):
            if fired[r]: continue
            d = max(float((a.scores[t][r] - b.scores[t][r]).abs().max()) for t in range(T))
            same_len = a.sequences[r].shape == b.sequences[r].shape and bool((a.sequences[r] == b.sequences[r]).all())
            n += 1; same += int(d == 0.0 and same_len); mx = max(mx, d)
    return n, same, mx

base_cache = None
for F in os.environ.get('FENCES', 'math,ml').split(','):
    probe = json.load(open(f'res/probe_{F}_{TAG}.json'))['probe']
    mon = json.load(open(f'res/mon_{F}_{TAG}.json'))
    if 'probe_latent' in CONDS:
        lat = json.load(open(f'res/latent_cfg_{F}_{TAG}.json'))
        V = torch.load(f'res/lat_{F}_{TAG}.pt')[lat['v']][lat['L2']] * lat['a']
        if lat.get('probe_layer') and lat['probe_layer'] != probe['layer']:
            probe_lat = json.load(open(f'res/probe_{F}_{TAG}_L{lat["probe_layer"]}.json'))['probe']
        else: probe_lat = probe
    for c in CONDS:
        if f'{F}/{c}' in out: base_cache = out[f'{F}/{c}'] if c == 'plain' else base_cache; continue
        if c == 'plain' and base_cache is not None: out[f'{F}/plain'] = base_cache; continue
        fz = None; gates = None; system = None
        if c == 'sys': system = SYSTEM[F]
        elif c == 'sys_strong': system = SYSTEM_STRONG[F]
        elif c == 'kw':
            fz = Fence(m, tok, 'text', phrase=PHRASE[F]); gates = [bool(KW[F].search(t)) for t in texts]; gm = [bool(KW[F].search(t)) for t in mtexts]
        elif c == 'kw_stop':
            fz = Fence(m, tok, 'text', phrase=PHRASE[F], stop=True); gates = [bool(KW[F].search(t)) for t in texts]; gm = [bool(KW[F].search(t)) for t in mtexts]
        elif c == 'probe_text': fz = Fence(m, tok, 'text', phrase=PHRASE[F], probe=probe)
        elif c == 'probe_text_stop': fz = Fence(m, tok, 'text', phrase=PHRASE[F], probe=probe, stop=True)
        elif c == 'probe_latent': fz = Fence(m, tok, 'latent', probe=probe_lat, vec=V, L2=lat['L2'], nsteps=lat['ns'])
        elif c == 'monitor': fz = Fence(m, tok, 'monitor', phrase=PHRASE[F], probe=dict(layer=mon['layer'], w=mon['w'], b=mon['b'], thr=mon['thr']), k=mon['k'])
        G = generate(m, tok, texts, fence=fz, gates=gates, system=system, max_new=120, bs=32)
        GM = generate(m, tok, mtexts, fence=fz, gates=gm if c in ('kw', 'kw_stop') else None, system=system, max_new=8, bs=32)
        rec = dict(gen=[{k: v for k, v in g.items()} for g in G], mmlu=[dict(text=g['text'], fired=g.get('fired', False), pred=letter(g['text'])) for g in GM])
        if fz is not None:
            rec['silent'] = silent_check(fz, texts[:640], gates=gates[:640] if gates else None)
            print(F, c, 'silent', rec['silent'], flush=True)
        if fz is not None: fz.detach()
        out[f'{F}/{c}'] = rec
        if c == 'plain': base_cache = rec
        fr = np.mean([g.get('fired', False) for g in G]); print(F, c, 'done; fire', round(fr, 3), flush=True)
        json.dump(out, open(OUT, 'w'), ensure_ascii=False)
