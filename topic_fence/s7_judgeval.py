"""Judge validation set (DEV prompts only). 40 (prompt, reply) items from mixed conditions; I hand-label them, then compare
with the LLM judge and the string check. usage: s7_judgeval.py gen | score labels.json"""
import sys, json, random, torch, numpy as np
from fc import *
if sys.argv[1] == 'gen':
    rows = load_jsonl('data/dev.jsonl'); rng = random.Random(7)
    items = []
    for F in ['math', 'ml']:
        pos = [r['text'] for r in rows if r['fence'] == F and r['label'] == 'pos' and r['clf_' + F] > 0]
        hard = [r['text'] for r in rows if r['fence'] == F and r['label'] == 'hard' and r['clf_' + F] < 0]
        rng.shuffle(pos); rng.shuffle(hard)
        items += [(F, 'pos', t) for t in pos[:16]] + [(F, 'hard', t) for t in hard[:8]]
    m, tok = load(HOST)
    V = {F: torch.load(f'res/lat_{F}_{TAG}.pt') for F in ['math', 'ml']}
    out = []
    for F in ['math', 'ml']:
        P = [t for f, l, t in items if f == F]; L = [l for f, l, t in items if f == F]
        conds = {}
        conds['plain'] = generate(m, tok, P, max_new=120)
        conds['sys_strong'] = generate(m, tok, P, system=SYSTEM_STRONG[F], max_new=120)
        fz = Fence(m, tok, 'text', phrase=PHRASE[F]); conds['text'] = generate(m, tok, P, fence=fz, gates=[True] * len(P), max_new=120); fz.detach()
        fz = Fence(m, tok, 'latent', vec=V[F]['v_ref'][18] * 1.0, L2=18); conds['latent'] = generate(m, tok, P, fence=fz, gates=[True] * len(P), max_new=120); fz.detach()
        for c, G in conds.items():
            for p, l, g in zip(P, L, G): out.append(dict(fence=F, label=l, cond=c, prompt=p, reply=g['text']))
    rng.shuffle(out)
    # stratify: 40 items, 10 per condition, mixed fences/labels
    sel = []
    for c in ['plain', 'sys_strong', 'text', 'latent']:
        sel += [o for o in out if o['cond'] == c][:10]
    json.dump(sel, open(f'res/judgeval_items_{TAG}.json', 'w'), indent=1, ensure_ascii=False)
else:
    sel = json.load(open(f'res/judgeval_items_{TAG}.json')); lab = json.load(open(sys.argv[2]))
    J = Judge()
    s = np.array(J([o['prompt'] for o in sel], [o['reply'] for o in sel])) > 0
    h = np.array([lab[str(i)] for i in range(len(sel))]) == 'C'
    st = np.array([not refusal_str(o['reply']) for o in sel])
    print('LLM judge agreement with hand labels: %d/%d' % ((s == h).sum(), len(sel)), ' string-check agreement: %d/%d' % ((st == h).sum(), len(sel)))
    for i, o in enumerate(sel):
        if s[i] != h[i]: print('DISAGREE', i, o['cond'], 'hand', 'C' if h[i] else 'D', '| judge', 'C' if s[i] else 'D', '|', o['reply'][:150].replace('\n', ' '))
    print('confusion (hand C/D x judge C/D):', int((h & s).sum()), int((h & ~s).sum()), int((~h & s).sum()), int((~h & ~s).sum()))
