"""Sanity-triggered restart cascade (simulated exactly from per-variant greedy runs): if the off answer is flagged
(missing / negative / non-integer), take the first unflagged answer among the listed restart prompts (else the last)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_trig as RT
def L(f): return {(x['set'], x['i']): x for x in map(json.loads, open(f))}
for D in sys.argv[1:]:
    V = {f[:-6]: L(f'{D}/{f}') for f in os.listdir(D) if f.endswith('.jsonl') and '@' not in f}
    off = V['off']
    for casc in (['T1s0'], ['plan1'], ['re2q'], ['plan1', 're2q'], ['plan1', 're2q', 'T1s0'], ['re2q', 'plan1', 'T1s0'], ['plan1', 're2nshots', 're2q', 'T1s0']):
        if any(c not in V for c in casc): continue
        for s in sorted({k[0] for k in off}):
            ks = [k for k in off if k[0] == s]; fx = br = fired = 0; extra = 0
            for k in ks:
                x = off[k]
                if not RT.t_san(x['txt']): continue
                fired += 1; y = x
                for c in casc:
                    y = V[c][k]; extra += y['ntok']
                    if not RT.t_san(y['txt']): break
                fx += (y['ok'] > 0) and not (x['ok'] > 0); br += (x['ok'] > 0) and not (y['ok'] > 0)
            base = sum(off[k]['ok'] for k in ks); tot = sum(off[k]['ntok'] for k in ks)
            print(f"{D.split('/')[-1]:9s} {s:4s} {'>'.join(casc):30s} n {len(ks)} fire {fired} fix {fx} brk {br} acc {100*base/len(ks):.2f}->{100*(base+fx-br)/len(ks):.2f} ({100*(fx-br)/len(ks):+.2f}) tok +{100*extra/tot:.1f}%")
