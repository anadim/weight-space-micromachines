"""Summarize run dirs: dev curve, train reward by fifths, writes/calls, host-only dev. python summ_runs.py 'runs/sweep/*'
--json: also dump one JSON line per run (for the RESULTS table)."""
import sys, json, glob, os
as_json = '--json' in sys.argv
pats = [p for p in sys.argv[1:] if p != '--json']
for d in sorted(sum([glob.glob(p) for p in pats], [])):
    f = os.path.join(d, 'steps.jsonl')
    if not os.path.exists(f): continue
    S, D, HC, H = [], [], [], None
    for l in open(f):
        r = json.loads(l)
        if 'dev' in r: D.append(r['dev'])
        elif 'dev_hostonly' in r: H = r['dev_hostonly']
        elif 'dev_host_curve' in r: HC.append(r['dev_host_curve'])
        elif 'u' in r: S.append(r)
    if not S: continue
    n = len(S); q = max(1, n // 5)
    fifth = [sum(s['reward'] for s in S[i:i + q]) / len(S[i:i + q]) for i in range(0, n - n % q if n >= 5 else n, q)][:5]
    dv = ' '.join(f"{x['u']}:{x['dev_acc']:.1f}" for x in D)
    hv = ' '.join(f"{x['u']}:{x['devhost_acc']:.1f}" for x in HC) + (f" {S[-1]['u']}:{H['devhost_acc']:.1f}" if H else '')
    print(f"{os.path.basename(d):14s} U={n:4d} R5=[{' '.join(f'{x:.3f}' for x in fifth)}] dev[{dv}] host[{hv}] "
          f"gsm/scaled={D[-1]['dev_gsm']:.1f}/{D[-1]['dev_scaled']:.1f} dev_writes={D[-1]['dev_writes']:.1f} dev_calls={D[-1]['dev_calls']:.2f} "
          f"train_writes={S[-1]['writes']:.1f} len={S[-1]['len']:.0f} t={S[-1]['t']:.0f}s")
    if as_json:
        print('JSON', json.dumps(dict(run=os.path.basename(d), U=n, dev=D, host=HC + ([dict(H, u=S[-1]['u'])] if H else []),
                                      reward_fifths=fifth, train_writes=S[-1]['writes'], train_calls=S[-1]['calls'])))
