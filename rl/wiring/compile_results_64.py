"""Parse runs/<arm>/steps.jsonl into a results table (markdown + json) next to arms 1/2/R."""
import os, sys, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
ORDER = ['1_s0', '2_s0', 'W2_s0', 'W3z_s0', 'W3r_s0', 'W1_s0_impllr1e5', 'W4_s0_impllr1e5', 'W1_s0_impllr1e6',
         '2_s1', 'W2_s1', 'W3z_s1']
LABEL = {'1_s0': 'arm1 no-impl', '2_s0': 'arm2 frozen (ref)', 'W2_s0': 'W2 write+links',
         'W3z_s0': 'W3z links-only, 0-init', 'W3r_s0': 'W3r links-only, rand', 'W1_s0_impllr1e5': 'W1 writer lr1e-5',
         'W4_s0_impllr1e5': 'W4 writer+links', 'W1_s0_impllr1e6': 'W1 writer lr1e-6',
         '2_s1': 'arm2 seed1', 'W2_s1': 'W2 seed1', 'W3z_s1': 'W3z seed1'}


def load(tag):
    p = os.path.join(RUNS, tag, 'steps.jsonl')
    if not os.path.exists(p): return None
    devs, drift, hostonly, linksoff, last = {}, {}, None, None, None
    gate_final = Wn_final = None
    for line in open(p):
        try: r = json.loads(line)
        except: continue
        if 'dev' in r: devs[r['dev']['u']] = r['dev']
        if 'drift' in r: drift[r['drift']['u']] = r['drift']
        if 'drift_final' in r: drift['final'] = r['drift_final']
        if 'dev_hostonly' in r: hostonly = r['dev_hostonly']
        if 'dev_linksoff_final' in r: linksoff = r['dev_linksoff_final']
        if 'link_gate_l2' in r: gate_final = r['link_gate_l2']; Wn_final = r.get('link_W_l2')
        if 'u' in r and 'reward' in r: last = r
    return dict(devs=devs, drift=drift, hostonly=hostonly, linksoff=linksoff, gate=gate_final, Wn=Wn_final, last=last)


def g(dev, k): return None if dev is None else round(dev.get(k), 1)

rows = []
out = {}
for tag in ORDER:
    d = load(tag)
    if d is None: continue
    out[tag] = d
    dv = d['devs']
    curve = ' / '.join(str(g(dv.get(u), 'dev_acc')) for u in (0, 16, 32, 48, 64))
    f = dv.get(64, {})
    row = dict(tag=tag, label=LABEL.get(tag, tag), curve=curve,
               gsm=g(f, 'dev_gsm'), scaled=g(f, 'dev_scaled'),
               writes=round(f.get('dev_writes', 0), 1) if f else None, calls=round(f.get('dev_calls', 0), 2) if f else None,
               hostonly=g(d['hostonly'], 'devhostonly_acc') if d['hostonly'] else None,
               linksoff=g(d['linksoff'], 'devlinksoff_final_acc') if d['linksoff'] else None,
               drift=(round(d['drift'].get('final', d['drift'].get(64, {})).get('drift_exact', 0), 3)
                      if d['drift'] else None),
               gate=d['gate'])
    rows.append(row)

# markdown
lines = ['| arm | dev @0/16/32/48/64 | GSM | scaled | writes | calls | host-only | links-off | writer-exact |',
         '|---|---|---|---|---|---|---|---|---|']
for r in rows:
    lines.append(f"| {r['label']} | {r['curve']} | {r['gsm']} | {r['scaled']} | {r['writes']} | {r['calls']} | "
                 f"{r['hostonly']} | {r['linksoff']} | {r['drift']} |")
md = '\n'.join(lines)
print(md)
# link gate norms
print('\n### link gate L2 (final) by layer')
for r in rows:
    if r['gate']: print(f"- {r['label']}: {r['gate']}")
json.dump(out, open(os.path.join(RUNS, 'results.json'), 'w'), indent=1)
open(os.path.join(RUNS, 'results.md'), 'w').write(md)
print('\nwrote runs/results.json, runs/results.md')
