"""Build the §2 comparison table from fetched files (no models run here).
Inputs: fetched/summ_<pod>.txt (summ_runs.py --json output), fetched/final/<name>.json (final_eval outputs).
python compile_abl.py > ../results/gsm_grpo/summary/table.md   (inputs: results/gsm_grpo/summary/summ_A100.txt, results/gsm_grpo/A100/final/)"""
import json, glob, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get('RL_RESULTS', os.path.join(HERE, '..', 'results', 'gsm_grpo'))   # released result files
runs = {}
for f in glob.glob(f'{R}/summary/summ_A100.txt'):
    pod = os.path.basename(f)[5:-4]
    for l in open(f):
        if l.startswith('JSON '):
            d = json.loads(l[5:]); d['gpu'] = pod; runs[d['run']] = d
fin = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f'{R}/A100/final/*.json')}


def dev_at(d, u, key='dev_acc', src='dev'):
    for x in d[src]:
        if x['u'] == u: return x.get(key)
    return None


def fmt(v, nd=1):
    return '—' if v is None else f'{v:.{nd}f}'


print('| run | GPU | dev @0 / 32 / 64 (implant on) | GSM / scaled half @64 | implant-off dev @0 / 64 | writes / calls per answer @0 -> @64 |'
      ' TEST GSM8K on / off | GSM-Hard on / off (201+ on / off) |')
print('|---|---|---|---|---|---|---|---|')
order = sorted(runs, key=lambda r: (r.split('_')[1], {'a1': 0, 'a2': 1, 'aR': 2}.get(r.split('_')[0], 9), r))
for r in order:
    d = runs[r]
    d0, d32, d64 = dev_at(d, 0), dev_at(d, 32), dev_at(d, 64)
    g64, s64 = dev_at(d, 64, 'dev_gsm'), dev_at(d, 64, 'dev_scaled')
    if r.startswith('a1'):
        h0, h64 = d0, d64
    else:
        h0 = dev_at(d, 0, 'devhost_acc', 'host'); h64 = dev_at(d, 64, 'devhost_acc', 'host')
    w0, w64 = dev_at(d, 0, 'dev_writes'), dev_at(d, 64, 'dev_writes'); c0, c64 = dev_at(d, 0, 'dev_calls'), dev_at(d, 64, 'dev_calls')
    fe = fin.get(r, {})
    t_on = fe.get('test_on', {}).get('acc'); t_off = fe.get('test_host', {}).get('acc')
    h_on = fe.get('hard_on', {}); h_off = fe.get('hard_host', {})
    print(f"| {r} | {d['gpu']} | {fmt(d0)} / {fmt(d32)} / {fmt(d64)} | {fmt(g64)} / {fmt(s64)} | {fmt(h0)} / {fmt(h64)} | "
          f"{fmt(w0)} / {fmt(c0, 2)} -> {fmt(w64)} / {fmt(c64, 2)} | {fmt(t_on)} / {fmt(t_off)} | "
          f"{fmt(h_on.get('acc'))} / {fmt(h_off.get('acc'))} ({fmt(h_on.get('acc_201plus'))} / {fmt(h_off.get('acc_201plus'))}) |")
print()
print('Base model (no RL), same harness, per GPU:')
print('| model | GPU | TEST GSM8K | GSM-Hard all (201+) | writes per question (test) | calc bursts exact / wrong (test) |')
print('|---|---|---|---|---|---|')
for k in sorted(fin):
    if not k.startswith('base_'): continue
    fe = fin[k]; gpu = k.split('_')[1]; kind = k.split('_')[2]
    if kind == 'calc':
        print(f"| base, no implant | {gpu} | {fmt(fe['test_host']['acc'])} | {fmt(fe['hard_host']['acc'])} ({fmt(fe['hard_host']['acc_201plus'])}) | 0 | — |")
    on = fe['test_on']
    print(f"| base + {'calc_s4' if kind == 'calc' else 'R (scrambled digits)'} | {gpu} | {fmt(on['acc'])} | {fmt(fe['hard_on']['acc'])} ({fmt(fe['hard_on']['acc_201plus'])}) | "
          f"{fmt(on.get('writes_per_q'))} | {on.get('calc_bursts_exact')} / {on.get('calc_bursts_wrong')} |")
