"""print the tool x task-type matrix (triggers per answer / % fired) from a run's steps.jsonl: dev evals + rollout windows"""
import json, sys
R = [json.loads(l) for l in open(sys.argv[1])]
TYPES = ['gsm', 'bbh', 'ifeval', 'combo']
TOOLS = ['calc', 'track_gen', 'order_gen', 'interval_gen', 'turtle_gen', 'truth_gen', 'expr_gen', 'calc_gen', 'bool_gen', 'ifeval']
ON = {'gsm': {'calc', 'expr_gen', 'calc_gen'},
      'bbh': {'track_gen', 'order_gen', 'interval_gen', 'turtle_gen', 'truth_gen', 'expr_gen', 'calc_gen', 'bool_gen'},
      'ifeval': {'ifeval'},
      'combo': {'calc', 'expr_gen', 'calc_gen', 'track_gen', 'order_gen', 'interval_gen', 'turtle_gen', 'truth_gen', 'bool_gen', 'ifeval'}}
def offdiag(get):
    out = []
    for t in TYPES:
        for n in TOOLS:
            v = get(t, n)
            if v and v.get('nb') and n not in ON[t]: out.append(f"{n}@{t} {v['bursts']:.2f}/{100 * v['fire']:.0f}%")
    print('  OFF-DIAGONAL (type-level; calc_s4 counts as off on bbh): ' + ('; '.join(out) if out else 'none'))
def show(title, get):
    print(title); print('tool'.ljust(13) + ''.join(t.rjust(15) for t in TYPES))
    for n in TOOLS:
        cells = []
        for t in TYPES:
            v = get(t, n)
            cells.append('-'.rjust(15) if not v or not v.get('nb') else f"{v['bursts']:.2f}/{100 * v['fire']:.0f}%".rjust(15))
        print(n.ljust(13) + ''.join(cells))
    offdiag(get)
for r in R:
    for k in ('dev', 'dev_host_curve', 'dev_toolsoff0'):
        if k in r and k == 'dev':
            d = r['dev']; print(f"\n== greedy dev u{d['u']}: acc " + ' '.join(f"{t} {d.get('dev_' + t, float('nan')):.1f}" for t in TYPES) + f" | all {d['dev_acc']:.1f}")
            show('triggers per answer / % fired (greedy)', lambda t, n: d.get(f'dev_tools_{t}', {}).get(n))
            print('IF latch/write by type:', {t: (round(d.get(f'dev_if_{t}', {}).get('dev_if_latch_any', 0), 2), round(d.get(f'dev_if_{t}', {}).get('dev_if_write_rows', 0), 2)) for t in TYPES})
        if k in r and k == 'dev_host_curve':
            d = r[k]; print(f"   host-only u{d['u']}: " + ' '.join(f"{t} {d.get('devhost_' + t, float('nan')):.1f}" for t in TYPES) + f" | all {d['devhost_acc']:.1f}")
U = [r for r in R if 'roll_s' in r]
for lo, hi in ((1, 5), (20, 30), (45, 50), (95, 100), (290, 300)):
    W = [r for r in U if lo <= r['u'] <= hi]
    if not W: continue
    def agg(t, n):
        nb = sum(r.get(f'roll_tools_{t}', {}).get(n, {}).get('nb', 0) for r in W)
        if not nb: return None
        fi = sum(r[f'roll_tools_{t}'][n]['fire'] * r[f'roll_tools_{t}'][n]['nb'] for r in W if r.get(f'roll_tools_{t}', {}).get(n, {}).get('nb'))
        rows = sum(r[f'roll_tools_{t}'][n]['nb'] / max(r[f'roll_tools_{t}'][n]['bursts'], 1e-9) for r in W if r.get(f'roll_tools_{t}', {}).get(n, {}).get('nb'))
        return dict(nb=nb, fire=fi / nb, bursts=nb / max(rows, 1))
    print(f"\n== rollouts u{lo}-{hi}: reward " + ' '.join(f"{t} {sum(r.get('r_' + t, 0) for r in W) / len(W):.2f}" for t in TYPES))
    show('triggers per answer / % fired (sampled, floor 5-95%)', agg)
