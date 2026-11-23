"""BBH-27 table + paired item-level bootstrap CIs for rl24/bbh (task [26]).
python analyze.py OUT.md --row "LABEL=eval_a.json[,eval_b.json,..]" ... --cmp "LABEL_X=LABEL_Y" (X minus Y) ...
A row with several files (seeds) is scored per item as the mean over its files (seed-averaged accuracy). mean27 = mean
over the 27 task accuracies; tool12 / other15 likewise. CIs: 10,000 bootstrap resamples of items, stratified by task
(items resampled within each task, the same resample for both sides of a paired difference), 95% percentile.
Also prints per-seed values and tok/q."""
import sys, json, gzip, argparse, numpy as np


def okf(x): return 1.0 if str(x) in ('True', '1', '1.0') else 0.0

TOOL12 = ['tracking_shuffled_objects_three_objects', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
          'logical_deduction_three_objects', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'temporal_sequences',
          'navigate', 'web_of_lies', 'multistep_arithmetic_two', 'boolean_expressions', 'object_counting']
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--row', action='append', default=[])
ap.add_argument('--cmp', action='append', default=[]); ap.add_argument('--json', default=''); ap.add_argument('--B', type=int, default=10000)
a = ap.parse_args()
rows, order, files = {}, [], {}
for spec in a.row:
    lab, fs = spec.split('=', 1); order.append(lab)
    rows[lab] = [json.load(gzip.open(f, 'rt') if f.endswith('.gz') else open(f)) for f in fs.split(',')]; files[lab] = fs.split(',')
ref = rows[order[0]][0]['rows']
tasks = sorted({r['task'] for r in ref}); tix = {t: [i for i, r in enumerate(ref) if r['task'] == t] for t in tasks}
for lab in order:
    for e in rows[lab]:
        assert len(e['rows']) == len(ref) and all(x['task'] == y['task'] and x['gold'] == y['gold'] for x, y in zip(e['rows'], ref)), lab
M = {lab: np.mean([[okf(r['ok']) for r in e['rows']] for e in rows[lab]], 0) for lab in order}
rng = np.random.default_rng(0)
boot = {t: rng.integers(0, len(ix), size=(a.B, len(ix))) for t, ix in tix.items()}


def stats(v):                                             # v: per-item vector -> (mean27, tool12, other15) point + boot
    acc = {t: v[ix].mean() * 100 for t, ix in tix.items()}
    bt = {t: v[np.array(ix)][boot[t]].mean(1) * 100 for t, ix in tix.items()}
    def agg(ts, d): return np.mean([d[t] for t in ts], 0)
    t12 = [t for t in tasks if t in TOOL12]; o15 = [t for t in tasks if t not in TOOL12]
    return {k: (agg(ts, acc), agg(ts, bt)) for k, ts in (('mean27', tasks), ('tool12', t12), ('other15', o15))}


def ci(b): return np.percentile(b, 2.5), np.percentile(b, 97.5)


L = ['| row | n seeds | BBH-27 mean [95% CI] | 12 tool tasks | 15 other tasks | tok/q | per seed mean27 |', '|---|---|---|---|---|---|---|']
for lab in order:
    s = stats(M[lab]); es = rows[lab]
    tq = np.mean([np.mean([float(r.get('ntok', 0)) for r in e['rows']]) for e in es])
    per_seed = ', '.join(f"{e['summary']['mean27']:.2f}" for e in es)
    cell = lambda k: f"{s[k][0]:.2f} [{ci(s[k][1])[0]:.2f}, {ci(s[k][1])[1]:.2f}]"
    L.append(f"| {lab} | {len(es)} | {cell('mean27')} | {cell('tool12')} | {cell('other15')} | {tq:.0f} | {per_seed} |")
L += ['', '| paired difference | BBH-27 mean [95% CI] | 12 tool tasks | 15 other tasks |', '|---|---|---|---|']
for c in a.cmp:
    x, y = c.split('=', 1) if '=' in c else c.split('-', 1)
    s = stats(M[x] - M[y])
    cell = lambda k: f"{s[k][0]:+.2f} [{ci(s[k][1])[0]:+.2f}, {ci(s[k][1])[1]:+.2f}]"
    L.append(f"| {x} − {y} | {cell('mean27')} | {cell('tool12')} | {cell('other15')} |")
L += ['', 'per task (seed-averaged):', '', '| task | ' + ' | '.join(order) + ' |', '|---|' + '---|' * len(order)]
for t in tasks:
    L.append(f"| {t}{' (tool)' if t in TOOL12 else ''} | " + ' | '.join(f"{M[l][tix[t]].mean() * 100:.1f}" for l in order) + ' |')
open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

NOTES = ('BBH-27 (6,511 items), BBH 3-shot CoT harness of bbh_generic/callgate (cg_gate_gen), greedy, max_new 600, fast engine. '
         'Rows 1-4 host = ft_int_v2 = Qwen3-0.6B-Base + \u00a76.1 tool-call FT (teams\' own 432 rows; whole-host weight change, '
         'shared by rows 1-4, [29]); row 0 = plain Base, row 2\' = RL from plain Base. Implants = 8 compiled BBH tools + call '
         'gate refit on ft_int_v2. RL: GRPO 300 u, 64x8, lr 1e-6, T 1, Dr.GRPO, kl 0, synthetic look-alike train data (seed 200). '
         'CI = paired item bootstrap (10k, stratified by task) over seed-averaged items.')
if a.json:                                                # rl24 [26] figure schema (rows.json); 'ctl ...' rows skipped
    import os, re
    def cfg(f): return re.sub(r'^eval_|\.json(\.gz)?$', '', os.path.basename(f))
    sets = {}
    for key, lab_set in (('mean27', 'BBH 27-task mean'), ('tool12', 'BBH 12 tool tasks')):
        R = []
        for lab in order:
            rid, _, name = lab.partition(' ')
            if rid == 'ctl': continue
            s_ = stats(M[lab])[key]; lo, hi = ci(s_[1])
            per = [stats(np.array([okf(r['ok']) for r in e['rows']]))[key][0] for e in rows[lab]]
            tq = np.mean([np.mean([float(r.get('ntok', 0)) for r in e['rows']]) for e in rows[lab]])
            R.append(dict(row=rid, label=name, configs=[cfg(f) for f in files[lab]], mean=round(float(s_[0]), 2),
                          seeds=[round(float(x), 2) for x in per], ci95=[round(float(lo), 2), round(float(hi), 2)], tokq=round(float(tq), 1)))
        sets[key] = dict(label=lab_set, rows=R)
    json.dump(dict(metric='BBH-27 mean acc (%)', sets=sets, notes=NOTES), open(a.json, 'w'), indent=1)
    print('wrote', a.json)
