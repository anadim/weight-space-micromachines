"""Rebuild the README tables from the result files in results/ (no model needed).

python results/summarize.py            -> prints the BBH 12-task table, the 27-task means, GSM and gate numbers
Also importable: figures/make_figs.py takes every plotted number from here.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
load = lambda p: json.load(open(os.path.join(HERE, p)))

TASKS12 = [  # (BBH task, label, implant result file)
    ('multistep_arithmetic_two', 'Multistep arithmetic', 'bbh/qwen3_implants_arith.json'),
    ('navigate', 'Navigate', 'bbh/qwen3_implants_navigate.json'),
    ('boolean_expressions', 'Boolean expressions', 'bbh/qwen3_implants_lies_bool_navv1.json'),
    ('web_of_lies', 'Web of lies', 'bbh/qwen3_implants_lies_bool_navv1.json'),
    ('temporal_sequences', 'Temporal sequences', 'bbh/qwen3_implants_temporal_dyck.json'),
    ('tracking_shuffled_objects_three_objects', 'Tracking, 3 objects', 'bbh/qwen3_implants_tracking.json'),
    ('tracking_shuffled_objects_five_objects', 'Tracking, 5 objects', 'bbh/qwen3_implants_tracking.json'),
    ('tracking_shuffled_objects_seven_objects', 'Tracking, 7 objects', 'bbh/qwen3_implants_tracking.json'),
    ('logical_deduction_three_objects', 'Deduction, 3 objects', 'bbh/qwen3_implants_deduction.json'),
    ('logical_deduction_five_objects', 'Deduction, 5 objects', 'bbh/qwen3_implants_deduction.json'),
    ('logical_deduction_seven_objects', 'Deduction, 7 objects', 'bbh/qwen3_implants_deduction.json'),
    ('object_counting', 'Object counting', 'bbh/qwen3_implants_arith.json'),
]
# which implant(s) serve each task (for the "add one implant at a time" view)
IMPLANT_OF = {'multistep_arithmetic_two': 'calculator + sum + expression', 'object_counting': 'sum', 'web_of_lies': 'truth tracker',
              'navigate': 'navigator', 'boolean_expressions': 'boolean', 'temporal_sequences': 'schedule reader',
              **{t: 'swap tracker' for t, *_ in TASKS12 if t.startswith('tracking')},
              **{t: 'deduction engine' for t, *_ in TASKS12 if t.startswith('logical')}}


def per_task(path):
    return load(path)['per_task']


# The first Qwen2.5-7B run returned empty outputs on these five tasks (its stop string was its first output); only for
# them the min_tokens-4 rerun replaces the first run. (The rerun also covered geometric_shapes, formal_fallacies and
# tracking-7, which were valid in the first run and are kept from it.)
EMPTY_7B = ('boolean_expressions', 'multistep_arithmetic_two', 'object_counting', 'dyck_languages', 'word_sorting')


def qwen25_7b():
    d = dict(per_task('bbh/qwen25_7b_vllm.json')); fix = per_task('bbh/qwen25_7b_vllm_mintok4.json')
    d.update({t: fix[t] for t in EMPTY_7B}); return d


def log_scores(name):
    """Per-task accuracy parsed from a log snapshot in results/bbh/."""
    p = os.path.join(HERE, 'bbh', name)
    if not os.path.exists(p): return {}
    return {m.group(1): float(m.group(2)) for m in re.finditer(r'^(\w+)\s+([\d.]+) \(n=\d+\)', open(p).read(), re.M)}


def base_hf_log():
    """Same-harness (HF loop, no implant) Qwen3-0.6B-Base per-task accuracy, all 27 tasks."""
    return per_task('bbh/qwen3_base_hf.json')


def bbh_table():
    base, b7, b14, hf = per_task('bbh/qwen3_base_vllm.json'), qwen25_7b(), per_task('bbh/qwen25_14b_vllm.json'), base_hf_log()
    rows = []
    for t, label, f in TASKS12:
        rows.append(dict(task=t, label=label, base=base[t], base_hf=hf.get(t), implants=per_task(f)[t], q7b=b7[t], q14b=b14[t], file=f))
    mean = lambda k: sum(r[k] for r in rows) / len(rows) if all(r[k] is not None for r in rows) else None
    return rows, {k: mean(k) for k in ('base', 'base_hf', 'implants', 'q7b', 'q14b')}


def bbh27():
    base, b7, b14 = per_task('bbh/qwen3_base_vllm.json'), qwen25_7b(), per_task('bbh/qwen25_14b_vllm.json')
    imp = dict(base)
    for t, _, f in TASKS12: imp[t] = per_task(f)[t]
    hf = base_hf_log(); imp_hf = dict(hf)
    for t, _, f in TASKS12: imp_hf[t] = per_task(f)[t]
    m = lambda d: sum(d.values()) / len(d)
    return {'base': m(base), 'implants_on_12_base_elsewhere': m(imp), 'base_hf': m(hf), 'implants_on_12_hf_base_elsewhere': m(imp_hf),
            'q7b': m(b7), 'q14b': m(b14), 'n': len(base)}


def adding_curve():
    """Mean over the 12 tasks when the implants are added one group at a time (each task's score switches from the
    base run to its implant run when its implant is added). Computed from the per-task files, not a separate run."""
    rows, _ = bbh_table()
    order = ['sum', 'calculator + sum + expression', 'truth tracker', 'navigator', 'boolean', 'schedule reader', 'swap tracker', 'deduction engine']
    cur = {r['task']: r['base'] for r in rows}; out = [('none', sum(cur.values()) / 12)]
    for imp in order:
        for r in rows:
            if IMPLANT_OF[r['task']] == imp: cur[r['task']] = r['implants']
        out.append((imp, sum(cur.values()) / 12))
    return out


def gsm():
    out = {}
    for s in ('test', 'hard'):
        for m in ('plain', 'calc'):
            d = load(f'gsm/qwen3_{s}_{m}_rethink.json'); n = len(d['rows'])
            out[s, m] = (100 * sum(r['first_ok'] for r in d['rows']) / n, 100 * sum(r['ok'] for r in d['rows']) / n, n)
    return out


def gate():
    """Parse results/gate/cost_curve.log -> {(src, split): {'never':(acc,tok), 'always':..., 'oracle':..., name: [(fire, acc, tok), ...]}}"""
    cur, out = None, {}
    for line in open(os.path.join(HERE, 'gate/cost_curve.log')):
        m = re.match(r'== (\w+) -> \w+ \| (\w+)', line)
        if m: cur = out.setdefault((m.group(1), m.group(2)), {}); continue
        m = re.match(r'\s+(never|always|oracle)\s*: acc\s*([\d.]+)% \| extra tok/problem\s*(\d+)', line)
        if m: cur[m.group(1)] = (float(m.group(2)), int(m.group(3))); continue
        m = re.match(r'\s+(.+?)\s+fire~\s*(\d+)% \(actual\s*(\d+)%\): acc\s*([\d.]+)% \| extra tok/problem\s*(\d+)', line)
        if m: cur.setdefault(m.group(1), []).append((int(m.group(3)), float(m.group(4)), int(m.group(5))))
    return out


if __name__ == '__main__':
    rows, mean = bbh_table()
    print('BBH, 12 tasks (official 3-shot CoT, greedy)')
    f = lambda v, w=8, p=1: f'{v:{w}.{p}f}' if v is not None else ' ' * (w - 3) + 'n/a'
    print(f"{'task':24s} {'Q3 vLLM':>8s} {'Q3 HF':>8s} {'+implants':>9s} {'Q2.5-7B':>8s} {'Q2.5-14B':>8s}")
    for r in rows: print(f"{r['label']:24s} {f(r['base'])} {f(r['base_hf'])} {f(r['implants'], 9)} {f(r['q7b'])} {f(r['q14b'])}")
    print(f"{'MEAN':24s} {f(mean['base'], 8, 2)} {f(mean['base_hf'], 8, 2)} {f(mean['implants'], 9, 2)} {f(mean['q7b'], 8, 2)} {f(mean['q14b'], 8, 2)}")
    print('\nBBH, all 27 tasks (implant runs on the 12, base elsewhere):', {k: round(v, 2) for k, v in bbh27().items()})
    a9, hf = log_scores('all_nine_mech.log'), base_hf_log()
    print('\nall nine attached at once (log snapshot) vs per-group implant runs:',
          {t: (a9[t], next(r['implants'] for r in rows if r['task'] == t)) for t in a9 if any(r['task'] == t for r in rows)})
    o9 = log_scores('all_nine_other.log')
    print('tasks without an implant, all nine attached vs no implant (HF loop):', {t: (o9[t], hf.get(t)) for t in o9})
    print('\nadd one implant at a time (12-task mean):', [(k, round(v, 2)) for k, v in adding_curve()])
    print('\nGSM (first pass -> after rethink):')
    for (s, m), (f, r, n) in gsm().items(): print(f'  {s:5s} {m:6s} {f:6.2f} -> {r:6.2f}  (n={n})')
    g = gate()[('base', 'test')]
    print('\ngate (GSM8K test, vLLM):', {k: v for k, v in g.items() if k in ('never', 'always', 'oracle')}, '| probe L28:', g['probe L28'])
    d = load('gate/detector_summary.json')
    print('detector test AUROC:', {src: round(d[src]['probe L28']['test_auc'], 3) for src in d})
