"""Aggregate BBH-27 call-gate results. Baseline = merged published v2+tools transcripts (out/b27_v2ft_tools_*.json),
new arms = gate.json / breaker.json (and optional none.json). Reports 27/12/15 means, 15-no-tool call rate,
no-answer rate (pred empty), mean <tool_call>/output, blocked rate.
python cg_report.py BASE_DIR "v2 + GATE=gate_even.json,gate_odd.json" "v2 + BREAKER=breaker.json" ...
Shipped results:  python cg_report.py ../../integrate/results "v2 + GATE=../results/bbh_gate_even.json,../results/bbh_gate_odd.json" \
                    "v2 + BREAKER=../results/bbh_breaker.json,../results/bbh_breaker_b.json"
"""
import json, glob, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from reduce_bbh import load_json            # plain or gzipped; the shipped results are reduced (no text: n_calls instead of out)


def _open(fn):
    for p in (fn, fn + '.gz', fn + '.partial', fn.replace('.json', '.partial.json') + '.gz'):
        if os.path.exists(p): return load_json(p)
    raise FileNotFoundError(fn)
TOOL = {'boolean_expressions', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects',
        'multistep_arithmetic_two', 'navigate', 'object_counting', 'temporal_sequences', 'tracking_shuffled_objects_five_objects',
        'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies'}
ALL27 = TOOL | {'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes',
        'hyperbaton', 'movie_recommendation', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names',
        'salient_translation_error_detection', 'snarks', 'sports_understanding', 'word_sorting'}
NOTOOL = ALL27 - TOOL


def merge_published(base_dir):
    best = {}
    for fn in glob.glob(f'{base_dir}/b27_v2ft_tools_*.json') + glob.glob(f'{base_dir}/b27_v2ft_tools_*.json.gz'):
        try: d = load_json(fn)
        except Exception: continue
        per = {}
        for r in d.get('rows', []): per.setdefault(r['task'], []).append(r)
        for t, rs in per.items():
            if t not in best or len(rs) > len(best[t]): best[t] = rs
    return best


def load_arm(fns):
    per = {}
    for fn in fns.split(','):
        for r in _open(fn)['rows']: per.setdefault(r['task'], []).append(r)
    return per


def stats(per):
    acc = {t: 100 * sum(r['ok'] for r in rs) / len(rs) for t, rs in per.items()}
    m = lambda S: sum(acc[t] for t in S if t in acc) / len([t for t in S if t in acc])
    nt = [r for t in NOTOOL if t in per for r in per[t]]
    nc = lambda r: r['out'].count('<tool_call>') if 'out' in r else r['n_calls']
    def callrate(rows): return 100 * sum(nc(r) > 0 for r in rows) / len(rows)
    def noans(rows): return 100 * sum(r['pred'].strip() == '' for r in rows) / len(rows)
    def meancalls(rows): return sum(nc(r) for r in rows) / len(rows)
    blk = None
    if nt and 'blocked' in nt[0]: blk = 100 * sum(r.get('blocked', False) for r in nt) / len(nt)
    return acc, {'m27': m(ALL27), 'm12': m(TOOL), 'm15': m(NOTOOL),
                 'call15': callrate(nt), 'noans15': noans(nt), 'calls15': meancalls(nt), 'blk15': blk}


base_dir = sys.argv[1]
arms = [('v2+tools (baseline)', merge_published(base_dir))]
for spec in sys.argv[2:]:
    name, fns = spec.split('=', 1); arms.append((name, load_arm(fns)))

accs = {}
print('\n| arm | 27 mean | 12 tool | 15 no-tool | 15 call% | 15 no-answer% | 15 calls/out | 15 blk% |')
print('|---|---|---|---|---|---|---|---|')
for name, per in arms:
    a, s = stats(per); accs[name] = a
    blk = f'{s["blk15"]:.1f}' if s['blk15'] is not None else '-'
    print(f'| {name} | {s["m27"]:.2f} | {s["m12"]:.2f} | {s["m15"]:.2f} | {s["call15"]:.1f} | {s["noans15"]:.1f} | {s["calls15"]:.1f} | {blk} |')
print('\nReference (15 no-tool means): base(no tools) 39.15 ; FT tools-off 30.01 ; no-answer tools-off 5.6%, tools-on 35.7%\n')

print('| task | tool | baseline | ' + ' | '.join(n for n, _ in arms[1:]) + ' |')
print('|---|---|---|' + '---|' * (len(arms) - 1))
for t in sorted(ALL27):
    cells = [f'{accs[n][t]:.1f}' if t in accs[n] else '-' for n, _ in arms]
    print(f'| {t} | {"Y" if t in TOOL else ""} | ' + ' | '.join(cells) + ' |')
