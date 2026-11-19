"""Summarize runs/<name>/steps.jsonl (copied to figdata/<name>_steps.jsonl): dev curve, per-tool rollout/dev stats,
LOO, IF silence, HF check, test. Prints markdown."""
import json, os, sys
FD = os.environ.get('LIB_FIGDATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'library', 'figdata'))
TOOLS = ['calc', 'track_gen', 'order_gen', 'interval_gen', 'turtle_gen', 'truth_gen', 'expr_gen', 'calc_gen', 'bool_gen', 'ifeval']
for name in sys.argv[1:] or ['LIB_s0', 'LIBscr_s0', 'LIB_s1']:
    p = os.path.join(FD, f'{name}_steps.jsonl')
    if not os.path.exists(p): continue
    R = [json.loads(l) for l in open(p)]
    print(f'\n### {name}')
    for r in R:
        if 'params' in r and isinstance(r['params'], dict): print('params', r['params'])
        if 'bitid_u0' in r: print('bit-identity u0', r['bitid_u0'])
        if 'hf_check' in r: print('hf_check', r['hf_check'])
        if 'deadline_check' in r: print('deadline', r['deadline_check'])
    dev = [r['dev'] for r in R if 'dev' in r]
    print('| u | dev | GSM | scaled | ' + ' | '.join(f'{t} burst/fire' for t in TOOLS if any(d.get(f'dev_t_{t}', {}).get('nb', 0) for d in dev)) + ' | IF latch |')
    act = [t for t in TOOLS if any(d.get(f'dev_t_{t}', {}).get('nb', 0) for d in dev)]
    for d in dev:
        cells = [f"{d[f'dev_t_{t}']['bursts']:.2f}/{100 * d[f'dev_t_{t}']['fire']:.0f}%" for t in act]
        print(f"| {d['u']} | {d['dev_acc']:.1f} | {d['dev_gsm']:.1f} | {d['dev_scaled']:.1f} | " + ' | '.join(cells) +
              f" | {d.get('dev_if_latch_any', float('nan')):.3f} |")
    ups = [r for r in R if 'u' in r and 'roll_s' in r]
    if ups:
        last = ups[-1]
        print('last update', last['u'], 'prompts', last.get('prompts'), 'upd_s', round(last.get('upd_s', 0), 1), 'roll_s', round(last['roll_s'], 1),
              'reward', round(last['reward'], 3))
        for t in TOOLS:
            seq = [(r['u'], r[f'roll_t_{t}']) for r in ups if f'roll_t_{t}' in r]
            nb = sum(x['nb'] for _, x in seq)
            if nb:
                pts = [(u, x) for u, x in seq if u in (1, 25, 50, 100, 150, 200, 250, 300) or u == last['u']]
                print(f'  roll {t}: total bursts {nb}; ' + '; '.join(f"u{u} b{x['bursts']:.2f} f{100 * x['fire']:.0f}%" for u, x in pts))
        print('  IF rollout latch any (mean over updates):', round(sum(r.get('roll_if_latch_any', 0) for r in ups) / len(ups), 4),
              'ans-mode', round(sum(r.get('roll_if_ans_mode', 0) for r in ups) / len(ups), 4),
              'write rows', round(sum(r.get('roll_if_write_rows', 0) for r in ups) / len(ups), 4))
    for r in R:
        if 'dev_loo' in r: l = r['dev_loo']; print('LOO', l['tool'], l.get('devloo_acc'), 'identical' if l.get('identical_by_construction') else '')
        for k in ('dev_toolsoff', 'dev_calconly', 'dev_hostonly', 'dev_toolsoff0'):
            if k in r: print(k, r[k].get('u'), {kk: round(v, 1) for kk, v in r[k].items() if kk.endswith('_acc') or kk.endswith('_scaled') or kk.endswith('_gsm')})
        if 'test' in r: print('TEST', {k: (v['acc'], v['n']) for k, v in r['test'].items()})
