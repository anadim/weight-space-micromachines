"""seeds3 summary for items (b) m4 noisy-SFT and (c) m2 scrambled library (mixed task, u150): the tables of RESULTS.md (b)/(c).
usage: cd checks/seeds3 && python3 summarize.py     (CPU, < 1 s)
Seed 0 = the original runs in $WSM_ROOT/rl/results/library/figdata (mix_m4_mixLIBnoisy_*, mix_m2_mixLIBscr_*);
seeds 1, 2 = res/<run>/{steps.jsonl,test_eval.json} (S3_RES overrides res/, e.g. S3_RES=runs to read new runs)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(os.environ.get('WSM_ROOT', os.path.join(H, '..', '..')), 'rl', 'results', 'library', 'figdata')
RES = os.environ.get('S3_RES', os.path.join(H, 'res'))
TYPES = ['gsm', 'bbh', 'ifeval', 'combo']
TOOLS = ['calc', 'track_gen', 'order_gen', 'interval_gen', 'turtle_gen', 'truth_gen', 'expr_gen', 'calc_gen', 'bool_gen', 'ifeval']
ON = {'gsm': {'calc', 'expr_gen', 'calc_gen'},       # same "own type" map as library/mixed/matrix.py
      'bbh': {'track_gen', 'order_gen', 'interval_gen', 'turtle_gen', 'truth_gen', 'expr_gen', 'calc_gen', 'bool_gen'},
      'ifeval': {'ifeval'}, 'combo': set(TOOLS)}
RUNS = {'m4': [('s0 (orig, H100)', 'mix_m4_mixLIBnoisy'), ('s1', 'm4_s1'), ('s2', 'm4_s2')],
        'm2': [('s0 (orig, A100)', 'mix_m2_mixLIBscr'), ('s1', 'm2_s1'), ('s2', 'm2_s2')]}


def load(tag):
    if tag.startswith('mix_'):
        st, te = f'{FD}/{tag}_steps.jsonl', f'{FD}/{tag}_test.json'
    else:
        st, te = f'{RES}/{tag}/steps.jsonl', f'{RES}/{tag}/test_eval.json'
    devs = []
    if os.path.exists(st):
        for l in open(st):
            r = json.loads(l)
            if 'dev' in r and isinstance(r['dev'], dict) and 'dev_acc' in r['dev']: devs.append(r['dev'])
    test = json.load(open(te)) if os.path.exists(te) else None
    return devs, test


def fired(d, pre, t, n):
    v = d.get(f'{pre}_tools_{t}', {}).get(n)
    return 0.0 if not v else v['bursts'] * v['fire']


def offdiag(d, pre):
    return {f'{n}@{t}': round(fired(d, pre, t, n), 3) for t in ('gsm', 'bbh', 'ifeval') for n in TOOLS
            if n not in ON[t] and fired(d, pre, t, n) > 0}


out = []
P = out.append
P('## (b) m4: noisy-routing SFT host + library, mixed task, 150 u. Off-type FIRED calls per answer (triggers x in-wire pass), greedy\n')
for lab, tag in RUNS['m4']:
    devs, test = load(tag)
    if not devs: P(f'- {lab}: no data yet'); continue
    d0, dl = devs[0], devs[-1]
    keys = sorted(set(offdiag(d0, 'dev')) | set(offdiag(dl, 'dev')) | (set(offdiag(test['full'], 'test')) if test else set()))
    P(f'- **{lab}**: dev all u{d0.get("u", 0)} {d0["dev_acc"]:.1f} -> u{dl.get("u")} {dl["dev_acc"]:.1f}'
      + (f'; test 400 tools on {test["full"]["test_acc"]:.1f} (gsm {test["full"]["test_gsm"]:.1f} / bbh {test["full"]["test_bbh"]:.1f} / ifeval {test["full"]["test_ifeval"]:.1f} / combo {test["full"]["test_combo"]:.1f}), tools off {test["toolsoff"]["test_acc"]:.1f}, host only {test["hostonly"]["test_acc"]:.1f}' if test else '; test pending'))
    P('  | off-type call | dev u0 | dev u%s | test u150 |' % dl.get('u'))
    P('  |---|---|---|---|')
    for k in keys:
        n, t = k.split('@')
        P(f'  | {k} | {fired(d0, "dev", t, n):.2f} | {fired(dl, "dev", t, n):.2f} | ' + (f'{fired(test["full"], "test", t, n):.2f} |' if test else '- |'))
    if test:
        f = test['full']
        for n in ('calc', 'expr_gen'):
            lk = test.get(f'loo_{n}')
            if lk: P(f'  - LOO {n} (test): ifeval {f["test_ifeval"]:.1f} -> {lk["test_ifeval"]:.1f}, bbh {f["test_bbh"]:.1f} -> {lk["test_bbh"]:.1f}')
    mp = f'{RES}/MF_{tag}/misfire_probe.json'
    if os.path.exists(mp):
        m = json.load(open(mp))
        P('  - misfire probe (u150, 64 train prompts x G8 per type): ' + '; '.join(
            f'{k}: {v["rows_with_call"]} rows, reward {v["mean_reward_with"]:.2f} vs {v["mean_reward_without"]:.2f}, within-group {v["within_group_diff"]:+.2f} ({v["groups_mixed"]} groups)'
            for k, v in m.items()))
P('\ncalculator FIRED calls per answer on greedy dev, by update (the (b) table of RESULTS.md):')
for lab, tag in RUNS['m4']:
    devs, _ = load(tag)
    for t in ('bbh', 'ifeval'):
        if devs: P(f'- m4 {lab} calc@{t}: ' + ' -> '.join(f'u{d.get("u", 0)} {fired(d, "dev", t, "calc"):.2f}' for d in devs))
P('\n## (c) m2: scrambled library (all but calc_s4), mixed task, 150 u. Greedy in-wire pass rate on test 400 at u150\n')
P('| seed | test all / tools off | ' + ' | '.join(TOOLS) + ' | closed / fire |')
P('|---|---|' + '---|' * len(TOOLS) + '---|')
for lab, tag in RUNS['m2']:
    devs, test = load(tag)
    if not test: P(f'| {lab} | pending |' + ' |' * len(TOOLS) + ' |'); continue
    f = test['full']; cells = []; nf = nc = 0; closed = []
    for n in TOOLS:
        v = f.get(f'test_t_{n}')
        if not v or not v['nb']: cells.append('-'); continue
        nf += 1; cells.append(f'{100 * v["fire"]:.0f}% ({v["nb"]})')
        if v['fire'] < 0.05: nc += 1; closed.append(n)
    P(f'| {lab} | {f["test_acc"]:.1f} / {test["toolsoff"]["test_acc"]:.1f} | ' + ' | '.join(cells) + f' | {nc} / {nf} ({", ".join(closed)}) |')
P('\ncells: % of triggers the in-wire lets through (number of triggers on the 400 test items); "-" = never triggers.')
for lab, tag in RUNS['m2']:
    devs, _ = load(tag)
    if devs: P(f'- m2 {lab} dev: ' + ', '.join(f'u{d.get("u")} {d["dev_acc"]:.1f}' for d in devs))
print('\n'.join(out))
