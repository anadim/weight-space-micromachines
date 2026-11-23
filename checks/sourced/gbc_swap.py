"""(b) §3.4 BBH "training the BBH implants never helped on BBH (one cost 10 points)".
Swap eval in the mixed_test record of each trainable-implant mixed GSM8K+BBH run (u50 final):
  full                    = trained host + trained implants
  trained_host_orig_tools = trained host + original (untrained) implants
  orig_host_trained_tools = original host + trained implants
Trained-implant effect on the trained host = full - trained_host_orig_tools. test split: BBH n=212, GSM n=188.
Per-item records were not saved, so only an unpaired (independent-binomial) 95% half-width is given; a paired CI is narrower.
python3 checks/sourced/gbc_swap.py   (input: rl/figures/v3/data, shipped)"""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get('WSM_ROOT') or os.path.join(HERE, '..', '..')
W = f'{ROOT}/rl/figures/v3/data'
NB, NG = 212, 188
print('| run | BBH trained impl. | BBH orig impl. | Δ BBH (count) | ±unpaired 95% | GSM trained impl. | GSM orig impl. | Δ GSM (count) | ±unpaired 95% | BBH orig host + trained impl. |')
print('|---|---|---|---|---|---|---|---|---|---|')
for r in ('gbC2_s0', 'gbC3_s0', 'gbC3_s1', 'gbC3_s2'):
    m = [json.loads(l) for l in open(f'{W}/{r}_steps.jsonl')]
    m = [x['mixed_test'] for x in m if 'mixed_test' in x][-1]
    f, o, oh = m['full'], m['trained_host_orig_tools'], m['orig_host_trained_tools']
    row = [r]
    for k, n in (('test_bbh', NB), ('test_gsm', NG)):
        a, b = f[k] / 100, o[k] / 100
        ca, cb = round(a * n), round(b * n)
        assert abs(ca / n - a) < 1e-9 and abs(cb / n - b) < 1e-9
        hw = 1.96 * math.sqrt(a * (1 - a) / n + b * (1 - b) / n) * 100
        row += [f'{f[k]:.1f}', f'{o[k]:.1f}', f'{(a - b) * 100:+.1f} ({ca - cb:+d}/{n})', f'±{hw:.1f}']
    row.append(f'{oh["test_bbh"]:.1f}')
    print('| ' + ' | '.join(row) + ' |')
