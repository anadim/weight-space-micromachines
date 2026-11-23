"""(a) §3.3 size-ordered veto tally.
Rule (blog: "RL first vetoes most calls, then lets them back in by size: big numbers first, 1-2 digit last"):
a run is flagged if at any logged update admit(5+ digits) - admit(1-2 digits) > 0.3 while admit(1-2) < 0.5.
C36 (checks/calcimp/tt/res/train_steps.jsonl.gz, records of run <arm>_s<k>): train-rollout admit_d1_2 / admit_d5p per update (rows with 'calls';
rows with only 'dev' are dev-eval lines and carry no per-size admit;
updates with < 100 in-wire decisions (iw_n) are skipped). Also reported on a 10-update moving mean.
Early runs (rl/results/wiring/H100/*/steps.jsonl): greedy-dev dev_veto_by_digits {bucket: [n_calls, n_vetoed]},
5+ = 5-6 and 7+ pooled.
python3 checks/sourced/veto33.py   [--seeds N: C36 arm fI seeds 0..N-1; default all in the file]"""
import json, os, gzip, re, sys, collections, statistics as S
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.environ.get('WSM_ROOT') or os.path.join(HERE, '..', '..')
NSEED = int(sys.argv[sys.argv.index('--seeds') + 1]) if '--seeds' in sys.argv else None
C36 = collections.defaultdict(list)                 # run -> records, from the calcimp release copy (floats rounded to 5 s.f.)
for _l in gzip.open(f'{W}/checks/calcimp/tt/res/train_steps.jsonl.gz', 'rt'):
    _r = json.loads(_l); C36[_r['run']].append(_r)
THR, LO = 0.3, 0.5
MIN_IW = 100


def flag(series):
    """series: list of (u, a12, a5p). Returns (flag, first_u, last_u, max_diff, min_a12)."""
    hits = [u for u, a, b in series if a is not None and b is not None and b - a > THR and a < LO]
    md = max((b - a for u, a, b in series if a is not None and b is not None), default=None)
    ma = min((a for u, a, b in series if a is not None), default=None)
    return bool(hits), (hits[0] if hits else None), (hits[-1] if hits else None), md, ma


out = []
out.append('## C36 (train-rollout admit, per update)\n')
out.append('| arm | seed | max(a5p − a12) per update | min a12 | flagged | max(a5p − a12) 10-upd mean | flagged (smoothed) | a12 / a5p at u300 (u291–300 mean) |')
out.append('|---|---|---|---|---|---|---|---|')
tot = {}
for arm in ('fI', 'fI1'):  # fI0/fI01/fItok/fItok1 have no trainable in-wire (iw_n = 0, admit 1.0 throughout)
    for p in sorted((r for r in C36 if re.fullmatch(arm + r'_s\d+', r)), key=lambda s: int(s.split('_s')[-1])):
        k = int(p.split('_s')[-1])
        if NSEED is not None and arm == 'fI' and k >= NSEED: continue
        rs = C36[p]
        rs = [r for r in rs if 'calls' in r and 'u' in r]
        if not rs or rs[-1]['u'] < 300: continue
        # updates with < MIN_IW in-wire decisions (after a collapse to no calls) give 0/1 admit rates from a handful of calls
        ser = [(r['u'], r.get('admit_d1_2'), r.get('admit_d5p')) if r.get('iw_n', 0) >= MIN_IW else (r['u'], None, None) for r in rs]
        if not any(x[1] is not None for x in ser):
            tot.setdefault(arm + ' (no admit logged)', [0, 0, False]); tot[arm + ' (no admit logged)'][1] += 1; continue
        f, u0, u1, md, ma = flag(ser)
        sm = []
        for i in range(len(ser)):
            w = ser[max(0, i - 9):i + 1]
            a = [x[1] for x in w if x[1] is not None]; b = [x[2] for x in w if x[2] is not None]
            sm.append((ser[i][0], S.mean(a) if a else None, S.mean(b) if b else None))
        fs, _, _, mds, _ = flag(sm)
        end = [x for x in ser if x[0] > 290]
        m = lambda v: ('%.3f' % S.mean(v)) if v else 'n/a'
        e12 = m([x[1] for x in end if x[1] is not None]); e5 = m([x[2] for x in end if x[2] is not None])
        lastu = max((x[0] for x in ser if x[1] is not None), default=None)
        varies = len({round(x[1], 6) for x in ser if x[1] is not None}) > 1
        tot.setdefault(arm, [0, 0, varies])
        tot[arm][0] += f; tot[arm][1] += 1; tot[arm][2] = tot[arm][2] or varies
        out.append(f'| {arm} | {k} | {md:+.3f} | {ma:.3f} | {"YES u%d–%d" % (u0, u1) if f else "no"} | {mds:+.3f} | {"YES" if fs else "no"} | {e12} / {e5}{'' if e12 != 'n/a' else ' (no calls; last admit logged u%s)' % lastu} |')
out.append('')
for arm, (n, t, v) in tot.items():
    out.append(f'- {arm}: flagged {n}/{t}' + ('' if v else ' (admit constant: no trainable in-wire)'))

out.append('\n## Early correct-calculator in-wire runs (greedy-dev veto_by_digits)\n')
out.append('| run | dev checkpoints: u: admit 1-2 / 3-4 / 5+ | max(a5p − a12) | flagged | transient? |')
out.append('|---|---|---|---|---|')
early = {'CW s0': 'L300_CW_s0_kl001evalall0', 'CW s1': 'L300_CW_s1_kl001evalall0', 'CW-lean s0': 'L300_CWlean_s0_kl001evalall0',
         'CWinC s0': 'L300_CWinC_s0_evalall0', 'CWinC s1': 'L300_CWinC_s1_evalall0'}
for lab, d in early.items():
    p = f'{W}/rl/results/wiring/H100/{d}/steps.jsonl'
    ser, cells = [], []
    for l in open(p):
        r = json.loads(l)
        if 'dev' not in r or 'dev_veto_by_digits' not in r['dev']: continue
        v = r['dev']['dev_veto_by_digits']; u = r['dev']['u']
        ad = lambda ks: 1 - sum(v[k][1] for k in ks) / max(1, sum(v[k][0] for k in ks))
        a12, a34, a5 = ad(['1-2']), ad(['3-4']), ad(['5-6', '7+'])
        ser.append((u, a12, a5)); cells.append(f'u{u}: {a12*100:.0f}/{a34*100:.0f}/{a5*100:.0f}')
    f, u0, u1, md, _ = flag(ser)
    last = ser[-1]
    trans = (last[2] - last[1] <= THR or last[1] >= LO) if f else None
    out.append(f'| {lab} | {"; ".join(cells)} | {md:+.3f} | {"YES u%d–%d" % (u0, u1) if f else "no"} | {"-" if not f else ("transient (gone by u%d)" % last[0] if trans else "persists at u%d" % last[0])} |')
print('\n'.join(out))
