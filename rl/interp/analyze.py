"""Interpretability analysis of dev-800 greedy traces (text analysis only; traces come from dump_traces.py on A100 machine).
python analyze.py traces/  -> prints tables, writes stats.json"""
import sys, json, re, statistics as st
from collections import Counter, defaultdict
from eqparse import equations

import os
D = os.environ.get('TRACES', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traces'))
CFG = {'base_off': 'base, no implant', 'base_on': 'base + calc', 'baseR_on': 'base + deranged calc',
       'a1_off': 'arm 1 RL, no implant', 'a1_on': 'arm 1 RL + calc (never trained with it)',
       'a2_on': 'arm 2 RL + calc', 'a2_off': 'arm 2 RL, implant off',
       'a1R_on': 'arm 1 RL + deranged calc', 'a2R_on': 'arm 2 RL + deranged calc'}
T = {}
for k in CFG:
    try: T[k] = [json.loads(l) for l in open(f'{D}/{k}.jsonl')]
    except FileNotFoundError: pass


def bursts(w):
    out, i = [], 0
    while i < len(w):
        if w[i]:
            j = i
            while j < len(w) and w[j]: j += 1
            out.append((i, j)); i = j
        else: i += 1
    return out


def annotate(r):
    """Per-trace features."""
    txt = r['text']
    body = txt.split('####')[0]
    eqs = equations(body)
    offs, c = [], 0
    for t in r['toks']: offs.append(c); c += len(t)
    calls = []
    for s, e in bursts(r['wflag']):
        p = offs[s]
        if p >= len(txt): continue                       # after the stop
        cand = [q for q in eqs if q['eq_pos'] < p and p <= q['rhs_end'] + 1]
        written = ''.join(r['toks'][s:e])
        calls.append(dict(start=p, eq=cand[-1] if cand else None, written=written))
    called_eq = {id(c['eq']) for c in calls if c['eq'] is not None}
    for q in eqs: q['called'] = id(q) in called_eq
    lines = [l for l in body.split('\n') if l.strip()]
    vwords = len(re.findall(r'\b(times|divided by|plus|minus|multiplied by|multiply|half of|twice|double)\b', body, re.I))
    return dict(eqs=eqs, calls=calls, nlines=len(lines), ntok=len(r['ids']), trunc=int(not r['fmt']),
                hit_max=int(len(r['ids']) >= 320), vwords=vwords, nwrong=sum(not q['ok'] for q in eqs))


A = {k: [annotate(r) for r in v] for k, v in T.items()}
N = 800
kinds = [r['kind'] for r in next(iter(T.values()))]
S = {}


def pct(x, n): return 100.0 * x / max(1, n)


def acc(k, kind=None):
    rs = [r for r in T[k] if kind is None or r['kind'] == kind]
    return pct(sum(r['ok'] for r in rs), len(rs))


print('## accuracy (this re-run, A100 machine A100, B_max 64, greedy)')
print('| config | all | GSM | scaled | fmt | tokens/ans | lines/ans | eqs/ans | wrong eqs/ans | eq error rate | calls/ans |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for k in CFG:
    if k not in T: continue
    a = A[k]; ne = sum(len(x['eqs']) for x in a); nw = sum(x['nwrong'] for x in a)
    row = dict(acc=acc(k), gsm=acc(k, 'gsm'), scaled=acc(k, 'scaled'), fmt=pct(sum(r['fmt'] for r in T[k]), N),
               tok=st.mean(x['ntok'] for x in a), lines=st.mean(x['nlines'] for x in a), eqs=ne / N, wrong=nw / N,
               err=pct(nw, ne), calls=sum(len(x['calls']) for x in a) / N, hitmax=pct(sum(x['hit_max'] for x in a), N),
               vwords=st.mean(x['vwords'] for x in a))
    S[k] = row
    print(f"| {CFG[k]} | {row['acc']:.1f} | {row['gsm']:.1f} | {row['scaled']:.1f} | {row['fmt']:.1f} | {row['tok']:.0f} | "
          f"{row['lines']:.2f} | {row['eqs']:.2f} | {row['wrong']:.2f} | {row['err']:.1f}% | {row['calls']:.2f} |")


def flips(k0, k1):
    res = {}
    for kind in ('gsm', 'scaled'):
        idx = [i for i in range(N) if kinds[i] == kind]
        up = sum(1 for i in idx if not T[k0][i]['ok'] and T[k1][i]['ok'])
        dn = sum(1 for i in idx if T[k0][i]['ok'] and not T[k1][i]['ok'])
        res[kind] = (up, dn)
    return res


print('\n## flips (right<-wrong / wrong<-right), per 400')
S['flips'] = {}
for k0, k1 in [('base_off', 'a1_off'), ('base_on', 'a2_on'), ('base_off', 'a2_on'), ('base_off', 'a2_off'), ('a1_off', 'a2_on'),
               ('a1_off', 'a2_off'), ('a2_off', 'a2_on'), ('base_off', 'base_on'), ('a1_off', 'a1_on')]:
    if k0 in T and k1 in T:
        f = flips(k0, k1); S['flips'][f'{k0}->{k1}'] = f
        print(f"| {k0} -> {k1} | GSM +{f['gsm'][0]}/-{f['gsm'][1]} | scaled +{f['scaled'][0]}/-{f['scaled'][1]} | net {sum(u - d for u, d in f.values()):+d} |")

# ---- Q1: cause of arm 2's edge over arm 1
print('\n## Q1: arm-2 (calc on) right & arm-1 wrong vs the reverse, by cause')
def cause(i):
    """Why did arm 1 fail where arm 2 (+calc) succeeded? arithmetic = arm-1 trace has a wrong 'a op b = c'."""
    a1 = A['a1_off'][i]
    if a1['nwrong'] > 0: return 'arith'
    if T['a1_off'][i]['fmt'] == 0: return 'no_answer'
    return 'plan'
win = [i for i in range(N) if T['a2_on'][i]['ok'] and not T['a1_off'][i]['ok']]
lose = [i for i in range(N) if not T['a2_on'][i]['ok'] and T['a1_off'][i]['ok']]
cw = Counter((kinds[i], cause(i)) for i in win)
def cause_l(i):
    a2 = A['a2_on'][i]
    if a2['nwrong'] > 0: return 'arith'
    if T['a2_on'][i]['fmt'] == 0: return 'no_answer'
    return 'plan'
cl = Counter((kinds[i], cause_l(i)) for i in lose)
# host-alone: does arm-2 with calc OFF also solve it?
hostalso = Counter((kinds[i], T['a2_off'][i]['ok']) for i in win)
S['q1'] = dict(win=len(win), lose=len(lose), win_cause={f'{a}|{b}': v for (a, b), v in cw.items()},
               lose_cause={f'{a}|{b}': v for (a, b), v in cl.items()},
               win_a2off_ok={f'{a}|{b}': v for (a, b), v in hostalso.items()})
print('wins', len(win), dict(cw)); print('losses', len(lose), dict(cl)); print('wins solved by arm2 with calc OFF:', dict(hostalso))
# finer: in arith wins, was arm-2's corresponding equation computed by the calculator?
arith_win = [i for i in win if cause(i) == 'arith']
calls_in = sum(1 for i in arith_win if A['a2_on'][i]['calls'])
print(f'arith wins with >=1 calc call in arm-2 trace: {calls_in}/{len(arith_win)}')
S['q1']['arith_win_with_calls'] = calls_in
# a2_on vs a2_off flips: what the calculator alone does for the RL'd host
cal = [i for i in range(N) if T['a2_on'][i]['ok'] and not T['a2_off'][i]['ok']]
cal_c = Counter((kinds[i], 'arith' if A['a2_off'][i]['nwrong'] else ('no_answer' if not T['a2_off'][i]['fmt'] else 'plan')) for i in cal)
print('a2_off wrong -> a2_on right:', len(cal), dict(cal_c))
S['q1']['calc_switch'] = {f'{a}|{b}': v for (a, b), v in cal_c.items()}

# ---- Q2: how the calculator is used
print('\n## Q2: calculator use (implant on)')
def opname(q):
    s = set(q['ops'])
    if len(q['ops']) > 1: return 'chain(' + ''.join(q['ops']) + ')'
    return {'*': 'x', '/': '/', '+': '+', '-': '-'}[q['ops'][0]]
def digits(x):
    return len(str(abs(int(x)))) if x.denominator == 1 else len(str(abs(int(x)))) + 2
S['q2'] = {}
for k in ('base_on', 'a1_on', 'a2_on', 'baseR_on', 'a1R_on', 'a2R_on'):
    if k not in A: continue
    a = A[k]
    calls = [c for x in a for c in x['calls']]
    eqs = [q for x in a for q in x['eqs']]
    linked = [c for c in calls if c['eq'] is not None]
    ops_called = Counter(opname(c['eq']) for c in linked)
    ops_all = Counter(opname(q) for q in eqs)
    ok_called = sum(c['eq']['ok'] for c in linked)
    unc = [q for q in eqs if not q['called']]
    ok_unc = sum(q['ok'] for q in unc)
    # coverage: fraction of single-op equations that went through the calculator
    single = [q for q in eqs if len(q['ops']) == 1]
    cov = pct(sum(q['called'] for q in single), len(single))
    maxd = Counter(min(6, max(digits(o) for o in c['eq']['operands'])) for c in linked)
    # chaining: called eq uses as operand the claimed result of an earlier called eq in the same trace
    chain = 0
    for x in a:
        prev = []
        for c in x['calls']:
            if c['eq'] is None: continue
            if any(o in prev for o in c['eq']['operands']): chain += 1
            prev.append(c['eq']['claimed'])
    row = dict(calls=len(calls) / N, linked=pct(len(linked), len(calls)), ops_called=dict(ops_called), ops_all=dict(ops_all),
               exact_called=pct(ok_called, len(linked)), exact_uncalled=pct(ok_unc, len(unc)), n_uncalled=len(unc),
               cov_single=cov, eqs=len(eqs) / N, maxdig=dict(sorted(maxd.items())), chain=pct(chain, len(linked)),
               writes=sum(sum(r['wflag']) for r in T[k]) / N)
    S['q2'][k] = row
    print(k, json.dumps(row))

# per-problem paired comparison base_on vs a2_on on the same problems
pc = [(len(A['base_on'][i]['calls']), len(A['a2_on'][i]['calls'])) for i in range(N)]
S['q2']['paired_calls_more'] = sum(b > a for a, b in pc); S['q2']['paired_calls_less'] = sum(b < a for a, b in pc)
print('paired calls: a2 more', S['q2']['paired_calls_more'], 'fewer', S['q2']['paired_calls_less'])

# ---- Q3: host alone after RL
print('\n## Q3: host alone (implant off): base vs arm 1 vs arm 2')
for k in ('base_off', 'a1_off', 'a2_off'):
    a = A[k]; eqs = [q for x in a for q in x['eqs']]
    byop = defaultdict(lambda: [0, 0])
    for q in eqs:
        o = opname(q) if len(q['ops']) == 1 else 'chain'; byop[o][0] += 1; byop[o][1] += (not q['ok'])
    big = [q for q in eqs if max(digits(o) for o in q['operands']) >= 4]
    S.setdefault('q3', {})[k] = dict(err_by_op={o: (n, pct(w, n)) for o, (n, w) in byop.items()},
                                     big_err=pct(sum(not q['ok'] for q in big), len(big)), n_big=len(big),
                                     ans_with_wrong_eq=pct(sum(x['nwrong'] > 0 for x in a), N))
    print(k, json.dumps(S['q3'][k]))

json.dump(S, open('stats.json', 'w'), indent=1, default=str)
