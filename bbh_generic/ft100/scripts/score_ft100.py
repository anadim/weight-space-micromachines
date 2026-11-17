"""Aggregate scores only (never prints items). Real BBH re-scored with peek/scripts/score.py's official norm (verbatim);
generated sets scored with each team's own harness 'ok'. call% = share of outputs containing <tool_call>."""
import json, re, os
import os as _os
FT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))          # bbh_generic/ft100
REPO = _os.environ.get('WSM_REPO', _os.path.dirname(_os.path.dirname(FT)))     # repo root
R = _os.environ.get('BFT_R', _os.path.join(FT, 'work'))                        # work dir: runs/ out/ logs/ locks/ jobs.txt
G = _os.path.join(REPO, 'bbh_generic')                                         # team code (unchanged)
BBH = _os.environ.get('BBH_DIR', _os.path.join(REPO, 'data', 'bbh', 'data'))  # official BBH task files (data/download_bbh.sh)
COT = _os.environ.get('BBH_COT', _os.path.join(REPO, 'data', 'bbh', 'cot'))
PEEK = _os.environ.get('PEEK_OUT', _os.path.join(G, 'peek', 'out'))           # bbh_peek arm outputs (peek/scripts/run_job.sh)
TEAM_A_FT = _os.environ.get('TEAM_A_FT', _os.path.join(REPO, 'runs', 'teamA_ft_v3'))  # not released; teamA/ft.py on train3
VETO = _os.environ.get('VETO', _os.path.join(REPO, 'runs', 'veto_qwen3.pt'))  # scripts/solve_veto.py
P = PEEK; RB = _os.environ.get('TEAMB_OUT', f'{G}/teamB/out')   # Team B's own run outputs (teamB/run_eval.py)


def norm(x, task):   # verbatim from peek/scripts/score.py (= scripts/run_bbh.py)
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x) and task not in ('word_sorting', 'dyck_languages'): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    if task in ('multistep_arithmetic_two', 'object_counting'): m = re.search(r'-?\d+', x); return m.group() if m else x
    return x.lower()


BN = {'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies'}


def load(path):
    return json.load(open(path))['rows'] if os.path.exists(path) else None


def agg(rows, group, official=False):
    """group(row) -> (bucket, task) or None"""
    out = {}
    for r in rows or []:
        g = group(r)
        if g is None: continue
        b, t = g
        ok = norm(r['pred'], t) == norm(r['gold'], t) if official else bool(r['ok'])
        s = out.setdefault(b, {}).setdefault(t, [0, 0, 0]); s[0] += 1; s[1] += ok; s[2] += '<tool_call>' in r.get('out', '')
    return out


def gA(r):
    t = r['task']
    if t.startswith('oos2_'): return ('oos', t)
    if re.fullmatch(r'(track|order)[357]', t): return ('test', t)
    return ('bbh', t)


def gB(r):
    s = r['set']; task, kind = s.split('_', 1)
    return {'test': ('test', task), 'oos': ('oos', task), 'bbh': ('bbh', BN.get(task, task))}.get(kind)


def merge(*ds):
    o = {}
    for d in ds:
        for b, tt in (d or {}).items(): o.setdefault(b, {}).update(tt)
    return o


def summ(d, b):
    if b not in d: return None
    n = sum(v[0] for v in d[b].values()); return (100 * sum(v[1] for v in d[b].values()) / n, 100 * sum(v[2] for v in d[b].values()) / n, n,
                                                  {t: round(100 * v[1] / v[0], 1) for t, v in d[b].items()})


res = {}
# ---- Team A
A = {'base': {'notools': merge(agg(load(f'{R}/out/A_base_testoos2.json'), gA), agg(load(f'{P}/A_base.json'), gA, False) and
                               agg([r for r in load(f'{P}/A_base.json') or []], lambda r: ('bbh', r['task']), True))}}
for c in ['n32_s0', 'n100_s0', 'n100_s1', 'n300_s0']:
    A[c] = {}
    for arm in ['notools', 'tools']:
        rows = load(f'{R}/out/A_{c}_{arm}.json')
        if rows is None: continue
        A[c][arm] = merge(agg([r for r in rows if gA(r)[0] != 'bbh'], gA), agg([r for r in rows if gA(r)[0] == 'bbh'], gA, True))
A['ftv3_14.5k'] = {arm: merge(agg(load(f'{R}/out/A_ftv3ref_{arm}_testoos2.json'), gA),
                              agg(load(f'{P}/A_ftv3_{arm}.json'), lambda r: ('bbh', r['task']), True)) for arm in ['notools', 'tools']}
# ---- Team B
SB = lambda rows, kind: [r for r in rows or [] if r['set'].endswith('_' + kind)]
B = {'base': {'notools': merge(agg(load(f'{RB}/plain_test.json'), gB), agg(load(f'{RB}/plain_oos.json'), gB),
                               agg(load(f'{P}/B_base.json'), lambda r: ('bbh', BN[r['set']]), True))}}
for c in ['n32_s0', 'n100_s0', 'n100_s1', 'n300_s0']:
    B[c] = {}
    for arm in ['notools', 'tools']:
        rows = load(f'{R}/out/B_{c}_{arm}.json')
        if rows is None: continue
        B[c][arm] = merge(agg(SB(rows, 'test') + SB(rows, 'oos'), gB), agg(SB(rows, 'bbh'), gB, True))
B['ft3_12k'] = {arm: merge(agg(load(f'{RB}/ft3_{a2}_test.json'), gB), agg(load(f'{RB}/ft3_{a2}_oos.json'), gB),
                           agg(load(f'{P}/B_ft3_{arm}.json'), lambda r: ('bbh', BN[r['set']]), True))
                for arm, a2 in [('notools', 'none'), ('tools', 'all')]}
table = {}
for team, D in [('A', A), ('B', B)]:
    print(f'=== Team {team}: acc% (call%) [n]   no tools | tools')
    for c, arms in D.items():
        line = f'{c:12s}'
        for b in ['test', 'oos', 'bbh']:
            cells = []
            for arm in ['notools', 'tools']:
                s = summ(arms.get(arm, {}), b) if arm in arms else None
                cells.append('   -       ' if s is None else f'{s[0]:5.1f} ({s[1]:3.0f})')
                if s: table.setdefault(team, {}).setdefault(c, {}).setdefault(b, {})[arm] = {'acc': round(s[0], 1), 'call': round(s[1], 1), 'n': s[2], 'per_task': s[3]}
            line += f' | {b}: ' + ' | '.join(cells)
        print(line)
json.dump(table, open(f'{R}/scores_ft100.json', 'w'), indent=1)   # shipped copy: ft100/scores_ft100.json
