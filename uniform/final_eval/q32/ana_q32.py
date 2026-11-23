"""[32](b)(c)(d) for GSM8K 0.6B: matched-compute majority vote, eval-seed spread, GSM-Plus. Paired item bootstrap, 2000 resamples.
usage: python3 uniform/final_eval/q32/ana_q32.py [b|c|d|r|v|s|a|t|T|m|p|u|h|x|X|E|U ...]. Configs without rows are skipped."""
import json, os, random, statistics, sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import items   # uniform/items.py: reduced per-item rows (bundles) or your own full rows via $FE_ROWS


def load(st, names):
    cols, toks = [], []
    for n in names:
        R = {}
        T = []
        for r in items.load(st, n):
            R[r['i']] = float(bool(r['ok'])); T.append(r['ntok'])
        cols.append(R); toks.append(sum(T) / len(T))
    ids = sorted(cols[0])
    return [sum(c[i] for c in cols) / len(cols) for i in ids], [100 * sum(c.values()) / len(c) for c in cols], sum(toks) / len(toks)


def have(st, names):
    return all(items.have(st, n) for n in names)


def table(st, ROWS, EFF):
    V = {}
    for k, names in ROWS.items():
        if not have(st, names):
            print(f'{st} row {k}: MISSING'); continue
        v, per, tq = load(st, names); V[k] = v
        print(f'{st} row {k:6s} mean {100*sum(v)/len(v):.2f}  per-seed {" / ".join("%.2f" % p for p in per)}  tok/q {tq:.0f}')
    rng = random.Random(0)
    for a, b in EFF:
        if a not in V or b not in V:
            continue
        d = [x - y for x, y in zip(V[a], V[b])]; n = len(d)
        bs = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
        print(f'  {a} - {b}: {100*sum(d)/n:+.2f} [{bs[50]:+.2f}, {bs[1949]:+.2f}]')


CW = ['r2_CW_s0', 'r2_CW_s1', 'r2_CW_s2']
what = sys.argv[1:] or ['b', 'c', 'd']
if 'b' in what:
    print('== (b) matched compute: N separate host-only samples + majority (N=12 test, 16 Hard)')
    for st, N in [('test', 12), ('hard', 16)]:
        ROWS = {'1': ['baseH'], '1-MV': [f'scH{N}_base'], '2': ['P64_s0', 'P64_s1', 'P64_s2'],
                '2-MV': [f'scH{N}_P64_s{s}' for s in range(3)], '3': ['r2_base'], '4': CW}
        table(st, ROWS, [('4', '2-MV'), ('2-MV', '2'), ('1-MV', '1'), ('3', '1-MV'), ('4', '2')])
    print('== (b2) Base host-only majority at row-3 compute (N=26 test, 30 Hard) vs row 3 (Base + rung 2)')
    for st, N in [('test', 26), ('hard', 30)]:
        table(st, {'1-MV3': [f'scH{N}_base'], '3': ['r2_base'], '3-vote0': ['q33_vote0']}, [('3', '1-MV3'), ('3-vote0', '1-MV3')])
if 'c' in what:
    print('== (c) eval seeds 7/11/13 (T = 1)')
    for st in ['test', 'hard']:
        for k, names in [('1', ['baseH', 'baseH_s11', 'baseH_s13']), ('3', ['r2_base', 'r2_base_s11', 'r2_base_s13']),
                         ("3'", ['r3_base', 'r3_base_s11', 'r3_base_s13'])]:
            ok = [n for n in names if have(st, [n])]
            per = [load(st, [n])[1][0] for n in ok]
            sd = statistics.stdev(per) if len(per) > 1 else float('nan')
            print(f'{st} row {k}: seeds {" / ".join("%.2f" % p for p in per)}  mean {statistics.mean(per):.2f}  sd {sd:.2f}  ({len(per)}/3)')
if 'r' in what:
    print('== row 4 reproducibility on stk_r2q (rq) and plain majority instead of the probe voter (rv), GSM8K test')
    table('test', {'4': CW, '4-rq': [f'rq_CW_s{s}' for s in range(3)], '4-vote0': [f'rq_CW_vote0_s{s}' for s in range(3)]},
          [('4-rq', '4'), ('4-vote0', '4-rq')])
    for s in range(3):
        table('test', {f'4_s{s}': [f'r2_CW_s{s}'], f'rq_s{s}': [f'rq_CW_s{s}']}, [(f'rq_s{s}', f'4_s{s}')])
if 'd' in what:
    print('== (d) GSM-Plus testmini (2100), rows 1-4')
    ROWS = {'1': ['gp_baseH'], '2': ['gp_P64_s0', 'gp_P64_s1', 'gp_P64_s2'], '3': ['gp_r2_base'],
            '4': ['gp_r2_CW_s0', 'gp_r2_CW_s1', 'gp_r2_CW_s2']}
    table('gsmplus', ROWS, [('2', '1'), ('3', '1'), ('4', '1'), ('4', '3'), ('4', '2')])
    print('-- (d) matched-compute majority on GSM-Plus (host-only samples, N=12 / 26)')
    table('gsmplus', {'2-MV': [f'gp_scH12_P64_s{s}' for s in range(3)], '2-MV16': [f'gp_scH16_P64_s{s}' for s in range(3)], '4': ROWS['4'],
                      '1-MV': ['gp_scH12_base'], '1-MV3': ['gp_scH26_base'], '3': ROWS['3']},
          [('4', '2-MV'), ('4', '2-MV16'), ('3', '1-MV3'), ('3', '1-MV')])   # 2-MV16 ([34] fill j23946): N matched to GSM-Plus row-4 tok/q
if 'v' in what:
    print('== [34] plain majority (VM 0) vs probe voter (VM 1) on GSM-Plus testmini, rows 3 and 4 (stk_r2q)')
    table('gsmplus', {'4': ['gp_r2_CW_s0', 'gp_r2_CW_s1', 'gp_r2_CW_s2'], '4-vote0': [f'gp_v0_CW_s{s}' for s in range(3)],
                      '3': ['gp_r2_base'], '3-vote0': ['gp_v0_base'], '1': ['gp_baseH']},
          [('4-vote0', '4'), ('3-vote0', '3'), ('4-vote0', '3-vote0'), ('3-vote0', '1')])
if 's' in what:
    print('== [34] seed-averaged matched-compute deltas (eval seeds 7/11/13)')
    E = ('', '_e11', '_e13')
    table('gsmplus', {'4': [f'gp_r2_CW_s{s}{x}' for x in ('', '_s11', '_s13') for s in range(3)],
                      '2-MV16': [f'gp_scH16_P64_s{s}{x}' for x in E for s in range(3)]}, [('4', '2-MV16')])
    for st, N in [('test', 26), ('hard', 30)]:
        table(st, {'3': ['r2_base', 'r2_base_s11', 'r2_base_s13'], '1-MV3': [f'scH{N}_base{x}' for x in E]}, [('3', '1-MV3')])
    table('gsmplus', {'3': ['gp_r2_base', 'gp_r2_base_s11', 'gp_r2_base_s13'], '1-MV3': [f'gp_scH26_base{x}' for x in E]},
          [('3', '1-MV3')])   # j23972 / j23981
    print('-- row 4 (CW s0-2; seed 7 = main row, seeds 11/13 on stk_r2q) vs 2-MV (N=12 test / 16 Hard)')
    for st, N in [('test', 12), ('hard', 16)]:
        table(st, {'4': [f'r2_CW_s{s}{x}' for x in ('', '_s11', '_s13') for s in range(3)],
                   '2-MV': [f'scH{N}_P64_s{s}{x}' for x in E for s in range(3)]}, [('4', '2-MV')])
    print('-- C16: rung 3 / rung 4 vs rung 2 on the CW hosts, eval seeds 7/11/13')
    for st in ('test', 'hard'):
        R = {f'r{r}': [f'r{r}_CW_s{s}{x}' for x in ('', '_s11', '_s13') for s in range(3)] for r in (2, 3, 4)}
        table(st, R, [('r3', 'r2'), ('r4', 'r2')])
    print('-- GSM-Plus 4 rows averaged over eval seeds 7/11/13')
    X3 = ('', '_s11', '_s13')
    table('gsmplus', {'1': ['gp_baseH' + x for x in X3], '2': [f'gp_P64_s{s}{x}' for x in X3 for s in range(3)],
                      '3': ['gp_r2_base' + x for x in X3], '4': [f'gp_r2_CW_s{s}{x}' for x in X3 for s in range(3)]},
          [('2', '1'), ('3', '1'), ('4', '2'), ('4', '3'), ('2', '3')])
    print('-- test / Hard 4 rows averaged over eval seeds 7/11/13')
    for st in ('test', 'hard'):
        table(st, {'1': ['baseH' + x for x in X3], '2': [f'P64_s{s}{x}' for x in X3 for s in range(3)],
                   '3': ['r2_base' + x for x in X3], '4': [f'r2_CW_s{s}{x}' for x in X3 for s in range(3)]},
              [('2', '1'), ('3', '1'), ('4', '3'), ('4', '2')])
    print('-- C16 on GSM-Plus (seed 7): rung 3 / rung 4 vs rung 2 on the CW hosts')
    table('gsmplus', {f'r{r}': [f'gp_r{r}_CW_s{s}' for s in range(3)] for r in (2, 3, 4)}, [('r3', 'r2'), ('r4', 'r2')])
    print('-- [33] ablation steps on 0.6B Base, test, eval seeds 7/11/13 (j23988)')
    table('test', {'1': ['baseH' + x for x in X3], 'N1': ['q33_N1' + x for x in X3], 'fixN4': ['q33_fixN4' + x for x in X3],
                   'fixN8': ['q33_fixN8' + x for x in X3], '3': ['r2_base' + x for x in X3]},
          [('N1', '1'), ('fixN4', 'N1'), ('fixN8', 'fixN4'), ('3', 'fixN8'), ('3', '1')])
    print('-- [33] ablation steps on 0.6B Base, GSM-Plus, seed 7 (j23992)')
    table('gsmplus', {'1': ['gp_baseH'], 'N1': ['gp_q33_N1'], 'fixN4': ['gp_q33_fixN4'], 'fixN8': ['gp_q33_fixN8'], '3': ['gp_r2_base']},
          [('N1', '1'), ('fixN4', 'N1'), ('fixN8', 'fixN4'), ('3', 'fixN8'), ('3', '1')])
    print('-- C16 on GSM-Plus, eval seeds 7/11/13 (j23978-79, j23983-84)')
    table('gsmplus', {f'r{r}': [f'gp_r{r}_CW_s{s}{x}' for x in X3 for s in range(3)] for r in (2, 3, 4)},
          [('r3', 'r2'), ('r4', 'r2')])
if 'a' in what:
    print('== [33] ablation steps on 0.6B Base, averaged over eval seeds 7/11/13 (test j23988; GSM-Plus j23992/j23995; Hard q34a33h)')
    X3 = ('', '_s11', '_s13')
    STEPS = [('N1', '1'), ('fixN4', 'N1'), ('fixN8', 'fixN4'), ('3', 'fixN8'), ('3', '1')]
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        R = {'1': [p + 'baseH' + x for x in X3], 'N1': [p + 'q33_N1' + x for x in X3], 'fixN4': [p + 'q33_fixN4' + x for x in X3],
             'fixN8': [p + 'q33_fixN8' + x for x in X3], '3': [p + 'r2_base' + x for x in X3]}
        table(st, R, STEPS)
    table('hard', {'3': ['r2_base'], 'nowpr': ['q33_nowpr']}, [('nowpr', '3')])
if 't' in what:
    print("== [34] 'attach after RL' (row 2b = P64 s0-2 + untrained rung 2 at eval; j23999 / j24002), seed 7")
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        table(st, {'2': [f'{p}P64_s{s}' for s in range(3)], '2b': [f'{p}r2_P64_s{s}' for s in range(3)],
                   '3': [p + 'r2_base'], '4': [f'{p}r2_CW_s{s}' for s in range(3)]},
              [('2b', '2'), ('4', '2b'), ('2b', '3')])
if 'T' in what:
    print("== [34] row 2b averaged over eval seeds 7/11/13 (j23999 + j24008), test / Hard")
    X3 = ('', '_s11', '_s13')
    for st in ('test', 'hard'):
        table(st, {'2': [f'P64_s{s}{x}' for x in X3 for s in range(3)], '2b': [f'r2_P64_s{s}{x}' for x in X3 for s in range(3)],
                   '3': ['r2_base' + x for x in X3], '4': [f'r2_CW_s{s}{x}' for x in X3 for s in range(3)]},
              [('2b', '2'), ('4', '2b'), ('2b', '3')])
    print("== [34] row 2b on GSM-Plus averaged over eval seeds 7/11/13 (j24002 + j24014)")
    table('gsmplus', {'2': [f'gp_P64_s{s}{x}' for x in X3 for s in range(3)], '2b': [f'gp_r2_P64_s{s}{x}' for x in X3 for s in range(3)],
                      '3': ['gp_r2_base' + x for x in X3], '4': [f'gp_r2_CW_s{s}{x}' for x in X3 for s in range(3)]},
          [('2b', '2'), ('4', '2b'), ('2b', '3')])
if 'm' in what:
    print("== [34] mechanism of 4 − 2b (0.6B, seed 7; j24020 P64, j24021 CW): rung-2 steps on the row-2 (P64) vs row-4 (CW) hosts")
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        R = {}
        for H, h in (('P', 'P64'), ('C', 'CW')):
            R[H + 'N1'] = [f'{p}m_N1_{h}_s{s}' for s in range(3)]
            R[H + 'N8'] = [f'{p}m_fixN8_{h}_s{s}' for s in range(3)]
            R[H + 'ad'] = [f'{p}r2_{h}_s{s}' for s in range(3)]
        table(st, R, [('CN1', 'PN1'), ('CN8', 'PN8'), ('Cad', 'Pad'), ('PN8', 'PN1'), ('CN8', 'CN1'), ('Pad', 'PN8'), ('Cad', 'CN8')])
if 'p' in what:
    print("== [34] row 4p: row-4 hosts (CW s0-2) with the stack removed, 1 sample (j24035), vs row 2 (P64) and row 4")
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        table(st, {'2': [f'{p}P64_s{s}' for s in range(3)], '4p': [f'{p}H_CW_s{s}' for s in range(3)],
                   '4': [f'{p}r2_CW_s{s}' for s in range(3)]}, [('4p', '2'), ('4', '4p')])
if 'u' in what:
    print("== [34] rung 2 at N=1 on P64 vs CW across RL updates (u0 = Base, j24039; u300 = j24020-21)")
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        R = {'u0': [f'{p}q33_N1']}
        for H, h in (('P', 'P64'), ('C', 'CW')):
            for u in (100, 200):
                R[f'{H}{u}'] = [f'{p}m_N1_{h}_s{s}_u{u}' for s in range(3)]
            R[f'{H}300'] = [f'{p}m_N1_{h}_s{s}' for s in range(3)]
        table(st, R, [('C100', 'P100'), ('C200', 'P200'), ('C300', 'P300')])
if 'h' in what:
    print("== [34] host only (no stack, 1 sample) on P64 vs CW across RL updates (u0 = baseH; u100/u200 j24042; u300 = row 2 / 4p)")
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        R = {'u0': [f'{p}baseH']}
        for H, h in (('P', 'P64'), ('C', 'CW')):
            for u in (100, 200):
                R[f'{H}{u}'] = [f'{p}H_{h}_s{s}_u{u}' for s in range(3)]
            R[f'{H}300'] = [f'{p}{h}_s{s}' if h == 'P64' else f'{p}H_{h}_s{s}' for s in range(3)]
        table(st, R, [('C100', 'P100'), ('C200', 'P200'), ('C300', 'P300')])
if 'x' in what:
    print("== [34] host only (no stack, 1 sample) on every base-host RL arm (j24045): which RL factor damages the bare host?")
    print("   Pb / Bb: 256 prompts (plain / stack N=1); P64 / C / CW: 64 prompts (plain / stack N=4 / stack N=4 + WPR); C600 / CW600: 600 u (1 seed)")
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        R = {'base': [f'{p}baseH'], 'Pb': [f'{p}H_Pb_s{s}' for s in (1, 2, 3)], 'Bb': [f'{p}H_Bb_s{s}' for s in (5, 6, 7)],
             'P64': [f'{p}P64_s{s}' for s in range(3)], 'C': [f'{p}H_{h}' for h in ('C', 'C_s2', 'C_s3')],
             'CW': [f'{p}H_CW_s{s}' for s in range(3)], 'C600': [f'{p}H_C600'], 'CW600': [f'{p}H_CW600'],
             'C_s5': [f'{p}H_C'], 'CW_s0': [f'{p}H_CW_s0']}
        table(st, R, [('Bb', 'Pb'), ('C', 'P64'), ('CW', 'P64'), ('CW', 'C'), ('C600', 'C_s5'), ('CW600', 'CW_s0')])
    print("== [34] rung 2 (row-4 settings) on the Bb and C hosts (j24046-47) vs the plain-RL hosts + the same stack")
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        R = {'r2_Pb': [f'{p}r2_Pb_s{s}' for s in (1, 2, 3)], 'r2_Bb': [f'{p}r2_Bb_s{s}' for s in (5, 6, 7)],
             'r2_P64': [f'{p}r2_P64_s{s}' for s in range(3)], 'r2_C': [f'{p}r2_{h}' for h in ('C', 'C_s2', 'C_s3')],
             'r2_CW': [f'{p}r2_CW_s{s}' for s in range(3)]}
        table(st, R, [('r2_Bb', 'r2_Pb'), ('r2_C', 'r2_P64'), ('r2_CW', 'r2_P64'), ('r2_CW', 'r2_C')])
if 'X' in what:
    X3 = ('', '_s11', '_s13')
    print("== [34] 'which RL factor' averaged over eval seeds 7/11/13: host only (j24045 + j24071), C600 / CW600 excluded")
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        R = {'Pb': [f'{p}H_Pb_s{s}{x}' for s in (1, 2, 3) for x in X3], 'Bb': [f'{p}H_Bb_s{s}{x}' for s in (5, 6, 7) for x in X3],
             'P64': [f'{p}P64_s{s}{x}' for s in range(3) for x in X3], 'C': [f'{p}H_{h}{x}' for h in ('C', 'C_s2', 'C_s3') for x in X3],
             'CW': [f'{p}H_CW_s{s}{x}' for s in range(3) for x in X3]}
        table(st, R, [('Bb', 'Pb'), ('C', 'P64'), ('CW', 'P64'), ('CW', 'C')])
    print("== [34] rung 2 (row-4 settings) averaged over eval seeds 7/11/13 (j24046-47 + j24062, j24067-68), test / Hard")
    for st in ('test', 'hard'):
        R = {'r2_Pb': [f'r2_Pb_s{s}{x}' for s in (1, 2, 3) for x in X3], 'r2_Bb': [f'r2_Bb_s{s}{x}' for s in (5, 6, 7) for x in X3],
             'r2_P64': [f'r2_P64_s{s}{x}' for s in range(3) for x in X3], 'r2_C': [f'r2_{h}{x}' for h in ('C', 'C_s2', 'C_s3') for x in X3],
             'r2_CW': [f'r2_CW_s{s}{x}' for s in range(3) for x in X3]}
        table(st, R, [('r2_Bb', 'r2_Pb'), ('r2_C', 'r2_P64'), ('r2_CW', 'r2_P64'), ('r2_CW', 'r2_C')])
if 'E' in what:
    print("== [34] 0.6B rung 2 at N=1 on P64 vs CW averaged over eval seeds 7/11/13 (j24020-21, j24027-28, j24050)")
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        R = {H + 'N1': [f'{p}m_N1_{h}_s{s}{x}' for s in range(3) for x in ('', '_s11', '_s13')] for H, h in (('P', 'P64'), ('C', 'CW'))}
        table(st, R, [('CN1', 'PN1')])
if 'U' in what:
    print("== [34] rung 2 (row-4 settings) on P64 (2b) vs CW (4) across RL updates (u100/u200 j24052-53, GSM-Plus j24059-60; u300 = 2b / row 4)")
    for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
        R = {}
        for H, h in (('P', 'P64'), ('C', 'CW')):
            for u in (100, 200):
                R[f'{H}{u}'] = [f'{p}r2_{h}_s{s}_u{u}' for s in range(3)]
            R[f'{H}300'] = [f'{p}r2_{h}_s{s}' for s in range(3)]
        table(st, R, [('C100', 'P100'), ('C200', 'P200'), ('C300', 'P300')])
