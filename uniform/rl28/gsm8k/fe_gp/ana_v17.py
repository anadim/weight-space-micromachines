"""[34] 1.7B GSM8K fill analysis (rl28/gsm8k/fe_gp): plain majority vs probe voter (rows 3/4, test + GSM-Plus), row-3 matched-compute
Base majority (N = 21 test / 26 Hard), row-4 eval seeds 7/11/13 (CW17_s2 test). Paired item bootstrap, 2000 resamples.
usage: python3 uniform/rl28/gsm8k/fe_gp/ana_v17.py   (rows: uniform/items.py)"""
import glob, json, os, random, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
import items   # uniform/items.py: reduced per-item rows (bundles) or your own full rows via $FE_ROWS

F = {f'{st}_{n}': (st, n) for st in ('test', 'hard', 'gsmplus') for n in items.names(st)}


def col(k):
    R, T = {}, []
    for r in items.load(*F[k]):
        R[r['i']] = float(bool(r['ok'])); T.append(r['ntok'])
    return R, sum(T) / len(T)


def table(st, ROWS, EFF):
    V = {}
    for k, names in ROWS.items():
        keys = [f'{st}_{n}' for n in names]
        miss = [x for x in keys if x not in F]
        if miss:
            print(f'{st} row {k}: MISSING {miss}'); continue
        cs = [col(x) for x in keys]; ids = sorted(cs[0][0])
        V[k] = [sum(c[0][i] for c in cs) / len(cs) for i in ids]
        print(f'{st} row {k:8s} mean {100*sum(V[k])/len(ids):.2f}  per-seed {" / ".join("%.2f" % (100*sum(c[0].values())/len(c[0])) for c in cs)}'
              f'  tok/q {sum(c[1] for c in cs)/len(cs):.0f}  n {len(ids)}')
    rng = random.Random(0)
    for a, b in EFF:
        if a in V and b in V:
            d = [x - y for x, y in zip(V[a], V[b])]; n = len(d)
            bs = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
            print(f'  {a} - {b}: {100*sum(d)/n:+.2f} [{bs[50]:+.2f}, {bs[1949]:+.2f}]')


CW = [f'r2_CW17_s{s}' for s in range(3)]
print('== plain majority (VM 0) vs probe voter, 1.7B')
table('test', {'4': CW, '4-vote0': [f'v0_{n}' for n in CW], '3': ['r2_17'], '3-vote0': ['v0_r2_17'], '1': ['base17H']},
      [('4-vote0', '4'), ('3-vote0', '3'), ('4-vote0', '3-vote0'), ('3-vote0', '1')])
table('gsmplus', {'4': [f'gp_{n}' for n in CW], '4-vote0': [f'gp_v0_{n}' for n in CW], '3': ['gp_r2_17'], '3-vote0': ['gp_v0_r2_17']},
      [('4-vote0', '4'), ('3-vote0', '3'), ('4-vote0', '3-vote0')])
print('== row-3 matched compute: Base, N host-only samples + majority (N = 21 test, 26 Hard)')
for st, N in (('test', 21), ('hard', 26)):
    table(st, {'1': ['base17H'], '1-MV3': [f'scH{N}_base17'], '3': ['r2_17']}, [('3', '1-MV3'), ('1-MV3', '1'), ('3', '1')])
print('== row-4 eval seeds (CW17_s2, test, T = 1)')
per = []
for k in ('test_r2_CW17_s2', 'test_r2_CW17_s2_s11', 'test_r2_CW17_s2_s13'):
    if k in F:
        R, _ = col(k); per.append(100 * sum(R.values()) / len(R))
print('  seeds 7/11/13:', ' / '.join('%.2f' % p for p in per), f' sd {statistics.stdev(per):.2f}' if len(per) > 1 else '')
print('== [33] ablation steps on 1.7B Base (rung 2 build stk_r2q; T = 1, seed 7, once)')
for st in ('test', 'hard'):
    table(st, {'1': ['base17H'], 'N1': ['q33_N1_17'], 'fixN4': ['q33_fixN4_17'], 'fixN8': ['q33_fixN8_17'], '3': ['r2_17'],
               'nowpr': ['q33_nowpr_17']},
          [('N1', '1'), ('fixN4', 'N1'), ('fixN8', 'fixN4'), ('3', 'fixN8'), ('nowpr', '3'), ('3', '1')])
print('== 2-MV vs row 4 over eval seeds 7/11/13 (test; 2-MV = P17 s0-2 N=16, row 4 = CW17 s0-2)')
S4 = {7: CW, 11: [f'{n}_s11' for n in CW], 13: [f'{n}_s13' for n in CW]}
SM = {7: [f'scH16_P17_s{s}' for s in range(3)], 11: [f'scH16_P17_s{s}_e11' for s in range(3)], 13: [f'scH16_P17_s{s}_e13' for s in range(3)]}
table('test', {'4': sum(S4.values(), []), '2-MV': sum(SM.values(), [])}, [('4', '2-MV')])
print('== 2-MV vs row 4 over eval seeds 7/11/13 (Hard; 2-MV = P17 s0-2 N=20)')
table('hard', {'4': sum(S4.values(), []), '2-MV': [f'scH20_P17_s{s}' + x for x in ('', '_e11', '_e13') for s in range(3)]}, [('4', '2-MV')])
print('== 2-MV vs row 4 over eval seeds 7/11/13 (GSM-Plus; 2-MV = P17 s0-2 N=19)')
table('gsmplus', {'4': [f'gp_{n}' for n in sum(S4.values(), [])], '2-MV': [f'gp_scH19_P17_s{s}' + x for x in ('', '_e11', '_e13') for s in range(3)]},
      [('4', '2-MV')])
print('== plain majority on Hard, rows 3/4')
table('hard', {'4': CW, '4-vote0': [f'v0_{n}' for n in CW], '3': ['r2_17'], '3-vote0': ['v0_r2_17']},
      [('4-vote0', '4'), ('3-vote0', '3'), ('4-vote0', '3-vote0')])
print('== row 3 vs 1-MV3 over eval seeds 7/11/13 (GSM-Plus; 1-MV3 = Base x 24)')
table('gsmplus', {'3': ['gp_r2_17', 'gp_r2_17_s11', 'gp_r2_17_s13'], '1-MV3': ['gp_scH24_base17', 'gp_scH24_base17_e11', 'gp_scH24_base17_e13']},
      [('3', '1-MV3')])
print('== row 3 vs 1-MV3 over eval seeds 7/11/13 (test N=21 / Hard N=26)')
for st, N in (('test', 21), ('hard', 26)):
    table(st, {'3': ['r2_17', 'r2_17_s11', 'r2_17_s13'], '1-MV3': [f'scH{N}_base17' + x for x in ('', '_e11', '_e13')]}, [('3', '1-MV3')])
print('== GSM-Plus 4 rows averaged over eval seeds 7/11/13')
X3 = ('', '_s11', '_s13')
table('gsmplus', {'1': ['gp_base17H' + x for x in X3], '2': [f'gp_P17_s{s}{x}' for x in X3 for s in range(3)],
                  '3': ['gp_r2_17' + x for x in X3], '4': [f'gp_{n}' for n in sum(S4.values(), [])]},
      [('2', '1'), ('3', '1'), ('4', '2'), ('4', '3'), ('2', '3')])
print('== test / Hard 4 rows averaged over eval seeds 7/11/13 (row 1 / 3 seeds from [32](c))')
for st in ('test', 'hard'):
    table(st, {'1': ['base17H', 'base17H_s11', 'base17H_s13'], '2': [f'P17_s{s}{x}' for x in X3 for s in range(3)],
               '3': ['r2_17', 'r2_17_s11', 'r2_17_s13'], '4': sum(S4.values(), [])},
          [('2', '1'), ('3', '1'), ('4', '3'), ('4', '2')])
print('== [33] ablation steps on 1.7B Base averaged over eval seeds 7/11/13 (j23986)')
for st in ('test', 'hard'):
    table(st, {'1': ['base17H' + x for x in X3], 'N1': ['q33_N1_17' + x for x in X3], 'fixN4': ['q33_fixN4_17' + x for x in X3],
               'fixN8': ['q33_fixN8_17' + x for x in X3], '3': ['r2_17' + x for x in X3]},
          [('N1', '1'), ('fixN4', 'N1'), ('fixN8', 'fixN4'), ('3', 'fixN8'), ('3', '1')])
print('== [33] ablation steps on 1.7B Base, GSM-Plus, seed 7 (j23993)')
table('gsmplus', {'1': ['gp_base17H'], 'N1': ['gp_q33_N1_17'], 'fixN4': ['gp_q33_fixN4_17'], 'fixN8': ['gp_q33_fixN8_17'], '3': ['gp_r2_17']},
      [('N1', '1'), ('fixN4', 'N1'), ('fixN8', 'fixN4'), ('3', 'fixN8'), ('3', '1')])
print('== [33] ablation steps on 1.7B Base, GSM-Plus, averaged over eval seeds 7/11/13 (j23993 + j23998)')
table('gsmplus', {'1': ['gp_base17H' + x for x in X3], 'N1': ['gp_q33_N1_17' + x for x in X3], 'fixN4': ['gp_q33_fixN4_17' + x for x in X3],
                  'fixN8': ['gp_q33_fixN8_17' + x for x in X3], '3': ['gp_r2_17' + x for x in X3]},
      [('N1', '1'), ('fixN4', 'N1'), ('fixN8', 'fixN4'), ('3', 'fixN8'), ('3', '1')])
print("== [34] 'attach after RL' (row 2b = P17 s0-2 + untrained rung 2 at eval; j24000, j24001, j24003), seed 7")
for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
    table(st, {'2': [f'{p}P17_s{s}' for s in range(3)], '2b': [f'{p}r2_P17_s{s}' for s in range(3)],
               '3': [p + 'r2_17'], '4': [p + n for n in CW]}, [('2b', '2'), ('4', '2b'), ('2b', '3')])
print("== [34] row 2b averaged over eval seeds 7/11/13 (j24000-01 + j24009-11), test / Hard")
for st in ('test', 'hard'):
    table(st, {'2': [f'P17_s{s}{x}' for x in X3 for s in range(3)], '2b': [f'r2_P17_s{s}{x}' for x in X3 for s in range(3)],
               '3': ['r2_17' + x for x in X3], '4': sum(S4.values(), [])}, [('2b', '2'), ('4', '2b'), ('2b', '3')])
print("== [34] row 2b on GSM-Plus averaged over eval seeds 7/11/13 (j24001, j24003 + j24015-17)")
table('gsmplus', {'2': [f'gp_P17_s{s}{x}' for x in X3 for s in range(3)], '2b': [f'gp_r2_P17_s{s}{x}' for x in X3 for s in range(3)],
                  '3': ['gp_r2_17' + x for x in X3], '4': [f'gp_{n}' for n in sum(S4.values(), [])]},
      [('2b', '2'), ('4', '2b'), ('2b', '3')])
print("== [34] mechanism of 4 − 2b (1.7B, seed 7; j24022-24): rung-2 steps on the row-2 (P17) vs row-4 (CW17) hosts")
for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
    R = {}
    for H, h in (('P', 'P17'), ('C', 'CW17')):
        R[H + 'N1'] = [f'{p}m_N1_{h}_s{s}' for s in range(3)]
        R[H + 'N8'] = [f'{p}m_fixN8_{h}_s{s}' for s in range(3)]
        R[H + 'ad'] = [f'{p}r2_{h}_s{s}' for s in range(3)]
    table(st, R, [('CN1', 'PN1'), ('CN8', 'PN8'), ('Cad', 'Pad'), ('PN8', 'PN1'), ('CN8', 'CN1'), ('Pad', 'PN8'), ('Cad', 'CN8')])
print("== [34] row 4p (1.7B): CW17 s0-2 with the stack removed, 1 sample (j24036), vs row 2 (P17) and row 4")
for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
    table(st, {'2': [f'{p}P17_s{s}' for s in range(3)], '4p': [f'{p}H_CW17_s{s}' for s in range(3)],
               '4': [f'{p}r2_CW17_s{s}' for s in range(3)]}, [('4p', '2'), ('4', '4p')])
# (u100/u200 trajectory dropped: the 1.7B step_000100/200 checkpoints were deleted after RL; j24040 failed)
print("== [34] 1.7B rung 2 at N=1 on P17 vs CW17 averaged over eval seeds 7/11/13 (j24022-24, j24029-31, j24048)")
for st, p in (('test', ''), ('hard', ''), ('gsmplus', 'gp_')):
    R = {}
    for H, h in (('P', 'P17'), ('C', 'CW17')):
        R[H + 'N1'] = [f'{p}m_N1_{h}_s{s}{x}' for s in range(3) for x in ('', '_s11', '_s13')]
    table(st, R, [('CN1', 'PN1')])
