"""Blog figure: RL learns the calculator's competence map (CWinOx). Reads figdata/*.jsonl, writes fig_competence_map.png.
Left: rollout fire rate per operation (CWinOx s0) at three points in training.
Middle: dev accuracy after 300 updates: no calculator / calculator with the hand trigger (calls everything) / learned trigger.
Right: what makes it learnable: x and ÷ fire rate at u300 with no floor, floor only, floor + gentle in-wire lr."""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, '..', 'results', 'wiring', 'figdata')
OUTD = os.path.join(HERE, '..', 'figures')                 # rendered figures live in rl/figures/
INK, INK2, GRID, GREEN, PALE, MID = '#222222', '#6b7280', '#e5e7eb', '#118659', '#e7f3ee', '#5fae8f'
for f in font_manager.findSystemFonts():
    if 'Avenir' in f:
        try: font_manager.fontManager.addfont(f)
        except Exception: pass
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': 11, 'axes.edgecolor': INK2, 'axes.linewidth': 0.8,
                     'figure.dpi': 160, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})
load = lambda t: [json.loads(l) for l in open(os.path.join(FD, t + '.jsonl'))]
OPS = ['+', '−', '×', '÷']; KEY = {'+': '+', '−': '-', '×': '*', '÷': '/'}


WIN = 10                 # every rollout fire rate in this figure: calls pooled over updates u-9 .. u (10 updates)
SIDE = {'window': f'rollout calls pooled over updates u-{WIN - 1}..u', 'rates': {}}


def counts_at(rows, u, win=WIN):
    acc = {k: [0, 0] for k in OPS}
    for r in rows:
        if 'iw_op' in r and u - win < r['u'] <= u:
            for k in OPS:
                c = r['iw_op'].get(KEY[k], [0, 0]); acc[k][0] += c[0]; acc[k][1] += c[1]
    return acc


def fire_at(rows, u, tag=None):
    acc = counts_at(rows, u)
    if tag: SIDE['rates'][f'{tag}@u{u}'] = {k: dict(calls=acc[k][0], fired=acc[k][1], pct=round(100 * acc[k][1] / max(1, acc[k][0]), 2)) for k in OPS}
    return [100 * acc[k][1] / max(1, acc[k][0]) for k in OPS]


def pooled(rows, u, ops, tag=None):
    acc = counts_at(rows, u); n = sum(acc[k][0] for k in ops); f = sum(acc[k][1] for k in ops)
    if tag: SIDE['rates'][f'{tag}@u{u}:{"".join(ops)}'] = dict(calls=n, fired=f, pct=round(100 * f / max(1, n), 2))
    return 100 * f / max(1, n)


def dev_at(rows, u, key='dev'):
    for r in rows:
        if key in r and r[key].get('u') == u:
            d = r[key]; t = [k for k in d if k.endswith('_acc')][0]; return d[t]


s0, s1, s2, nofl, flo, lro = load('CWinOx_s0'), load('CWinOx_s1'), load('CWinOx_s2'), load('CWinO_s0'), load('CWinOfloor_s0'), load('CWinOlr_s0')
lro1 = load('CWinOlr_s1')
fig, (a, b, c) = plt.subplots(1, 3, figsize=(17.5, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))

U = [25, 100, 300]; cols = [PALE, MID, GREEN]; w = 0.26
for j, (u, col) in enumerate(zip(U, cols)):
    v = fire_at(s0, u, 'CWinOx_s0'); xs = [i + (j - 1) * w for i in range(4)]
    a.bar(xs, v, w, color=col, edgecolor=GREEN, linewidth=0.6, label=f'updates {u - WIN + 1}-{u}')
    for x, y in zip(xs, v): a.text(x, y + 1.5, f'{y:.0f}', ha='center', fontsize=8)
a.set_xticks(range(4)); a.set_xticklabels([f'{o}\n{"broken" if o in "+−" else "exact"}' for o in OPS])
a.set_ylim(0, 105); a.set_ylabel('calls let through by the learned trigger (%)')
a.set_title('RL learns where the tool is right', loc='left', fontsize=12, pad=20)
a.text(0, 1.01, 'exact on × ÷, wrong digits on + −; seed 0; rollout calls pooled over 10 updates', transform=a.transAxes, fontsize=8.5, color=INK2, va='bottom')
a.legend(frameon=False, fontsize=9, loc='upper left')

labs = ['no\ncalculator', 'hand\ntrigger', 'learned\ntrigger']
for j, (rows, sd) in enumerate(((s0, 's0'), (s1, 's1'), (s2, 's2'))):
    v = [dev_at(rows, 300, 'dev_hostonly'), dev_at(rows, 300, 'dev_cw_hand'), dev_at(rows, 300)]
    xs = [i + (j - 1) * 0.27 for i in range(3)]
    b.bar(xs, v, 0.27, color=[PALE, PALE, [GREEN, MID, '#8cc7ae'][j]], edgecolor=GREEN, linewidth=0.6, label=f'seed {j}')
    for x, y in zip(xs, v): b.text(x, y + 1, f'{y:.0f}', ha='center', fontsize=8)
b.set_xticks(range(3)); b.set_xticklabels(labs, fontsize=9); b.set_ylim(0, 80)
b.set_ylabel('dev accuracy after 300 RL updates (%)')
b.set_title('Calling it blindly is a disaster', loc='left', fontsize=12, pad=20)
b.text(0, 1.01, 'bars = seeds 0, 1, 2 (left to right); same trained host within a seed; dev n = 800', transform=b.transAxes, fontsize=8.5, color=INK2, va='bottom')

cases = [('no floor\nlr 1e-2\n', nofl), ('floor\nlr 1e-2\n', flo), ('no floor\nlr 1e-3\nseed 0', lro), ('no floor\nlr 1e-3\nseed 1', lro1), ('floor\nlr 1e-3\nseed 0', s0), ('floor\nlr 1e-3\nseed 1', s1), ('floor\nlr 1e-3\nseed 2', s2)]
xs = range(len(cases))
tags = ['CWinO_nofloor_s0', 'CWinO_flooronly_s0', 'CWinO_lronly_s0', 'CWinO_lronly_s1', 'CWinOx_s0', 'CWinOx_s1', 'CWinOx_s2']
md = [pooled(r, 300, ['×', '÷'], t) for (_, r), t in zip(cases, tags)]; asb = [pooled(r, 300, ['+', '−'], t) for (_, r), t in zip(cases, tags)]
c.bar([x - 0.18 for x in xs], md, 0.36, color=GREEN, label='× ÷ (exact)')
c.bar([x + 0.18 for x in xs], asb, 0.36, color=PALE, edgecolor=GREEN, linewidth=0.6, label='+ − (broken)')
for x, y in zip(xs, md): c.text(x - 0.18, y + 1.5, f'{y:.0f}', ha='center', fontsize=8)
for x, y in zip(xs, asb): c.text(x + 0.18, y + 1.5, f'{y:.0f}', ha='center', fontsize=8, color=INK2)
c.set_xticks(list(xs)); c.set_xticklabels([n for n, _ in cases], fontsize=9); c.set_ylim(0, 105)
c.set_ylabel('calls let through, updates 291-300 (%)')
c.set_title('It needs small steps', loc='left', fontsize=12, pad=20)
c.text(0, 1.01, 'lr = in-wire learning rate; floor = fires ≥ 5% of the time; pooled by calls', transform=c.transAxes, fontsize=8.5, color=INK2, va='bottom')
c.legend(frameon=False, fontsize=9, loc='upper left')
for ax in (a, b, c):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
plt.tight_layout()
out = os.path.join(OUTD, 'fig_competence_map.png'); plt.savefig(out, bbox_inches='tight'); plt.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
json.dump(SIDE, open(out.replace('.png', '.json'), 'w'), indent=1)
print('wrote', out, 'and sidecar json')
