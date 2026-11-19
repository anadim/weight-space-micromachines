"""rl6_wiring: trainable wiring (CW seed 0). Left: calls vetoed by the learned in-wire, by largest operand, update 25 vs 300.
Right: dev accuracy after 300 updates for hand wiring, learned wiring (CW), new links only (W3e). Data: rl/results/wiring/figdata."""
import os, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, '..', 'results', 'wiring', 'figdata')
INK, INK2, GRID, GREEN, PALE, GREY = '#222222', '#6b7280', '#e5e7eb', '#118659', '#e7f3ee', '#b9bec6'
for f in font_manager.findSystemFonts():
    if 'Avenir' in f:
        try: font_manager.fontManager.addfont(f)
        except Exception: pass
TITLE, LABEL, TICK, VAL = 22, 18, 16, 16
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': TICK, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK2, 'axes.edgecolor': INK2, 'axes.linewidth': 1.5,
    'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'axes.labelsize': LABEL, 'figure.facecolor': 'white'})
load = lambda t: [json.loads(l) for l in open(os.path.join(FD, t + '.jsonl'))]
def dev_at(rows, u):
    for r in rows:
        if 'dev' in r and r['dev'].get('u') == u:
            d = r['dev']; return d[[k for k in d if k.endswith('_acc')][0]]
B = ['1-2', '3-4', '5-6', '7+']; veto = {}
for r in load('CW_s0'):
    d = r.get('dev', {})
    if d.get('u') in (25, 300) and d.get('dev_veto_by_digits'):
        v = d['dev_veto_by_digits']; veto[d['u']] = [100 * v[b][1] / max(1, v[b][0]) for b in B]
def style(ax, t, yl, ylim):
    ax.set_title(t, loc='left', fontsize=TITLE, fontweight='bold', pad=16); ax.set_ylabel(yl); ax.set_ylim(*ylim)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=1.5); ax.set_axisbelow(True); ax.tick_params(width=1.5, length=6)
fig, (L, R) = plt.subplots(1, 2, figsize=(16, 7))
w = 0.38
for j, (u, kw) in enumerate([(25, dict(color=GREY, edgecolor=GREY)), (300, dict(color=GREEN, edgecolor=GREEN))]):
    xs = [i + (j - 0.5) * w for i in range(4)]
    L.bar(xs, veto[u], w, linewidth=2, **kw)
    for x, y in zip(xs, veto[u]): L.text(x, y + 1.5, f'{y:.0f}', ha='center', fontsize=VAL, fontweight='bold')
L.text(0 - 0.5 * w, veto[25][0] + 9, 'early', ha='center', fontsize=VAL, color=INK2, fontweight='bold')
L.text(0 + 0.5 * w, veto[300][0] + 9, 'late', ha='center', fontsize=VAL, color=GREEN, fontweight='bold')
L.set_xticks(range(4)); L.set_xticklabels([b + ' digits' for b in B])
style(L, 'RL skips the calculator on small numbers', 'calls vetoed (%)', (0, 115))
arms = [('REF_a2long', 'hand\nwiring', GREEN), ('CW_s0', 'learned\nwiring', GREEN), ('W3e_s0', 'new links\nonly', GREY)]
for i, (t, lab, c) in enumerate(arms):
    v = dev_at(load(t), 300); R.bar(i, v, 0.6, color=c)
    R.text(i, v + 0.8, f'{v:.0f}', ha='center', fontsize=VAL, fontweight='bold')
R.set_xticks(range(3)); R.set_xticklabels([a[1] for a in arms])
style(R, 'Learned wiring ends about level', 'dev accuracy (%)', (40, 80))
fig.tight_layout(w_pad=4)
for ext in ('png', 'svg'): fig.savefig(os.path.join(HERE, 'rl6_wiring.' + ext), dpi=100)
print(veto, [dev_at(load(a[0]), 300) for a in arms])
