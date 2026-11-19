"""Blog figure: (left) accuracy ladder, calculator off vs on, before/after RL; (right) how RL changed the host's writing."""
import json, io, contextlib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
with contextlib.redirect_stdout(io.StringIO()):
    from analyze import T, A, N
    from fmt_probe import feats
    import aR_analysis as AR
A = dict(A); A['aR0_on'] = AR.A['aR0_on']
INK, INK2, GRID, GREEN, PALE = '#222222', '#6b7280', '#e5e7eb', '#118659', '#e7f3ee'
GREY = '#c9cdd3'
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': 13, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID})
acc = lambda k: 100 * sum(r['ok'] for r in T[k]) / N
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.22))
# ---- left: accuracy, off vs on per host
hosts = [('base', 'base_off', 'base_on'), ('RL without\ncalculator', 'a1_off', 'a1_on'), ('RL with\ncalculator', 'a2_off', 'a2_on')]
w = 0.36
for j, (lab, off, on) in enumerate(hosts):
    for dx, k, col in ((-w / 2, off, GREY), (w / 2, on, GREEN)):
        v = acc(k); ax1.bar(j + dx, v, w * 0.92, color=col, zorder=3)
        ax1.text(j + dx, v + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=14, fontweight='bold', color=INK if col == GREEN else INK2)
ax1.set_xticks(range(3)); ax1.set_xticklabels([h[0] for h in hosts], fontsize=13, color=INK)
ax1.set_ylim(0, 75); ax1.set_yticks([0, 20, 40, 60]); ax1.yaxis.grid(True, color=GRID, zorder=0)
for s in ('top', 'right', 'left'): ax1.spines[s].set_visible(False)
ax1.tick_params(length=0)
ax1.set_title('Dev accuracy (%), 800 problems, greedy', loc='left', fontsize=15, fontweight='bold', pad=26)
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color=GREY, label='calculator off'), Patch(color=GREEN, label='calculator on')], loc='lower left',
           bbox_to_anchor=(0, 0.99), ncol=2, frameon=False, fontsize=12, labelcolor=INK2, handlelength=1, handleheight=1, borderaxespad=0)
# ---- right: writing habits
def share(k, f):
    e = [q for x in A[k] for q in x['eqs']]; return 100 * sum(f(q) for q in e) / len(e)
cfg = [('base', 'base_on'), ('RL', 'a1_on'), ('RL + calc', 'a2_on'), ('RL + wrong calc', 'aR0_on')]
mets = [('answered by\nthe calculator', lambda q: q['called']), ('unit words written\nbetween operands', lambda q: feats(q)['units'])]
cols = [PALE, '#9fcfb8', GREEN, INK2]
for m, (mlab, f) in enumerate(mets):
    for j, (lab, k) in enumerate(cfg):
        v = share(k, f); x = m + (j - 1.5) * 0.21
        ax2.bar(x, v, 0.19, color=cols[j], zorder=3, edgecolor=GREEN if j == 0 else 'none', linewidth=1)
        ax2.text(x, v + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color=INK)
ax2.set_xticks([0, 1]); ax2.set_xticklabels([m[0] for m in mets], fontsize=13, color=INK)
ax2.set_ylim(0, 100); ax2.set_yticks([0, 25, 50, 75]); ax2.yaxis.grid(True, color=GRID, zorder=0)
for s in ('top', 'right', 'left'): ax2.spines[s].set_visible(False)
ax2.tick_params(length=0)
ax2.set_title('How RL changed the writing (% of equations)', loc='left', fontsize=15, fontweight='bold', pad=26)
ax2.legend(handles=[Patch(facecolor=cols[j], edgecolor=GREEN if j == 0 else cols[j], label=cfg[j][0]) for j in range(4)], loc='lower left',
           bbox_to_anchor=(0, 0.99), ncol=4, columnspacing=1.0, handletextpad=0.5, frameon=False, fontsize=12, labelcolor=INK2, handlelength=1, handleheight=1, borderaxespad=0)
fig.savefig('fig_rl_calc_use.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig('fig_rl_calc_use.svg', bbox_inches='tight', facecolor='white')
print('ok')
