"""Blog figure v2 (2026-11-18): only the right panel of fig_rl_calc_use (how RL changed the writing).
The v1 left panel (accuracy, calculator off/on) was 64-update / one-seed dev data and contradicted the 300-update
test result, so it is dropped. The remaining bars are also 64-update dev data, one seed per arm; the title says so.
Same data path as make_fig.py (analyze.T/A, fmt_probe.feats, aR_analysis.A). Hatching instead of extra shades."""
import io, contextlib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
with contextlib.redirect_stdout(io.StringIO()):
    from analyze import T, A, N
    from fmt_probe import feats
    import aR_analysis as AR
A = dict(A); A['aR0_on'] = AR.A['aR0_on']
INK, INK2, GRID, GREEN, GREY = '#222222', '#6b7280', '#e5e7eb', '#118659', '#b9bec6'
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': 16, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID, 'hatch.linewidth': 2.0})
def share(k, f):
    e = [q for x in A[k] for q in x['eqs']]; return 100 * sum(f(q) for q in e) / len(e)
cfg = [('base', 'base_on'), ('RL', 'a1_on'), ('RL + calc', 'a2_on'), ('RL + wrong calc', 'aR0_on')]
# (facecolor, hatch colour, hatch)
sty = [(GREY, GREY, ''), ('white', GREY, '///'), (GREEN, GREEN, ''), ('white', GREEN, '///')]
mets = [('answered by\nthe calculator', lambda q: q['called']), ('unit words written\nbetween operands', lambda q: feats(q)['units'])]
fig, ax = plt.subplots(figsize=(9.5, 6.2))
vals = {}
for m, (mlab, f) in enumerate(mets):
    for j, (lab, k) in enumerate(cfg):
        v = share(k, f); vals[(m, lab)] = v; x = m + (j - 1.5) * 0.21
        fc, hc, h = sty[j]
        ax.bar(x, v, 0.19, color=fc, edgecolor=hc, hatch=h, linewidth=2, zorder=3)
        ax.text(x, v + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=16, fontweight='bold', color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels([m[0] for m in mets], fontsize=17, color=INK)
ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75]); ax.yaxis.grid(True, color=GRID, lw=1.5, zorder=0)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.tick_params(length=0, labelsize=15)
fig.text(0.02, 0.98, 'How RL changed the writing (% of equations)', fontsize=20, fontweight='bold', va='top')
fig.text(0.02, 0.915, 'after 64 RL updates; dev split, 800 problems, greedy, one seed per arm', fontsize=15, color=INK2, va='top')
ax.legend(handles=[Patch(facecolor=sty[j][0], edgecolor=sty[j][1], hatch=sty[j][2], linewidth=1.5, label=cfg[j][0]) for j in range(4)],
          loc='lower left', bbox_to_anchor=(0, 0.99), ncol=4, columnspacing=1.2, handletextpad=0.5, frameon=False,
          fontsize=15, labelcolor=INK2, handlelength=1.4, handleheight=1.1, borderaxespad=0)
fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.14)
fig.savefig('fig_rl_calc_use_v2.png', dpi=200, facecolor='white')
fig.savefig('fig_rl_calc_use_v2.svg', facecolor='white')
for k, v in vals.items(): print(k, round(v, 2))
