"""v3 copy (2026-11-20) for blog_figs_rl/v3: sizes raised to the bars_v2 convention (title 22, labels 22, ticks 20, values 20, legend 20).
Shared blog style for the review re-renders (2026-11-19): same rules as rl/figures/make_blog_figs_rl.py.
Avenir Next, green palette, titles 22 pt, labels 18 pt, ticks 16 pt, thick lines, direct labels, white bg, 1600 x 700 px."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID, GREEN, PALE, GREY, MID = '#222222', '#6b7280', '#e5e7eb', '#118659', '#e7f3ee', '#b9bec6', '#5fae8f'
for f in ('/System/Library/Fonts/Avenir Next.ttc',):
    if os.path.exists(f): font_manager.fontManager.addfont(f)
TITLE, LABEL, TICK, VAL, LG = 22, 22, 20, 20, 20
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': TICK, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK2, 'axes.edgecolor': INK2, 'axes.linewidth': 1.5,
                     'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'axes.labelsize': LABEL,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                     'hatch.linewidth': 2.0, 'svg.fonttype': 'none', 'lines.linewidth': 3.5})
DPI, SIZE = 100, (16, 7)


def style(ax, title, ylabel, ylim):
    ax.set_title(title, loc='left', fontsize=TITLE, fontweight='bold', pad=16)
    ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=1.5); ax.set_axisbelow(True)
    ax.tick_params(axis='both', width=1.5, length=6)


def save(fig, name, tight=True):
    if tight: fig.tight_layout(w_pad=4)
    p = os.path.join(os.path.dirname(HERE), name + '_v6')
    fig.savefig(p + '.png', dpi=DPI); fig.savefig(p + '.svg'); plt.close(fig); print('wrote', p + '.png/.svg')


from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def legend(ax, cols, names, loc='upper left', ncol=None, edges=None, **kw):
    """Colour legend (bars_v2 convention). edges: optional edge colours for outlined bars."""
    edges = edges or [None] * len(cols)
    h = [Patch(facecolor=c, edgecolor=e or c, linewidth=2.5, label=n) for c, e, n in zip(cols, edges, names)]
    return ax.legend(handles=h, loc=loc, fontsize=LG, frameon=False, handlelength=1.4, handleheight=1.0,
                     ncol=ncol or len(names), columnspacing=1.2, **kw)


def xcats(ax, labels, bold=True):
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=TICK, fontweight='bold' if bold else 'normal', linespacing=1.05)
    ax.tick_params(axis='x', length=0, pad=10)


def val(ax, x, y, txt, strong=True, dy=1.2):
    ax.text(x, y + dy, txt, ha='center', va='bottom', fontsize=VAL, fontweight='bold' if strong else 'normal', color=INK if strong else INK2)


def dots(ax, x, ys, spread=0.06, ms=10):
    xs = [x + (i - (len(ys) - 1) / 2) * spread for i in range(len(ys))]
    ax.plot(xs, ys, 'o', color=INK, ms=ms, zorder=6, mec='white', mew=1.2)
