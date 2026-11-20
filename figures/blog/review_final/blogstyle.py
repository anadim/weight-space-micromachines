"""Shared blog style for the review re-renders (2026-11-19): same rules as blog_figs_rl/make_blog_figs_rl.py.
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
TITLE, LABEL, TICK, VAL = 22, 18, 16, 16
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': TICK, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK2, 'axes.edgecolor': INK2, 'axes.linewidth': 1.5,
                     'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'axes.labelsize': LABEL,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                     'hatch.linewidth': 2.0, 'svg.fonttype': 'none'})
DPI, SIZE = 100, (16, 7)


def style(ax, title, ylabel, ylim):
    ax.set_title(title, loc='left', fontsize=TITLE, fontweight='bold', pad=16)
    ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=1.5); ax.set_axisbelow(True)
    ax.tick_params(axis='both', width=1.5, length=6)


def save(fig, name, tight=True):
    if tight: fig.tight_layout(w_pad=4)
    p = os.path.join(HERE, name)
    fig.savefig(p + '.png', dpi=DPI); fig.savefig(p + '.svg'); plt.close(fig); print('wrote', p + '.png/.svg')
