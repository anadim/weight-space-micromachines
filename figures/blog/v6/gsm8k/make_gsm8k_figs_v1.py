"""GSM8K section-5 figures (2026-11-20), blog style of figs_v6 (blogstyle.py + bars_v2/bbh_bars_v6 conventions).
Model: Qwen3-0.6B-Base after RL with the calculator, frozen, T=1; every implant in weights, one generation call.
Numbers handed over by the blog lead (GSM8K test n=1319; GSM-Hard). Writes gsm8k_*.png/.svg next to this file."""
import os, sys, glob
OUTDIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(OUTDIR), "src")
sys.path.insert(0, SRC)
from matplotlib import font_manager as _fm
from blogstyle import *
DARK = '#0b5e3e'


def out(fig, name):
    fig.tight_layout()
    p = os.path.join(OUTDIR, name)
    fig.savefig(p + '.png', dpi=DPI); fig.savefig(p + '.svg'); plt.close(fig); print('wrote', p + '.png/.svg')


def f1_bars():
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax, 'Implants on GSM8K (Qwen3-0.6B-Base, frozen, all implants in weights)', 'accuracy (%)', (0, 112))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    cols = [GREY, INK2, GREEN, DARK]
    groups = [('GSM8K test', [67.7, 69.5, 72.3, 79.8]), ('GSM-Hard (big numbers)', [39.4, 53.9, 55.2, 63.2])]
    w, n = 0.19, 4
    for g, (_, vs) in enumerate(groups):
        for k, v in enumerate(vs):
            x = g + (k - (n - 1) / 2) * (w + 0.02)
            ax.bar(x, v, w, color=cols[k], zorder=3)
            ax.text(x, v + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=VAL,
                    fontweight='bold' if k == n - 1 else 'normal', color=INK if k == n - 1 else INK2)
    xcats(ax, [g for g, _ in groups]); ax.set_xlim(-0.6, 1.6)
    legend(ax, cols, ['model alone', '+ calculators', '+ sanity check + forget-mask', '+ RL with the stack, 8 tries in one call'], ncol=2)
    out(fig, 'gsm8k_bars')


def f2_ablation():
    # drop in GSM8K test accuracy (points) when one part is switched off; full RL'd stack = 79.76
    R = [('forget-mask', -8.19, -10.16, -6.29), ('both calculators', -2.50, -4.25, -0.76),
         ('voter (plain majority instead)', -1.36, -2.65, -0.15), ('chain2 (long +/− chains)', -1.14, -2.35, 0.15),
         ('calculator', -0.76, -2.27, 0.76), ('sanity check (fires on 1 item)', 0.08, None, None)]
    fig, ax = plt.subplots(figsize=SIZE)
    ax.set_title('Each implant, one by one: switch it off (full stack 79.8% on GSM8K test)', loc='left',
                 fontsize=TITLE, fontweight='bold', pad=16, x=-0.345)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=1.5); ax.set_axisbelow(True)
    for i, (nm, d, lo, hi) in enumerate(R):
        sig = hi is not None and hi < 0
        c = GREEN if sig else GREY
        ax.barh(i, d, 0.6, color=c, zorder=3)
        if lo is not None:
            ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], color=INK, lw=2.5, capsize=6, capthick=2.5, zorder=4)
        left = lo if lo is not None else min(d, 0)
        right = hi if hi is not None else max(d, 0)
        if d < 0:
            ax.text(left - 0.25, i, f'{d:+.1f}'.replace('-', '−'), ha='right', va='center', fontsize=VAL,
                    fontweight='bold' if sig else 'normal', color=INK if sig else INK2)
        else:
            ax.text(right + 0.25, i, f'{d:+.1f}', ha='left', va='center', fontsize=VAL, color=INK2)
    ax.axvline(0, color=INK2, lw=1.5, zorder=2)
    ax.set_yticks(range(len(R))); ax.set_yticklabels([r[0] for r in R], fontsize=TICK, color=INK)
    ax.tick_params(axis='y', length=0, pad=10); ax.tick_params(axis='x', width=1.5, length=6)
    ax.set_ylim(len(R) - 0.5, -0.6); ax.set_xlim(-12, 1.5)
    ax.set_xticks([-10, -8, -6, -4, -2, 0]); ax.set_xticklabels(['−10', '−8', '−6', '−4', '−2', '0'])
    ax.set_xlabel('change in GSM8K test accuracy when switched off (points, 95% CI)')
    legend(ax, [GREEN, GREY], ['CI below zero', 'within noise'], loc='lower left', ncol=1)
    out(fig, 'gsm8k_one_by_one')


def f3_curve():
    labs = ['no implants', '+ calculator', '+ chain2', '+ sanity check\n+ forget-mask', '+ 8 tries\n(retry, voter)', '+ RL with\nthe stack']
    g8 = [67.7, 68.7, 69.5, 72.3, 71.9, 79.8]
    gh = [39.4, None, 53.9, 55.2, 57.5, 63.2]
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax, 'Add one implant at a time (Qwen3-0.6B-Base, frozen)', 'accuracy (%)', (30, 92))
    ax.set_yticks([30, 40, 50, 60, 70, 80, 90])
    ax.axvspan(4.5, 5.35, color=PALE, zorder=0)
    ax.text(5, 90.5, 'after RL', ha='center', va='top', fontsize=VAL, color=GREEN, fontweight='bold')
    for ys, c, lw, ms, nm, strong in ((g8, GREEN, 6, 13, 'GSM8K test', True), (gh, INK2, 3.5, 10, 'GSM-Hard', False)):
        pts = [(i, y) for i, y in enumerate(ys) if y is not None]
        pre = [p for p in pts if p[0] <= 4]
        ax.plot([p[0] for p in pre], [p[1] for p in pre], '-o', color=c, lw=lw, ms=ms, zorder=3)
        ax.plot([4, 5], [ys[4], ys[5]], '--', color=c, lw=lw, zorder=3)
        ax.plot([5], [ys[5]], 'o', color=c, ms=ms, zorder=4)
        for i, y in pts:
            ax.text(i, y + 1.6, f'{y:.1f}' if strong else f'{y:.0f}', ha='center', va='bottom', fontsize=VAL - 2,
                    fontweight='bold' if strong else 'normal', color=INK if strong else INK2)
        ax.text(5.45, ys[5], f'{nm}\n{ys[0]:.0f} to {ys[5]:.0f}', va='center', fontsize=VAL, color=c,
                fontweight='bold' if strong else 'normal', linespacing=1.1)
    ax.set_xlim(-0.3, 6.4); ax.set_xticks(range(6)); ax.set_xticklabels(labs, fontsize=TICK - 2, linespacing=1.0)
    ax.tick_params(axis='x', length=0, pad=10)
    out(fig, 'gsm8k_curve')


if __name__ == '__main__':
    f1_bars(); f2_ablation(); f3_curve()
