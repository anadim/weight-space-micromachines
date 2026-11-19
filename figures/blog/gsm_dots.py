import make_figs as M
from make_figs import *
def gsm_dots():
    fig = plt.figure(figsize=(16, 9))
    title(fig, "Two different fixes, and they stack", "Frozen Qwen3-0.6B-Base, full test sets, greedy")
    arms = ["as is", "+ calculator", "+ rethink", "+ both"]
    cols = [GRAY, BLUE, "#7cc3a0", "#03623f"]
    data = {"GSM-Hard": [26.99, 34.50, 29.26, 37.23], "GSM8K": [51.93, 54.06, 56.10, 57.62]}
    for p, (ds, vals) in enumerate(data.items()):
        ax = fig.add_axes([0.08 + p * 0.47, 0.11, 0.42, 0.58])
        ax.axhline(vals[0], color=GRAY, lw=2.5, ls=(0, (5, 4)), zorder=1)
        for k, v in enumerate(vals):
            ax.scatter(k, v, s=900, color=cols[k], edgecolor=BG, linewidth=3, zorder=3)
            ax.text(k, v + 2.2, f"{v:.0f}" if False else f"{v:.1f}", ha="center", fontsize=26,
                    fontweight="bold" if k == 3 else "normal", color=INK)
        ax.set_xlim(-0.5, 3.5); ax.set_ylim(20, 62)
        ax.set_xticks(range(4)); ax.set_xticklabels(arms, fontsize=19)
        ax.set_yticks([20, 30, 40, 50, 60]); ax.tick_params(labelsize=20, length=0)
        if p: ax.set_yticklabels([])
        for yv in (20, 30, 40, 50, 60):
            ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(ds, fontsize=28, fontweight="bold", loc="left", pad=14)
        if p == 0: ax.set_ylabel("accuracy (%)", fontsize=22)
    save(fig, "3_gsm_dots.png")
gsm_dots()
