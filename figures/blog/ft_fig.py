from make_figs import *
def ft():
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, "Or fine-tune Qwen to call the implant", "Qwen2.5-0.5B, fine-tuned on GSM8K solutions, 5 seeds each")
    ax = fig.add_axes([0.07, 0.12, 0.90, 0.56])
    arms = ["plain solutions", "calls, model writes the answer", "calls + calculator implant"]
    cols = [GRAY, INK2, BLUE]
    data = {"GSM8K": ([33.6, 35.2, 35.1], [0.6, 0.2, 1.2]),
            "GSM-Hard": ([13.4, 14.9, 23.3], [0.4, 0.9, 0.6]),
            "arithmetic,\nnew phrasings": ([50.8, 54.5, 99.0], [3.4, 1.5, 0.4])}
    w = 0.25
    for g, (ds, (vals, errs)) in enumerate(data.items()):
        for k, (v, e) in enumerate(zip(vals, errs)):
            x = g + (k - 1) * (w + 0.02)
            ax.bar(x, v, width=w, color=cols[k], edgecolor=BG, linewidth=2, zorder=2)
            ax.errorbar(x, v, yerr=e, color=INK, lw=2, capsize=6, zorder=3)
            ax.text(x, v + e + 1.5, f"{v:.0f}", ha="center", fontsize=26, fontweight="bold" if k == 2 else "normal")
        ax.text(g, -9, ds, ha="center", va="top", fontsize=25, fontweight="bold", linespacing=1.1)
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 108)
    ax.set_xticks([]); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=20, length=0)
    for yv in (25, 50, 75, 100):
        ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    ax.set_ylabel("accuracy (%)", fontsize=22)
    xs = [0.07, 0.30, 0.64]
    for k, (a, c) in enumerate(zip(arms, cols)):
        fig.patches.append(FancyBboxPatch((xs[k], 0.755), 0.018, 0.032, boxstyle="round,pad=0.002", transform=fig.transFigure, color=c))
        fig.text(xs[k] + 0.024, 0.771, a, va="center", fontsize=22)
    save(fig, "8_finetune.png")
ft()
