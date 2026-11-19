"""Same implants, other models. Data: ifeval_transfer NOTEBOOK (ALL 541 prompt-strict), bbh_transfer RESULTS (12 implant tasks),
SUMMARY.md GSM-Hard calc (Qwen3 Base 0.6/1.7/4B, xfamily SmolLM2-1.7B, Granite-3.3-2B)."""
from make_figs import *
from make_figs3 import legend_rows
panels = [
    ("IFEval (14 rules)", "every instruction followed (%)",
     [("Qwen3\n0.6B", 57.9, 72.1), ("Qwen3\n1.7B", 68.8, 78.2), ("Qwen3\n4B", 80.4, 84.8)]),
    ("BBH (12 implant tasks)", "accuracy (%)",
     [("Qwen3\n0.6B", 43, 84), ("Qwen3\n1.7B", 56, 96), ("Qwen3\n4B", 83, 98)]),
    ("GSM-Hard (calculator)", "accuracy (%)",
     [("Qwen3\n0.6B", 27.3, 35.0), ("Qwen3\n1.7B", 45.4, 51.0), ("Qwen3\n4B", 59.0, 60.4),
      ("SmolLM2\n1.7B", 14.0, 19.9), ("Granite\n3.3-2B", 23.2, 30.4)]),
]
fig = plt.figure(figsize=(22, 10.5))
title(fig, "Same implants, other models", "built and debugged on Qwen3-0.6B; only the reader and the write directions are re-fit for each new model")
legend_rows(fig, [("model alone", GRAY), ("with the implants", BLUE)], 0.05, 0.80, dy=0.05, bh=0.034)
widths = [3, 3, 5]; tot = sum(widths); x0 = 0.05; gap = 0.045; W = 0.93 - gap * 2
for (name, ylab, rows), w in zip(panels, widths):
    ax = fig.add_axes([x0, 0.18, W * w / tot, 0.44]); x0 += W * w / tot + gap
    for i, (lab, a, b) in enumerate(rows):
        for k, (v, c) in enumerate(((a, GRAY), (b, BLUE))):
            xx = i + (k - 0.5) * 0.4
            ax.bar(xx, v, width=0.4, color=c, edgecolor=BG, linewidth=2, zorder=2)
            ax.text(xx, v + 2, f"{v:.0f}", ha="center", fontsize=25, fontweight="bold" if k else "normal",
                    color=INK if k else INK2)
        ax.text(i, -7, lab, ha="center", va="top", fontsize=22, linespacing=1.05)
    ax.set_xlim(-0.6, len(rows) - 0.4); ax.set_ylim(0, 108)
    ax.set_xticks([]); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=20, length=0)
    for yv in (25, 50, 75, 100): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(name, fontsize=26, fontweight="bold", loc="left", pad=14)
save(fig, "24_transfer.png")
