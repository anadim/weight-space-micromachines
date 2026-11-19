"""BBH add-one-tool curve, 0.6B v2 FT host, all 27 tasks, strict. Data: <notes>/bbh_transfer/curve_table.md"""
from make_figs import *
labs = ["no implants\n(fine-tuned)", "+ schedules", "+ swaps", "+ ordering", "+ who lies", "+ directions", "+ arithmetic\n& logic"]
t12 = [38.10, 40.60, 59.90, 71.63, 72.47, 75.00, 84.33]
a27 = [33.61, 33.31, 39.33, 43.51, 43.61, 44.08, 48.36]
n15 = [30.01, 27.49, 22.88, 21.00, 20.52, 19.35, 19.58]
fig = plt.figure(figsize=(16, 10))
title(fig, "Add one implant at a time, all 27 BBH tasks",
      "Qwen3-0.6B fine-tuned to call its implants; each step switches on one more implant")
ax = fig.add_axes([0.08, 0.17, 0.68, 0.62])
x = list(range(7))
for v, nm, c, lw, ms in ((t12, "12 tasks\nwith an implant", BLUE, 6, 13),
                         (n15, "15 tasks\nwith no implant", INK2, 3.5, 9)):
    ax.plot(x, v, "-o", color=c, lw=lw, ms=ms, zorder=3)
    ax.text(6.3, v[-1], f"{nm}\n{v[0]:.0f} to {v[-1]:.0f}", va="center", fontsize=19, color=c,
            fontweight="bold" if c == BLUE else "normal")
ax.set_xlim(-0.3, 6.2); ax.set_ylim(0, 100)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=16, linespacing=1.0)
ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(axis="y", labelsize=20, length=0); ax.tick_params(axis="x", length=0)
for yv in (25, 50, 75, 100): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
ax.set_ylabel("accuracy (%)", fontsize=22)
fig.text(0.08, 0.02, "real BBH runs for every step (not added up); order picked on our own practice questions.\nAll 27 tasks together: 34 to 48 (the model alone, not fine-tuned: 41).", fontsize=17, color=INK2)
save(fig, "23_bbh_curve27.png")
