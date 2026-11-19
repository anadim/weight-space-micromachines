"""CW-full s0: in-wire veto rate by operand size over RL (greedy dev-800). Data: wiring agent report 2026-11-18 (rl_wiring NOTEBOOK)."""
from make_figs import *
from make_figs3 import legend_rows
u = [25, 50, 64, 100, 125, 150, 175, 200, 225, 250]
V = {"1-2 digits": [97, 95, 93, 69, 36, 52, 25, 38, 16, 18], "3-4 digits": [90, 81, 77, 24, 12, 20, 7, 8, 1, 3],
     "5-6 digits": [44, 21, 16, 0, 0, 16, 2, 1, 0, 0], "7+ digits": [40, 21, 9, 2, 2, 10, 0, 1, 0, 0]}
cols = ["#b9bec6", "#7aa892", "#3f8f68", BLUE]
fig = plt.figure(figsize=(16, 10))
title(fig, "RL redesigns when the calculator fires",
      "a trainable in-wire can veto each calculator call; % vetoed, by number size")
ax = fig.add_axes([0.08, 0.13, 0.70, 0.60])
for (nm, v), c in zip(V.items(), cols):
    ax.plot(u, v, "-o", color=c, lw=6 if nm.startswith("1-2") or nm.startswith("7") else 4, ms=11)
    ax.text(256, v[-1] + {"1-2 digits": 4, "3-4 digits": 12, "5-6 digits": 6, "7+ digits": 0}[nm], nm, fontsize=21, color=c if c != "#b9bec6" else INK2,
            va="center", fontweight="bold" if nm.startswith("7") else "normal")
ax.set_xlim(20, 255); ax.set_ylim(-4, 103)
ax.set_xticks([25, 50, 100, 150, 200, 250]); ax.set_yticks([0, 25, 50, 75, 100])
ax.tick_params(labelsize=19, length=0); ax.set_xlabel("RL updates", fontsize=21); ax.set_ylabel("calls vetoed (%)", fontsize=21)
for yv in (25, 50, 75, 100): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
for s in ax.spines.values(): s.set_visible(False)
fig.text(0.08, 0.025, "Qwen3-0.6B-Base, GRPO on GSM8K-style problems, frozen exact calculator; one seed",
         fontsize=16, color=INK2)
save(fig, "27_cw_veto.png")
