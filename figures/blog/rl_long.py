"""Long RL run (calculator arm, H100, seed 0, 500 updates): reward + dev with calculator on/off, split by half.
Data: <notes>/rl_curves/long_a2/steps.jsonl"""
from make_figs import *
import json, os, numpy as np
D = os.path.join(os.environ.get("RL_CURVES", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "rl", "results", "curves")), "long_a2", "steps.jsonl")
R, on, off = {}, {}, {}
for l in open(D):
    j = json.loads(l)
    if "reward" in j: R[j["u"]] = j["reward"]
    if "dev" in j: on[j["dev"]["u"]] = j["dev"]
    if "dev_host_curve" in j: off[j["dev_host_curve"]["u"]] = j["dev_host_curve"]
us = sorted(R); r = np.array([R[u] for u in us]); k = 16
sm = np.convolve(r, np.ones(k) / k, mode="valid")
fig = plt.figure(figsize=(22, 9.5))
title(fig, "500 updates of RL with the calculator attached",
      "GRPO on Qwen3-0.6B-Base, lr 1e-6, one seed (H100); dev = 800 held-out problems, greedy")
axes = [fig.add_axes([0.05 + i * 0.325, 0.13, 0.26, 0.52]) for i in range(3)]
a = axes[0]
a.plot(us, r * 100, color=GRAY, lw=1.5, alpha=0.7)
a.plot(us[k - 1:], sm * 100, color=BLUE, lw=6)
a.set_title("training reward\n(% of rollouts correct)", loc="left", fontsize=22, fontweight="bold", pad=14)
for ax, key, nm in ((axes[1], "gsm", "normal GSM8K problems"), (axes[2], "scaled", "the same problems, bigger numbers")):
    u1 = sorted(on); u0 = sorted(off)
    y1 = [on[u]["dev_" + key] for u in u1]; y0 = [off[u]["devhost_" + key] for u in u0]
    ax.plot(u1, y1, color=BLUE, lw=6, marker="o", ms=11)
    ax.plot(u0, y0, color=INK2, lw=5, marker="o", ms=10, ls=(0, (3, 2)))
    ax.text(u1[-1], y1[-1] + 4, f"{y1[-1]:.0f}", ha="center", va="bottom", fontsize=21, color=BLUE, fontweight="bold")
    ax.text(u0[-1], y0[-1] - 4, f"{y0[-1]:.0f}", ha="center", va="top", fontsize=21, color=INK2)
    ax.set_title(f"dev accuracy:\n{nm}", loc="left", fontsize=22, fontweight="bold", pad=14)
for ax in axes:
    ax.set_xlim(0, 505); ax.set_ylim(0, 90); ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_yticks([0, 25, 50, 75]); ax.tick_params(labelsize=17, length=0); ax.set_xlabel("RL updates", fontsize=19)
    for yv in (25, 50, 75): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
from matplotlib.lines import Line2D
fig.legend([Line2D([0], [0], color=BLUE, lw=6), Line2D([0], [0], color=INK2, lw=5, ls=(0, (3, 2)))],
           ["calculator on", "same model, calculator switched off"], loc="upper left", bbox_to_anchor=(0.045, 0.835),
           ncol=2, frameon=False, fontsize=20)
fig.text(0.05, 0.015, "Dashed: the same trained model with the calculator switched off. On normal problems the two match; on big numbers the calculator is worth ~27 points.",
         fontsize=17, color=INK2)
save(fig, "26_rl_long.png")
