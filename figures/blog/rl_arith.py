"""Is the model learning arithmetic? Dev accuracy vs RL updates on the normal vs big-number halves, H100, lr 1e-6,
mean of 2 seeds. Data: <notes>/rl_curves/h300/*.jsonl"""
from make_figs import *
import json, os, numpy as np
from matplotlib.lines import Line2D
D = os.path.join(os.environ.get("RL_CURVES", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "rl", "results", "curves")), "h300")
def load(name, key):
    out = {}
    for l in open(f"{D}/{name}.jsonl"):
        j = json.loads(l)
        if key in j:
            v = j[key]; p = "dev" if key == "dev" else "devhost"
            out[v["u"]] = (v[p + "_gsm"], v[p + "_scaled"])
    return out
U = [0, 50, 100, 150, 200, 250, 300]
def mean(runs, key, half):
    ds = [load(r, key) for r in runs]
    us = [u for u in U if all(u in d for d in ds)]
    return us, [np.mean([d[u][half] for d in ds]) for u in us]
lines = [(["a1_lr1e-6_s0", "a1_u300_s1"], "dev", INK2, "-", "RL, no implant"),
         (["a2_lr1e-6_s0", "a2_u300_s1"], "dev", BLUE, "-", "RL with calculator (calculator on)"),
         (["a2_lr1e-6_s0", "a2_u300_s1"], "dev_host_curve", BLUE, (0, (3, 2)), "same model, calculator switched off")]
fig = plt.figure(figsize=(20, 9.5))
title(fig, "Is the model learning arithmetic itself?",
      "dev accuracy over RL, split into normal GSM8K problems and the same problems with bigger numbers (mean of 2 seeds)")
fig.legend([Line2D([0], [0], color=c, lw=5, ls=ls) for _, _, c, ls, _ in lines], [n for *_, n in lines],
           loc="upper left", bbox_to_anchor=(0.045, 0.835), ncol=3, frameon=False, fontsize=19)
for i, (half, nm) in enumerate(((0, "normal GSM8K problems"), (1, "same problems, bigger numbers"))):
    ax = fig.add_axes([0.06 + i * 0.48, 0.12, 0.40, 0.52])
    for runs, key, c, ls, n in lines:
        us, v = mean(runs, key, half)
        ax.plot(us, v, color=c, ls=ls, lw=6 if key == "dev" and c == BLUE else 5, marker="o", ms=10)
        if half == 1: ax.text(us[-1] + 8, v[-1], f"{v[-1]:.0f}", va="center", fontsize=21, color=c, fontweight="bold" if c == BLUE and key == "dev" else "normal")
    if half == 0: ax.text(300, 67, "all three ≈ 77", ha="right", fontsize=20, color=INK2)
    ax.set_title(nm, loc="left", fontsize=23, fontweight="bold", pad=12)
    ax.set_xlim(-8, 325); ax.set_ylim(15, 85); ax.set_xticks(U); ax.set_yticks([25, 50, 75])
    ax.tick_params(labelsize=18, length=0); ax.set_xlabel("RL updates", fontsize=20)
    for yv in (25, 50, 75): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
save(fig, "28_rl_arith.png")
