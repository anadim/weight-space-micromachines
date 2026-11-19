"""RL reward + dev curves, GSM8K-train + scaled, Qwen3-0.6B-Base, lr 1e-6. Data: <notes>/rl_curves/*/steps.jsonl
(copied from A100 node A100 runs/abl; long run from H100 pod runs/long)."""
from make_figs import *
import json, os, numpy as np
D = os.environ.get("RL_CURVES", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "rl", "results", "curves"))  # rl/results/curves/<run>/steps.jsonl
def load(r):
    R, V = {}, {}
    for l in open(f"{D}/{r}/steps.jsonl"):
        j = json.loads(l)
        if "reward" in j: R[j["u"]] = j["reward"]
        if "dev" in j: V[j["dev"]["u"]] = j["dev"]["dev_acc"]
    return R, V
def smooth(y, k=8):
    y = np.asarray(y); out = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.concatenate([np.full(k - 1, np.nan), out])
arms = [("a2", "calculator", BLUE, "-", 6), ("a1", "no implant", INK2, "-", 5), ("aR", "wrong calculator", INK2, (0, (3, 2)), 5)]
fig = plt.figure(figsize=(20, 9.5))
title(fig, "RL with the calculator implant attached",
      "GRPO on Qwen3-0.6B-Base, GSM8K-style problems, lr 1e-6; mean of 2 seeds")
axL = fig.add_axes([0.06, 0.13, 0.33, 0.58]); axR = fig.add_axes([0.57, 0.13, 0.38, 0.58])
for key, nm, c, ls, lw in arms:
    Rs, Vs = zip(*(load(f"{key}_lr1e-6_s{s}") for s in (0, 1)))
    us = sorted(Rs[0]); r = np.mean([[R[u] for u in us] for R in Rs], 0)
    axL.plot(us, smooth(r) * 100, color=c, ls=ls, lw=lw)
    axL.text(66, smooth(r)[-1] * 100, nm, va="center", fontsize=20, color=c if c != INK2 else INK2,
             fontweight="bold" if key == "a2" else "normal")
    vu = sorted(Vs[0]); v = np.mean([[V[u] for u in vu] for V in Vs], 0)
    axR.plot(vu, v, color=c, ls=ls, lw=lw, marker="o", ms=11)
    axR.text(vu[-1] + 3, v[-1] - (4 if key == "a2" else 0), f"{v[-1]:.0f}", va="center", fontsize=22, color=c,
             fontweight="bold" if key == "a2" else "normal")
R, V = load("long_a2"); vu = sorted(V)
axR.plot(vu, [V[u] for u in vu], color=BLUE, ls=(0, (1, 1.6)), lw=4, marker="o", ms=8, alpha=0.8)
axR.text(200, 74, "calculator, longer run (1 seed): 69 at 200", ha="center", va="bottom", fontsize=18, color=BLUE)
axL.set_title("training reward (% of rollouts correct, smoothed)", loc="left", fontsize=22, fontweight="bold", pad=14)
axR.set_title("held-out dev accuracy (%)", loc="left", fontsize=22, fontweight="bold", pad=14)
for ax, xl in ((axL, (0, 64)), (axR, (-5, 255))):
    ax.set_xlim(*xl); ax.set_ylim(0, 80); ax.set_xlabel("RL updates", fontsize=20)
    ax.set_yticks([0, 25, 50, 75]); ax.tick_params(labelsize=18, length=0)
    for yv in (25, 50, 75): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
axR.set_xticks([0, 64, 100, 150, 200, 250])
save(fig, "25_rl_curves.png")
