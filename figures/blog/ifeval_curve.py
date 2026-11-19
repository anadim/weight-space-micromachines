"""IFEval add-one-rule curve, full test set (541), prompt-strict, A100 node runs.
Data: <notes>/ifeval_transfer/results/{c06,c17,c4}_curve_score.txt (== all section)."""
from make_figs import *
import json
C = json.load(open("ifeval_curves.json"))
labs = ["none", "+ no commas", "+ N paragraphs", "+ repeat request", "+ forbid words", "+ para k starts", "+ all caps",
        "+ two answers", "+ lowercase", "+ end phrase", "+ highlights", "+ quotes", "+ bullets", "+ title", "+ P.S."]
fig = plt.figure(figsize=(16, 11))
title(fig, "Add one rule implant at a time",
      "IFEval, all 541 prompts, every instruction followed; the same 14 rules on three Qwen3 sizes")
ax = fig.add_axes([0.09, 0.27, 0.75, 0.52])
x = list(range(15))
for key, nm, c, lw, ms in (("c4", "Qwen3-4B", INK, 6, 13), ("c17", "Qwen3-1.7B", INK2, 6, 13), ("c06", "Qwen3-0.6B", BLUE, 8.5, 17)):
    v = C[key]
    ax.plot(x, v, "-o", color=c, lw=lw, ms=ms, zorder=3)
    ax.text(14.5, v[-1], f"{nm}\n{v[0]:.0f} to {v[-1]:.0f}", va="center", fontsize=25, color=c,
            fontweight="bold" if key == "c06" else "normal")
ax.set_xlim(-0.4, 14.3); ax.set_ylim(50, 90)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=19, rotation=40, ha="right", rotation_mode="anchor")
ax.set_yticks([50, 60, 70, 80, 90]); ax.tick_params(axis="y", labelsize=24, length=0); ax.tick_params(axis="x", length=0)
for yv in (60, 70, 80, 90): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
ax.set_ylabel("accuracy (%)", fontsize=26)
fig.text(0.09, 0.025, "left to right: switch on one more rule (order picked on our own practice prompts, not IFEval)",
         fontsize=21, color=INK2)
save(fig, "22_ifeval_curve.png")
