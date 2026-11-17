import json, os, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-0.6B-Base"
RES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")
INK, SEC, GRID, GREEN, PALE = "#222222", "#6b7280", "#e5e7eb", "#118659", "#e7f3ee"
plt.rcParams.update({"font.family": ["Avenir Next", "DejaVu Sans"], "text.color": INK, "axes.labelcolor": INK, "xtick.color": SEC,
                     "ytick.color": SEC, "axes.edgecolor": SEC, "axes.spines.top": False, "axes.spines.right": False,
                     "font.weight": "regular", "axes.titleweight": "demibold"})
cells = [("muladd_mid", "(a*b)+c, mid"), ("muladd_hard", "(a*b)+c, hard"), ("mulsub_mid", "(a*b)-c, mid"),
         ("lost_mid", "word problem (pens lost)")]
cells = [(k, t) for k, t in cells if os.path.exists(f"{RES}/{k}_{TAG}.json")]
fig, axs = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 3.6), sharey=True)
axs = [axs] if len(cells) == 1 else axs
for ax, (k, title) in zip(axs, cells):
    d = json.load(open(f"{RES}/{k}_{TAG}.json"))
    Ls = sorted(int(x[1:].split("_")[0]) for x in d if x.startswith("L") and x.endswith("_comp_true"))
    acc = [100 * d[f"L{l}_comp_true"]["acc"] for l in Ls]
    fol = [100 * d[f"L{l}_comp_corr"]["follow"] for l in Ls]
    mea = [100 * d[f"L{l}_comp_mean"]["acc"] for l in Ls]
    ax.fill_between(Ls, 0, acc, color=PALE, zorder=0)
    ax.axhline(100 * d["text_oracle_true"]["acc"], color=INK, ls="--", lw=1, zorder=1)
    ax.axhline(100 * d["plain"]["acc"], color=SEC, ls=":", lw=1.4, zorder=1)
    ax.plot(Ls, acc, "o-", color=GREEN, lw=2, ms=4.5, label="compiled write: accuracy", zorder=3)
    ax.plot(Ls, fol, "s--", color=INK, lw=1.2, ms=3.5, mfc="white", label="corrupted value: output follows it", zorder=3)
    ax.plot(Ls, mea, "^-", color=SEC, lw=1.2, ms=3.5, label="mean-value control", zorder=3)
    ax.text(Ls[-1], 100 * d["text_oracle_true"]["acc"] + 1.5, "text oracle", color=INK, fontsize=8, ha="right", va="bottom")
    ax.text(Ls[0] + 0.3, 100 * d["plain"]["acc"] + 1.5, "plain model", color=SEC, fontsize=8, ha="left", va="bottom")
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_xlabel("layer L the write enters (of 28)", fontsize=9)
    ax.set_xticks(Ls); ax.tick_params(labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(-3, 105)
axs[0].set_ylabel("% of 300 problems", fontsize=9)
axs[0].legend(fontsize=7.5, frameon=False, loc="center left")
plt.tight_layout()
out = os.path.join(HERE if len(sys.argv) <= 2 else RES, f"latent_inweights_{TAG}.png")
plt.savefig(out, dpi=160, facecolor="white")
print(out)
