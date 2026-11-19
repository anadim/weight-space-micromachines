"""Figures for the Weight-Space Micromachines blog, draft V0.
Every number below is copied from calc/NOTEBOOK.md (2026-11-15/16 entries) or the pod logs quoted there.
Run: python3 make_figs.py  -> figs/*.png (1600 px wide, for X)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID, BG = "#222222", "#6b7280", "#e5e7eb", "#ffffff"
GRAY = "#b9bec6"            # frozen model, no machines
BLUE = "#118659"            # + machines (the protagonist)
ORANGE = "#6b7280"          # Qwen2.5-7B
AQUA = "#222222"            # Qwen2.5-14B

from matplotlib import font_manager as _fm
import glob as _glob
for _f in _glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "*.ttf")):
    _fm.fontManager.addfont(_f)

plt.rcParams.update({
    "font.family": "Avenir Next", "font.size": 24, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
})

def title(fig, t, sub=None):
    fig.text(0.04, 0.94, t, fontsize=38, fontweight="bold", va="top")
    if sub:
        fig.text(0.04, 0.865, sub, fontsize=24, color=INK2, va="top")

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=100)
    plt.close(fig)

# ---------------------------------------------------------------- 1. hero: 12-task mean
def hero():
    fig = plt.figure(figsize=(16, 9))
    title(fig, "A frozen 0.6B model with nine written machines", "BIG-Bench Hard, 12 algorithmic tasks, official 3-shot prompts")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.66])
    rows = [("Qwen3-0.6B-Base", 43.0, GRAY), ("Qwen2.5-7B", 76.0, ORANGE),
            ("Qwen2.5-14B", 82.2, AQUA), ("Qwen3-0.6B-Base\n+ written machines", 90.1, BLUE)]
    for i, (lab, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.62, color=c, edgecolor=BG, linewidth=2)
        ax.text(v + 1.2, y, f"{v:.1f}", va="center", fontsize=40, fontweight="bold",
                color=INK)
        ax.text(-1.5, y, lab, va="center", ha="right", fontsize=27,
                fontweight="bold" if c == BLUE else "normal")
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    fig.subplots_adjust(left=0.3)
    ax.set_position([0.32, 0.08, 0.62, 0.70])
    save(fig, "1_hero.png")

# ---------------------------------------------------------------- 2. per task dumbbell
TASKS = [  # name, 0.6B base, 0.6B + machines, 7B, 14B
    ("Multistep arithmetic", 47.2, 97.6, 90.0, 95.2),
    ("Navigate",             63.2, 99.6, 97.6, 98.8),
    ("Boolean expressions",  82.8, 98.4, 91.6, 94.8),
    ("Web of lies",          60.4, 97.6, 99.2, 100.0),
    ("Temporal sequences",   26.0, 98.4, 71.2, 86.8),
    ("Tracking, 3 objects",  42.4, 100.0, 90.0, 82.4),
    ("Tracking, 5 objects",  21.6, 100.0, 66.8, 78.0),
    ("Tracking, 7 objects",  11.6, 78.8, 46.0, 59.2),
    ("Deduction, 3 objects", 56.0, 100.0, 80.0, 95.6),
    ("Deduction, 5 objects", 28.8, 81.6, 54.0, 62.0),
    ("Deduction, 7 objects", 11.2, 49.6, 38.4, 42.4),
    ("Object counting",      65.2, 79.2, 87.2, 91.2),
]

def per_task():
    fig = plt.figure(figsize=(16, 13))
    title(fig, "Task by task", "Same 12 BBH tasks. The bar runs from the frozen 0.6B to the 0.6B with machines.")
    ax = fig.add_axes([0.33, 0.07, 0.62, 0.70])
    n = len(TASKS)
    for i, (name, b, m, s7, s14) in enumerate(TASKS):
        y = n - 1 - i
        ax.plot([b, m], [y, y], color=BLUE, alpha=0.28, lw=10, solid_capstyle="round", zorder=1)
        ax.scatter([s7], [y], s=260, marker="D", color=ORANGE, edgecolor=BG, linewidth=2, zorder=3)
        ax.scatter([s14], [y], s=260, marker="s", color=AQUA, edgecolor=BG, linewidth=2, zorder=3)
        ax.scatter([b], [y], s=300, color=BG, edgecolor=GRAY, linewidth=4, zorder=4)
        ax.scatter([m], [y], s=420, color=BLUE, edgecolor=BG, linewidth=2, zorder=5)
        ax.text(-3, y, name, ha="right", va="center", fontsize=23)
    for x in (0, 25, 50, 75, 100):
        ax.axvline(x, color=GRID, lw=1.5, zorder=0)
    ax.set_xlim(0, 103); ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=22, length=0)
    ax.set_xlabel("accuracy (%)", fontsize=22)
    # legend row, under the subtitle
    lx, ly = 0.04, 0.82
    items = [("o", BG, GRAY, "0.6B"), ("o", BLUE, BLUE, "0.6B + machines"),
             ("D", ORANGE, ORANGE, "Qwen2.5-7B"), ("s", AQUA, AQUA, "Qwen2.5-14B")]
    lax = fig.add_axes([lx, ly - 0.03, 0.92, 0.04]); lax.axis("off")
    xs = [0.0, 0.14, 0.40, 0.62]
    for (mk, fc, ec, lab), x in zip(items, xs):
        lax.scatter([x + 0.012], [0.5], s=260, marker=mk, color=fc, edgecolor=ec, linewidth=3)
        lax.text(x + 0.03, 0.5, lab, va="center", fontsize=22, color=INK)
    lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    save(fig, "2_per_task.png")

# ---------------------------------------------------------------- 3. calculator + rethink stack
def gsm():
    # <repo>/results/rethink_v2/summary.json + audit_part3 (always-on arms: final_eval/*_alwaysA_*)
    # color = calculator (gray off, green on); texture = rethink (solid none, // detector-gated, xx every time)
    import matplotlib as mpl
    mpl.rcParams["hatch.linewidth"] = 2.2
    fig = plt.figure(figsize=(16, 10.5))
    title(fig, "Two different fixes, and they compose", "frozen Qwen3-0.6B, implants in weights, full test sets")
    ax = fig.add_axes([0.07, 0.10, 0.90, 0.50])
    # order: no calc (none, detector, always), calc (none, detector, always)
    arms = [("as is", GRAY, ""), ("+ rethink when the detector fires", GRAY, "//"), ("+ rethink every time", GRAY, "xx"),
            ("+ calculator", BLUE, ""), ("+ calculator + detector", BLUE, "//"), ("+ calculator + rethink every time", BLUE, "xx")]
    data = {"GSM-Hard": [26.61, 29.04, 29.57, 34.34, 35.18, 37.15], "GSM8K": [52.01, 55.65, 56.56, 53.30, 56.71, 57.54]}
    w = 0.13
    def swatch(axx, x, y, wd, h, c, hatch, transform=None):
        hc = "#7c838d" if c == GRAY else "#cfe9dc"
        kw = dict(facecolor=c, edgecolor=hc, hatch=hatch, linewidth=0)
        if transform is None:
            axx.bar(x, y, width=wd, **kw)
        else:
            fig.patches.append(mpl.patches.Rectangle((x, y), wd, h, transform=transform, **kw))
    for g, (ds, vals) in enumerate(data.items()):
        for k, v in enumerate(vals):
            x = g + (k - 2.5) * (w + 0.015) + (0.03 if k >= 3 else 0)
            swatch(ax, x, v, w, None, arms[k][1], arms[k][2])
            ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=23, fontweight="bold" if k == 4 else "normal")
        ax.text(g, -6.5, ds, ha="center", fontsize=28, fontweight="bold")
    ax.set_xlim(-0.58, 1.6); ax.set_ylim(0, 66)
    ax.set_xticks([]); ax.set_yticks([0, 20, 40, 60]); ax.tick_params(labelsize=20, length=0)
    for yv in (20, 40, 60):
        ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    ax.set_ylabel("accuracy (%)", fontsize=22)
    for k, (a, c, h) in enumerate(arms):
        col, row = k // 3, k % 3
        x0, y0 = 0.07 + col * 0.45, 0.80 - row * 0.052
        swatch(None, x0, y0 - 0.018, 0.024, 0.036, c, h, transform=fig.transFigure)
        fig.text(x0 + 0.032, y0, a, va="center", fontsize=21, fontweight="bold" if k == 4 else "normal")
    fig.text(0.07, 0.635, "color = calculator off / on;  texture = rethink (none, when the detector fires, every time)",
             fontsize=17, color=INK2)
    save(fig, "3_gsm_stack.png")

# ---------------------------------------------------------------- 4. reading the model: cost curve
def gate():
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, "Rethink only when the model looks wrong",
          "Qwen3-0.6B-Base, GSM8K test. A probe on layer 28 decides when to write \"Hmm, wait.\"")
    ax = fig.add_axes([0.09, 0.13, 0.86, 0.64])
    probe = [(0, 52.08), (45, 55.27), (71, 56.18), (85, 56.25), (96, 56.48), (101, 56.56)]
    margin = [(0, 52.08), (41, 54.36), (64, 54.89), (83, 56.03), (96, 56.41), (101, 56.56)]
    ax.plot([0, 101], [52.08, 56.56], ls=(0, (6, 5)), color=GRAY, lw=3.5, zorder=1)
    ax.plot(*zip(*margin), color=ORANGE, lw=3.5, marker="o", ms=11, zorder=2)
    ax.plot(*zip(*probe), color=BLUE, lw=5, marker="o", ms=14, zorder=3)
    ax.scatter([49], [58.30], s=380, facecolor=BG, edgecolor=INK, linewidth=3, zorder=4)
    ax.text(53, 58.30, "perfect detector", va="center", fontsize=23)
    ax.text(18, 56.05, "residual probe", color=BLUE, fontsize=25, fontweight="bold", ha="center")
    ax.text(52, 54.25, "logit margin (nothing fitted)", color=ORANGE, fontsize=23, ha="left", va="top")
    ax.text(24, 52.95, "random trigger", color=INK2, fontsize=23, ha="left", va="top")
    ax.annotate("always rethink", xy=(101, 56.56), xytext=(88, 57.6), fontsize=23, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1.5))
    ax.set_xlim(-3, 108); ax.set_ylim(51.5, 59.2)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([52, 54, 56, 58])
    ax.tick_params(labelsize=21, length=0)
    for yv in (52, 54, 56, 58):
        ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    ax.set_xlabel("extra tokens per problem", fontsize=22)
    ax.set_ylabel("accuracy (%)", fontsize=22)
    save(fig, "4_gate.png")

# ---------------------------------------------------------------- 0. diagram
def diagram():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 90); ax.axis("off")
    ax.text(6, 84, "Implants, not tools", fontsize=38, fontweight="bold", va="top")
    ax.text(6, 76.5, "The model is frozen. The machines are written next to it, in weights.", fontsize=24, color=INK2, va="top")
    # model stack
    ax.add_patch(FancyBboxPatch((8, 12), 62, 54, boxstyle="round,pad=0,rounding_size=3", fc="none", ec=GRAY, lw=3))
    ax.text(39, 61, "Qwen3-0.6B-Base", ha="center", fontsize=26, fontweight="bold")
    ax.text(39, 56.5, "frozen, bit-identical to the download", ha="center", fontsize=19, color=INK2)
    for i in range(6):
        ax.add_patch(FancyBboxPatch((14, 18 + i * 5.6), 50, 3.6, boxstyle="round,pad=0,rounding_size=1", fc=GRID, ec="none"))
    ax.text(39, 13.8, "token in, layers, next token", ha="center", fontsize=18, color=INK2)
    # compute implant
    ax.add_patch(FancyBboxPatch((96, 44), 56, 20, boxstyle="round,pad=0,rounding_size=3", fc=BLUE + "22", ec=BLUE, lw=3))
    ax.text(124, 57.5, "compute machine", ha="center", fontsize=25, fontweight="bold")
    ax.text(124, 51.8, "reads the tokens, writes exact ones", ha="center", fontsize=19, color=INK2)
    ax.text(124, 47, "2595 + 3075 = 5670", ha="center", fontsize=20, family="Menlo", color=BLUE)
    # sensor implant
    ax.add_patch(FancyBboxPatch((96, 12), 56, 20, boxstyle="round,pad=0,rounding_size=3", fc=ORANGE + "22", ec=ORANGE, lw=3))
    ax.text(124, 25.5, "sensor machine", ha="center", fontsize=25, fontweight="bold")
    ax.text(124, 19.8, "reads a hidden state, writes a nudge", ha="center", fontsize=19, color=INK2)
    ax.text(124, 15, "#### 42  Hmm, wait.", ha="center", fontsize=20, family="Menlo", color=ORANGE)
    arr = dict(arrowstyle="-|>", mutation_scale=34, lw=3.5)
    ax.add_patch(FancyArrowPatch((70, 58), (96, 58), color=BLUE, **arr))
    ax.add_patch(FancyArrowPatch((96, 50), (70, 50), color=BLUE, **arr))
    ax.add_patch(FancyArrowPatch((70, 28), (96, 28), color=ORANGE, **arr))
    ax.add_patch(FancyArrowPatch((96, 18), (70, 18), color=ORANGE, **arr))
    ax.text(83, 60, "tokens", ha="center", fontsize=17, color=INK2)
    ax.text(83, 30, "layer 28", ha="center", fontsize=17, color=INK2)
    ax.text(83, 3.5, "Silent unless triggered: when no machine fires, the logits are bit-identical.", ha="center", fontsize=21, color=INK)
    save(fig, "0_diagram.png")

# ---------------------------------------------------------------- 5. adding machines one at a time
STEPS = [(0, "", 43.03), (1, "sum", 44.20), (2, "calculator,\nexpression", 48.40), (3, "truth\ntracker", 51.50), (4, "navigator", 54.53), (5, "boolean", 55.83), (6, "schedule\nreader", 61.87), (7, "swap\ntracker", 78.80), (8, "deduction\nengine", 90.07)]

F27 = (40.37, 61.27)

def adding():
    fig = plt.figure(figsize=(16, 10))
    title(fig, "Add one implant at a time", "Frozen Qwen3-0.6B-Base, mean over the same 12 BBH tasks")
    ax = fig.add_axes([0.08, 0.17, 0.74, 0.62])
    for v, lab, c in ((76.0, "Qwen2.5-7B", ORANGE), (82.2, "Qwen2.5-14B", AQUA)):
        ax.axhline(v, color=c, lw=3, ls=(0, (7, 5)), zorder=2)
        ax.text(8.6, v, lab, va="center", fontsize=23, color=c, fontweight="bold")
    xs = [x for x, _, _ in STEPS]; ys = [y for _, _, y in STEPS]
    cols = [GRAY] + [BLUE] * (len(xs) - 1)
    ax.bar(xs, [y - 30 for y in ys], bottom=30, width=0.66, color=cols, edgecolor=BG, linewidth=2, zorder=3)
    ax.text(0, ys[0] + 1.5, "43", fontsize=26, color=INK, ha="center", va="bottom", zorder=5)
    ax.text(8, ys[-1] + 1.5, "90", fontsize=30, fontweight="bold", color=INK, ha="center", va="bottom", zorder=5)
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(30, 100)
    ax.set_xticks(xs); ax.set_xticklabels(["none"] + [l for _, l, _ in STEPS[1:]], fontsize=17)
    ax.set_yticks([40, 60, 80, 100]); ax.tick_params(labelsize=21, length=0)
    for yv in (40, 60, 80, 100):
        ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    ax.set_ylabel("BBH accuracy (%)", fontsize=22)
    fig.text(0.08, 0.03, "left to right: add one more implant (per-task runs, added up)", fontsize=21, color=INK2)
    save(fig, "5_adding.png")

def full27():
    fig = plt.figure(figsize=(16, 8))
    title(fig, "All 27 BBH tasks", "Official 3-shot prompts, same harness; 15 tasks have no implant yet")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.64])
    rows = [("Qwen3-0.6B-Base", F27[0], GRAY), ("+ nine implants", F27[1], BLUE),
            ("Qwen2.5-7B", 67.3, ORANGE), ("Qwen2.5-14B", 74.45, AQUA)]
    for i, (lab, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.62, color=c, edgecolor=BG, linewidth=2)
        ax.text(v + 1.2, y, f"{int(v + 0.5)}", va="center", fontsize=38, fontweight="bold", color=INK)
        ax.text(1.5, y, lab, va="center", fontsize=26, color=BG if c != GRAY else INK, fontweight="bold")
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axis("off")
    save(fig, "7_full27.png")

# ---------------------------------------------------------------- 6. eyes
def eyes():
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, "Giving Qwen eyes", "Qwen2.5-0.5B, arithmetic on 3\u20134 handwritten MNIST digits per question")
    ax = fig.add_axes([0.09, 0.14, 0.80, 0.62])
    import math
    n = [8, 32, 4000, 16000]
    yes = [78.5, 94.7, 97.67, 97.67]; no = [4.0, 3.7, 93.26, 97.52]
    ax.plot(n, no, color=GRAY, lw=5, marker="o", ms=15, zorder=2)
    ax.plot(n, yes, color=BLUE, lw=6, marker="o", ms=17, zorder=3)
    ax.set_xscale("log")
    ax.text(40, 89, "eyes + calculator implants", color=BLUE, fontsize=27, fontweight="bold")
    ax.text(8.6, 17, "no implants", color=INK2, fontsize=27, fontweight="bold")
    ax.annotate("95 vs 4", xy=(32, 50), fontsize=30, fontweight="bold", color=INK, ha="center")
    ax.set_xticks(n); ax.set_xticklabels(["8", "32", "4,000", "16,000"])
    ax.minorticks_off()
    ax.set_xlim(6, 22000); ax.set_ylim(-3, 105)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=22, length=0)
    for yv in (0, 25, 50, 75, 100):
        ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    ax.set_xlabel("fine-tuning questions", fontsize=23)
    ax.set_ylabel("accuracy (%)", fontsize=23)
    save(fig, "6_eyes.png")

if __name__ == "__main__":
    hero(); per_task(); gsm(); gate(); adding(); eyes(); full27()
    print(sorted(os.listdir(OUT)))
