"""BBH all 27 tasks, integration v2 (one Qwen3-0.6B host, 8 implants, 432-row fine-tune).
Data: <notes>/bbh_integrate/TABLES_v1_vs_v2.md (real BBH, official 3-shot prompts, one run per arm)."""
from make_figs import *
from matplotlib.lines import Line2D
# task: (nice name, tool?, base, FT no tools, FT + implants)
D = [
("boolean_expressions","boolean expressions",1,82.8,66.8,80.0),
("causal_judgement","causal judgement",0,47.1,45.5,32.1),
("date_understanding","date understanding",0,54.4,38.8,14.4),
("disambiguation_qa","disambiguation QA",0,34.0,28.0,24.0),
("dyck_languages","Dyck languages",0,0.0,1.2,1.2),
("formal_fallacies","formal fallacies",0,52.4,43.6,15.6),
("geometric_shapes","geometric shapes",0,28.4,18.4,3.2),
("hyperbaton","hyperbaton",0,60.4,47.2,16.4),
("ld5","logical deduction (5)",1,29.2,22.0,69.6),
("ld7","logical deduction (7)",1,16.4,15.2,69.2),
("ld3","logical deduction (3)",1,57.2,30.8,73.6),
("movie","movie recommendation",0,35.2,26.0,26.0),
("multistep","multistep arithmetic",1,45.2,16.4,100.0),
("navigate","navigate",1,61.6,42.0,72.8),
("object_counting","object counting",1,66.0,39.2,57.2),
("penguins","penguins in a table",0,52.1,28.8,18.5),
("colored","colored objects",0,57.2,32.8,24.8),
("ruin","ruin names",0,26.4,27.2,19.2),
("salient","translation errors",0,38.4,24.0,20.8),
("snarks","snarks",0,51.7,38.8,27.5),
("sports","sports understanding",0,46.4,46.8,46.8),
("temporal","temporal sequences",1,24.8,77.6,99.6),
("ts5","shuffled objects (5)",1,22.8,16.8,95.2),
("ts7","shuffled objects (7)",1,14.8,10.0,96.4),
("ts3","shuffled objects (3)",1,40.4,30.0,98.4),
("lies","web of lies",1,58.0,90.4,100.0),
("word_sorting","word sorting",0,3.2,3.2,3.2),
]
import numpy as np
tool = sorted([d for d in D if d[2]], key=lambda d: -(d[5]-d[3]))
rest = sorted([d for d in D if not d[2]], key=lambda d: -(d[5]-d[3]))
def m(rows, i): return np.mean([r[i] for r in rows])
fig = plt.figure(figsize=(16, 24))
title(fig, "All 27 BIG-Bench Hard tasks", "Qwen3-0.6B, one model with 8 implants, real BBH test (run once)")
ax = fig.add_axes([0.33, 0.05, 0.63, 0.78])
ys, labels, y = [], [], 0
rows = []
def block(name, rr):
    global y
    rows.append(("H", name, y)); y += 1.1
    for r in rr:
        rows.append(("R", r, y)); y += 1
    y += 0.9
block(f"Tasks with an implant (12)   ·   average {m(tool,3):.0f}% to {m(tool,5):.0f}%", tool)
block(f"Tasks with no implant (15)   ·   average {m(rest,3):.0f}% to {m(rest,5):.0f}%", rest)
ax.set_ylim(y - 0.6, -0.4)
for kind, r, yy in rows:
    if kind == "H":
        fig.text(0.04, ax.transData.transform((0, yy + 0.25))[1] / fig.dpi / fig.get_figheight(), r, fontsize=24, fontweight="bold", color=INK, va="center")
        continue
    _, nm, _, b, f, t = r
    ax.axhline(yy, color=GRID, lw=1, zorder=0)
    ax.plot([b, t], [yy, yy], color=BLUE if t >= b else INK2, lw=3, alpha=0.5, zorder=1, solid_capstyle="round")
    ax.scatter([f], [yy], s=170, facecolor=BG, edgecolor=INK2, lw=2.5, zorder=3)
    ax.scatter([b], [yy], s=200, color=GRAY, zorder=4)
    ax.scatter([t], [yy], s=230, color=BLUE, zorder=5)
    ax.text(-0.02, yy, nm, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=21, color=INK)
ax.set_ylim(y - 0.6, -0.4)
ax.set_xlim(-2, 102)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.tick_params(axis="x", labelsize=20, length=0)
ax.xaxis.tick_top()
for xv in (0, 25, 50, 75, 100): ax.axvline(xv, color=GRID, lw=1, zorder=0)
h = [Line2D([], [], marker="o", ls="", ms=15, color=GRAY, label="model alone"),
     Line2D([], [], marker="o", ls="", ms=14, mfc=BG, mec=INK2, mew=2.5, label="fine-tuned, implants off"),
     Line2D([], [], marker="o", ls="", ms=16, color=BLUE, label="fine-tuned, implants on")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=21, bbox_to_anchor=(0.5, 0.0))
save(fig, "16_bbh27.png")
print(m(tool,3), m(tool,4), m(tool,5), m(rest,3), m(rest,4), m(rest,5))
