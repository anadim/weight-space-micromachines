"""v2 (2026-11-18): IFEval bars + per-rule redrawn from the SEEDED all-541 run (thinking off, greedy,
max_new 2048): prompt-strict 57.86 -> 72.09, instruction-strict 67.63 -> 79.62, and the per-rule
instruction-strict passes (ALL, 0.6B). Source tables: ifeval/ (see REPRODUCE.md section 5).
v1 plotted an older unseeded run (58.2 -> 72.8).
Third batch of blog figures (2026-11-17): bar versions for IFEval, BBH, topic blocker; same style as make_figs.py.
Data sources:
  IFEval      : ifeval/heldout_results/ (ODD = never-opened half, n=299; IFBench n=298)
  BBH         : bbh_generic/ integration v2 tables (real BBH, one run per arm)
  topic block : topic_fence/ q17 math/ml tables (Qwen3-1.7B; block = LLM judge says the reply does not help;
                implant = probe_text_stop; 'normal requests' = Alpaca retention set, n=567)
"""
from make_figs2 import *

LIGHT = "#d1d5db"
DARK = "#374151"


def legend_rows(fig, items, x, y, dy=0.048, bold_last=True, bh=0.032):
    W, H = fig.get_size_inches()
    for k, (a, c) in enumerate(items):
        yy = y - k * dy
        fig.patches.append(FancyBboxPatch((x, yy - bh / 2), bh * H / W * 0.9, bh, boxstyle="round,pad=0.002",
                                          transform=fig.transFigure, color=c))
        fig.text(x + 0.024, yy, a, va="center", fontsize=22,
                 fontweight="bold" if (bold_last and k == len(items) - 1) else "normal")


def grouped(ax, data, cols, w, labfs=26, lab_y=-9, fmt="{:.0f}"):
    n = len(cols)
    for g, (name, vals) in enumerate(data):
        for k, v in enumerate(vals):
            x = g + (k - (n - 1) / 2) * (w + 0.02)
            ax.bar(x, v, width=w, color=cols[k], edgecolor=BG, linewidth=2, zorder=2)
            ax.text(x, v + 1.5, fmt.format(v), ha="center", fontsize=labfs,
                    fontweight="bold" if k == n - 1 else "normal")
        ax.text(g, lab_y, name, ha="center", va="top", fontsize=24, fontweight="bold", linespacing=1.1)
    ax.set_xlim(-0.55, len(data) - 0.45); ax.set_ylim(0, 108)
    ax.set_xticks([]); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=20, length=0)
    grid(ax, (25, 50, 75, 100))


# ------------------------------------------------------------ IFEval headline
def ifeval_bars():
    fig = plt.figure(figsize=(16, 10.5))
    title(fig, "Rule implants on IFEval",
          "frozen Qwen3-0.6B, 14 rule implants, all 541 IFEval prompts")
    legend_rows(fig, [("model alone", GRAY), ("with the 14 rule implants", BLUE)], 0.07, 0.765)
    ax = fig.add_axes([0.07, 0.17, 0.90, 0.52])
    data = [("every instruction in the prompt\nfollowed (main IFEval score)", [57.86, 72.09]),
            ("single instructions followed\n(each scored on its own)", [67.63, 79.62])]
    grouped(ax, data, [GRAY, BLUE], 0.32, labfs=24, lab_y=15.5, fmt="{:.0f}")
    for ln in list(ax.lines): ln.remove()
    ax.set_ylim(20, 86); ax.set_yticks([20, 40, 60, 80])
    grid(ax, (40, 60, 80))
    for t in ax.texts:
        if "\n" in t.get_text(): t.set_fontsize(20)
    ax.set_ylabel("accuracy (%)", fontsize=22)
    save(fig, "17_ifeval_bars_v2.png")


# ------------------------------------------------------------ IFEval per rule
def ifeval_rules():
    R = [("no commas", 31, 62, 66), ("forbidden words", 23, 42, 49), ("N paragraphs", 13, 21, 27),
         ("paragraph k starts with X", 4, 11, 12), ("two responses", 11, 19, 24), ("exactly N bullets", 22, 29, 31),
         ("repeat the request first", 26, 34, 41), ("wrap in quotes", 33, 36, 41), ("end with a phrase", 24, 26, 26),
         ("all lowercase", 28, 31, 39), ("all capitals", 15, 16, 25), ("title in << >>", 35, 36, 37),
         ("highlight N sections", 45, 46, 48), ("add a P.S.", 26, 26, 26)]
    R.sort(key=lambda r: -(r[2] - r[1]) / r[3])
    fig = plt.figure(figsize=(16, 15))
    title(fig, "Each rule, one by one",
          "instructions followed, full IFEval test set (541 prompts), frozen Qwen3-0.6B")
    legend_rows(fig, [("model alone", GRAY), ("with all 14 rule implants on", BLUE)], 0.07, 0.80, dy=0.032, bh=0.021)
    ax = fig.add_axes([0.36, 0.04, 0.58, 0.66])
    h = 0.36
    for i, (nm, a, b, n) in enumerate(R):
        y = i
        for k, (v, c) in enumerate(((a, GRAY), (b, BLUE))):
            yy = y + (k - 0.5) * (h + 0.04)
            ax.barh(yy, 100 * v / n, height=h, color=c, zorder=2)
        ax.text(100 * b / n + 1.2, y + 0.5 * (h + 0.04), f"{b}/{n}", va="center", fontsize=19, fontweight="bold")
        ax.text(100 * a / n + 1.2, y - 0.5 * (h + 0.04), f"{a}/{n}", va="center", fontsize=17, color=INK2)
        ax.text(-2, y, nm, ha="right", va="center", fontsize=22)
    ax.set_ylim(len(R) - 0.4, -0.6); ax.set_xlim(0, 112)
    ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(labelsize=19, length=0); ax.xaxis.tick_top()
    for xv in (25, 50, 75, 100): ax.axvline(xv, color=GRID, lw=1.5, zorder=0)
    save(fig, "18_ifeval_rules_v2.png")


# ------------------------------------------------------------ BBH, the three arms
BBH = [  # nice name, tool task?, base, FT no tools, FT + implants  (integration v2)
    ("boolean expressions", 1, 82.8, 66.8, 80.0), ("causal judgement", 0, 47.1, 45.5, 32.1),
    ("date understanding", 0, 54.4, 38.8, 14.4), ("disambiguation QA", 0, 34.0, 28.0, 24.0),
    ("Dyck languages", 0, 0.0, 1.2, 1.2), ("formal fallacies", 0, 52.4, 43.6, 15.6),
    ("geometric shapes", 0, 28.4, 18.4, 3.2), ("hyperbaton", 0, 60.4, 47.2, 16.4),
    ("logical deduction (5)", 1, 29.2, 22.0, 69.6), ("logical deduction (7)", 1, 16.4, 15.2, 69.2),
    ("logical deduction (3)", 1, 57.2, 30.8, 73.6), ("movie recommendation", 0, 35.2, 26.0, 26.0),
    ("multistep arithmetic", 1, 45.2, 16.4, 100.0), ("navigate", 1, 61.6, 42.0, 72.8),
    ("object counting", 1, 66.0, 39.2, 57.2), ("penguins in a table", 0, 52.1, 28.8, 18.5),
    ("colored objects", 0, 57.2, 32.8, 24.8), ("ruin names", 0, 26.4, 27.2, 19.2),
    ("translation errors", 0, 38.4, 24.0, 20.8), ("snarks", 0, 51.7, 38.8, 27.5),
    ("sports understanding", 0, 46.4, 46.8, 46.8), ("temporal sequences", 1, 24.8, 77.6, 99.6),
    ("shuffled objects (5)", 1, 22.8, 16.8, 95.2), ("shuffled objects (7)", 1, 14.8, 10.0, 96.4),
    ("shuffled objects (3)", 1, 40.4, 30.0, 98.4), ("web of lies", 1, 58.0, 90.4, 100.0),
    ("word sorting", 0, 3.2, 3.2, 3.2)]
ARMS = [("model alone", GRAY), ("fine-tuned, implants off", INK2), ("fine-tuned, implants on", BLUE)]


def _mean(rows, i):
    return sum(r[i] for r in rows) / len(rows)


def bbh_bars():
    tool = [r for r in BBH if r[1]]; rest = [r for r in BBH if not r[1]]
    fig = plt.figure(figsize=(16, 11))
    title(fig, "Eight implants on BIG-Bench Hard",
          "Qwen3-0.6B fine-tuned on 432 of our own examples, real BBH test, run once")
    legend_rows(fig, ARMS, 0.07, 0.775)
    ax = fig.add_axes([0.07, 0.14, 0.90, 0.47])
    data = [("12 tasks\nwith an implant", [_mean(tool, i) for i in (2, 3, 4)]),
            ("15 tasks\nwith no implant", [_mean(rest, i) for i in (2, 3, 4)]),
            ("all 27 tasks", [_mean(BBH, i) for i in (2, 3, 4)])]
    grouped(ax, data, [c for _, c in ARMS], 0.25)
    ax.set_ylabel("accuracy (%)", fontsize=22)
    save(fig, "19_bbh_bars.png")


def bbh_tasks():
    tool = sorted([r for r in BBH if r[1]], key=lambda r: -(r[4] - r[2]))
    rest = sorted([r for r in BBH if not r[1]], key=lambda r: -(r[4] - r[2]))
    fig = plt.figure(figsize=(16, 30))
    title(fig, "All 27 BIG-Bench Hard tasks", "Qwen3-0.6B, same three arms, real BBH test, run once")
    legend_rows(fig, ARMS, 0.07, 0.85, dy=0.016, bh=0.011)
    ax = fig.add_axes([0.33, 0.03, 0.62, 0.76])
    h = 0.26; y = 0; ticks = []
    def block(name, rows):
        nonlocal y
        ax.text(-0.50, y - 0.2, name, transform=ax.get_yaxis_transform(), fontsize=24, fontweight="bold", va="center")
        y += 0.9
        for r in rows:
            for k, (v, (_, c)) in enumerate(zip(r[2:], ARMS)):
                yy = y + (k - 1) * (h + 0.03)
                ax.barh(yy, max(v, 0.4), height=h, color=c, zorder=2)
                if k == 2:
                    ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=17, fontweight="bold")
            ax.text(-1.5, y, r[0], ha="right", va="center", fontsize=21)
            y += 1.05
        y += 0.8
    block("Tasks with an implant (12)", tool)
    block("Tasks with no implant (15)", rest)
    ax.set_ylim(y - 0.6, -0.9); ax.set_xlim(0, 106)
    ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(labelsize=19, length=0); ax.xaxis.tick_top()
    for xv in (25, 50, 75, 100): ax.axvline(xv, color=GRID, lw=1.5, zorder=0)
    save(fig, "16_bbh27.png")


# ------------------------------------------------------------ BBH, add one implant at a time (per-task runs, added up)
def bbh_adding():
    # tasks per implant; order = gain each gave on OUR OWN fresh-test sets (not BBH): interval 83, track 68,
    # order 42, truth 36, walk 24, expr/calc/bool (multistep 32, boolean 10, counting 6)
    groups = [("schedules\n(interval)", ["temporal sequences"]),
              ("swaps\n(track)", ["shuffled objects (3)", "shuffled objects (5)", "shuffled objects (7)"]),
              ("ordering\n(order)", ["logical deduction (3)", "logical deduction (5)", "logical deduction (7)"]),
              ("who lies\n(truth)", ["web of lies"]),
              ("directions\n(walk)", ["navigate"]),
              ("arithmetic,\nlogic (3)", ["multistep arithmetic", "boolean expressions", "object counting"])]
    by = {r[0]: r for r in BBH}
    tool = [r for r in BBH if r[1]]
    cur = {r[0]: r[3] for r in tool}  # fine-tuned, no implants
    vals = [_mean(tool, 2), _mean(tool, 3)]
    for _, ts in groups:
        for t in ts: cur[t] = by[t][4]
        vals.append(sum(cur.values()) / len(cur))
    labs = ["model\nalone", "fine-tuned,\nno implants"] + [g for g, _ in groups]
    fig = plt.figure(figsize=(16, 10))
    title(fig, "Add one implant at a time",
          "Qwen3-0.6B, mean over the 12 BBH tasks that have an implant")
    ax = fig.add_axes([0.08, 0.2, 0.89, 0.6])
    for i, (v, l) in enumerate(zip(vals, labs)):
        c = GRAY if i == 0 else (INK2 if i == 1 else BLUE)
        ax.bar(i, v, width=0.68, color=c, zorder=2)
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=26, fontweight="bold" if i == len(vals) - 1 else "normal")
        ax.text(i, -4, l, ha="center", va="top", fontsize=19, color=INK2, linespacing=1.1)
    ax.set_xlim(-0.6, len(vals) - 0.4); ax.set_ylim(0, 100)
    ax.set_xticks([]); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=20, length=0)
    grid(ax, (25, 50, 75, 100)); ax.set_ylabel("BBH accuracy (%)", fontsize=22)
    fig.text(0.08, 0.035, "left to right: switch on one more implant (per-task runs, added up)", fontsize=19, color=INK2)
    save(fig, "20_bbh_adding.png")
    return vals


# ------------------------------------------------------------ topic blocker
def topic_blocker():
    # one gray for every baseline, told apart by texture; green = the implant
    import matplotlib as mpl
    mpl.rcParams["hatch.linewidth"] = 2.0
    HC = "#7c838d"
    methods = [("model alone", GRAY, ""), ("system prompt: \"don't help with X\"", GRAY, "//"),
               ("strict system prompt", GRAY, "xx"), ("keyword list", GRAY, ".."), ("the blocker implant", BLUE, "")]
    D = {"math": {"blocked: requests to do X": [11.0, 8.9, 73.6, 63.3, 98.7],
                  "blocked by mistake: near-misses": [1.4, 5.7, 82.9, 57.1, 1.4],
                  "blocked by mistake: normal requests": [5.5, 7.9, 85.5, 7.2, 9.5]},
         "ML development": {"blocked: requests to do X": [4.9, 1.6, 61.5, 93.4, 98.4],
                            "blocked by mistake: near-misses": [2.9, 2.9, 15.7, 24.3, 8.6],
                            "blocked by mistake: normal requests": [5.5, 4.8, 20.3, 7.1, 5.5]}}
    fig = plt.figure(figsize=(16, 17))
    title(fig, "Block a topic by reading the model's state",
          "frozen Qwen3-1.7B, prompts written separately from the training prompts")
    for k, (nm, c, h) in enumerate(methods):
        y0 = 0.80 - k * 0.03
        fig.patches.append(mpl.patches.Rectangle((0.07, y0 - 0.011), 0.024, 0.022, transform=fig.transFigure,
                                                 facecolor=c, edgecolor=HC, hatch=h, linewidth=0))
        fig.text(0.07 + 0.032, y0, nm, va="center", fontsize=22, fontweight="bold" if k == 4 else "normal")
    for p, (topic, d) in enumerate(D.items()):
        ax = fig.add_axes([0.07, 0.41 - p * 0.34, 0.90, 0.22])
        names = [("requests to\ndo " + ("math" if p == 0 else "ML")), "near-misses\n(should NOT block)",
                 "normal requests\n(should NOT block)"]
        for g, (key, vals) in enumerate(d.items()):
            for k, v in enumerate(vals):
                x = g + (k - 2) * 0.165
                ax.bar(x, max(v, 0.6), width=0.15, facecolor=methods[k][1], edgecolor=HC, hatch=methods[k][2],
                       linewidth=0, zorder=2)
                ax.text(x, v + 2, f"{v:.0f}", ha="center", fontsize=18, fontweight="bold" if k == 4 else "normal")
            ax.text(g, -10, names[g], ha="center", va="top", fontsize=21, fontweight="bold", linespacing=1.1)
        ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 110)
        ax.set_xticks([]); ax.set_yticks([0, 50, 100]); ax.tick_params(labelsize=18, length=0)
        grid(ax, (50, 100)); ax.set_ylabel("% of replies blocked", fontsize=18)
        ax.set_title(f"Block {topic}", loc="left", fontsize=26, fontweight="bold", pad=10)
    save(fig, "21_topic_blocker.png")


if __name__ == "__main__":
    ifeval_bars(); ifeval_rules()
