"""Second batch of blog figures (2026-11-17), same style as make_figs.py.
Data sources:
  calc_digits : evidence/item1/*.json  (frozen Qwen3-0.6B-Base, 2-shot, n=100 per cell)
  finetune    : same numbers as ft_fig.py (Qwen2.5-0.5B, 5 seeds), clearer labels
  ifeval      : ifeval/heldout/NOTES.md + runs17 scores (held-out odd keys; Multi-IF English turns 1-3)
  latent      : latent_bbh (not released; figure 12 is not in the blog)
"""
import json, os
from make_figs import *
from matplotlib.patches import FancyBboxPatch, Rectangle

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repository root
H = REPO
GREEN_PALE = "#e7f3ee"


def bar_legend(fig, items, x0, y, gap):
    x = x0
    for lab, c in items:
        fig.patches.append(FancyBboxPatch((x, y - 0.016), 0.018, 0.032, boxstyle="round,pad=0.002",
                                          transform=fig.transFigure, color=c))
        fig.text(x + 0.024, y, lab, va="center", fontsize=22)
        x += gap


def grid(ax, ys):
    for yv in ys:
        ax.axhline(yv, color=GRID, lw=1.5, zorder=0)


# ------------------------------------------------------------ calculator vs number of digits
def calc_digits():
    d = os.path.join(REPO, "evidence/item1")
    add = json.load(open(f"{d}/add_bylen.json")); sub = json.load(open(f"{d}/sub_bylen.json"))
    mul = json.load(open(f"{d}/mult_heatmap.json"))
    n = list(range(1, 10))
    A = [100 * add[f"d{k}"]["base"] for k in n]
    S = [100 * sub[f"d{k}"]["base"] for k in n]
    M = [100 * mul[f"{k}x{k}"]["base"] for k in n]
    impl = [100 * min(add[f"d{k}"]["impl"], sub[f"d{k}"]["impl"], mul[f"{k}x{k}"]["impl"]) for k in n]
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, "Accuracy as the numbers get longer",
          "frozen Qwen3-0.6B, 2 worked examples in the prompt; each point is 100 problems")
    ax = fig.add_axes([0.08, 0.13, 0.74, 0.62])
    for ys, lab in ((A, "adding,\nmodel alone"), (S, "subtracting,\nmodel alone"), (M, "multiplying,\nmodel alone")):
        ax.plot(n, ys, color=GRAY, lw=5, marker="o", ms=12, zorder=2)
        ax.text(9.25, ys[-1], lab, va="center", fontsize=22, color=INK2, linespacing=1.0)
    ax.plot(n, impl, color=BLUE, lw=6, marker="o", ms=14, zorder=3)
    ax.text(9.25, 100, "all three, with\nthe calculator\nimplant", va="center", fontsize=22, color=BLUE,
            fontweight="bold", linespacing=1.05)
    ax.set_xlim(0.7, 9.2); ax.set_ylim(-3, 106)
    ax.set_xticks(n); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=22, length=0)
    grid(ax, (0, 25, 50, 75, 100))
    ax.set_xlabel("digits in each number  (e.g. 4 means 4-digit × 4-digit)", fontsize=22)
    ax.set_ylabel("accuracy (% exactly right)", fontsize=22)
    save(fig, "9_calc_digits.png")


# ------------------------------------------------------------ fine-tune, clearer labels
def finetune():
    fig = plt.figure(figsize=(16, 11))
    title(fig, "Or fine-tune Qwen to make the calls",
          "Qwen2.5-0.5B fine-tuned on the same GSM8K solutions three ways, 5 seeds each")
    ax = fig.add_axes([0.07, 0.14, 0.90, 0.44])
    arms = ["plain solutions, no calls",
            "solutions with calls; the model fills in each result itself",
            "solutions with calls; the calculator implant fills in each result"]
    cols = [GRAY, INK2, BLUE]
    data = {"GSM8K": ([33.6, 35.2, 35.1], [0.6, 0.2, 1.2]),
            "GSM-Hard\n(big numbers)": ([13.4, 14.9, 23.3], [0.4, 0.9, 0.6]),
            "arithmetic,\nnew phrasings": ([50.8, 54.5, 99.0], [3.4, 1.5, 0.4])}
    w = 0.25
    for g, (ds, (vals, errs)) in enumerate(data.items()):
        for k, (v, e) in enumerate(zip(vals, errs)):
            x = g + (k - 1) * (w + 0.02)
            ax.bar(x, v, width=w, color=cols[k], edgecolor=BG, linewidth=2, zorder=2)
            ax.errorbar(x, v, yerr=e, color=INK, lw=2, capsize=6, zorder=3)
            ax.text(x, v + e + 1.5, f"{v:.0f}", ha="center", fontsize=26, fontweight="bold" if k == 2 else "normal")
        ax.text(g, -9, ds, ha="center", va="top", fontsize=25, fontweight="bold", linespacing=1.1)
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 108)
    ax.set_xticks([]); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=20, length=0)
    grid(ax, (25, 50, 75, 100))
    ax.set_ylabel("accuracy (%)", fontsize=22)
    for k, (a, c) in enumerate(zip(arms, cols)):
        y = 0.775 - k * 0.048
        fig.patches.append(FancyBboxPatch((0.07, y - 0.016), 0.018, 0.032, boxstyle="round,pad=0.002",
                                          transform=fig.transFigure, color=c))
        fig.text(0.094, y, a, va="center", fontsize=22, fontweight="bold" if k == 2 else "normal")
    fig.text(0.094, 0.775 - 3 * 0.048, "a call looks like:  <tool_call>48 / 2</tool_call> 24",
             family="monospace", fontsize=18, color=INK2, va="center")
    save(fig, "8_finetune.png")


# ------------------------------------------------------------ IFEval
def ifeval():
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, "Rule-keeper implants on instruction following",
          "frozen models, prompt-level strict accuracy, prompts we never tuned on")
    bar_legend(fig, [("model alone", GRAY), ("with the rule-keeper implants", BLUE)], 0.07, 0.765, 0.25)
    # left: IFEval held-out
    ax = fig.add_axes([0.07, 0.13, 0.36, 0.53])
    groups = [("Qwen3-0.6B", 58.2, 69.6), ("Qwen3-1.7B", 70.6, 75.9)]
    w = 0.36
    for g, (lab, a, b) in enumerate(groups):
        for k, (v, c) in enumerate(((a, GRAY), (b, BLUE))):
            x = g + (k - 0.5) * (w + 0.03)
            ax.bar(x, v, width=w, color=c, edgecolor=BG, linewidth=2, zorder=2)
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=28, fontweight="bold" if k else "normal")
        ax.text(g, -7, lab, ha="center", va="top", fontsize=24, fontweight="bold")
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 100)
    ax.set_xticks([]); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=20, length=0)
    grid(ax, (25, 50, 75, 100))
    ax.set_title("IFEval, held-out half", fontsize=26, color=INK, pad=18, loc="left", fontweight="bold")
    # right: Multi-IF
    ax2 = fig.add_axes([0.53, 0.13, 0.44, 0.53])
    turns = [("turn 1", 59.8, 71.3), ("turn 2", 34.5, 53.6), ("turn 3", 15.5, 28.6)]
    for g, (lab, a, b) in enumerate(turns):
        for k, (v, c) in enumerate(((a, GRAY), (b, BLUE))):
            x = g + (k - 0.5) * (w + 0.03)
            ax2.bar(x, v, width=w, color=c, edgecolor=BG, linewidth=2, zorder=2)
            ax2.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=28, fontweight="bold" if k else "normal")
        ax2.text(g, -7, lab, ha="center", va="top", fontsize=24, fontweight="bold")
    ax2.set_xlim(-0.6, 2.6); ax2.set_ylim(0, 100)
    ax2.set_xticks([]); ax2.set_yticks([0, 25, 50, 75, 100]); ax2.tick_params(labelsize=20, length=0)
    grid(ax2, (25, 50, 75, 100))
    ax2.set_title("Multi-IF, multi-turn chats (Qwen3-0.6B)", fontsize=26, color=INK, pad=18, loc="left",
                  fontweight="bold")
    save(fig, "10_ifeval.png")


# ------------------------------------------------------------ latent write: what it does
def latent_diagram():
    fig = plt.figure(figsize=(16, 8.4))
    title(fig, "Writing a value into the middle of the model",
          "BBH multi-step arithmetic, answered directly, no chain of thought")
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.80]); ax.set_xlim(0, 100); ax.set_ylim(-3, 52); ax.axis("off")
    toks = ["((", "-5", "+", "9", "*", "-4", "-", "0", ")", "*", "(", "4", "+", "-7", "+", "0", "*", "-5", "))", "="]
    A, B = list(range(1, 8)), list(range(11, 18))
    x0, cw, gap = 10.5, 2.85, 0.4
    xs = [x0 + i * (cw + gap) for i in range(len(toks))]
    rows = [("layer 1", 1), ("layer 2", 2), ("layer 3", 3), ("layer 4", 4), ("\u2026", None), ("layer 28", 28)]
    y0, rh = 12.0, 5.2
    for r, (lab, L) in enumerate(rows):
        y = y0 + r * rh
        ax.text(x0 - 1.2, y + rh * 0.4, lab, ha="right", va="center", fontsize=19, color=INK2)
        if L is None:
            continue
        for i, xc in enumerate(xs):
            hidden = L >= 4 and ((i in A and i != A[0]) or (i in B and i != B[0]))
            written = L == 4 and i in (A[0], B[0])
            if hidden:
                ax.add_patch(FancyBboxPatch((xc, y), cw, rh * 0.78, boxstyle="round,pad=0.12", fc=BG, ec=GRID,
                                            lw=1.5, ls=(0, (3, 3))))
            elif written:
                ax.add_patch(FancyBboxPatch((xc - 0.2, y - 0.2), cw + 0.4, rh * 0.78 + 0.4,
                                            boxstyle="round,pad=0.12", fc=GREEN_PALE, ec=BLUE, lw=2.5, zorder=4))
                ax.text(xc + cw / 2, y + rh * 0.39, "\u221241" if i == A[0] else "\u22123", ha="center",
                        va="center", fontsize=19, color=BLUE, fontweight="bold", zorder=5)
            else:
                ax.add_patch(FancyBboxPatch((xc, y), cw, rh * 0.78, boxstyle="round,pad=0.12", fc="#f3f4f6",
                                            ec=GRID, lw=1.2))
    # tokens
    yt = 4.5
    for i, (t, xc) in enumerate(zip(toks, xs)):
        ec = BLUE if (i in A or i in B) else INK
        ax.add_patch(FancyBboxPatch((xc, yt), cw, 4.4, boxstyle="round,pad=0.15", fc=BG, ec=ec, lw=2))
        ax.text(xc + cw / 2, yt + 2.2, t, ha="center", va="center", fontsize=20, family="monospace")
    for span, lab in ((A, "bracket A = \u221241"), (B, "bracket B = \u22123")):
        ax.plot([xs[span[0]], xs[span[-1]] + cw], [yt - 1.0, yt - 1.0], color=BLUE, lw=2)
        ax.text((xs[span[0]] + xs[span[-1]] + cw) / 2, yt - 4.0, lab, ha="center", fontsize=19, color=BLUE)
    # answer
    xa = xs[-1] + cw / 2
    ytop = y0 + 5 * rh + rh * 0.78
    ax.annotate("", xy=(xa, ytop + 3.2), xytext=(xa, ytop + 0.3), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.add_patch(FancyBboxPatch((xa - 3.6, ytop + 3.6), 7.2, 4.4, boxstyle="round,pad=0.15", fc=BG, ec=INK, lw=2))
    ax.text(xa, ytop + 5.8, "123", ha="center", va="center", fontsize=22, family="monospace", fontweight="bold")
    # legend / explanation on the right
    xr = xs[-1] + cw + 2.5
    ax.add_patch(FancyBboxPatch((xr, 36), 2.4, 3.0, boxstyle="round,pad=0.1", fc=GREEN_PALE, ec=BLUE, lw=2))
    ax.text(xr + 3.4, 37.5, "value written\nby the implant", va="center", fontsize=18, linespacing=1.0)
    ax.add_patch(FancyBboxPatch((xr, 28), 2.4, 3.0, boxstyle="round,pad=0.1", fc=BG, ec=GRID, lw=1.5, ls=(0, (3, 3))))
    ax.text(xr + 3.4, 29.5, "hidden from\nlayer 4 up", va="center", fontsize=18, linespacing=1.0)
    ax.add_patch(FancyBboxPatch((xr, 20), 2.4, 3.0, boxstyle="round,pad=0.1", fc="#f3f4f6", ec=GRID, lw=1.2))
    ax.text(xr + 3.4, 21.5, "the model's\nown states", va="center", fontsize=18, linespacing=1.0)
    ax.text(xr, 11.5, "the model\nfinishes:\n\u221241 \u00d7 \u22123 = 123", va="center", fontsize=18,
            color=INK2, linespacing=1.1)
    save(fig, "11_latent_diagram.png")


def _latent(model):
    p = {"0.6B": "results/bbh_Qwen3-0.6B-Base_dropc.json", "1.7B": "results/bbh_Qwen3-1.7B-Base_dropc.json",
         "4B": "4b/bbh_Qwen3-4B-Base_dropc.json"}[model]
    return json.load(open(os.path.join(os.environ.get("LATENT_BBH_DIR", "latent_bbh"), p)))


def latent_results():
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, "The model treats the written value as its own",
          "BBH multi-step arithmetic, all 250 questions, frozen Qwen3 base models, no chain of thought")
    bar_legend(fig, [("model alone", GRAY), ("values written into layer 4", BLUE),
                     ("values given as text", INK2)], 0.07, 0.765, 0.27)
    ax = fig.add_axes([0.07, 0.13, 0.48, 0.53])
    w = 0.26
    for g, m in enumerate(["0.6B", "1.7B", "4B"]):
        d = _latent(m)
        vals = [d["plain_generate_EM"], d["brackets:L4:synth_true"]["true"], d["text_brackets"]["true"]]
        for k, (v, c) in enumerate(zip(vals, (GRAY, BLUE, INK2))):
            x = g + (k - 1) * (w + 0.02)
            ax.bar(x, max(v, 0.6), width=w, color=c, edgecolor=BG, linewidth=2, zorder=2)
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=24, fontweight="bold" if k == 1 else "normal")
        ax.text(g, -7, f"Qwen3-{m}", ha="center", va="top", fontsize=23, fontweight="bold")
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 108)
    ax.set_xticks([]); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=20, length=0)
    grid(ax, (25, 50, 75, 100))
    ax.set_ylabel("exact answers (%)", fontsize=22)
    # right: which layer
    ax2 = fig.add_axes([0.64, 0.13, 0.33, 0.53])
    shades = {"0.6B": "#9fd3bd", "1.7B": "#3fa37c", "4B": BLUE}
    for m in ["0.6B", "1.7B", "4B"]:
        d = _latent(m)
        Ls = d["layers"]; ys = [d[f"brackets:L{L}:synth_true"]["true"] for L in Ls]
        ax2.plot(Ls, ys, color=shades[m], lw=5, marker="o", ms=9, zorder=3)
        ax2.text(Ls[-1] + 0.6, ys[-1] + {"0.6B": -5, "1.7B": 5, "4B": 0}[m], m, va="center", fontsize=21, color=shades[m], fontweight="bold")
    ax2.set_xlim(-0.5, 21); ax2.set_ylim(-3, 106)
    ax2.set_xticks([0, 4, 8, 12, 16]); ax2.set_yticks([0, 25, 50, 75, 100]); ax2.tick_params(labelsize=20, length=0)
    grid(ax2, (0, 25, 50, 75, 100))
    ax2.set_xlabel("layer the values are written into", fontsize=21)
    ax2.set_title("works until about 40% deep", fontsize=24, color=INK, pad=18, loc="left", fontweight="bold")
    save(fig, "12_latent_results.png")


if __name__ == "__main__":
    calc_digits(); finetune(); ifeval(); latent_diagram(); latent_results()


# ------------------------------------------------------------ the agent's loop (no circle)
def agent_loop():
    fig = plt.figure(figsize=(16, 7.4))
    title(fig, "How Claude builds an implant",
          "the model stays frozen the whole time")
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.72]); ax.set_xlim(0, 100); ax.set_ylim(2, 42); ax.axis("off")
    steps = [("1", "Run the\nfrozen model"), ("2", "Read its\nfailures"), ("3", "Pick the ones\nan implant\ncan fix"),
             ("4", "Write the\nimplant"), ("5", "Attach it\nand rerun")]
    w, gap, x0, y0, h = 16.4, 3.0, 4.0, 22.0, 16.0
    xs = [x0 + i * (w + gap) for i in range(len(steps))]
    for i, ((n, lab), x) in enumerate(zip(steps, xs)):
        green = i == 3
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.4",
                                    fc=GREEN_PALE if green else "#f3f4f6", ec=BLUE if green else GRID, lw=2.5))
        ax.text(x + 1.4, y0 + h - 2.6, n, fontsize=22, fontweight="bold", color=BLUE if green else INK2, va="center")
        ax.text(x + w / 2, y0 + h / 2 - (0.0 if green else 1.0), lab, ha="center", va="center", fontsize=25,
                fontweight="bold" if green else "normal", color=INK, linespacing=1.15)
        if green:
            ax.text(x + w / 2, y0 + 1.6, "tested alone until exact", ha="center", fontsize=17, color=BLUE)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.5, y0 + h / 2), xytext=(x + w + 0.5, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2, mutation_scale=22))
    # return path underneath
    yb = 10.0
    xa, xb = xs[-1] + w / 2, xs[1] + w / 2
    ax.plot([xa, xa], [y0 - 0.8, yb], color=INK2, lw=2.2)
    ax.plot([xa, xb], [yb, yb], color=INK2, lw=2.2)
    ax.annotate("", xy=(xb, y0 - 0.8), xytext=(xb, yb), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2.2,
                                                                       mutation_scale=22))
    ax.text((xa + xb) / 2, yb - 3.6, "the new failures feed the next pass", ha="center", fontsize=22, color=INK2)
    save(fig, "13_agent_loop.png")


# ------------------------------------------------------------ GSM, calculator only, per model
GSM_CALC = {  # model: (gsm8k plain, gsm8k calc, hard plain, hard calc)
    "Qwen3-0.6B": (52.24, 53.37, 27.29, 35.03),
    "Qwen3-1.7B": (73.09, 72.33, 45.41, 51.02),
    "Qwen3-4B": (84.31, 84.69, 58.98, 60.35),
}
def gsm_calc():
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, "The calculator on grade-school math",
          "frozen Qwen3 base models, 8 worked examples in the prompt, full test sets")
    bar_legend(fig, [("model alone", GRAY), ("with the calculator implant", BLUE)], 0.07, 0.765, 0.25)
    models = list(GSM_CALC)
    for p, (name, idx) in enumerate((("GSM8K", (0, 1)), ("GSM-Hard", (2, 3)))):
        ax = fig.add_axes([0.07 + p * 0.47, 0.13, 0.42, 0.53])
        w = 0.34
        for g, m in enumerate(models):
            vals = [GSM_CALC[m][idx[0]], GSM_CALC[m][idx[1]]]
            for k, (v, c) in enumerate(zip(vals, (GRAY, BLUE))):
                x = g + (k - 0.5) * (w + 0.03)
                ax.bar(x, v, width=w, color=c, edgecolor=BG, linewidth=2, zorder=2)
                ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=28, fontweight="bold" if k else "normal")
            ax.text(g, -7, m, ha="center", va="top", fontsize=23, fontweight="bold")
        ax.set_xlim(-0.6, len(models) - 0.4); ax.set_ylim(0, 100)
        ax.set_xticks([]); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=20, length=0)
        grid(ax, (25, 50, 75, 100))
        ax.set_title(name, fontsize=26, color=INK, pad=18, loc="left", fontweight="bold")
    save(fig, "14_gsm_calc.png")


# ------------------------------------------------------------ IFEval, compiled rule-keepers (held-out)
def ifeval_compiled():
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, "Rule-keeper implants, fully in weights",
          "IFEval held-out half (299 prompts), frozen Qwen3-0.6B, three compiled rules")
    bar_legend(fig, [("model alone", GRAY), ("with the rule-keeper implants", BLUE)], 0.07, 0.765, 0.25)
    ax = fig.add_axes([0.07, 0.13, 0.26, 0.53])
    w = 0.36
    for k, (v, c) in enumerate(((56.7, GRAY), (61.7, BLUE))):
        x = (k - 0.5) * (w + 0.03)
        ax.bar(x, v, width=w, color=c, edgecolor=BG, linewidth=2, zorder=2)
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=28, fontweight="bold" if k else "normal")
    ax.text(0, -7, "all 299 prompts", ha="center", va="top", fontsize=22, fontweight="bold")
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 100)
    ax.set_xticks([]); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=20, length=0)
    grid(ax, (25, 50, 75, 100))
    ax.set_title("prompt-level strict", fontsize=24, color=INK, pad=18, loc="left", fontweight="bold")
    ax2 = fig.add_axes([0.43, 0.13, 0.54, 0.53])
    rules = [("no commas\n(36 prompts)", 18 / 36 * 100, 34 / 36 * 100),
             ("all lowercase\n(23)", 18 / 23 * 100, 21 / 23 * 100),
             ("wrap in quotes\n(23)", 21 / 23 * 100, 22 / 23 * 100)]
    for g, (lab, a, b) in enumerate(rules):
        for k, (v, c) in enumerate(((a, GRAY), (b, BLUE))):
            x = g + (k - 0.5) * (w + 0.03)
            ax2.bar(x, v, width=w, color=c, edgecolor=BG, linewidth=2, zorder=2)
            ax2.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=26, fontweight="bold" if k else "normal")
        ax2.text(g, -7, lab, ha="center", va="top", fontsize=21, fontweight="bold", linespacing=1.1)
    ax2.set_xlim(-0.6, 2.6); ax2.set_ylim(0, 100)
    ax2.set_xticks([]); ax2.set_yticks([0, 25, 50, 75, 100]); ax2.tick_params(labelsize=20, length=0)
    grid(ax2, (25, 50, 75, 100))
    ax2.set_title("instruction followed (%), by rule", fontsize=24, color=INK, pad=18, loc="left", fontweight="bold")
    save(fig, "10_ifeval_compiled.png")


# ---------------------------------------------------------------- latent write, compiled in weights
def latent_inw():
    import json as _j
    R = os.path.join(REPO, "latent_write/results")
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, "The model treats the written value as its own",
          "frozen Qwen3-0.6B, no chain of thought, 300 problems each")
    panels = [("muladd_mid", "What is (23*7)+58?"), ("lost_mid", "Each box has 23 pens... 58 are lost.")]
    for p, (k, name) in enumerate(panels):
        d = _j.load(open(f"{R}/{k}_Qwen3-0.6B-Base.json"))
        Ls = sorted(int(x[1:].split("_")[0]) for x in d if x.startswith("L") and x.endswith("_comp_true"))
        acc = [100 * d[f"L{l}_comp_true"]["acc"] for l in Ls]
        fol = [100 * d[f"L{l}_comp_corr"]["follow"] for l in Ls]
        mea = [100 * d[f"L{l}_comp_meanall"]["acc"] for l in Ls]
        ax = fig.add_axes([0.07 + p * 0.48, 0.13, 0.41, 0.52])
        ax.axhline(100 * d["plain"]["acc"], color=GRAY, ls=(0, (4, 4)), lw=4, zorder=1)
        ax.plot(Ls, acc, "-o", color=BLUE, lw=7, ms=16, zorder=3)
        ax.plot(Ls, fol, "--s", color=INK, lw=4, ms=12, mfc=BG, mew=3, zorder=4)
        ax.plot(Ls, mea, "-", color=INK2, lw=4.5, zorder=2)
        ax.text(-0.5, 100 * d["plain"]["acc"] + 3, "model alone", ha="left", fontsize=19, color=INK2)
        tl = 100 * d["text_tool_line"]["acc"]; to = 100 * d["text_oracle_true"]["acc"]
        ax.axhline(tl, color="#c2410c", ls=(0, (2, 3)), lw=3.5, zorder=1)
        ax.text(Ls[-1] + 0.8, tl - 2, f"tool line in text: {tl:.0f}", ha="right", va="top",
                fontsize=17, color="#c2410c", zorder=8, bbox=dict(fc=BG, ec="none", pad=1))
        ax.set_xlim(-1, Ls[-1] + 1); ax.set_ylim(-4, 108)
        ax.set_xticks([0, 4, 8, 12, 16, 20]); ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=19, length=0)
        grid(ax, (25, 50, 75, 100))
        ax.set_xlabel("layer the value is written into (of 28)", fontsize=20, color=INK2)
        ax.set_title(name, fontsize=24, color=INK, pad=16, loc="left", fontweight="bold")
    keys = [("correct value written: accuracy", BLUE, "-"), ("wrong value written: model follows", INK, "--"),
            ("average value written", INK2, "-")]
    fig.text(0.04, 0.80, "orange dotted line: the same value handed to the model as a tool line in the text instead",
             fontsize=18, color="#c2410c")
    for i, (lab, c, ls) in enumerate(keys):
        x0 = 0.07 + i * 0.315
        fig.add_artist(plt.Line2D([x0, x0 + 0.03], [0.745, 0.745], color=c, lw=7 if c == BLUE else 4.5, ls=ls,
                                  transform=fig.transFigure))
        fig.text(x0 + 0.038, 0.745, lab, va="center", fontsize=19)
    save(fig, "15_latent_inweights.png")
