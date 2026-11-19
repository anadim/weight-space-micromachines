"""CW-full s0: in-wire veto rate by operand size over RL (greedy dev-800). Data: the run's own log,
rl/results/wiring/H100/L300_CW_s0_kl001evalall0/steps.jsonl (dev.dev_veto_by_digits = [calls, vetoed] per bucket).
The blog figure shows u25-u250; the run continues to u300 (1-2 digits: 59% vetoed at u275, 32% at u300)."""
from make_figs import *
from make_figs3 import legend_rows
import json, os
_R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_F = os.path.join(_R, "rl", "results", "wiring", "H100", "L300_CW_s0_kl001evalall0", "steps.jsonl")
u = [25, 50, 64, 100, 125, 150, 175, 200, 225, 250]
_B = {"1-2 digits": "1-2", "3-4 digits": "3-4", "5-6 digits": "5-6", "7+ digits": "7+"}
_D = {}
for _l in open(_F):
    _j = json.loads(_l)
    if "dev" in _j and "dev_veto_by_digits" in _j["dev"]: _D[_j["dev"]["u"]] = _j["dev"]["dev_veto_by_digits"]
V = {nm: [round(100 * _D[x][b][1] / max(_D[x][b][0], 1)) for x in u] for nm, b in _B.items()}
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
