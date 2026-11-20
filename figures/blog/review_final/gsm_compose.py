"""Review re-render of 'Two different fixes, and they compose' (orig: weight-space-micromachines/figures/blog/make_figs.py:
gsm() -> figs/3_gsm_stack.png). Numbers = the constants of the original script, checked against
results/rethink_v2/summary.json (hard_calc none/hybrid/alwaysA = 34.34/35.18/37.15,
hard_plain hybrid/alwaysA = 29.04/29.57; frozen Qwen3-0.6B-Base, full test sets, n = 1319 each).
Changes: 1600x700; one panel per benchmark; x = when to rethink, color = calculator off/on (no hatches, no legend block)."""
import numpy as np
from blogstyle import *
D = {'GSM-Hard': [26.61, 29.04, 29.57, 34.34, 35.18, 37.15], 'GSM8K': [52.01, 55.65, 56.56, 53.30, 56.71, 57.54]}
groups = ['never', 'when the\ndetector fires', 'every\ntime']
fig, axs = plt.subplots(1, 2, figsize=SIZE)
w = 0.36
for p, (ax, (bench, v)) in enumerate(zip(axs, D.items())):
    off, on = v[:3], v[3:]
    x = np.arange(3)
    ax.bar(x - w / 2 - 0.015, off, w, color=GREY, zorder=3)
    ax.bar(x + w / 2 + 0.015, on, w, color=GREEN, zorder=3)
    for i in range(3):
        ax.text(x[i] - w / 2 - 0.015, off[i] + 1, f'{off[i]:.0f}', ha='center', va='bottom', fontsize=VAL, color=INK2)
        ax.text(x[i] + w / 2 + 0.015, on[i] + 1, f'{on[i]:.0f}', ha='center', va='bottom', fontsize=VAL,
                color=INK, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(groups, linespacing=1.0); ax.tick_params(axis='x', length=0, pad=8)
    style(ax, bench, 'accuracy (%)' if p == 0 else '', (0, 72))
    ax.tick_params(axis='x', length=0)
    ax.set_xlabel('rethink before answering', labelpad=8)
    ax.set_xlim(-0.6, 2.6)
    if p == 0:
        ax.text(x[0] - w / 2 - 0.015, off[0] / 2, 'no\ncalc', ha='center', va='center', fontsize=15, color=INK,
                fontweight='bold', linespacing=0.95)
        ax.text(x[0] + w / 2 + 0.015, on[0] / 2, '+ calc', ha='center', va='center', fontsize=15, color='white',
                fontweight='bold')
save(fig, 'gsm_compose')
