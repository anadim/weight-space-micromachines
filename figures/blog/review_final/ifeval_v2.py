"""IFEval bars v2: legend instead of rotated in-bar labels, bigger tick/axis fonts (user comment 2026-11-19)."""
from blogstyle import *
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=SIZE)
style(ax, 'Rule implants on IFEval (frozen Qwen3-0.6B, all 541 prompts)', 'accuracy (%)', (0, 95))
groups = [('every instruction in a prompt\nfollowed (main IFEval score)', [57.86, 72.09]),
          ('each instruction\nscored on its own', [67.63, 79.62])]
w = 0.34
for g, (_, vals) in enumerate(groups):
    for k, v in enumerate(vals):
        x = g + (k - 0.5) * (w + 0.02)
        ax.bar(x, v, w, color=[GREY, GREEN][k], zorder=3)
        ax.text(x, v + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=24, fontweight='bold',
                color=[INK2, INK][k])
ax.set_xticks(range(2)); ax.set_xticklabels([g for g, _ in groups], fontsize=21, fontweight='bold', linespacing=1.1)
ax.tick_params(axis='x', length=0, pad=10); ax.tick_params(axis='y', labelsize=21)
ax.set_ylabel('accuracy (%)', fontsize=23)
ax.set_xlim(-0.6, 1.6)
ax.legend(handles=[Patch(color=GREY, label='model alone'), Patch(color=GREEN, label='+ 14 rule implants')],
          loc='upper left', fontsize=21, frameon=False, handlelength=1.4, handleheight=1.0)
save(fig, 'ifeval_bars_v2')
