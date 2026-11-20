"""Review re-render of 'Block a topic by reading the model's state' (orig: weight-space-micromachines/figures/blog/
make_figs3_v2.py: topic_blocker() -> figs/21_topic_blocker.png, one 1600x1700 figure with math + ML panels).
Numbers = original constants, checked against topic_fence/results/q17_{math,ml}_table.txt
(rows pos / hard / ret; columns plain, sys, sys_strong, kw_stop, probe_text_stop). Frozen Qwen3-1.7B, held-out prompts.
Changes: one 1600x700 figure per topic; left panel = what it should block, right panel = what it blocks by mistake;
methods as rows with direct labels; no hatches, no legend block."""
import numpy as np
from blogstyle import *
M = ['model alone', 'system prompt', 'strict system prompt', 'keyword list', 'blocker implant']
D = {'math': {'pos': [11.0, 8.9, 73.6, 63.3, 98.7], 'near': [1.4, 5.7, 82.9, 57.1, 1.4], 'normal': [5.5, 7.9, 85.5, 7.2, 9.5]},
     'ML development': {'pos': [4.9, 1.6, 61.5, 93.4, 98.4], 'near': [2.9, 2.9, 15.7, 24.3, 8.6], 'normal': [5.5, 4.8, 20.3, 7.1, 5.5]}}
for topic, d in D.items():
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={'width_ratios': [1, 1.25]})
    y = np.arange(len(M))[::-1]
    cols = [GREY] * 4 + [GREEN]
    a.barh(y, d['pos'], 0.62, color=cols, zorder=3)
    for yi, v, c in zip(y, d['pos'], cols):
        a.text(v + 1.5, yi, f'{v:.0f}', va='center', fontsize=VAL, fontweight='bold' if c == GREEN else 'normal',
               color=INK if c == GREEN else INK2)
    a.set_yticks(y); a.set_yticklabels(M)
    for t, c in zip(a.get_yticklabels(), cols):
        if c == GREEN: t.set_fontweight('bold'); t.set_color(GREEN)
    h = 0.36
    b.barh(y + h / 2 + 0.01, d['near'], h, color=[GREY] * 4 + [GREEN], zorder=3)
    b.barh(y - h / 2 - 0.01, d['normal'], h, color=[PALE] * 4 + [PALE], edgecolor=[GREY] * 4 + [GREEN], lw=2, zorder=3)
    for yi, v1, v2, c in zip(y, d['near'], d['normal'], cols):
        b.text(v1 + 1.5, yi + h / 2 + 0.01, f'{v1:.0f}', va='center', fontsize=15, color=INK2 if c != GREEN else INK,
               fontweight='bold' if c == GREEN else 'normal')
        b.text(v2 + 1.5, yi - h / 2 - 0.01, f'{v2:.0f}', va='center', fontsize=15, color=INK2 if c != GREEN else INK,
               fontweight='bold' if c == GREEN else 'normal')
    b.set_yticks(y); b.set_yticklabels([])
    word = 'math' if topic == 'math' else 'ML'
    for ax, ttl in ((a, f'Should block: requests to do {word}'), (b, 'Blocked by mistake')):
        ax.set_title(ttl, loc='left', fontsize=TITLE, fontweight='bold', pad=16)
        ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel('% of replies blocked')
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        ax.xaxis.grid(True, color=GRID, lw=1.5); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0)
    # direct labels for the two bar kinds, next to the top row (model alone, where both bars are short)
    i0 = 0 if topic == 'math' else 1
    b.text(d['near'][0] + 9, y[0] + h / 2 + 0.01, f'only mentions {word} (near-miss)', va='center', fontsize=15, color=INK)
    b.text(d['normal'][0] + 9, y[0] - h / 2 - 0.01, 'normal requests', va='center', fontsize=15, color=INK)
    save(fig, 'topic_blocker' if topic == 'math' else 'topic_blocker_ml')
