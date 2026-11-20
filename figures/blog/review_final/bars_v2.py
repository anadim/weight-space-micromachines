"""v2 of the grouped-bar re-renders (user comment 2026-11-19): legend instead of rotated in-bar names, bigger fonts."""
import numpy as np
from blogstyle import *
from matplotlib.patches import Patch
TK, LB, VL, LG = 20, 22, 20, 20

def legend(ax, cols, names, loc='upper left', ncol=None):
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(cols, names)], loc=loc, fontsize=LG, frameon=False,
              handlelength=1.4, handleheight=1.0, ncol=ncol or len(names), columnspacing=1.2)

def grouped(ax, groups, cols, w, errs=None):
    n = len(cols)
    for g, (_, vals) in enumerate(groups):
        for k, v in enumerate(vals):
            x = g + (k - (n - 1) / 2) * (w + 0.02)
            ax.bar(x, v, w, color=cols[k], zorder=3)
            e = errs[g][k] if errs else 0
            if errs: ax.errorbar(x, v, yerr=e, color=INK, lw=2.5, capsize=6, capthick=2.5, zorder=4)
            ax.text(x, v + e + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=VL,
                    fontweight='bold' if k == n - 1 else 'normal', color=INK if k == n - 1 else INK2)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups], fontsize=TK, fontweight='bold', linespacing=1.05)
    ax.tick_params(axis='x', length=0, pad=10); ax.tick_params(axis='y', labelsize=TK)
    ax.set_xlim(-0.6, len(groups) - 0.4)

def bbh():
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax, 'Eight implants on BIG-Bench Hard (Qwen3-0.6B, real BBH test)', 'accuracy (%)', (0, 112))
    ax.yaxis.label.set_size(LB); ax.set_yticks([0, 20, 40, 60, 80, 100])
    cols = [GREY, INK2, GREEN]
    grouped(ax, [('12 tasks with an implant', [43.27, 38.10, 84.33]), ('15 tasks with no implant', [39.15, 30.01, 19.58]),
                 ('all 27 tasks', [40.98, 33.61, 48.36])], cols, 0.26)
    legend(ax, cols, ['model alone', 'call fine-tune', 'fine-tune + implants'])
    save(fig, 'bbh_bars_v2')

def finetune():
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax, 'Fine-tune Qwen2.5-0.5B to make the calls (5 seeds)', 'accuracy (%)', (0, 125))
    ax.yaxis.label.set_size(LB); ax.set_yticks([0, 20, 40, 60, 80, 100])
    cols = [GREY, INK2, GREEN]
    grouped(ax, [('GSM8K', [33.6, 35.2, 35.1]), ('GSM-Hard (big numbers)', [13.4, 14.9, 23.3]),
                 ('arithmetic, new phrasings', [50.8, 54.5, 99.0])], cols, 0.26,
            errs=[[0.6, 0.2, 1.2], [0.4, 0.9, 0.6], [3.4, 1.5, 0.4]])
    legend(ax, cols, ['no calls', 'calls, model answers', 'calls, implant answers'])
    save(fig, 'finetune_v2')

def transfer():
    panels = [('IFEval (14 rules)', [('Qwen3\n0.6B', 57.9, 72.1), ('Qwen3\n1.7B', 68.8, 78.2), ('Qwen3\n4B', 80.4, 84.8)]),
              ('BBH (12 implant tasks)', [('Qwen3\n0.6B', 43.27, 84.33), ('Qwen3\n1.7B', 55.73, 96.37), ('Qwen3\n4B', 82.83, 97.77)]),
              ('GSM-Hard (calculator)', [('Qwen3\n0.6B', 27.3, 35.0), ('Qwen3\n1.7B', 45.4, 51.0), ('Qwen3\n4B', 59.0, 60.4),
                                         ('SmolLM2\n1.7B', 14.0, 19.9), ('Granite\n3.3-2B', 23.2, 30.4)])]
    fig, axs = plt.subplots(1, 3, figsize=(16, 7.6), gridspec_kw={'width_ratios': [3, 3, 5]})
    for p, (ax, (name, rows)) in enumerate(zip(axs, panels)):
        for i, (lab, a, b) in enumerate(rows):
            ax.bar(i - 0.2, a, 0.4, color=GREY, zorder=3); ax.bar(i + 0.2, b, 0.4, color=GREEN, zorder=3)
            ax.text(i - 0.2, a + 1.5, f'{a:.0f}', ha='center', va='bottom', fontsize=16, color=INK2)
            ax.text(i + 0.2, b + 1.5, f'{b:.0f}', ha='center', va='bottom', fontsize=16, color=INK, fontweight='bold')
        style(ax, name, 'accuracy (%)' if p == 0 else '', (0, 108))
        if p == 0: ax.yaxis.label.set_size(LB)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=17, linespacing=1.0)
        ax.tick_params(axis='x', length=0); ax.tick_params(axis='y', labelsize=TK)
        ax.set_xlim(-0.6, len(rows) - 0.4); ax.set_yticks([0, 25, 50, 75, 100])
    fig.legend(handles=[Patch(color=GREY, label='model alone'), Patch(color=GREEN, label='+ implants')],
               loc='lower center', ncol=2, fontsize=LG, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=3, rect=(0, 0.07, 1, 1)); save(fig, 'transfer_v2', tight=False)

def compose():
    D = {'GSM-Hard': [26.61, 29.04, 29.57, 34.34, 35.18, 37.15], 'GSM8K': [52.01, 55.65, 56.56, 53.30, 56.71, 57.54]}
    groups = ['never', 'when the\ndetector fires', 'every\ntime']
    fig, axs = plt.subplots(1, 2, figsize=SIZE); w = 0.36
    for p, (ax, (bench, v)) in enumerate(zip(axs, D.items())):
        off, on = v[:3], v[3:]; x = np.arange(3)
        ax.bar(x - w / 2 - 0.015, off, w, color=GREY, zorder=3); ax.bar(x + w / 2 + 0.015, on, w, color=GREEN, zorder=3)
        for i in range(3):
            ax.text(x[i] - w / 2 - 0.015, off[i] + 1, f'{off[i]:.0f}', ha='center', va='bottom', fontsize=VL, color=INK2)
            ax.text(x[i] + w / 2 + 0.015, on[i] + 1, f'{on[i]:.0f}', ha='center', va='bottom', fontsize=VL, color=INK, fontweight='bold')
        style(ax, bench, 'accuracy (%)' if p == 0 else '', (0, 80))
        ax.yaxis.label.set_size(LB)
        ax.set_xticks(x); ax.set_xticklabels(groups, linespacing=1.0, fontsize=TK)
        ax.tick_params(axis='x', length=0, pad=8); ax.tick_params(axis='y', labelsize=TK)
        ax.set_xlabel('rethink before answering', labelpad=8, fontsize=LB); ax.set_xlim(-0.6, 2.6)
        if p == 0: legend(ax, [GREY, GREEN], ['no calculator', '+ calculator'])
    save(fig, 'gsm_compose_v2')

if __name__ == '__main__':
    bbh(); finetune(); transfer(); compose()
