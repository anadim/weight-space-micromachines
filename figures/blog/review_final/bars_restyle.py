"""Review re-renders (2026-11-19) of four grouped-bar figures in the RL-section style: 1600x700, one short title,
no subtitle, no legend block; the arms are named directly on the first group's bars.
Numbers are the constants of the original scripts in figures/blog/, re-checked against:
  ifeval_bars  (make_figs3_v2.ifeval_bars)  REPRODUCE.md sec 5 / ifeval/transfer/results/q06_score.txt: 57.86->72.09, 67.63->79.62
  bbh_bars     (make_figs3_v2.bbh_bars)     bbh_generic/integrate/TABLES_v1_vs_v2.md (v2): 43.27/38.10/84.33, 39.15/30.01/19.58, 40.98/33.61/48.36
  finetune     (make_figs2.finetune)        finetuned_phase/results/reported_numbers.json five_seed_mean_std (Qwen2.5-0.5B)
  transfer     (transfer_fig_v2.py)         bbh_generic/transfer/README.md, ifeval REPRODUCE.md, transfer/xfamily/NOTEBOOK.md"""
import numpy as np
from blogstyle import *
ARMCOL = {'alone': GREY, 'ftoff': INK2, 'on': GREEN}


def grouped(ax, groups, arms, cols, names, w, ylim, errs=None, name_y=None, lab_fs=VAL, label_group=0):
    n = arms
    for g, (gname, vals) in enumerate(groups):
        for k, v in enumerate(vals):
            x = g + (k - (n - 1) / 2) * (w + 0.02)
            ax.bar(x, v, w, color=cols[k], zorder=3)
            e = errs[g][k] if errs else 0
            if errs: ax.errorbar(x, v, yerr=e, color=INK, lw=2.5, capsize=6, capthick=2.5, zorder=4)
            ax.text(x, v + e + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=lab_fs + 2,
                    fontweight='bold' if k == n - 1 else 'normal', color=INK if k == n - 1 else INK2)
            if g == label_group:
                c = 'white' if cols[k] in (GREEN, INK2) else INK
                ax.text(x, (name_y if name_y is not None else 2), names[k], ha='center', va='bottom', rotation=90,
                        fontsize=15, color=c, fontweight='bold', zorder=5)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups], fontsize=LABEL, fontweight='bold',
                                                           linespacing=1.05)
    ax.tick_params(axis='x', length=0, pad=10)
    ax.set_xlim(-0.6, len(groups) - 0.4); ax.set_ylim(*ylim)


def ifeval():
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax, 'Rule implants on IFEval (frozen Qwen3-0.6B, all 541 prompts)', 'accuracy (%)', (0, 90))
    grouped(ax, [('every instruction in a prompt followed\n(main IFEval score)', [57.86, 72.09]),
                 ('each instruction scored on its own', [67.63, 79.62])],
            2, [GREY, GREEN], ['model alone', '+ 14 rule implants'], 0.34, (0, 90))
    ax.tick_params(axis='x', length=0)
    save(fig, 'ifeval_bars')


def bbh():
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax, 'Eight implants on BIG-Bench Hard (Qwen3-0.6B, real BBH test)', 'accuracy (%)', (0, 100))
    grouped(ax, [('12 tasks with an implant', [43.27, 38.10, 84.33]), ('15 tasks with no implant', [39.15, 30.01, 19.58]),
                 ('all 27 tasks', [40.98, 33.61, 48.36])],
            3, [GREY, INK2, GREEN], ['model alone', 'call fine-tune', 'fine-tune + implants'], 0.26, (0, 100))
    ax.tick_params(axis='x', length=0)
    save(fig, 'bbh_bars')


def finetune():
    fig, ax = plt.subplots(figsize=SIZE)
    style(ax, 'Fine-tune Qwen2.5-0.5B to make the calls (5 seeds)', 'accuracy (%)', (0, 110))
    grouped(ax, [('GSM8K', [33.6, 35.2, 35.1]), ('GSM-Hard (big numbers)', [13.4, 14.9, 23.3]),
                 ('arithmetic, new phrasings', [50.8, 54.5, 99.0])],
            3, [GREY, INK2, GREEN], ['no calls', 'calls, model answers', 'calls, implant answers'], 0.26, (0, 110),
            errs=[[0.6, 0.2, 1.2], [0.4, 0.9, 0.6], [3.4, 1.5, 0.4]], label_group=2)
    ax.tick_params(axis='x', length=0)
    save(fig, 'finetune')


def transfer():
    panels = [('IFEval (14 rules)', [('Qwen3\n0.6B', 57.9, 72.1), ('Qwen3\n1.7B', 68.8, 78.2), ('Qwen3\n4B', 80.4, 84.8)]),
              ('BBH (12 implant tasks)', [('Qwen3\n0.6B', 43.27, 84.33), ('Qwen3\n1.7B', 55.73, 96.37), ('Qwen3\n4B', 82.83, 97.77)]),
              ('GSM-Hard (calculator)', [('Qwen3\n0.6B', 27.3, 35.0), ('Qwen3\n1.7B', 45.4, 51.0), ('Qwen3\n4B', 59.0, 60.4),
                                         ('SmolLM2\n1.7B', 14.0, 19.9), ('Granite\n3.3-2B', 23.2, 30.4)])]
    fig, axs = plt.subplots(1, 3, figsize=SIZE, gridspec_kw={'width_ratios': [3, 3, 5]})
    for p, (ax, (name, rows)) in enumerate(zip(axs, panels)):
        for i, (lab, a, b) in enumerate(rows):
            ax.bar(i - 0.2, a, 0.4, color=GREY, zorder=3); ax.bar(i + 0.2, b, 0.4, color=GREEN, zorder=3)
            ax.text(i - 0.2, a + 1.5, f'{a:.0f}', ha='center', va='bottom', fontsize=15, color=INK2)
            ax.text(i + 0.2, b + 1.5, f'{b:.0f}', ha='center', va='bottom', fontsize=15, color=INK, fontweight='bold')
        style(ax, name, 'accuracy (%)' if p == 0 else '', (0, 108))
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=15, linespacing=1.0)
        ax.tick_params(axis='x', length=0); ax.set_xlim(-0.6, len(rows) - 0.4); ax.set_yticks([0, 25, 50, 75, 100])
        if p == 0:
            ax.text(-0.2, 3, 'model alone', rotation=90, ha='center', va='bottom', fontsize=14, color=INK, fontweight='bold')
            ax.text(0.2, 3, '+ implants', rotation=90, ha='center', va='bottom', fontsize=14, color='white', fontweight='bold')
    save(fig, 'transfer')


if __name__ == '__main__':
    ifeval(); bbh(); finetune(); transfer()
