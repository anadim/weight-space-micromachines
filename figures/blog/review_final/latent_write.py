"""Review re-render of 'The model treats the written value as its own' (orig: weight-space-micromachines/figures/blog/
make_figs2.py: latent_inw -> figs/15_latent_inweights.png). Same data, read from
latent_write/results/{muladd,lost}_mid_Qwen3-0.6B-Base.json.
Changes: 1600x700, one title per panel, direct labels instead of a legend row + red caption, orange tool line -> dark
dotted (palette), 'average value written' control dropped (flat ~0 everywhere; mention in text if needed)."""
import json, os
from blogstyle import *
R = os.path.join(HERE, '..', '..', '..', 'latent_write', 'results')
panels = [('muladd_mid', 'What is (23×7)+58?  (one shot)'), ('lost_mid', 'Word problem: 23 pens per box, 58 lost')]
fig, axs = plt.subplots(1, 2, figsize=SIZE)
for p, (ax, (k, name)) in enumerate(zip(axs, panels)):
    d = json.load(open(f'{R}/{k}_Qwen3-0.6B-Base.json'))
    Ls = sorted(int(x[1:].split('_')[0]) for x in d if x.startswith('L') and x.endswith('_comp_true'))
    acc = [100 * d[f'L{l}_comp_true']['acc'] for l in Ls]
    fol = [100 * d[f'L{l}_comp_corr']['follow'] for l in Ls]
    base = 100 * d['plain']['acc']; tl = 100 * d['text_tool_line']['acc']
    ax.axhline(base, color=GREY, ls=(0, (4, 3)), lw=3.5, zorder=1)
    ax.axhline(tl, color=INK2, ls=(0, (1, 2)), lw=3.5, zorder=1)
    ax.plot(Ls, acc, '-o', color=GREEN, lw=5, ms=11, zorder=3)
    ax.plot(Ls, fol, '--', color=INK, lw=2.5, zorder=4)
    style(ax, name, 'accuracy (%)' if p == 0 else '', (-4, 112))
    ax.set_xlim(-0.8, Ls[-1] + 0.8); ax.set_xticks([0, 4, 8, 12, 16, 20]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel('layer the value is written into (of 28)')
    bb = dict(fc='white', ec='none', pad=1.5)
    ax.text(Ls[0] - 0.3, base + 2.5, f'model alone: {base:.0f}', ha='left', va='bottom', fontsize=VAL, color=INK2, bbox=bb)
    if p == 0:
        ax.text(Ls[-1] + 0.6, tl + 3, f'value as text: {tl:.0f}', ha='right', va='bottom', fontsize=VAL, color=INK2,
                fontweight='bold', bbox=bb)
    else:
        ax.text(Ls[0] - 0.3, tl - 3, f'value as text: {tl:.0f}', ha='left', va='top', fontsize=VAL, color=INK2,
                fontweight='bold', bbox=bb)
    ax.text(Ls[0] + 0.3, acc[0] - 7, f'written into the model: {acc[0]:.0f}', ha='left', va='top', fontsize=VAL + 1,
            color=GREEN, fontweight='bold', bbox=dict(fc='white', ec='none', pad=1))
    print(k, 'base', round(base, 1), 'tool', round(tl, 1), 'acc', [round(a) for a in acc], 'follow', [round(f) for f in fol])
axs[1].annotate('dashed: a wrong value written,\nmodel follows it', xy=(16.9, 45), xytext=(4.5, 40), fontsize=15,
                color=INK, arrowprops=dict(arrowstyle='-', color=INK, lw=1.5))
save(fig, 'latent_write')
