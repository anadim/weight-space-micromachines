"""Data figures. Every plotted number comes from results/ via results/summarize.py (nothing is typed in by hand).

python figures/make_figs.py   -> figures/*.png (1600 px wide)
Figures 6 and 8 (earlier, fine-tuned phase) read finetuned_phase/results/reported_numbers.json; those numbers were not
re-run for this repo (see finetuned_phase/README.md).
Fonts: Avenir Next is optional. Drop its .ttf files into figures/fonts/ (not distributed) to match the blog; otherwise the
figures fall back to DejaVu Sans with the same layout.
"""
import os, sys, glob, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'results'))
import summarize as S

INK, INK2, GRID, BG = '#222222', '#6b7280', '#e5e7eb', '#ffffff'
GRAY, GREEN, C7B, C14B = '#b9bec6', '#118659', '#6b7280', '#222222'
for f in glob.glob(os.path.join(HERE, 'fonts', '*.ttf')): fm.fontManager.addfont(f)
FONT = 'Avenir Next' if any('Avenir' in f.name for f in fm.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': FONT, 'font.size': 24, 'text.color': INK, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG})


def title(fig, t, sub=None):
    fig.text(0.04, 0.94, t, fontsize=38, fontweight='bold', va='top')
    if sub: fig.text(0.04, 0.865, sub, fontsize=24, color=INK2, va='top')


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=100); plt.close(fig); print('wrote', name)


def hero():
    _, m = S.bbh_table()
    fig = plt.figure(figsize=(16, 9))
    title(fig, 'A frozen 0.6B model with nine written implants', 'BIG-Bench Hard, 12 algorithmic tasks, official 3-shot prompts')
    ax = fig.add_axes([0.32, 0.08, 0.62, 0.70])
    rows = [('Qwen3-0.6B-Base', m['base'], GRAY), ('Qwen2.5-7B', m['q7b'], C7B), ('Qwen2.5-14B', m['q14b'], C14B),
            ('Qwen3-0.6B-Base\n+ implants', m['implants'], GREEN)]
    for i, (lab, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.62, color=c, edgecolor=BG, linewidth=2)
        ax.text(v + 1.2, y, f'{v:.1f}', va='center', fontsize=40, fontweight='bold')
        ax.text(-1.5, y, lab, va='center', ha='right', fontsize=27, fontweight='bold' if c == GREEN else 'normal')
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, 3.6); ax.set_yticks([]); ax.set_xticks([]); ax.spines['bottom'].set_visible(False)
    save(fig, '1_hero.png')


def per_task():
    rows, _ = S.bbh_table()
    fig = plt.figure(figsize=(16, 13))
    title(fig, 'Task by task', 'Same 12 BBH tasks. Each bar runs from the frozen 0.6B to the 0.6B with implants.')
    ax = fig.add_axes([0.33, 0.07, 0.62, 0.70]); n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i; b, m = r['base'], r['implants']
        ax.plot([b, m], [y, y], color=GREEN, alpha=0.28, lw=10, solid_capstyle='round', zorder=1)
        ax.scatter([r['q7b']], [y], s=260, marker='D', color=C7B, edgecolor=BG, linewidth=2, zorder=3)
        ax.scatter([r['q14b']], [y], s=260, marker='s', color=C14B, edgecolor=BG, linewidth=2, zorder=3)
        ax.scatter([b], [y], s=300, color=BG, edgecolor=GRAY, linewidth=4, zorder=4)
        ax.scatter([m], [y], s=420, color=GREEN, edgecolor=BG, linewidth=2, zorder=5)
        ax.text(-3, y, r['label'], ha='right', va='center', fontsize=23)
    for x in (0, 25, 50, 75, 100): ax.axvline(x, color=GRID, lw=1.5, zorder=0)
    ax.set_xlim(0, 103); ax.set_ylim(-0.7, n - 0.3); ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='x', labelsize=22, length=0); ax.set_xlabel('accuracy (%)', fontsize=22)
    lax = fig.add_axes([0.04, 0.79, 0.92, 0.04]); lax.axis('off')
    for (mk, fc, ec, lab), x in zip([('o', BG, GRAY, '0.6B'), ('o', GREEN, GREEN, '0.6B + implants'), ('D', C7B, C7B, 'Qwen2.5-7B'),
                                     ('s', C14B, C14B, 'Qwen2.5-14B')], [0.0, 0.14, 0.40, 0.62]):
        lax.scatter([x + 0.012], [0.5], s=260, marker=mk, color=fc, edgecolor=ec, linewidth=3)
        lax.text(x + 0.03, 0.5, lab, va='center', fontsize=22)
    lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    save(fig, '2_per_task.png')


def gsm():
    g = S.gsm()
    fig = plt.figure(figsize=(16, 9))
    title(fig, 'Two different fixes, and they stack', 'Frozen Qwen3-0.6B-Base, full test sets (1,319 each), greedy')
    ax = fig.add_axes([0.07, 0.12, 0.90, 0.58])
    arms, cols = ['as is', '+ calculator', '+ rethink', '+ both'], [GRAY, GREEN, '#9fd1b6', '#03623f']
    data = {name: [g[s, 'plain'][0], g[s, 'calc'][0], g[s, 'plain'][1], g[s, 'calc'][1]] for name, s in (('GSM-Hard', 'hard'), ('GSM8K', 'test'))}
    w = 0.2
    for gi, (ds, vals) in enumerate(data.items()):
        for k, v in enumerate(vals):
            x = gi + (k - 1.5) * (w + 0.02)
            ax.bar(x, v, width=w, color=cols[k], edgecolor=BG, linewidth=2)
            ax.text(x, v + 1.2, f'{v:.1f}', ha='center', fontsize=25, fontweight='bold' if k == 3 else 'normal')
        ax.text(gi, -6.5, ds, ha='center', fontsize=28, fontweight='bold')
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 66); ax.set_xticks([]); ax.set_yticks([0, 20, 40, 60]); ax.tick_params(labelsize=20, length=0)
    for yv in (20, 40, 60): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    ax.set_ylabel('accuracy (%)', fontsize=22)
    for k, (a, c) in enumerate(zip(arms, cols)):
        fig.patches.append(FancyBboxPatch((0.07 + k * 0.215, 0.755), 0.018, 0.032, boxstyle='round,pad=0.002', transform=fig.transFigure, color=c))
        fig.text(0.093 + k * 0.215, 0.771, a, va='center', fontsize=22)
    save(fig, '3_gsm_stack.png')


def gate():
    G = S.gate()[('base', 'test')]
    curve = lambda name: [(0, G['never'][0])] + [(tok, acc) for _, acc, tok in G[name]] + [(G['always'][1], G['always'][0])]
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, 'Rethink only when the model looks wrong', 'Qwen3-0.6B-Base, GSM8K test (vLLM). A layer-28 probe decides when to rethink.')
    ax = fig.add_axes([0.09, 0.13, 0.86, 0.64])
    ax.plot([0, G['always'][1]], [G['never'][0], G['always'][0]], ls=(0, (6, 5)), color=GRAY, lw=3.5, zorder=1)
    ax.plot(*zip(*curve('margin')), color=C7B, lw=3.5, marker='o', ms=11, zorder=2)
    ax.plot(*zip(*curve('probe L28')), color=GREEN, lw=5, marker='o', ms=14, zorder=3)
    ox, oy = G['oracle'][1], G['oracle'][0]
    ax.scatter([ox], [oy], s=380, facecolor=BG, edgecolor=INK, linewidth=3, zorder=4); ax.text(ox + 4, oy, 'perfect detector', va='center', fontsize=23)
    ax.text(18, 56.05, 'residual probe', color=GREEN, fontsize=25, fontweight='bold', ha='center')
    ax.text(52, 54.25, 'logit margin (nothing fitted)', color=C7B, fontsize=23, ha='left', va='top')
    ax.text(24, 52.95, 'random trigger', color=INK2, fontsize=23, ha='left', va='top')
    ax.annotate('always rethink', xy=(G['always'][1], G['always'][0]), xytext=(84, G['always'][0] + 1.05), fontsize=23,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=1.5))
    ax.set_xlim(-3, 108); ax.set_ylim(51.5, 59.2); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([52, 54, 56, 58])
    ax.tick_params(labelsize=21, length=0)
    for yv in (52, 54, 56, 58): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    ax.set_xlabel('extra tokens per problem', fontsize=22); ax.set_ylabel('accuracy (%)', fontsize=22)
    save(fig, '4_gate.png')


def adding():
    steps = S.adding_curve(); _, m = S.bbh_table()
    fig = plt.figure(figsize=(16, 10))
    title(fig, 'Add one implant at a time', 'Frozen Qwen3-0.6B-Base, mean over the same 12 BBH tasks')
    ax = fig.add_axes([0.08, 0.17, 0.74, 0.62])
    for v, lab, c in ((m['q7b'], 'Qwen2.5-7B', C7B), (m['q14b'], 'Qwen2.5-14B', C14B)):
        ax.axhline(v, color=c, lw=3, ls=(0, (7, 5)), zorder=2); ax.text(len(steps) - 0.4, v, lab, va='center', fontsize=23, color=c, fontweight='bold')
    xs = list(range(len(steps))); ys = [y for _, y in steps]
    ax.bar(xs, [y - 30 for y in ys], bottom=30, width=0.66, color=[GRAY] + [GREEN] * (len(xs) - 1), edgecolor=BG, linewidth=2, zorder=3)
    ax.text(0, ys[0] + 1.5, f'{ys[0]:.0f}', fontsize=26, ha='center', va='bottom', zorder=5)
    ax.text(xs[-1], ys[-1] + 1.5, f'{ys[-1]:.0f}', fontsize=30, fontweight='bold', ha='center', va='bottom', zorder=5)
    ax.set_xlim(-0.5, len(xs) - 0.5); ax.set_ylim(30, 100); ax.set_xticks(xs)
    ax.set_xticklabels([k.replace(' + ', ',\n').replace(' ', '\n', 1) for k, _ in steps], fontsize=15)
    ax.set_yticks([40, 60, 80, 100]); ax.tick_params(labelsize=21, length=0)
    for yv in (40, 60, 80, 100): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    ax.set_ylabel('BBH accuracy (%)', fontsize=22)
    fig.text(0.08, 0.03, 'left to right: add one more implant (per-task runs, added up)', fontsize=21, color=INK2)
    save(fig, '5_adding.png')


def full27():
    a = S.bbh27()
    fig = plt.figure(figsize=(16, 8))
    title(fig, 'All 27 BBH tasks', 'Official 3-shot prompts; 15 tasks have no implant')
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.64])
    rows = [('Qwen3-0.6B-Base', a['base'], GRAY), ('+ nine implants', a['implants_on_12_base_elsewhere'], GREEN),
            ('Qwen2.5-7B', a['q7b'], C7B), ('Qwen2.5-14B', a['q14b'], C14B)]
    for i, (lab, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.62, color=c, edgecolor=BG, linewidth=2)
        ax.text(v + 1.2, y, f'{v:.1f}', va='center', fontsize=38, fontweight='bold')
        ax.text(1.5, y, lab, va='center', fontsize=26, color=BG if c != GRAY else INK, fontweight='bold')
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(rows) - 0.4); ax.axis('off')
    save(fig, '7_full27.png')


FT = json.load(open(os.path.join(HERE, '..', 'finetuned_phase', 'results', 'reported_numbers.json')))


def eyes():
    e = FT['eyes']; n = e['seed0']['train_questions']
    yes = e['seed0']['with_implants'][:2] + [e['five_seed_with_implants']['4000'], e['five_seed_with_implants']['16000']]
    no = e['seed0']['no_implants'][:2] + [e['five_seed_mean_no_implants']['4000'], e['five_seed_mean_no_implants']['16000']]
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, 'Giving Qwen eyes', 'Qwen2.5-0.5B (fine-tuned), arithmetic on 3\u20134 handwritten MNIST digits per question')
    ax = fig.add_axes([0.09, 0.14, 0.80, 0.62])
    ax.plot(n, no, color=GRAY, lw=5, marker='o', ms=15, zorder=2)
    ax.plot(n, yes, color=GREEN, lw=6, marker='o', ms=17, zorder=3)
    ax.set_xscale('log')
    ax.text(40, 89, 'eyes + calculator implants', color=GREEN, fontsize=27, fontweight='bold')
    ax.text(8.6, 17, 'no implants', color=INK2, fontsize=27, fontweight='bold')
    ax.annotate(f'{yes[1]:.0f} vs {no[1]}', xy=(32, 50), fontsize=30, fontweight='bold', ha='center')
    ax.set_xticks(n); ax.set_xticklabels([f'{x:,}' for x in n]); ax.minorticks_off()
    ax.set_xlim(6, 22000); ax.set_ylim(-3, 105); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=22, length=0)
    for yv in (0, 25, 50, 75, 100): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    ax.set_xlabel('fine-tuning questions', fontsize=23); ax.set_ylabel('accuracy (%)', fontsize=23)
    fig.text(0.09, 0.02, '8 and 32: seed 0; 4,000 and 16,000: mean of 5 seeds. Not re-run for this repository.', fontsize=17, color=INK2)
    save(fig, '6_eyes.png')


def finetune():
    f = FT['finetune_comparison']['five_seed_mean_std']
    fig = plt.figure(figsize=(16, 9.5))
    title(fig, 'Or fine-tune Qwen to call the implant', 'Qwen2.5-0.5B, fine-tuned on GSM8K solutions, 5 seeds each')
    ax = fig.add_axes([0.07, 0.12, 0.90, 0.56])
    cols = [GRAY, INK2, GREEN]; w = 0.25
    for g, ds in enumerate(['GSM8K', 'GSM-Hard', 'arithmetic, new phrasings']):
        for k, (v, e) in enumerate(zip(f[ds]['mean'], f[ds]['std'])):
            x = g + (k - 1) * (w + 0.02)
            ax.bar(x, v, width=w, color=cols[k], edgecolor=BG, linewidth=2, zorder=2)
            ax.errorbar(x, v, yerr=e, color=INK, lw=2, capsize=6, zorder=3)
            ax.text(x, v + e + 1.5, f'{v:.0f}', ha='center', fontsize=26, fontweight='bold' if k == 2 else 'normal')
        ax.text(g, -9, ds.replace(', ', ',\n'), ha='center', va='top', fontsize=25, fontweight='bold', linespacing=1.1)
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 108); ax.set_xticks([]); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(labelsize=20, length=0)
    for yv in (25, 50, 75, 100): ax.axhline(yv, color=GRID, lw=1.5, zorder=0)
    ax.set_ylabel('accuracy (%)', fontsize=22)
    for k, (a, c) in enumerate(zip(f['arms'], cols)):
        x = [0.07, 0.30, 0.64][k]
        fig.patches.append(FancyBboxPatch((x, 0.755), 0.018, 0.032, boxstyle='round,pad=0.002', transform=fig.transFigure, color=c))
        fig.text(x + 0.024, 0.771, a, va='center', fontsize=22)
    save(fig, '8_finetune.png')


if __name__ == '__main__':
    hero(); per_task(); gsm(); gate(); adding(); full27(); eyes(); finetune()
