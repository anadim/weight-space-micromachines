"""Blog figures rl4_toolbox.png and rl5_mixed.png (+ .svg). Data from rl/results/library/figdata (run logs).
Style: Avenir Next, blog green palette, titles >= 20 pt, labels >= 17 pt, ticks >= 15 pt, lines >= 3 px, direct labels,
2 panels, ~1600x700 px, white background."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, '..', '..', 'results', 'library', 'figdata')
if os.path.exists('/System/Library/Fonts/Avenir Next.ttc'): fm.fontManager.addfont('/System/Library/Fonts/Avenir Next.ttc')
INK, SEC, GRID, GREEN, GRAY, LGREEN = '#222', '#6b7280', '#e5e7eb', '#118659', '#b9bec6', '#8cc9ab'
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': 16, 'axes.titlesize': 21, 'axes.titleweight': 'bold',
                     'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'axes.edgecolor': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'axes.linewidth': 1.4})


def load(n):
    return [json.loads(l) for l in open(os.path.join(FD, n))]


def roll(R, tool):
    xs, ys = [], []
    for r in R:
        v = r.get(f'roll_t_{tool}')
        if 'u' in r and v and v['nb']: xs.append(r['u']); ys.append(100 * v['fire'])
    k = 15; sm = [np.mean(ys[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(ys))]
    return xs, sm


def save(fig, name):
    for ext in ('png', 'svg'): fig.savefig(os.path.join(HERE, f'{name}.{ext}'), dpi=100, facecolor='white')
    plt.close(fig)


# ---------------- rl4: overnight GSM tool library ----------------
lib, scr = load('LIB_s0_steps.jsonl'), load('LIBscr_s0_steps.jsonl')
fig, ax = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.28))
a = ax[0]
for tool, lab in (('calc', 'calculator'), ('expr_gen', 'expression tool'), ('calc_gen', 'second calculator')):
    x, y = roll(lib, tool); a.plot(x, y, color=GREEN, lw=3.5)
for tool, lab in (('expr_gen', 'expression tool'), ('calc_gen', 'second calculator')):
    x, y = roll(scr, tool); a.plot(x, y, color=GRAY, lw=3.5)
a.text(150, 99, '3 correct arithmetic tools', color=GREEN, fontsize=17, ha='center', va='bottom', fontweight='bold')
a.text(150, 12, '2 scrambled tools', color=SEC, fontsize=17, ha='center', va='bottom', fontweight='bold')
a.set_ylim(0, 110); a.set_xlim(0, 300); a.set_yticks([0, 25, 50, 75, 100])
a.set_xlabel('RL update'); a.set_ylabel('calls let through (%)'); a.set_title('RL shuts off the broken tools')
a.grid(axis='y', color=GRID, lw=1); a.set_axisbelow(True)
# right: accuracy all tools vs all off (dev, u300), by half; seed 1 as dots
b = ax[1]
fin = [r['dev'] for r in lib if 'dev' in r and r['dev']['u'] == 300][-1]
off = [r['dev_toolsoff'] for r in lib if 'dev_toolsoff' in r and r['dev_toolsoff'].get('u') == 300][-1]
s1 = load('LIB_s1_steps.jsonl') if os.path.exists(os.path.join(FD, 'LIB_s1_steps.jsonl')) else []
groups = [('GSM8K problems', 'gsm'), ('big-number problems', 'scaled')]
x = np.arange(2); w = 0.36
on_v = [fin[f'dev_{k}'] for _, k in groups]; off_v = [off[f'devtoolsoff_{k}'] for _, k in groups]
b.bar(x - w / 2, on_v, w, color=GREEN, label='all tools on')
b.bar(x + w / 2, off_v, w, color='white', edgecolor=SEC, hatch='//', lw=2, label='all tools off')
if s1:
    f1 = [r['dev'] for r in s1 if 'dev' in r and r['dev']['u'] == 300]; o1 = [r['dev_toolsoff'] for r in s1 if 'dev_toolsoff' in r]
    if f1 and o1:
        b.scatter(x - w / 2, [f1[-1][f'dev_{k}'] for _, k in groups], s=90, color=INK, zorder=5)
        b.scatter(x + w / 2, [o1[-1][f'devtoolsoff_{k}'] for _, k in groups], s=90, color=INK, zorder=5)
for i in range(2):
    b.text(x[i] - w / 2, on_v[i] / 2, f'{on_v[i]:.0f}', ha='center', va='center', fontsize=19, color='white', fontweight='bold')
    b.text(x[i] + w / 2, off_v[i] / 2, f'{off_v[i]:.0f}', ha='center', va='center', fontsize=19, color=INK, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'))
for i in range(2):
    b.text(x[i] - w / 2, 5, 'tools on', ha='center', fontsize=15, color='white', fontweight='bold')
    b.text(x[i] + w / 2, 5, 'tools off', ha='center', fontsize=15, color=INK, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
if s1:
    b.scatter([0.72], [0.955], s=90, color=INK, transform=b.transAxes, clip_on=False)
    b.text(0.75, 0.955, 'second seed', transform=b.transAxes, ha='left', va='center', fontsize=15, color=INK)
b.set_xticks(x); b.set_xticklabels([g for g, _ in groups]); b.set_ylim(0, 95); b.set_ylabel('accuracy after RL (%)')
b.set_title('The tools carry the big numbers'); b.grid(axis='y', color=GRID, lw=1); b.set_axisbelow(True)
fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12)
save(fig, 'rl4_toolbox')

# ---------------- rl5: mixed tasks ----------------
T = ['gsm', 'bbh', 'ifeval', 'combo']; TL = ['GSM', 'BBH', 'IFEval', 'combo']
m1 = json.load(open(os.path.join(FD, 'mix_m1_mixLIB_test.json')))
plain = dict(gsm=31.8, bbh=46.0, ifeval=41.3, combo=14.8)          # m3-64 plain RL test (coordinator report)
fig, ax = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.28))
a = ax[0]; x = np.arange(4); w = 0.26
vals = [('plain RL', [plain[t] for t in T], dict(color=GRAY)),
        ('tools off', [m1['toolsoff'][f'test_{t}'] for t in T], dict(color='white', edgecolor=SEC, hatch='//', lw=2)),
        ('tools on', [m1['full'][f'test_{t}'] for t in T], dict(color=GREEN))]
for i, (lab, v, kw) in enumerate(vals):
    a.bar(x + (i - 1) * w, v, w, **kw)
for i, t in enumerate(T):
    v = vals[2][1][i]; a.text(x[i] + w, v + 1.5, f'{v:.0f}', ha='center', fontsize=16, color=GREEN, fontweight='bold')
a.text(x[0] - w, vals[0][1][0] + 2, 'plain RL', ha='center', fontsize=15, color=SEC, rotation=90, va='bottom', fontweight='bold')
a.text(x[0], vals[1][1][0] + 2, 'tools off', ha='center', fontsize=15, color=SEC, rotation=90, va='bottom', fontweight='bold')
a.text(x[0] + w, vals[2][1][0] + 8, 'tools on', ha='center', fontsize=15, color=GREEN, rotation=90, va='bottom', fontweight='bold')
a.set_xticks(x); a.set_xticklabels(TL); a.set_ylim(0, 100); a.set_ylabel('test accuracy (%)')
a.set_title('Tools on wins on every task type (1 seed)', pad=14); a.grid(axis='y', color=GRID, lw=1); a.set_axisbelow(True)
b = ax[1]
m4 = [json.loads(l)['dev'] for l in open(os.path.join(FD, 'mix_m4_mixLIBnoisy_steps.jsonl')) if l.startswith('{"dev"')]
def calls(d, t): v = d.get(f'dev_tools_{t}', {}).get('calc'); return v['bursts'] * v['fire'] if v and v['nb'] else 0.0
us = [d['u'] for d in m4]
b.plot(us, [calls(d, 'ifeval') for d in m4], color=GRAY, lw=3.5, marker='o', ms=8)
b.plot(us, [calls(d, 'bbh') for d in m4], color=GREEN, lw=3.5, marker='o', ms=8)
b.text(152, calls(m4[-1], 'ifeval'), '  in IFEval answers\n  costs −0.07 reward', color=SEC, fontsize=15, va='center')
b.text(152, calls(m4[-1], 'bbh'), '  in BBH answers\n  costs −0.39 reward', color=GREEN, fontsize=15, va='center', fontweight='bold')
b.set_xlim(0, 150); b.set_ylim(0, 0.32); b.set_xticks([0, 50, 100, 150])
b.set_xlabel('RL update'); b.set_ylabel('wrong calculator calls per answer')
b.set_title('RL drops costly wrong calls (1 seed)', pad=14); b.grid(axis='y', color=GRID, lw=1); b.set_axisbelow(True)
fig.subplots_adjust(left=0.07, right=0.84, top=0.88, bottom=0.12)
save(fig, 'rl5_mixed')
print('saved')
