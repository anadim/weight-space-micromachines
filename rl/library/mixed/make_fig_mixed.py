"""fig_mixed_routing.png (EARLY CHECKPOINT): tool x task-type heatmaps (fired calls per answer on greedy dev = triggers x
in-wire fire rate) at u0 and the latest dev point, for m1 (LIB, clean SFT) and m4 (LIB, noisy-routing SFT); plus dev
accuracy per task type (u0 vs latest; hatched = u0). Plotting only."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, '..', '..', 'results', 'library', 'figdata')
if os.path.exists('/System/Library/Fonts/Avenir Next.ttc'): fm.fontManager.addfont('/System/Library/Fonts/Avenir Next.ttc')
INK, SEC, GRID, GREEN, GRAY = '#222', '#6b7280', '#e5e7eb', '#118659', '#b9bec6'
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': 14, 'axes.edgecolor': INK, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.titlesize': 15.5})
TYPES = ['gsm', 'bbh', 'ifeval', 'combo']; TL = ['GSM', 'BBH', 'IFEval', 'combo']
TOOLS = ['calc', 'expr_gen', 'calc_gen', 'track_gen', 'order_gen', 'interval_gen', 'turtle_gen', 'truth_gen', 'bool_gen', 'ifeval']
TN = ['calculator', 'expr (BBH)', 'calc (BBH)', 'track', 'order', 'interval', 'walk', 'truth', 'bool', 'IFEval']
ON = {'gsm': {'calc', 'expr_gen', 'calc_gen'}, 'bbh': set(TOOLS) - {'calc', 'ifeval'}, 'ifeval': {'ifeval'}, 'combo': set(TOOLS)}
cmap = LinearSegmentedColormap.from_list('g', ['#ffffff', '#cfe9dc', GREEN])


def devs(name):
    p = os.path.join(FD, f'{name}_steps.jsonl')
    return [json.loads(l)['dev'] for l in open(p) if '"dev"' in l[:8]] if os.path.exists(p) else []


def mat(d):
    M = np.zeros((len(TOOLS), len(TYPES)))
    for j, t in enumerate(TYPES):
        for i, n in enumerate(TOOLS):
            v = d.get(f'dev_tools_{t}', {}).get(n)
            if v and v['nb']: M[i, j] = v['bursts'] * v['fire']
    return M


runs = [('mix_m1_mixLIB', 'clean SFT (m1)'), ('mix_m4_mixLIBnoisy', 'noisy-routing SFT (m4)')]
D = {k: devs(k) for k, _ in runs}
fig = plt.figure(figsize=(21, 8.6))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.35], wspace=0.35)
vmax = 2.5
for r, (k, lab) in enumerate(runs):
    ds = D[k]
    if not ds: continue
    for c, d in enumerate((ds[0], ds[-1])):
        ax = fig.add_subplot(gs[0, 2 * r + c]); M = mat(d)
        ax.imshow(np.minimum(M, vmax), cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        for i in range(len(TOOLS)):
            for j in range(len(TYPES)):
                if M[i, j] > 0.005:
                    off = TOOLS[i] not in ON[TYPES[j]]
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=11.5, color=INK if M[i, j] < 1.6 else 'white',
                            fontweight='bold' if off else 'normal')
                    if off: ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch='///', edgecolor=SEC, lw=0))
        ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(TL, fontsize=12)
        ax.set_yticks(range(len(TOOLS))); ax.set_yticklabels(TN if (r == 0 and c == 0) else [''] * len(TOOLS))
        ax.set_title(f"{lab}\nupdate {d['u']}", color=INK)
        for s in ax.spines.values(): s.set_color(GRID)
        ax.set_xticks(np.arange(-.5, len(TYPES)), minor=True); ax.set_yticks(np.arange(-.5, len(TOOLS)), minor=True)
        ax.grid(which='minor', color=GRID, lw=1); ax.tick_params(which='minor', length=0)
ax = fig.add_subplot(gs[0, 4])
w = 0.2; x = np.arange(len(TYPES))
for r, (k, lab) in enumerate(runs):
    ds = D[k]
    if not ds: continue
    col = GREEN if r == 0 else INK
    a0 = [ds[0].get(f'dev_{t}', 0) for t in TYPES]; a1 = [ds[-1].get(f'dev_{t}', 0) for t in TYPES]
    ax.bar(x + (2 * r - 1.5) * w, a0, w, color='white', edgecolor=col, hatch='//', lw=1.6, label=f'{lab}, u0')
    ax.bar(x + (2 * r - 0.5) * w, a1, w, color=col, edgecolor=col, lw=1.6, label=f"{lab}, u{ds[-1]['u']}")
ax.set_xticks(x); ax.set_xticklabels(TL); ax.set_ylabel('greedy dev accuracy (%)')
ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2); ax.set_title('Accuracy by task type', color=INK)
_fin = all(D[k] and D[k][-1]['u'] >= 150 for k, _ in runs)
fig.suptitle(('Tool library under RL on mixed tasks, u0 vs u150 (final)' if _fin else 'Tool library under RL on mixed tasks — EARLY CHECKPOINT') + ' — fired calls per answer, greedy dev; hatched = tool fired off its task type', fontsize=15, color=INK, y=1.0)
fig.savefig(os.path.join(HERE, '..', 'fig_mixed_routing.png'), dpi=150, bbox_inches='tight')
print('saved', {k: [d['u'] for d in v] for k, v in D.items()})
