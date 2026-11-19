"""Blog figure: does trainable wiring / a trainable calculator help, and does W3 discover the calculator?"""
import os, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'runs', 'results.json')))
INK, INK2, GRID, GREEN, PALE = '#222222', '#6b7280', '#e5e7eb', '#118659', '#e7f3ee'
for f in font_manager.findSystemFonts():
    if 'Avenir' in f:
        try: font_manager.fontManager.addfont(f)
        except Exception: pass
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': 11, 'axes.edgecolor': INK2,
                     'axes.linewidth': 0.8, 'figure.dpi': 140})


def dev64(tag, k='dev_acc'):
    d = R.get(tag, {}).get('devs', {})
    return d.get('64', d.get(64, {})).get(k)

def dev0(tag, k='dev_acc'):
    d = R.get(tag, {}).get('devs', {})
    return d.get('0', d.get(0, {})).get(k)

# Left: final dev (overall) per arm, with host-only(implant off) overlaid.
arms = [('2_s0', 'arm 2\nfixed write'), ('W2_s0', 'W2\nwrite+links'), ('W1_s0_impllr1e5', 'W1\ntrain writer'),
        ('W4_s0_impllr1e5', 'W4\ntrain both'), ('W3z_s0', 'W3z\nlinks only\n(0-init)'),
        ('W3r_s0', 'W3r\nlinks only\n(rand)'), ('1_s0', 'arm 1\nno impl')]
arms = [(t, l) for t, l in arms if t in R]
labels = [l for _, l in arms]
full = [dev64(t) for t, _ in arms]
off = [(R[t].get('hostonly') or {}).get('devhostonly_acc') for t, _ in arms]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
x = range(len(arms))
axL.bar([i - 0.2 for i in x], full, 0.4, color=GREEN, label='calculator on')
axL.bar([i + 0.2 for i in x], [o if o else 0 for o in off], 0.4, color=PALE, edgecolor=GREEN, label='implant off (host)')
for i, v in enumerate(full):
    if v is not None: axL.text(i - 0.2, v + 0.5, f'{v:.0f}', ha='center', fontsize=9, color=INK)
a1 = dev64('1_s0')
if a1: axL.axhline(a1, ls='--', lw=0.8, color=INK2)
axL.set_xticks(list(x)); axL.set_xticklabels(labels, fontsize=8)
axL.set_ylabel('dev-800 accuracy'); axL.set_title('Does trainable wiring help?', color=INK, loc='left')
axL.legend(frameon=False, fontsize=9); axL.set_ylim(0, 75)
for s in ('top', 'right'): axL.spines[s].set_visible(False)
axL.yaxis.grid(True, color=GRID, lw=0.6); axL.set_axisbelow(True)

# Right: scaled-half accuracy curve over updates for arm1 / W3z / W3r / arm2 -> does W3 discover the calculator?
def curve(tag, k='dev_scaled'):
    d = R.get(tag, {}).get('devs', {})
    us = sorted(int(u) for u in d)
    return us, [d[str(u) if str(u) in d else u].get(k) for u in us]

for tag, lab, col, ls in [('1_s0', 'arm 1 (no impl)', INK2, '--'), ('W3z_s0', 'W3z links 0-init', GREEN, '-'),
                          ('W3r_s0', 'W3r links rand', '#7bbfa6', '-'), ('2_s0', 'arm 2 fixed write', INK, ':')]:
    if tag not in R: continue
    us, ys = curve(tag)
    if us: axR.plot(us, ys, marker='o', ms=4, color=col, ls=ls, label=lab)
axR.set_xlabel('GRPO update'); axR.set_ylabel('scaled-half dev accuracy')
axR.set_title('Does RL discover the calculator organically?', color=INK, loc='left')
axR.legend(frameon=False, fontsize=9)
for s in ('top', 'right'): axR.spines[s].set_visible(False)
axR.yaxis.grid(True, color=GRID, lw=0.6); axR.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(HERE, 'fig_rl_wiring.png'), bbox_inches='tight')
plt.savefig(os.path.join(HERE, 'fig_rl_wiring.svg'), bbox_inches='tight')
print('wrote fig_rl_wiring.png/.svg')
