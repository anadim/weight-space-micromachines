"""Blog figure for the RL-wiring study. Reads pulled logs in figdata/ (steps.jsonl per run) and writes fig_rl_wiring.png.
Left: CW-full, share of calculator calls the learned in-wire vetoes, by the call's largest operand, early/mid/late.
Right: dev-800 accuracy at u300 per wiring arm; pale bar = same trained host with the ORIGINAL hand wiring."""
import os, json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, '..', 'results', 'wiring', 'figdata')
INK, INK2, GRID, GREEN, PALE = '#222222', '#6b7280', '#e5e7eb', '#118659', '#e7f3ee'
MID = '#5fae8f'
for f in font_manager.findSystemFonts():
    if 'Avenir' in f:
        try: font_manager.fontManager.addfont(f)
        except Exception: pass
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': 11, 'axes.edgecolor': INK2, 'axes.linewidth': 0.8,
                     'figure.dpi': 160, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})


def load(tag):
    p = os.path.join(FD, tag + '.jsonl')
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def dev_at(rows, u, key='dev', field='acc'):
    for r in rows:
        if key in r and r[key].get('u') == u:
            d = r[key]; t = [k for k in d if k.endswith('_' + field)][0]; return d[t]
    return None


cw = load('CW_s0')
BUCKETS = ['1-2', '3-4', '5-6', '7+']
CKPT = [int(x) for x in os.environ.get('CW_CKPT', '25,100,300').split(',')]
veto = {}
for r in cw:
    if 'dev' in r and r['dev'].get('u') in CKPT and r['dev'].get('dev_veto_by_digits'):
        d = r['dev']['dev_veto_by_digits']
        veto[r['dev']['u']] = [100 * d[b][1] / max(1, d[b][0]) if b in d else 0 for b in BUCKETS]
CKPT = [u for u in CKPT if u in veto]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
w = 0.8 / max(1, len(CKPT)); cols = [GREEN, MID, PALE][:len(CKPT)]
for j, (u, c) in enumerate(zip(CKPT, cols)):
    xs = [i + (j - (len(CKPT) - 1) / 2) * w for i in range(len(BUCKETS))]
    axL.bar(xs, veto[u], w, color=c, edgecolor=GREEN, linewidth=0.6, label=f'update {u}')
    for x, v in zip(xs, veto[u]):
        axL.text(x, v + 1.5, f'{v:.0f}', ha='center', fontsize=8, color=INK)
axL.set_xticks(range(len(BUCKETS))); axL.set_xticklabels([f'{b} digits' for b in BUCKETS])
axL.set_ylabel('calls vetoed by the learned trigger (%)'); axL.set_ylim(0, 110)
axL.set_title('RL redesigns when to call the calculator', loc='left', fontsize=12, pad=20)
axL.text(0, 1.01, 'largest operand of the equation; greedy dev, CW seed 0', transform=axL.transAxes, fontsize=8.5, va='bottom', color=INK2)
axL.legend(frameon=False, fontsize=9)

ARMS = [('REF_a2long', 'fixed\nhand wiring', None), ('W2_s0', 'fixed +\nnew links', None),
        ('W2w_s0_lr1e2', 'trainable\nhand write', 'dev_reset_curve'), ('W3e_s0', 'links only\n(no write)', None),
        ('CW_s0', 'co-designed\n(in + out)', 'dev_cw_hand')]
ARMS = [a for a in ARMS if load(a[0])]
for i, (tag, lab, ctrl) in enumerate(ARMS):
    rows = load(tag)
    v = dev_at(rows, 300)
    if v is not None:
        axR.bar(i - 0.18 if ctrl else i, v, 0.36, color=GREEN)
        axR.text(i - 0.18 if ctrl else i, v + 0.8, f'{v:.0f}', ha='center', fontsize=9)
    if ctrl:
        cv = dev_at(rows, 300, key=ctrl)
        if cv is not None:
            axR.bar(i + 0.18, cv, 0.36, color=PALE, edgecolor=GREEN, linewidth=0.6)
            axR.text(i + 0.18, cv + 0.8, f'{cv:.0f}', ha='center', fontsize=9, color=INK2)
axR.set_xticks(range(len(ARMS))); axR.set_xticklabels([a[1] for a in ARMS], fontsize=9)
axR.set_ylabel('dev accuracy after 300 RL updates (%)'); axR.set_ylim(40, 80)
axR.set_title('Trainable wiring does not beat the hand wiring', loc='left', fontsize=12, pad=20)
axR.text(0, 1.01, 'pale bar: same trained host, original hand wiring restored', transform=axR.transAxes, fontsize=8.5, va='bottom', color=INK2)
for ax in (axL, axR):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
plt.tight_layout()
out = os.path.join(HERE, 'fig_rl_wiring.png')
plt.savefig(out, bbox_inches='tight'); plt.savefig(out.replace('.png', '.svg'), bbox_inches='tight')
print('wrote', out)
