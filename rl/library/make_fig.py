"""fig_rl_library.png: (a) rollout fire rate per tool vs update (LIB solid, scrambled dashed), (b) tool bursts per answer
(rollout, LIB), (c) greedy dev accuracy vs update: LIB vs LIB-scrambled vs CW-lean (calc only) vs arm 2 (calc, hand),
(d) leave-one-out on dev at u300 (hatched bars). Plotting only (laptop)."""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, '..', 'results', 'library', 'figdata')
for f in ('/System/Library/Fonts/Avenir Next.ttc',):
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 15,
                     'axes.edgecolor': '#222', 'axes.labelcolor': '#222', 'xtick.color': '#6b7280', 'ytick.color': '#6b7280',
                     'text.color': '#222', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e5e7eb', 'grid.linewidth': 1.0, 'legend.frameon': False, 'legend.fontsize': 12.5})
INK, SEC, GREEN, GRAY = '#222', '#6b7280', '#118659', '#b9bec6'


def load(name):
    p = os.path.join(FD, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def dev_curve(rows):
    return sorted((r['dev']['u'], r['dev']['dev_acc']) for r in rows if 'dev' in r)


def roll(rows, tool, key):
    out = []
    for r in rows:
        if 'u' in r and f'roll_t_{tool}' in r:
            d = r[f'roll_t_{tool}']
            if key == 'fire' and d['nb'] == 0: continue
            out.append((r['u'], d[key]))
    return out


def smooth(xy, k=9):
    if not xy: return [], []
    xs, ys = zip(*xy); sm = []
    for i in range(len(ys)):
        w = ys[max(0, i - k // 2): i + k // 2 + 1]; sm.append(sum(w) / len(w))
    return xs, sm


lib = load('LIB_s0_steps.jsonl'); scr = load('LIBscr_s0_steps.jsonl'); cwl = load('cwlean_s0_steps.jsonl')
a2 = load('ref_a2long_H100abl.jsonl'); lib1 = load('LIB_s1_steps.jsonl')
TOOLS = ['calc', 'track_gen', 'order_gen', 'interval_gen', 'turtle_gen', 'truth_gen', 'expr_gen', 'calc_gen', 'bool_gen', 'ifeval']
active = [t for t in TOOLS if any(r.get(f'roll_t_{t}', {}).get('nb', 0) > 0 for r in lib + scr)]
silent = [t for t in TOOLS if t not in active]
STY = {'calc': dict(color=GREEN, ls='-', lw=3.2), 'expr_gen': dict(color=INK, ls='-', lw=2.6),
       'calc_gen': dict(color=SEC, ls='-', lw=2.6), 'bool_gen': dict(color=GRAY, ls='-', lw=2.6)}
LBL = {'calc': 'calculator (calc_s4)', 'expr_gen': 'expr_gen (BBH arithmetic)', 'calc_gen': 'calc_gen (BBH calculator)',
       'bool_gen': 'bool_gen', 'ifeval': 'IFEval circuit'}

fig, ax = plt.subplots(1, 4, figsize=(26, 6.6), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.15, 0.95]))
# (a) fire rate
for t in active:
    st = STY.get(t, dict(color=GRAY, ls='-', lw=2.4))
    xs, ys = smooth(roll(lib, t, 'fire'))
    if xs: ax[0].plot(xs, [100 * y for y in ys], label=LBL.get(t, t), **st)
    xs, ys = smooth(roll(scr, t, 'fire'))
    if xs: ax[0].plot(xs, [100 * y for y in ys], **dict(st, ls=(0, (4, 2.5)), lw=st['lw'] - 0.6), alpha=0.9)
ax[0].plot([], [], color=SEC, ls=(0, (4, 2.5)), lw=2, label='dashed: scrambled library')
ax[0].set_title('In-wire fire rate per tool (rollouts)'); ax[0].set_xlabel('update'); ax[0].set_ylabel('% of tool calls fired')
ax[0].set_ylim(-3, 103); ax[0].legend(loc=(0.36, 0.42))
if silent:
    ax[0].text(0.02, 0.03, 'never triggered on GSM8K: ' + ', '.join(s.replace('_gen', '') for s in silent), transform=ax[0].transAxes,
               fontsize=11.5, color=SEC)
# (b) bursts per answer
for t in active:
    st = STY.get(t, dict(color=GRAY, ls='-', lw=2.4))
    xs, ys = smooth(roll(lib, t, 'bursts'))
    if xs: ax[1].plot(xs, ys, **st)
    xs, ys = smooth(roll(scr, t, 'bursts'))
    if xs: ax[1].plot(xs, ys, **dict(st, ls=(0, (4, 2.5)), lw=st['lw'] - 0.6))
ax[1].set_title('Tool triggers per answer (rollouts)'); ax[1].set_xlabel('update'); ax[1].set_ylabel('triggers / answer')
# (c) dev
for rows, lab, st in ((lib, 'library (s0)', dict(color=GREEN, lw=3.4)), (lib1, 'library (s1, A40, 24-32 prompts)', dict(color=GREEN, lw=2.2, ls=(0, (1, 1.5)))),
                      (load('LIBnoIF_s0_steps.jsonl'), 'library w/o IFEval (A40, 40 prompts)', dict(color=GREEN, lw=1.6, ls=(0, (6, 2)))),
                      (scr, 'scrambled library', dict(color=GRAY, lw=3.2, ls=(0, (4, 2.5)))),
                      (cwl, 'CW-lean: calculator only', dict(color=INK, lw=2.6)),
                      (a2, 'arm 2: calculator, hand wiring', dict(color=SEC, lw=2.2, ls=(0, (1, 1.5))))):
    c = [(u, v) for u, v in dev_curve(rows) if u <= 300]
    if c: ax[2].plot(*zip(*c), label=lab, marker='o', ms=4, **st)
ax[2].set_title('Greedy dev accuracy (split B, n=800)'); ax[2].set_xlabel('update'); ax[2].set_ylabel('accuracy (%)')
ax[2].legend(loc='lower right', fontsize=11.5); ax[2].set_ylim(14, 74)
# (d) LOO
loo = [r['dev_loo'] for r in lib if 'dev_loo' in r]
full = [r['dev'] for r in lib if 'dev' in r and r['dev']['u'] == 300]
if loo and full:
    f0 = full[-1]['dev_acc']
    names = ['full'] + [l['tool'] for l in loo if not l.get('identical_by_construction')]
    vals = [f0] + [l['devloo_acc'] for l in loo if not l.get('identical_by_construction')]
    tf = [r['dev_toolsoff'] for r in lib if 'dev_toolsoff' in r and r['dev_toolsoff'].get('u') == 300]
    if tf: names.append('all off'); vals.append(tf[-1]['devtoolsoff_acc'])
    hat = ['', '//', '\\\\', 'xx', '..', '--', '++']
    for i, (n, v) in enumerate(zip(names, vals)):
        ax[3].bar(i, v, color='white' if i else GREEN, edgecolor=INK if i else GREEN, hatch=hat[i % len(hat)] if i else '', lw=1.8)
        ax[3].text(i, v + 0.6, f'{v:.1f}', ha='center', fontsize=12.5)
    ax[3].set_xticks(range(len(names))); SH = {'full': 'full', 'calc': '− calc_s4', 'expr_gen': '− expr_gen', 'calc_gen': '− calc_gen', 'all off': 'all tools off'}
    ax[3].set_xticklabels([SH.get(n, '− ' + n) for n in names], rotation=20)
    ax[3].set_ylim(min(vals) - 8, max(vals) + 9)
    nid = [l['tool'] for l in loo if l.get('identical_by_construction')]
    if nid: ax[3].text(0.0, 0.95, f'the other {len(nid)} tools never triggered on dev:\nremoving them changes nothing', transform=ax[3].transAxes, fontsize=11.5, color=SEC, va='top')
ax[3].set_title('Leave one tool out (dev, u300)'); ax[3].set_ylabel('accuracy (%)'); ax[3].grid(axis='x', visible=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_rl_library.png'), dpi=150)
print('saved', active, silent)
