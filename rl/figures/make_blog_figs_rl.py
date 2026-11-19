"""Blog RL figures, redone 2026-11-19 in the blog style (large fonts, few captions, thick lines, direct labels).

rl1_calc_after_rl : GSM8K / GSM-Hard test after 300 RL updates (3 seeds: mean bar, min-max range) + dev curve (mean of 3 seeds).
rl2_writing       : how RL bends the writing (64-update interp analysis, rl_interp/; one seed per arm, arm R seed 1 as a dot).
rl3_competence    : half-broken calculator (CWinOx, 3 seeds): per-op calls let through, start vs end; dev at u300.

Sources:
  test numbers : rl/results/gsm_grpo/H100/final + rl/gsm_grpo/RESULTS_ledger.md (seeds 0, 1, H100) + rl/wiring/NOTEBOOK.md 'Headline seed 2' (A100,
                 confirmed against A100 host runs/RAW_test_a{1,2}_s2.out).
  dev curves   : data/a{1,2}_s{0,1}.jsonl (= rl/results/curves/h300) and data/a{1,2}_s2.out (A100 host runs/RAW_arm{1,2}_s2.out).
  writing      : rl/interp/NOTEBOOK.md Q2 table (unit words 29.8 / 23.5 / 16.7) and rl/interp/stats_aR.json (fires 74.6 -> 12.1 / 11.2).
  competence   : rl/results/wiring/figdata/CWinOx_s{0,1,2}.jsonl.
"""
import os, json, ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'data')
FD = os.path.join(HERE, '..', 'results', 'wiring', 'figdata')
INK, INK2, GRID, GREEN, PALE, GREY, MID = '#222222', '#6b7280', '#e5e7eb', '#118659', '#e7f3ee', '#b9bec6', '#5fae8f'
for f in font_manager.findSystemFonts():
    if 'Avenir' in f:
        try: font_manager.fontManager.addfont(f)
        except Exception: pass
TITLE, LABEL, TICK, VAL = 22, 18, 16, 16
plt.rcParams.update({'font.family': 'Avenir Next', 'font.size': TICK, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK2, 'axes.edgecolor': INK2, 'axes.linewidth': 1.5,
                     'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'axes.labelsize': LABEL,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
DPI = 100; SIZE = (16, 7)          # 1600 x 700 px


def style(ax, title, ylabel, ylim):
    ax.set_title(title, loc='left', fontsize=TITLE, fontweight='bold', pad=16)
    ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=1.5); ax.set_axisbelow(True)
    ax.tick_params(axis='both', width=1.5, length=6)


def save(fig, name):
    fig.tight_layout(w_pad=4)
    p = os.path.join(HERE, name)
    fig.savefig(p + '.png', dpi=DPI); fig.savefig(p + '.svg'); plt.close(fig); print('wrote', p + '.png/.svg')


def barlabel(ax, x, y, txt=None, dy=1.2, color=INK):
    ax.text(x, y + dy, txt if txt is not None else f'{y:.0f}', ha='center', va='bottom', fontsize=VAL, fontweight='bold', color=color)


# ---------------------------------------------------------------- rl1
TEST = {  # seeds 0, 1, 2
    'GSM8K':    {'plain\nRL': [68.2, 70.1, 70.88703563305535], 'RL +\ncalc': [68.8, 70.7, 71.34192570128886], 'RL + calc,\ncalc off': [68.0, 70.1, 70.05307050796057]},
    'GSM-Hard': {'plain\nRL': [42.4, 43.7, 44.95830174374526], 'RL +\ncalc': [54.7, 56.0, 55.799848369977255], 'RL + calc,\ncalc off': [40.2, 39.3, 40.56103108415466]},
}
ARMSTY = {'plain\nRL': dict(color=GREY, edgecolor=GREY), 'RL +\ncalc': dict(color=GREEN, edgecolor=GREEN),
          'RL + calc,\ncalc off': dict(color=PALE, edgecolor=GREEN)}


def dev_curve(arm, seed):
    if seed < 2:
        L = [json.loads(l) for l in open(os.path.join(D, f'a{arm}_s{seed}.jsonl'))]
        return {r['dev']['u']: r['dev']['dev_acc'] for r in L if 'dev' in r and r['dev']['u'] <= 300}
    out, k = {}, 0
    for line in open(os.path.join(D, f'a{arm}_s2.out')):
        if line.startswith('dev@0 '): out[0] = ast.literal_eval(line[6:])['dev_acc']
        elif line.startswith('dev '): k += 25; out[k] = ast.literal_eval(line[4:])['dev_acc']
    return out


def rl1():
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={'width_ratios': [1.25, 1]})
    arms = list(ARMSTY); w = 0.26; ticks, labs = [], []
    for g, (bench, d) in enumerate(TEST.items()):
        for j, arm in enumerate(arms):
            x = g * 1.15 + (j - 1) * (w + 0.03); v = np.array(d[arm]); m = v.mean()
            a.bar(x, m, w, linewidth=2.5, **ARMSTY[arm], zorder=3)
            a.errorbar(x, m, yerr=[[m - v.min()], [v.max() - m]], color=INK, lw=3, capsize=7, capthick=3, zorder=4)
            barlabel(a, x, v.max(), f'{m:.0f}', dy=1.0)
            ticks.append(x); labs.append(arm)
        a.text(g * 1.15, -19, bench, ha='center', va='top', fontsize=LABEL, fontweight='bold', transform=a.transData)
    a.set_xticks(ticks); a.set_xticklabels(labs, fontsize=15, linespacing=1.0)
    a.tick_params(axis='x', length=0, pad=8)
    style(a, 'Test accuracy after 300 RL updates', 'accuracy (%)', (0, 85))
    a.set_xlim(-0.5, 1.65)
    # dev curves, mean of 3 seeds at the updates every seed logged
    us = list(range(0, 301, 50))
    for arm, col, lab in [(1, INK2, 'plain RL'), (2, GREEN, 'RL + calc')]:
        cs = [dev_curve(arm, s) for s in range(3)]
        m = [np.mean([c[u] for c in cs]) for u in us]
        b.plot(us, m, color=col, lw=3.5, marker='o', ms=10, zorder=3)
        b.text(us[-1] + 8, m[-1], lab, color=col, fontsize=LABEL, fontweight='bold', va='center')
    b.set_xlim(-10, 380); b.set_xticks(us); b.set_xlabel('RL updates')
    style(b, 'Dev accuracy during RL', 'accuracy (%)', (40, 75))
    fig.subplots_adjust(bottom=0.2)
    save(fig, 'rl1_calc_after_rl')


# ---------------------------------------------------------------- rl2
def rl2():
    st = json.load(open(os.path.join(HERE, '..', 'interp', 'stats_aR.json')))
    fire0, fire1, fire1b = st['h_base_on']['called'], st['aR0_on']['called'], st['aR1_on']['called']
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE)
    xs = [0, 1, 2]; vals = [29.8, 23.5, 16.7]; labs = ['start', 'plain RL', 'RL + calc']
    sty = [dict(color=GREY, edgecolor=GREY), dict(color='white', edgecolor=GREY, hatch='//'), dict(color=GREEN, edgecolor=GREEN)]
    for x, v, s in zip(xs, vals, sty):
        a.bar(x, v, 0.6, linewidth=3, zorder=3, **s); barlabel(a, x, v, f'{v:.1f}')
    a.set_xticks(xs); a.set_xticklabels(labs, fontsize=LABEL); a.tick_params(axis='x', length=0, pad=8)
    style(a, 'Good calculator: fewer unit words', 'equations with unit words (%)', (0, 40))
    xs = [0, 1]; vals = [fire0, (fire1 + fire1b) / 2]
    for x, v, s in zip(xs, vals, [dict(color=GREY, edgecolor=GREY), dict(color=PALE, edgecolor=GREEN)]):
        b.bar(x, v, 0.6, linewidth=3, zorder=3, **s)
    barlabel(b, 0, fire0); barlabel(b, 1, max(fire1, fire1b), f'{vals[1]:.0f}', dy=3)
    b.plot([0.9, 1.1], [fire1, fire1b], 'o', color=INK, ms=10, zorder=5)   # the two RL seeds
    b.set_xticks(xs); b.set_xticklabels(['start', 'after RL'], fontsize=LABEL); b.tick_params(axis='x', length=0, pad=8)
    style(b, 'Broken calculator: RL stops triggering it', 'equations it fires on (%)', (0, 90))
    save(fig, 'rl2_writing')
    return dict(fire_start=fire0, fire_end_s0=fire1, fire_end_s1=fire1b)


# ---------------------------------------------------------------- rl3
OPS = ['+', '−', '×', '÷']; KEY = {'+': '+', '−': '-', '×': '*', '÷': '/'}


def rl3():
    runs = [[json.loads(l) for l in open(os.path.join(FD, f'CWinOx_s{s}.jsonl'))] for s in range(3)]

    def pooled(lo, hi):
        acc = {k: [0, 0] for k in OPS}
        for rows in runs:
            for r in rows:
                if 'iw_op' in r and lo <= r['u'] <= hi:
                    for k in OPS:
                        c = r['iw_op'].get(KEY[k], [0, 0]); acc[k][0] += c[0]; acc[k][1] += c[1]
        return {k: 100 * acc[k][1] / max(1, acc[k][0]) for k in OPS}, acc

    start, sa = pooled(1, 10); end, ea = pooled(291, 300)
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE)
    w = 0.36
    for i, k in enumerate(OPS):
        a.bar(i - w / 2 - 0.02, start[k], w, color=PALE, edgecolor=GREEN, linewidth=2.5, zorder=3)
        a.bar(i + w / 2 + 0.02, end[k], w, color=GREEN, edgecolor=GREEN, linewidth=2.5, zorder=3)
        barlabel(a, i - w / 2 - 0.02, start[k], color=INK2); barlabel(a, i + w / 2 + 0.02, end[k])
    a.text(-w / 2 - 0.02, start['+'] + 9, 'start', ha='center', fontsize=VAL, color=INK2)
    a.text(w / 2 + 0.02, end['+'] + 9, 'end', ha='center', fontsize=VAL, color=GREEN, fontweight='bold')
    a.set_xticks(range(4)); a.set_xticklabels([f'{k}\n{"broken" if k in "+−" else "exact"}' for k in OPS], fontsize=LABEL)
    a.tick_params(axis='x', length=0, pad=8)
    style(a, 'RL learns where the implant is right', 'calls let through (%)', (0, 112))
    a.set_yticks([0, 25, 50, 75, 100])

    def dev_at(rows, key):
        for r in rows:
            if key in r and r[key].get('u') == 300:
                d = r[key]; return d[[t for t in d if t.endswith('_acc')][0]]
    cases = [('no\ncalculator', 'dev_hostonly', PALE), ('calls\neverything', 'dev_cw_hand', GREY), ('learned\ngate', 'dev', GREEN)]
    devs = {}
    for i, (lab, key, col) in enumerate(cases):
        v = np.array([dev_at(r, key) for r in runs]); devs[lab.replace('\n', ' ')] = v.tolist()
        b.bar(i, v.mean(), 0.6, color=col, edgecolor=GREEN if col != GREY else GREY, linewidth=2.5, zorder=3)
        b.plot([i - 0.12, i, i + 0.12], v, 'o', color=INK, ms=10, zorder=5)
        barlabel(b, i, v.max(), f'{v.mean():.0f}', dy=1.5)
    b.set_xticks(range(3)); b.set_xticklabels([c[0] for c in cases], fontsize=LABEL); b.tick_params(axis='x', length=0, pad=8)
    style(b, 'Calling it blindly hurts', 'dev accuracy (%)', (0, 80))
    save(fig, 'rl3_competence')
    return dict(start={k: round(v, 1) for k, v in start.items()}, end={k: round(v, 1) for k, v in end.items()},
                start_calls={k: sa[k] for k in OPS}, end_calls={k: ea[k] for k in OPS}, dev_u300=devs)


if __name__ == '__main__':
    rl1(); side = {'rl2': rl2(), 'rl3': rl3(),
                   'rl1_test': {b: {k.replace('\n', ' '): v for k, v in d.items()} for b, d in TEST.items()}}
    json.dump(side, open(os.path.join(HERE, 'blog_figs_rl.json'), 'w'), indent=1)
    print(json.dumps(side['rl3'], indent=0)[:600])
