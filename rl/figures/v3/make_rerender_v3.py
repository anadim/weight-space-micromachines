"""v3 re-renders of rl1, rl2, rl3, rl4, rl6 (2026-11-20): same data as the v2 scripts
(../make_blog_figs_rl.py, ../make_rl6.py, ../../review_final/figs/make_rl4_toolbox.py), restyled to the v3 rules:
bigger fonts (labels 22, ticks 20, values 20), colour legends instead of in-plot / in-bar text, values above bars,
seed dots where there are seeds (bar = seed mean)."""
import os, json, ast
import numpy as np
from blogstyle import *

W = os.path.join(HERE, '..', '..')                     # rl/
D1 = os.path.join(W, 'figures', 'data'); FDW = os.path.join(W, 'results', 'wiring', 'figdata'); FDL = os.path.join(W, 'results', 'library', 'figdata')
OUT = {}
jl = lambda p: [json.loads(l) for l in open(p)]


# ---------------------------------------------------------------- rl1
TEST = {  # seeds 0, 1, 2 (same as make_blog_figs_rl.py)
    'GSM8K':    [[68.2, 70.1, 70.88703563305535], [68.8, 70.7, 71.34192570128886], [68.0, 70.1, 70.05307050796057]],
    'GSM-Hard': [[42.4, 43.7, 44.95830174374526], [54.7, 56.0, 55.799848369977255], [40.2, 39.3, 40.56103108415466]],
}


def dev_curve(arm, seed):
    if seed < 2:
        return {r['dev']['u']: r['dev']['dev_acc'] for r in jl(os.path.join(D1, f'a{arm}_s{seed}.jsonl')) if 'dev' in r and r['dev']['u'] <= 300}
    out, k = {}, 0
    for line in open(os.path.join(D1, f'a{arm}_s2.out')):
        if line.startswith('dev@0 '): out[0] = ast.literal_eval(line[6:])['dev_acc']
        elif line.startswith('dev '): k += 25; out[k] = ast.literal_eval(line[4:])['dev_acc']
    return out


def rl1():
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={'width_ratios': [1.2, 1]})
    cols, edges = [GREY, GREEN, PALE], [GREY, GREEN, GREEN]; w = 0.27
    for g, (bench, arms) in enumerate(TEST.items()):
        for k, v in enumerate(arms):
            x = g + (k - 1) * (w + 0.03); m = np.mean(v)
            a.bar(x, m, w, color=cols[k], edgecolor=edges[k], linewidth=2.5, zorder=3)
            dots(a, x, v, spread=0.06, ms=9)
            val(a, x, max(v), f'{m:.0f}', strong=(k == 1), dy=1.5)
    xcats(a, list(TEST))
    style(a, 'Test accuracy after 300 RL updates', 'accuracy (%)', (0, 112)); a.set_xlim(-0.55, 1.55); a.set_yticks([0, 20, 40, 60, 80, 100])
    legend(a, cols, ['plain RL', 'RL + calculator', 'same, calculator off'], edges=edges, loc='upper right', ncol=1)
    us = list(range(0, 301, 50)); curves = {}
    for arm, col, lab in [(1, INK2, 'plain RL'), (2, GREEN, 'RL +\ncalculator')]:
        cs = [dev_curve(arm, s) for s in range(3)]
        m = [np.mean([c[u] for c in cs]) for u in us]; curves[lab.replace('\n', ' ')] = [round(x, 1) for x in m]
        b.plot(us, m, color=col, lw=4, marker='o', ms=11, zorder=3)
        b.text(us[-1] + 10, m[-1], lab, color=col, fontsize=LABEL, fontweight='bold', va='center', linespacing=1.0)
    b.set_xlim(-10, 410); b.set_xticks(us); b.set_xlabel('RL update')
    style(b, 'Dev accuracy during RL', 'accuracy (%)', (40, 75))
    save(fig, 'rl1_calc_after_rl')
    OUT['rl1'] = dict(test={k: [[round(x, 1) for x in v] for v in arms] for k, arms in TEST.items()}, dev_mean=curves)


# ---------------------------------------------------------------- rl2
def rl2():
    st = json.load(open(os.path.join(W, 'interp', 'stats_aR.json')))
    f0, f1, f1b = st['h_base_on']['called'], st['aR0_on']['called'], st['aR1_on']['called']
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE)
    vals = [29.8, 23.5, 16.7]; cols = [GREY, 'white', GREEN]; eds = [GREY, GREY, GREEN]
    for x, (v, c, e) in enumerate(zip(vals, cols, eds)):
        a.bar(x, v, 0.6, color=c, edgecolor=e, linewidth=3, hatch='//' if c == 'white' else None, zorder=3)
        val(a, x, v, f'{v:.1f}', strong=(x == 2), dy=0.6)
    xcats(a, ['start', 'plain RL', 'RL + calculator'])
    style(a, 'Good calculator: fewer unit words', 'equations with unit words (%)', (0, 36)); a.set_yticks([0, 10, 20, 30])
    m1 = (f1 + f1b) / 2
    b.bar(0, f0, 0.6, color=GREY, zorder=3); b.bar(1, m1, 0.6, color=PALE, edgecolor=GREEN, linewidth=3, zorder=3)
    val(b, 0, f0, f'{f0:.0f}', strong=False, dy=2); dots(b, 1, [f1, f1b], spread=0.16, ms=11); val(b, 1, max(f1, f1b), f'{m1:.0f}', dy=3)
    xcats(b, ['start', 'after RL (2 seeds)'])
    style(b, 'Broken calculator: RL stops triggering it', 'equations it fires on (%)', (0, 90))
    save(fig, 'rl2_writing')
    OUT['rl2'] = dict(unit_words=vals, fires_start=round(f0, 1), fires_after=[round(f1, 1), round(f1b, 1)])


# ---------------------------------------------------------------- rl3
OPS = ['+', '−', '×', '÷']; KEY = {'+': '+', '−': '-', '×': '*', '÷': '/'}


def rl3():
    runs = [jl(os.path.join(FDW, f'CWinOx_s{s}.jsonl')) for s in range(3)]

    def pooled(lo, hi):
        acc = {k: [0, 0] for k in OPS}
        for rows in runs:
            for r in rows:
                if 'iw_op' in r and lo <= r['u'] <= hi:
                    for k in OPS:
                        c = r['iw_op'].get(KEY[k], [0, 0]); acc[k][0] += c[0]; acc[k][1] += c[1]
        return {k: 100 * acc[k][1] / max(1, acc[k][0]) for k in OPS}
    start, end = pooled(1, 10), pooled(291, 300)
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE)
    w = 0.36
    for i, k in enumerate(OPS):
        a.bar(i - w / 2 - 0.02, start[k], w, color=PALE, edgecolor=GREEN, linewidth=2.5, zorder=3)
        a.bar(i + w / 2 + 0.02, end[k], w, color=GREEN, edgecolor=GREEN, linewidth=2.5, zorder=3)
        val(a, i - w / 2 - 0.02, start[k], f'{start[k]:.0f}', strong=False); val(a, i + w / 2 + 0.02, end[k], f'{end[k]:.0f}')
    xcats(a, [f'{k} {"broken" if k in "+−" else "exact"}' for k in OPS])
    style(a, 'RL learns where the implant is right', 'calls let through (%)', (0, 125)); a.set_yticks([0, 25, 50, 75, 100])
    legend(a, [PALE, GREEN], ['updates 1–10', 'updates 291–300'], edges=[GREEN, GREEN], loc='upper left', ncol=2)

    def dev_at(rows, key):
        for r in rows:
            if key in r and r[key].get('u') == 300:
                d = r[key]; return d[[t for t in d if t.endswith('_acc')][0]]
    cases = [('no\ncalculator', 'dev_hostonly', PALE, GREEN), ('calls\neverything', 'dev_cw_hand', GREY, GREY), ('learned\ngate', 'dev', GREEN, GREEN)]
    devs = {}
    for i, (lab, key, col, ed) in enumerate(cases):
        v = [dev_at(r, key) for r in runs]; devs[lab.replace('\n', ' ')] = v
        b.bar(i, np.mean(v), 0.6, color=col, edgecolor=ed, linewidth=2.5, zorder=3)
        dots(b, i, v, spread=0.13, ms=11); val(b, i, max(v), f'{np.mean(v):.0f}', strong=(i == 2), dy=2)
    xcats(b, [c[0] for c in cases])
    style(b, 'Calling it blindly hurts (3 seeds)', 'dev accuracy (%)', (0, 80))
    save(fig, 'rl3_competence')
    OUT['rl3'] = dict(start={k: round(v, 1) for k, v in start.items()}, end={k: round(v, 1) for k, v in end.items()}, dev_u300=devs)


# ---------------------------------------------------------------- rl4
def roll(R, tool):
    xs, ys = [], []
    for r in R:
        v = r.get(f'roll_t_{tool}')
        if 'u' in r and v and v['nb']: xs.append(r['u']); ys.append(100 * v['fire'])
    k = 15
    return xs, [np.mean(ys[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(ys))]


def rl4():
    lib, scr = jl(os.path.join(FDL, 'LIB_s0_steps.jsonl')), jl(os.path.join(FDL, 'LIBscr_s0_steps.jsonl'))
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw=dict(width_ratios=[1.3, 1]))
    for t in ('calc', 'expr_gen', 'calc_gen'):
        x, y = roll(lib, t); a.plot(x, y, color=GREEN, lw=4)
    for t in ('expr_gen', 'calc_gen'):
        x, y = roll(scr, t); a.plot(x, y, color=GREY, lw=4)
    a.text(150, 100, '3 correct arithmetic implants', color=GREEN, fontsize=LABEL, ha='center', va='bottom', fontweight='bold')
    a.text(150, 12, '2 scrambled implants', color=INK2, fontsize=LABEL, ha='center', va='bottom', fontweight='bold')
    a.set_xlim(0, 300); a.set_xlabel('RL update')
    style(a, 'RL shuts off the broken implants', 'calls let through (%)', (0, 112)); a.set_yticks([0, 25, 50, 75, 100])
    groups = [('GSM8K\nproblems', 'gsm'), ('big-number\nproblems', 'scaled')]; seeds = {}
    for s in (0, 1):
        R = jl(os.path.join(FDL, f'LIB_s{s}_steps.jsonl'))
        fin = [r['dev'] for r in R if 'dev' in r and r['dev']['u'] == 300][-1]
        off = [r['dev_toolsoff'] for r in R if 'dev_toolsoff' in r and r['dev_toolsoff'].get('u') == 300][-1]
        seeds[s] = ([fin[f'dev_{k}'] for _, k in groups], [off[f'devtoolsoff_{k}'] for _, k in groups])
    w = 0.36; cols, eds = [GREEN, PALE], [GREEN, GREEN]
    for i in range(2):
        for j in range(2):   # j 0 = on, 1 = off
            v = [seeds[s][j][i] for s in (0, 1)]; x = i + (j - 0.5) * (w + 0.03); m = np.mean(v)
            b.bar(x, m, w, color=cols[j], edgecolor=eds[j], linewidth=2.5, zorder=3)
            dots(b, x, v, spread=0.1, ms=10); val(b, x, max(v), f'{m:.0f}', strong=(j == 0), dy=1.5)
    xcats(b, [g for g, _ in groups])
    style(b, 'The implants carry the big numbers', 'dev accuracy after RL (%)', (0, 105)); b.set_xlim(-0.6, 1.6)
    b.set_yticks([0, 20, 40, 60, 80])
    legend(b, cols, ['implants on', 'implants off'], edges=eds, loc='upper right', ncol=2)
    save(fig, 'rl4_toolbox')
    OUT['rl4'] = dict(dev_u300={f's{s}': dict(on=v[0], off=v[1]) for s, v in seeds.items()})


# ---------------------------------------------------------------- rl6
def rl6():
    B = ['1-2', '3-4', '5-6', '7+']; veto = {}
    for r in jl(os.path.join(FDW, 'CW_s0.jsonl')):
        d = r.get('dev', {})
        if d.get('u') in (25, 300) and d.get('dev_veto_by_digits'):
            v = d['dev_veto_by_digits']; veto[d['u']] = [100 * v[k][1] / max(1, v[k][0]) for k in B]

    def dev_at(rows, u):
        for r in rows:
            if 'dev' in r and r['dev'].get('u') == u:
                d = r['dev']; return d[[k for k in d if k.endswith('_acc')][0]]
    fig, (L, R) = plt.subplots(1, 2, figsize=SIZE)
    w = 0.38
    for j, (u, col) in enumerate([(25, GREY), (300, GREEN)]):
        for i in range(4):
            x = i + (j - 0.5) * (w + 0.02); L.bar(x, veto[u][i], w, color=col, zorder=3)
            val(L, x, veto[u][i], f'{veto[u][i]:.0f}', strong=(j == 1), dy=1.5)
    xcats(L, [k + '\ndigits' for k in B])
    style(L, 'RL skips the calculator on small numbers', 'calls vetoed (%)', (0, 125)); L.set_yticks([0, 25, 50, 75, 100])
    legend(L, [GREY, GREEN], ['update 25', 'update 300'], loc='upper right', ncol=2)
    arms = [('REF_a2long', 'hand\nwiring', GREEN), ('CW_s0', 'learned\nwiring', GREEN), ('W3e_s0', 'new links\nonly', GREY)]
    dv = []
    for i, (t, lab, c) in enumerate(arms):
        v = dev_at(jl(os.path.join(FDW, t + '.jsonl')), 300); dv.append(v)
        R.bar(i, v, 0.6, color=c, zorder=3); val(R, i, v, f'{v:.0f}', strong=(c == GREEN), dy=0.8)
    xcats(R, [a[1] for a in arms])
    style(R, 'Learned wiring ends about level', 'dev accuracy (%)', (40, 80))
    save(fig, 'rl6_wiring')
    OUT['rl6'] = dict(veto={u: [round(x, 1) for x in v] for u, v in veto.items()}, dev_u300=dict(zip([a[0] for a in arms], dv)))


if __name__ == '__main__':
    rl1(); rl2(); rl3(); rl4(); rl6()
    json.dump(OUT, open(os.path.join(HERE, 'v3_rerender_numbers.json'), 'w'), indent=1, default=str)
    print(json.dumps(OUT, default=str))
