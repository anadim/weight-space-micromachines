"""v3 RL blog figures (2026-11-20): rl5_mixed_v3, rl7_better_tool, rl8_train_implant. Plotting only; data in ./data.

rl5  : mixed-task test (400 stratified items) after 150 updates. Implants arms m1_mixLIB s0 + s1 (test_eval.json:
       'full' = implants on, 'toolsoff' = implants off, same host and wiring); plain RL = m3_mixPlain64 (1 seed).
       Bars = mean of the 2 seeds, dots = each seed. The old right panel ("wrong calls", m4 noisy routing) is dropped:
       its IFEval curve is not monotone (0.22 -> 0.01 -> 0.15 -> 0.28 -> 0.23) and seeds 1/2 stop at u50.
rl7  : share of GSM rollout answers that contain <calc>, per RL update, from rollouts.jsonl (computed on the pods with
       calc_share_remote.py, same rule as rl_library/causal_analyze.py) -> data/calc_share_*.jsonl.
       Lines = mean over seeds of a 5-update rolling mean; band = min-max over seeds.
rl8  : left = rung C module exactness (10k random calls) logged at u0/25/50 (raw Adam, sweep v2 swC2_*) and u0/5/10
       (snapped SGD, swSGD_*). Snapped Adam (L300cf_Csnap_*) logs exactness only at u0 -> not plotted.
       right = arm C per-implant exactness on the BBH drift streams (impl_drift in steps.jsonl): gbC2_s0, gbC3_s0/s1/s2.
"""
import json, os, glob
import numpy as np
from blogstyle import *

D = os.path.join(HERE, 'data')
FDL = os.path.join(HERE, '..', '..', 'results', 'library', 'figdata')   # shared run logs (m1 s0 / m4 s0)
_SHARED = {'m1_mixLIB_steps.jsonl': 'mix_m1_mixLIB_steps.jsonl', 'm4_s0_steps.jsonl': 'mix_m4_mixLIBnoisy_steps.jsonl',
           'm1_mixLIB_test_eval.json': 'mix_m1_mixLIB_test.json'}
def dpath(n): return os.path.join(FDL, _SHARED[n]) if n in _SHARED else os.path.join(D, n)
OUT = {}


# ---------------------------------------------------------------- rl5
def rl5():
    T = ['gsm', 'bbh', 'ifeval', 'combo']; TL = ['GSM8K', 'BBH', 'IFEval', 'GSM + IFEval\ncombo']
    s = [json.load(open(dpath(f'm1_mixLIB{x}_test_eval.json'))) for x in ('', '_s1')]
    pl = json.load(open(os.path.join(D, 'm3_mixPlain64_test_eval.json')))['full']
    plain = [pl[f'test_{t}'] for t in T]
    off = [[r['toolsoff'][f'test_{t}'] for r in s] for t in T]
    on = [[r['full'][f'test_{t}'] for r in s] for t in T]
    fig, ax = plt.subplots(figsize=SIZE)
    w = 0.26; cols = [GREY, PALE, GREEN]; edges = [GREY, GREEN, GREEN]
    for g in range(4):
        for k, (v, seeds) in enumerate([(plain[g], None), (np.mean(off[g]), off[g]), (np.mean(on[g]), on[g])]):
            x = g + (k - 1) * (w + 0.03)
            ax.bar(x, v, w, color=cols[k], edgecolor=edges[k], linewidth=2.5, zorder=3)
            top = v if seeds is None else max(max(seeds), v)
            if seeds is not None: dots(ax, x, seeds)
            val(ax, x, top, f'{v:.0f}', strong=(k == 2), dy=1.5)
    xcats(ax, TL)
    style(ax, 'Implants on win on every task type', 'test accuracy (%)', (0, 112))
    ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_xlim(-0.6, 3.6)
    lg = legend(ax, cols, ['plain RL (1 seed)', 'implants off', 'implants on'], edges=edges, loc='upper left', ncol=4)
    lg.legend_handles.append(Line2D([], [], marker='o', ls='', color=INK, ms=10, mec='white'))
    ax.legend(handles=lg.legend_handles, labels=['plain RL (1 seed)', 'implants off', 'implants on', 'each seed (2)'],
              loc='upper left', fontsize=LG, frameon=False, handlelength=1.4, handleheight=1.0, ncol=4, columnspacing=1.4)
    save(fig, 'rl5_mixed_v3')
    OUT['rl5'] = dict(plain=dict(zip(T, [round(v, 1) for v in plain])),
                      off={t: [round(v, 1) for v in off[i]] for i, t in enumerate(T)},
                      on={t: [round(v, 1) for v in on[i]] for i, t in enumerate(T)})


# ---------------------------------------------------------------- rl7
def calc_share():
    S = {}
    for f in glob.glob(os.path.join(D, 'calc_share_*.jsonl')):
        for l in open(f):
            d = json.loads(l); S[d['run']] = {int(u): 100 * v[1] / v[0] for u, v in d['u'].items()}
    return S


def rl7():
    S = calc_share()
    conds = [('exact ‘=’ implant (control)', ['causal_i_control', 'causal_i_control_s1'], GREEN, '-'),
             ('no ‘=’ implants', ['causal_ii_noPassive', 'causal_ii_noPassive_s1'], INK, '--'),
             ('‘=’ implant 25% corrupted', ['compete_c9_s0', 'compete_c9_s1', 'compete_c9_s2'], GREY, '-'),
             ('‘=’ implant 55% corrupted', ['compete_c1_s0', 'compete_c1_s1', 'compete_c1_s2'], '#8b919b', '-'),
             ('‘=’ implant 87% corrupted', ['compete_c3_s0', 'compete_c3_s1', 'compete_c3_s2'], INK2, '-')]
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={'width_ratios': [1.55, 1]})
    us = np.arange(1, 51); ends = []
    for lab, runs, col, ls in conds:
        cur = []
        for r in runs:
            y = np.array([S[r][u] for u in us]); k = 5
            cur.append(np.array([y[max(0, i - k // 2): i + k // 2 + 1].mean() for i in range(len(y))]))
        cur = np.array(cur); m = cur.mean(0)
        a.fill_between(us, cur.min(0), cur.max(0), color=col, alpha=0.18, lw=0, zorder=2)
        a.plot(us, m, color=col, lw=4 if col == GREEN else 3.2, ls=ls, zorder=4 if col == GREEN else 3, label=lab)
        ends.append([np.mean([S[r][u] for u in range(41, 51)]) for r in runs])
    a.set_xlim(1, 50); a.set_xticks([1, 10, 20, 30, 40, 50]); a.set_xlabel('RL update')
    style(a, 'RL drops <calc> only when a better implant exists', 'GSM answers using <calc> (%)', (0, 105))
    a.set_yticks([0, 25, 50, 75, 100])
    a.legend(loc='lower left', fontsize=18, frameon=False, handlelength=1.8, ncol=1, borderaxespad=0.2)
    # right: end share, updates 41-50, one dot per seed
    short = ['exact', 'none', '25%', '55%', '87%']
    for i, ((lab, runs, col, ls), e) in enumerate(zip(conds, ends)):
        m = np.mean(e)
        b.bar(i, m, 0.62, color=col, zorder=3, edgecolor=col, linewidth=2, hatch='//' if ls == '--' else None,
              **({'facecolor': 'white'} if ls == '--' else {}))
        dots(b, i, e, spread=0.14)
        val(b, i, max(e), f'{m:.0f}', strong=(col == GREEN), dy=2)
    b.set_xticks(range(5)); b.set_xticklabels(short, fontsize=TICK, fontweight='bold'); b.tick_params(axis='x', length=0, pad=10)
    b.set_xlabel('the ‘=’ implant', labelpad=6)
    style(b, 'Share at updates 41–50', '', (0, 112)); b.set_yticks([0, 25, 50, 75, 100]); b.set_xlim(-0.6, 4.6)
    save(fig, 'rl7_better_tool')
    OUT['rl7'] = {c[0]: dict(start_u1_5=[round(np.mean([S[r][u] for u in range(1, 6)]), 1) for r in c[1]],
                             end_u41_50=[round(v, 1) for v in e]) for c, e in zip(conds, ends)}


# ---------------------------------------------------------------- rl8
def me_series(name):
    out = {}
    for l in open(os.path.join(D, 'cf', name + '.jsonl')):
        r = json.loads(l)
        if 'module_exact' in r: out[r['module_exact']['u']] = r['module_exact']['exact']
    return dict(sorted(out.items()))


def rl8():
    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={'width_ratios': [1, 1.25]})
    a.plot([0, 50], [100, 100], color=GREEN, lw=5, zorder=4, solid_capstyle='butt')
    a.text(25, 103.5, 'frozen calculator', color=GREEN, fontsize=LABEL, fontweight='bold', ha='center', va='bottom')
    raw = {n: me_series(n) for n in ['swC2_core1e2', 'swC2_core1e3', 'swC2_core1e4', 'swC2_core1e3_rd1e4']}
    sgd = {n: me_series(n) for n in ['swSGD_1e-3', 'swSGD_1e-4', 'swSGD_1e-5']}
    for d, col in [(raw, GREY), (sgd, INK2)]:
        for s in d.values():
            a.plot(list(s), list(s.values()), color=col, lw=2.5, ls=(0, (1.5, 2.5)), zorder=3)   # dotted: not measured in between
            a.plot(list(s), list(s.values()), 'o', color=col, ms=13, mec='white', mew=1.5, zorder=5)
    a.text(27, 7, 'Adam, 4 learning rates', color=INK2, fontsize=18, fontweight='bold', ha='left', va='bottom')
    a.text(6.5, 30, 'SGD,\n3 learning\nrates', color=INK, fontsize=18, fontweight='bold', ha='left', va='bottom', linespacing=1.0)
    a.set_xlim(-2, 52); a.set_xticks([0, 5, 10, 25, 50]); a.set_xlabel('RL update')
    style(a, 'The trained calculator breaks', 'exact on random calls (%)', (-6, 115)); a.set_yticks([0, 25, 50, 75, 100])
    # right: arm C per-implant exactness
    runs = ['gbC2_s0', 'gbC3_s0', 'gbC3_s1', 'gbC3_s2']; imp = ['interval_gen', 'track_gen', 'truth_gen', 'turtle_gen', 'order_gen']
    lab = ['interval', 'track', 'truth', 'turtle', 'order']
    E = {}
    for r in runs:
        for l in open(dpath(r + '_steps.jsonl')):
            x = json.loads(l)
            if 'impl_drift' in x: E[(r, x['u'])] = {k: 100 * v['exact'] for k, v in x['impl_drift'].items()}
    cols = [PALE, MID, GREEN]; w = 0.27; tab = {}
    for g, k in enumerate(imp):
        for j, u in enumerate([0, 25, 50]):
            v = [E[(r, u)][k] for r in runs]; m = np.mean(v); x = g + (j - 1) * (w + 0.03); tab[(lab[g], u)] = [round(t, 1) for t in v]
            b.bar(x, m, w, color=cols[j], edgecolor=GREEN, linewidth=2, zorder=3)
            if u: dots(b, x, v, spread=0.045, ms=8)
            if g == 0 and u: val(b, x, max(v), f'{m:.0f}', strong=(u == 50), dy=3.5)
    xcats(b, lab)
    style(b, 'RL erodes the BBH implants it trains', 'same output as the original (%)', (0, 125))
    b.set_yticks([0, 25, 50, 75, 100]); b.set_xlim(-0.6, 4.6)
    legend(b, cols, ['start', 'update 25', 'update 50'], edges=[GREEN] * 3, loc='upper right', ncol=3)
    save(fig, 'rl8_train_implant')
    OUT['rl8'] = dict(raw_adam={k: v for k, v in raw.items()}, sgd_snapped={k: v for k, v in sgd.items()},
                      armC_exact={f'{a_}@u{u}': v for (a_, u), v in tab.items()})


if __name__ == '__main__':
    rl5(); rl7(); rl8()
    json.dump(OUT, open(os.path.join(HERE, 'v3_new_numbers.json'), 'w'), indent=1, default=str)
    print(json.dumps(OUT, indent=0, default=str)[:4000])
