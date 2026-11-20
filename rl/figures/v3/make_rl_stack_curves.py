"""RL with the all-weights implant stack attached (gsm_push RL_STACK study): dev curves, per-attempt accuracy, training reward.
Data: data/rlstack/<arm>_steps.jsonl (copied from the pods' final/rl/<arm>/steps.jsonl). Style: blogstyle.py."""
import json, os, numpy as np
import blogstyle as bs
import matplotlib.pyplot as plt
D = os.path.join(bs.HERE, 'data', 'rlstack')
ARMS = [('A', 'A: implant stack, N = 4, vote reward', bs.GREEN, '-'),
        ('A1', "A': A + per-attempt credit", bs.MID, '--'),
        ('Areset', 'A-reset: A + programmatic position reset (16 prompts)', bs.INK2, ':'),
        ('B', 'B: S2F implants, one attempt', bs.GREY, '-'),
        ('C', 'C: implant stack on the base host (16 prompts)', '#c27c0e', '-')]
from matplotlib.lines import Line2D


def load(a):
    f = os.path.join(D, f'{a}_steps.jsonl')
    if not os.path.exists(f): return None, None
    L = [json.loads(l) for l in open(f)]
    dev = [x['dev'] for x in L if isinstance(x.get('dev'), dict)]
    rw = [x for x in L if 'reward' in x and 'u' in x]
    return dev, rw


def comb(d, k):
    """dev600 + dev2 pooled (900 items)."""
    if f'dev_{k}' not in d: return None
    return (600 * d[f'dev_{k}'] + 300 * d[f'dev2_{k}']) / 900


def sm(y, w=9):
    y = np.array(y, float); p = np.pad(y, (w // 2, w // 2), mode='edge'); return np.convolve(p, np.ones(w) / w, 'valid')


dat = {a: load(a) for a, *_ in ARMS}
# ---- figure 1: dev curves
panels = [('vote4', 'Voted accuracy, N = 4', (20, 100)), ('vote8', 'Voted accuracy, N = 8 (H100 arms)', (20, 100)),
          ('allsame4', 'All-same rate, N = 4', (0, 100)), ('oracle4', 'Any attempt correct, N = 4', (20, 100))]
fig, ax = plt.subplots(1, 4, figsize=(26, 7))
for a_, (k, t, yl) in zip(ax, panels):
    for arm, lab, col, ls in ARMS:
        dev = dat[arm][0]
        if not dev: continue
        if arm == 'B':
            if k not in ('vote4', 'vote8'): continue
            xs = [d['u'] for d in dev]; ys = [(600 * d['dev_acc'] + 300 * d['dev2_acc']) / 900 for d in dev]
        else:
            pts = [(d['u'], comb(d, k)) for d in dev if comb(d, k) is not None]
            if not pts: continue
            xs, ys = zip(*pts)
        a_.plot(xs, ys, ls, color=col, marker='o', ms=7, label=lab)
    bs.style(a_, t, 'dev600 + dev2 (%)', yl); a_.set_xlabel('RL update')
h = [Line2D([], [], ls=ls, color=col, marker='o', ms=7, lw=2.5) for arm, lab, col, ls in ARMS if dat[arm][0]]
l = [lab for arm, lab, col, ls in ARMS if dat[arm][0]]
fig.legend(h, l, loc='lower center', ncol=3, fontsize=bs.LG, frameon=False, bbox_to_anchor=(.5, 0.0))
fig.tight_layout(rect=(0, .17, 1, 1), w_pad=4); bs.save(fig, 'rl10_stack_dev', tight=False)
# ---- figure 2: per-attempt accuracy, first vs last dev point
fig, ax = plt.subplots(1, 3, figsize=(bs.SIZE[0] * 1.5, bs.SIZE[1]))
for a_, arm in zip(ax, ('A', 'A1', 'Areset')):
    dev = dat[arm][0]
    if not dev: continue
    for d, col, lab in ((dev[0], bs.GREY, f"update {dev[0]['u']}"), (dev[-1], bs.GREEN, f"update {dev[-1]['u']}"))[:1 + (len(dev) > 1)]:
        y = [(600 * p + 300 * q) / 900 for p, q in zip(d['dev_attempt_acc'], d['dev2_attempt_acc'])]
        a_.plot(range(1, len(y) + 1), y, '-', color=col, marker='o', ms=9, label=lab)
    bs.style(a_, dict(A='A: accuracy of each attempt', A1="A': accuracy of each attempt", Areset='A-reset (N = 4 only)')[arm], 'dev600 + dev2 (%)', (0, 100))
    a_.set_xlabel('attempt in the stream'); a_.set_xlim(0.5, 8.5); a_.legend(frameon=False, fontsize=bs.LG, loc='lower left')
bs.save(fig, 'rl11_stack_attempts')
# ---- figure 3: training reward
fig, ax = plt.subplots(1, 1, figsize=bs.SIZE)
for arm, lab, col, ls in ARMS:
    rw = dat[arm][1]
    if not rw: continue
    ax.plot([x['u'] for x in rw], sm([x['reward'] for x in rw]), ls, color=col, label=lab)
bs.style(ax, 'Training reward (final answer correct, 9-update mean)', 'reward', (0.0, 1.0)); ax.set_xlabel('RL update')
ax.legend(frameon=False, fontsize=bs.LG - 4, loc='upper left', bbox_to_anchor=(1.01, 1.0))
bs.save(fig, 'rl12_stack_reward')
json.dump({a: dict(dev=dat[a][0], last_reward=(dat[a][1][-1]['reward'] if dat[a][1] else None)) for a, *_ in ARMS},
          open(os.path.join(bs.HERE, 'rlstack_numbers.json'), 'w'), indent=1)
