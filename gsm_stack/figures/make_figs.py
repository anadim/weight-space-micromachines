"""Blog figures for the RL-stack GSM8K story (task [14]); python3 gsm_stack/figures/make_figs.py  (needs matplotlib, numpy).
Inputs: $FE_OUT_DIR (default $WSM_WORK/fe, i.e. runs/gsm_stack/work/fe)/{test,hard}_*.jsonl (eval/fe.sh rows, one per item with 'ok';
not shipped: rerun the eval lists), results/curves/ARM.steps.jsonl (dev curves, shipped). Missing inputs are skipped. Bars = accuracy with 95% bootstrap CIs (several seeds: mean bar,
seed dots, CI over the pooled items). Hatched grey bars = programmatic baselines (several generation calls + majority)."""
import os, json, glob
import numpy as np
import blogstyle as S
plt = S.plt
H = os.path.dirname(os.path.abspath(__file__)); FE = os.environ.get('FE_OUT_DIR', os.path.join(os.environ.get('WSM_WORK', f'{H}/../../runs/gsm_stack/work'), 'fe')); ST = f'{H}/../results/curves'


def rows(f):
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else None


def acc_ci(oks, B=4000):
    rng = np.random.RandomState(0); d = np.asarray(oks, float); n = len(d)
    m = d[rng.randint(0, n, (B, n))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def bars(name, title, spec, ylim, note='', prog_label='programmatic baseline (N separate calls + majority)'):
    """spec: list of (label, [files], style) with style 'stack' (green), 'host' (pale), 'prog' (hatched grey)."""
    got = []
    for lab, fs, sty in spec:
        R = [r for r in (rows(f'{FE}/{f}') for f in fs) if r]
        if not R: continue
        a, lo, hi = acc_ci([x['ok'] for r in R for x in r])
        seeds = [100 * np.mean([x['ok'] for x in r]) for r in R]
        tok = np.mean([x.get('ntok', np.nan) for r in R for x in r])
        got.append((lab, a, lo, hi, seeds, sty, tok))
    if not got: print('skip', name); return
    fig, ax = plt.subplots(figsize=(max(S.SIZE[0], 1.35 * len(got) + 1.5), S.SIZE[1]))   # wide ladders: keep labels apart
    for i, (lab, a, lo, hi, seeds, sty, tok) in enumerate(got):
        kw = dict(color=S.GREEN, edgecolor=S.GREEN) if sty == 'stack' else dict(color=S.PALE, edgecolor=S.MID) if sty == 'host' \
            else dict(color=S.INK2, edgecolor=S.INK2) if sty == 'rl' else dict(color='white', edgecolor=S.GREY, hatch='//')
        ax.bar(i, a, width=0.7, lw=2, **kw)
        ax.errorbar(i, a, yerr=[[a - lo], [hi - a]], color=S.INK, capsize=6, lw=2)
        if len(seeds) > 1: ax.scatter([i] * len(seeds), seeds, color=S.INK, s=28, zorder=5)
        ax.text(i, hi + 1.0, f'{a:.1f}', ha='center', va='bottom', fontsize=S.VAL, fontweight='bold', color=S.INK)
    ax.set_xticks(range(len(got)))
    ax.set_xticklabels([g[0] + (f'\n{g[6]:.0f} tok/q' if np.isfinite(g[6]) else '') for g in got], fontsize=13)
    S.style(ax, title, 'accuracy (%)', ylim)
    from matplotlib.patches import Patch
    used = {g[5] for g in got}
    hs = [(st, h) for st, h in (('host', Patch(color=S.PALE, ec=S.MID, lw=2, label='host / implants, no RL')),
                                ('rl', Patch(color=S.INK2, label='RL without implants')),
                                ('stack', Patch(color=S.GREEN, label='RL with the implants attached (one generation call)')),
                                ('prog', Patch(facecolor='white', ec=S.GREY, hatch='//', lw=2, label=prog_label))) if st in used]
    ax.legend(handles=[h for _, h in hs], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(hs), frameon=False, fontsize=12)
    if note: fig.text(0.01, 0.01, note, fontsize=11, color=S.INK2)
    S.save(fig, name)


def steps(arm):
    f = f'{ST}/{arm}.steps.jsonl'
    if not os.path.exists(f): return None
    d = [json.loads(l)['dev'] for l in open(f) if '"dev"' in l]
    return [x for x in d if isinstance(x, dict)]


def curves():
    panels = [('From the implant-trained host u0', [
                  ('A1: serial N=8, attempt credit', ['A1', 'A1_s2', 'A1_s3', 'A1_s4'], 'dev_vote8', S.GREEN),
                  ('A1 + diversity term', ['A1D_s0', 'A1D_s1', 'A1D_s2'], 'dev_vote8', S.MID),
                  ('A: serial N=8, final-vote reward', ['A', 'A_s2', 'A_s3'], 'dev_vote8', S.INK2),
                  ('B: one attempt (S2F)', ['B', 'B_s2', 'B_s3'], 'dev_acc', S.GREY)], (76, 90)),
              ('From the base host (Qwen3-0.6B-Base)', [
                  ('C: serial N=4 stack', ['C', 'C_s2', 'C_s3'], 'dev_vote4', S.GREEN),
                  ('B_base: one attempt (S2F stack)', ['Bb_s5', 'Bb_s6', 'Bb_s7'], 'dev_acc', S.MID),
                  ('plain RL, no implants', ['Pb_s1', 'Pb_s2'], 'dev_acc', S.INK2)], (30, 90))]
    fig, axs = plt.subplots(1, 2, figsize=S.SIZE)
    for ax, (title, arms, ylim) in zip(axs, panels):
        for lab, names, key, col in arms:
            ser = [s for s in (steps(n) for n in names) if s]
            ser = [[(x['u'], x[key]) for x in s if key in x] for s in ser]; ser = [s for s in ser if s]
            if not ser: continue
            for s in ser: ax.plot(*zip(*s), color=col, lw=1.2, alpha=0.45)
            umax = max(s[-1][0] for s in ser); full = [s for s in ser if s[-1][0] == umax]   # thick: mean of the furthest seeds
            us = sorted(set.intersection(*[set(u for u, _ in s) for s in full]))
            m = [np.mean([dict(s)[u] for s in full]) for u in us]
            ax.plot(us, m, color=col, lw=3.5, label=f'{lab} ({len(full)}/{len(ser)} seed{"s" if len(ser) > 1 else ""} at u{umax})')
            ax.text(us[-1] + 4, m[-1], f'{m[-1]:.1f}', color=col, fontsize=14, fontweight='bold', va='center')
        S.style(ax, title, 'dev accuracy (%)', ylim); ax.set_xlabel('RL update'); ax.set_xlim(0, 330)
        ax.legend(loc='lower right', frameon=False, fontsize=12)
    S.save(fig, 'rl_dev_curves')


if __name__ == '__main__':
    curves()
    note = 'GSM8K test (1319), T = 1, seed 7, each config run once; 95% bootstrap CIs. Weight-space implants only.'
    bars('test_bars', 'GSM8K test: implants, then RL with the implants attached', [
        ('host only', ['test_lad_H.jsonl'], 'host'), ('+ calculator', ['test_lad_C4.jsonl'], 'host'),
        ('+ chain\ncalc', ['test_lad_D.jsonl'], 'host'), ('+ sanity\n& forget', ['test_S2Fw_u0.jsonl'], 'host'),
        ('+ serial\nN=8', ['test_u0_N8.jsonl'], 'host'),
        ('+ position\nreset (WPR)', ['test_r1_u0_N8.jsonl'], 'host'),
        ('+ adaptive\ntries (rung 2)', ['test_r2_u0.jsonl'], 'host'),
        ('+ learned\nvoter (rung 3)', ['test_r3_u0.jsonl'], 'host'),
        ('RL B (S2F)', ['test_B_u300w.jsonl', 'test_B_s2_u300w.jsonl', 'test_B_s3_u300w.jsonl'], 'stack'), ('RL C N=4\n(from base)', ['test_C_N4.jsonl', 'test_C_s2_N4.jsonl', 'test_C_s3_N4.jsonl'], 'stack'),
        ('RL C + WPR\nN=4 (from base)', [f'test_CW_s{k}_N4.jsonl' for k in (0, 1, 2)], 'stack'),
        ('RL A1 N=8', ['test_A1_N8.jsonl', 'test_A1_s2_N8.jsonl', 'test_A1_s3_N8.jsonl'], 'stack'),
        ('SC ×8, u0', ['test_sc8_u0.jsonl'], 'prog'), ('SC ×8, A1', ['test_sc8_A1.jsonl'], 'prog')], (50, 90), note)
    bars('hard_bars', 'GSM-Hard: implants, then RL with the implants attached', [
        ('host only', ['hard_lad_H.jsonl'], 'host'), ('+ calculator', ['hard_lad_C4.jsonl'], 'host'),
        ('+ chain\ncalc', ['hard_lad_D.jsonl'], 'host'), ('+ forget', ['hard_S2Fw_u0.jsonl'], 'host'),
        ('+ serial\nN=8', ['hard_u0_N8.jsonl'], 'host'),
        ('+ position\nreset (WPR)', ['hard_r1_u0_N8.jsonl'], 'host'),
        ('+ adaptive\ntries (rung 2)', ['hard_r2_u0.jsonl'], 'host'),
        ('+ learned\nvoter (rung 3)', ['hard_r3_u0.jsonl'], 'host'),
        ('RL B (S2F)', ['hard_B_u300w.jsonl', 'hard_B_s2_u300w.jsonl', 'hard_B_s3_u300w.jsonl'], 'stack'), ('RL C N=4\n(from base)', ['hard_C_N4.jsonl', 'hard_C_s2_N4.jsonl', 'hard_C_s3_N4.jsonl'], 'stack'),
        ('RL C + WPR\nN=4 (from base)', [f'hard_CW_s{k}_N4.jsonl' for k in (0, 1, 2)], 'stack'),
        ('RL A1 N=8', ['hard_A1_N8.jsonl', 'hard_A1_s2_N8.jsonl', 'hard_A1_s3_N8.jsonl'], 'stack'),
        ('SC ×8, u0', ['hard_sc8_u0.jsonl'], 'prog'), ('SC ×8, A1', ['hard_sc8_A1.jsonl'], 'prog')], (30, 70),
        note.replace('GSM8K test', 'GSM-Hard') + ' Serial: plain majority, no sanity filter.')
    for SS in ('test', 'hard'):
        bars('base_split' + ('' if SS == 'test' else '_hard'),
             ('GSM8K test' if SS == 'test' else 'GSM-Hard') + ' from the base host: RL alone vs RL + tools vs RL + serial stack', [
            ('base host,\nno RL', [f'{SS}_baseH.jsonl'], 'host'),
            ('base + stack N=4,\nno RL', [f'{SS}_base_N4.jsonl'], 'host'),
            ('plain RL\n(no implants)', [f'{SS}_Pb_s1_u300.jsonl', f'{SS}_Pb_s2_u300.jsonl'], 'rl'),
            ('RL + one-attempt\nstack (B_base)', [f'{SS}_Bb_s{k}_u300w.jsonl' for k in (5, 6, 7)], 'stack'),
            ('RL + serial\nstack N=4 (C)', [f'{SS}_C_N4.jsonl', f'{SS}_C_s2_N4.jsonl', f'{SS}_C_s3_N4.jsonl'], 'stack'),
            ('RL + serial N=4\n+ WPR (CW)', [f'{SS}_CW_s{k}_N4.jsonl' for k in (0, 1, 2)], 'stack'),
            ('CW + adaptive\ntries (rung 2)', [f'{SS}_r2_CW_s{k}.jsonl' for k in (0, 1, 2)], 'stack')], (0, 100),
            note if SS == 'test' else note.replace('GSM8K test', 'GSM-Hard'))
    for SS, lim in (('test', (50, 90)), ('hard', (30, 70))):
        bars(f'wpr_{SS}', ('GSM8K test' if SS == 'test' else 'GSM-Hard') + ': weight-space position reset vs RL (N=8)', [
            ('serial N=8,\nno reset', [f'{SS}_u0_N8.jsonl'], 'host'), ('+ WPR\n(all weights)', [f'{SS}_r1_u0_N8.jsonl'], 'host'),
            ('+ adaptive\ntries (rung 2)', [f'{SS}_r2_u0.jsonl'], 'host'),
            ('+ Nmax 24,\nno voter', [f'{SS}_r2n24_u0.jsonl'], 'host'),
            ('+ learned\nvoter (rung 3)', [f'{SS}_r3_u0.jsonl'], 'host'),
            ('programmatic\nreset (ref.)', [f'{SS}_u0_N8pr1.jsonl'], 'prog'),
            ('RL A1,\nno reset', [f'{SS}_A1_N8.jsonl', f'{SS}_A1_s2_N8.jsonl', f'{SS}_A1_s3_N8.jsonl'], 'stack'), ('RL A1\n+ WPR', [f'{SS}_A1_N8wpr.jsonl'], 'stack'),
            ('RL A1 with\nWPR in RL (A1W)', [f'{SS}_A1W_s{k}_N8.jsonl' for k in (0, 1, 2)], 'stack'),
            ('A1W + adaptive\ntries (rung 2)', [f'{SS}_r2_A1W_s{k}.jsonl' for k in (0, 1, 2)], 'stack')], lim,
            note.replace('GSM8K test', 'GSM8K test' if SS == 'test' else 'GSM-Hard') + ' Programmatic reset: non-compliant reference.',
            prog_label='programmatic baseline (position reset done outside the weights)')
    D = lambda c: [f'dev_{c}_s{k}.jsonl' for k in (7, 11, 13)]
    bars('controls_dev', 'One call with N tries vs N separate calls, before and after RL (dev, 3 seeds)', [
        ('u0: serial\nN=8, one call', D('ser8'), 'host'), ('u0: SC ×8\n(8 calls)', D('sc8'), 'prog'),
        ('u0: serial N=8,\nforget off', D('fo8'), 'host'),
        ('A1: serial\nN=8, one call', D('ser8A1'), 'stack'), ('A1: SC ×8\n(8 calls)', D('sc8A1'), 'prog'),
        ('A1: serial N=8,\nforget off', D('fo8A1'), 'stack')], (70, 90),
        'GSM8K dev (600 train items held out), T = 1, sampling seeds 7 / 11 / 13 (dots); 95% bootstrap CIs over the pooled items. SC: plain majority.')
    bars('ablation_A1', 'RL A1 (N=8) at test: turn off one weight-space part', [
        ('full', ['test_A1_N8.jsonl'], 'stack'), ('calculator off', ['test_abA1_nocalc.jsonl'], 'stack'),
        ('chain calc off', ['test_abA1_nochain.jsonl'], 'stack'), ('both tools off', ['test_abA1_notools.jsonl'], 'stack'),
        ('compiled voter\n→ majority', ['test_abA1_vote0.jsonl'], 'stack'), ('forget off', ['test_abA1_nofgt.jsonl'], 'stack'),
        ('SC ×8, A1', ['test_sc8_A1.jsonl'], 'prog')], (50, 90), note + ' Same RL checkpoint, no retraining.')
    bars('ablation_hard', 'GSM-Hard: the in-weight calculator still carries the RL host', [
        ('A1 full', ['hard_A1_N8.jsonl'], 'stack'), ('A1 calculator off', ['hard_abhA1_nocalc.jsonl'], 'stack'),
        ('A1 chain calc off', ['hard_abhA1_nochain.jsonl'], 'stack'), ('A1 both tools off', ['hard_abhA1_notools.jsonl'], 'stack'),
        ('B full', ['hard_B_u300.jsonl'], 'stack'), ('B both tools off', ['hard_abhB_notools.jsonl'], 'stack')], (30, 70),
        note.replace('GSM8K test', 'GSM-Hard') + ' Same RL checkpoints, no retraining.')
