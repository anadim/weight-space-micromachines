"""Dev curves (408 look-alike dev items, greedy) + training stats for the rl24/bbh runs. Local, stdlib only.
python3 devcurves.py [RUN ...]   (default: P0_s0..2 I0_s0..2). Keeps the last record per update (resumed attempts)."""
import json, os, sys
runs = sys.argv[1:] or [f'{a}_s{s}' for a in ('P0', 'I0') for s in (0, 1, 2)]


def last_per_u(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            try: r = json.loads(line)
            except ValueError: continue
            d[r['u']] = r
    return d


for run in runs:
    dev, lg = last_per_u(f'out/{run}/dev.jsonl'), last_per_u(f'out/{run}/log.jsonl')
    if not dev and not lg: print(run, '(no logs yet)'); continue
    print(f'{run}: dev acc ' + ' '.join(f'u{u}:{dev[u]["acc"]:.1f}' for u in sorted(dev)))
    if lg:
        us = sorted(lg); n = len(us)
        def m(k, sel): v = [lg[u][k] for u in sel if k in lg[u]]; return sum(v) / max(1, len(v))
        first, last = us[:25], us[-25:]
        print(f'   updates {n} (last u{us[-1]}); reward first25 {m("reward", first):.3f} last25 {m("reward", last):.3f}; '
              f's/update {m("total_s", us):.1f} (rollout {m("rollout_s", us):.1f}, train {m("train_s", us):.1f}); '
              f'tok/s {m("tok_s", us):.0f}; tok/q {m("tok_q", us):.0f}; ratio {m("ratio", us):.4f}')
        if 'u1_tf' in lg.get(1, {}): print('   u1 TF', lg[1]['u1_tf'], 'wflag mismatch', lg[1].get('u1_wflag_mismatch'))
