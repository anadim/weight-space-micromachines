"""Off-task analysis: python3 ana_ot.py [DIR ...] -> RESULTS tables (stdout, markdown) + rows06.json / rows17.json.
  OT_SIZE=06 (default, Qwen3-0.6B-Base) | 17 (Qwen3-1.7B-Base). DIR default: j06/res (OT_SIZE=17: j17/res), the shipped
  reduced per-item rows <ROW>.items.jsonl.gz (mk_items.py). A fresh ot_hook.py output dir (<ROW>_<task>.jsonl, e.g.
  $WSM_WORK/offtask/j06/ot) is read directly as well. OT_ROWS overrides the JSON output path.
Rows: 1 Base | 3 Base + rung 2 | 2 P64 s0-2 | 2b P64 + rung 2 | 4 CW s0-2 + rung 2 | 4h CW host only.
Per item the job records host-only ('_h') and implants-attached ('_a') scores from the same forward pass."""
import json, glob, gzip, os, random, sys

SIZE = os.environ.get('OT_SIZE', '06')
HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = sys.argv[1:] or [f'{HERE}/j{SIZE}/res']
if SIZE == '06':
    HOSTS = dict(base=['base'], P64=['P64_s0', 'P64_s1', 'P64_s2'], CW=['CW_s0', 'CW_s1', 'CW_s2'])
    ROWS = [('1', 'Base (model alone)', 'base', 'h'), ('3', 'Base + rung 2', 'base', 'a'),
            ('2', 'P64 s0/1/2 (plain RL)', 'P64', 'h'), ('2b', 'P64 + untrained rung 2 attached', 'P64', 'a'),
            ('4', 'CW s0/1/2 + rung 2 (RL with implants)', 'CW', 'a'), ('4h', 'CW host alone (implants detached)', 'CW', 'h')]
else:
    HOSTS = dict(base=['base17'], P64=['P17_s0', 'P17_s1', 'P17_s2'], CW=['CW17_s0', 'CW17_s1', 'CW17_s2'])
    ROWS = [('1', '1.7B-Base (model alone)', 'base', 'h'), ('3', '1.7B-Base + rung 2', 'base', 'a'),
            ('2', 'P17 s0/1/2 (plain RL)', 'P64', 'h'), ('2b', 'P17 + untrained rung 2 attached', 'P64', 'a'),
            ('4', 'CW17 s0/1/2 + rung 2 (RL with implants)', 'CW', 'a'), ('4h', 'CW17 host alone', 'CW', 'h')]
BENCH = [('mmlu', 'MMLU 5-shot', 'ok'), ('arcc', 'ARC-C acc_norm', 'okn'), ('hs', 'HellaSwag acc_norm', 'okn'),
         ('wg', 'WinoGrande', 'ok'), ('nq', 'NQ-open EM (gen.)', 'ok')]


_RED = {}


def load(tag, task):
    for d in DIRS:
        p = f'{d}/{tag}_{task}.jsonl'
        if os.path.exists(p): return [json.loads(l) for l in open(p)]
        p = f'{d}/{tag}.items.jsonl.gz'
        if os.path.exists(p):
            if p not in _RED:
                _RED[p] = {}
                for l in gzip.open(p, 'rt'):
                    x = json.loads(l); _RED[p].setdefault(x['task'], []).append(x)
            if task in _RED[p]: return _RED[p][task]
    return None


def boot1(v, nb=2000, seed=1):
    rng = random.Random(seed); n = len(v); ms = sorted(100 * sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(nb))
    return ms[int(.025 * nb)], ms[int(.975 * nb) - 1]


def hboot(A, B, nb=2000, seed=2):
    """two-level bootstrap of mean(A) - mean(B) (calcimp/ana38.py): resample train seeds per arm, then items (shared)."""
    rng = random.Random(seed); n = len(A[0]); ms = []
    for _ in range(nb):
        sa = [A[rng.randrange(len(A))] for _ in A]; sb = [B[rng.randrange(len(B))] for _ in B]
        ma = [sum(c) / len(c) for c in zip(*sa)]; mb = [sum(c) / len(c) for c in zip(*sb)]
        idx = [rng.randrange(n) for _ in range(n)]
        ms.append(100 * sum(ma[i] - mb[i] for i in idx) / n)
    ms.sort(); return ms[int(0.025 * nb)], ms[int(0.975 * nb) - 1]


def hboot1(A, nb=2000, seed=3):
    """CI of the seed-mean accuracy: resample seeds, then items."""
    rng = random.Random(seed); n = len(A[0]); ms = []
    for _ in range(nb):
        sa = [A[rng.randrange(len(A))] for _ in A]; m = [sum(c) / len(c) for c in zip(*sa)]
        ms.append(100 * sum(m[rng.randrange(n)] for _ in range(n)) / n)
    ms.sort(); return ms[int(.025 * nb)], ms[int(.975 * nb) - 1]


V = {}; FIRE = {}
for rid, lab, hk, mode in ROWS:
    for b, bl, key in BENCH:
        runs = [load(t, b) for t in HOSTS[hk]]
        if any(r is None for r in runs): continue
        V[(rid, b)] = [[x[f'{key}_{mode}'] for x in r] for r in runs]
        if mode == 'a':
            n = sum(len(r) for r in runs)
            FIRE[(rid, b)] = dict(seq=sum(1 for r in runs for x in r if x.get('fsc')) / n,
                                  pos=sum(x.get('fsc', 0) for r in runs for x in r) / max(1, sum(x.get('npos', 1) for r in runs for x in r)),
                                  any_prompt=sum(1 for r in runs for x in r if x.get('fall')) / n if 'fall' in runs[0][0] else None,
                                  end=sum(1 for r in runs for x in r if x.get('end')) / n,
                                  changed=sum(1 for r in runs for x in r if x[f'{key}_a'] != x[f'{key}_h']) / n)
out = dict(rows={}, diffs={}, fire={})
L = ['| row | config | ' + ' | '.join(bl for _, bl, _ in BENCH) + ' |', '|---|---|' + '---|' * len(BENCH)]
for rid, lab, hk, mode in ROWS:
    cells = []
    for b, bl, _ in BENCH:
        A = V.get((rid, b))
        if A is None: cells.append('–'); continue
        per = [100 * sum(r) / len(r) for r in A]; m = sum(per) / len(per)
        lo, hi = boot1(A[0]) if len(A) == 1 else hboot1(A)
        s = f'{m:.2f} [{lo:.1f}, {hi:.1f}]' + (' (' + '/'.join(f'{p:.1f}' for p in per) + ')' if len(A) > 1 else '')
        cells.append(s); out['rows'].setdefault(rid, {'config': lab})[b] = dict(acc=m, ci=[lo, hi], per_seed=per, n=len(A[0]))
    L.append(f'| {rid} | {lab} | ' + ' | '.join(cells) + ' |')
print('\n'.join(L))
DIFFS = [('4 − 2', '4', '2'), ('4 − 2b', '4', '2b'), ('4h − 2', '4h', '2'), ('2 − 1', '2', '1'), ('4 − 1', '4', '1'), ('2b − 1', '2b', '1'), ('3 − 1', '3', '1')]
L = ['\n| difference | ' + ' | '.join(bl for _, bl, _ in BENCH) + ' |', '|---|' + '---|' * len(BENCH)]
for nm, x, y in DIFFS:
    cells = []
    for b, bl, _ in BENCH:
        A, B = V.get((x, b)), V.get((y, b))
        if A is None or B is None: cells.append('–'); continue
        d = sum(100 * sum(r) / len(r) for r in A) / len(A) - sum(100 * sum(r) / len(r) for r in B) / len(B)
        lo, hi = hboot(A, B)
        cells.append(f'{d:+.2f} [{lo:+.2f}, {hi:+.2f}]'); out['diffs'].setdefault(nm, {})[b] = dict(d=d, ci=[lo, hi])
    L.append(f'| {nm} | ' + ' | '.join(cells) + ' |')
print('\n'.join(L))
L = ['\n| implants attached: row | benchmark | items with an implant write on a scored position | scored positions with a write | '
     'items with a write anywhere in the prompt | serial END trigger on a scored position | items whose decision changed |', '|---|---|---|---|---|---|---|']
for (rid, b), f in FIRE.items():
    ap = '–' if f['any_prompt'] is None else f"{100 * f['any_prompt']:.2f}%"
    L.append(f"| {rid} | {b} | {100 * f['seq']:.3f}% | {100 * f['pos']:.4f}% | {ap} | "
             f"{100 * f['end']:.3f}% | {100 * f['changed']:.3f}% |")
    out['fire'][f'{rid}/{b}'] = f
print('\n'.join(L))
json.dump(out, open(os.environ.get('OT_ROWS', f'{HERE}/rows{SIZE}.json'), 'w'), indent=1)
