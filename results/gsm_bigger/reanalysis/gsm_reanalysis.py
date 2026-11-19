"""Job 2 (critic item 4): GSM re-analysis from SAVED per-problem JSONs, no new runs (CPU, pure analysis).

Subsets: all 1319; idx>=200 (problems 201+, 1-based; the first 200 GSM-Hard problems were used to hill-climb the 8-shot
prompt); clean = gold is a non-negative integer; 201+ & clean. Paired bootstrap (10k resamples of problems) for the gain of
each arm over its baseline, plain percentile bootstrap for single accuracies. Also token budgets / truncation.
"""
import json, glob, os, random, re, sys
import numpy as np

W = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo results/
FULL = os.environ.get('WSM_FULL_OUTPUTS', '')  # optional: unreduced outputs (with text) for the truncation check; not released
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gsm_reanalysis.json')
rng = np.random.default_rng(0)
B = 10000


def ok_of(v): return v is True or v == 'True'


def load(paths):
    rows = {}
    meta = None
    for p in paths:
        d = json.load(open(p)); meta = meta or {k: v for k, v in d.items() if k != 'rows'}
        for r in d['rows']: rows[int(r['i'])] = r
    assert len(rows) == 1319, (paths, len(rows))
    return rows, meta


def clean(g):
    try: x = float(str(g).replace(',', ''))
    except ValueError: return False
    return x >= 0 and abs(x - round(x)) < 1e-9


ARMS = {}   # (split, name) -> rows
for sp in ['test', 'hard']:
    for m in ['plain', 'calc']:
        ARMS[(sp, f'{m} (320 tok)')] = load([f'{W}/gsm_full_implant/{sp}_{m}.json'])
        for imp, nm in [('none', 'no rethink'), ('alwaysA', 'rethink always'), ('hybrid', 'rethink gated (detector)')]:
            ARMS[(sp, f'v2 {m} + {nm} (700 tok)')] = load(sorted(glob.glob(f'{W}/rethink_v2/final_eval/{sp}_{m}_{imp}_s*.json')))
    for q in ['q17', 'q4']:
        for m in ['plain', 'calc']:
            p = f'{W}/gsm_bigger/{q}_{sp}_{m}.json'
            if not os.path.exists(p): p = f'{FULL}/gsm_bigger/{q}_{sp}_{m}.json'
            if os.path.exists(p): ARMS[(sp, f'{q} {m} (320 tok)')] = load([p])

BASE = {'calc (320 tok)': 'plain (320 tok)', 'v2 plain + no rethink (700 tok)': 'plain (320 tok)',
        'v2 calc + no rethink (700 tok)': 'v2 plain + no rethink (700 tok)',
        'v2 plain + rethink always (700 tok)': 'v2 plain + no rethink (700 tok)', 'v2 plain + rethink gated (detector) (700 tok)': 'v2 plain + no rethink (700 tok)',
        'v2 calc + rethink always (700 tok)': 'v2 plain + no rethink (700 tok)', 'v2 calc + rethink gated (detector) (700 tok)': 'v2 plain + no rethink (700 tok)',
        'q17 calc (320 tok)': 'q17 plain (320 tok)', 'q4 calc (320 tok)': 'q4 plain (320 tok)'}
EXTRA = {('v2 calc + rethink always (700 tok)', 'v2 calc + no rethink (700 tok)'), ('v2 calc + rethink gated (detector) (700 tok)', 'v2 calc + rethink always (700 tok)'),
         ('v2 plain + rethink gated (detector) (700 tok)', 'v2 plain + rethink always (700 tok)')}

res = {}
lines = []
for sp in ['test', 'hard']:
    gold = {i: r['gold'] for i, r in ARMS[(sp, 'plain (320 tok)')][0].items()}
    subsets = {'all': [i for i in range(1319)], '201+': [i for i in range(200, 1319)],
               'clean': [i for i in range(1319) if clean(gold[i])], '201+ & clean': [i for i in range(200, 1319) if clean(gold[i])],
               'first 200': list(range(200)), 'weird labels': [i for i in range(1319) if not clean(gold[i])]}
    for nm, (rows, _) in ARMS.items():
        if nm[0] != sp: continue
        assert all(str(rows[i]['gold']) == str(gold[i]) for i in range(1319)), nm
    lines.append(f'\n## {"GSM8K" if sp == "test" else "GSM-Hard"} (n by subset: ' + ', '.join(f'{k} {len(v)}' for k, v in subsets.items()) + ')')
    for sname, idx in subsets.items():
        idx = np.array(idx); bs = rng.integers(0, len(idx), size=(B, len(idx)))
        lines.append(f'\n### {sp} / {sname} (n={len(idx)})\n| arm | acc [95% CI] | vs | gain [95% CI] |\n|---|---|---|---|')
        for (s2, nm), (rows, meta) in ARMS.items():
            if s2 != sp: continue
            v = np.array([ok_of(rows[i]['ok']) for i in idx], float)
            boot = v[bs].mean(1) * 100
            e = {'n': len(idx), 'acc': v.mean() * 100, 'ci': list(np.percentile(boot, [2.5, 97.5]))}
            cell = f'| {nm} | {e["acc"]:.1f} [{e["ci"][0]:.1f}, {e["ci"][1]:.1f}] |'
            comps = [(nm, BASE[nm])] if nm in BASE else []
            comps += [c for c in EXTRA if c[0] == nm]
            if not comps: lines.append(cell + ' | |')
            for a, b in comps:
                u = np.array([ok_of(ARMS[(sp, b)][0][i]['ok']) for i in idx], float)
                dd = (v - u)[bs].mean(1) * 100
                e.setdefault('gain', {})[b] = {'d': (v - u).mean() * 100, 'ci': list(np.percentile(dd, [2.5, 97.5])), 'p_le0': float((dd <= 0).mean())}
                g = e['gain'][b]
                lines.append(cell + f' {b} | {g["d"]:+.1f} [{g["ci"][0]:+.1f}, {g["ci"][1]:+.1f}] |'); cell = '| | |'
            res.setdefault(sp, {}).setdefault(sname, {})[nm] = e

# ---------------- token budgets / truncation
lines.append('\n## Token budgets and truncation')
for (sp, nm), (rows, meta) in ARMS.items():
    if nm.startswith('v2'):
        nt = [r.get('n_tok', 0) for r in rows.values()]; cap = meta['max_new']
        lines.append(f'- {sp} {nm}: max_new {cap}; outputs at the cap {sum(x >= cap for x in nt)}/1319; >=320 tokens {sum(x >= 320 for x in nt)}; '
                     f'mean tokens {np.mean(nt):.0f}; no #### answer {sum(r["pred"] in (None, "", "None") for r in rows.values())}')
if not FULL:   # keep the shipped truncation lines (they need the unreleased full-text outputs)
    try:
        prev = json.load(open(OUT)).get('trunc', {})
        res['trunc'] = prev
        lines += [l for l in open(OUT.replace('.json', '.md')).read().splitlines() if '(320 tok, full text re-tokenized)' in l]
    except Exception:
        pass
else:
  try:
      from transformers import AutoTokenizer
      tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
      for sp in ['test', 'hard']:
          for m in ['plain', 'calc']:
              d = json.load(open(f'{FULL}/gsm_full_implant/a100/{sp}_{m}.json'))['rows']
              n = [len(tok(r['out'], add_special_tokens=False)['input_ids']) for r in d]
              trunc = [r for r, k in zip(d, n) if k >= 316 and '\n\n' not in r['out']]
              noans = [r for r in d if r['pred'] in (None, '', 'None')]
              tr_noans = [r for r in trunc if r['pred'] in (None, '', 'None')]
              lines.append(f'- {sp} {m} (320 tok, full text re-tokenized): mean {np.mean(n):.0f} tok; >=316 tok w/o blank-line stop (truncated) {len(trunc)}/1319 '
                           f'({len(tr_noans)} of them with no #### answer); no answer overall {len(noans)}')
              res.setdefault('trunc', {})[f'{sp}_{m}'] = {'truncated': len(trunc), 'truncated_noans': len(tr_noans), 'noans': len(noans), 'mean_tok': float(np.mean(n))}
  except Exception as ex:
    lines.append(f'- tokenizer check failed: {ex}')

open(OUT.replace('.json', '.md'), 'w').write('\n'.join(lines) + '\n')
json.dump(res, open(OUT, 'w'), indent=1)
print('\n'.join(lines))
