"""Causal <calc>-drop analysis from run3 rollout dumps (CPU). Per update window: calc-row share, within-prompt reward gap
(<calc> rows minus other rows, prompts having both), advantage x token mass per style (Dr.GRPO token-sum), tag-token cost,
and the items driving a negative gap. Usage: python causal_analyze.py run_dir [run_dir ...]"""
import sys, json, collections, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
WIN = [(1, 2), (4, 6), (9, 11), (19, 21), (39, 41), (48, 50)]
def style(t):
    if '<calc>' in t: return 'calc'
    if re.search(r'\d\s*[-+*/x]\s*\d[^\n]{0,20}=\s*-?\d', t): return 'eq'
    return 'other'
for rd in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f'{rd}/rollouts.jsonl')]
    seen = {}
    for r in rows: seen[(r['u'], r['p'], tuple(r['toks'][:8]), r['n'], r['r'])] = r   # dedupe resumed duplicates
    rows = list(seen.values())
    for r in rows:
        t = tok.decode(r['toks']); r['st'] = style(t)
        r['tagtok'] = sum(len(m) for m in re.findall(r'<calc>.*?</calc>', t)) / 3.5 if r['st'] == 'calc' else 0   # ~chars/3.5
    print(f'== {rd}  rows {len(rows)}')
    print('win      type | calc%  eq%  | gap(calc-other) n_prompts | advxtok calc  eq  other (per row) | adv calc | len calc/other')
    item = collections.defaultdict(lambda: [0.0, 0])
    for lo, hi in WIN:
        for ty in ('all', 'gsm', 'bbh'):
            R = [r for r in rows if lo <= r['u'] <= hi and (ty == 'all' or r['type'] == ty)]
            if not R: continue
            byp = collections.defaultdict(list)
            for r in R: byp[(r['u'], r['p'])].append(r)
            gaps = []; gce = []; rs = {'calc': [], 'eq': [], 'other': []}
            for r in R: rs[r['st']].append(r['r'])
            for k, g in byp.items():
                c_ = [x['r'] for x in g if x['st'] == 'calc']; e_ = [x['r'] for x in g if x['st'] == 'eq']
                if c_ and e_: gce.append(sum(c_) / len(c_) - sum(e_) / len(e_))
                c = [x['r'] for x in g if x['st'] == 'calc']; o = [x['r'] for x in g if x['st'] != 'calc']
                if c and o:
                    gp = sum(c) / len(c) - sum(o) / len(o); gaps.append(gp)
                    if ty == 'all' and hi <= 21: item[(g[0]['id'], g[0]['task'])][0] += gp; item[(g[0]['id'], g[0]['task'])][1] += 1
            n = len(R); f = lambda s: sum(1 for r in R if r['st'] == s) / n
            am = {s: sum(r['adv'] * r['n'] for r in R if r['st'] == s and r['kept']) / n for s in ('calc', 'eq', 'other')}
            ac = [r['adv'] for r in R if r['st'] == 'calc' and r['kept']]
            lc = [r['n'] for r in R if r['st'] == 'calc']; lo_ = [r['n'] for r in R if r['st'] != 'calc']
            print(f'u{lo:>2}-{hi:<3} {ty:>4} | {100*f("calc"):5.1f} {100*f("eq"):5.1f} | {sum(gaps)/max(1,len(gaps)):+.3f} {len(gaps):4d} | '
                  f'{am["calc"]:+7.2f} {am["eq"]:+7.2f} {am["other"]:+7.2f} | {sum(ac)/max(1,len(ac)):+.3f} | '
                  f'{sum(lc)/max(1,len(lc)):5.0f}/{sum(lo_)/max(1,len(lo_)):5.0f} | gap calc-eq {sum(gce)/max(1,len(gce)):+.3f} n{len(gce)} | '
                  f'r calc/eq/other ' + '/'.join(f'{sum(v)/max(1,len(v)):.2f}' for v in rs.values()))
    worst = sorted(((v[0] / v[1], v[1], k) for k, v in item.items() if v[1] >= 2), key=lambda x: x[0])[:8]
    bytask = collections.defaultdict(lambda: [0.0, 0])
    for k, v in item.items(): bytask[k[1]][0] += v[0]; bytask[k[1]][1] += v[1]
    print('gap by task (u<=21):', {t: (round(v[0] / v[1], 3), v[1]) for t, v in sorted(bytask.items(), key=lambda x: x[1][0] / x[1][1])})
    print('most negative items (u<=21):', [(round(g, 2), n, k[0], k[1]) for g, n, k in worst])
