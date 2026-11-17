"""Score runs with the official lm_eval IFEval checkers (instruction kwargs from our generator / the dataset).
python score.py DATA.jsonl RUN_A_PREFIX RUN_B_PREFIX ...   (prefix = runs/dev_plain -> merges runs/dev_plain.s*.jsonl)
Prints prompt/instruction strict+loose, per-type strict pass counts, gained/lost prompts, and silent identity (B vs A)."""
import json, sys, glob, argparse
from collections import defaultdict
from lm_eval.tasks.ifeval import utils as U
import langdetect; langdetect.DetectorFactory.seed = 0
ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('runs', nargs='+'); ap.add_argument('--filter', default='')
ap.add_argument('--json', default=''); ap.add_argument('--keys', default='')
a = ap.parse_args()
docs = {}
for l in open(a.data):
    d = json.loads(l); docs[d['key']] = d
SHORT = {'punctuation:no_comma': 'no_comma', 'change_case:english_lowercase': 'lower', 'change_case:english_capital': 'caps', 'startend:quotation': 'quote',
         'startend:end_checker': 'end', 'detectable_content:postscript': 'ps', 'keywords:forbidden_words': 'forbid', 'combination:repeat_prompt': 'repeat',
         'combination:two_responses': 'two', 'length_constraints:number_paragraphs': 'paras', 'length_constraints:nth_paragraph_first_word': 'nth',
         'detectable_format:title': 'title', 'detectable_format:number_highlighted_sections': 'high', 'detectable_format:number_bullet_lists': 'bullets'}
def load(prefix):
    fs = sorted(glob.glob(prefix + '.s*.jsonl')) or [prefix]
    rows = {}
    for f in fs:
        for l in open(f): r = json.loads(l); rows[r['key']] = r
    return rows
def keep(d):
    if a.keys == 'even' and d['key'] % 2: return False
    if a.keys == 'odd' and not d['key'] % 2: return False
    if a.filter == 'fresh_phrasing': return 'fresh' in d.get('meta', {}).get('tags', [])
    if a.filter.startswith('style='): return d.get('style') == a.filter[6:]
    return True
res = {}
LOADED = {p: load(p) for p in a.runs}
common = set.intersection(*[set(R) for R in LOADED.values()])          # compare runs on the same prompts
for p in a.runs:
    R = LOADED[p]; ks = sorted(k for k in R if k in docs and keep(docs[k]) and k in common)
    per, inst = {}, defaultdict(lambda: [0, 0, 0]); ps = pl = 0
    for k in ks:
        d = docs[k]; resp = R[k]['response']
        doc = {'key': k, 'instruction_id_list': d['instruction_id_list'], 'prompt': d['prompt'], 'kwargs': d['kwargs']}
        o = U.process_results(doc, [resp])
        ps += o['prompt_level_strict_acc']; pl += o['prompt_level_loose_acc']; per[k] = o['prompt_level_strict_acc']
        for iid, s, l_ in zip(d['instruction_id_list'], o['inst_level_strict_acc'], o['inst_level_loose_acc']):
            inst[iid][0] += 1; inst[iid][1] += s; inst[iid][2] += l_
    n = len(ks); ni = sum(v[0] for v in inst.values())
    res[p] = dict(n=n, ps=100 * ps / n, pl=100 * pl / n, is_=100 * sum(v[1] for v in inst.values()) / ni, il=100 * sum(v[2] for v in inst.values()) / ni,
                  inst=dict(inst), per=per, R=R, trunc=sum(R[k]['finish'] != 'stop' for k in ks), ks=ks)
print(f"{'run':40s} {'n':>4s} {'P-strict':>8s} {'P-loose':>8s} {'I-strict':>8s} {'I-loose':>8s} {'trunc':>5s}")
for p, r in res.items(): print(f"{p[-40:]:40s} {r['n']:4d} {r['ps']:8.2f} {r['pl']:8.2f} {r['is_']:8.2f} {r['il']:8.2f} {r['trunc']:5d}")
ids = sorted({k for r in res.values() for k in r['inst']}, key=lambda k: (k not in SHORT, k))
print('\nper-instruction strict pass (pass/n)')
for k in ids: print(f"{SHORT.get(k, k):45s}" + ''.join(f" {r['inst'].get(k, [0, 0])[1]:>4d}/{r['inst'].get(k, [0])[0]:<4d}" for r in res.values()))
if len(res) >= 2:
    A, B = list(res.values())[0], list(res.values())[-1]
    gained = [k for k in A['per'] if not A['per'][k] and B['per'].get(k)]; lost = [k for k in A['per'] if A['per'][k] and not B['per'].get(k)]
    print(f'\nprompt-strict first->last: gained {len(gained)} lost {len(lost)}; lost keys {lost[:40]}')
    sil = [k for k in B['ks'] if (B['R'][k].get('circ') or {}).get('fires', 1) == 0]
    same = sum(A['R'][k]['response'] == B['R'][k]['response'] for k in sil if k in A['R'])
    print(f'silent prompts (circuit wrote nothing): {len(sil)}; byte-identical to first run: {same}/{len(sil)}')
    KEYS = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul']
    unl = [k for k in B['ks'] if not any((B['R'][k].get('circ') or {}).get(x) for x in KEYS)]
    same2 = sum(A['R'][k]['response'] == B['R'][k]['response'] for k in unl if k in A['R'])
    print(f'unlatched prompts (no rule latched): {len(unl)}; byte-identical to first run: {same2}/{len(unl)}')
if a.json: json.dump({p: {k: v for k, v in r.items() if k not in ('R', 'per', 'ks')} for p, r in res.items()}, open(a.json, 'w'), indent=1)
