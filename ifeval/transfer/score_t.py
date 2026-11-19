"""Score plain vs implant runs on IFEval with the official lm_eval checkers (same calls as Team D's score.py), split ODD/EVEN.
python score_t.py OUT.json PLAIN_PREFIX RUN_PREFIX [RUN_PREFIX ...]   (prefix runs/x -> merges runs/x.s*.jsonl)
Per run and split: prompt/instruction strict+loose, fixed/broken (prompt-strict) vs PLAIN, per-rule instruction-strict passes,
silent prompts (circuit wrote nothing) byte-identical to PLAIN, truncations."""
import json, sys, glob, random
from collections import defaultdict
from lm_eval.tasks.ifeval import utils as U
import langdetect; langdetect.DetectorFactory.seed = 0
RID = {'punctuation:no_comma': 'nc', 'change_case:english_lowercase': 'lower', 'change_case:english_capital': 'caps', 'startend:quotation': 'quote',
       'startend:end_checker': 'end', 'detectable_content:postscript': 'ps', 'keywords:forbidden_words': 'forbid', 'combination:repeat_prompt': 'repeat',
       'combination:two_responses': 'two', 'length_constraints:number_paragraphs': 'para', 'length_constraints:nth_paragraph_first_word': 'nth',
       'detectable_format:title': 'title', 'detectable_format:number_highlighted_sections': 'high', 'detectable_format:number_bullet_lists': 'bul'}
docs = {}
for l in open('data/ifeval.jsonl'): d = json.loads(l); docs[d['key']] = d
def load(p):
    R = {}
    for f in sorted(glob.glob(p + '.s*.jsonl')) or [p]:
        for l in open(f): r = json.loads(l); R[r['key']] = r
    return R
cache = {}
def sc(k, resp):
    if (k, resp) in cache: return cache[(k, resp)]
    random.seed(k)   # lm_eval builds a missing kwarg with random (keys 1122/1129: letter_frequency); seed per key -> order-independent
    d = docs[k]; o = U.process_results({'key': k, 'instruction_id_list': d['instruction_id_list'], 'prompt': d['prompt'], 'kwargs': d['kwargs']}, [resp])
    cache[(k, resp)] = o; return o
out = {}
P = load(sys.argv[2])
for p in sys.argv[2:]:
    R = load(p); out[p] = {}
    for split in ['odd', 'even', 'all']:
        ks = sorted(k for k in R if k in P and (split == 'all' or (k % 2 == 1) == (split == 'odd')))
        ps = pl = 0; ni = is_ = il = 0; fixed = broken = 0; inst = defaultdict(lambda: [0, 0]); sil = same = 0; trunc = 0
        for k in ks:
            o = sc(k, R[k]['response']); o0 = sc(k, P[k]['response'])
            ps += o['prompt_level_strict_acc']; pl += o['prompt_level_loose_acc']
            fixed += o['prompt_level_strict_acc'] and not o0['prompt_level_strict_acc']; broken += o0['prompt_level_strict_acc'] and not o['prompt_level_strict_acc']
            for iid, s, l_ in zip(docs[k]['instruction_id_list'], o['inst_level_strict_acc'], o['inst_level_loose_acc']):
                ni += 1; is_ += s; il += l_
                r = RID.get(iid, 'other'); inst[r][0] += 1; inst[r][1] += s
            c = R[k].get('circ')
            if c is not None and c.get('fires', 1) == 0: sil += 1; same += R[k]['response'] == P[k]['response']
            trunc += R[k]['finish'] != 'stop'
        n = len(ks)
        out[p][split] = dict(n=n, ps=100 * ps / n, pl=100 * pl / n, is_=100 * is_ / ni, il=100 * il / ni, fixed=fixed, broken=broken,
                             inst=dict(inst), silent=sil, silent_same=same, trunc=trunc,
                             ident=sum(R[k]['response'] == P[k]['response'] for k in ks))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for split in ['odd', 'even', 'all']:
    print(f'== {split}')
    print(f"{'run':34s} {'n':>4s} {'P-str':>6s} {'P-loo':>6s} {'I-str':>6s} {'I-loo':>6s} {'fix':>4s} {'brk':>4s} {'silent=':>9s} {'ident':>5s} {'trunc':>5s}")
    for p, r in out.items():
        x = r[split]; print(f"{p[-34:]:34s} {x['n']:4d} {x['ps']:6.2f} {x['pl']:6.2f} {x['is_']:6.2f} {x['il']:6.2f} {x['fixed']:4d} {x['broken']:4d} {x['silent_same']:4d}/{x['silent']:<4d} {x['ident']:5d} {x['trunc']:5d}")
    rules = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'forbid', 'repeat', 'two', 'para', 'nth', 'title', 'high', 'bul', 'other']
    print('per-rule instruction-strict passes: ' + ' '.join(f"{r}" for r in rules))
    for p, r in out.items():
        x = r[split]['inst']; print(f"{p[-34:]:34s} " + ' '.join(f"{x.get(q, [0, 0])[1]}/{x.get(q, [0, 0])[0]}" for q in rules))
