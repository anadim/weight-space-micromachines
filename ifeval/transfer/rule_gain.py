"""Per-rule gain of the frozen 14-rule implant on Team D's FRESH (seed 1) and DEV runs -> the implant-count order.
python rule_gain.py data/fresh.jsonl runs_teamd/fresh_plain runs_teamd/fresh_imp
Gain of rule r = (instruction-strict passes of r's instruction id with implants) - (same, plain), over prompts present in both runs.
Also prompt-strict net (fixed - broken) among prompts whose instruction list contains r."""
import json, sys, glob, random
from collections import defaultdict
from lm_eval.tasks.ifeval import utils as U
import langdetect; langdetect.DetectorFactory.seed = 0
RID = {'punctuation:no_comma': 'nc', 'change_case:english_lowercase': 'lower', 'change_case:english_capital': 'caps', 'startend:quotation': 'quote',
       'startend:end_checker': 'end', 'detectable_content:postscript': 'ps', 'keywords:forbidden_words': 'forbid', 'combination:repeat_prompt': 'repeat',
       'combination:two_responses': 'two', 'length_constraints:number_paragraphs': 'para', 'length_constraints:nth_paragraph_first_word': 'nth',
       'detectable_format:title': 'title', 'detectable_format:number_highlighted_sections': 'high', 'detectable_format:number_bullet_lists': 'bul'}
docs = {json.loads(l)['key']: json.loads(l) for l in open(sys.argv[1])}
def load(p):
    R = {}
    for f in sorted(glob.glob(p + '.s*.jsonl')) or [p]:
        for l in open(f): r = json.loads(l); R[r['key']] = r
    return R
A, B = load(sys.argv[2]), load(sys.argv[3]); ks = sorted(set(A) & set(B) & set(docs))
def sc(k, resp):
    random.seed(k)   # order-independent scoring (see score_t.py)
    d = docs[k]; o = U.process_results({'key': k, 'instruction_id_list': d['instruction_id_list'], 'prompt': d['prompt'], 'kwargs': d['kwargs']}, [resp])
    return o['prompt_level_strict_acc'], o['inst_level_strict_acc']
inst = defaultdict(lambda: [0, 0, 0]); pnet = defaultdict(lambda: [0, 0])
for k in ks:
    pa, ia = sc(k, A[k]['response']); pb, ib = sc(k, B[k]['response'])
    for iid, x, y in zip(docs[k]['instruction_id_list'], ia, ib):
        if iid in RID: r = RID[iid]; inst[r][0] += 1; inst[r][1] += x; inst[r][2] += y
    for r in {RID[i] for i in docs[k]['instruction_id_list'] if i in RID}:
        pnet[r][0] += (pb and not pa); pnet[r][1] += (pa and not pb)
print('n prompts', len(ks))
rows = sorted(inst, key=lambda r: -(inst[r][2] - inst[r][1]))
print(f"{'rule':7s} {'n':>4s} {'plain':>6s} {'imp':>6s} {'gain':>5s} {'P-fix':>5s} {'P-brk':>5s}")
for r in rows: n, a, b = inst[r]; print(f"{r:7s} {n:4d} {a:6d} {b:6d} {b - a:5d} {pnet[r][0]:5d} {pnet[r][1]:5d}")
json.dump({r: dict(n=inst[r][0], plain=inst[r][1], imp=inst[r][2], gain=inst[r][2] - inst[r][1], pfix=pnet[r][0], pbrk=pnet[r][1]) for r in inst}, open(sys.argv[4], 'w'), indent=1)
