"""IFBench official checkers (allenai/IFBench evaluation_lib) on our jsonl outputs. python score_ifbench.py run1.jsonl [run2.jsonl ...]"""
import sys, json, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IFBench')); sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pydeps'))
import evaluation_lib as EL
import langdetect; langdetect.DetectorFactory.seed = 0
inputs = EL.read_prompt_list('data/ifbench_orig.jsonl')
res = []
for path in sys.argv[1:]:
    rows = {json.loads(l)['prompt']: json.loads(l)['response'] for l in open(path)}
    out = {}
    for name, fn in [('strict', EL.test_instruction_following_strict), ('loose', EL.test_instruction_following_loose)]:
        o = [fn(inp, rows) for inp in inputs if inp.prompt in rows]
        P = 100 * sum(x.follow_all_instructions for x in o) / len(o)
        I = 100 * sum(sum(x.follow_instruction_list) for x in o) / sum(len(x.follow_instruction_list) for x in o)
        per = collections.Counter(); tot = collections.Counter()
        for x in o:
            for i, f in zip(x.instruction_id_list, x.follow_instruction_list): tot[i] += 1; per[i] += f
        out[name] = (len(o), P, I, per, tot)
    res.append((path, out))
    print(f"{path[-40:]:40s} n={out['strict'][0]} P-strict {out['strict'][1]:.2f} P-loose {out['loose'][1]:.2f} I-strict {out['strict'][2]:.2f} I-loose {out['loose'][2]:.2f}")
if len(res) > 1:
    a, b = res[0][1]['strict'], res[-1][1]['strict']
    print('instructions whose strict pass count changed (first -> last):')
    for k in sorted(a[4]):
        if a[3][k] != b[3][k]: print(f'  {k:45s} {a[3][k]}/{a[4][k]} -> {b[3][k]}/{b[4][k]}')
