"""NLTK punkt_tab check for the IFEval seeds 3-5 (see IFEVAL.md): on the 46 IFEval-541 prompts with a
length_constraints:number_sentences instruction, count how often that instruction itself passes (strict), per run.
python3 ifana/nltk_check.py     (CPU, reads ../res/ifeval/ and uniform/rl24/ifeval/res/)"""
import json, os, sys
sys.dont_write_bytecode = True
from ifitems import rows as _rows   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); O = os.environ.get('AO', os.path.join(HERE, '..', 'res', 'ifeval'))
NS = 'length_constraints:number_sentences'


def count(run, ev):
    rows = _rows(run, ev)
    if rows is None: return None
    ns = [r for r in rows if NS in r['ids']]
    return len(ns), sum(r['strict'][r['ids'].index(NS)] for r in ns), sum(all(r['strict']) for r in ns)


print('| run | eval | number_sentences prompts | instruction pass | whole prompt pass |')
print('|---|---|---|---|---|')
for a, evs in (('P', ('eval_u300.plain', 'eval_u300.imp')), ('I', ('eval_u300.imp',))):
    for ev in evs:
        for s in range(6):
            c = count(f'{a}_s{s}', ev)
            if c: print(f'| {a}_s{s} | {ev.split(".")[1]} | {c[0]} | {c[1]} | {c[2]} |')
