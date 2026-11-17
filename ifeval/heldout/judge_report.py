"""Per-item report for a judge's variants: circuit latches/arguments (from the implant run's circuit report), official per-instruction
strict scores plain vs implant, whether the circuit wrote anything, byte-identity with plain, finish/ntok, and both responses.
python judge_report.py judge/all_variants_r1.jsonl runs/j1_plain runs/j1_imp TOOLNAME > judge/TOOL_results_r1.md"""
import json, sys, glob
from lm_eval.tasks.ifeval import utils as U
import langdetect; langdetect.DetectorFactory.seed = 0
data = [json.loads(l) for l in open(sys.argv[1])]
def load(p):
    R = {}
    for f in sorted(glob.glob(p + '.s*.jsonl')) or [p]:
        for l in open(f): r = json.loads(l); R[r['key']] = r
    return R
P, I = load(sys.argv[2]), load(sys.argv[3]); tool = sys.argv[4]
KEEP = ['nc', 'lower', 'caps', 'quote', 'end', 'ps', 'pps', 'forbid', 'repeat', 'two', 'para', 'star3', 'nth', 'title', 'high', 'bul']
print(f'# Run results for judge {tool} (circuit version after the round-1 guard changes; see message)\n')
for d in data:
    if d.get('judge') != tool: continue
    k = d['key']; p, i = P.get(k), I.get(k)
    if p is None or i is None: print(f'## {k}: MISSING RUN\n'); continue
    doc = {'key': k, 'instruction_id_list': d['instruction_id_list'], 'prompt': d['prompt'], 'kwargs': d['kwargs']}
    sp = U.process_results(doc, [p['response']])['inst_level_strict_acc'] if d['instruction_id_list'] else []
    si = U.process_results(doc, [i['response']])['inst_level_strict_acc'] if d['instruction_id_list'] else []
    c = i.get('circ') or {}
    lat = [x for x in KEEP if c.get(x)]
    args = {a: c[a] for a in ['N_par', 'N_sec', 'N_bul', 'N_high', 'nth_k', 'nth_last', 'nth_word', 'end_phrase', 'rep', 'forbid_h'] if c.get(a)}
    if args.get('forbid_h'):
        def wh(w):
            h = 0
            for ch in w.lower(): h = (h * 131 + ord(ch)) % 2147483647
            return h
        exp = [w for kw in d['kwargs'] for w in (kw or {}).get('forbidden_words', [])]
        args['forbid_h'] = f"{len(args['forbid_h'])} words captured; expected words captured: {[w for w in exp if wh(w) in args['forbid_h']]} of {exp}"
    print(f"## {k}  ({d.get('note', '')[:160]})")
    print(f"- latched: {lat}  args: {args}  circuit wrote on {c.get('fires', 0)} steps; forced-token misses {c.get('ffail', 0)}")
    print(f"- expected ids: {d['instruction_id_list']}  strict plain {sp}  implant {si}")
    print(f"- implant output byte-identical to plain: {p['response'] == i['response']}; finish plain {p['finish']}/{p['ntok']} implant {i['finish']}/{i['ntok']}")
    print(f"- PLAIN: {json.dumps(p['response'][:500])}{' ...' if len(p['response']) > 500 else ''}")
    print(f"- IMPLANT: {json.dumps(i['response'][:700])}{' ... [end: ' + json.dumps(i['response'][-200:]) + ']' if len(i['response']) > 700 else ''}\n")
