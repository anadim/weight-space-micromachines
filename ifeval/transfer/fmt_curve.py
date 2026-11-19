"""Markdown table of the implant-count curve from score_t.py JSON. python fmt_curve.py TAG"""
import json, sys
ORDER = ['nc', 'para', 'repeat', 'forbid', 'nth', 'caps', 'two', 'lower', 'end', 'high', 'quote', 'bul', 'title', 'ps']
tag = sys.argv[1]; J = json.load(open(f'runs/{tag}_curve_score.json'))
print('| k | rule added | ALL P-strict | ALL I-strict | ODD P-strict | ODD I-strict | EVEN P-strict | EVEN I-strict | ALL fixed/broken | ODD fixed/broken | EVEN fixed/broken |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for k in range(15):
    r = J[f'runs/{tag}_k{k}.jsonl']; a, o, e = r['all'], r['odd'], r['even']
    print(f"| {k} | {ORDER[k-1] if k else '(plain)'} | {a['ps']:.2f} | {a['is_']:.2f} | {o['ps']:.2f} | {o['is_']:.2f} | {e['ps']:.2f} | {e['is_']:.2f} | +{a['fixed']}/-{a['broken']} | +{o['fixed']}/-{o['broken']} | +{e['fixed']}/-{e['broken']} |")
