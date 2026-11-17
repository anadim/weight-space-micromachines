"""Compare a rethink-IMPLANT run with a reference rethink run (the evaluation-code path, --rethink), problem by problem.

python scripts/compare_rethink.py IMPLANT.json REFERENCE.json [--gate GATE.json]
  first answer text : reference `out.rstrip() + "\\n" + PHRASE` vs implant `out + forced` (the text up to the end of the phrase)
  rethink text      : the text generated after the phrase (`rt_text`; for saved runs only `out2`, kept when it has '####')
  final answer      : the last '####' after the phrase if there is one, else the first answer; and its correctness
--gate GATE.json    : the reference is an ungated run; apply the eval-code gate (scripts/gate/eval_gate.py output) to it first
"""
import json, re, sys, argparse
PH = "Hmm, wait. Let me rethink this.\n"
NUM = r'####\s*(-?[\d,]*\.?\d+)'
ap = argparse.ArgumentParser(); ap.add_argument('imp'); ap.add_argument('ref'); ap.add_argument('--gate'); ap.add_argument('--show', type=int, default=3)
a = ap.parse_args()
I = {r['i']: r for r in json.load(open(a.imp))['rows']}; R = {r['i']: r for r in json.load(open(a.ref))['rows']}
for r in I.values():                                  # a reference-style run as the first argument (e.g. today's vs the saved one)
    if 'implant_wrote' not in r: r.update(implant_wrote=True, forced='', out=r['out'].rstrip() + '\n' + PH)
G = {int(k): v for k, v in json.load(open(a.gate))['fires'].items()} if a.gate else None
ks = sorted(set(I) & set(R))


def final(r):
    t2 = r.get('rt_text', r.get('out2'))
    m2 = re.findall(NUM, t2) if t2 is not None else []
    return m2[-1].replace(',', '') if m2 else r['pred']


c = dict(n=len(ks), first_text=0, rethink_compared=0, rethink_text=0, final_pred=0, final_ok=0, first_ok=0, gate_agree=0)
diff = []
for k in ks:
    i, r = I[k], R[k]
    fires = G.get(k, False) if G is not None else True
    if G is not None: c['gate_agree'] += fires == i['implant_wrote']
    c['first_ok'] += i['first_ok'] == r['first_ok']
    if fires:
        rf, rok = final(r), r['ok']
        ref_head = r['out'].rstrip() + '\n' + PH
    else:
        rf, rok = r['pred'], r['first_ok']
        ref_head = r['out']
    imp_head = i['out'] + i.get('forced', '')
    c['first_text'] += imp_head == ref_head
    if fires and i['implant_wrote'] and ('rt_text' in r or 'out2' in r) and ('rt_text' in r or 'out2' in i):
        c['rethink_compared'] += 1
        same = i.get('rt_text') == r.get('rt_text', r.get('out2')) if 'rt_text' in r else i.get('out2') == r.get('out2')
        c['rethink_text'] += same
        if not same and imp_head == ref_head: diff.append(k)
    c['final_pred'] += final(i) == rf if i['implant_wrote'] else i['pred'] == rf
    c['final_ok'] += i['ok'] == rok
ia = 100 * sum(I[k]['ok'] for k in ks) / len(ks)
ra = 100 * sum((R[k]['ok'] if (G is None or G.get(k, False)) else R[k]['first_ok']) for k in ks) / len(ks)
print(json.dumps(c), f'| accuracy implant {ia:.2f} / reference {ra:.2f}')
for k in diff[:a.show]:
    x, y = I[k].get('rt_text', I[k].get('out2', '')), R[k].get('rt_text', R[k].get('out2', ''))
    p = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
    print(f'  i={k}: rethink text diverges at char {p}: implant {x[max(0, p - 40):p + 40]!r} | reference {y[max(0, p - 40):p + 40]!r}')
