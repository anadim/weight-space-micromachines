"""Arm R after RL (H100 machine traces): how does the host stop the deranged calculator from hurting it?"""
import io, json, re, contextlib, os
from collections import Counter
with contextlib.redirect_stdout(io.StringIO()):
    from analyze import annotate
    from fmt_probe import feats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traces')
CF = [('h_base_off', 'base, no implant (H100)'), ('h_base_on', 'base + deranged calc'), ('aR0_on', 'arm R s0 RL + deranged calc'),
      ('aR0_off', 'arm R s0 RL, implant off'), ('aR0calc_on', 'arm R s0 RL + CORRECT calc'), ('aR1_on', 'arm R s1 RL + deranged calc'),
      ('aR1_off', 'arm R s1 RL, implant off')]
T, A = {}, {}
for k, _ in CF:
    p = f'{D}/{k}.jsonl'
    if os.path.exists(p): T[k] = [json.loads(l) for l in open(p)]; A[k] = [annotate(r) for r in T[k]]
N = 800
def expr_form(q):
    f = feats(q); return f
S = {}
print('| config | acc | GSM | scaled | calls/ans | eqs/ans | called eqs | unit-form eqs | bare single-op eqs/ans | lines w/o "=" | "=" signs/ans | verbal ops/ans | comma nums | tok/ans | wrong-eq rate |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, lab in CF:
    if k not in T: continue
    a, t = A[k], T[k]
    eqs = [q for x in a for q in x['eqs']]
    F = [feats(q) for q in eqs]
    bare = sum(1 for q, f in zip(eqs, F) if not f['units'] and len(q['ops']) == 1)
    body = [r['text'].split('####')[0] for r in t]
    noeq = sum(sum('=' not in l for l in b.split('\n') if l.strip()) for b in body)
    neq = sum(b.count('=') for b in body)
    commas = sum(len(re.findall(r'\d,\d{3}', b)) for b in body)
    row = dict(acc=100*sum(r['ok'] for r in t)/N, gsm=100*sum(r['ok'] for r in t if r['kind']=='gsm')/400,
               scaled=100*sum(r['ok'] for r in t if r['kind']=='scaled')/400, calls=sum(len(x['calls']) for x in a)/N,
               eqs=len(eqs)/N, called=100*sum(q['called'] for q in eqs)/max(1,len(eqs)), units=100*sum(f['units'] for f in F)/max(1,len(eqs)),
               bare=bare/N, noeq=noeq/N, eqsigns=neq/N, verbal=sum(x['vwords'] for x in a)/N, commas=commas/N,
               tok=sum(x['ntok'] for x in a)/N, err=100*sum(not q['ok'] for q in eqs)/max(1,len(eqs)))
    S[k] = row
    print(f"| {lab} | {row['acc']:.1f} | {row['gsm']:.1f} | {row['scaled']:.1f} | {row['calls']:.2f} | {row['eqs']:.2f} | {row['called']:.1f}% | {row['units']:.1f}% | {row['bare']:.2f} | {row['noeq']:.2f} | {row['eqsigns']:.2f} | {row['verbal']:.2f} | {row['commas']:.2f} | {row['tok']:.0f} | {row['err']:.1f}% |")
# where do the remaining calls in aR land, and is the written (wrong) result copied?
for k in ('h_base_on', 'aR0_on', 'aR1_on'):
    if k not in A: continue
    calls = [c for x in A[k] for c in x['calls']]
    linked = [c for c in calls if c['eq']]
    print(k, 'calls', len(calls), 'linked', len(linked), 'result correct when called %.1f%%' % (100*sum(c['eq']['ok'] for c in linked)/max(1,len(linked))))
    # unlinked burst contexts
    ctx = Counter()
    for r, x in zip(T[k], A[k]):
        for c in x['calls']:
            if not c['eq']: ctx[r['text'][max(0,c['start']-25):c['start']+10]] += 1
    print('   unlinked samples', list(ctx)[:5])
# trigger-ignoring forms of the "a op b" step in aR: classify every equation the trigger missed
for k in ('h_base_on', 'aR0_on'):
    if k not in A: continue
    eqs = [q for x in A[k] for q in x['eqs'] if not q['called']]
    c = Counter()
    for q in eqs:
        f = feats(q)
        c['units between operands' if f['units'] else 'multi-op chain' if len(q['ops'])>1 else 'comma number' if f['comma'] else 'parens' if f['paren'] else 'other'] += 1
    print(k, 'uncalled eqs by form', dict(c))
# per-problem paired: base_on vs aR0_on, what replaced "a op b =" ?
json.dump(S, open('stats_aR.json', 'w'), indent=1)
