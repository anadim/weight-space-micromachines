"""(f) §4.2 IFEval: (1) instructions per prompt (P3-4) and (2) share of generated tokens written by the 14-rule implant (P3-5).
Input: ifeval/transfer/results/reduced/q06_ifeval_{imp,plain}.json (R§5.1 run: Qwen3-0.6B chat, greedy, 57.86 -> 72.09).
circ.fires = number of decode steps on which the circuit's emitter was active and wrote a forced-token vector at the
output (heldout/circuit.py:783 `self.fires += active`; one forced token per active step). ffail = active steps where the
forced token still lost the argmax (diagnostic). So fires - ffail = tokens the implant chose instead of the model.
python3 checks/sourced/ifeval_42.py"""
import json, os, ast, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get('WSM_ROOT') or os.path.join(HERE, '..', '..')
D = f'{ROOT}/ifeval/transfer/results/reduced'
L = lambda f: {r['key']: r for r in json.load(open(f'{D}/{f}'))['rows']}
I, P = L('q06_ifeval_imp.json'), L('q06_ifeval_plain.json')
pv = lambda v: ast.literal_eval(v) if isinstance(v, str) else v
K = sorted(I, key=int); n = len(K)
ni = [len(pv(I[k]['ids'])) for k in K]
h = collections.Counter(ni)
print(f'## instructions per prompt (n = {n})\n\n| # instructions | prompts | % |\n|---|---|---|')
for c in sorted(h): print(f'| {c} | {h[c]} | {100*h[c]/n:.1f} |')
print(f'\nmean {sum(ni)/n:.2f}; >= 2: {sum(x >= 2 for x in ni)} ({100*sum(x >= 2 for x in ni)/n:.1f}%); 2-3: {sum(2 <= x <= 3 for x in ni)} '
      f'({100*sum(2 <= x <= 3 for x in ni)/n:.1f}%); total instructions {sum(ni)}')
c = {k: pv(I[k]['circ']) for k in K}
fi = {k: int(c[k]['fires']) for k in K}; ff = {k: int(c[k]['ffail']) for k in K}; nt = {k: int(pv(I[k]['ntok'])) for k in K}
acc = lambda R: 100 * sum(all(pv(R[k]['strict'])) for k in K) / n
F = [k for k in K if fi[k] > 0]
T, Tf = sum(nt.values()), sum(nt[k] for k in F)
W, Wf = sum(fi.values()), sum(ff.values())
print(f'\n## implant-written tokens (prompt-strict plain {acc(P):.2f} -> implant {acc(I):.2f})\n')
print('| scope | prompts | generated tokens | implant writes (fires) | forced token lost (ffail) | implant share | model share |\n|---|---|---|---|---|---|---|')
for lab, ks in [('all 541', K), ('prompts where the implant wrote', F)]:
    t = sum(nt[k] for k in ks); w = sum(fi[k] for k in ks); x = sum(ff[k] for k in ks)
    print(f'| {lab} | {len(ks)} | {t} | {w} | {x} | {100*(w-x)/t:.1f}% (≤ {100*w/t:.1f}%) | {100-100*w/t:.1f}% (≥) |')
sil = [k for k in K if fi[k] == 0]
byte = sum(I[k]['resp_sha256'] == P[k]['resp_sha256'] for k in sil)
print(f'\nsilent prompts (fires = 0): {len(sil)}, byte-identical to plain: {byte}')
sh = sorted(100 * fi[k] / max(nt[k], 1) for k in F)
q = lambda p: sh[min(len(sh) - 1, int(p * len(sh)))]
over = [k for k in F if fi[k] > nt[k]]
print(f'per fired prompt fires/ntok: median {q(.5):.1f}%, p90 {q(.9):.1f}%; prompts with fires > ntok: {len(over)} '
      f'(all case rules: {sum(any(x.startswith("change_case") or x == "detectable_format:title" for x in pv(I[k]["ids"])) for k in over)}/{len(over)}; '
      f'{sum(fi[k] for k in over)} fires on {sum(nt[k] for k in over)} tokens)')
# run_t.py keeps stepping the circuit on finished rows (fed EOS) until the batch ends, so `fires` also counts steps after
# a row's stop. min(fires, ntok) per prompt is an upper bound on implant-written tokens inside the response.
cap = {k: min(fi[k], nt[k]) for k in K}
print('\n| scope | generated tokens | implant-written tokens, upper bound sum(min(fires, ntok)) | implant share ≤ | model share ≥ |\n|---|---|---|---|---|')
for lab, ks in [('all 541', K), ('217 prompts where the implant wrote', F), ('the same, without the 17 fires>ntok prompts', [k for k in F if k not in over])]:
    t = sum(nt[k] for k in ks); w = sum(cap[k] for k in ks)
    print(f'| {lab} | {t} | {w} | {100*w/t:.1f}% | {100-100*w/t:.1f}% |')
