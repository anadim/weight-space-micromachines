"""Drop generated prompts that equal / near-duplicate an IFEval test prompt (word 6-gram containment or char-5gram Jaccard),
and train prompts that equal a dev prompt. Writes gen/train.jsonl, gen/dev.jsonl and gen/filter_report.json."""
import json, re, hashlib
norm = lambda s: re.sub(r'\s+', ' ', s.lower()).strip()
def ch(s, n=5): s = norm(s); return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}
def w6(s): w = re.findall(r'[a-z0-9]+', s.lower()); return {' '.join(w[i:i + 6]) for i in range(max(1, len(w) - 5))}
IF = [json.loads(l) for l in open('ifdata/ifeval.jsonl')]
IFc = [ch(d['prompt']) for d in IF]; IFw = [w6(d['prompt']) for d in IF]; IFn = {norm(d['prompt']) for d in IF}
def worst(p):
    c, w = ch(p), w6(p); bj = bc = 0.0; kj = kc = None
    for d, a, b in zip(IF, IFc, IFw):
        j = len(c & a) / len(c | a)
        if j > bj: bj, kj = j, d['key']
        if w:
            cc = len(w & b) / len(w)
            if cc > bc: bc, kc = cc, d['key']
    return bj, kj, bc, kc
rep = {}
dev = [json.loads(l) for l in open('gen/dev_s201.jsonl')]; tr = [json.loads(l) for l in open('gen/train_s200.jsonl')]
devn = {norm(r['prompt']) for r in dev}
for name, rows in (('dev', dev), ('train', tr)):
    keep, drop, sims, seen = [], [], [], set()
    for r in rows:
        p = r['prompt']; j, kj, c, kc = worst(p); sims.append(j)
        bad = norm(p) in IFn or norm(p) in seen or j >= 0.5 or c >= 0.5 or (name == 'train' and norm(p) in devn)
        if not bad: seen.add(norm(p))
        (drop if bad else keep).append(dict(key=r['key'], jacc=round(j, 3), jkey=kj, w6=round(c, 3), wkey=kc) if bad else r)
    with open(f'gen/{name}.jsonl', 'w') as f:
        for r in keep: f.write(json.dumps(r) + '\n')
    sims.sort()
    rep[name] = dict(n_in=len(rows), n_keep=len(keep), n_drop=len(drop), dropped=drop[:50], jacc_max=sims[-1], jacc_p99=sims[int(.99 * len(sims))],
                     unique_prompts=len({norm(r['prompt']) for r in keep}),
                     sha256=hashlib.sha256(open(f'gen/{name}.jsonl', 'rb').read()).hexdigest())
    print(name, {k: v for k, v in rep[name].items() if k != 'dropped'}, flush=True)
json.dump(rep, open('gen/filter_report.json', 'w'), indent=1)
