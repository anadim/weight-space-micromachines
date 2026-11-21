"""Rung-1 never items (0 right over all attempts) x the S3 never/near-never hand labels (atlas/data/never_labels.json)."""
import sys, json, gzip, collections
d = sys.argv[1]
lab = {(x['set'], x['idx']): x for x in json.load(open('atlas/data/never_labels.json'))}
for s in ('dev', 'dev2'):
    I = [json.loads(l) for l in gzip.open(f'{d}/items_{s}.jsonl.gz', 'rt')]
    nv = [r for r in I if r['n_right'] == 0]
    c = collections.Counter(lab[(s, r['idx'])]['category'] if (s, r['idx']) in lab else 'unlabelled (S3 n_right>4)' for r in nv)
    s3n = [k for k, x in lab.items() if k[0] == s and x['n_right'] == 0]
    now = {r['idx']: r for r in I}
    rescued = [k for k in s3n if now[k[1]]['n_right'] > 0]
    print(f'{s}: rung-1 never {len(nv)}; S3 never {len(s3n)} -> now some right {len(rescued)}; by S3 label:', dict(c.most_common()))
