"""Job input: dev/dev2 mixed attempts (probe/branch jobs, with fork lists) + always-right items' seed-7 attempts (atlas)."""
import gzip, json, os
WS = os.path.expanduser('gsm_stack/implants'); H = os.path.dirname(os.path.abspath(__file__))
rd = lambda fn: [json.loads(l) for l in gzip.open(fn, 'rt')]
for S in ['dev', 'dev2']:
    J = rd(f'{WS}/probe/branch/job/tables/jobs_{S}.jsonl.gz')
    for j in J: j['cls'] = 'mixed'
    I = {r['idx']: r for r in rd(f'{WS}/probe/tables/items_{S}.jsonl.gz')}
    A = [a for a in rd(f'{WS}/probe/tables/attempts_{S}.jsonl.gz') if I[a['idx']]['class'] == 'always' and a['seed'] == 7]
    for a in A:
        J.append(dict(gid=a['gid'], idx=a['idx'], q=a['q'], seg_tokens=a['seg_tokens'], seg_wf=a['seg_wf'], ok=int(bool(a['ok'])),
                      plen=a['phrase_len'], b=-1, forks=[], ctrl=True, seg_logp=a['seg_logp'], cls='always'))
    with gzip.open(f'{H}/job/tables/jobs_{S}.jsonl.gz', 'wt') as f:
        for j in J: f.write(json.dumps(j) + '\n')
    print(S, len(J), sum(j['cls'] == 'always' for j in J), sum(len(j['seg_tokens']) for j in J))
