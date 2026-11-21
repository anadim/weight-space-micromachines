"""-> tables/valtok_{set}.jsonl.gz: per attempt of a mixed item, token offsets (seg_tokens) of computed-value steps
(kind 'eq': '... = v') and stated-number steps, from the atlas step parser (controls for divprobe.py)."""
import sys, os, gzip, json, collections
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f'{H}/../atlas')
from parse import Tok
from steps import question_numbers
from diverge import Att
tk = Tok()
for S in sys.argv[1:] or ['dev', 'dev2']:
    rows = [json.loads(l) for l in gzip.open(f'{H}/tables/attempts_{S}.jsonl.gz', 'rt')]
    by = collections.defaultdict(list)
    for r in rows: by[r['idx']].append(r)
    n = 0
    with gzip.open(f'{H}/tables/valtok_{S}.jsonl.gz', 'wt') as f:
        for i, rs in by.items():
            if not (0 < sum(r['ok'] for r in rs) < len(rs)): continue
            Q = question_numbers(rs[0]['q'])
            for r in rs:
                if isinstance(r['seg_tokens'], str): r['seg_tokens'] = json.loads(r['seg_tokens'])
                at = Att(r, tk, Q)
                f.write(json.dumps(dict(gid=r['gid'], eq=[s['tok'] for s in at.steps if s['kind'] == 'eq'],
                                        st=[s['tok'] for s in at.steps if s['kind'] == 'stated'])) + '\n'); n += 1
    print(S, n)
