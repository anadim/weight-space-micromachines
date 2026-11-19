"""v2 training union (432 rows): the A (102) and B (100) rows of train_union.jsonl unchanged + c_passive_v2.jsonl (Team C's
100 rows in passive form, make_v2_rows.py) + nocall_v2.jsonl (100 base-self-distilled no-tool rows, make_nocall.py) +
multi_v2.jsonl (30 multi-tool rows, make_v2_rows.py); shuffled with random.Random(2).
python make_union_v2.py DATA_DIR OUT.jsonl"""
import json, random, sys, os
D, out = sys.argv[1], sys.argv[2]
L = lambda f: [l for l in open(os.path.join(D, f))]
rows = [l for l in L('train_union.jsonl') if json.loads(l)['team'] in ('A', 'B')]
rows += L('c_passive_v2.jsonl') + L('nocall_v2.jsonl') + L('multi_v2.jsonl')
random.Random(2).shuffle(rows)
open(out, 'w').writelines(r if r.endswith('\n') else r + '\n' for r in rows)
