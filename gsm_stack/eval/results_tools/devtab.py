"""Dev curves of the RL-stack arms from steps.jsonl copies: python3 devtab.py ARM=path/steps.jsonl ..."""
import json, sys
for spec in sys.argv[1:]:
    arm, p = spec.split('=', 1)
    d = [json.loads(l)['dev'] for l in open(p) if '"dev"' in l]
    d = [x for x in d if isinstance(x, dict)]
    if not d: print(arm, 'no dev'); continue
    ks = [k for k in ('dev_vote8', 'dev2_vote8', 'dev_vote4', 'dev2_vote4', 'dev_vote1', 'dev2_vote1', 'dev_acc', 'dev2_acc') if k in d[0]]
    print(f'\n{arm}: u | ' + ' | '.join(ks) + ' | allsame8 | att8')
    for x in d:
        print(f"  {x['u']:>3} | " + ' | '.join(f"{x.get(k, float('nan')):.2f}" for k in ks)
              + f" | {x.get('dev_allsame8', float('nan')):.1f} | {(x.get('dev_attempt_acc') or [None])[-1]}")
