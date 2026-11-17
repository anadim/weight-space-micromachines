import json
import os, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scores_ft100.json')))
for team in d:
    for c in d[team]:
        for b in ['test', 'oos', 'bbh']:
            x = d[team][c].get(b, {})
            if x: print(team, c, b, ' || '.join(f"{a}: " + ' '.join(f'{k.replace("tracking_shuffled_objects_","ts").replace("logical_deduction_","ld")}={v}' for k, v in x[a]['per_task'].items()) for a in x))
