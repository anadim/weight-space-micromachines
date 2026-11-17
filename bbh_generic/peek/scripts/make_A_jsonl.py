"""Build Team-A-format jsonl (fields tool/input/id/task/target) from the official BBH json files. No inspection of items."""
import json, sys
D = sys.argv[1]; out = sys.argv[2]
T = {'tracking_shuffled_objects_three_objects': 'track', 'tracking_shuffled_objects_five_objects': 'track',
     'tracking_shuffled_objects_seven_objects': 'track', 'logical_deduction_three_objects': 'order',
     'logical_deduction_five_objects': 'order', 'logical_deduction_seven_objects': 'order'}
with open(out, 'w') as f:
    for t, tool in T.items():
        for i, e in enumerate(json.load(open(f'{D}/{t}.json'))['examples']):
            f.write(json.dumps({'id': f'{t}_{i}', 'tool': tool, 'task': t, 'input': e['input'], 'target': e['target']}) + '\n')
