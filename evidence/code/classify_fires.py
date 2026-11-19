"""Inspect every item-2 fire: does the 12-token context before it contain an 'a op b =' equation?
python classify_fires.py ../item2/fires.json"""
import json, re, sys
from collections import Counter
F = json.load(open(sys.argv[1]))
eq = re.compile(r'[\d)]\s*[-+*/x×]\s*\$?\(?[\d.,]+\)?\s*=')
print('fires:', len(F), '| passages with a fire:', len({f['passage'] for f in F}))
print('by source:', Counter(f['source'] for f in F))
print('fire token:', Counter(f['token'].strip() or '<space>' for f in F).most_common(8))
out = [f for f in F if not eq.search(f['context'])]
print('fires with no "a op b =" in context:', len(out))
for f in out: print('  ', repr(f['context']), repr(f['token']), f['machines'])
