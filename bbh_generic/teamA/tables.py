"""Final tables: frozen base / FT100 no tools / FT100 + tools / FT100 age-price ablation + tools, per set.
python tables.py [DIR=res]   (python tables.py results  -> from the shipped compact files)"""
import json, sys
D = sys.argv[1] if len(sys.argv) > 1 else 'res'
T = ['track3', 'track5', 'track7', 'order3', 'order5', 'order7']


def row(f):
    try: d = json.load(open(f))['per_task']
    except Exception: return ' | '.join(['n/a'] * len(T))
    return ' | '.join(f"{d[t]:.1f}" if t in d else '-' for t in T)


print('| set | host | ' + ' | '.join(T) + ' |'); print('|---' * (len(T) + 2) + '|')
for s in ['dev', 'test', 'oos2', 'oos3', 'oos']:
    base = f'{D}/dev2_base_plain.json' if s == 'dev' else f'{D}/{s}_base_plain.json'
    for h, f in [('frozen base', base), ('FT100 no tools', f'{D}/{s}_ft100a_notools.json'), ('FT100 + tools', f'{D}/{s}_ft100a_tools.json'),
                 ('FT100 age/price-ablated + tools', f'{D}/{s}_ft100a_abl_tools.json')]:
        r = row(f)
        if not r.startswith('n/a'): print(f'| {s} | {h} | {r} |')
