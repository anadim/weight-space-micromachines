import json, sys, glob, os
O = sys.argv[1]
for f in sorted(glob.glob(f'{O}/*.score.json')):
    n = os.path.basename(f)[:-11]; R = json.load(open(f))['rows']
    ps = 100 * sum(all(r['strict']) for r in R) / len(R); pl = 100 * sum(all(r['loose']) for r in R) / len(R)
    ins = [x for r in R for x in r['strict']]; il = 100 * sum(ins) / len(ins)
    ntok = sum(r['ntok'] for r in R) / len(R); fin = 100 * sum(r['finish'] == 'stop' for r in R) / len(R)
    lat = [r.get('circ', {}) for r in R]; anyl = 100 * sum(any(k.startswith('L.') for k in c) for c in lat) / len(R)
    fires = sum(c.get('fires', 0) > 0 for c in lat)
    print(f'{n:18s} n={len(R)} prompt_strict={ps:5.1f} inst_strict={il:5.1f} prompt_loose={pl:5.1f} tok/q={ntok:6.0f} stop%={fin:5.1f} latch%={anyl:5.1f} fired_rows={fires}')
