"""Aggregate eyes seed sweep: mean +/- std of test macro per (n, arm). Reads summary.txt."""
import sys, re, statistics, json, os
f=sys.argv[1] if len(sys.argv)>1 else 'evidence/eyes_sweep/results/summary.txt'
D={}
for line in open(f):
    m=re.match(r'n=(\d+) arm=(\w+) seed=(\d+) test_macro=([\d.]+)',line.strip())
    if m:
        n,arm,seed,mac=int(m[1]),m[2],int(m[3]),float(m[4]); D.setdefault((n,arm),[]).append((seed,mac))
rep={}
for (n,arm),vals in sorted(D.items()):
    macs=[v for _,v in sorted(vals)]
    mean=statistics.mean(macs); sd=statistics.stdev(macs) if len(macs)>1 else 0.0
    rep[f'n{n}_{arm}']={'seeds':[s for s,_ in sorted(vals)],'macros':macs,'mean':round(mean,2),'std':round(sd,2),'min':min(macs),'max':max(macs)}
    print(f'n={n:2d} {arm:6s} seeds={[s for s,_ in sorted(vals)]} macros={[round(x,1) for x in macs]} -> {mean:.2f} +/- {sd:.2f}')
json.dump(rep,open(os.path.join(os.path.dirname(f),'eyes_seeds_summary.json'),'w'),indent=1)
print('wrote eyes_seeds_summary.json')
