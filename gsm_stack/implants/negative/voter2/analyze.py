"""Compare v2 runs (job/out/*_v2.jsonl.gz) with the base stack (gen/out): prefix exactness, vote accuracy, fixes/breaks."""
import gzip, json, os, sys
B = os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants/gen/out'); V = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job/out')

def load(p): return {r['idx']: r for r in map(json.loads, gzip.open(p))}

def qstart(wf):
    return max(i for i, w in enumerate(wf) if w & 8)   # final queue line starts at the last token with the queue-start bit

def okv(r, p):
    try: return float(r['gold'].replace(',', '')) == r['val'][p]
    except Exception: return False

W = 128   # items per wave (B_max); the sampler seed is a global step counter, so a wave whose step count changes shifts later waves
print('| set | seed | prefix identical (all) | wave 0 | waves >0 | pick0==base pick (aligned rows) | base run acc | v2 run acc | fixes | breaks '
      '| same-stream base voter (pick0) acc | same-stream v2 acc | same-stream fixes/breaks | tok/q base | tok/q v2 |')
print('|' + '---|' * 15)
for st in sys.argv[1].split(',') if len(sys.argv) > 1 else ['dev', 'dev2']:
    for sd in sys.argv[2].split(',') if len(sys.argv) > 2 else ['7', '11']:
        f = f'{st}_N8_R1_v1_s1_m0_T1.0_seed{sd}'
        pv = f'{V}/{f}_v2.jsonl.gz'
        if not os.path.exists(pv): print(f'| {st} | {sd} | missing |'); continue
        b, v = load(f'{B}/{f}.jsonl.gz'), load(pv)
        ks = sorted(v); assert set(ks) == set(b), (len(ks), len(b))
        same = sum(b[k]['tokens'][:qstart(b[k]['wf'])] == v[k]['tokens'][:qstart(v[k]['wf'])] for k in ks)
        ab = sum(b[k]['ok'] for k in ks); av = sum(v[k]['ok'] for k in ks)
        fx = sum(v[k]['ok'] and not b[k]['ok'] for k in ks); br = sum(b[k]['ok'] and not v[k]['ok'] for k in ks)
        al = [k for k in ks if b[k]['tokens'][:qstart(b[k]['wf'])] == v[k]['tokens'][:qstart(v[k]['wf'])]]
        w0 = [k for k in ks if k < W]; s0 = sum(k in al for k in w0); s1 = len(al) - s0
        p0 = sum(v[k]['pick0'] == b[k]['pick'] for k in al)
        bad = sum((okv(v[k], v[k]['pick']) != bool(v[k]['ok'])) for k in ks)   # sanity: ok == (val[pick] == gold)
        o0 = sum(okv(v[k], v[k]['pick0']) for k in ks); sf = sum(v[k]['ok'] and not okv(v[k], v[k]['pick0']) for k in ks)
        sb = sum(okv(v[k], v[k]['pick0']) and not v[k]['ok'] for k in ks)
        tb = sum(b[k]['ntok'] for k in ks) / len(ks); tv = sum(v[k]['ntok'] for k in ks) / len(ks)
        n = len(ks)
        print(f'| {st} | {sd} | {same}/{n} | {s0}/{len(w0)} | {s1}/{n-len(w0)} | {p0}/{len(al)} | {100*ab/n:.2f} | {100*av/n:.2f} | {fx} | {br} '
              f'| {100*o0/n:.2f} | {100*av/n:.2f} | {sf}/{sb} | {tb:.1f} | {tv:.1f} |' + (f' ok-check mismatches {bad}' if bad else ''))
