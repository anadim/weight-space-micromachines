"""[41] dev-only sweep table: dev acc (800 = 400 GSM8K-train held-out + 400 scaled) at u100 for every sweep run.
T at host lr 1e-6 = T_s* runs at u100 (same schedule). Reads the dev records of tt/res/train_steps.jsonl.gz (B200 runs + the A100 sweep
replicates; = the dev lines of each run's train log).  python3 ana_sweep.py"""
import gzip, json, os, re, statistics as S
RES = os.environ.get('C36_RES', os.path.dirname(os.path.abspath(__file__)) + '/tt/res')
STEPS = {}   # run -> steps.jsonl records in order (tt/res/train_steps.jsonl.gz, built by mk_res.py)
for _r in map(json.loads, gzip.open(f'{RES}/train_steps.jsonl.gz', 'rt')):
    STEPS.setdefault(_r.pop('run'), []).append(_r)


def dev_at(nm, u=100):
    for r in STEPS[nm]:
        if 'dev' in r and r['dev'].get('u') == u: return r['dev']
    return None


BEST = 'll1e-2_f0.05_r'      # [41] best I config at host lr 1e-6 (I* = BEST crossed with host lr)
rows = {}
for nm in sorted(STEPS, key=lambda n: (STEPS[n][0]['gpu'] == 'A100', n)):   # B200 runs first, then the A100 replicates
    f = nm
    m = re.fullmatch(r'T_s(\d+)', nm) or re.fullmatch(r'swT_lr([0-9e.-]+)_s(\d+)', nm) or re.fullmatch(r'swI_(.+)_s(\d+)', nm) \
        or re.fullmatch(r'swIx_lr([0-9e.-]+)_s(\d+)', nm)
    if not m: continue
    d = dev_at(f)
    if d is None: continue
    cfg = 'T lr1e-6' if nm.startswith('T_') else ('T lr' + m.group(1) if nm.startswith('swT') else
                                                  (f'I* lr{m.group(1)}' if nm.startswith('swIx') else 'I ' + m.group(1)))
    if cfg == 'I ' + BEST: rows.setdefault('I* lr1e-6', []).append((int(m.groups()[-1]), dev_at(f)))
    rows.setdefault(cfg, []).append((int(m.groups()[-1]), d))
rows = {c: list({k: d for k, d in reversed(v)}.items())[::-1] for c, v in rows.items()}   # one run per seed
print(f'| config | n seeds | dev acc (mean; per seed) | dev GSM | dev scaled | admit | written/q |')
print('|---|---|---|---|---|---|---|')
for cfg, v in sorted(rows.items(), key=lambda kv: -S.mean(d['dev_acc'] for _, d in kv[1])):
    v.sort(key=lambda x: x[0])
    acc = [d['dev_acc'] for _, d in v]
    g = S.mean(d['dev_gsm'] for _, d in v); sc = S.mean(d['dev_scaled'] for _, d in v)
    adm = S.mean(d.get('dev_admit', float('nan')) for _, d in v); wr = S.mean(d.get('dev_written', float('nan')) for _, d in v)
    print(f"| {cfg} | {len(v)} | **{S.mean(acc):.2f}** ({' / '.join(f's{s} {d['dev_acc']:.1f}' for s, d in v)}) | {g:.2f} | {sc:.2f} | {adm:.3f} | {wr:.1f} |")
