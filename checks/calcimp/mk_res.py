"""Reduce the per-run outputs that job.sh copies into an out/ directory to the compact, text-free res/ form that the
analysis scripts (ana38.py, ana_tt.py, sel300.py, ana_sweep.py, links38.py) read.

  python3 mk_res.py --out tt/out --res tt/res [--evals REGEX] [--runs REGEX] [--bundles DIR]          (C36, this folder)
  python3 mk_res.py --out ../tooltext/out --res ../tooltext/res --evals '^(ev_|e37a_)' --runs '^a[12X]_s'   (C35)

Writes into RES:
  items.jsonl.gz       one line per eval dir <NAME>/ (fe_tool.py output): {"ev", "summary", "sets"}; summary = summary.json
                       without the tool_examples lists; sets["<set>_s<seed>"] = {"n", "ok": "0101...", "sum": {field: int}}
                       = per-item correctness in item order + the per-run sums of ntok / ntok_model / nforced / nwritten /
                       calls / calls_in_answer / parsed. No generated text, no predictions.
  configs.json         {run: config.json} for every train run (paths scrubbed), plus "gpu" (A100 iff the run came back in an
                       A100 bundle: run_dir outside the local runs root; fIa_s14/17 used A100 batching on a B200).
  train_steps.jsonl.gz one line per steps.jsonl record: {"run", "gpu", ...}. dev records are kept whole (rounded to 7
                       significant digits); per-update records keep STEP_KEYS (5 significant digits). The arm-I wiring
                       runs (fI, fI1, fIr, fIa, fItok, fItok1; read by links38.py and ../sourced) also keep STEP_KEYS_I,
                       the in-wire admit split / counts (ADMIT_KEYS) and the wiring norms (CI_KEYS). Sweep runs (sw*) keep
                       their dev records only.
--bundles DIR: extra train runs in self-contained bundle dirs DIR/<tag>_<RUN>/run/{steps.jsonl,config.json} (how the
A100 sweep replicates came back)."""
import argparse, glob, gzip, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--res', required=True)
ap.add_argument('--evals', default=r'^(ev_|eb_base)'); ap.add_argument('--runs', default=r'^(T_s|f[IT]|sw)')
ap.add_argument('--bundles', default=None)
a = ap.parse_args()
os.makedirs(a.res, exist_ok=True)
SUMS = ('ntok', 'ntok_model', 'nforced', 'nwritten', 'calls', 'calls_in_answer', 'parsed')
STEP_KEYS = ('u', 'reward', 'fmt', 'model_tok', 'calls', 'tool_calls', 'admit')
STEP_KEYS_I = STEP_KEYS + ('r_gsm', 'r_scaled', 'len', 'written', 'forced', 'ci_err', 'loss', 'clipfrac', 'grad_norm',
                           'ci_grad_norm')
ADMIT_KEYS = ('admit_d1_2', 'admit_d3_4', 'admit_d5p', 'admit_add', 'admit_sub', 'admit_mul', 'admit_div',
              'iw_n', 'iw_ratio', 'iw_clip', 'ci_calls')
CI_KEYS = ('writer_dW', 'writer_gate', 'cw7_B', 'cw14_B', 'cw21_B', 'cwfinal_B', 'cw7_out', 'cw14_out', 'cw21_out',
           'cwfinal_out', 'iw_b', 'iw_v')


def scrub(p):
    if not isinstance(p, str) or not p.startswith('/'): return p
    m = re.search(r'/(calcimp|tooltext)/runs/(.*)$', p)
    if m: return f'$WSM_DATA/{m[1]}/runs/{m[2]}'
    m = re.search(r'/[A-Za-z0-9]+_([^/]+)/out/(.*)$', p)
    if m: return f'a100_bundle_{m[1]}/out/{m[2]}'
    return '<path>/' + os.path.basename(p)


def rnd(x, sig):
    if isinstance(x, float): return float(f'{x:.{sig}g}')
    if isinstance(x, dict): return {k: rnd(v, sig) for k, v in x.items()}
    return x


# ---- eval dirs -> items.jsonl.gz
n_ev = 0
with gzip.open(f'{a.res}/items.jsonl.gz', 'wt') as f:
    for d in sorted(os.listdir(a.out)):
        p = f'{a.out}/{d}'
        if not re.search(a.evals, d) or not os.path.isfile(f'{p}/summary.json'): continue
        sm = json.load(open(f'{p}/summary.json'))
        summ = {k: ({kk: vv for kk, vv in v.items() if kk != 'tool_examples'} if isinstance(v, dict) else scrub(v))
                for k, v in sm.items()}
        sets = {}
        for fn in sorted(glob.glob(f'{p}/*_s*.jsonl')):
            key = os.path.basename(fn)[:-6]
            rows = [json.loads(l) for l in open(fn)]
            assert [r['i'] for r in rows] == list(range(len(rows))), fn
            sets[key] = dict(n=len(rows), ok=''.join(str(int(r['ok'])) for r in rows),
                             sum={k: sum(int(r[k]) for r in rows) for k in SUMS if k in rows[0]})
        f.write(json.dumps(dict(ev=d, summary=summ, sets=sets), separators=(',', ':')) + '\n'); n_ev += 1

# ---- train runs -> configs.json + train_steps.jsonl.gz
runs = [(os.path.basename(s)[:-len('.steps.jsonl')], s, s[:-len('.steps.jsonl')] + '.config.json')
        for s in sorted(glob.glob(f'{a.out}/*.steps.jsonl'))]
if a.bundles:
    for b in sorted(glob.glob(f'{a.bundles}/*_*/run/steps.jsonl')):
        bd = os.path.basename(os.path.dirname(os.path.dirname(b)))
        runs.append((bd.split('_', 1)[1], b, os.path.dirname(b) + '/config.json'))
runs = [r for r in runs if re.search(a.runs, r[0])]
cfgs = {}; n_rec = 0
with gzip.open(f'{a.res}/train_steps.jsonl.gz', 'wt') as f:
    for name, sp, cp in runs:
        if name in cfgs: continue                       # out/ first, then bundles
        c = json.load(open(cp)); c = {k: scrub(v) for k, v in c.items()}
        gpu = 'A100' if str(c.get('run_dir', '')).startswith('a100_bundle_') else 'B200'
        cfgs[name] = dict(c, gpu=gpu)
        wiring = name.startswith('fI') and not name.startswith('fI0')
        for l in open(sp):
            r = json.loads(l)
            if 'dev' in r:
                o = dict(dev=rnd({k: v for k, v in r['dev'].items() if k != 't'}, 7))
            elif name.startswith('sw') or 'u' not in r:
                continue
            else:
                o = {k: r[k] for k in (STEP_KEYS_I if wiring else STEP_KEYS) if k in r}
                if wiring:
                    o.update({k: r[k] for k in ADMIT_KEYS if k in r})
                    if 'ci' in r: o['ci'] = {k: r['ci'][k] for k in CI_KEYS if k in r['ci']}
                o = rnd(o, 5)
            f.write(json.dumps(dict(run=name, gpu=gpu, **o), separators=(',', ':')) + '\n'); n_rec += 1
json.dump(cfgs, open(f'{a.res}/configs.json', 'w'), indent=0, sort_keys=True)
print(f'{n_ev} eval dirs -> {a.res}/items.jsonl.gz; {len(cfgs)} train runs, {n_rec} step records -> {a.res}/train_steps.jsonl.gz, configs.json')
