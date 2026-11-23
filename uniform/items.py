"""Reduced per-item GSM8K / GSM-Hard / GSM-Plus eval rows of the uniform-row configs (REPRODUCE §17, §18).

The eval harness (gsm_stack/eval/fe.sh, uniform/final_eval/fe.sh) writes one jsonl per config, <SET>_<NAME>.jsonl, with
the generated tokens. The release keeps, per item, only i, gold, last (extracted answer), ok, ntok (generated tokens),
natt (attempts, serial-stack rows) and, for the configs the per-attempt analyses read, att (each attempt's answer string or
null). They are stored column-wise, one gzipped bundle per set:

    <dir>/<SET>.items.jsonl.gz   line 1: {"set", "i": [...], "gold": [...]}      (the set's gold answers)
                                 then one line per config: {"cfg": NAME, "i": [...], "last": [...], "ok": "0110...",
                                                            "ntok": [...], ["natt": [...]], ["att": [[...], ...]]}

load(SET, NAME) returns the rows in the original order as dicts {i, gold, last, ok, ntok[, natt, att]}, i.e. what the
analysis scripts read from the full jsonl. Directories searched, first hit wins:
  1. $FE_ROWS (':'-separated) holding full <SET>_<NAME>.jsonl files from your own runs (e.g. uniform/final_eval/out/fe, the fe.sh default);
  2. uniform/final_eval/items (0.6B) and uniform/rl28/gsm8k/items (1.7B), or $FE_ITEMS (':'-separated) if set.

    python3 uniform/items.py pack OUTDIR [--att SET_NAME,...] FILE.jsonl ...   # bundle full rows (merges into OUTDIR)
    python3 uniform/items.py ls [SET]                                        # list the configs in the bundles
"""
import gzip, json, os, sys

U = os.path.dirname(os.path.abspath(__file__))
CH = '0123456789-.'
_B = {}


def _dirs():
    e = os.environ.get('FE_ITEMS')
    return e.split(':') if e else [os.path.join(U, 'final_eval', 'items'), os.path.join(U, 'rl28', 'gsm8k', 'items')]


def _bundle(d, st):
    k = (d, st)
    if k not in _B:
        p = os.path.join(d, f'{st}.items.jsonl.gz')
        B = {}
        if os.path.exists(p):
            with gzip.open(p, 'rt') as f:
                h = json.loads(f.readline()); gold = dict(zip(h['i'], h['gold']))
                for l in f:
                    c = json.loads(l); c['_gold'] = gold; B[c['cfg']] = c
        _B[k] = B
    return _B[k]


def _raw(st, n):
    for d in filter(None, os.environ.get('FE_ROWS', '').split(':')):
        p = os.path.join(d, f'{st}_{n}.jsonl')
        if os.path.exists(p): return p
    return None


def have(st, n):
    return _raw(st, n) is not None or any(n in _bundle(d, st) for d in _dirs())


def names(st):
    """config names available for SET (bundles + $FE_ROWS)."""
    s = set()
    for d in _dirs(): s |= set(_bundle(d, st))
    for d in filter(None, os.environ.get('FE_ROWS', '').split(':')):
        if os.path.isdir(d): s |= {f[len(st) + 1:-6] for f in os.listdir(d) if f.startswith(st + '_') and f.endswith('.jsonl')}
    return sorted(s)


def att_of(r):
    """per-attempt answer strings (None = no answer) from a full row (sym / rok) or a reduced row (att)."""
    if 'att' in r: return r['att']
    return [''.join(CH[v] for v in r['sym'][t] if v < len(CH)) if r['rok'][t] else None for t in range(r['natt'])]


def load(st, n):
    p = _raw(st, n)
    if p: return [json.loads(l) for l in open(p)]
    for d in _dirs():
        c = _bundle(d, st).get(n)
        if c is None: continue
        g = c['_gold']; R = []
        for j, i in enumerate(c['i']):
            r = dict(i=i, gold=g[i], last=c['last'][j], ok=int(c['ok'][j]), ntok=c['ntok'][j])
            for k in ('natt', 'att'):
                if k in c: r[k] = c[k][j]
            R.append(r)
        return R
    raise FileNotFoundError(f'{st}_{n}: not in $FE_ROWS nor in the item bundles {_dirs()}')


def pack(out, files, att=()):
    os.makedirs(out, exist_ok=True)
    by = {}
    for f in files:
        b = os.path.basename(f)[:-len('.jsonl')]; st, n = b.split('_', 1)
        by.setdefault(st, {})[n] = f
    for st, cf in sorted(by.items()):
        p = os.path.join(out, f'{st}.items.jsonl.gz')
        old = {}
        if os.path.exists(p):
            with gzip.open(p, 'rt') as fh:
                h = json.loads(fh.readline()); gold = dict(zip(h['i'], h['gold']))
                old = {c['cfg']: c for c in map(json.loads, fh)}
        else:
            gold = {}
        for n, f in sorted(cf.items()):
            R = [json.loads(l) for l in open(f)]
            for r in R:
                if gold.setdefault(r['i'], r['gold']) != r['gold']: sys.exit(f'{f}: gold of item {r["i"]} differs')
            c = dict(cfg=n, i=[r['i'] for r in R], last=[r.get('last') for r in R], ok=''.join(str(int(bool(r['ok']))) for r in R),
                     ntok=[r['ntok'] for r in R])
            if all('natt' in r for r in R): c['natt'] = [r['natt'] for r in R]
            if f'{st}_{n}' in att: c['att'] = [att_of(r) for r in R]
            old[n] = c
        ids = sorted(gold)
        with gzip.open(p, 'wt', compresslevel=9) as fh:
            fh.write(json.dumps(dict(set=st, i=ids, gold=[gold[i] for i in ids]), separators=(',', ':')) + '\n')
            for n in sorted(old): fh.write(json.dumps(old[n], separators=(',', ':')) + '\n')
        print(p, len(old), 'configs', os.path.getsize(p), 'bytes')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['pack'] and len(a) >= 3:
        out, rest = a[1], a[2:]; att = ()
        if rest[0] == '--att': att = set(rest[1].split(',')); rest = rest[2:]
        pack(out, rest, att)
    elif a[:1] == ['ls']:
        for st in a[1:] or ('test', 'hard', 'gsmplus', 'dev'):
            print(st, len(names(st)), ' '.join(names(st)))
    else:
        sys.exit(__doc__)
