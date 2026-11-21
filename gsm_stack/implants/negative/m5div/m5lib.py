"""M5 (first-step diversity) analysis helpers: load serial-stack rows (fe format = lane wpr, or gen format = aw_hook) into
per-attempt records with the attempt's FIRST segment tokens and its register answer; voter replica; first-step keys."""
import os, re, sys, json, gzip, math
WS = os.path.expanduser(os.environ.get('WSM_ROOT', '.') + '/gsm_stack/implants')
sys.path.insert(0, os.path.join(WS, 'atlas'))
import parse as AP                      # Tok (pure-python decode), Q_IDS

TK = AP.Tok()
WAITS = AP.Q_IDS['wait']                # [no-\n variant, '\n'+WAIT]
W0 = WAITS[0]
REPS = AP.Q_IDS['rep'][0]
PAD = 12


def sym_val(s):
    t = ''.join('0123456789-.'[x] for x in s if x != PAD)
    try: return float(t)
    except ValueError: return None


def gold_f(g):
    return float(str(g).replace(',', ''))


def find_waits(T):
    """start index of every wait phrase (the 8-token core, '\\n' variant counted from the core) -> list of (start_core, end)."""
    out = []; n = len(W0); i = 0
    while i <= len(T) - n:
        if T[i] == W0[0] and T[i:i + n] == W0: out.append((i, i + n)); i += n
        else: i += 1
    return out


def split_attempts(T, wf=None):
    """[(start, end)] token spans of the FIRST segment of each attempt (attempt k>=1 starts after its wait phrase).
    gen format: wf bit 8 = queue start; else the wait phrase is searched."""
    if wf is not None:
        st = [i for i, w in enumerate(wf) if w & 8]
        bounds = []
        for i in st:
            if T[i:i + len(W0)] == W0: bounds.append((i, i + len(W0)))
            elif T[i] == 198 and T[i + 1:i + 1 + len(W0)] == W0: bounds.append((i, i + 1 + len(W0)))
            else: bounds.append((i, None))          # repair / final queue
    else:
        bounds = [(s - (1 if s > 0 and T[s - 1] == 198 else 0), e) for s, e in find_waits(T)]
        r = [i for i in range(len(T) - 3) if T[i:i + 4] == REPS]
        bounds += [(i, None) for i in r]
        bounds.sort()
    atts = []; cur = 0
    for s, e in bounds:
        if e is not None:
            atts.append((cur, s)); cur = e
        else:
            atts.append((cur, s)); cur = None
            # the rest until the next wait belongs to a repair / final: skip
    if cur is not None: atts.append((cur, len(T)))
    # keep only spans that begin at stream start or at a wait end (drop repair/final spans)
    starts = {0} | {e for _, e in bounds if e is not None}
    return [(a, b) for a, b in atts if a in starts]


def first_line(ids):
    """token ids of the first non-empty line (through its '\\n' token)."""
    out = []; seen = False
    for t in ids:
        s = TK.ts(t); out.append(t)
        if s.strip(): seen = True
        if '\n' in s and seen: break
    return out


NUMRE = re.compile(r'\d+(?:[.,]\d+)*')


def first_eq(txt):
    """normalized LHS of the first '=' in the attempt: numbers and operators only, e.g. '60/15'."""
    i = txt.find('=')
    if i < 0: return None
    ln = txt[:i].split('\n')[-1]
    toks = re.findall(r'\d+(?:[.,]\d+)*|[+\-*/x×÷]', ln)
    return ''.join(t.replace('×', '*').replace('x', '*').replace('÷', '/').replace(',', '') for t in toks) or None


def first_nums(txt, k=2):
    return tuple(m.group(0).replace(',', '') for m in NUMRE.finditer(txt))[:k]


def load_rows(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        return [json.loads(l) for l in f]


def attempts_of_row(r, fmt):
    """-> list of dict(k, ids (first segment), fl (first-line ids), val, ok_reg, bad, lg, right)"""
    T = r['tokens']; wf = r.get('wf') if fmt == 'gen' else None
    spans = split_attempts(T, wf)
    natt = r['natt']
    if fmt == 'gen':
        vals = [r['val'][i] if r['rok'][i] else None for i in range(len(r['rok']))]
        syms = None
    else:
        vals = [sym_val(s) if ok else None for s, ok in zip(r['sym'], r['rok'])]
    g = gold_f(r['gold'])
    out = []
    for k in range(min(natt, len(spans))):
        a, b = spans[k]; ids = T[a:b]
        v = vals[k] if k < len(vals) else None
        out.append(dict(k=k, ids=ids, fl=first_line(ids), val=v, ok=bool(r['rok'][k]), bad=bool(r['bad'][k]), lg=float(r['lg'][k]),
                        right=v is not None and abs(v - g) <= 1e-6 * max(1, abs(g))))
    return out, len(spans)


def vote(atts, san=True):
    """replica of comp_serial._vote (vmode 1): returns the winning attempt dict or None."""
    ok = [a for a in atts if a['ok']]
    if not ok: return None
    sane = [a for a in ok if not a['bad']]
    keep = sane if (san and sane) else ok
    sg = lambda x: 1 / (1 + math.exp(-x))
    best = None; bs = None
    for i, a in enumerate(keep):
        sc = sum(sg(b['lg']) for b in keep if b['val'] == a['val'])
        if bs is None or sc > bs + 1e-6: best, bs = a, sc
    return best


def load_set(src, st, seeds):
    """src = 'fe' (lane wpr rung-1 rows) or a gen-format dir.  -> {seed: [ (row, attempts) ] }"""
    D = {}
    for s in seeds:
        if src == 'fe':
            p = f'{WS}/implants/wpr/fe/out/fe/{st}_wpr_s{s}.jsonl'; fmt = 'fe'
        else:
            import glob
            c = sorted(glob.glob(f'{src.replace("{st}", st)}/{st}_N8_*seed{s}.jsonl.gz')); assert len(c) == 1, (src, st, s, c); p = c[0]; fmt = 'gen'
        rows = load_rows(p)
        D[s] = [(r, attempts_of_row(r, fmt)[0]) for r in rows]
    return D
