"""Lane-1 rollouts (S3-serial stack, N=8 serial tries x 4 seeds + N1_R2 greedy refs) -> canonical atlas tables.

usage: python3 atlas/parse.py [--sets dev dev2] [--gen gen/out] [--out atlas/data] [--notes notes/atlas_counts.md]

Stream layout (gen/stk/serial_stack.py, `_act`): wf bit 8 marks the first token of every queue the writer emits, bit 2 every
queue token.  Queue kinds:  wait = ['\\n'] "Wait, let me solve it again.\\n" (new attempt),  rep = " We need to find" (S2F
repair inside the attempt; the failed segment is hidden),  final = ['\\n'] "####" + voted answer tokens + eos (only the
last one; a vote with no valid slot writes a bare eos).  Segment j = [queue start j, next queue start); segment 0 starts at
the stream start.  The final line is excluded from the last attempt.

Answer per attempt = the scorer's regex C.NUM on the attempt's final segment (first match), re-checked with a python port
of the register automaton (`_read` / commit in `_act`) and against the stored register (val/rok/bad/end per slot).
"""
import os, re, ast, sys, json, gzip, glob, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
COMMON = os.path.join(WS, 'gen', 'wiring', 'common.py')
TOKJ = os.path.join(HERE, 'data', 'qwen3_tokenizer.json')
WAIT = "Wait, let me solve it again.\n"
REP = " We need to find"
EOS_TXT = '<|endoftext|>'
# Qwen3 ids of the writer queues (serial_setup: q_wait = [T(WAIT), T('\n' + WAIT)], c_rep = T(" We need to find"))
Q_IDS = {'wait': [[14190, 11, 1077, 752, 11625, 432, 1549, 624], [198, 14190, 11, 1077, 752, 11625, 432, 1549, 624]],
         'rep': [[1205, 1184, 311, 1477]]}


def _load_scorer():
    """C.NUM and C.correct taken verbatim from gen/wiring/common.py (importing it needs numpy + the repo)."""
    src = open(COMMON).read(); tree = ast.parse(src); ns = {}
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == 'NUM' for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == 'correct')]
    assert len(keep) == 2, 'NUM / correct not found in common.py'
    exec(compile(ast.Module(body=keep, type_ignores=[]), COMMON, 'exec'), ns)
    return ns['NUM'], ns['correct']


NUM, correct = _load_scorer()


# ---------------------------------------------------------------- tokenizer (pure python byte-level decode)
def _bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Tok:
    """Qwen3 tokenizer decode (ByteLevel, no cleanup) = HF tok.decode; enc() is a greedy longest-match encoder (tests)."""
    def __init__(self, path=TOKJ):
        d = json.load(open(path)); u2b = _bytes_to_unicode()
        V = max(max(d['model']['vocab'].values()), max(a['id'] for a in d['added_tokens'])) + 1
        self.b = [b''] * V
        for s, i in d['model']['vocab'].items(): self.b[i] = bytes(u2b[c] for c in s)
        for a in d['added_tokens']: self.b[a['id']] = a['content'].encode()
        self.eos = next(a['id'] for a in d['added_tokens'] if a['content'] == EOS_TXT)
        self.s1 = [x.decode('utf-8', 'replace') for x in self.b]        # tok.decode([i]) (register token classes)
        self._enc = None

    def tb(self, i): return self.b[i] if 0 <= i < len(self.b) else b''
    def ts(self, i): return self.s1[i] if 0 <= i < len(self.s1) else ''
    def decode(self, ids): return b''.join(self.tb(i) for i in ids).decode('utf-8', 'replace')

    def enc(self, s):
        if self._enc is None:
            self._enc = {x: i for i, x in enumerate(self.b) if x and i != self.eos and i < 151643}
            self._ml = max(map(len, self._enc))
        bs, out, i = s.encode(), [], 0
        while i < len(bs):
            for L in range(min(self._ml, len(bs) - i), 0, -1):
                j = self._enc.get(bs[i:i + L])
                if j is not None: out.append(j); i += L; break
            else: raise ValueError(bs[i:i + 8])
        return out


# ---------------------------------------------------------------- register automaton (port of SerialEngine._read)
class Reg:
    def __init__(self, tk):
        self.tk = tk; self._cls = {}; self.reset()

    def cls(self, t):
        if t not in self._cls:
            s0 = self.tk.ts(t); s = s0.strip()
            nh = len(s) if (s and set(s) == {'#'}) else 0
            if s0 != '' and s == '': c = 1
            elif s == '-' and s0.lstrip() == '-': c = 2
            elif len(s0) == 1 and s0 in '0123456789': c = 3
            elif s0 == ',': c = 4
            elif s0 == '.': c = 5
            else: c = 0
            self._cls[t] = (c, int(s0) if c == 3 else 0, nh)
        return self._cls[t]

    def reset(self):
        self.ph = self.hc = self.an = 0; self.val = self.fsc = 0.0
        self.neg = self.fnz = self.hasd = False

    def read(self, t, live=True):
        """-> (start: '####' completed here, close: the number closed at this token)"""
        c, d, nh = self.cls(t); ph = self.ph
        self.hc = self.hc + nh if nh > 0 else 0
        start = ph == 0 and self.hc >= 4
        ws, mi, dg, cm, dt = c == 1, c == 2, c == 3, c == 4, c == 5
        nph = ph
        if ph == 1 and ws: nph = 1
        if ph == 1 and mi: nph = 2
        if ph in (1, 2) and (dg or cm): nph = 3
        if ph in (1, 2) and dt: nph = 4
        if ph == 3 and (dg or cm): nph = 3
        if ph == 3 and dt: nph = 4
        if ph == 4 and dg: nph = 5
        if ph == 5 and dg: nph = 5
        inph = 1 <= ph <= 5
        close = inph and nph == ph and not ((ph == 1 and ws) or (ph == 3 and (dg or cm)) or (ph == 5 and dg))
        if close: nph = 6
        acc = inph and not close
        if live:
            if acc and dg and ph in (1, 2, 3): self.val = self.val * 10 + d
            frd = acc and dg and ph in (4, 5)
            if frd: self.fsc = self.fsc / 10
            if self.fsc == 0: self.fsc = 1.0
            if frd: self.val += d * self.fsc; self.fnz |= d > 0
            self.hasd |= acc and dg
            self.neg |= acc and mi and ph == 1
            self.an += int(acc)
            self.ph = 1 if start else nph
        else:
            if self.fsc == 0: self.fsc = 1.0
        return live and start, live and close

    def commit(self):
        valid = 3 <= self.ph <= 6 and self.hasd
        bad = (not valid) or self.neg or self.fnz
        return valid, bad, (-self.val if self.neg else self.val)


# ---------------------------------------------------------------- stream parsing
def queues(tokens, wf, tk):
    """written queues: list of dict(start, n, kind, txt).  n = run of bit-2 tokens from a bit-8 token."""
    out = []
    for i, w in enumerate(wf):
        if not (w & 8): continue
        j = i + 1
        while j < len(wf) and (wf[j] & 2) and not (wf[j] & 8): j += 1
        txt = tk.decode(tokens[i:j])
        b = txt.replace(EOS_TXT, '').lstrip('\n')
        kind = ('wait' if b.startswith(WAIT.rstrip('\n')) else 'rep' if txt.startswith(REP)
                else 'final' if (b.startswith('####') or txt == EOS_TXT) else 'unk')
        n = j - i; c2first = False
        if kind == 'unk':                       # calc v2 overwrote the queue's first token (number open at a cap end)
            for kd, es in Q_IDS.items():
                if any(len(e) > 1 and tokens[i + 1:i + len(e)] == e[1:] for e in es): kind = kd; c2first = True
        for e in Q_IDS.get(kind, []):           # calc v2 forced tokens also carry bit 2: cut the run at the known phrase
            if tokens[i + c2first:i + len(e)] == e[c2first:]: n = len(e); txt = tk.decode(tokens[i:i + n]); break
        out.append(dict(start=i, n=n, kind=kind, txt=txt, c2first=c2first))
    return out


def line_spans(ids, tk, skip=0):
    """[s, e) token spans of lines (a line ends after a token containing '\\n'); spans inside the first `skip` tokens dropped."""
    sp, s = [], 0
    for i, t in enumerate(ids):
        if '\n' in tk.ts(t):
            sp.append([s, i + 1]); s = i + 1
    if s < len(ids): sp.append([s, len(ids)])
    return [x for x in sp if x[1] > skip and tk.decode(ids[x[0]:x[1]]).strip()]


def parse_stream(row, tk):
    """-> dict(attempts=[...], final=dict, anomalies=[...]).  Each attempt: list of segments (start, end, phrase_len)."""
    T, W = row['tokens'], row['wf']
    assert len(T) == len(W), (len(T), len(W))
    Qs = queues(T, W, tk); an = []
    for q in Qs:
        if q['kind'] == 'unk': an.append(f"unknown queue at {q['start']}: {q['txt']!r}")
        exp = Q_IDS.get(q['kind'])
        if q['c2first']: an.append(f"c2first: {q['kind']} queue at {q['start']} starts with a calc-v2 token: {q['txt']!r}")
        elif exp and T[q['start']:q['start'] + q['n']] not in exp: an.append(f"{q['kind']} queue ids differ at {q['start']}")
    fin = [q for q in Qs if q['kind'] == 'final']
    if len(fin) > 1: an.append('several final queues')
    if fin and fin[-1] is not Qs[-1]: an.append('final queue is not the last queue')
    if not fin: an.append('nofinal: stream ended without a final vote line')
    stop = fin[-1]['start'] if fin else len(T)
    segs = [dict(start=0, end=None, plen=0, kind='first')]
    for q in Qs:
        if q['start'] >= stop or q['kind'] not in ('wait', 'rep'): continue
        segs.append(dict(start=q['start'], end=None, plen=q['n'], kind=q['kind']))
    for a, b in zip(segs, segs[1:]): a['end'] = b['start']
    segs[-1]['end'] = stop
    atts = []
    for s in segs:
        if s['kind'] in ('first', 'wait'): atts.append([s])
        else: atts[-1].append(s)
    # register port over each segment (reader reset at every new segment; queue tokens except the last are not read)
    reg = Reg(tk)
    for s in segs:
        reg.reset(); s['hash'] = s['close'] = None
        for i in range(s['start'], s['end']):
            live = not (i - s['start'] < s['plen'] - 1)
            st, cl = reg.read(T[i], live)
            if st and s['hash'] is None: s['hash'] = i - s['start']
            if cl and s['close'] is None: s['close'] = i - s['start']
        s['valid'], s['bad'], s['val'] = reg.commit()
        if s['close'] is not None and s['close'] != s['end'] - s['start'] - 1:
            an.append(f"segment at {s['start']} continues after its answer closed")
    final = None
    if fin:
        ft = tk.decode(T[fin[-1]['start']:]).replace(EOS_TXT, '')
        m = re.search(NUM, ft); final = dict(start=fin[-1]['start'], txt=ft, ans=m.group(1).replace(',', '') if m else None)
    return dict(attempts=atts, final=final, anomalies=an, nq=collections.Counter(q['kind'] for q in Qs))


def bad_of(ans):
    """register semantics: impossible = missing, negative sign written, or a nonzero fraction digit"""
    if ans is None: return True
    frac = ans.split('.', 1)[1] if '.' in ans else ''
    return ans.startswith('-') or any(c in '123456789' for c in frac)


def attempt_rows(row, P, tk, set_, seed):
    T, LP, W = row['tokens'], row.get('logp'), row['wf']
    out = []
    for k, segs in enumerate(P['attempts']):
        s = segs[-1]; ids = T[s['start']:s['end']]
        text = tk.decode(ids).replace(EOS_TXT, '')
        m = re.search(NUM, text); ans = m.group(1).replace(',', '') if m else None
        reg = lambda key, d=None: (row[key][k] if key in row and k < len(row[key]) else d)
        out.append(dict(
            set=set_, idx=row['idx'], seed=seed, k=k, gid=f'{set_}/{row["idx"]}/s{seed}/k{k}', q=row['q'], gold=row['gold'],
            ok=int(correct(ans, row['gold'])), ans=ans, bad=bool(bad_of(ans)), lg=reg('lg'), nrep=len(segs) - 1,
            seg_tokens=ids, steps=line_spans(ids, tk, s['plen']), hash_off=s['hash'], text=text,
            prev_segments=[tk.decode(T[p['start']:p['end']]).replace(EOS_TXT, '') for p in segs[:-1]],
            phrase_len=s['plen'], seg_start=s['start'], seg_wf=W[s['start']:s['end']],
            seg_logp=[round(x, 3) for x in LP[s['start']:s['end']]] if LP else None,
            voted=int(row.get('pick', -1) == k), complete=bool(P['final']) or k < len(P['attempts']) - 1,
            reg_val=reg('val'), reg_ok=reg('rok'), reg_bad=reg('bad'), reg_end=reg('end'),
            port_valid=s['valid'], port_val=s['val'], port_bad=s['bad']))
    return out


def same(a, b): return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def xcheck(att, row, P, S):
    """agreement of parsed answer / register port with the stored register (per slot) -> counters in S, examples."""
    S['att'] += 1
    a = att; ex = []
    rx_ok = a['ans'] is not None
    if a['reg_ok'] is None: S['no_reg'] += 1; return ex
    agree = (rx_ok == bool(a['reg_ok'])) and (not rx_ok or same(float(a['ans']), a['reg_val']))
    agree_bad = (not a['reg_ok']) or (a['bad'] == bool(a['reg_bad']))
    port = (a['port_valid'] == bool(a['reg_ok'])) and (not a['port_valid'] or same(a['port_val'], a['reg_val'])) \
        and (a['port_bad'] == bool(a['reg_bad']))
    end_ok = a['reg_end'] == a['seg_start'] + len(a['seg_tokens']) - 1 or not a['complete']   # truncated: no commit
    port = port or not a['complete']                                   # truncated attempt: slot never committed
    S['ans_agree'] += agree; S['bad_agree'] += agree_bad; S['port_agree'] += port; S['end_agree'] += end_ok
    if not (agree and agree_bad): ex.append(('ans', a['gid'], a['ans'], a['bad'], a['reg_ok'], a['reg_val'], a['reg_bad'], a['text'][-80:]))
    if not port: ex.append(('port', a['gid'], a['port_valid'], a['port_val'], a['port_bad'], a['reg_ok'], a['reg_val'], a['reg_bad']))
    if not end_ok: ex.append(('end', a['gid'], a['reg_end'], a['seg_start'] + len(a['seg_tokens']) - 1))
    return ex


def rd(fn):
    with gzip.open(fn, 'rt') as f:
        return [json.loads(l) for l in f]


def seed_of(fn): return int(re.search(r'_seed(\d+)', fn).group(1))


def pct(x, n): return f'{100 * x / n:.2f}' if n else 'n/a'


def run_set(set_, gen, outd, tk, L):
    fs = sorted(glob.glob(f'{gen}/{set_}_N8_*seed*.jsonl.gz'), key=seed_of)
    gs = sorted(glob.glob(f'{gen}/{set_}_N1_R2_*greedy*seed*.jsonl.gz'), key=seed_of)
    if not fs: L(f'## {set_}: no N8 files in {gen}'); return None
    S = collections.Counter(); EX = []; items = {}; atts_all = []; seeds = []
    nq = collections.Counter(); anom = []
    for fn in fs:
        seed = seed_of(fn); seeds.append(seed)
        for row in rd(fn):
            P = parse_stream(row, tk); nq.update(P['nq'])
            anom += [(seed, row['idx'], x) for x in P['anomalies']]
            A = attempt_rows(row, P, tk, set_, seed)
            S['streams'] += 1
            S['natt_agree'] += len(P['attempts']) == row.get('natt')
            S['nseg_agree'] += sum(len(s) for s in P['attempts']) == row.get('nseg')
            S['n8'] += len(A) == 8
            fa = P['final']['ans'] if P['final'] else None
            S['final_eq_last'] += (fa == row.get('last')) or (fa is None and row.get('last') is None)
            pk = row.get('pick', -1)
            if fa is not None and 0 <= pk < len(A): S['final_eq_pick'] += A[pk]['port_valid'] and same(float(fa), A[pk]['port_val'])
            else: S['final_eq_pick'] += fa is None and pk < 0
            for a in A: EX += xcheck(a, row, P, S)
            it = items.setdefault(row['idx'], dict(set=set_, idx=row['idx'], q=row['q'], gold=row['gold'], n_right=0, n_att=0,
                                                   vote_ok={}, vote_ans={}, n_right_seed={}, greedy_ok=None))
            it['n_right'] += sum(a['ok'] for a in A); it['n_att'] += len(A)
            it['vote_ok'][str(seed)] = int(row['ok']); it['vote_ans'][str(seed)] = row.get('last')
            it['n_right_seed'][str(seed)] = sum(a['ok'] for a in A)
            atts_all += A
    G = {}
    for fn in gs:
        sd = seed_of(fn)
        for row in rd(fn):
            P = parse_stream(row, tk); A = attempt_rows(row, P, tk, set_, sd)
            G.setdefault(row['idx'], []).append((sd, int(row['ok']), A[-1]['ok'] if A else 0, row.get('last')))
    for idx, it in items.items():
        g = sorted(G.get(idx, []))
        if g:
            it['greedy_ok'] = g[0][1]; it['greedy_ans'] = g[0][3]; it['greedy_ok_seeds'] = {str(s): o for s, o, _, _ in g}
        it['complete'] = it['n_att'] == 8 * len(seeds)     # class is relative to the attempts present (n_att)
        it['class'] = 'always' if it['n_right'] == it['n_att'] else 'never' if it['n_right'] == 0 else 'mixed'
    os.makedirs(outd, exist_ok=True)
    with gzip.open(f'{outd}/attempts_{set_}.jsonl.gz', 'wt') as f:
        for a in atts_all: f.write(json.dumps(a) + '\n')
    IT = [items[i] for i in sorted(items)]
    with gzip.open(f'{outd}/items_{set_}.jsonl.gz', 'wt') as f:
        for it in IT: f.write(json.dumps(it) + '\n')
    # ------------------------------------------------ summary
    n = len(IT); K = 8 * len(seeds)
    L(f'\n## {set_}: {n} items, seeds {seeds} ({len(fs)} N8 files), greedy refs: {len(gs)} files')
    L(f'- streams {S["streams"]}, attempts {S["att"]} (streams with 8 attempts: {S["n8"]}); queues: {dict(nq)}')
    L(f'- parser vs stored register: answer/valid {pct(S["ans_agree"], S["att"])}%, bad flag {pct(S["bad_agree"], S["att"])}%, '
      f'register port {pct(S["port_agree"], S["att"])}%, attempt end index {pct(S["end_agree"], S["att"])}%; natt {pct(S["natt_agree"], S["streams"])}%, '
      f'nseg {pct(S["nseg_agree"], S["streams"])}%, final line == `last` {pct(S["final_eq_last"], S["streams"])}%, final line == picked slot {pct(S["final_eq_pick"], S["streams"])}%')
    if anom:
        ak = collections.Counter(x.split(':')[0] if x.split(':')[0] in ('c2first', 'nofinal') else 'other' for _, _, x in anom)
        L(f'- stream anomalies: {dict(ak)} (c2first = calc v2 overwrote the first token of a written queue; nofinal = stream '
          f'ended early, its last attempt kept with complete=false); examples: {anom[:4]}')
    if EX: L(f'- mismatch examples ({len(EX)}): ' + '; '.join(map(repr, EX[:6])))
    cl = collections.Counter(it['class'] for it in IT)
    L(f'- classes ({K}/item): always {cl["always"]} ({pct(cl["always"], n)}%), mixed {cl["mixed"]} ({pct(cl["mixed"], n)}%), '
      f'never {cl["never"]} ({pct(cl["never"], n)}%); items with fewer than {K} attempts: {sum(not it["complete"] for it in IT)}')
    na = sum(it['n_att'] for it in IT)
    L(f'- mean per-attempt accuracy: {pct(sum(it["n_right"] for it in IT), na)}%')
    L(f'- oracle@{K}: {pct(sum(it["n_right"] > 0 for it in IT), n)}%')
    gk = [it['greedy_ok'] for it in IT if it['greedy_ok'] is not None]
    gcons = sum(len(set(it.get('greedy_ok_seeds', {}).values())) <= 1 for it in IT)
    L(f'- greedy (N1_R2) accuracy: {pct(sum(gk), len(gk))}% (n={len(gk)}; greedy ok identical across ref seeds on {gcons}/{n})'
      if gk else '- greedy (N1_R2): no greedy refs for this set')
    L('\n| seed | vote acc | oracle@8 | vote wrong, some attempt right (selection gap) | vote wrong, none right | mean attempt acc |')
    L('|---|---|---|---|---|---|')
    by = collections.defaultdict(list)
    for a in atts_all: by[(a['seed'], a['idx'])].append(a['ok'])
    tot = collections.Counter()
    for sd in seeds:
        v = sum(it['vote_ok'].get(str(sd), 0) for it in IT)
        orc = sum(any(by[(sd, it['idx'])]) for it in IT)
        gap = sum((not it['vote_ok'].get(str(sd), 0)) and any(by[(sd, it['idx'])]) for it in IT)
        none = sum((not it['vote_ok'].get(str(sd), 0)) and not any(by[(sd, it['idx'])]) for it in IT)
        ma = sum(sum(by[(sd, it['idx'])]) for it in IT); mn = sum(len(by[(sd, it['idx'])]) for it in IT)
        tot.update(v=v, orc=orc, gap=gap, none=none)
        L(f'| {sd} | {pct(v, n)} | {pct(orc, n)} | {gap} ({pct(gap, n)}%) | {none} ({pct(none, n)}%) | {pct(ma, mn)} |')
    m = len(seeds)
    L(f'| mean | {pct(tot["v"] / m, n)} | {pct(tot["orc"] / m, n)} | {tot["gap"] / m:.1f} ({pct(tot["gap"] / m, n)}%) | '
      f'{tot["none"] / m:.1f} ({pct(tot["none"] / m, n)}%) | |')
    vote_any = sum(any(it['vote_ok'].values()) for it in IT); vote_all = sum(all(it['vote_ok'].values()) for it in IT)
    L(f'\n- vote right in >=1 of {m} seeds: {vote_any} ({pct(vote_any, n)}%), in all seeds: {vote_all} ({pct(vote_all, n)}%)')
    kk = collections.defaultdict(lambda: [0, 0])
    for a in atts_all: kk[a['k']][0] += a['ok']; kk[a['k']][1] += 1
    L('- per-attempt-index accuracy (k: acc%, n): ' + ', '.join(f'{k}: {pct(*kk[k])}' for k in sorted(kk)))
    rr = collections.Counter(a['nrep'] for a in atts_all); ok_r = collections.defaultdict(lambda: [0, 0])
    for a in atts_all: ok_r[a['nrep']][0] += a['ok']; ok_r[a['nrep']][1] += 1
    L('- attempts by repairs (nrep: n, acc%): ' + ', '.join(f'{r}: {rr[r]}, {pct(*ok_r[r])}' for r in sorted(rr)))
    bb = sum(a['bad'] for a in atts_all); miss = sum(a['ans'] is None for a in atts_all)
    L(f'- bad attempts (final segment): {bb} ({pct(bb, len(atts_all))}%), missing answer: {miss}')
    h = collections.Counter(it['n_right'] for it in IT)
    L(f'\nn_right histogram (0..{K}):\n```\n' + '\n'.join(f'{r:2d} {h[r]:4d} ' + '#' * ((h[r] + 2) // 3) for r in range(K + 1)) + '\n```')
    return dict(n=n, classes=cl, S=S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', default=['dev', 'dev2'])
    ap.add_argument('--gen', default=os.path.join(WS, 'gen', 'out'))
    ap.add_argument('--out', default=os.path.join(HERE, 'data'))
    ap.add_argument('--notes', default=os.path.join(WS, 'notes', 'atlas_counts.md'))
    a = ap.parse_args()
    tk = Tok(); lines = []
    def L(s): print(s, flush=True); lines.append(s)
    L('# Atlas counts (lane-1 rollouts: S3-serial N=8 at T=1 x seeds; greedy = N1_R2)\n')
    L('Produced by `atlas/parse.py`. class: always = all attempts right, never = none, else mixed. Attempt answer = first '
      "'#### n' (C.NUM) of the attempt's final segment; ok = C.correct vs gold. Selection gap = vote wrong but some attempt of "
      'the same stream right.\n')
    L('Data quirks (found by the parser cross-checks): (a) `nofinal`: calc v2 forced "<number>." then a veto-eos (wf bit 4) that '
      'ends the row inside an attempt, bypassing the serial writer: no vote line, fewer than 8 attempts, the last attempt is '
      'kept with complete=false (its register slot was never committed); (b) `c2first`: an attempt capped mid-number, calc v2 '
      'overwrote the first token of the repair queue (" We" -> digit), the host then saw "<digit> need to find"; parsed as a '
      'repair; (c) a vote with no valid slot writes a bare eos: the stream `ok` then falls back to the last "####" of the earlier '
      'attempts (gen_hook grades the last match of the whole text).')
    head = len(lines); secs = {}
    for s in a.sets:
        n0 = len(lines); run_set(s, a.gen, a.out, tk, L); secs[s] = '\n'.join(lines[n0:]).strip('\n')
    if os.path.exists(a.notes):                         # keep the sections of sets not re-run now
        old = re.split(r'\n(?=## )', open(a.notes).read())
        for blk in old[1:]:
            nm = blk.split(':', 1)[0][3:].strip()
            if nm not in secs: secs[nm] = blk.strip('\n')
    order = [x for x in ('dev', 'dev2') if x in secs] + sorted(x for x in secs if x not in ('dev', 'dev2'))
    os.makedirs(os.path.dirname(a.notes), exist_ok=True)
    open(a.notes, 'w').write('\n'.join(lines[:head]).strip('\n') + '\n\n' + '\n\n'.join(secs[x] for x in order) + '\n')


if __name__ == '__main__':
    main()
