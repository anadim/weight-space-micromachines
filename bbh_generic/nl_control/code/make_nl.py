"""Natural-language control rows for the BBH ft100 study (critic item 5).

For every row of a Team A / Team B ft subsample (../../ft100/scripts/subsample.py output, rows unchanged), keep the prompt /
question and replace the completion by a worked solution in plain language, written deterministically from the reference
parse that the original completion already contains (slot names, values, ops / clue codes, the reference call) and from the
reference solver's step-by-step state. No call syntax, no tool tokens, no slot or code numbers in the text. The final
"So the answer is X." is the original row's answer (checked against the NL trace's own result: every row must agree).

Formats follow the official BBH CoT exemplars of each task (tracking: "(0) At the start: A: x, B: y." per step; logical
deduction: numbered clues + "Combining these, we get the following ordering: ..."; navigate: "(k) ...: (x, y), facing ...";
web of lies: "(k) X says Y lies. Y ..., so X ..."; temporal: busy list + hour-by-hour sweep).

python make_nl.py A IN.jsonl OUT.jsonl
python make_nl.py B IN_DIR OUT_DIR          (IN_DIR/data/{task}_train.jsonl -> OUT_DIR/data/{task}_train.jsonl)
"""
import json, re, sys, os, itertools, collections

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'teamA'))   # repo bbh_generic/teamA
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'teamB'))   # repo bbh_generic/teamB
from callfmt import track_ref, order_ref, order_violates          # noqa: E402
import tools_ref                                                     # noqa: E402

CALL = re.compile(r'<tool_call>(.*?)</tool_call><tool_response>(.*?)</tool_response>')
ANS = re.compile(r'So the answer is (.*?)\.?\s*$')
ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']


def kv(line):
    """'Slots: 1 = a, 2 = b c.' -> {1: 'a', 2: 'b c'}"""
    body = line.split(':', 1)[1].strip().rstrip('.')
    return {int(k): v.strip() for k, v in re.findall(r'(\d+) = (.*?)(?=, \d+ = |$)', body)}


def options(prompt):
    last = prompt.rsplit('\nQ: ', 1)[-1] if '\nQ: ' in prompt else prompt
    return dict(re.findall(r'^\(([A-R])\) (.*)$', last, re.M))


def answer_of(comp):
    m = ANS.search(comp.strip()); return m.group(1).strip()


def desc_of(line, code):
    """'Pedro and Malia swap: 14.' / 'X replaces .... 4=6' -> 'Pedro and Malia swap'"""
    s = line.strip()
    i = s.rfind(code)
    assert i > 0, (line, code)
    return s[:i].rstrip().rstrip(':').rstrip('.').rstrip()


# ------------------------------------------------------------------ Team A: track
def nl_track(r):
    comp = r['completion']; lines = [l for l in comp.strip().split('\n')]
    (call, resp), = CALL.findall(comp)
    m = re.fullmatch(r'track (\d+):(.*?)(?:([?@])(\d))?', call)
    start, ops, qk, qx = m.group(1), (m.group(2).split(',') if m.group(2) else []), m.group(3), int(m.group(4))
    slots = kv(next(l for l in lines if l.startswith('Slots:')))
    code = any(l.startswith('Start values (the digits themselves)') for l in lines)
    vl = next((l for l in lines if l.startswith('Values:') or (l.startswith('Start values:'))), None)
    vals = {int(c): c for c in '0123456789'} if code else kv(vl)
    hdr = [i for i, l in enumerate(lines) if l.startswith(('Slots:', 'Values:', 'Start values'))]
    qi = next(i for i, l in enumerate(lines) if l.startswith('Question:'))
    oplines = lines[max(hdr) + 1:qi]
    assert len(oplines) == len(ops), (oplines, ops)
    st = [int(c) for c in start]; n = len(st)
    fmt = (lambda: ', '.join(f'{slots[i + 1]} = {vals[st[i]]}' for i in range(n))) if code else \
          (lambda: ', '.join(f'{slots[i + 1]}: {vals[st[i]]}' for i in range(n)))
    out = [f'(0) At the start: {fmt()}.']
    for k, (op, line) in enumerate(zip(ops, oplines), 1):
        if '=' in op: st[int(op[0]) - 1] = int(op[2])
        elif '<' in op: st[int(op[0]) - 1] = st[int(op[2]) - 1]
        else: i, j = int(op[0]) - 1, int(op[1]) - 1; st[i], st[j] = st[j], st[i]
        out.append(f'({k}) {desc_of(line, op)}: {fmt()}.')
    if qk == '?':
        res = str(st[qx - 1]); v = vals[st[qx - 1]]
        out.append(f'At the end, {slots[qx]} = {v}.' if code else f'At the end, {slots[qx]} has {v}.')
        said = v
    else:
        hit = [i + 1 for i, x in enumerate(st) if x == qx]; assert len(hit) == 1; res = str(hit[0])
        out.append(f'At the end, {slots[hit[0]]} holds {vals[qx]}.' if code else f'At the end, {vals[qx]} is with {slots[hit[0]]}.')
        said = slots[hit[0]]
    assert res == resp == track_ref(call), (res, resp, call)
    ans = answer_of(comp); opt = options(r['prompt'])
    L = ans.strip('()')
    assert opt[L].strip().lower().rstrip('.') == said.lower(), (opt[L], said)
    return '\n' + '\n'.join(out) + f' So the answer is {ans}.'


# ------------------------------------------------------------------ Team A: order
def propagate(n, cons, D, log, names, cid):
    """Arc-consistency + all-different on domains D (dict obj -> set of positions). Logs narrowing in prose.
    Returns False on contradiction (and the clue number that emptied a domain)."""
    def show(o): return f'{names[o]} is at position {min(D[o])}' if len(D[o]) == 1 else \
        f'{names[o]} can be at positions {", ".join(map(str, sorted(D[o])))}'
    changed = True
    while changed:
        changed = False
        for k, (op, a, b) in enumerate(cons):
            if op in ('<', '>', '+', '~'):
                newa = {x for x in D[a] if any(not order_violates_pos(op, x, y) for y in D[b] if y != x)}
                newb = {y for y in D[b] if any(not order_violates_pos(op, x, y) for x in newa if x != y)}
                ch = [o for o, new in ((a, newa), (b, newb)) if new != D[o]]
                D[a], D[b] = newa, newb
            else:
                tgt = b if op in ('=', '!=') else n + 1 - b
                new = {x for x in D[a] if (x == tgt) == (op in ('=', '=-'))}
                ch = [a] if new != D[a] else []; D[a] = new
            if any(not D[o] for o in D): log.append(f'That breaks ({cid[k]}).'); return False
            if ch: log.append(f'From ({cid[k]}): ' + '; '.join(show(o) for o in ch) + '.'); changed = True
        # all-different: a fixed object takes its position away from the others
        for o in sorted(D):
            if len(D[o]) == 1:
                p = min(D[o]); ch = [q for q in sorted(D) if q != o and p in D[q]]
                for q in ch: D[q] = D[q] - {p}
                if any(not D[q] for q in D): log.append('That leaves no position for someone.'); return False
                if ch: log.append(f'Position {p} is taken by {names[o]}, so ' + '; '.join(show(q) for q in ch) + '.'); changed = True
        for p in range(1, n + 1):          # hidden single: a position only one object can take
            can = [o for o in sorted(D) if p in D[o]]
            if not can: log.append(f'No one can be at position {p}.'); return False
            if len(can) == 1 and len(D[can[0]]) > 1:
                D[can[0]] = {p}; log.append(f'Only {names[can[0]]} can be at position {p}, so {names[can[0]]} is at position {p}.'); changed = True
    return True


def order_violates_pos(op, pa, pb):
    return not {'<': pa < pb, '>': pa > pb, '+': pb == pa + 1, '~': abs(pa - pb) == 1}[op]


def solve_log(n, cons, names, cid):
    D = {o: set(range(1, n + 1)) for o in range(1, n + 1)}; log = []
    ok = propagate(n, cons, D, log, names, cid)
    assert ok

    def search(D, log, depth):
        if all(len(D[o]) == 1 for o in D): return D
        o = min((o for o in D if len(D[o]) > 1), key=lambda o: (len(D[o]), o))
        for p in sorted(D[o]):
            D2 = {k: set(v) for k, v in D.items()}; D2[o] = {p}; sub = []
            if propagate(n, cons, D2, sub, names, cid):
                log.append(f'Suppose {names[o]} is at position {p}.')
                log.extend(sub)
                res = search(D2, log, depth + 1)
                if res is not None: return res
            else:
                log.append(f'Suppose {names[o]} is at position {p}. ' + sub[-1])
        return None
    final = search(D, log, 0)
    assert final is not None
    perm = tuple(sorted(D, key=lambda o: min(final[o])))
    perm = tuple(sorted(range(1, n + 1), key=lambda o: min(final[o])))
    return perm, log


def nl_order(r):
    comp = r['completion']; lines = comp.strip().split('\n')
    (call, resp), = CALL.findall(comp)
    m = re.fullmatch(r'order (\d):(.*?)(?:(\?-|\?|@)(\d))?', call)
    n = int(m.group(1)); q, x = m.group(3), int(m.group(4))
    cons = [(op, int(a), int(b)) for a, op, b in re.findall(r'(\d)(<|>|\+|~|=-|=|!=-|!=)(\d)', m.group(2))]
    names = kv(next(l for l in lines if l.startswith('Objects:')))
    pos = kv(next(l for l in lines if l.startswith('Positions:')))
    i0 = next(i for i, l in enumerate(lines) if l.startswith('Positions:')); qi = next(i for i, l in enumerate(lines) if l.startswith('Question:'))
    clines = lines[i0 + 1:qi]; assert len(clines) == len(cons), (clines, cons)
    codes = re.findall(r'\d(?:<|>|\+|~|=-|=|!=-|!=)\d', m.group(2))
    out = []
    for k, (line, c) in enumerate(zip(clines, codes), 1): out.append(f'({k}) {desc_of(line, c)}.')
    cid = list(range(1, len(cons) + 1))
    out.append(f'Positions go from 1 ({pos[1]}) to {n} ({pos[n]}).')
    perm, log = solve_log(n, cons, names, cid)
    out += log
    sols = [p for p in itertools.permutations(range(1, n + 1)) if not any(order_violates(p, op, a, b) for op, a, b in cons)]
    assert perm in sols
    out.append(f'({len(cons) + 1}) Combining these, we get the following ordering: "({pos[1]}) ' + ' '.join(names[o] for o in perm) + f' ({pos[n]})".')
    if q in ('?', '?-'):
        k = x if q == '?' else n + 1 - x; res = str(perm[k - 1]); who = names[perm[k - 1]]
        where = f'position {k}' if q == '?' else f'position {x} counted from the {pos[n]} end'
        out.append(f'According to this ordering, {where} is {who}.'); said = who
    else:
        res = str(perm.index(x) + 1); out.append(f'According to this ordering, {names[x]} is at position {res}, the {ORD[int(res) - 1]}.'); said = ORD[int(res) - 1]
    assert res == resp == order_ref(call), (res, resp, call)
    ans = answer_of(comp); opt = options(r['prompt']); L = ans.strip('()')
    o = opt[L].lower()
    assert said.lower() in o or o in said.lower(), (opt[L], said)
    return '\n' + '\n'.join(out) + f' So the answer is {ans}.'


# ------------------------------------------------------------------ Team B
def nl_interval(x):
    (call, resp), = CALL.findall(x['completion'])
    m = re.fullmatch(r'interval (\S+)-(\S+) busy (.*)', call)
    a, ap = tools_ref.parse_time(m.group(1)); b, _ = tools_ref.parse_time(m.group(2), end=True)
    busy = set(); ivs = []
    for iv in m.group(3).split(', '):
        u, v = iv.split('-'); uu, _ = tools_ref.parse_time(u); vv, _ = tools_ref.parse_time(v, end=True); busy |= set(range(uu, vv)); ivs.append((u, v))
    f = lambda h: tools_ref.fmt_time(h, ap)
    out = [f'The open window is {m.group(1)} to {m.group(2)}.', 'Busy: ' + ', '.join(f'{u} to {v}' for u, v in ivs) + '.',
           'Hour by hour: ' + ', '.join(f'{f(h)} {"busy" if h in busy else "free"}' for h in range(a, b)) + '.']
    free = tools_ref.interval_ref(call); assert free == resp, (free, resp)
    out.append('Free: ' + ('nothing' if free == 'none' else ', '.join(s.replace('-', ' to ') for s in free.split(', '))) + '.')
    return '\n' + '\n'.join(out) + '\n' + x['completion'].split('</tool_response>')[-1].strip()


def nl_walk(x, km):
    (call, resp), = CALL.findall(x['completion'])
    cmds = call[len('walk '):].split(', ')
    H = tools_ref.HEAD; h = xx = yy = 0
    unit = 'km' if km else 'steps'
    out = ['We start at (0, 0), facing north.']
    for k, c in enumerate(cmds, 1):
        w = c.split(' ')
        if w[0] == 'turn':
            h = (h + {'left': 1, 'around': 2, 'right': 3}[w[1]]) % 4; say = {'left': 'Turn left', 'right': 'Turn right', 'around': 'Turn around'}[w[1]]
        else:
            rr = {'forward': 0, 'left': 1, 'backward': 2, 'right': 3}[w[0]]; dx, dy = tools_ref.VEC[H[(h + rr) % 4]]; nn = int(w[1])
            xx += dx * nn; yy += dy * nn
            say = {'forward': f'Go {nn} {unit} forward', 'backward': f'Go {nn} {unit} backward', 'left': f'Go {nn} {unit} to the left', 'right': f'Go {nn} {unit} to the right'}[w[0]]
        out.append(f'({k}) {say}: ({xx}, {yy}), facing {H[h]}.')
    assert f'({xx}, {yy}) facing {H[h]}' == resp == tools_ref.turtle_ref(call)
    return '\n' + '\n'.join(out) + '\n' + x['completion'].split('</tool_response>')[-1].strip()


def nl_truth(x):
    comp = x['completion']; (call, resp), = CALL.findall(comp)
    head = ((x.get('masked_prefix') or '') + comp).split('<tool_call>')[0]
    nm = {int(k): v for v, k in re.findall(r'([^,=\n]+?) = (\d+)', head)}
    nm = {k: v.strip() for k, v in nm.items()}
    cls = call[len('truth '):].split('; ')
    facts, edges = [], []
    for c in cls:
        w = c.split(' ')
        if len(w) == 2: facts.append((int(w[0]), w[1] == 'true'))
        else: edges.append((int(w[0]), int(w[2]), w[3] == 'true'))
    st = lambda v: 'tells the truth' if v else 'lies'
    val = {}; out = []; k = 0
    for p, v in facts:
        k += 1; val[p] = v; out.append(f'({k}) {nm[p]} {st(v)}. So, we know that {nm[p]} {st(v)}.')
    done = set(); prog = True
    while prog:
        prog = False
        for e in edges:
            if e in done: continue
            a, b, c = e; claim = f'{nm[a]} says {nm[b]} {st(c)}.'
            if b in val and a not in val:
                val[a] = (val[b] == c); k += 1; done.add(e); prog = True
                out.append(f'({k}) {claim} We know {nm[b]} {st(val[b])}, so {nm[a]} {"is right" if val[a] else "is wrong"}: {nm[a]} {st(val[a])}.')
            elif a in val and b not in val:
                val[b] = (c if val[a] else not c); k += 1; done.add(e); prog = True
                out.append(f'({k}) {claim} We know {nm[a]} {st(val[a])}, so the claim is {"true" if val[a] else "false"}: {nm[b]} {st(val[b])}.')
            elif a in val and b in val:
                done.add(e)
    got = ', '.join(f'{p} {("true" if val[p] else "false") if p in val else "unknown"}' for p in sorted(nm))
    assert got == resp == tools_ref.truth_ref(call), (got, resp, call)
    who = re.search(r'(\S[^\n]*?) is (\d+), and \d+ is (true|false)\.', comp.split('</tool_response>')[-1])
    p = int(who.group(2)); assert nm[p] == who.group(1).strip()
    ans = answer_of(comp)
    return '\n' + '\n'.join(out) + f'\nSo {nm[p]} {st(val[p])}. So the answer is {ans}.'


def main():
    team, src, dst = sys.argv[1:4]; cnt = collections.Counter()
    if team == 'A':
        rows = [json.loads(l) for l in open(src)]
        with open(dst, 'w') as f:
            for r in rows:
                c = nl_track(r) if r['tool'] == 'track' else nl_order(r)
                assert '<tool' not in c and answer_of(c) == answer_of(r['completion'])
                f.write(json.dumps(dict(r, completion=c, orig_completion=r['completion'])) + '\n'); cnt[r['tool']] += 1
    else:
        os.makedirs(f'{dst}/data', exist_ok=True)
        for task in ['temporal', 'lab', 'navigate', 'hiker', 'lies', 'rumor']:
            rows = [json.loads(l) for l in open(f'{src}/data/{task}_train.jsonl')]
            with open(f'{dst}/data/{task}_train.jsonl', 'w') as f:
                for x in rows:
                    fn = {'temporal': nl_interval, 'lab': nl_interval, 'navigate': lambda x: nl_walk(x, False), 'hiker': lambda x: nl_walk(x, True),
                          'lies': nl_truth, 'rumor': nl_truth}[task]
                    c = fn(x)
                    assert '<tool' not in c and answer_of(c) == answer_of(x['completion']), (c, x['completion'])
                    y = {k: v for k, v in x.items() if k != 'masked_prefix'}; y['completion'] = c; y['orig_completion'] = x['completion']
                    y['had_masked_prefix'] = bool(x.get('masked_prefix'))
                    f.write(json.dumps(y) + '\n'); cnt[task] += 1
    print(dict(cnt))


if __name__ == '__main__':
    main()
