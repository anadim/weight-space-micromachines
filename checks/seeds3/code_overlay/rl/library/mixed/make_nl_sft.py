"""Arm aNL (checks/seeds3, item (e)): the SFT rows of mixed/sft_warm.py with every tool call replaced by a plain worked solution.

Input : bbh_generic/integrate/data/train_union_v2.jsonl (the 432 union rows sft_warm.py uses).
Output: the same rows (same prompt / team / kind), completion rewritten so that no '<tool_call>' / '<tool_response>' remains:
  - Team A track / order, Team B interval / walk / truth: the call+response (and the code header lines the call was built
    from: 'Slots:', 'Objects:', op lines with digit codes, 'Question: ?2') are replaced by the step-by-step trace of the
    reference solver, rendered with bbh_generic/nl_control/code/make_nl.py (the renderers of the BBH NL control); every
    trace is checked against the row's own tool response, and the final answer line is the row's original one.
  - multi rows (M1..M5: tool + arithmetic / boolean rule): the tool part is rendered the same way, the text after the
    response (the arithmetic / boolean rule, already plain text) is kept.
  - Team C passive rows and no-call rows have no tags: unchanged.
  - '*/recovery' rows (recovering from an invalid call that sits in the PROMPT) have no plain-text analogue: dropped.
Every output row is asserted tag-free and to end in the original answer.
python make_nl_sft.py IN.jsonl OUT.jsonl"""
import os, sys, re, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WSM_REPO_ROOT', os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'bbh_generic', 'nl_control', 'code'))
import make_nl as NL                       # noqa: E402
from make_nl import CALL, kv, desc_of, tools_ref, track_ref, order_ref, order_violates, solve_log, ORD  # noqa: E402

ANS = re.compile(r'So the answer is (.*?)\.?\s*$')


def after(comp):
    """text after the tool response, minus the references to the call's slot / object codes (the trace states them)"""
    t = comp.split('</tool_response>', 1)[1].strip()
    t = re.sub(r'^(?:Value|Slot|Object|Position) ?\d* = [^\n]*?\.(?=\s|$)\s*', '', t)
    t = re.sub(r'^Position \d+\.\s*', '', t)
    t = re.sub(r'\b([A-Z][\w-]*) is \d+, at position', r'\1 is at position', t)
    t = re.sub(r'(^|\. )[A-Z][\w-]* is \d+, so "', lambda m: m.group(1) + 'So "', t)
    return t


def trace_track(comp):
    lines = comp.strip().split('\n')
    (call, resp), = CALL.findall(comp)
    m = re.fullmatch(r'track (\d+):(.*?)(?:([?@])(\d))?', call)
    start, ops, qk, qx = m.group(1), (m.group(2).split(',') if m.group(2) else []), m.group(3), int(m.group(4))
    slots = kv(next(l for l in lines if l.startswith('Slots:')))
    code = any(l.startswith('Start values (the digits themselves)') for l in lines)
    vl = next((l for l in lines if l.startswith('Values:') or l.startswith('Start values:')), None)
    vals = {int(c): c for c in '0123456789'} if code else kv(vl)
    hdr = [i for i, l in enumerate(lines) if l.startswith(('Slots:', 'Values:', 'Start values'))]
    qi = next(i for i, l in enumerate(lines) if l.startswith('Question:'))
    oplines = lines[max(hdr) + 1:qi]
    for l in oplines:                                  # track/mix: '-> new value 8 = Radar; 7=8'
        for k_, v_ in re.findall(r'new value (\d+) = (.*?);', l): vals[int(k_)] = v_.strip()
    st = [int(c) for c in start]; n = len(st)
    fmt = (lambda: ', '.join(f'{slots[i + 1]} = {vals[st[i]]}' for i in range(n))) if code else \
          (lambda: ', '.join(f'{slots[i + 1]}: {vals[st[i]]}' for i in range(n)))
    out = [f'(0) At the start: {fmt()}.']; p_ = 0
    for k, line in enumerate(oplines, 1):
        if '->' in line:                               # one line may carry several codes ('two swaps: 56,26')
            desc, tail = line.split(' ->', 1)[0].strip().rstrip('.'), line.split('->', 1)[1]
            mine = []
            while p_ < len(ops) and ops[p_] in tail:
                pos_ = tail.find(ops[p_]); mine.append(ops[p_]); tail = tail[pos_ + len(ops[p_]):]; p_ += 1
            assert mine, (line, ops, p_)
        else:
            desc, mine = desc_of(line, ops[p_]), [ops[p_]]; p_ += 1
        for op in mine:
            if '=' in op: st[int(op[0]) - 1] = int(op[2])
            elif '<' in op: st[int(op[0]) - 1] = st[int(op[2]) - 1]
            else: i, j = int(op[0]) - 1, int(op[1]) - 1; st[i], st[j] = st[j], st[i]
        out.append(f'({k}) {desc}: {fmt()}.')
    assert p_ == len(ops), (oplines, ops)
    if qk == '?':
        res = str(st[qx - 1]); v = vals[st[qx - 1]]
        out.append(f'At the end, {slots[qx]} = {v}.' if code else f'At the end, {slots[qx]} has {v}.')
    else:
        hit = [i + 1 for i, x in enumerate(st) if x == qx]; assert len(hit) == 1; res = str(hit[0])
        out.append(f'At the end, {slots[hit[0]]} holds {vals[qx]}.' if code else f'At the end, {vals[qx]} is with {slots[hit[0]]}.')
    assert res == resp == track_ref(call), (res, resp, call)
    return out


def trace_order(comp):
    lines = comp.strip().split('\n')
    (call, resp), = CALL.findall(comp)
    m = re.fullmatch(r'order (\d):(.*?)(?:(\?-|\?|@)(\d))?', call)
    n = int(m.group(1)); q, x = m.group(3), (int(m.group(4)) if m.group(4) else None)
    cons = [(op, int(a_), int(b_)) for a_, op, b_ in re.findall(r'(\d)(<|>|\+|~|=-|=|!=-|!=)(\d)', m.group(2))]
    names = kv(next(l for l in lines if l.startswith('Objects:')))
    pos = kv(next(l for l in lines if l.startswith('Positions:')))
    i0 = next(i for i, l in enumerate(lines) if l.startswith('Positions:')); qi = next(i for i, l in enumerate(lines) if l.startswith('Question:'))
    clines = lines[i0 + 1:qi]; assert len(clines) == len(cons), (clines, cons)
    codes = re.findall(r'\d(?:<|>|\+|~|=-|=|!=-|!=)\d', m.group(2))
    out = [f'({k}) {desc_of(line, c)}.' for k, (line, c) in enumerate(zip(clines, codes), 1)]
    out.append(f'Positions go from 1 ({pos[1]}) to {n} ({pos[n]}).')
    perm, log = solve_log(n, cons, names, list(range(1, len(cons) + 1)))
    out += log
    sols = [p for p in __import__('itertools').permutations(range(1, n + 1)) if not any(order_violates(p, op, a_, b_) for op, a_, b_ in cons)]
    assert perm in sols
    out.append(f'({len(cons) + 1}) Combining these, we get the following ordering: "({pos[1]}) ' + ' '.join(names[o] for o in perm) + f' ({pos[n]})".')
    if q in ('?', '?-'):
        k = x if q == '?' else n + 1 - x; res = str(perm[k - 1])
        where = f'position {k}' if q == '?' else f'position {x} counted from the {pos[n]} end'
        out.append(f'According to this ordering, {where} is {names[perm[k - 1]]}.')
    elif q == '@':
        res = str(perm.index(x) + 1); out.append(f'According to this ordering, {names[x]} is at position {res}, the {ORD[int(res) - 1]}.')
    else:
        res = ''.join(map(str, perm))
        out.append('So the positions are: ' + ', '.join(f'{names[o]} at position {i}' for i, o in enumerate(perm, 1)) + '.')
    assert res == resp == order_ref(call), (res, resp, call)
    return out


def trace_truth(comp):
    (call, resp), = CALL.findall(comp)
    head = comp.split('<tool_call>')[0]
    nm = {int(k): v.strip() for v, k in re.findall(r'([^,=\n]+?) = (\d+)', head)}
    st = lambda v: 'tells the truth' if v else 'lies'
    facts, edges = [], []
    for c in call[len('truth '):].split('; '):
        w = c.split(' ')
        if len(w) == 2: facts.append((int(w[0]), w[1] == 'true'))
        else: edges.append((int(w[0]), int(w[2]), w[3] == 'true'))
    val = {}; out = []; k = 0
    for p, v in facts:
        k += 1; val[p] = v; out.append(f'({k}) {nm[p]} {st(v)}. So, we know that {nm[p]} {st(v)}.')
    done = set(); prog = True
    while prog:
        prog = False
        for e in edges:
            if e in done: continue
            a_, b_, c = e; claim = f'{nm[a_]} says {nm[b_]} {st(c)}.'
            if b_ in val and a_ not in val:
                val[a_] = (val[b_] == c); k += 1; done.add(e); prog = True
                out.append(f'({k}) {claim} We know {nm[b_]} {st(val[b_])}, so {nm[a_]} {"is right" if val[a_] else "is wrong"}: {nm[a_]} {st(val[a_])}.')
            elif a_ in val and b_ not in val:
                val[b_] = (c if val[a_] else not c); k += 1; done.add(e); prog = True
                out.append(f'({k}) {claim} We know {nm[a_]} {st(val[a_])}, so the claim is {"true" if val[a_] else "false"}: {nm[b_]} {st(val[b_])}.')
            elif a_ in val and b_ in val:
                done.add(e)
    got = ', '.join(f'{p} {("true" if val[p] else "false") if p in val else "unknown"}' for p in sorted(nm))
    assert got == resp == tools_ref.truth_ref(call), (got, resp, call)
    return out, nm, val


def convert(r):
    """-> (new completion | None, how)"""
    comp, kind = r['completion'], r['kind']
    if '<tool' not in comp and '<tool' not in r['prompt']: return comp, 'unchanged'
    if 'recovery' in kind: return None, 'drop_recovery'
    tool = CALL.findall(comp)[0][0].split(' ')[0]
    if tool == 'track': steps = trace_track(comp)
    elif tool == 'order': steps = trace_order(comp)
    elif tool == 'interval':
        return NL.nl_interval(r), 'nl_interval'
    elif tool == 'walk':
        return NL.nl_walk(r, km=kind.startswith('hiker')), 'nl_walk'
    elif tool == 'truth':
        steps, nm, val = trace_truth(comp)
        tail = after(comp)
        who = re.match(r'(\S[^\n]*?) is (\d+), and \d+ is (true|false)\.\s*', tail)
        if who:                                   # 'Kofi is 7, and 7 is false. [So Kofi lies.] So the answer is No.'
            p = int(who.group(2)); assert nm[p] == who.group(1).strip()
            ans = ANS.search(tail).group(1).strip()
            return '\n' + '\n'.join(steps) + f'\nSo {nm[p]} {"tells the truth" if val[p] else "lies"}. So the answer is {ans}.', 'nl_truth'
        return '\n' + '\n'.join(steps) + '\n' + tail, 'nl_truth+tail'
    else:
        raise ValueError(tool)
    return '\n' + '\n'.join(steps) + '\n' + after(comp), f'nl_{tool}+tail'


def main():
    src, dst = sys.argv[1:3]; cnt = collections.Counter(); by = collections.Counter()
    with open(dst, 'w') as f:
        for l in open(src):
            r = json.loads(l)
            try:
                c, how = convert(r)
            except Exception as e:
                c, how = None, 'drop_fail:' + type(e).__name__
            cnt[how] += 1; by[(r['kind'], how)] += 1
            if c is None: continue
            assert '<tool' not in c and '</tool' not in c, c
            m0 = ANS.search(r['completion'].strip())
            if m0: assert ANS.search(c.strip()).group(1).strip() == m0.group(1).strip(), (c, r['completion'])
            f.write(json.dumps(dict(r, completion=c, orig_completion=r['completion'], nl_how=how)) + '\n')
    print(json.dumps(dict(cnt), indent=0)); print({f'{k}|{h}': v for (k, h), v in sorted(by.items())})


if __name__ == '__main__':
    main()
