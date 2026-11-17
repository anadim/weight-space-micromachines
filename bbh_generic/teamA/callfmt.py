"""Call format and exact Python REFERENCES for the two generic Team-A tools (used for tests and for writing training data;
never at inference -- at inference the compiled ReLU circuits in track_gen.py / order_gen.py do the work).

Call format (Qwen3 tokens, all single tokens):  <tool_call>NAME ARGS</tool_call><tool_response>RESULT</tool_response>
The host writes up to and including </tool_call>; the tool writes <tool_response>RESULT</tool_response> (RESULT empty =
error / not determined).

------------------------------------------------------------------------------------------------------------------------
track  -- a STATE tracker over up to 9 numbered slots holding one digit each (a tiny register file).
    track S:OPS[Q]
    S    initial values, one digit per slot (slot 1 first), 1..9 digits             e.g. 1234567
    OPS  comma-separated operations, applied left to right (may be empty):
           ij     swap the values of slots i and j          (i != j)
           i=v    set slot i to the digit v
           i<j    copy: slot i gets the value of slot j     (i != j)
    Q    optional query:  ?i  -> the value in slot i
                          @v  -> the slot that holds value v (must be exactly one)
                          (none) -> the whole final state, one digit per slot
    e.g. track 123:12,23?1  -> 3      track 1234:2=7,3<2 -> 1774
------------------------------------------------------------------------------------------------------------------------
order  -- an ORDERING-CONSTRAINT solver: the orderings of objects 1..N (N = 2..7) that satisfy every constraint.
    order N:C,C,...[Q]
    C    a<b   a comes before b            a>b   a comes after b
         a+b   a comes immediately before b                a~b   a and b are next to each other
         a=k   a is at position k          a=-k  a is at position k counted from the end
         a!=k  a is not at position k      a!=-k a is not at position k from the end
    Q    ?k / ?-k  -> the object at position k (from the start / from the end)
         @a        -> the position (from the start) of object a
         (none)    -> the whole order: the object at position 1, 2, ..., N
    The answer is given only if it is the same in every ordering that satisfies all constraints (and there is >= 1);
    otherwise the result is empty.
    e.g. order 3:2>1,3=2?1 -> 1      order 4:1+2,3=1,4>2 -> 3124
"""
import re, itertools

CS, CE, OS, OE = '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>'
TRACK_NMAX, TRACK_MAXOPS = 9, 20
ORDER_NMIN, ORDER_NMAX, ORDER_MAXC = 2, 7, 20


def track_ref(args: str) -> str:
    m = re.fullmatch(r'track (\d{1,9}):((?:\d\d|\d=\d|\d<\d)(?:,(?:\d\d|\d=\d|\d<\d))*)?(?:([?@])(\d))?', args)
    if not m: return ''
    st = [int(c) for c in m.group(1)]; n = len(st)
    ops = m.group(2).split(',') if m.group(2) else []
    if len(ops) > TRACK_MAXOPS: return ''
    for op in ops:
        if '=' in op:
            i, v = int(op[0]), int(op[2])
            if not 1 <= i <= n: return ''
            st[i - 1] = v
        elif '<' in op:
            i, j = int(op[0]), int(op[2])
            if not (1 <= i <= n and 1 <= j <= n and i != j): return ''
            st[i - 1] = st[j - 1]
        else:
            i, j = int(op[0]), int(op[1])
            if not (1 <= i <= n and 1 <= j <= n and i != j): return ''
            st[i - 1], st[j - 1] = st[j - 1], st[i - 1]
    if m.group(3) == '?':
        q = int(m.group(4))
        return str(st[q - 1]) if 1 <= q <= n else ''
    if m.group(3) == '@':
        v = int(m.group(4)); hits = [i + 1 for i, x in enumerate(st) if x == v]
        return str(hits[0]) if len(hits) == 1 else ''
    return ''.join(map(str, st))


ORDER_OPS = ('<', '>', '+', '~', '=', '=-', '!=', '!=-')


def order_violates(p, op, a, b):
    """p: tuple, p[k] = object at position k+1. True if the ordering violates constraint (a op b)."""
    n = len(p)
    if not 1 <= a <= n: return True
    if op in ('<', '>', '+', '~'):
        if b > n or a == b: return True
        pa, pb = p.index(a), p.index(b)
        return not {'<': pa < pb, '>': pa > pb, '+': pb == pa + 1, '~': abs(pa - pb) == 1}[op]
    if not 1 <= b <= n: return True
    pa = p.index(a) + 1
    tgt = b if op in ('=', '!=') else n + 1 - b
    return (pa != tgt) if op in ('=', '=-') else (pa == tgt)


def order_ref(args: str) -> str:
    m = re.fullmatch(r'order (\d):((?:\d(?:<|>|\+|~|=-|=|!=-|!=)\d)(?:,\d(?:<|>|\+|~|=-|=|!=-|!=)\d)*)?(?:(\?-|\?|@)(\d))?', args)
    if not m: return ''
    n = int(m.group(1))
    if not ORDER_NMIN <= n <= ORDER_NMAX: return ''
    cons = re.findall(r'(\d)(<|>|\+|~|=-|=|!=-|!=)(\d)', m.group(2) or '')
    if len(cons) > ORDER_MAXC: return ''
    cons = [(op, int(a), int(b)) for a, op, b in cons]
    if any(not (1 <= a <= ORDER_NMAX and 1 <= b <= ORDER_NMAX) for _, a, b in cons): return ''
    sols = [p for p in itertools.permutations(range(1, n + 1)) if not any(order_violates(p, op, a, b) for op, a, b in cons)]
    if not sols: return ''
    q, x = m.group(3), (int(m.group(4)) if m.group(4) else None)
    if q == '?-' and not cons: return ''          # tokenizer artifact: ':?-' is two tokens ':?' '-' (not supported)
    if q in ('?', '?-'):
        if not 1 <= x <= n: return ''
        k = x if q == '?' else n + 1 - x
        vals = {p[k - 1] for p in sols}
        return str(vals.pop()) if len(vals) == 1 else ''
    if q == '@':
        if not 1 <= x <= n: return ''
        vals = {p.index(x) + 1 for p in sols}
        return str(vals.pop()) if len(vals) == 1 else ''
    if len(sols) != 1: return ''
    return ''.join(map(str, sols[0]))


def run_ref(args: str) -> str:
    if args.startswith('track '): return track_ref(args)
    if args.startswith('order '): return order_ref(args)
    return ''


CALL_RE = re.compile(re.escape(CS) + r'(.*?)' + re.escape(CE))


def fill_calls(text: str) -> str:
    """Reference: write each call's response (used to build training transcripts)."""
    return CALL_RE.sub(lambda m: f'{CS}{m.group(1)}{CE}{OS}{run_ref(m.group(1))}{OE}', text)


if __name__ == '__main__':
    for c in ['track 123:12,23?1', 'track 1234:2=7,3<2', 'track 123:@3', 'track 123:', 'track 12:11', 'track 1234567:12,35,2=4?2',
              'order 3:2>1,3=2?1', 'order 4:1+2,3=1,4>2', 'order 3:1<2', 'order 3:1<2?-1', 'order 3:1<2@3', 'order 5:1=-1,2!=-1,2~1?-2']:
        print(c, '->', repr(run_ref(c)))
