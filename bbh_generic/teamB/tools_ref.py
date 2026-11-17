"""Reference semantics of Team B's three generic tools (Python; used ONLY to write training completions and to test the
compiled circuits. At inference the tools run as compiled ReLU circuits, see circuits_B.py).

Call format = the shared envelope of all BBH teams (protocol rule 14, reference bbh_generic/teamA/callfmt.py):
    <tool_call>NAME ARGS</tool_call><tool_response>RESULT</tool_response>
The host writes up to and including </tool_call>; the tool writes <tool_response>RESULT</tool_response>.
RESULT empty = the call is malformed / out of range (error). A call with another NAME: the tool stays silent.

1. interval  ("free time = window minus busy intervals")
   interval A-B busy C-D, E-F, ...        (>= 1 busy interval, any order, may overlap; exactly one ' busy')
   time   = an hour of the day '7am', '12pm', '11pm' (1..12 + am|pm; 12am = 0 at a start, 24 at an end), or a plain integer
            0..99 (e.g. day numbers, 24-hour clock hours). Whole units only (no '9:30'). Both ends of one interval use the
            same notation. The window's first endpoint decides the output notation.
   RESULT = the maximal free sub-intervals of [A, B) in increasing order 'X-Y, Z-W' (at most 4; more -> error), or 'none'.
2. walk  ("2-D position/heading accumulator", dead reckoning on a grid)
   walk CMD, CMD, ...      1..24 commands, CMD = turn left | turn right | turn around | forward N | backward N | left N | right N
   N = 0..99 (one or two digits). Start (0, 0) facing north (+y); 'left N'/'right N' step sideways relative to the current
   heading (no turn). RESULT = '(X, Y) facing <north|east|south|west>'.
3. truth  ("truth propagation": same/opposite constraints between people who are truthful or lying)
   truth CLAUSE; CLAUSE; ...   CLAUSE = 'K true' (K is truthful) | 'K false' (K lies) | 'K says J true' (K claims J is
   truthful) | 'K says J false' (K claims J lies).   K, J in 1..12, any order, any number of clauses per person.
   Semantics (knights and knaves): 'K says J c' means v_K = (v_J == c), i.e. the same truth value (c = true) or the
   opposite (c = false); it is used in both directions. Facts and these constraints are propagated (12 rounds).
   RESULT = 'K value' for every K mentioned, increasing K; value = true / false / unknown (not connected to a fact).
   Error ('') if the known values contradict a clause or each other.
Limits common to all: the whole call must fit in the tool's window (128 tokens, '<tool_call>' included), otherwise the tool
is silent; a call that breaks the grammar or a limit gets the empty RESULT.
"""
import re

CS, CE, OS, OE = '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>'


# ---------------- interval ----------------
def parse_time(s, end=False):
    m = re.fullmatch(r'(\d{1,2})(am|pm)?', s)
    if not m: raise ValueError(s)
    n, u = int(m.group(1)), m.group(2)
    if u is None:
        return n, False
    if not 1 <= n <= 12: raise ValueError(s)
    h = n % 12 + (12 if u == 'pm' else 0)
    if end and u == 'am' and n == 12: h = 24
    return h, True


def fmt_time(h, ampm):
    if not ampm: return str(h)
    if h in (0, 24): return '12am'
    if h == 12: return '12pm'
    return f'{h % 12}{"pm" if h >= 12 else "am"}'


T_RE = r'\d{1,2}(?:am|pm)?'


def interval_ref(args):
    m = re.fullmatch(rf'interval ({T_RE})-({T_RE}) busy ({T_RE}-{T_RE}(?:, {T_RE}-{T_RE})*)', args)
    if not m: return ''
    a, ap = parse_time(m.group(1)); b, bp = parse_time(m.group(2), end=True)
    if ap != bp: return ''
    busy = set()
    for iv in m.group(3).split(', '):
        x, y = iv.split('-'); x, xp = parse_time(x); y, yp = parse_time(y, end=True)
        if xp != yp: return ''
        busy |= set(range(x, y))
    free = [h for h in range(a, b) if h not in busy]
    gaps = []
    for h in free:
        if gaps and gaps[-1][1] == h: gaps[-1][1] = h + 1
        else: gaps.append([h, h + 1])
    if not gaps: return 'none'
    if len(gaps) > 4: return ''
    return ', '.join(f'{fmt_time(x, ap)}-{fmt_time(y, ap)}' for x, y in gaps)


# ---------------- turtle ----------------
HEAD = ['north', 'west', 'south', 'east']          # index = number of left turns from north
VEC = {'north': (0, 1), 'west': (-1, 0), 'south': (0, -1), 'east': (1, 0)}


CMD_RE = r'(?:turn (?:left|right|around)|(?:forward|backward|left|right) \d{1,2})'


def turtle_ref(args):
    m = re.fullmatch(rf'walk ({CMD_RE}(?:, {CMD_RE})*)', args)
    if not m or len(m.group(1).split(', ')) > 24: return ''
    h, x, y = 0, 0, 0
    for c in m.group(1).split(', '):
        w = c.split(' ')
        if w[0] == 'turn':
            h = (h + {'left': 1, 'around': 2, 'right': 3}[w[1]]) % 4
        else:
            r = {'forward': 0, 'left': 1, 'backward': 2, 'right': 3}[w[0]]
            dx, dy = VEC[HEAD[(h + r) % 4]]; n = int(w[1]); x += dx * n; y += dy * n
    return f'({x}, {y}) facing {HEAD[h]}'


def turtle_cmds(steps):
    """steps: list of ('turn', 'left'|'right'|'around') or ('move', 'forward'|'backward'|'left'|'right', n)."""
    return 'walk ' + ', '.join(f'turn {s[1]}' if s[0] == 'turn' else f'{s[1]} {s[2]}' for s in steps)


# ---------------- truth ----------------
CL_RE = r'\d{1,2}(?: says \d{1,2})? (?:true|false)'


def truth_ref(args):
    m = re.fullmatch(rf'truth ({CL_RE}(?:; {CL_RE})*)', args)
    if not m: return ''
    facts, edges, ment = [], [], set()
    for cl in m.group(1).split('; '):
        w = cl.split(' ')
        if len(w) == 2: k = int(w[0]); facts.append((k, w[1] == 'true')); ment.add(k)
        else: k, j = int(w[0]), int(w[2]); edges.append((k, j, w[3] == 'true')); ment |= {k, j}
    if any(not 1 <= k <= 12 for k in ment): return ''
    known = {k for k, _ in facts}; val = {k: any(v for kk, v in facts if kk == k) for k in known}
    for _ in range(12):                                # same update as the circuit: OR of the derived values
        nk, nv = set(known), {}
        for k in ment:
            d = [v for kk, v in facts if kk == k]
            for a, b, c in edges:
                for x, y in ((a, b), (b, a)):
                    if x == k and y in known: d.append(val[y] if c else not val[y])
            if d or k in known: nk.add(k); nv[k] = any(d)
        known, val = nk, nv
    bad = any(k in known and val[k] != v for k, v in facts)
    bad |= any(a in known and b in known and val[a] != (val[b] == c) for a, b, c in edges)
    if bad: return ''
    return ', '.join(f'{k} {("true" if val[k] else "false") if k in known else "unknown"}' for k in sorted(ment))


def run_ref(args):
    """NAME ARGS -> RESULT ('' = error), or None if NAME is not one of ours."""
    m = re.match(r'(interval|truth) |walk(?![a-z])', args)      # as the circuits see it: 'walk' then anything but a letter
    f = {'interval': interval_ref, 'walk': turtle_ref, 'truth': truth_ref}.get(m.group(0).strip() if m else None)
    if f is None: return None
    try:
        return f(args)
    except ValueError:
        return ''


if __name__ == '__main__':
    print(interval_ref('interval 7am-10pm busy 1pm-2pm, 2pm-3pm, 3pm-6pm, 6pm-9pm, 9pm-10pm'))
    print(interval_ref('interval 9am-5pm busy 9am-11am, 10am-12pm, 2pm-3pm'))
    print(interval_ref('interval 1-30 busy 3-8, 12-20'))
    print(turtle_ref('walk turn left, forward 7, backward 12, left 3, right 5, turn around'))
    print(truth_ref('truth 1 true; 2 says 1 true; 3 says 2 true; 4 says 3 false; 5 says 4 false'))
    print(truth_ref('truth 2 says 3 false; 1 says 2 true; 3 false; 4 says 9 true'))
    print(truth_ref('truth 1 true; 1 says 2 false; 3 says 2 true'), '|', truth_ref('truth 1 true; 2 says 1 true; 2 false'), '|', truth_ref('truth 1 true; 1 true'))
