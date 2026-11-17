"""Team B question generators (own vocabulary, several phrasings, BBH-like surface form) + training completions.

Tasks (BBH style): temporal (temporal_sequences), navigate, lies (web_of_lies).
Train-only auxiliary styles: lab (interval, plain integer hours), hiker (turtle), rumor (truth, any order).
Out-of-style EVAL sets (rule 12, never trained): meeting (interval tool on a meeting calendar), robot (turtle tool on a delivery-robot route),
knights (truth tool on a knights-and-knaves island story, references in any order).

split: 'train' (seed 2), 'dev' (seed 0), 'test' (seed 1), 'oos' (seed 3 for the out-of-style sets).
The test split mixes in EXTRA phrasings (marked X_*) that train/dev never use.
Every item: {task, id, question, target, completion (training target after "A: Let's think step by step."), call}.
Nothing here is taken from BBH data files; names/places/activities are our own lists.
"""
import random, json, sys
from tools_ref import interval_ref, turtle_ref, truth_ref, turtle_cmds, fmt_time, CS, CE, OS, OE

NAMES = ['Amara', 'Bjorn', 'Chidi', 'Dalia', 'Esteban', 'Farida', 'Goran', 'Hiroshi', 'Ingrid', 'Jamal', 'Kalani', 'Lucia',
         'Mei', 'Nikolai', 'Oluwaseun', 'Priya', 'Quentin', 'Rahel', 'Santiago', 'Tamar', 'Ulla', 'Vikram', 'Wanjiru', 'Ximena',
         'Yusuf', 'Zofia', 'Anouk', 'Bilal', 'Carmen', 'Dmitri', 'Eun-ji', 'Fatou', 'Gustavo', 'Hana', 'Idris', 'Jorge',
         'Keiko', 'Leilani', 'Mateus', 'Nadia', 'Omar', 'Paloma', 'Raj', 'Sione', 'Thiago', 'Uriel', 'Valentina', 'Wei',
         'Yara', 'Zainab', 'Aiko', 'Bogdan', 'Cyrus', 'Delphine', 'Emeka', 'Freya', 'Gideon', 'Hamid', 'Isolde', 'Jun',
         'Kofi', 'Lars', 'Malika', 'Nils', 'Oksana', 'Pedro', 'Rosalind', 'Sven', 'Tendai', 'Ugo', 'Veda', 'Wiremu', 'Xiu',
         'Yannick', 'Zeynep', 'Ayesha', 'Bruno', 'Catalina', 'Darius', 'Elif', 'Fintan', 'Gulnara', 'Hector', 'Ilse', 'Joaquin',
         'Kwame', 'Ludmila', 'Mbali', 'Noor', 'Orla', 'Pavel', 'Renata', 'Soren', 'Talia', 'Umar', 'Vesna', 'Wilhelmina', 'Yosef']
PLACES = ['aquarium', 'planetarium', 'bowling alley', 'flower market', 'pottery workshop', 'climbing gym', 'ferry terminal',
          'bike shop', 'ice rink', 'laundromat', 'observatory', 'farmers market', 'post office', 'community garden',
          'karaoke bar', 'sauna', 'archery range', 'record store', 'botanical garden', 'chess club', 'tailor', 'pharmacy',
          'hardware store', 'swimming pool', 'barber shop', 'zoo', 'car wash', 'antique shop', 'tea house', 'skate park']
ACTS = ['repairing a kite at the hobby store', 'feeding ducks at the pond', 'jogging along the river', 'buying bread at the corner shop',
        'painting a fence near the school', 'reading comics at the newsstand', 'waiting for a bus at the depot',
        'playing chess in the town square', 'washing a van at the garage', 'stretching on the pier', 'eating noodles at the food court',
        'sketching birds at the marsh', 'carrying boxes into the warehouse', 'buying stamps at the kiosk', 'fishing off the jetty',
        'singing in the choir hall', 'practicing violin at the conservatory', 'picking apples at the orchard',
        'fixing a sink at a neighbor\'s flat', 'walking a dog along the canal', 'shopping for shoes at the outlet',
        'repotting plants in the greenhouse', 'riding a scooter through the plaza', 'studying at the reading room',
        'taking a dance lesson at the community center', 'queuing at the bank', 'watering lettuce at the allotment',
        'polishing trophies at the clubhouse', 'testing a drone in the field', 'swimming laps at the lido']

# ======================================================================================================================
# temporal (interval tool)
# ======================================================================================================================
T_WAKE = ['{X} woke up at {t}.', '{X} got out of bed at {t}.', "{X}'s day started at {t}, when they woke up."]
X_WAKE = ['{X} was awake from {t} onward.', 'At {t}, {X} rolled out of bed.']
T_SEEN = ['{Y} saw {X} {act} from {a} to {b}.', '{Y} spotted {X} {act} from {a} to {b}.', 'From {a} to {b}, {Y} saw {X} {act}.',
          '{Y} saw {X} {act} between {a} and {b}.']
X_SEEN = ['{Y} ran into {X}, who was {act}, from {a} until {b}.', 'Between {a} and {b}, {X} was seen {act} by {Y}.',
          '{X} was {act} from {a} to {b}, according to {Y}.']
T_CLOSE = ['The {P} was closed after {t}.', 'The {P} closed at {t}.', 'After {t}, the {P} was closed.']
X_CLOSE = ['The {P} shut its doors at {t}.', 'Nobody could enter the {P} after {t}.']
T_INTRO = ['Today, {X} went to the {P}. Between what times could they have gone?', '{X} visited the {P} today. Between what times could they have gone?']
X_INTRO = ['At some point today, {X} stopped by the {P}. Between what times could they have gone?']
T_ASK = ['Between what times could {X} have gone to the {P}?', 'When could {X} have gone to the {P}?']
X_ASK = ['During which window could {X} have visited the {P}?']
X_NOISE = ['{X} had soup for lunch.', 'It rained in the afternoon.', '{Y} owns a red bicycle.']


def pick(r, base, extra, test):
    return r.choice(base + extra) if test and r.random() < 0.5 else r.choice(base)


def temporal_item(r, test=False, idx=0):
    X = r.choice(NAMES); P = r.choice(PLACES)
    nsee = r.choice([3, 4, 4, 5, 5, 6])
    while True:
        w = r.randint(5, 12); c = r.randint(w + nsee + 1, min(w + 16, 23))
        cuts = sorted(r.sample(range(w + 1, c), nsee))           # nsee+1 segments of [w, c)
        segs = list(zip([w] + cuts, cuts + [c]))
        if len(segs) == nsee + 1: break
    gi = r.randrange(len(segs)); gap = segs[gi]; busy = [s for i, s in enumerate(segs) if i != gi]
    seers = r.sample([n for n in NAMES if n != X], nsee); acts = r.sample(ACTS, nsee)
    f = lambda h: fmt_time(h, True)
    lines = [pick(r, T_WAKE, X_WAKE, test).format(X=X, t=f(w))]
    order = list(range(nsee))
    if test and r.random() < 0.3: r.shuffle(order)
    for i in order:
        a, b = busy[i]; lines.append(pick(r, T_SEEN, X_SEEN, test).format(X=X, Y=seers[i], act=acts[i], a=f(a), b=f(b)))
    if test and r.random() < 0.25: lines.insert(r.randint(1, len(lines)), r.choice(X_NOISE).format(X=X, Y=r.choice(seers)))
    lines.append(pick(r, T_CLOSE, X_CLOSE, test).format(P=P, t=f(c)))
    opts = [gap] + r.sample(busy, 3)
    r.shuffle(opts); L = 'ABCD'[opts.index(gap)]
    q = pick(r, T_INTRO, X_INTRO, test).format(X=X, P=P) + '\nWe know that:\n' + '\n'.join(lines) + '\n' + \
        pick(r, T_ASK, X_ASK, test).format(X=X, P=P) + '\nOptions:\n' + '\n'.join(f'({"ABCD"[i]}) {f(a)} to {f(b)}' for i, (a, b) in enumerate(opts))
    call = f'interval {f(w)}-{f(c)} busy ' + ', '.join(f'{f(a)}-{f(b)}' for a, b in [busy[i] for i in order])
    resp = interval_ref(call)
    assert resp == f'{f(gap[0])}-{f(gap[1])}', (resp, gap)
    comp = f'\n{CS}{call}{CE}{OS}{resp}{OE}\nThe only time when {X} could have gone to the {P} was {f(gap[0])} to {f(gap[1])}. So the answer is ({L}).'
    return dict(task='temporal', question=q, target=f'({L})', completion=comp, call=call)


# ======================================================================================================================
# navigate (turtle tool)
# ======================================================================================================================
def n_steps(n): return f'{n} step' + ('' if n == 1 else 's')


T_TURN = {'left': ['Turn left.'], 'right': ['Turn right.'], 'around': ['Turn around.', 'Turn around to face the other way.']}
X_TURN = {'left': ['Pivot to your left.', 'Rotate 90 degrees to the left.'], 'right': ['Pivot to your right.', 'Rotate 90 degrees to the right.'],
          'around': ['Do an about-face.', 'Spin 180 degrees.']}
T_MOVE = {'forward': ['Take {s}.', 'Take {s} forward.', 'Walk {s} forward.'], 'backward': ['Take {s} backward.', 'Walk {s} backward.'],
          'left': ['Take {s} left.', 'Take {s} to the left.'], 'right': ['Take {s} right.', 'Take {s} to the right.']}
X_MOVE = {'forward': ['Advance {n} paces.', 'Go straight ahead for {s}.'], 'backward': ['Back up {s}.', 'Retreat {n} paces.'],
          'left': ['Sidestep {s} to your left.', 'Shuffle {n} paces left.'], 'right': ['Sidestep {s} to your right.', 'Shuffle {n} paces right.']}
T_NQ = ['If you follow these instructions, do you return to the starting point?', 'If you follow these instructions, do you end up where you started?']
X_NQ = ['After carrying out these instructions, are you back at your starting spot?']


def nav_steps(r, ff, nmove, maxn):
    st = []
    for _ in range(nmove):
        if not ff and r.random() < 0.45: st.append(('turn', r.choice(['left', 'right', 'around'])))
        d = r.choice(['forward', 'backward', 'left', 'right']) if ff else r.choice(['forward'] * 4 + ['backward'])
        st.append(('move', d, r.randint(1, maxn)))
    return st


def pos_of(steps):
    out = turtle_ref(turtle_cmds(steps)); xy = out.split(')')[0][1:]; return tuple(int(v) for v in xy.split(', '))


def make_return(r, steps):
    """Adjust step counts (and maybe append moves) so the walk ends at the origin. Returns None if it fails."""
    for _ in range(50):
        x, y = pos_of(steps)
        if (x, y) == (0, 0): return steps
        mv = [i for i, s in enumerate(steps) if s[0] == 'move']
        i = r.choice(mv)
        s = steps[i]
        # effect of +1 on this move
        base = pos_of(steps)
        steps2 = list(steps); steps2[i] = (s[0], s[1], s[2] + 1); px, py = pos_of(steps2); dx, dy = px - base[0], py - base[1]
        # change n so that the component along (dx,dy) cancels
        k = -(x * dx + y * dy)
        if k != 0 and 1 <= s[2] + k <= 10:
            steps[i] = (s[0], s[1], s[2] + k)
    return steps if pos_of(steps) == (0, 0) else None


def navigate_item(r, test=False, idx=0):
    want_yes = r.random() < 0.5
    ff = r.random() < 0.4
    while True:
        nmove = r.choice([2, 3, 3, 4, 4, 5, 6]); steps = nav_steps(r, ff, nmove, 10)
        if want_yes:
            s2 = make_return(r, list(steps))
            if s2 is None: continue
            steps = s2
        elif pos_of(steps) == (0, 0): continue
        break
    sents = ['Always face forward.'] if ff else []
    for s in steps:
        if s[0] == 'turn': sents.append(r.choice(T_TURN[s[1]] + (X_TURN[s[1]] if test and r.random() < 0.5 else [])))
        else:
            pool = T_MOVE[s[1]] + (X_MOVE[s[1]] if test and r.random() < 0.5 else [])
            if ff and s[1] == 'forward': pool = [p for p in pool if p != 'Take {s}.']
            sents.append(r.choice(pool).format(s=n_steps(s[2]), n=s[2]))
    q = pick(r, T_NQ, X_NQ, test) + ' ' + ' '.join(sents) + '\nOptions:\n- Yes\n- No'
    call = turtle_cmds(steps); resp = turtle_ref(call); x, y = pos_of(steps)
    ans = 'Yes' if (x, y) == (0, 0) else 'No'
    concl = f'Since ({x}, {y}) is (0, 0), we are indeed where we started.' if ans == 'Yes' else f'Since ({x}, {y}) is not (0, 0), we are not where we started.'
    comp = f'\n{CS}{call}{CE}{OS}{resp}{OE}\n{concl} So the answer is {ans}.'
    return dict(task='navigate', question=q, target=ans, completion=comp, call=call)


# ======================================================================================================================
# web of lies (truth tool)
# ======================================================================================================================
T_BASE = {True: ['{A} tells the truth.', '{A} is honest.'], False: ['{A} lies.', '{A} is a liar.']}
X_BASE = {True: ['{A} always speaks truthfully.'], False: ['{A} never tells the truth.']}
T_SAY = {True: ['{B} says {A} tells the truth.', '{B} claims that {A} tells the truth.', '{B} says {A} is honest.'],
         False: ['{B} says {A} lies.', '{B} claims that {A} lies.', '{B} says {A} is lying.']}
X_SAY = {True: ['According to {B}, {A} is truthful.', '{B} insists that {A} is telling the truth.'],
         False: ['According to {B}, {A} is a liar.', '{B} insists that {A} is not telling the truth.']}
T_LQ = ['Does {Z} tell the truth?', 'Is {Z} telling the truth?']
X_LQ = ['Is {Z} honest?']


def lies_item(r, test=False, idx=0):
    n = r.choice([3, 4, 5, 5, 5, 6, 7]); ppl = r.sample(NAMES, n)
    base = r.random() < 0.5; claims = [r.random() < 0.5 for _ in range(n - 1)]
    sents = [pick(r, T_BASE[base], X_BASE[base], test).format(A=ppl[0])]
    for i in range(1, n):
        pool = T_SAY[claims[i - 1]] + (X_SAY[claims[i - 1]] if test and r.random() < 0.5 else [])
        sents.append(r.choice(pool).format(B=ppl[i], A=ppl[i - 1]))
    q = 'Question: ' + ' '.join(sents) + ' ' + pick(r, T_LQ, X_LQ, test).format(Z=ppl[-1])
    call = 'truth ' + '; '.join([f'1 {"true" if base else "false"}'] + [f'{i + 1} says {i} {"true" if c else "false"}' for i, c in enumerate(claims, 1)])
    resp = truth_ref(call); last = resp.split(', ')[-1].split(' ')[1]
    ans = 'Yes' if last == 'true' else 'No'
    mapping = ', '.join(f'{p} = {i + 1}' for i, p in enumerate(ppl))
    comp = f'\n{mapping}.\n{CS}{call}{CE}{OS}{resp}{OE}\n{ppl[-1]} is {n}, and {n} is {last}. So {ppl[-1]} {"tells the truth" if last == "true" else "lies"}. So the answer is {ans}.'
    return dict(task='lies', question=q, target=ans, completion=comp, call=call)


# ======================================================================================================================
# out-of-style EVAL settings (rule 12), never used for training. Prompt for these: zero-shot "Q: ...\nA: Let's think step by step."
# ======================================================================================================================
def meeting_item(r, test=False, idx=0):
    k = r.choice([2, 3, 4]); ppl = r.sample(NAMES, k)
    while True:
        a = r.randint(7, 10); b = r.randint(a + 6, min(a + 10, 19))
        busy = []
        for p in ppl:
            for _ in range(r.randint(1, 3)):
                x = r.randint(a - 1, b - 1); y = min(x + r.randint(1, 3), b + 1); busy.append((p, x, y))
        call = f'interval {fmt_time(a, True)}-{fmt_time(b, True)} busy ' + ', '.join(f'{fmt_time(x, True)}-{fmt_time(y, True)}' for _, x, y in busy)
        resp = interval_ref(call)
        if resp != 'none' and ',' not in resp: break          # exactly one common free slot
    lines = []
    for p in ppl:
        mine = [(x, y) for q, x, y in busy if q == p]
        lines.append(f"{p}'s calendar has " + ' and '.join(f'a meeting from {fmt_time(x, True)} to {fmt_time(y, True)}' for x, y in mine) + '.')
    r.shuffle(lines)
    q = (f'{", ".join(ppl[:-1])} and {ppl[-1]} want to find a time to meet today. The office is open from {fmt_time(a, True)} to '
         f'{fmt_time(b, True)}. ' + ' '.join(lines) + ' When are all of them free? Answer with the time range.')
    x, y = resp.split('-')
    comp = f'\n{CS}{call}{CE}{OS}{resp}{OE}\nEveryone is free from {x} to {y}. So the answer is {x} to {y}.'
    return dict(task='meeting', question=q, target=f'{x} to {y}', completion=comp, call=call)


R_MOVE = {'forward': ['drives {n} blocks forward', 'rolls ahead {n} blocks'], 'backward': ['reverses {n} blocks', 'backs up {n} blocks']}
R_TURN = {'left': ['turns left'], 'right': ['turns right'], 'around': ['makes a U-turn']}


def robot_item(r, test=False, idx=0):
    steps = []
    for _ in range(r.randint(3, 7)):
        if r.random() < 0.5: steps.append(('turn', r.choice(['left', 'right', 'around'])))
        steps.append(('move', r.choice(['forward', 'forward', 'backward']), r.randint(1, 12)))
    parts = [r.choice(R_TURN[s[1]]) if s[0] == 'turn' else r.choice(R_MOVE[s[1]]).format(n=s[2]) for s in steps]
    q = ('A delivery robot leaves its charging dock facing north. Then it ' + ', then it '.join(parts) +
         '. Where is the robot now, as (east, north) blocks from the dock?')
    call = turtle_cmds(steps); resp = turtle_ref(call); pos = resp.split(' facing')[0]
    comp = f'\n{CS}{call}{CE}{OS}{resp}{OE}\nThe robot is at {pos}. So the answer is {pos}.'
    return dict(task='robot', question=q, target=pos, completion=comp, call=call)


def knights_item(r, test=False, idx=0):
    n = r.randint(3, 6); ppl = r.sample(NAMES, n)
    # a random tree: person 0 is described directly; every other person speaks about someone earlier in the tree
    stm, cls = [], []
    root = r.random() < 0.5
    stm.append(f'{ppl[0]} is a {"knight" if root else "knave"}.'); cls.append((1, None, root))
    for i in range(1, n):
        j = r.randrange(i); c = r.random() < 0.5
        stm.append(f'{ppl[i]} says: "{ppl[j]} is a {"knight" if c else "knave"}."'); cls.append((i + 1, j + 1, c))
    order = list(range(n)); r.shuffle(order)
    Z = r.randrange(n)
    q = ('On an island, knights always tell the truth and knaves always lie. ' + ' '.join(stm[i] for i in order) +
         f' Is {ppl[Z]} a knight?')
    # number people in order of appearance in the story
    num = {o: k + 1 for k, o in enumerate(order)}
    cl = []
    for o in order:
        k, j, c = cls[o]
        cl.append(f'{num[o]} {"true" if c else "false"}' if j is None else f'{num[o]} says {num[j - 1]} {"true" if c else "false"}')
    call = 'truth ' + '; '.join(cl); resp = truth_ref(call)
    val = dict(x.split(' ') for x in resp.split(', '))[str(num[Z])]
    ans = 'Yes' if val == 'true' else 'No'
    mapping = ', '.join(f'{ppl[o]} = {num[o]}' for o in order)
    comp = f'\n{mapping}.\n{CS}{call}{CE}{OS}{resp}{OE}\n{ppl[Z]} is {num[Z]}, and {num[Z]} is {val}. So the answer is {ans}.'
    return dict(task='knights', question=q, target=ans, completion=comp, call=call)


# ---------------- auxiliary TRAIN-ONLY styles (never evaluated): teach the host that the tools are general -------------
def lab_item(r, test=False, idx=0):
    """interval tool with plain integers (24-hour clock hours)"""
    while True:
        a = r.randint(6, 10); b = r.randint(a + 6, 22); k = r.randint(2, 8)
        bk = []
        for _ in range(k):
            x = r.randint(a, b - 1); y = min(x + r.randint(1, 4), b); bk.append((x, y))
        call = f'interval {a}-{b} busy ' + ', '.join(f'{x}-{y}' for x, y in bk)
        resp = interval_ref(call)
        if resp != 'none': break
    q = (f'The shared lab is open from hour {a} to hour {b} (24-hour clock). It is already booked ' +
         ', '.join(f'from {x} to {y}' for x, y in bk) + '. Which hours are still open for booking?')
    nice = ' and '.join(g.replace('-', ' to ') for g in resp.split(', '))
    comp = f'\n{CS}{call}{CE}{OS}{resp}{OE}\nThe lab is free from {nice}. So the answer is {nice}.'
    return dict(task='lab', question=q, target=nice, completion=comp, call=call)


H_MOVE = {'forward': ['walks {n} km straight on', 'hikes {n} km ahead', 'marches {n} km forward', 'continues for {n} km'],
          'backward': ['walks {n} km backwards', 'goes {n} km back without turning', 'steps {n} km backwards'],
          'left': ['steps {n} km to her left', 'shifts {n} km sideways to the left'], 'right': ['steps {n} km to her right', 'shifts {n} km sideways to the right']}
H_TURN = {'left': ['turns to her left', 'turns left'], 'right': ['turns to her right', 'turns right'], 'around': ['turns back', 'faces the opposite way']}


def hiker_item(r, test=False, idx=0):
    """turtle tool, final position and back-at-camp questions"""
    steps = []
    for _ in range(r.randint(2, 6)):
        if r.random() < 0.5: steps.append(('turn', r.choice(['left', 'right', 'around'])))
        steps.append(('move', r.choice(['forward', 'forward', 'backward', 'left', 'right']), r.randint(1, 15)))
    parts = [r.choice(H_TURN[s[1]]) if s[0] == 'turn' else r.choice(H_MOVE[s[1]]).format(n=s[2]) for s in steps]
    call = turtle_cmds(steps); resp = turtle_ref(call); pos = resp.split(' facing')[0]
    if r.random() < 0.3:
        q = 'A hiker leaves camp heading north. She ' + ', then '.join(parts) + '. Is she back at camp?'
        ans = 'Yes' if pos == '(0, 0)' else 'No'
        comp = f'\n{CS}{call}{CE}{OS}{resp}{OE}\nShe ends at {pos}, which is {"" if ans == "Yes" else "not "}the camp (0, 0). So the answer is {ans}.'
        return dict(task='hiker', question=q, target=ans, completion=comp, call=call)
    q = 'A hiker leaves camp heading north. She ' + ', then '.join(parts) + '. How far east and how far north of camp is she, in km?'
    comp = f'\n{CS}{call}{CE}{OS}{resp}{OE}\nShe ends at {pos}. So the answer is {pos}.'
    return dict(task='hiker', question=q, target=pos, completion=comp, call=call)


def rumor_item(r, test=False, idx=0):
    """truth tool: a tree of 'X says that Y is honest/a liar' with the one known fact on ANY person (so values also flow
    from a speaker to the person spoken about), statements in any order, question about anyone"""
    n = r.randint(3, 6); ppl = r.sample(NAMES, n)
    edges = []
    for i in range(1, n):
        j = r.randrange(i); a, b = (i, j) if r.random() < 0.5 else (j, i); edges.append((a, b, r.random() < 0.5))
    f = r.randrange(n); fv = r.random() < 0.5
    st = [(f, None, fv)] + edges; r.shuffle(st)
    txt = [f'{ppl[a]} is {"honest" if c else "a liar"}.' if b is None else f'{ppl[a]} says that {ppl[b]} is {"honest" if c else "a liar"}.' for a, b, c in st]
    Z = r.randrange(n)
    q = 'In the village, every person is either always honest or always a liar. ' + ' '.join(txt) + f' Is {ppl[Z]} honest?'
    order = []
    for a, b, c in st:
        for x in (a, b):
            if x is not None and x not in order: order.append(x)
    num = {o: k + 1 for k, o in enumerate(order)}
    cl = [f'{num[a]} {"true" if c else "false"}' if b is None else f'{num[a]} says {num[b]} {"true" if c else "false"}' for a, b, c in st]
    call = 'truth ' + '; '.join(cl); resp = truth_ref(call)
    val = dict(x.split(' ') for x in resp.split(', '))[str(num[Z])]
    ans = 'Yes' if val == 'true' else 'No'
    mapping = ', '.join(f'{ppl[o]} = {num[o]}' for o in order)
    comp = f'\n{mapping}.\n{CS}{call}{CE}{OS}{resp}{OE}\n{ppl[Z]} is {num[Z]}, and {num[Z]} is {val}. So the answer is {ans}.'
    return dict(task='rumor', question=q, target=ans, completion=comp, call=call)


# ---------------- error recovery (TRAIN only): a first call that breaks the grammar gets the empty response, then the host
# writes the corrected call. The broken call is MASKED from the loss (ft_B.py), so the host only learns the recovery. -------
SYN = {' forward': [' ahead', ' forwards', ' straight'], ' backward': [' back', ' backwards', ' reverse'], ' around': [' back', ' about'],
       ' busy': [' booked', ' taken'], ' true': [' honest', ' truthful'], ' false': [' liar', ' lying']}


def corrupt_call(call, r):
    opts = [w for w in SYN if w in ' ' + call]
    for _ in range(10):
        k = r.random()
        if k < 0.6 and opts:
            w = r.choice(opts); c2 = (' ' + call).replace(w, r.choice(SYN[w]), 1)[1:]
        elif k < 0.8: c2 = call.replace(', ', ' ', 1).replace('; ', ' ', 1)
        else: c2 = call.replace('-', ' to ', 1) if '-' in call else call.replace(' ', '  ', 1)
        if c2 != call: return c2
    return None


def with_retry(x, r):
    call = x['call']; bad = corrupt_call(call, r)
    if bad is None: return x
    full = f'{CS}{call}{CE}'; i = x['completion'].index(full)
    y = dict(x)
    y['masked_prefix'] = x['completion'][:i] + f'{CS}{bad}{CE}{OS}{OE}'
    y['completion'] = '\nThat call was not valid, so I fix it.\n' + x['completion'][i:]
    return y


GENS = {'temporal': temporal_item, 'navigate': navigate_item, 'lies': lies_item,
        'meeting': meeting_item, 'robot': robot_item, 'knights': knights_item,
        'lab': lab_item, 'hiker': hiker_item, 'rumor': rumor_item}
SEEDS = {'dev': 0, 'test': 1, 'train': 2, 'oos': 3, 'oosdev': 4}
BBH_NAME = {'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies'}


def make(task, split, n):
    r = random.Random(f'{task}-{SEEDS[split]}')
    out = []
    for i in range(n):
        x = GENS[task](r, test=(split == 'test'), idx=i); x['id'] = f'{task}-{split}-{i}'
        if split == 'train' and r.random() < 0.15: x = with_retry(x, r)
        out.append(x)
    return out


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    for t in ('temporal', 'navigate', 'lies'):
        for split, n in (('dev', 30), ('test', 100), ('train', 3000)):
            with open(f'data/{t}_{split}.jsonl', 'w') as f:
                for x in make(t, split, n): f.write(json.dumps(x) + '\n')
    for t in ('meeting', 'robot', 'knights'):                     # out-of-style EVAL sets, never trained on
        with open(f'data/{t}_oos.jsonl', 'w') as f:                # held out (seed 3): only scores are looked at
            for x in make(t, 'oos', 50): f.write(json.dumps(x) + '\n')
        with open(f'data/{t}_oosdev.jsonl', 'w') as f:             # seed 4: host failures may be read (never trained on)
            for x in make(t, 'oosdev', 30): f.write(json.dumps(x) + '\n')
    for t in ('lab', 'hiker', 'rumor'):                           # auxiliary TRAIN-only styles
        for split, n in (('train', 1000),):
            with open(f'data/{t}_{split}.jsonl', 'w') as f:
                for x in make(t, split, n): f.write(json.dumps(x) + '\n')
    for t in GENS:
        x = make(t, 'dev' if t in BBH_NAME else 'oos' if t in ('meeting', 'robot', 'knights') else 'train', 1)[0]
        print('=' * 80); print(x['question']); print('---'); print(x['completion']); print('target', x['target'])
