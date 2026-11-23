"""BBH-style environment definitions. OUR generated questions only; real BBH (bbh27/data) is read by eval_bbh.py once.

env 'bbhc'    : Team-C tasks (multistep_arithmetic_two, boolean_expressions, object_counting), Team-C PASSIVE tools
                (calc_gen, expr_gen, bool_gen; fire at the host's own '=' / 'is'), host Qwen3-0.6B-Base for all arms.
env 'bbhcall' : all teams' tasks (A: track3/5/7, order3/5/7; B: temporal, navigate, lies; C: the three above) with the
                8 integrated tools of integ.TOOLS_V2 (A/B in CALL form <tool_call>..</tool_call><tool_response>..,
                Team C passive) + Team-C veto; host = ft_int_v2 (SFT'd on OUR generated call transcripts).

Splits = generator seeds: TR (RL pool), DV (HP tuning / dev curves), LK (locked). Seeds 100/101/102, 'test'-style
(widest) phrasing; any item equal to a frozen team dev/test/oos item is dropped.
Prompts: the official BBH 3-shot CoT prompt of the task (team cot/ copies) + "Q: ..\\nA: Let's think step by step."
Scoring: the team harness norms (eval_integ.py normA/normB/normC) -- the same rule the final BBH scorer uses.
"""
import os, sys, re, json, random, glob

REPO = os.environ.get('WSM_REPO_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))   # repo root
INTEG = os.environ.get('WSM_INTEG_DIR', os.path.join(REPO, 'bbh_generic', 'integrate', 'code'))
G = os.environ.get('WSM_BBH_GENERIC', os.path.join(REPO, 'bbh_generic'))   # team{A,B,C} generators + cot/ (pod layout: INTEG/bbh_generic)
SEED = {'TR': 100, 'DV': 101, 'LK': 102}
TASKS_C = ('multistep_arithmetic_two', 'boolean_expressions', 'object_counting')
BN = {'temporal': 'temporal_sequences', 'navigate': 'navigate', 'lies': 'web_of_lies'}
QA = lambda q: f"Q: {q}\nA: Let's think step by step."
_COT = {}


def cot(path):
    if path not in _COT:
        _COT[path] = open(path).read().split('-----', 1)[1].strip() + '\n\n'
    return _COT[path]


# ---------------- scoring (copied from bbh_integrate/code/eval_integ.py) ----------------
def extract(text):
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', text.split('\n\nQ:')[0]); return m[-1].strip() if m else ''


def normA(x):
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x): return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    return x.lower()


def normB(x, task):
    x = x.strip().strip('.').strip()
    if task == 'temporal':
        m = re.match(r'\(?([A-D])\)?(\s.*)?$', x); return '(' + m.group(1) + ')' if m else x
    x = re.sub(r'\s+', ' ', x.lower())
    return re.sub(r'(\d(?:am|pm)?)\s*(?:-|to)\s*(\d)', r'\1-\2', x)


def normC(x, task):
    x = x.strip().strip('.').strip()
    if task in ('multistep_arithmetic_two', 'object_counting') or task.startswith('num'):
        m = re.search(r'-?\d+', x.replace(',', '')); return m.group() if m else x
    return x.lower()


def score(text, row):
    t = text.replace('<|endoftext|>', ''); p = extract(t); fam, task, gold = row['fam'], row['task'], row['gold']
    if fam == 'A': return float(normA(p) == normA(gold))
    if fam == 'B': return float(normB(p, task) == normB(gold, task))
    return float(normC(p, task) == normC(gold, task))


# ---------------- generators ----------------
def _path(p):
    if p not in sys.path: sys.path.insert(0, p)


def _gen_C(split, n):
    _path(os.path.join(G, 'teamC')); import gen_tasks as GT
    rows = []
    for t in TASKS_C:
        banned = set()
        for f in glob.glob(os.path.join(G, 'teamC', 'data', '*', f'{t}.json')):
            banned |= {e['input'] for e in json.load(open(f))['examples']}
        r1 = random.Random(f'{t}-1')
        for _ in range(100):                                # frozen FRESH (seed 1), whether or not on disk
            res = GT.TASKS[t](r1, 'test'); banned.add(res if t == 'multistep_arithmetic_two' else res[0])
        r = random.Random(f'{t}-{SEED[split]}'); seen = set()
        while len([x for x in rows if x['task'] == t]) < n:
            res = GT.TASKS[t](r, 'test')
            inp, tgt = (res, str(eval(res[:-2]))) if t == 'multistep_arithmetic_two' else res
            if inp in seen or inp in banned: continue
            seen.add(inp)
            rows.append({'fam': 'C', 'task': t, 'q': inp, 'gold': tgt,
                         'prompt': cot(os.path.join(G, 'teamC', 'cot', f'{t}.txt')) + QA(inp)})
    return rows


def _gen_A(split, n_per):
    _path(os.path.join(G, 'teamA')); import gen_track, gen_order
    banned = set()
    for f in glob.glob(os.path.join(G, 'teamA', 'data', '*.jsonl')):
        for l in open(f):
            e = json.loads(l); banned.add(e.get('input', ''))
    P = {'track': cot(os.path.join(G, 'teamA', 'cot', 'tracking_shuffled_objects_three_objects.txt')),
         'order': cot(os.path.join(G, 'teamA', 'cot', 'logical_deduction_three_objects.txt'))}
    r = random.Random(f'A-{SEED[split]}'); rows = []
    for tool, Gm in (('track', gen_track), ('order', gen_order)):
        for nn in (3, 5, 7):
            k = 0
            while k < n_per:
                x = Gm.make(r, nn, 'test', 'bbh')
                if x['input'] in banned: continue
                banned.add(x['input']); k += 1
                rows.append({'fam': 'A', 'task': f'{tool}{nn}', 'q': x['input'], 'gold': x['target'],
                             'prompt': P[tool] + QA(x['input'])})
    return rows


def _gen_B(split, n):
    _path(os.path.join(G, 'teamB')); import gens
    banned = set()
    for f in glob.glob(os.path.join(G, 'teamB', 'data', '*.jsonl')):
        for l in open(f):
            banned.add(json.loads(l).get('question', ''))
    rows = []
    old = gens.SEEDS['test']
    try:
        gens.SEEDS['test'] = SEED[split]                    # 'test'-style phrasing, held-out seed
        for t in ('temporal', 'navigate', 'lies'):
            for x in gens.make(t, 'test', n + 20):
                if x['question'] in banned: continue
                if len([y for y in rows if y['task'] == t]) >= n: break
                rows.append({'fam': 'B', 'task': t, 'q': x['question'], 'gold': x['target'],
                             'prompt': cot(os.path.join(G, 'teamB', 'cot', f'{BN[t]}.txt')) + QA(x['question'])})
    finally:
        gens.SEEDS['test'] = old
    return rows


class Env:
    def __init__(self, name):
        self.name = name
        if name == 'bbhc':
            self.machines = ('calc_gen', 'expr_gen', 'bool_gen')
            self.host = 'Qwen/Qwen3-0.6B-Base'
            self.max_new = 256; self.eval_max_new = 400
        elif name == 'bbhcall':
            self.machines = ('track_gen', 'order_gen', 'interval_gen', 'turtle_gen', 'truth_gen',
                             'expr_gen', 'calc_gen', 'bool_gen')          # == integ.TOOLS_V2
            self.host = os.environ.get('FT_INT_V2', os.path.join(REPO, 'bbh_generic', 'integrate', 'runs', 'ft_int_v2'))
            self.max_new = 400; self.eval_max_new = 600
        else:
            raise ValueError(name)

    def rows(self, split, n):
        """n = approx items per task."""
        if self.name == 'bbhc':
            return _gen_C(split, n)
        return _gen_A(split, max(1, n // 2)) + _gen_B(split, n) + _gen_C(split, n)

    def reward(self, text, row):
        return score(text, row)


if __name__ == '__main__':
    for name in sys.argv[1:] or ['bbhc', 'bbhcall']:
        e = Env(name)
        for sp in ('TR', 'DV'):
            rs = e.rows(sp, 4)
            from collections import Counter
            print(name, sp, len(rs), Counter(r['task'] for r in rs))
        print(rs[0]['prompt'][-300:], '| gold', rs[0]['gold'])
        tr = {r['q'] for r in e.rows('TR', 30)}; dv = {r['q'] for r in e.rows('DV', 30)}
        print('TR∩DV', len(tr & dv))
