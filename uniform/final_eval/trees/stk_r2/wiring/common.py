"""Data, prompt and scoring, identical to scripts/run_gsm.py (the GSM8K/GSM-Hard harness behind the ledger numbers).

Prompt = 8 worked GSM8K train solutions (annotations stripped) tokenized shot by shot, then 'Question: q\\nAnswer:'.
Splits of GSM8K TRAIN by md5(question)%10: A = 0..5 (RL pool), B = 6,7 (dev / HP), C = 8,9 (locked). The scaled-number
(GSM-Hard-like) train set (scripts/rethink_v2/make_scaled_train.py) is split by the md5 of its SOURCE question, so a
source problem and its scaled copy are always in the same split. GSM8K test / GSM-Hard: read only by final_eval.py."""
import os, re, json, hashlib, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WSM_ROOT') or os.path.abspath(os.path.join(HERE, '..', '..', '..'))   # repo root (micromachines/, data/); launchers set WSM_ROOT
sys.path.insert(0, REPO)
from micromachines.calcfmt import strip_solution   # noqa: E402

NUM = r'####\s*(-?[\d,]*\.?\d+)'


def part(q):
    h = int(hashlib.md5(q.encode()).hexdigest(), 16) % 10
    return 'A' if h < 6 else ('B' if h < 8 else 'C')


CALLFORM = os.environ.get('CALLFORM', '0') == '1'      # call-form calculator ('<calc>a op b</calc> c') prompts / SFT data
_BIN = re.compile(r'\s*(-?\d+(?:\.\d+)?)\s*([-+*/x])\s*(-?\d+(?:\.\d+)?)\s*')
_PRE = re.compile(r'(?:\$?\d[\d,]*(?:\.\d+)?%?(?:\s+[A-Za-z/]+){0,3}\s*[-+*/x\u00d7]\s*)+\$?\d[\d,]*(?:\.\d+)?%?(?:\s+[A-Za-z/]+){0,3}\s*=\s*$')


def callform_solution(ans):
    """GSM8K gold solution with <<a op b=c>>c annotations -> (body, final) in call form: each BINARY annotation becomes
    '<calc>a op b</calc> c' (c = the calculator's exact output), replacing the restated 'a op b =' text before it; other
    annotations are stripped as in strip_solution. Returns (body, final, n_calls)."""
    from micromachines.calcfmt import calc_ref
    sol, fin = ans.split('####'); fin = fin.strip()
    out, i, n = '', 0, 0
    for m in re.finditer(r'<<([^=<>]*)=([^<>]*)>>', sol):
        pre = sol[i:m.start()]; expr = m.group(1).replace(',', ''); val = m.group(2)
        rest_start = m.end()
        if sol[rest_start:rest_start + len(val)] == val: rest_start += len(val)
        b = _BIN.fullmatch(expr)
        res = calc_ref(f"{b.group(1)}{'*' if b.group(2) == 'x' else b.group(2)}{b.group(3)}") if b else ''
        if b and res:
            pm = _PRE.search(pre)
            if pm: pre = pre[:pm.start()]
            if pre and not pre.endswith((' ', '\n', '(', '$')): pre += ' '
            op = '*' if b.group(2) == 'x' else b.group(2)
            out += pre + f'<calc>{b.group(1)} {op} {b.group(3)}</calc> {res}'; n += 1
        else:
            out += pre + sol[m.end():rest_start] if False else pre + val
        i = rest_start
    out += sol[i:]
    return out.strip(), fin, n


def prefix_ids(tok):
    shots = json.load(open(f'{REPO}/data/gsm8k/shots_v1.json'))
    T = lambda s: tok(s, add_special_tokens=False)['input_ids']
    ids = []
    for s in shots:
        if CALLFORM:
            body, fin, _ = callform_solution(s['solution'] + f"\n#### {s['answer']}")
            ids += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n"); continue
        body, fin = strip_solution(s['solution'] + f"\n#### {s['answer']}")
        ids += T(f"Question: {s['question']}\nAnswer: {body}\n#### {fin}\n\n")
    return ids, {s['question'] for s in shots}


def q_ids(tok, q):
    return tok(f"Question: {q}\nAnswer:", add_special_tokens=False)['input_ids']


def gold_of(ans):
    return ans.split('####')[1].strip().replace(',', '')


def load_train(split, which=('gsm', 'scaled'), shot_q=()):
    """-> list of dict(q, gold, kind) for split in A/B/C."""
    tr = [json.loads(l) for l in open(f'{REPO}/data/gsm8k/train.jsonl')]
    out = []
    if 'gsm' in which:
        out += [dict(q=r['question'], gold=gold_of(r['answer']), kind='gsm') for r in tr
                if part(r['question']) == split and r['question'] not in shot_q]
    if 'scaled' in which:
        for l in open(f'{REPO}/data/gsm8k/train_scaled.jsonl'):
            r = json.loads(l)
            src = tr[r['src']]['question']
            if part(src) == split and src not in shot_q:
                out.append(dict(q=r['question'], gold=gold_of(r['answer']), kind='scaled'))
    return out


def load_test(name, shot_q=()):
    """name: 'test' (GSM8K test) or 'hard' (GSM-Hard). Keeps the ORIGINAL index (for the 201+ slice)."""
    global ROUND2
    if name == 'gsmplus':   # task [20] R9: GSM-Plus testmini, 7 answerable types (data/prepare_gsmplus.py)
        ROUND2 = True
        rows = [json.loads(l) for l in open(f'{REPO}/data/gsmplus/testmini.jsonl')]
        return [dict(i=i, q=r['question'], gold=gold_of(r['answer']), ptype=r['ptype']) for i, r in enumerate(rows)
                if r['question'] not in shot_q]
    p = f'{REPO}/data/gsm8k/test.jsonl' if name == 'test' else f'{REPO}/data/gsmhard/test.jsonl'
    rows = [json.loads(l) for l in open(p)]
    return [dict(i=i, q=r['question'], gold=gold_of(r['answer'])) for i, r in enumerate(rows) if r['question'] not in shot_q]


ROUND2 = False   # set by load_test('gsmplus') only: GSM8K / GSM-Hard scoring is unchanged


def _ndec(x):
    x = str(x).strip(); return len(x.split('.')[1]) if '.' in x and 'e' not in x.lower() else 0


def _round2(pred, gold):
    """pred and gold both have >= 2 decimals and agree when rounded half-up to the shorter decimal count."""
    from decimal import Decimal, ROUND_HALF_UP
    d = min(_ndec(pred), _ndec(gold))
    if d < 2: return False
    q = Decimal(1).scaleb(-d)
    return Decimal(str(pred).strip()).quantize(q, ROUND_HALF_UP) == Decimal(str(gold).strip()).quantize(q, ROUND_HALF_UP)


def correct(pred, gold):
    try:
        if pred is None: return False
        if abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold))): return True
        return ROUND2 and _round2(pred, gold)
    except (ValueError, ArithmeticError): return False


def score(tok, ids, gold):
    """run_gsm.py scoring: text up to the first blank line, the first '#### number'."""
    txt = tok.decode(ids).replace('<|endoftext|>', '').split('\n\n')[0]
    m = re.search(NUM, txt); pred = m.group(1).replace(',', '') if m else None
    return float(correct(pred, gold)), pred is not None, txt
