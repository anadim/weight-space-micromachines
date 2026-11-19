"""Data, prompt and scoring, identical to scripts/run_gsm.py (the GSM8K/GSM-Hard harness behind the ledger numbers).

Prompt = 8 worked GSM8K train solutions (annotations stripped) tokenized shot by shot, then 'Question: q\\nAnswer:'.
Splits of GSM8K TRAIN by md5(question)%10: A = 0..5 (RL pool), B = 6,7 (dev / HP), C = 8,9 (locked). The scaled-number
(GSM-Hard-like) train set (scripts/rethink_v2/make_scaled_train.py) is split by the md5 of its SOURCE question, so a
source problem and its scaled copy are always in the same split. GSM8K test / GSM-Hard: read only by final_eval.py."""
import os, re, json, hashlib, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WSM_ROOT', os.path.dirname(os.path.dirname(HERE)))   # repo root (micromachines/, data/)
sys.path.insert(0, REPO)
from micromachines.calcfmt import strip_solution   # noqa: E402

NUM = r'####\s*(-?[\d,]*\.?\d+)'


def part(q):
    h = int(hashlib.md5(q.encode()).hexdigest(), 16) % 10
    return 'A' if h < 6 else ('B' if h < 8 else 'C')


def prefix_ids(tok):
    shots = json.load(open(f'{REPO}/data/gsm8k/shots_v1.json'))
    T = lambda s: tok(s, add_special_tokens=False)['input_ids']
    ids = []
    for s in shots:
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
    p = f'{REPO}/data/gsm8k/test.jsonl' if name == 'test' else f'{REPO}/data/gsmhard/test.jsonl'
    rows = [json.loads(l) for l in open(p)]
    return [dict(i=i, q=r['question'], gold=gold_of(r['answer'])) for i, r in enumerate(rows) if r['question'] not in shot_q]


def correct(pred, gold):
    try: return pred is not None and abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    except ValueError: return False


def score(tok, ids, gold):
    """run_gsm.py scoring: text up to the first blank line, the first '#### number'."""
    txt = tok.decode(ids).replace('<|endoftext|>', '').split('\n\n')[0]
    m = re.search(NUM, txt); pred = m.group(1).replace(',', '') if m else None
    return float(correct(pred, gold)), pred is not None, txt
