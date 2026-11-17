"""Register format for GSM8K (experiment 4): a calculator operand may be a REFERENCE vK = the value of the K-th box of the
current answer, instead of a re-copied number. Targets the 'went wrong midway' / 'copied a wrong number' failures.

Training data is converted mechanically from the GSM8K solutions: an operand becomes vK when it equals the K-th earlier box
(the most recent such box) and the number does not appear in the question (so true literals stay literals).
Prototype first with the Python tool (gen_eval --calc oracle derefs vK); compile into the machine only if it helps.
python regfmt.py   -> data/gsm_calc_train.jsonl, data/gsm_reg_train.jsonl (same rows; only the operands differ)"""
import json, re
from calcfmt import calc_ref, convert_solution, CALL_RE, CS, CE, OS, OE

REF = re.compile(r'v(\d{1,2})')
BOX = re.compile(re.escape(OS) + '(.*?)' + re.escape(OE), flags=re.S)
CALL = re.compile(re.escape(CS) + '(.*?)' + re.escape(CE) + re.escape(OS) + '(.*?)' + re.escape(OE), flags=re.S)


def deref(expr, before):
    """Replace vK by the K-th calculator box found in `before` (the answer text so far). None if a reference is bad."""
    boxes = BOX.findall(before)
    def sub(m):
        k = int(m.group(1))
        if not 1 <= k <= len(boxes) or not boxes[k - 1]: raise KeyError
        return boxes[k - 1]
    try: return REF.sub(sub, expr)
    except KeyError: return None


def reg_ref(expr, before):
    e = deref(expr, before) if 'v' in expr else expr
    return '' if e is None else calc_ref(e)


def operands(e):
    m = CALL_RE.match(e)
    if not m: return None
    sa, ai, af, op, sb, bi, bf = m.groups()
    a = sa + ai + ('.' + af if af else ''); b = sb + bi + ('.' + bf if bf else '')
    return a, op, b


def to_reg(question, body):
    qnums = set(re.findall(r'\d+(?:\.\d+)?', question.replace(',', '')))
    boxes, out, pos = [], [], 0
    for m in CALL.finditer(body):
        out.append(body[pos:m.start()]); pos = m.end()
        e, o = m.group(1), m.group(2)
        ops = operands(e)
        if ops:
            a, op, b = ops
            def ref(x):
                if x.lstrip('-') in qnums or x.startswith('-'): return x
                ks = [k for k, v in enumerate(boxes) if v == x]
                return f'v{ks[-1] + 1}' if ks else x
            e2 = ref(a) + op + ref(b)
            assert calc_ref(deref(e2, ''.join(f'{OS}{v}{OE}' for v in boxes)) or '') == o or not o
            e = e2
        out.append(f'{CS}{e}{CE}{OS}{o}{OE}'); boxes.append(o)
    out.append(body[pos:])
    return ''.join(out)


if __name__ == '__main__':
    import collections
    import random
    rows = [json.loads(l) for l in open('data/train.jsonl')]
    random.Random(1234).shuffle(rows); rows = rows[500:]          # same split as ft.py: the first 500 are dev
    st = collections.Counter()
    with open('data/gsm_calc_train.jsonl', 'w') as fc, open('data/gsm_reg_train.jsonl', 'w') as fr:
        for ex in rows:
            body, fin = convert_solution(ex['answer'])
            reg = to_reg(ex['question'], body)
            st['rows'] += 1; st['calls'] += body.count(CS); st['refs'] += len(REF.findall(''.join(re.findall(re.escape(CS) + '(.*?)' + re.escape(CE), reg))))
            p = f"Question: {ex['question']}\nAnswer:"
            fc.write(json.dumps({'prompt': p, 'completion': f' {body}\n#### {fin}'}) + '\n')
            fr.write(json.dumps({'prompt': p, 'completion': f' {reg}\n#### {fin}'}) + '\n')
    print(dict(st), f"refs per call {st['refs'] / st['calls']:.2f}")
    ex = next(json.loads(l) for i, l in enumerate(open('data/gsm_reg_train.jsonl')) if i == 3)
    print(ex['completion'][:600])
