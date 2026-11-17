"""Boolean evaluator micromachine (MM3) for BBH boolean_expressions: reference semantics and training data.

    <|repo_name|>EXPR<|file_sep|>  ->  box  <|fim_middle|>True|False<|fim_pad|>
EXPR is a sequence of the words True False not and or ( ), at most LMAX words, nested at most DMAX deep.
Precedence is Python's: not, then and, then or. Anything else gives an empty box.
"""
import re, json, random

CS4, CE4, OS4, OE4 = '<|repo_name|>', '<|file_sep|>', '<|fim_middle|>', '<|fim_pad|>'
WORDS = ['True', 'False', 'not', 'and', 'or', '(', ')']
LMAX, DMAX = 16, 3


def bool_ref_words(words):
    """Reference on a list of words. '' = malformed."""
    if not 1 <= len(words) <= LMAX or any(w not in WORDS for w in words): return ''
    depth, prev = 0, None                                    # prev = None at the start
    for w in words:
        want_operand = prev is None or prev in ('not', 'and', 'or', '(')
        if want_operand != (w in ('True', 'False', 'not', '(')): return ''
        if w == '(':
            depth += 1
            if depth > DMAX: return ''
        if w == ')':
            depth -= 1
            if depth < 0: return ''
        prev = w
    if depth != 0 or prev not in ('True', 'False', ')'): return ''
    return str(bool(eval(' '.join(words))))


def bool_ref_toks(tokens):
    """Reference on tokenizer tokens ('Ġand', 'True', ...): a leading space is ignored."""
    return bool_ref_words([t[1:] if t[:1] in ('Ġ', ' ') else t for t in tokens])


def bool_ref(text):
    ws = re.findall(r'True|False|not|and|or|\(|\)|[^\s()]+', text)
    return bool_ref_words(ws)


def gen_expr(r, n):
    """A random valid expression with exactly n words."""
    after_ok = lambda left, d: left == d or left >= d + 2          # can still finish after an operand
    out, depth, operand = [], 0, True
    while True:
        left = n - len(out)
        if operand:
            opts = [w for w in ('True', 'False') if after_ok(left - 1, depth)] * 3
            if left - 1 >= 1 + depth: opts += ['not'] * 2
            if depth < DMAX and left - 1 >= 2 + depth: opts += ['(']
        else:
            if left == 0: return out
            opts = [w for w in ('and', 'or') if left - 1 >= 1 + depth] * 2
            if depth > 0 and after_ok(left - 1, depth - 1): opts += [')'] * (3 if left == depth else 1)
        w = r.choice(opts); out.append(w)
        depth += (w == '(') - (w == ')')
        operand = w in ('not', 'and', 'or', '(')


def all_valid(n):
    """Every valid expression with exactly n words."""
    res = []
    def go(out, depth, operand):
        left = n - len(out)
        if not operand and left == 0:
            if depth == 0: res.append(list(out))
            return
        if left == 0: return
        for w in (('True', 'False', 'not', '(') if operand else ('and', 'or', ')')):
            d = depth + (w == '(') - (w == ')')
            if d < 0 or d > DMAX: continue
            need = (1 + d) if w in ('not', 'and', 'or', '(') else d
            if left - 1 < need: continue
            out.append(w); go(out, d, w in ('not', 'and', 'or', '(')); out.pop()
    go([], 0, True)
    return res


def make_rows(n, seed, exclude=()):
    r = random.Random(seed); rows = []; ex = set(exclude)
    while len(rows) < n:
        e = ' '.join(gen_expr(r, r.choice([2, 3, 4, 5, 6, 7, 8, 8, 8, 9, 10, 11, 12, 14, 16])))
        if e in ex: continue
        v = bool_ref(e)
        rows.append({'prompt': f'{e} is\nAnswer:', 'completion': f' {CS4}{e}{CE4}{OS4}{v}{OE4}\n#### {v}', 'kind': 'bool',
                     'question': f'{e} is\nAnswer:', 'answer': f'#### {v}', 'raw_prompt': True})
    return rows


if __name__ == '__main__':
    import sys
    bbh = []
    for path in sys.argv[1:]:
        exs = json.load(open(path))['examples']; ok = 0
        with open('data/bbh_bool.jsonl', 'w') as f:
            for e in exs:
                expr = e['input'].rsplit(' is', 1)[0]; bbh.append(expr); ok += bool_ref(expr) == e['target']
                f.write(json.dumps({'question': e['input'] + '\nAnswer:', 'answer': '#### ' + e['target'], 'raw_prompt': True}) + '\n')
        print(path.split('/')[-1], f'reference matches gold on {ok}/{len(exs)}')
    rows = make_rows(2000, 31, exclude=bbh)
    with open('data/bool_train.jsonl', 'w') as f:
        for x in rows: f.write(json.dumps(x) + '\n')
    print(rows[0]['prompt'] + rows[0]['completion']); print('valid expressions by length:', {n: len(all_valid(n)) for n in range(1, 9)})
