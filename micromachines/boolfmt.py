"""Reference semantics for the boolean evaluator (circuits/boolean.py, used by the natural-text boolean implant).

EXPR is a sequence of the words True False not and or ( ), at most LMAX words, nested at most DMAX deep.
Precedence is Python's: not, then and, then or. Anything else gives an empty box.
"""
import re

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
