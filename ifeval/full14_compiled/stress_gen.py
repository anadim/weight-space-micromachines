"""Random stress sequences for lockstep.py --seqs (focus: the forbidden-word hash paths; plus everything else).

python stress_gen.py OUT.jsonl --n 480 --seed 0 --kind forbid|mixed|random
forbid: IFEval-style forbidden-word instructions with random words (single- and multi-token, odd case/punctuation), answers
        that write those words in many tokenizations (teacher tokens; the lockstep runs closed-loop).
mixed : random combinations of all 14 instruction kinds + random answers.
random: uniformly random token ids (prompt and answer), with role tokens in place.
"""
import argparse, json, random, re
from transformers import AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--n', type=int, default=480); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--kind', default='forbid'); ap.add_argument('--ans_len', type=int, default=300)
a = ap.parse_args()
R = random.Random(a.seed)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
vocab = tok.convert_ids_to_tokens(list(range(151643)))
words = sorted({t.lstrip('Ġ').lower() for t in vocab if re.fullmatch(r'Ġ?[a-zA-Z]{3,10}', t)})


def rword():
    r = R.random()
    if r < 0.5: return R.choice(words)
    if r < 0.75: return ''.join(R.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(R.randint(4, 14)))      # multi-piece
    if r < 0.85: return R.choice(words) + R.choice(["'s", '-' + R.choice(words), '2', 'ing', 'ed'])
    return R.choice(words) + R.choice(words)


def variant(w):
    r = R.random()
    v = w.upper() if r < 0.15 else (w.capitalize() if r < 0.4 else w)
    return R.choice(['', ' ', ' ', ' ', '"', "'", '(', ' "', '\n']) + v + R.choice(['', '', ',', '.', '!', '"', "'s", 's', '-like', ' ', ':'])


TASKS = ['Write a short story about {}.', 'Write an essay on {}.', 'Describe {} to a child.', 'Write a poem about {}.',
         'Explain how {} works.', 'Give me three facts about {}.', 'Write a letter to a friend about {}.']
FORB = ['Do not include keywords {} in the response.', 'Do not include the words {} in your response.', 'Avoid using the words {}.',
        'Answer without using the word {}.', 'The words {} should not appear in your answer.', 'Do not use any of the following words: {}.',
        "Please don't say {} in your reply.", 'Make sure the word {} is not used.', 'Never mention {}.', 'Refrain from using {} anywhere.']
OTHER = ['Do not use any commas.', 'Your entire response should be in lowercase letters.', 'Your answer must be in all capital letters.',
         'Wrap your entire response with double quotation marks.', 'Finish your response with this exact phrase "{}". No other words should follow.',
         'At the end of your response, please explicitly add a postscript starting with P.S.', 'Give two different responses separated by 6 asterisk symbols ******.',
         'There should be {} paragraphs. Paragraphs are separated with the markdown divider: ***', 'Your response should contain {} paragraphs separated by two new lines. The {} paragraph must start with the word {}.',
         'Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.', 'Highlight at least {} sections in your answer with markdown, i.e. *highlighted section*.',
         'Your answer must contain exactly {} bullet points. Use the markdown bullet points such as:\n* This is point 1.', 'First repeat the request word for word without change, then give your answer.']
ORD = ['first', 'second', 'third', '2nd', '3rd', '1st']


def quoted(ws):
    st = R.random()
    if st < 0.3: return ', '.join(f'"{w}"' for w in ws)
    if st < 0.5: return ', '.join(f"'{w}'" for w in ws)
    if st < 0.7: return ' and '.join(ws)
    if st < 0.85: return ', '.join(ws[:-1]) + (' or ' + ws[-1] if len(ws) > 1 else ws[-1])
    return ' '.join(ws)


def make(kind):
    topic = ' '.join(R.choice(words) for _ in range(R.randint(1, 3)))
    if kind == 'random':
        p = [R.randrange(151643) for _ in range(R.randint(5, 120))]
        pre = tok('<|im_start|>user\n', add_special_tokens=False).input_ids
        post = tok('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False).input_ids
        ans = [R.randrange(151643) for _ in range(R.randint(10, a.ans_len))]
        return pre + p + post, ans, []
    ws = [rword() for _ in range(R.randint(1, 4))]
    parts = [R.choice(TASKS).format(topic)]
    if kind == 'forbid' or R.random() < 0.4:
        parts.append(R.choice(FORB).format(quoted(ws)))
    if kind == 'mixed' or R.random() < 0.3:
        for o in R.sample(OTHER, R.randint(1, 3)):
            parts.append(o.format(*([R.choice([str(R.randint(1, 6)), 'three', 'two', 'four']), R.choice(ORD), rword()] if o.count('{}') == 3 else
                                    [R.choice(['Is there anything else I can help with?', 'Peace!', rword()])] if 'phrase' in o else
                                    [str(R.randint(1, 6))] * o.count('{}'))))
    R.shuffle(parts)
    prompt = ' '.join(parts) if R.random() < 0.7 else '\n'.join(parts)
    ids = tok(tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False),
              add_special_tokens=False).input_ids
    # answer: random text, forbidden words in many spellings, paragraph breaks, stars, bullets
    out = []
    while len(out) < R.randint(40, a.ans_len):
        r = R.random()
        if r < 0.35: s = variant(R.choice(ws))
        elif r < 0.8: s = ' ' + ' '.join(R.choice(words) for _ in range(R.randint(1, 6)))
        else: s = R.choice(['\n\n', '.', ',', '\n* ', '\n- ', ' ***\n\n', '\n\n******\n\n', ' *bold*', ' "', 'P.S. ', '<<', '>>'])
        out += tok(s, add_special_tokens=False).input_ids
    return ids, out, ws


with open(a.out, 'w') as f:
    for i in range(a.n):
        p, ans, ws = make(a.kind)
        f.write(json.dumps({'key': i, 'prompt_ids': p, 'answer_ids': ans, 'words': ws}) + '\n')
print('wrote', a.n)
