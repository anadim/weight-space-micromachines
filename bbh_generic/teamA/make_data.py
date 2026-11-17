"""Build our own data sets (no BBH data anywhere):
  data/dev.jsonl    seed 0, 30 per task  (track3/5/7, order3/5/7), BBH-like style, base vocabulary
  data/test.jsonl   seed 1, 100 per task, BBH-like style, base + EXTRA vocabulary/paraphrases/domains (never trained on)
  data/oos.jsonl    seed 3, 30 per (tool, n in 3/5/7), out-of-style settings never trained on (card game; theater seating
                    and contest ranking)
  data/train.jsonl  seed 2, fine-tuning rows {prompt, completion}: BBH-like style (n = 3..7) + the train-only 'alt' style,
                    with prompts that are bare or carry OUR OWN few-shot prefix (our questions + plain CoT answers), so the
                    host learns to call the tool whatever plain-CoT examples precede the question.
"""
import json, random, os, sys
import gen_track, gen_order, gen_mix

TASKS = {'track': gen_track, 'order': gen_order}
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'data'); os.makedirs(D, exist_ok=True)

DESC = {'track': ['Work out which object each person holds after a series of exchanges.',
                  'Each story describes people trading items; determine who ends up with what.',
                  'Determine the final assignment of items to people after a sequence of swaps.'],
        'order': ['Determine the order of a set of objects from clues about their relative and absolute positions.',
                  'Each paragraph describes objects in a fixed order; use the clues to decide which statement is true.',
                  'Deduce the arrangement of several items from the given statements.']}


def eval_set(seed, per, split, style, ns=(3, 5, 7)):
    r = random.Random(seed); rows = []
    for tool, G in TASKS.items():
        for n in ns:
            for i in range(per):
                x = G.make(r, n, split, style); x.update(task=f'{tool}{n}', tool=tool, id=f'{tool}{n}_{style}_{seed}_{i}'); rows.append(x)
    return rows


def prefix(r, tool):
    """Our own few-shot prefix: a description line + k of our questions with plain CoT answers (official-prompt layout)."""
    k = r.choice([1, 2, 3, 3, 3])
    G = TASKS[tool]; shots = []
    for _ in range(k):
        x = G.make(r, r.choice([3, 3, 3, 4, 5]), 'train', 'bbh')
        shots.append(f"Q: {x['input']}\nA:{x['cot']}")
    return r.choice(DESC[tool]) + '\n\n' + '\n\n'.join(shots) + '\n\n'


MIX = {'track': gen_mix.make_track_mix, 'order': gen_mix.make_order_mix}


def train_set(seed, n_bbh, n_alt, n_mix=0):
    r = random.Random(seed); rows = []
    for tool, G in TASKS.items():
        for style, cnt in (('bbh', n_bbh), ('alt', n_alt), ('code', n_alt // 2 if tool == 'track' else 0), ('mix', n_mix)):
            for _ in range(cnt):
                nn = r.choice([3, 4, 5, 6, 7])
                x = MIX[tool](r, nn) if style == 'mix' else G.make(r, nn, 'train', style)
                pre = prefix(r, tool) if r.random() < 0.65 else ''
                body = x['solution'].lstrip()
                rows.append({'prompt': pre + f"Q: {x['input']}\nA: Let's think step by step.", 'completion': '\n' + body, 'tool': tool, 'style': style})
    r.shuffle(rows); return rows


SMALL = {'track': (('bbh', 25), ('alt', 8), ('code', 7), ('mix', 12)), 'order': (('bbh', 25), ('alt', 10), ('mix', 15))}


def train_small(seed):
    """~100 rows: per tool and style a fixed count, sizes cycled over 3..7 (stratified), 65% with our own few-shot prefix."""
    r = random.Random(seed); rows = []
    for tool, G in TASKS.items():
        for style, cnt in SMALL[tool]:
            for i in range(cnt):
                nn = [3, 4, 5, 6, 7][i % 5]
                x = MIX[tool](r, nn) if style == 'mix' else G.make(r, nn, 'train', style)
                pre = prefix(r, tool) if r.random() < 0.65 else ''
                rows.append({'prompt': pre + f"Q: {x['input']}\nA: Let's think step by step.", 'completion': '\n' + x['solution'].lstrip(), 'tool': tool, 'style': style})
    r.shuffle(rows); return rows


def dump(rows, name):
    with open(os.path.join(D, name), 'w') as f:
        for x in rows: f.write(json.dumps(x) + '\n')
    print(name, len(rows))


if __name__ == '__main__':
    what = sys.argv[1:] or ['dev', 'test', 'oos', 'train']
    if 'dev' in what: dump(eval_set(0, 30, 'dev', 'bbh'), 'dev.jsonl')
    if 'test' in what: dump(eval_set(1, 100, 'test', 'bbh'), 'test.jsonl')
    if 'oos' in what: dump(eval_set(3, 30, 'dev', 'oos'), 'oos.jsonl')
    if 'train' in what: dump(train_set(2, 4000, 1500), 'train.jsonl')
    if 'train3' in what: dump(train_set(2, 4000, 1000, 2000), 'train3.jsonl')          # v3: + generic 'mix' styles
    if 'train3_ablate' in what:
        gen_order.EXCLUDE = {'age_obj', 'age_live', 'price'}; dump(train_set(2, 4000, 1000, 2000), 'train3_ablate.jsonl'); gen_order.EXCLUDE = set()
    if 'train4' in what: dump(train_set(2, 4000, 1000, 3000), 'train4.jsonl')          # v4: mix v4 (copy/rotation/inline values/property axes)
    if 'train4_ablate' in what:
        gen_order.EXCLUDE = {'age_obj', 'age_live', 'price'}; dump(train_set(2, 4000, 1000, 3000), 'train4_ablate.jsonl'); gen_order.EXCLUDE = set()
    if 'train100' in what:                                      # author: ~100-row fine-tune, stratified over tools/styles/sizes
        dump(train_small(2), 'train100.jsonl')
    if 'train100_ablate' in what:
        gen_order.EXCLUDE = {'age_obj', 'age_live', 'price'}; dump(train_small(2), 'train100_ablate.jsonl'); gen_order.EXCLUDE = set()
    if 'train_ablate' in what:                                   # judge round 1: no age / price scales in training
        gen_order.EXCLUDE = {'age_obj', 'age_live', 'price'}; dump(train_set(2, 4000, 1500), 'train_ablate.jsonl'); gen_order.EXCLUDE = set()
