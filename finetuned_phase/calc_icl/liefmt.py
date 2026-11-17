"""Truth tracker micromachine (MM3) for BBH web_of_lies: reference semantics and a generator of training stories.

    <|fim_prefix|>D<|fim_suffix|>  ->  box  <|vision_start|>B<|vision_end|>
    D = 1: this person tells the truth   D = 0: this person lies            (start of a chain)
    D = 2: says the previous person tells the truth (keep the bit)          D = 3: says the previous person lies (flip it)
"""
import re, json, random

CS3, CE3, OS3, OE3 = '<|fim_prefix|>', '<|fim_suffix|>', '<|vision_start|>', '<|vision_end|>'


def truth_ref(call: str, state):
    """state: content of the most recent box (None if none). Returns the new box content ('' = error)."""
    if call in ('0', '1'): return call
    if call in ('2', '3') and state in ('0', '1'):
        return state if call == '2' else str(1 - int(state))
    return ''


def last_state3(text):
    m = re.findall(re.escape(OS3) + '(.*?)' + re.escape(OE3), text, flags=re.S)
    return m[-1] if m else None


NAMES = ['Liam', 'Noah', 'Olivia', 'Emma', 'Ava', 'Mia', 'Lucas', 'Ethan', 'Mason', 'Logan', 'Sofia', 'Isla', 'Zoe', 'Nora', 'Leo', 'Owen', 'Ruby', 'Ivy',
         'Hugo', 'Felix', 'Maya', 'Nina', 'Omar', 'Priya', 'Quinn', 'Rosa', 'Tariq', 'Uma', 'Victor', 'Wendy', 'Xavier', 'Yara', 'Zane', 'Hana', 'Kenji',
         'Lars', 'Ines', 'Joel', 'Kira', 'Mateo', 'Anton', 'Bella', 'Cyrus', 'Dalia', 'Edgar', 'Freya', 'Gus', 'Heidi']


def solve(text):
    """Parse a web_of_lies question -> (people in order, calls, final bit)."""
    sents = [s.strip() for s in text.replace('Question:', '').split('.') if s.strip()]
    people, calls, st = [], [], None
    for s in sents[:-1]:
        m = re.fullmatch(r'(\w+) says (\w+) (tells the truth|lies)', s)
        if m: people.append(m.group(1)); c = '2' if m.group(3) == 'tells the truth' else '3'
        else:
            m = re.fullmatch(r'(\w+) (tells the truth|lies)', s); people.append(m.group(1)); c = '1' if m.group(2) == 'tells the truth' else '0'
        calls.append(c); st = truth_ref(c, st)
    return people, calls, st


def make_rows(n, seed):
    r = random.Random(seed); rows = []
    for _ in range(n):
        k = r.randint(3, 8); ppl = r.sample(NAMES, k)
        sents = [f'{ppl[0]} {r.choice(["tells the truth", "lies"])}.'] + [f'{ppl[i]} says {ppl[i - 1]} {r.choice(["tells the truth", "lies"])}.' for i in range(1, k)]
        q = 'Question: ' + ' '.join(sents) + f' Does {ppl[-1]} tell the truth?'
        people, calls, st = solve(q)
        lines, cur = [], None
        for p, c in zip(people, calls):
            cur = truth_ref(c, cur); lines.append(f'{p}: {CS3}{c}{CE3}{OS3}{cur}{OE3}')
        ans = 'Yes' if st == '1' else 'No'
        rows.append({'prompt': f'{q}\nAnswer:', 'completion': ' ' + '\n'.join(lines) + f'\n{people[-1]}: {st}, {ans}.\n#### {ans}', 'kind': 'lies',
                     'question': q, 'answer': f'#### {ans}', 'raw_prompt': True})
    return rows


if __name__ == '__main__':
    import sys
    with open('data/lies_train.jsonl', 'w') as f:
        for x in make_rows(2000, 21): f.write(json.dumps({**x, 'prompt': x['prompt']}) + '\n')
    x = make_rows(1, 5)[0]; print(x['prompt'] + x['completion'])
    for path in sys.argv[1:]:
        ex = json.load(open(path))['examples']; ok = 0
        with open('data/bbh_lies.jsonl', 'w') as f:
            for e in ex:
                _, _, st = solve(e['input']); ok += ('Yes' if st == '1' else 'No') == e['target']
                f.write(json.dumps({'question': e['input'] + '\nAnswer:', 'answer': '#### ' + e['target'], 'raw_prompt': True}) + '\n')
        print(path.split('/')[-1], f'parser+reference matches gold on {ok}/{len(ex)}')
