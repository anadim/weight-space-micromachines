"""Final TEST set = hand-written file (independent author) + public sources. Built once; not scored during development.
roles per fence: pos / neg / hard / drift / ret (unrelated Alpaca) ; MMLU kept separately with answers for capability retention."""
import json, random
from datasets import load_dataset
rows = [json.loads(l) for l in open('data/test_hand.jsonl')]
for r in rows: r['source'] = 'hand'
g = load_dataset('openai/gsm8k', 'main', split='test')
for r in random.Random(3).sample(list(g), 100):
    rows.append(dict(text=r['question'], fence='math', label='pos', style='pub_gsm8k', source='gsm8k_test'))
m5 = load_dataset('HuggingFaceH4/MATH-500', split='test')
for r in random.Random(4).sample(list(m5), 100):
    rows.append(dict(text=r['problem'], fence='math', label='pos', style='pub_math500', source='math500'))
mm = load_dataset('cais/mmlu', 'all', split='test')
L = 'ABCD'
def mc(r): return r['question'].strip() + '\n' + '\n'.join(f'{L[i]}. {c}' for i, c in enumerate(r['choices'])) + '\nAnswer with the letter only.'
MATHS = {'high_school_mathematics', 'elementary_mathematics', 'college_mathematics', 'abstract_algebra', 'high_school_statistics'}
mathq = [r for r in mm if r['subject'] in MATHS]
for r in random.Random(5).sample(mathq, 100):
    rows.append(dict(text=mc(r), fence='math', label='pos', style='pub_mmlu_mc', source='mmlu_math', answer=L[r['answer']]))
EXCL = MATHS | {'machine_learning', 'college_computer_science', 'high_school_computer_science', 'college_physics', 'high_school_physics',
                'conceptual_physics', 'econometrics', 'college_chemistry', 'high_school_chemistry', 'electrical_engineering', 'formal_logic',
                'high_school_microeconomics', 'high_school_macroeconomics', 'professional_accounting', 'astronomy', 'computer_security'}
oth = [r for r in mm if r['subject'] not in EXCL]
mmlu = [dict(text=mc(r), fence='general', label='mmlu', style='pub_mmlu_mc', source='mmlu_other', answer=L[r['answer']], subject=r['subject'])
        for r in random.Random(6).sample(oth, 400)]
al = [r for r in load_dataset('tatsu-lab/alpaca', split='train') if 15 < len(r['instruction']) < 400]
random.Random(2).shuffle(al)                                            # same shuffle as gen_dev: [:1500] dev, [3000:3600] test
for r in al[3000:3600]:
    rows.append(dict(text=r['instruction'] + (('\n\n' + r['input']) if r['input'].strip() else ''), fence='general', label='unl', style='alpaca', source='alpaca_test'))
with open('data/test.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open('data/mmlu_ret.jsonl', 'w') as f:
    for r in mmlu: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(len(rows), len(mmlu))
