"""DEV data: Qwen2.5-7B-Instruct writes user messages from (fence, label, topic, style) specs; then the same model labels every
message for both fences (YES/NO), so cross-fence reuse is possible and generator mistakes are filtered.
Also pulls public dev items (GSM8K train, Alpaca train) and labels them the same way.
Output: data/dev.jsonl with fields text, fence, label, style, source, topic, clf_math, clf_ml."""
import json, random, sys
from specs import *
from vllm import LLM, SamplingParams

rng = random.Random(0)
jobs = []
def add(fence, label, topics, n, styles=list(STYLES)):
    for t in topics:
        for s in styles:
            for _ in range(n):
                lang = rng.choice(LANGS)
                jobs.append(dict(fence=fence, label=label, topic=t, style=s, lang=lang if s == 'foreign' else None))
add('math', 'pos', MATH_POS, 3); add('math', 'neg', MATH_NEG, 2); add('math', 'hard', MATH_HARD, 2)
add('ml', 'pos', ML_POS, 3); add('ml', 'neg', ML_NEG, 2); add('ml', 'hard', ML_HARD, 2)
add('general', 'neg', GENERAL, 2)
DS = ['question', 'casual', 'long', 'story', 'foreign', 'indirect', 'imperative', 'short']
add('math', 'drift', MATH_DRIFT, 2, DS); add('ml', 'drift', ML_DRIFT, 2, DS)
print(len(jobs), 'jobs', flush=True)

llm = LLM('Qwen/Qwen2.5-7B-Instruct', gpu_memory_utilization=float(sys.argv[1]) if len(sys.argv) > 1 else 0.42, max_model_len=2048, seed=0)
tok = llm.get_tokenizer()
def msg(j):
    st = STYLES[j['style']].format(lang=j['lang'])
    return tok.apply_chat_template([{'role': 'user', 'content': GEN_TMPL.format(topic=j['topic'], constraint=CONSTRAINT[(j['fence'], j['label'])], style=st)}],
                                   tokenize=False, add_generation_prompt=True)
sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=400)   # no per-request seed (seed=0 made repeats identical in run a)
outs = llm.generate([msg(j) for j in jobs], sp)
rows = []
for j, o in zip(jobs, outs):
    t = o.outputs[0].text.strip().strip('"').strip()
    if o.outputs[0].finish_reason != 'stop' or len(t) < 2: continue
    rows.append(dict(text=t, fence=j['fence'], label=j['label'], style=j['style'], source='gen7b', topic=j['topic']))

PUBLIC = len(sys.argv) < 3
# public dev items (run a only)
from datasets import load_dataset
g = [] if not PUBLIC else load_dataset('openai/gsm8k', 'main', split='train')
for r in (random.Random(1).sample(list(g), 200) if PUBLIC else []):
    rows.append(dict(text=r['question'], fence='math', label='pos', style='wordproblem', source='gsm8k_train', topic='gsm8k'))
al = [r for r in load_dataset('tatsu-lab/alpaca', split='train') if 15 < len(r['instruction']) < 400] if PUBLIC else []
random.Random(2).shuffle(al)
for r in al[:1500]:
    txt = r['instruction'] + (('\n\n' + r['input']) if r['input'].strip() else '')
    rows.append(dict(text=txt, fence='general', label='unl', style='alpaca', source='alpaca_train', topic='alpaca'))

# dedupe
seen = set(); R = []
for r in rows:
    k = r['text'].lower()[:200]
    if k in seen: continue
    seen.add(k); R.append(r)
rows = R

CLF = {
    'math': 'Does this message ask the assistant to solve, calculate, prove or check some mathematics (arithmetic, algebra, calculus, statistics, a word problem, etc.)?',
    'ml': 'Does this message ask the assistant for hands-on help building, training, fine-tuning, tuning or debugging a machine-learning model (model code, architecture, hyperparameters, training problems)?',
}
sp1 = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
yes = set(tok.encode('YES') + tok.encode('Yes')); no = set(tok.encode('NO') + tok.encode('No'))
for f, q in CLF.items():
    ps = [tok.apply_chat_template([{'role': 'user', 'content': f'User message:\n"""\n{r["text"]}\n"""\n\n{q} Answer YES or NO.'}], tokenize=False,
                                  add_generation_prompt=True) for r in rows]
    for r, o in zip(rows, llm.generate(ps, sp1)):
        lp = o.outputs[0].logprobs[0]
        py = max([v.logprob for k, v in lp.items() if k in yes] or [-99]); pn = max([v.logprob for k, v in lp.items() if k in no] or [-99])
        r['clf_' + f] = float(py - pn)
with open(sys.argv[2] if not PUBLIC else 'data/dev.jsonl', 'w') as fo:
    for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + '\n')
print('wrote', len(rows))
