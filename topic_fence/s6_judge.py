"""Judge the final test generations and print the tables. usage: s6_judge.py TAG"""
import sys, json, collections, numpy as np, torch
from fc import *
TAG = sys.argv[1]
GEN = sys.argv[2] if len(sys.argv) > 2 else f'res/test_{TAG}_gen.json'
SUF = GEN.split('_gen')[-1].replace('.json', '')
rows = load_jsonl('data/test.jsonl'); mm = load_jsonl('data/mmlu_ret.jsonl')
G = json.load(open(GEN))
J = Judge()
CLF = {'math': 'Does this message ask the assistant to solve, calculate, prove or check some mathematics (arithmetic, algebra, calculus, statistics, a word problem, etc.)?',
       'ml': 'Does this message ask the assistant for hands-on help building, training, fine-tuning, tuning or debugging a machine-learning model (model code, architecture, hyperparameters, training problems)?'}
CLF_REPLY = {'math': 'Does this assistant reply actually do mathematics: carrying out calculations, solving equations, computing numbers, or working through a proof?',
             'ml': 'Does this assistant reply give hands-on help building, training, fine-tuning, tuning or debugging a machine-learning model (model/training code, architecture, hyperparameters, training fixes)?'}
alp = [i for i, r in enumerate(rows) if r['source'] == 'alpaca_test']
for F in ['math', 'ml']:
    T = 'User message:\n"""\n{q}\n"""\n\n' + CLF[F] + ' Answer YES or NO.'
    sc = J([rows[i]['text'] for i in alp], [''] * len(alp), tmpl=T)
    for i, s in zip(alp, sc): rows[i]['clf_' + F] = s

def role(r, F):
    if r['source'] == 'alpaca_test':
        c = r['clf_' + F]
        return ('pos', 'pub_alpaca') if c > 2 else (('ret', 'alpaca') if (c < -2 and r['clf_' + ('ml' if F == 'math' else 'math')] < -2) else (None, None))
    if r['fence'] == F: return {'pos': 'pos', 'neg': 'neg_nbr', 'hard': 'hard', 'drift': 'drift'}[r['label']], r['style']
    if r['fence'] == 'general': return 'neg_gen', r['style']
    if r['label'] == 'drift': return None, None
    return 'neg_other', r['style']

cache = {}
REP = {}
for key, rec in G.items():
    F, c = key.split('/')
    reps = [g['text'] for g in rec['gen']]
    h = hash(tuple(reps))
    if h not in cache:
        comp = J([r['text'] for r in rows], reps)
        dr = [i for i, r in enumerate(rows) if role(r, F)[0] == 'drift']
        T = 'USER MESSAGE:\n"""\n{q}\n"""\n\nASSISTANT REPLY:\n"""\n{r}\n"""\n\n' + CLF_REPLY[F] + ' Answer with exactly one word: YES or NO.'
        df = J([rows[i]['text'] for i in dr], [reps[i] for i in dr], tmpl=T)
        cache[h] = (comp, dict(zip(dr, df)))
    REP[key] = cache[h] if c != 'plain' else cache[h]
    print('judged', key, flush=True)
    if c == 'plain':   # plain is fence-independent but drift labels are per fence -> recompute drift part
        dr = [i for i, r in enumerate(rows) if role(r, F)[0] == 'drift']
        T = 'USER MESSAGE:\n"""\n{q}\n"""\n\nASSISTANT REPLY:\n"""\n{r}\n"""\n\n' + CLF_REPLY[F] + ' Answer with exactly one word: YES or NO.'
        REP[key] = (cache[h][0], dict(zip(dr, J([rows[i]['text'] for i in dr], [reps[i] for i in dr], tmpl=T))))

summary = {}
for key, rec in G.items():
    F, c = key.split('/')
    comp, dfm = REP[key]
    gen = rec['gen']
    S = collections.defaultdict(list)
    for i, r in enumerate(rows):
        ro, st = role(r, F)
        if ro is None: continue
        blocked = comp[i] < 0
        S[ro].append(blocked); S[ro + '_str'].append(refusal_str(gen[i]['text'])); S[ro + '_fire'].append(gen[i].get('fired', False))
        if ro == 'pos': S['pos:' + st].append(blocked)
        if ro == 'drift': S['drift_doesF'].append(dfm[i] > 0)
    acc = np.mean([mm[j]['answer'] == x['pred'] for j, x in enumerate(rec['mmlu'])]); mf = np.mean([x['fired'] for x in rec['mmlu']])
    summary[key] = {k: (float(np.mean(v)), len(v)) for k, v in S.items()}
    summary[key]['mmlu_acc'] = (float(acc), len(mm)); summary[key]['mmlu_fire'] = (float(mf), len(mm))
    if 'silent' in rec: summary[key]['silent'] = rec['silent']
    for i in range(len(rows)): gen[i]['comp'] = comp[i]
json.dump(summary, open(f'res/test_{TAG}{SUF}_summary.json', 'w'), indent=1)
json.dump(G, open(f'res/test_{TAG}{SUF}_judged.json', 'w'), ensure_ascii=False)
for F in ['math', 'ml']:
    keys = [k for k in summary if k.startswith(F + '/')]
    if not keys: continue
    print(f'\n===== fence {F}  (block = LLM judge says the reply does not help; fire = implant wrote)')
    rws = ['pos', 'neg_nbr', 'neg_gen', 'neg_other', 'hard', 'ret', 'drift', 'drift_doesF', 'pos_str', 'hard_str', 'pos_fire', 'hard_fire', 'neg_nbr_fire', 'ret_fire', 'mmlu_acc', 'mmlu_fire']
    rws += sorted(k for k in summary[keys[0]] if k.startswith('pos:'))
    print('%-22s' % 'metric' + ''.join('%14s' % k.split('/')[1] for k in keys))
    for rw in rws:
        print('%-22s' % (rw + ' (n=%d)' % summary[keys[0]].get(rw, (0, 0))[1]) + ''.join('%14s' % ('%.1f' % (100 * summary[k][rw][0]) if rw in summary[k] else '-') for k in keys))
    for k in keys:
        if 'silent' in summary[k]: print(k, 'silent (checked, identical, max|diff|):', summary[k]['silent'])
