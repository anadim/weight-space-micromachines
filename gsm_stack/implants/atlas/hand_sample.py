"""Stratified hand-label sample: 20 dev + 10 dev2 mixed items per n_right band (seed 0); one representative wrong attempt
per item (attractor answer, longest common token prefix with a right attempt) vs that closest right attempt.
-> data/hand_sample.json (ids) + data/hand_dump.txt (texts to read).  Hand labels live in data/hand_labels.tsv (set, idx, category, note); calib.py compares them with diverge.py."""
import os, sys, json, gzip, random, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from diverge import rd, BANDS, band

def main():
    out = []; dump = []
    for set_, per in (('dev', 20), ('dev2', 10)):
        D = rd(f'{HERE}/data/diverge_{set_}.jsonl.gz')
        items = [r for r in D if r['rec'] == 'item']; atts = {r['gid']: r for r in D if r['rec'] == 'att'}
        A = {a['gid']: a for a in rd(f'{HERE}/data/attempts_{set_}.jsonl.gz') if a['gid'] in atts}
        rng = random.Random(0)
        for a, b in BANDS:
            pool = sorted([it for it in items if a <= it['n_right'] <= b], key=lambda x: x['idx'])
            for it in sorted(rng.sample(pool, min(per, len(pool))), key=lambda x: x['idx']):
                ws = [r for r in atts.values() if r['idx'] == it['idx'] and not r['ok'] and r['ans'] == it['attractor']]
                w = max(ws, key=lambda r: (r.get('b_tok', -1), -r['k']))
                rp = w.get('partner') or w['sem_partner']
                out.append(dict(set=set_, idx=it['idx'], band=band(it['n_right']), wrong=w['gid'], right=rp))
                q = A[w['gid']]['q']
                dump.append(f"### {set_} {it['idx']}  n_right={it['n_right']}  gold={it['gold']}  wrong answers={it['wrong_answers']}\n"
                            f"Q: {q}\n--- WRONG {w['gid']} (ans {w['ans']}):\n{A[w['gid']]['text'].replace('Wait, let me solve it again.', '').strip()}\n"
                            f"--- RIGHT {rp}:\n{A[rp]['text'].replace('Wait, let me solve it again.', '').strip()}\n")
    json.dump(out, open(f'{HERE}/data/hand_sample.json', 'w'), indent=0)
    open(f'{HERE}/data/hand_dump.txt', 'w').write('\n'.join(dump))
    print(len(out), collections.Counter((o['set'], o['band']) for o in out))

if __name__ == '__main__':
    main()
