"""Heuristic vs hand labels (data/hand_labels.tsv, sampled wrong attempt per item; see hand_sample.py).
usage: python3 atlas/calib.py [-v]"""
import os, sys, json, gzip, collections
HERE = os.path.dirname(os.path.abspath(__file__))
CATS = ['operand', 'relation', 'dropped', 'unit', 'target', 'plan', 'arithmetic', 'early_stop', 'format', 'other']


def load():
    S = {(x['set'], x['idx']): x for x in json.load(open(f'{HERE}/data/hand_sample.json'))}
    H = {}
    for l in open(f'{HERE}/data/hand_labels.tsv'):
        s, i, c, n = l.rstrip('\n').split('\t'); H[(s, int(i))] = (c, n)
    R = {}
    for s in ('dev', 'dev2'):
        with gzip.open(f'{HERE}/data/diverge_{s}.jsonl.gz', 'rt') as f:
            for l in f:
                r = json.loads(l)
                if r['rec'] == 'att': R[r['gid']] = r
    return S, H, R


def confusion(S, H, R):
    pairs = [(H[k][0], R[S[k]['wrong']]['cat'], k) for k in H]
    return pairs


def main():
    S, H, R = load(); P = confusion(S, H, R)
    agree = sum(h == a for h, a, _ in P)
    print(f'agreement {agree}/{len(P)} = {agree / len(P):.2f}')
    G = {'relation': 'misread', 'operand': 'misread', 'plan': 'misread', 'unit': 'misread', 'dropped': 'omission',
         'target': 'omission'}
    g = lambda c: G.get(c, 'mechanical')
    ca = sum(g(h) == g(a) for h, a, _ in P)
    print(f'coarse agreement (misread / omission / mechanical) {ca}/{len(P)} = {ca / len(P):.2f}')
    M = collections.Counter((h, a) for h, a, _ in P)
    print('hand \\ heur ' + ' '.join(f'{c[:5]:>5}' for c in CATS))
    for h in CATS:
        print(f'{h:11s} ' + ' '.join(f'{M[(h, a)]:5d}' for a in CATS))
    if '-v' in sys.argv:
        for h, a, k in P:
            if h != a:
                r = R[S[k]['wrong']]
                print(k, 'hand', h, '| heur', a, r['sub'], '|', r.get('detail', ''), '|', H[k][1], '| div:', r.get('div_line', '')[:90])


if __name__ == '__main__':
    main()
