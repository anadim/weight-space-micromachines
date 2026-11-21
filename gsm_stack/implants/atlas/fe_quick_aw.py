"""Quick per-attempt-index accuracy from fe-format rows (no wf/logp): segment the token stream on the fixed writer phrases.
usage: python3 atlas/fe_quick_aw.py implants/wpr/fe/out/fe dev_wpr dev2_wpr   (cross-check for the gen-format r1atl rows)"""
import sys, os, re, json, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse import Tok, NUM, correct, Q_IDS, EOS_TXT

def find(T, pat, i0=0):
    out = []; n = len(pat)
    for i in range(i0, len(T) - n + 1):
        if T[i] == pat[0] and T[i:i + n] == pat: out.append(i)
    return out

def attempts(T, tk):
    W = Q_IDS['wait'][0]; R = Q_IDS['rep'][0]
    st = [0] + find(T, W); en = st[1:] + [len(T)]
    out = []
    for a, b in zip(st, en):
        seg = T[a:b]; rp = find(seg, R); s0 = rp[-1] if rp else 0
        txt = tk.decode(seg[s0:]).replace(EOS_TXT, '')
        m = re.search(NUM, txt); out.append((m.group(1).replace(',', '') if m else None, len(rp)))
    return out

def main():
    d = sys.argv[1]; tk = Tok()
    for pre in sys.argv[2:]:
        kk = collections.defaultdict(lambda: [0, 0]); vote = []; orc = []
        for fn in sorted(glob.glob(f'{d}/{pre}_s*.jsonl')):
            for l in open(fn):
                r = json.loads(l); A = attempts(r['tokens'], tk)
                oks = [int(correct(a, r['gold'])) for a, _ in A[:8]]
                for k, o in enumerate(oks): kk[k][0] += o; kk[k][1] += 1
                vote.append(r['ok']); orc.append(any(oks))
        print(pre, 'vote', round(100 * sum(vote) / len(vote), 2), 'oracle@8', round(100 * sum(orc) / len(orc), 2),
              'per-k', ' '.join(f'{k}:{100 * a / max(1, n):.1f}' for k, (a, n) in sorted(kk.items())),
              'mean', round(100 * sum(a for a, n in kk.values()) / sum(n for a, n in kk.values()), 2))

if __name__ == '__main__': main()
