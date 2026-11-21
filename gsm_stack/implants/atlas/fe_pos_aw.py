"""None-rate (attempt without '####') and accuracy binned by the attempt's start index in the generated stream (fe rows)."""
import sys, os, json, glob, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_quick_aw import find
from parse import Tok, Q_IDS, NUM, EOS_TXT, correct
tk = Tok(); W = Q_IDS['wait'][0]; R = Q_IDS['rep'][0]
for pre in sys.argv[2:]:
    B = collections.defaultdict(lambda: [0, 0, 0])
    for fn in sorted(glob.glob(f'{sys.argv[1]}/{pre}_s*.jsonl')):
        for l in open(fn):
            r = json.loads(l); T = r['tokens']; st = [0] + find(T, W); en = st[1:] + [len(T)]
            for k, (a, b) in enumerate(zip(st, en)):
                if k == len(st) - 1: continue            # last attempt carries the vote line
                seg = T[a:b]; rp = find(seg, R); s0 = rp[-1] if rp else 0
                m = re.search(NUM, tk.decode(seg[s0:]).replace(EOS_TXT, ''))
                ans = m.group(1).replace(',', '') if m else None
                bn = min(a // 500, 8); B[bn][0] += 1; B[bn][1] += ans is None; B[bn][2] += int(correct(ans, r['gold']))
    print(pre, ' | '.join(f'{bn*500}-: n{n} none{100*x/n:.1f}% acc{100*c/n:.1f}' for bn, (n, x, c) in sorted(B.items())))
