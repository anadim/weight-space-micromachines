import sys, os, json, re
sys.path.insert(0, 'atlas')
from fe_quick_aw import find
from parse import Tok, Q_IDS, NUM, EOS_TXT
tk = Tok(); W = Q_IDS['wait'][0]; shown = 0
for l in open(sys.argv[1]):
    r = json.loads(l); T = r['tokens']; st = [0] + find(T, W); en = st[1:] + [len(T)]
    for k, (a, b) in enumerate(zip(st, en)):
        txt = tk.decode(T[a:b]).replace(EOS_TXT, '')
        if k >= 5 and k < 7 and not re.search(NUM, txt):
            print(f'=== item {r["i"]} k{k} len {b-a}\n', txt[:300], '\n ... \n', txt[-400:]); shown += 1
            break
    if shown >= 5: break
