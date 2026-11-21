"""build aw/stk/sn_w.pt = fixed side-net vector (job/out/w_TAG_CFG.pt, fold-mean folded) + BLD writer theta (vfsn_BLD_L2.json from
repick_sn.py --save CFG@BLD@L2). python3.11 mk_snw.py W.pt THETA.json"""
import sys, json, torch
sys.path.insert(0, 'aw/stk')
import aw_sn as ASN
W = torch.load(sys.argv[1]); T = json.load(open(sys.argv[2]))
assert T['mode'] == 'BLD' and T['cfg'] == W['cfg'], (T['cfg'], W['cfg'])
out = dict(w=W['w'], b=float(W['b']), layers=list(W['layers']), kinds=W['kinds'], D=int(W['D']), cfg=W['cfg'], epoch=W['epoch'],
           th=T['th'], c0=float(T['c0']), feats=T['feats'], l2=T['l2'], th_vf=ASN.to_vf_order(T['th'], T['feats']))
torch.save(out, 'aw/stk/sn_w.pt')
print('sn_w.pt', W['cfg'], 'ep', W['epoch'], 'layers', out['layers'], '|w|', float(W['w'].norm()), 'b', out['b'], 'check', W['check'])
print('th_vf', [round(x, 4) for x in out['th_vf']], 'c0', out['c0'])
