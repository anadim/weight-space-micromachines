"""export the 14-rule implant against a given host: python mk_sd.py OUT.pt MODEL  (run in heldout/, tables.pt built)"""
import sys, torch
sys.path.insert(0, '../f14')
import export as X
sd, meta, model, progs = X.make('cuda', torch.bfloat16, model_name=sys.argv[2])
torch.save(dict(sd), sys.argv[1] + '.tmp'); __import__('os').replace(sys.argv[1] + '.tmp', sys.argv[1]); print('saved', sys.argv[1], sys.argv[2])
