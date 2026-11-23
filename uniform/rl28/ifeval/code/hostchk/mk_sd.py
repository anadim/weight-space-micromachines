"""export the 14-rule implant against a given host: python mk_sd.py OUT.pt MODEL [VECS.pt]  (run in heldout/, tables.pt built)
VECS = the host's force_fix rows (default vecs2_forcefix.pt = solved on Qwen3-0.6B; rl28: vecs_q17.pt solved on Qwen3-1.7B)"""
import sys, torch
sys.path.insert(0, '../f14')
import export as X
kw = dict(vecs=sys.argv[3]) if len(sys.argv) > 3 else {}
sd, meta, model, progs = X.make('cuda', torch.bfloat16, model_name=sys.argv[2], **kw)
D = model.config.hidden_size
for k in ('voc.E', 'voc.g', 'force.F'):
    assert k not in sd or sd[k].shape[-1] == D, (k, tuple(sd[k].shape), D)
print('host', sys.argv[2], 'D', D, {k: tuple(sd[k].shape) for k in ('voc.E', 'voc.g', 'force.F') if k in sd})
torch.save(dict(sd), sys.argv[1] + '.tmp'); __import__('os').replace(sys.argv[1] + '.tmp', sys.argv[1]); print('saved', sys.argv[1], sys.argv[2], kw)
