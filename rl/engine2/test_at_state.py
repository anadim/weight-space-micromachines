"""Engine faithfulness at a later point of an RL run: load the trainable state saved after update u (host + wiring),
then greedy 128 tokens on the 32 reference prompts with the OLD engine and with engine2 -> all fields bitwise."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--state', required=True); A = ap.parse_args()
import ref_env
G = ref_env.load(os.environ.get('E2_LIB', 'SET_E2_LIB') + '/engine2/runs/test_at_state', ['--B_max', '32', '--fused', '0'])
import torch, eng2
sd = torch.load(A.state, map_location='cuda')
with torch.no_grad():
    for k, v in sd['P32'].items(): G['P32'][k].copy_(v)
for k, l in G['links'].items(): l.load_state_dict(sd['links'][str(k)])
G['INW'].load_state_dict(sd['inw'])
G['sync_weights']()
ref = json.load(open(os.environ.get('E2_REF', os.environ.get('E2_LIB', 'SET_E2_LIB') + '/runs/engine_ref/engine_reference.json')))
old = G['eng']
o_old = old.generate(ref['q_ids'], G=1, greedy=True, max_new=128)
new = eng2.LibEngine2Lazy(old.W, G['pre'], G['tok'], G['imp'], G['TOOLS'], G['IFT'], G['E0'], links=G['links'], B_max=32, S_max=G['a'].S_max)
new.chat_stop = old.chat_stop; new.set_mwire(G['INW']); new.refresh()
o_new = new.generate(ref['q_ids'], G=1, greedy=True, max_new=128)
res = dict(state=A.state, rows=len(o_old['tokens']),
           tokens=sum(a == b for a, b in zip(o_old['tokens'], o_new['tokens'])),
           logp=sum(a == b for a, b in zip(o_old['logp'], o_new['logp'])),
           fire=sum(torch.equal(a, b) for a, b in zip(o_old['wt'], o_new['wt'])),
           dec=sum(torch.equal(a, b) for a, b in zip(o_old['dec'], o_new['dec'])),
           ifw=sum(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(o_old['ifw'], o_new['ifw'])),
           calc_tokens=sum(int((w[:, 0] > 0).sum()) for w in o_new['wt']), tk_flag=int(new.TK.flag.item()))
print('ATSTATE', json.dumps(res), flush=True)
