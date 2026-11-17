"""Build the implants and report their sizes.

python scripts/build_implants.py [--machines all] [--out results/implant_sizes.json]          circuits only (no model, CPU)
python scripts/build_implants.py --attach [--model Qwen/Qwen3-0.6B-Base] [--veto runs/veto_qwen3.pt]
        also links them to the host: solves the readers (exact on every vocabulary row) and the write directions, and
        checks through the host's own norm + LM head that every write wins the argmax (prints the smallest margin).

Nothing is saved but the report: implants are rebuilt from code in seconds to minutes (logic_nat is the big one:
about 200k ReLU units and 75M nonzero weights, ~10 minutes and ~40 GB of RAM to compile on CPU).
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from micromachines import SPECS, BBH_MACHINES, DESCRIPTION
from micromachines.hdl import ReLUNet

ap = argparse.ArgumentParser()
ap.add_argument('--machines', default='all')
ap.add_argument('--attach', action='store_true')
ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--veto', default='runs/veto_qwen3.pt')
ap.add_argument('--out', default='results/implant_sizes.json')
a = ap.parse_args()
names = BBH_MACHINES + ('calc_s3',) if a.machines == 'all' else tuple(a.machines.split(','))
rep = {}
if not a.attach:
    for n in names:
        t = time.time(); c1, c2 = SPECS[n]['build'](); s1, s2 = ReLUNet(c1), ReLUNet(c2)
        rep[n] = dict(description=DESCRIPTION.get(n, ''), relu_units=s1.n_units + s2.n_units, layers=s1.L + s2.L,
                      nonzero_weights=s1.nnz + s2.nnz, window_tokens=SPECS[n]['K'], input_symbols=len(SPECS[n]['syms']),
                      output_tokens=len(SPECS[n]['out']), build_seconds=round(time.time() - t, 1))
        print(f"{n:10s} {rep[n]['relu_units']:7d} units {rep[n]['layers']:3d} layers {rep[n]['nonzero_weights']:10d} nonzeros "
              f"K={rep[n]['window_tokens']:4d} | {rep[n]['build_seconds']}s | {rep[n]['description']}", flush=True)
    tot = {k: sum(r[k] for m, r in rep.items() if m in BBH_MACHINES) for k in ('relu_units', 'nonzero_weights')}
    print('nine BBH implants together:', tot)
    rep['_nine_bbh_total'] = tot
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, 'w'), indent=1)
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from micromachines import attach, VETO_MACHINES
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev).eval()
    t = time.time()
    net = attach(model, tok, machines=[n for n in names if n != 'calc_s3'],
                 veto=a.veto if os.path.exists(a.veto) else None, veto_machines=VETO_MACHINES)
    print(f'attached in {time.time() - t:.0f}s')
    worst = min(net.margins.items(), key=lambda kv: kv[1][0])
    print('readers exact on all', model.get_input_embeddings().weight.shape[0], 'vocabulary rows (asserted)')
    print('smallest write margin through the host head (logit units):', worst[0], round(worst[1][0], 2),
          '| largest leaked probability:', max(v[1] for v in net.margins.values()))
