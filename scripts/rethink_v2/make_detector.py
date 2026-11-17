"""Turn detector_base.pt (policy.py) into the implant's detector file for one policy: fire iff P(correct) logit < theta,
i.e. pass = step(-logit - (-theta)).  python scripts/rethink_v2/make_detector.py BASE.pt POLICY OUT.pt"""
import sys, torch
d = torch.load(sys.argv[1]); pol = sys.argv[2]
d['thr_logit'] = -float(d['thetas'][pol]); d['policy'] = pol
torch.save(d, sys.argv[3]); print(pol, d['cfg'], 'thr_logit', d['thr_logit'])
