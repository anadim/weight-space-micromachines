"""Arm-4 curve: dev on/off, rethink fire / fix / break per dev eval. python summ4.py RUN_DIR"""
import sys, json
on, off = {}, {}
for l in open(sys.argv[1] + '/steps.jsonl'):
    r = json.loads(l)
    if 'dev' in r: on[r['dev']['u']] = r['dev']
    for k in ('dev_host_curve', 'dev_hostonly'):
        if k in r: off[r[k]['u']] = r[k]
for u in sorted(on):
    d = on[u]; h = off.get(u, {})
    print(f"u{u:3d} on {d['dev_acc']:5.1f} (gsm {d['dev_gsm']:.1f} / scaled {d['dev_scaled']:.1f})  off {h.get('devhost_acc', float('nan')):5.1f}  "
          f"fire {100*d.get('dev_rt_fire', 0):4.1f}% (hash {100*d.get('dev_rt_fire_hash', 0):4.1f} end {100*d.get('dev_rt_fire_end', 0):4.1f})  "
          f"changed {100*d.get('dev_rt_changed', 0):4.1f} fixed {100*d.get('dev_rt_fixed', 0):4.1f} broke {100*d.get('dev_rt_broke', 0):4.1f}  len {d['dev_len']:.0f}"
          + (f"  | AUROC sB {d['aud_auroc_sB']:.3f} s0 {d['aud_auroc_s0']:.3f} fire wrong/right {100*d['aud_fire_wrong']:.1f}/{100*d['aud_fire_right']:.1f}" if 'aud_auroc_sB' in d else ''))
