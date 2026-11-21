"""Parse runs/L500_*/steps.jsonl (+ external refs runs/ref_<tag>.jsonl) into a results table (markdown + json)."""
import os, sys, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
SHOW_U = (0, 25, 50, 100, 150, 200, 250, 300)


def load(p):
    devs, host, loff, drift, last, link = {}, {}, {}, {}, None, None
    reset, cw, params, gain, pfire = {}, {}, None, {}, {}
    for line in open(p):
        try: r = json.loads(line)
        except Exception: continue
        if 'dev' in r: devs[r['dev']['u']] = r['dev']
        for k in ('dev_host_curve', 'dev_hostonly'):
            if k in r:
                d = r[k]; host[d['u']] = d.get('devhost_acc', d.get('devhostonly_acc'))
        for k in ('dev_linksoff_curve', 'dev_linksoff_final'):
            if k in r:
                d = r[k]; loff[d['u']] = d.get('devlinksoff_acc', d.get('devlinksoff_final_acc'))
        if 'drift' in r: drift[r['drift']['u']] = r['drift']['drift_exact']
        if 'dev_reset_curve' in r: reset[r['dev_reset_curve']['u']] = r['dev_reset_curve']['devreset_acc']
        for md in ('cw_hand', 'cw_out', 'cw_in', 'cw_mask21'):
            if f'dev_{md}' in r:
                d = r[f'dev_{md}']; cw.setdefault(md, {})[d['u']] = (d[f'dev{md}_acc'], d[f'dev{md}_scaled'])
        if 'params' in r: params = r['params']
        if 'reward' in r and 'gain' in r: gain[r['u']] = r['gain']
        if 'reward' in r and 'p_fire' in r: pfire[r['u']] = r['p_fire']
        if 'drift_final' in r: drift[r['drift_final']['u']] = r['drift_final']['drift_final_exact']
        if 'reward' in r:
            last = r
            if 'link_eff' in r: link = (r['u'], r['link_eff'])
    return dict(devs=devs, host=host, linksoff=loff, drift=drift, last_u=last['u'] if last else 0, link=link,
                reset=reset, cw=cw, params=params, gain=gain, pfire=pfire, last=last)


res = {}
for d in sorted(glob.glob(os.path.join(RUNS, 'L500_*')) + glob.glob(os.path.join(RUNS, 'L300_*'))):
    p = os.path.join(d, 'steps.jsonl')
    if os.path.isdir(d) and os.path.exists(p): res[os.path.basename(d)] = load(p)
for p in sorted(glob.glob(os.path.join(RUNS, 'ref_*.jsonl'))):
    res['REF_' + os.path.basename(p)[4:-6]] = load(p)


def cell(dv, u, k='dev_acc'):
    x = dv.get(u); return '' if x is None else f"{x[k]:.1f}"


hdr = ('| run | last u | ' + ' | '.join(f'dev@{u}' for u in SHOW_U) +
       ' | GSM@last | scaled@last | calls@last | host-only (u) | links-off (u) | writer-exact (u) |')
lines = [hdr, '|' + '---|' * (hdr.count('|') - 1)]
for tag, r in res.items():
    dv = r['devs']; ul = max(dv) if dv else None
    f = dv.get(ul, {}) if ul is not None else {}
    h = ', '.join(f'{v:.1f}@{u}' for u, v in sorted(r['host'].items()))
    lo = ', '.join(f'{v:.1f}@{u}' for u, v in sorted(r['linksoff'].items()))
    dr = ', '.join(f'{v:.3f}@{u}' for u, v in sorted(r['drift'].items()))
    lines.append(f"| {tag} | {r['last_u']} | " + ' | '.join(cell(dv, u) for u in SHOW_U) +
                 f" | {f.get('dev_gsm', 0):.1f} | {f.get('dev_scaled', 0):.1f} | {f.get('dev_calls', 0):.2f} | {h} | {lo} | {dr} |")
md = '\n'.join(lines)
md += '\n\nlink effective norm ||diag(gate)W||_F at the last logged update:\n'
for tag, r in res.items():
    if r['link']: md += f"- {tag} (u{r['link'][0]}): " + ', '.join(f'{k} {v:.3g}' for k, v in r['link'][1].items()) + '\n'
md += '\nwiring params (trainable / % host / % implant nonzeros), reset curve, CW evals, gain, p(written):\n'
for tag, r in res.items():
    bits = []
    if r['params']: bits.append(f"params {r['params']['wire']:,} / {r['params']['pct_host']:.4f}% / {r['params']['pct_implant'] or 0:.2f}%")
    if r['reset']: bits.append('reset ' + ', '.join(f'{v:.1f}@{u}' for u, v in sorted(r['reset'].items())))
    for md, cv in r['cw'].items(): bits.append(md + ' ' + ', '.join(f'{v[0]:.1f}(sc {v[1]:.1f})@{u}' for u, v in sorted(cv.items())))
    if r['gain']: bits.append('gain ' + ', '.join(f'{r["gain"][u]:.1f}@{u}' for u in sorted(r['gain']) if u in (1, 25, 50, 100, 150, 200, 250, 300)))
    if r['pfire']: bits.append('p_fire ' + ', '.join(f'{r["pfire"][u]:.3f}@{u}' for u in sorted(r['pfire']) if u in (1, 25, 50, 100, 150, 200, 250, 300)))
    if bits: md += f'- {tag}: ' + ' | '.join(bits) + '\n'
md += '\nfull dev curves (u: acc / gsm / scaled / calls):\n'
for tag, r in res.items():
    md += f"- {tag}: " + '; '.join(f"{u}: {v['dev_acc']:.1f}/{v['dev_gsm']:.1f}/{v['dev_scaled']:.1f}/{v.get('dev_calls', 0):.2f}"
                                  for u, v in sorted(r['devs'].items())) + '\n'
print(md)
json.dump(res, open(os.path.join(RUNS, 'results.json'), 'w'), indent=1, default=str)
open(os.path.join(RUNS, 'results.md'), 'w').write(md)
