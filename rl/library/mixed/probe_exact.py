"""Offline (CPU): for a probe_<tool>.json, compare each tool reply in the full-library outputs with the Python reference
on the SAME call string (integ.ref), and split item accuracy by whether the tool fired / was exact."""
import os, sys, re, json
sys.path.insert(0, os.environ.get('WSM_INTEG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'bbh_generic', 'integrate', 'code')))
import integ
P = json.load(open(sys.argv[1]))
pat = re.compile(r'<tool_call>(.*?)</tool_call>\s*<tool_response>(.*?)</tool_response>', re.S)
n_calls = n_exact = 0; per = []; examples = []
for it, txt, ok in zip(P['items'], P['full']['texts'], P['full']['ok']):
    calls = pat.findall(txt)
    ex = []
    for c, resp in calls:
        try: ref = integ.ref(c.strip())
        except Exception as e: ref = f'ERR {e}'
        good = (ref is not None and str(ref).strip() == resp.strip())
        n_calls += 1; n_exact += good; ex.append(good)
        if not good and len(examples) < 5: examples.append((c.strip()[:80], resp.strip()[:40], str(ref)[:40]))
    per.append((len(calls) > 0, all(ex) if ex else None, ok))
fired = [p for p in per if p[0]]; nof = [p for p in per if not p[0]]
print(json.dumps(dict(items=len(per), with_call=len(fired), calls=n_calls, tool_exact=n_exact,
                      acc_full=P['full']['acc'], acc_loo=P[[k for k in P if k.startswith('loo_')][0]]['acc'],
                      acc_items_with_call=100 * sum(p[2] for p in fired) / max(1, len(fired)),
                      acc_items_without_call=100 * sum(p[2] for p in nof) / max(1, len(nof)),
                      mismatches=examples), indent=1))
