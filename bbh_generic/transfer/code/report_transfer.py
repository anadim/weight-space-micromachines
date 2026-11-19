"""Aggregate transfer arms with strict + lenient + no-answer rate.
python report_transfer.py OUT_DIR arm1 arm2 ...
Reads <arm>.json (or .partial) = run27_integ output {rows:[{task,gold,pred,ok,out}]}, or the reduced rows shipped in
../results/ (no text; precomputed ok_lenient and n_calls, see reduce_transfer.py)."""
import json, os, sys, re, glob, gzip
TOOL12 = {'boolean_expressions','logical_deduction_five_objects','logical_deduction_seven_objects',
 'logical_deduction_three_objects','multistep_arithmetic_two','navigate','object_counting','temporal_sequences',
 'tracking_shuffled_objects_five_objects','tracking_shuffled_objects_seven_objects',
 'tracking_shuffled_objects_three_objects','web_of_lies'}

def norm(x, task):   # verbatim from run27_integ.py
    x = x.strip().strip('.').strip()
    if re.fullmatch(r'\(?[A-R]\)?(\s.*)?', x) and task not in ('word_sorting','dyck_languages'):
        return '(' + re.match(r'\(?([A-R])', x).group(1) + ')'
    if task in ('multistep_arithmetic_two','object_counting'):
        m = re.search(r'-?\d+', x); return m.group() if m else x
    return x.lower()

def lenient_pred(out, task):
    body = out.split('\n\nQ:')[0]
    m = re.findall(r'[Tt]he answer is\s*(.*?)(?:\.\s*$|\.\n|\n|$)', body)
    if m and m[-1].strip(): return m[-1].strip()
    # lenient fallbacks when the template drifted
    m2 = re.findall(r'(?:answer|result|final|conclusion|therefore|so|thus)[^A-Za-z0-9]{0,8}(?:is|:|=|would be)?\s*\(?([A-R])\)?\b', body, re.I)
    if task in ('multistep_arithmetic_two','object_counting'):
        nums = re.findall(r'-?\d+', body); return nums[-1] if nums else ''
    if task not in ('word_sorting','dyck_languages'):
        mc = re.findall(r'\(([A-R])\)', body)
        if mc: return '('+mc[-1]+')'
        if m2: return '('+m2[-1].upper()+')'
    if task in ('web_of_lies',):
        yn = re.findall(r'\b(Yes|No)\b', body); return yn[-1] if yn else ''
    if task=='boolean_expressions':
        tf = re.findall(r'\b(True|False)\b', body); return tf[-1] if tf else ''
    if task=='navigate':
        yn = re.findall(r'\b(Yes|No)\b', body); return yn[-1] if yn else ''
    return ''

def load(a, O):
    for p in (f'{O}/{a}.json', f'{O}/{a}.json.gz', f'{O}/{a}.json.partial'):
        if os.path.exists(p):
            with (gzip.open(p, 'rt') if p.endswith('.gz') else open(p)) as f: return json.load(f)
    return None

def stats(d):
    if not d: return None
    rows=d.get('rows',[]); per={}; call={}; na={}; len_ok={}
    for r in rows:
        t=r['task']
        per.setdefault(t,[]).append(r['ok'])
        call.setdefault(t,[]).append(('<tool_call>' in r['out']) if 'out' in r else r.get('n_calls', 0) > 0)
        na.setdefault(t,[]).append(r.get('pred','').strip()=='')
        lo = r['ok_lenient'] if 'ok_lenient' in r else (r['ok'] or (norm(lenient_pred(r.get('out',''),t),t)==norm(r['gold'],t)))
        len_ok.setdefault(t,[]).append(lo)
    tasks=sorted(per)
    acc={t:100*sum(per[t])/len(per[t]) for t in tasks}
    lacc={t:100*sum(len_ok[t])/len(len_ok[t]) for t in tasks}
    nar={t:100*sum(na[t])/len(na[t]) for t in tasks}
    def mean(d,sub): s=[d[t] for t in tasks if (sub is None or (t in TOOL12)==sub)]; return sum(s)/len(s) if s else 0
    n15=[t for t in tasks if t not in TOOL12]
    c15=100*sum(sum(call[t]) for t in n15)/sum(len(call[t]) for t in n15) if n15 else 0
    return {'acc':acc,'lacc':lacc,'nar':nar,'tasks':tasks,'call15':c15,
            'm27':mean(acc,None),'m12':mean(acc,True),'m15':mean(acc,False),
            'l27':mean(lacc,None),'l12':mean(lacc,True),'l15':mean(lacc,False),
            'na27':mean(nar,None),'na15':mean(nar,False)}

if __name__ == '__main__':
    O = sys.argv[1]
    ARMS=sys.argv[2:]
    print('present:', sorted(os.path.basename(f).replace('.json','').replace('.partial','') for f in glob.glob(f'{O}/*.json*')))
    S={a:stats(load(a, O)) for a in ARMS}
    alltasks=sorted(set(t for s in S.values() if s for t in s['tasks']))
    print('\n| task | tool |'+''.join(f' {a} (L / na) |' for a in ARMS)); print('|---|---|'+'---|'*len(ARMS))
    for t in alltasks:
        row=f"| {t} | {'Y' if t in TOOL12 else ''} |"
        for a in ARMS:
            s=S[a]
            row += (f" {s['acc'][t]:.1f} / {s['lacc'][t]:.1f} / {s['nar'][t]:.0f} |" if s and t in s['acc'] else '  -  |')
        print(row)
    print()
    for lab,ks in [('27 mean strict','m27'),('27 mean lenient','l27'),('27 no-answer%','na27'),
                   ('12-tool strict','m12'),('12-tool lenient','l12'),
                   ('15-notool strict','m15'),('15-notool lenient','l15'),('15-notool no-answer%','na15'),
                   ('call% 15-notool','call15')]:
        print(f"| **{lab}** |"+''.join(f" {S[a][ks]:.2f} |" if S[a] else '  -  |' for a in ARMS))

