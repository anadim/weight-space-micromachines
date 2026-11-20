# exec'd inside tw_dump_lean.py --eval_ckpt.  Single-call rethink implants, evaluated by an EXACT token-stream proxy:
# the implant's stream = [host's own greedy attempt up to the end of its '#### X' line] + [implant-written phrase] +
# [host continues]. Under greedy decoding this is the same token stream as prefilling that prefix and continuing,
# which is what this script does (one prefill + decode per fired row; unfired rows keep the plain greedy answer).
# Triggers (all computable at the '####' line by fixed circuits):
#   san   = answer impossible (negative / non-integer / missing)
#   probe = L18 probe logit at '####' < thr (thr calibrated on split C for precision 0.6)
#   unused= a number stated in the question never appears in the solution text (number-register circuit)
# Answer graded = the first '#### number' AFTER the phrase (no answer after the phrase -> keep the original answer
# is NOT done: missing counts as wrong).
import os as _o, sys as _s, re as _re, json as _j, torch as _torch, random as _r
SETS = _o.environ.get('RP_SETS', 'dev,dev2').split(',')
OUTD = _o.environ['SEL_OUT']; PR = _torch.load(f'{OUTD}/ver_probe.pt')
thr = PR['thresholds'][_o.environ.get('RP_PREC', '0.6')]['thr_logit']
NUMQ = _re.compile(r'\d[\d,]*(?:\.\d+)?')


def _norm(x):
    x = x.replace(',', '').rstrip('.')
    try:
        v = float(x); return str(int(v)) if v == int(v) else repr(v)
    except ValueError: return x


PH = {'check': "Wait, let me check that. We need to find",
      'unused': "Wait, we have not used the {N} yet. Let me solve it again using all the numbers.\n",
      'neg': "Wait, the answer can't be negative. Let me solve it again.\n",
      'nonint': "Wait, the answer should be a whole number. Let me solve it again.\n",
      'missing': "Wait, I did not give a final answer. Let me solve it again.\n"}
summ = {}
for nm in SETS:
    D = _torch.load(f'{OUTD}/{nm}_greedy_G1_s0.pt')
    n = len(D['q'])
    lg = (D['hcap'][:, 1].float() @ PR['w'] + PR['b']).numpy()
    rows = []
    for j in range(n):
        t = D['tokens'][j]; p = int(D['hpos'][j])
        txt_full = tok.decode(t).replace('<|endoftext|>', '')
        # prefix = tokens up to the first token after '####' that contains a newline (end of the answer line)
        if p >= 0:
            e = next((i for i in range(p + 1, len(t)) if '\n' in tok.decode([t[i]]) or t[i] == eng.eos), len(t))
            pre = [x for x in t[:e + 1] if x != eng.eos]
        else:
            pre = [x for x in t if x != eng.eos]
            while pre and '\n\n' in tok.decode([pre[-1]]): pre = pre[:-1]
        sol = tok.decode(pre).split('####')[0]
        qn = {_norm(x) for x in NUMQ.findall(D['q'][j])}
        sn = {_norm(x) for x in NUMQ.findall(sol)}
        sn |= {_norm(str(float(x) * 100)) for x in sn if x.replace('.', '', 1).isdigit()}
        un = sorted(qn - sn, key=lambda x: D['q'][j].replace(',', '').find(x))
        fl = D['flag'][j]
        rows.append(dict(pre=pre, flag=fl, lg=float(lg[j]), unused=un, ok=D['ok'][j], gold=D['gold'][j], ends_nl=tok.decode(pre[-1:]).endswith('\n')))
    variants = {
        'san_typed': lambda r: ({1: 'neg', 2: 'nonint', 3: 'missing'}[r['flag']] if r['flag'] else None),
        'probe_check': lambda r: ('check' if (r['flag'] or r['lg'] < thr) else None),
        'unused_hint': lambda r: ('unused' if r['unused'] else None),
        'all_typed': lambda r: ({1: 'neg', 2: 'nonint', 3: 'missing'}[r['flag']] if r['flag'] else ('unused' if r['unused'] else ('check' if r['lg'] < thr else None))),
    }
    cache = {}
    for vn, f in variants.items():
        fired = [(j, f(r)) for j, r in enumerate(rows) if f(r)]
        need = []
        for j, ph in fired:
            text = PH[ph].replace('{N}', rows[j]['unused'][0] if rows[j]['unused'] else '')
            key = (j, text)
            if key not in cache: need.append(key)
        if need:
            qids = []
            for j, text in need:
                r = rows[j]
                ids = C.q_ids(tok, D['q'][j]) + r['pre'] + tok(('' if r['ends_nl'] else '\n') + text, add_special_tokens=False)['input_ids']
                qids.append(ids)
            o = eng.generate(qids, G=1, greedy=True, max_new=320)
            for (j, text), t2 in zip(need, o['tokens']):
                cont = text + tok.decode(t2).replace('<|endoftext|>', '')
                m = _re.search(C.NUM, cont.split('\n\n')[0])
                pred = m.group(1).replace(',', '') if m else None
                cache[(j, text)] = dict(pred=pred, ok=int(C.correct(pred, rows[j]['gold'])), ntok=len(t2) + len(tok(text)['input_ids']), cont=cont[:600])
        fx = br = 0; ex = []; extra = 0; same = 0
        for j, ph in fired:
            text = PH[ph].replace('{N}', rows[j]['unused'][0] if rows[j]['unused'] else '')
            c = cache[(j, text)]; extra += c['ntok']
            if c['ok'] and not rows[j]['ok']: fx += 1
            if rows[j]['ok'] and not c['ok']: br += 1
            same += int(c['pred'] is not None and D['pred'][j] is not None and C.correct(c['pred'], D['pred'][j]))
            if len(ex) < 8 and c['ok'] != rows[j]['ok']: ex.append(dict(j=j, gold=rows[j]['gold'], before=D['pred'][j], after=c['pred'], phrase=ph, cont=c['cont'][:300]))
        acc0 = sum(r['ok'] for r in rows) / n
        rec = dict(n=n, base=round(100 * acc0, 2), acc=round(100 * (acc0 + (fx - br) / n), 2), fires=len(fired),
                   fire_rate=round(100 * len(fired) / n, 2), precision=round(sum(1 for j, _ in fired if not rows[j]['ok']) / max(1, len(fired)), 3),
                   fixes=fx, breaks=br, same_answer_after=same, extra_tok_per_item=round(extra / n, 1),
                   extra_tok_per_fire=round(extra / max(1, len(fired)), 1), examples=ex)
        summ[f'{nm}/{vn}'] = rec
        print('RP', nm, vn, _j.dumps({k: v for k, v in rec.items() if k != 'examples'}), flush=True)
_j.dump(summ, open(f'{OUTD}/rethink_proxy.json', 'w'), indent=1)
print('RPDONE', flush=True)
