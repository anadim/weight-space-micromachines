# exec'd inside dig_san/tw_dump_lean.py after --eval_ckpt weights are loaded (eng, tok, C, a, shot_q in scope).
# Reading/planning implants on GSM8K split-B dev (600) + dev2 (300). Writes one jsonl per variant.
#   prompt variants (an unconditional write at answer start == a prompt change; run on all items):
#     off | re2q | re2a | plan1 | plan2 | T1s<seed> (off prompt, T=1 sample)
#   resume variants (pause at the end of the off answer, write a nudge, let the host continue; only on trigger rows):
#     nudge_<trigger>_<style>
import json as _j, os as _o, sys as _s, time as _t, random as _r
_s.path.insert(0, _o.environ['RD_DIR'])
import rd_trig as RT
OUT = _o.environ['RD_OUT']; _o.makedirs(OUT, exist_ok=True)
VARS = _o.environ.get('RD_VARIANTS', 'off').split(',')
NLIM = int(_o.environ.get('RD_N', 0))
sp = _j.load(open(_o.path.join(_o.environ['RD_DIR'], 'dev_split.json')))
tr = [_j.loads(l) for l in open(f'{C.REPO}/data/gsm8k/train.jsonl')]
items = [dict(i=i, set=nm, q=tr[i]['question'], gold=C.gold_of(tr[i]['answer'])) for nm in ('dev', 'dev2') for i in sp[nm]]
if _o.environ.get('RD_SET') == 'dev3':        # the rest of split B (never used for choosing variants)
    import hashlib as _h
    _used = set(sp['dev']) | set(sp['dev2'])
    items = [dict(i=i, set='dev', q=r_['question'], gold=C.gold_of(r_['answer'])) for i, r_ in enumerate(tr)
             if C.part(r_['question']) == 'B' and r_['question'] not in shot_q and i not in _used]
if _o.environ.get('RD_SET') == 'scaled':      # GSM-Hard-like scaled-number train items of split B (not test): first 300 by seed 0
    _sc = C.load_train('B', which=('scaled',), shot_q=shot_q); _r.Random(0).shuffle(_sc)
    items = [dict(i=k, set='dev', q=x['q'], gold=x['gold']) for k, x in enumerate(_sc[:300])]
items = [r for r in items if len(C.q_ids(tok, r['q'])) + a.max_new <= 544]
if NLIM: items = items[:NLIM]
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
PROMPTS = {
    'off': lambda q: f"Question: {q}\nAnswer:",
    're2q': lambda q: f"Question: {q}\nRead the question again: {q}\nAnswer:",
    're2a': lambda q: f"Question: {q}\nAnswer: Let me read the question again: {q}\n",
    're2ask': lambda q: f"Question: {q}\nWhat is asked: {RT.ask_sentence(q)}\nAnswer:",
    're2n': lambda q: f"Question: {q}\nQuestion again: {q}\nAnswer:",
    're3q': lambda q: f"Question: {q}\nRead the question again: {q}\nRead the question again: {q}\nAnswer:",
    're2plan': lambda q: f"Question: {q}\nRead the question again: {q}\nAnswer: We need to find",
    'plan1': lambda q: f"Question: {q}\nAnswer: We need to find",
    'plan2': lambda q: f"Question: {q}\nAnswer: First, the quantities given and what is asked.\n",
}
def mark(t, w):
    s = ''; on = 0
    for x, f in zip(t, w):
        c = 1 if f & 1 else 0
        if c != on: s += '⟦' if c else '⟧'; on = c
        s += tok.decode([x])
    if on: s += '⟧'
    return s.replace('<|endoftext|>', '')
def dump(var, rows, dt):
    with open(f'{OUT}/{var}.jsonl', 'w') as f:
        for x in rows: f.write(_j.dumps(x) + '\n')
    for nm in ('dev', 'dev2'):
        rs = [x for x in rows if x['set'] == nm]
        if rs: print('RD', var, nm, len(rs), round(100 * sum(x['ok'] for x in rs) / len(rs), 2), 'tok', sum(x['ntok'] for x in rs), 'fires', sum(x.get('fire', 0) for x in rs), f'{dt:.0f}s', flush=True)
ENG0 = eng
def shots_engine(fmt):
    import json as _jj
    from micromachines.calcfmt import strip_solution
    shots = _jj.load(open(f'{C.REPO}/data/gsm8k/shots_v1.json')); ids = []
    for s_ in shots:
        b_, f_ = strip_solution(s_['solution'] + f"\n#### {s_['answer']}")
        ids += T(fmt(s_['question']) + f" {b_}\n#### {f_}\n\n")
    e2 = Engine(W, ids, tok, implant=imp, links=links, B_max=a.B_max, S_max=int(_o.environ.get('SAN_SMAX', 544)), use_write=USE_WRITE)
    e2.refresh(); return e2
for var in VARS:
    t0 = _t.time()
    GS = 0
    if '@T1g' in var:                    # T=1, G samples per item; ok = mean over the G samples
        var, _g = var.split('@T1g'); GS = int(_g); _gtag = f'{var}@T1g{GS}'
    if var in ('re2shots', 're2nshots', 're3shots') or ':' in var:      # 'SHOTFMT:QUERYPROMPT'
        _sk, _pk = var.split(':') if ':' in var else ({'re2shots': 're2q', 're2nshots': 're2n', 're3shots': 're3q'}[var],) * 2
        eng = shots_engine(PROMPTS[_sk]); _tag = var.replace(':', '__'); var = _pk
    else: eng = ENG0; _tag = None
    if var in PROMPTS or var.startswith('T1s'):
        pk = 'off' if var.startswith('T1s') else var
        qids = [T(PROMPTS[pk](r['q'])) for r in items]
        if GS:
            torch.manual_seed(1234); eng.seed.fill_(1234)
            o = eng.generate(qids, G=GS, T=1.0, greedy=False, max_new=a.max_new)
            rows = []
            for j, r in enumerate(items):
                ts = o['tokens'][j * GS:(j + 1) * GS]
                oks = [C.score(tok, t, r['gold'])[0] for t in ts]
                rows.append(dict(i=r['i'], set=r['set'], gold=r['gold'], ok=sum(oks) / GS, oks=oks, ntok=sum(map(len, ts)) / GS,
                                 preds=[RT.answer(C.score(tok, t, r['gold'])[2]) for t in ts], txt=mark(ts[0], o['wflag'][j * GS])))
            dump(_gtag.replace(':', '__'), rows, _t.time() - t0)
            if _tag: del eng; torch.cuda.empty_cache()
            continue
        if var.startswith('T1s'):
            sd = int(var[3:]); torch.manual_seed(sd); eng.seed.fill_(sd)
            o = eng.generate(qids, G=1, T=1.0, greedy=False, max_new=a.max_new)
        else:
            o = eng.generate(qids, G=1, greedy=True, max_new=a.max_new)
        rows = []
        for r, t, w in zip(items, o['tokens'], o['wflag']):
            ok, fmt, txt = C.score(tok, t, r['gold'])
            rows.append(dict(i=r['i'], set=r['set'], gold=r['gold'], ok=ok, ntok=len(t), pred=RT.answer(txt), txt=mark(t, w), tokens=t))
        dump(_tag or var, rows, _t.time() - t0)
        if _tag: del eng; torch.cuda.empty_cache()
        continue
    if var.startswith('casc'):            # casc_plan1-re2q-T1s0: on a sanity fire, restart greedily with the next prompt
        steps = var.split('_', 1)[1].split('-')
        cur = {(x['set'], x['i']): x for x in map(_j.loads, open(f'{OUT}/off.jsonl'))}
        rows = {}
        for k, x in cur.items():
            rows[k] = dict(i=x['i'], set=x['set'], gold=x['gold'], ok=x['ok'], ntok=x['ntok'], pred=x['pred'], fire=0, base_ok=x['ok'],
                           txt=x['txt'], tokens=x['tokens'], path='off')
        byk = {(r['set'], r['i']): r for r in items}
        for st in steps:
            todo = [k for k, x in rows.items() if RT.t_san(tok.decode(x['tokens']).replace('<|endoftext|>', ''))]
            if not todo: break
            pk = 'off' if st.startswith('T1s') else st
            qids = [T(PROMPTS[pk](byk[k]['q'])) for k in todo]
            if st.startswith('T1s'):
                sd = int(st[3:]); torch.manual_seed(sd); eng.seed.fill_(sd)
                o = eng.generate(qids, G=1, T=1.0, greedy=False, max_new=a.max_new)
            else:
                o = eng.generate(qids, G=1, greedy=True, max_new=a.max_new)
            for k, t, w in zip(todo, o['tokens'], o['wflag']):
                ok, fmt, txt = C.score(tok, t, rows[k]['gold'])
                rows[k].update(ok=ok, ntok=rows[k]['ntok'] + len(t), pred=RT.answer(txt), fire=rows[k]['fire'] + 1, tokens=t,
                               txt=rows[k]['txt'] + '【RESTART ' + st + '】' + mark(t, w), path=rows[k]['path'] + '>' + st)
        dump(var, list(rows.values()), _t.time() - t0); continue
    if var.startswith('nudge_'):
        _, trig, style = var.split('_')
        base = {(x['set'], x['i']): x for x in map(_j.loads, open(f'{OUT}/off.jsonl'))}
        todo, rows = [], []
        for r in items:
            b = base[(r['set'], r['i'])]
            txt = tok.decode(b['tokens']).replace('<|endoftext|>', '')
            fire = {'unused': RT.t_unused(r['q'], txt), 'unused3': RT.t_unused(r['q'], txt, 3), 'notlast': RT.t_notlast(txt), 'san': RT.t_san(txt),
                    'all': '####' in RT.body(txt)}[trig]
            row = dict(i=r['i'], set=r['set'], gold=r['gold'], ok=b['ok'], ntok=b['ntok'], pred=b['pred'], fire=int(fire), base_ok=b['ok'])
            if fire:
                cut = txt.split('\n\n')[0].rstrip('\n')
                if style == 'goal':               # write the asked sentence just before the host commits '####'
                    cut = cut.split('####')[0].rstrip('\n')
                    nud = f'The question asks: {RT.ask_sentence(r["q"])}\n'
                elif style in ('clause', 'number'): nud = RT.nudge_unused(r['q'], txt, style)
                elif style == 'generic': nud = 'Wait, let me check this. Let me read the question again and solve it step by step.\n'
                else: nud = 'Hmm, wait. Let me rethink this.\n'
                if nud is None: nud = 'Hmm, wait. Let me rethink this.\n'
                ctx = T(PROMPTS['off'](r['q'])) + T(cut + '\n' + nud)
                todo.append((len(rows), ctx, cut + '\n' + nud))
            rows.append(row)
        if todo:
            o = eng.generate([c for _, c, _ in todo], G=1, greedy=True, max_new=a.max_new)
            for (k, _, pre), t, w in zip(todo, o['tokens'], o['wflag']):
                ok, fmt, txt2 = C.score(tok, t, rows[k]['gold'])
                rows[k].update(ok=ok, pred=RT.answer(txt2), ntok=rows[k]['ntok'] + len(t) + len(T(pre)) - 0, txt=pre + '【' + mark(t, w) + '】')
        dump(var, rows, _t.time() - t0); continue
    print('unknown variant', var, flush=True)
print('RDDONE', flush=True)
