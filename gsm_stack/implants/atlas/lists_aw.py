"""Item lists for the mechanistic lanes (M1-M5), from an all-weights atlas dir (parse.py + diverge_aw.py outputs).

usage: python3 atlas/lists_aw.py DATA_DIR OUT_DIR [--seeds ...] [--hand atlas/data_aw/hand_labels_aw.tsv]
writes OUT_DIR/{m12_mixed,m3_dropped_target,m4_unit_world,m5_attractor}_aw.json (+ a printed summary)

Token offsets: seg_start = index of the attempt's final visible segment in the row's generated `tokens` (aw/out rows);
div_tok = offset (in seg_tokens, i.e. from seg_start) of the first token of the first divergent value step (diverge.py step D);
div_line_tok = offset of the start of that line; b_tok = token branch point vs the right attempt of the same context group with
the longest common prefix (b_partner), offsets from seg_start too.  Absolute stream index = seg_start + offset.
"""
import os, sys, re, json, gzip, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steps import question_numbers      # noqa: E402

FACTS = [  # (question regex, fact needed)
    (r'\bdozens?\b', 'dozen = 12'), (r'\bweeks?\b|\bweekly\b', 'week = 7 days (5 weekdays / school days, 2 weekend days)'),
    (r'\bhours?\b|\bhourly\b', 'hour = 60 minutes'), (r'\bminutes?\b', 'minute = 60 seconds; hour = 60 minutes'),
    (r'\bdays?\b|\bdaily\b', 'day = 24 hours'), (r'\bmonths?\b|\bmonthly\b', 'month ~ 4 weeks / 30 days; year = 12 months'),
    (r'\byears?\b|\bannual', 'year = 12 months = 52 weeks = 365 days'), (r'\bpairs?\b', 'pair = 2'),
    (r'\bquarts?\b', 'quart = 32 oz; gallon = 4 quarts'), (r'\bgallons?\b', 'gallon = 4 quarts = 128 oz'),
    (r'\bpints?\b', 'pint = 16 oz; quart = 2 pints'), (r'\bcups?\b', 'cup = 8 oz'),
    (r'\bfeet\b|\bfoot\b', 'foot = 12 inches'), (r'\byards?\b', 'yard = 3 feet = 36 inches'), (r'\bmiles?\b', 'mile = 5280 feet'),
    (r'\bpounds?\b', 'pound = 16 ounces'), (r'\bkilo|\bkg\b', 'kg = 1000 g'), (r'\bliters?\b|\blitres?\b', 'liter = 1000 ml'),
    (r'\bcents?\b|\bpenn(y|ies)\b', 'dollar = 100 cents'), (r'\bquarters?\b', 'quarter = 25 cents / 1/4'),
    (r'\bdimes?\b', 'dime = 10 cents'), (r'\bnickels?\b', 'nickel = 5 cents'), (r'\bdecades?\b', 'decade = 10 years'),
    (r'\bcentur', 'century = 100 years'), (r'\bscore\b', 'score = 20 (if numeric)'), (r'\bgross\b', 'gross = 144'),
    (r'\bhorse', 'horse = 4 horseshoes / 4 legs'), (r'\btires?\b|\bwheels?\b', 'car = 4 wheels, bike = 2, tricycle = 3'),
    (r'\bcouples?\b', 'couple = 2 people'), (r'\btwins?\b', 'twins = 2'), (r'\btriplets?\b', 'triplets = 3'),
    (r'\bBC\b|\bborn\b', 'calendar: born later = larger AD year; BC years count down'),
    (r'round trip|one way|and back|back home', 'round trip = 2 x one way'), (r'\bleap\b', 'leap year = 366 days'),
    (r'\blegs?\b', 'legs: person 2, dog/cat/horse 4, spider 8, insect 6'), (r'\bdeck\b|\bcards?\b', 'deck = 52 cards'),
    (r'\bsides?\b|\bsquare\b|\btriangle\b|\bhexagon|\bpentagon|\boctagon', 'square 4 sides, triangle 3, pentagon 5, hexagon 6, octagon 8'),
    (r'\bpercent|%', 'percent = /100'), (r'\bseconds?\b', 'minute = 60 seconds'),
]
UNIT_SUBS = {'world_fact', 'missing_factor', 'factor', 'const', 'count_off_by_one'}


def rd(fn):
    with gzip.open(fn, 'rt') as f:
        return [json.loads(l) for l in f]


def first_line(t, plen_txt=''):
    for l in t.split('\n'):
        l = l.strip()
        if l and not l.startswith('Wait, let me') and l != 'We need to find': return l[:160]
    return ''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('d'); ap.add_argument('out'); ap.add_argument('--seeds', default='')
    ap.add_argument('--sets', default='dev,dev2'); ap.add_argument('--hand', default='')
    a = ap.parse_args(); seeds = {int(x) for x in a.seeds.split(',') if x}
    hand = collections.defaultdict(list)                      # (set, idx) -> [(gid, cat, fact/note)]
    if a.hand and os.path.exists(a.hand):
        for l in open(a.hand):
            if l.startswith('#') or not l.strip(): continue
            p = l.rstrip('\n').split('\t')
            gid, cat = p[0], p[1]; note = p[2] if len(p) > 2 else ''
            s_, i_ = gid.split('/')[:2]; hand[(s_, int(i_))].append((gid, cat, note))
    M12, M3, M4, M5 = [], [], [], []
    for s in a.sets.split(','):
        A = [x for x in rd(f'{a.d}/attempts_{s}.jsonl.gz') if not seeds or x['seed'] in seeds]
        I = {it['idx']: it for it in rd(f'{a.d}/items_{s}.jsonl.gz')}
        D = rd(f'{a.d}/diverge_{s}.jsonl.gz')
        dv = {r['gid']: r for r in D if r['rec'] == 'att'}; di = {r['idx']: r for r in D if r['rec'] == 'item'}
        by = collections.defaultdict(list)
        for x in A: by[x['idx']].append(x)
        for idx, L in sorted(by.items()):
            it = I[idx]; nr = sum(x['ok'] for x in L); nw = len(L) - nr; item = di.get(idx)
            atts = []
            for x in sorted(L, key=lambda x: (x['seed'], x['k'])):
                r = dv.get(x['gid'], {})
                atts.append(dict(gid=x['gid'], seed=x['seed'], k=x['k'], ok=x['ok'], ans=x['ans'], lg=x['lg'], nrep=x['nrep'],
                                 seg_start=x['seg_start'], phrase_len=x['phrase_len'], ntok=len(x['seg_tokens']),
                                 cat=r.get('cat'), sub=r.get('sub'), detail=r.get('detail'), div_tok=r.get('div_tok'),
                                 div_line_tok=r.get('div_line_tok'), div_line=r.get('div_line'), div_kind=r.get('div_kind'),
                                 b_tok=r.get('b_tok'), b_partner=r.get('partner'), b_wtok=r.get('b_wtok'), b_rtok=r.get('b_rtok'),
                                 lp_w=r.get('lp_w'), lp_r=r.get('lp_r'), right_line=r.get('right_line'),
                                 first_line=first_line(x['text'])))
            wrong = [t for t in atts if not t['ok']]
            cats = collections.Counter(t['cat'] for t in wrong if t['cat'])
            base = dict(set=s, idx=idx, q=it['q'], gold=it['gold'], n_right=nr, n_wrong=nw, n_att=len(L),
                        vote_ok=it['vote_ok'], cats=dict(cats), dominant_cat=item['dominant_cat'] if item else None)
            # (a) M1/M2: balanced mixed items
            if nr >= 4 and nw >= 4:
                M12.append(dict(base, attempts=atts))
            # (b) M3: dropped quantity / wrong target
            if wrong:
                dt = sum(cats[c] for c in ('dropped', 'target'))
                hd = [h for h in hand.get((s, idx), []) if h[1] in ('dropped', 'target')]
                if (cats and dt / max(1, sum(cats.values())) >= 0.5 and dt >= 2) or hd:
                    qn = sorted(float(q) for q in question_numbers(it['q']) if q is not None)
                    M3.append(dict(base, dropped_target_share=round(dt / max(1, sum(cats.values())), 2), question_numbers=qn,
                                   hand=hd, examples=[dict(gid=t['gid'], ans=t['ans'], cat=t['cat'], sub=t['sub'], detail=t['detail'],
                                                           div_line=t['div_line'], right_line=t['right_line'], div_tok=t['div_tok'],
                                                           seg_start=t['seg_start'])
                                                      for t in wrong if t['cat'] in ('dropped', 'target')][:6]))
            # (c) M4: unit / world knowledge
            uw = [t for t in wrong if t['cat'] == 'unit' or t['sub'] in UNIT_SUBS]
            hu = [h for h in hand.get((s, idx), []) if h[1] == 'unit']
            if len(uw) >= 2 or (hu and wrong):
                facts = [f for rx, f in FACTS if re.search(rx, it['q'], re.I)]
                M4.append(dict(base, n_unit_wrong=len(uw), facts_from_question=facts,
                               hand_fact=[h[2] for h in hu], details=collections.Counter(t['detail'] for t in uw).most_common(5),
                               examples=[dict(gid=t['gid'], ans=t['ans'], sub=t['sub'], detail=t['detail'], div_line=t['div_line'],
                                              right_line=t['right_line'], div_tok=t['div_tok'], seg_start=t['seg_start']) for t in uw[:4]]))
            # (d) M5: attractor (attempts collapse onto one wrong answer)
            if nw >= 4:
                wa = collections.Counter(t['ans'] for t in wrong)
                att, an = wa.most_common(1)[0]
                per_seed = collections.Counter((t['seed'], t['ans']) for t in wrong if t['ans'] is not None)
                streams_collapsed = sum(1 for (sd, v), c in per_seed.items() if c >= 3 and v == att)
                if att is not None and an >= 4 and an / nw >= 0.5:
                    fr = [t['first_line'] for t in atts if t['ans'] == att][:4]
                    fr_r = [t['first_line'] for t in atts if t['ok']][:4]
                    M5.append(dict(base, attractor=att, attractor_n=an, attractor_share_wrong=round(an / nw, 2),
                                   attractor_beats_right=an > nr, streams_with_3plus_attractor=streams_collapsed,
                                   attractor_type=item['attractor_type'] if item else ('never' if nr == 0 else None),
                                   wrong_answers=dict(wa.most_common(5)), attractor_first_lines=fr, right_first_lines=fr_r,
                                   attractor_k=sorted(t['k'] for t in wrong if t['ans'] == att)))
    os.makedirs(a.out, exist_ok=True)
    for nm, L in (('m12_mixed', M12), ('m3_dropped_target', M3), ('m4_unit_world', M4), ('m5_attractor', M5)):
        with open(f'{a.out}/{nm}_aw.json', 'w') as f: json.dump(L, f, indent=1)
        cs = collections.Counter(x['set'] for x in L)
        print(f'{nm}_aw.json: {len(L)} items ({dict(cs)})')
    print('M5: attractor beats right count in', sum(x['attractor_beats_right'] for x in M5), 'items; never-right attractors',
          sum(x['n_right'] == 0 for x in M5))


if __name__ == '__main__':
    main()
