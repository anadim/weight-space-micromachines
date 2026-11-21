"""Unit tests for atlas/parse.py on synthetic token streams.  run: python3 atlas/test_parse.py"""
import os, sys, json, gzip, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse as PA

tk = PA.Tok()
WAIT_IDS, WAIT_NL_IDS = PA.Q_IDS['wait']
REP_IDS = PA.Q_IDS['rep'][0]


def build(pieces):
    """pieces: ('h', text) host tokens | ('c', text) calc-implant tokens (wf 1) | ('wait',) | ('waitnl',) | ('rep',)
    | ('final', ' 12') = '\\n####' + answer + eos | ('final0',) = bare eos.  -> tokens, wf"""
    T, W = [], []
    def q(ids):
        T.extend(ids); W.extend([10] + [2] * (len(ids) - 1))
    for p in pieces:
        if p[0] == 'h': ids = tk.enc(p[1]); T.extend(ids); W.extend([0] * len(ids))
        elif p[0] == 'c': ids = tk.enc(p[1]); T.extend(ids); W.extend([1] * len(ids))
        elif p[0] == 'f': ids = tk.enc(p[1]); T.extend(ids); W.extend([2] * len(ids))       # calc v2 forced token
        elif p[0] == 'wait': q(WAIT_IDS)
        elif p[0] == 'waitnl': q(WAIT_NL_IDS)
        elif p[0] == 'rep': q(REP_IDS)
        elif p[0] == 'final': q([198, 820] + tk.enc(p[1]) + [tk.eos])
        elif p[0] == 'final0': q([tk.eos])
    return T, W


def row_of(pieces, gold='12', **kw):
    T, W = build(pieces)
    txt = tk.decode(T).replace(PA.EOS_TXT, '')
    import re
    ms = list(re.finditer(PA.NUM, txt)); last = ms[-1].group(1).replace(',', '') if ms else None
    r = dict(idx=0, q='q?', gold=gold, tokens=T, wf=W, logp=[-0.5] * len(T), txt=txt, last=last,
             ok=int(PA.correct(last, gold)))
    r.update(kw)
    return r


S3 = [('h', ' 3 * 4 = 12\n#### 12\n'),                                   # k0 right
      ('wait',), ('h', 'He has 5 - 9 = -4\n#### -4\n'), ('rep',),           # k1: impossible, repaired
      ('h', ' the total. 7 + 5 = 12\n#### 12'),                             # (ends without newline: stop rule)
      ('waitnl',), ('h', 'So 2 * 6 = 13\n#### 13\n'),                       # k2 wrong
      ('final', ' 12')]


def test_split_and_answers():
    r = row_of(S3)
    P = PA.parse_stream(r, tk)
    assert P['anomalies'] == [], P['anomalies']
    assert [len(a) for a in P['attempts']] == [1, 2, 1]
    A = PA.attempt_rows(r, P, tk, 'dev', 7)
    assert [a['ans'] for a in A] == ['12', '12', '13']
    assert [a['ok'] for a in A] == [1, 1, 0]
    assert [a['nrep'] for a in A] == [0, 1, 0]
    assert A[1]['prev_segments'][0].startswith('Wait, let me solve it again.\nHe has') and '#### -4' in A[1]['prev_segments'][0]
    assert A[1]['text'].startswith(' We need to find the total.') and A[1]['phrase_len'] == len(REP_IDS)
    assert A[2]['text'] == '\nWait, let me solve it again.\nSo 2 * 6 = 13\n#### 13\n'     # final vote line excluded
    assert A[0]['phrase_len'] == 0 and A[0]['seg_start'] == 0
    assert P['final']['ans'] == '12' and P['final']['start'] == A[2]['seg_start'] + len(A[2]['seg_tokens'])
    for a in A:                                                              # hash_off = token completing '####'
        assert tk.ts(a['seg_tokens'][a['hash_off']]).strip() == '####'
        assert tk.decode(a['seg_tokens'][:a['hash_off'] + 1]).endswith('####')
    # steps: line spans after the written phrase, non-empty, contiguous ends
    st = A[2]['steps']
    assert [tk.decode(A[2]['seg_tokens'][s:e]) for s, e in st] == ['So 2 * 6 = 13\n', '#### 13\n'], st
    assert [tk.decode(A[1]['seg_tokens'][s:e]) for s, e in A[1]['steps']] == [' We need to find the total. 7 + 5 = 12\n', '#### 12']
    assert A[0]['bad'] is False and A[1]['bad'] is False and len(A[0]['seg_logp']) == len(A[0]['seg_tokens'])


def test_register_port_matches_regex():
    cases = {' #### 1,234.50\n': (True, 1234.5, True), '####\n-7\n': (True, -7.0, True), '#### 12.0\n': (True, 12.0, False),
             '#### 0.05': (True, 0.05, True), 'no answer here\n': (False, 0.0, True), '#### ,\n': (False, 0.0, True),
             '#### $5\n': (False, 0.0, True), '#### 40 dollars\n': (True, 40.0, False)}
    for text, (valid, val, bad) in cases.items():
        reg = PA.Reg(tk)
        for t in tk.enc(text): reg.read(t)
        v, b, x = reg.commit()
        assert (v, b) == (valid, bad) and abs(x - val) < 1e-9, (text, v, b, x)
        import re
        m = re.search(PA.NUM, text); ans = m.group(1).replace(',', '') if m else None
        if valid: assert abs(float(ans) - val) < 1e-9 and PA.bad_of(ans) == bad, text
    # '####' split over several all-'#' tokens still opens the reader; non-live (queue) tokens are not read
    reg = PA.Reg(tk); ids = tk.enc('##') + tk.enc('##') + tk.enc(' 9\n')
    starts = [reg.read(t)[0] for t in ids]
    assert starts.count(True) == 1 and reg.commit()[2] == 9.0
    reg = PA.Reg(tk)
    for t in [820] + tk.enc(' 9'): reg.read(t, live=False)
    assert reg.commit()[0] is False


def test_bare_eos_final_and_empty_segment_and_calc_tokens():
    pieces = [('h', ' no idea\n'), ('rep',), ('wait',), ('f', 'x'), ('h', ' = '), ('c', '6'), ('h', '\n#### 6\n'), ('final0',)]
    r = row_of(pieces, gold='6')
    P = PA.parse_stream(r, tk)
    assert P['anomalies'] == [], P['anomalies']
    A = PA.attempt_rows(r, P, tk, 'dev', 7)
    assert len(A) == 2 and A[0]['nrep'] == 1 and A[0]['ans'] is None and A[0]['bad'] and A[0]['hash_off'] is None
    assert A[0]['text'] == ' We need to find' and A[0]['steps'] == []
    assert A[1]['ans'] == '6' and A[1]['ok'] == 1 and 1 in A[1]['seg_wf'] and A[1]['phrase_len'] == len(WAIT_IDS)
    assert A[1]['text'].startswith(PA.WAIT + 'x = 6')
    assert P['final']['ans'] is None and P['final']['txt'] == ''
    # calc v2 overwrote the first token of a repair queue (number still open when the attempt was capped)
    r = row_of([('h', ' 20 - 24 = -'), ('h', '4'), ('h', ' need to find'), ('h', ' x\n#### 3\n'), ('final', ' 3')], gold='3')
    r['wf'] = [0] * len(r['wf']); i = len(tk.enc(' 20 - 24 = -')); r['wf'][i] = 10
    for j in range(i + 1, i + 4): r['wf'][j] = 2
    r['wf'][-5:] = [10, 2, 2, 2, 2]
    P = PA.parse_stream(r, tk)
    assert [x.split(':')[0] for x in P['anomalies']] == ['c2first'] and [len(a) for a in P['attempts']] == [2]
    A = PA.attempt_rows(r, P, tk, 'dev', 7)
    assert A[0]['text'] == '4 need to find x\n#### 3\n' and A[0]['phrase_len'] == 4 and A[0]['ok'] == 1 and A[0]['complete']


def test_unknown_queue_and_closed_answer_continuation_flagged():
    r = row_of([('h', ' 1\n#### 4\nmore text\n'), ('final', ' 4')])
    assert any('continues after' in x for x in PA.parse_stream(r, tk)['anomalies'])
    r['wf'] = list(r['wf']); r['wf'][2] |= 8
    assert any('unknown queue' in x for x in PA.parse_stream(r, tk)['anomalies'])


def test_end_to_end_tables():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_test_tmp')
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    try:
        def reg_of(r):
            P = PA.parse_stream(r, tk); A = PA.attempt_rows(r, P, tk, 'x', 0)
            return dict(val=[a['port_val'] for a in A] + [0.0] * (8 - len(A)), rok=[a['port_valid'] for a in A] + [False] * (8 - len(A)),
                        bad=[a['port_bad'] for a in A] + [False] * (8 - len(A)), lg=[1.0] * 8,
                        end=[a['seg_start'] + len(a['seg_tokens']) - 1 for a in A] + [0] * (8 - len(A)),
                        natt=len(A), nseg=sum(len(s) for s in P['attempts']), pick=0)
        rows = {7: [], 11: []}
        for sd in rows:
            for idx, gold in enumerate(['12', '99']):
                r = row_of(S3, gold=gold, idx=idx); r.update(reg_of(r)); r['idx'] = idx; rows[sd].append(r)
            with gzip.open(f'{d}/x_N8_R1_v1_s1_m0_T1.0_seed{sd}.jsonl.gz', 'wt') as f:
                for r in rows[sd]: f.write(json.dumps(r) + '\n')
            with gzip.open(f'{d}/x_N1_R2_v1_s1_m0_greedy_seed{sd}.jsonl.gz', 'wt') as f:
                for r in rows[sd]: f.write(json.dumps(dict(r, ok=1 - r['idx'])) + '\n')
        out = []
        res = PA.run_set('x', d, d, tk, out.append)
        assert res['S']['ans_agree'] == res['S']['att'] == 12 and res['S']['end_agree'] == 12 and res['S']['port_agree'] == 12
        with gzip.open(f'{d}/items_x.jsonl.gz', 'rt') as f: IT = [json.loads(l) for l in f]
        assert [(i['n_right'], i['n_att'], i['class'], i['greedy_ok']) for i in IT] == [(4, 6, 'mixed', 1), (0, 6, 'never', 0)]
        assert IT[0]['vote_ok'] == {'7': 1, '11': 1} and IT[1]['vote_ok'] == {'7': 0, '11': 0}
        with gzip.open(f'{d}/attempts_x.jsonl.gz', 'rt') as f: AT = [json.loads(l) for l in f]
        assert len(AT) == 12 and len({a['gid'] for a in AT}) == 12
        txt = '\n'.join(out)
        assert '| 7 | 50.00 | 50.00 | 0 (0.00%) | 1 (50.00%) |' in txt, txt
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): f(); print('ok', k)
    print('ALL OK')
