"""Branch-pair / fork tables for the branch probe (todo 'branch-probe'). Local, CPU.

usage: python3 probe/branch/build.py  -> probe/branch/tables/{forks,pairs}_{set}.jsonl.gz + job/tables/jobs_{set}.jsonl.gz

Per mixed item and visible-context group (k0 / waitN / rep, as atlas/diverge.group_of) we build the token trie of the
attempts' seg_tokens. A FORK = trie node (identical visible context => identical hidden state) with >= 2 child tokens.
Child value v = n_right / n of the attempts through that child (soft label).
  forks_{set}: one line per fork node: fid, idx, group, L (= offset in seg_tokens of the branching token), rep (gid of a
     representative attempt), children [[tok, n, n_right, sampled lp of one attempt (stored seg_logp)], ...] (by n desc),
     clean (no child token was written / calc-replaced), on_branch (some atlas branch pair sits here).
  pairs_{set}: atlas branch pairs (wrong attempt vs closest right attempt, atlas/diverge), deduped by (fork, tok_w, tok_r):
     fid, tok_r, tok_w, lp_r, lp_w (stored sampled), n_wrong (wrong attempts with this branch), cat (category of the
     first one), b_vs_div, b_type, b_rel, wtxt/rtxt (branch token strings), soft values v_r, v_w from the trie.
GPU job input jobs_{set}: attempts to teacher-force: gid, idx, q, seg_tokens, seg_wf, ok, plen, b (branch offset or -1),
  forks [[fid, L, [cand toks]]], ctrl (sample control forks along this attempt).
"""
import os, sys, json, gzip, collections, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); WS = os.path.dirname(os.path.dirname(HERE))
TAB = f'{WS}/probe/tables'; ATL = f'{WS}/atlas/data'
DIV = {'trainC': f'{TAB}/diverge_trainC.jsonl.gz', 'dev': f'{ATL}/diverge_dev.jsonl.gz', 'dev2': f'{ATL}/diverge_dev2.jsonl.gz'}


def rd(fn):
    with gzip.open(fn, 'rt') as f: return [json.loads(l) for l in f]


def wr(fn, rows):
    with gzip.open(fn, 'wt') as f:
        for r in rows: f.write(json.dumps(r) + '\n')


def group_of(a):
    if a['phrase_len'] == 0: return 'k0'
    return 'rep' if a['nrep'] > 0 else f"wait{a['phrase_len']}"


def trie_forks(atts):
    """atts: attempts of one (item, group). -> list of (L, prefix_owner_attempt, {tok: [atts]})"""
    out = []
    stack = [(0, atts)]
    while stack:
        L, S = stack.pop()
        ch = collections.defaultdict(list)
        for a in S:
            if len(a['seg_tokens']) > L: ch[a['seg_tokens'][L]].append(a)
        if len(ch) >= 2: out.append((L, S[0], dict(ch)))
        for t, sub in ch.items():
            if len(sub) >= 2: stack.append((L + 1, sub))
    return out


def run(S):
    A = rd(f'{TAB}/attempts_{S}.jsonl.gz'); I = {r['idx']: r for r in rd(f'{TAB}/items_{S}.jsonl.gz')}
    Dv = [r for r in rd(DIV[S]) if r['rec'] == 'att' and not r['ok'] and 'partner' in r]
    byg = {a['gid']: a for a in A}
    mixed = [a for a in A if I[a['idx']]['class'] == 'mixed']
    G = collections.defaultdict(list)
    for a in mixed: G[(a['idx'], group_of(a))].append(a)
    forks = []; fid_of = {}
    for (idx, g), atts in sorted(G.items()):
        for L, rep, ch in trie_forks(atts):
            ph = hashlib.md5(json.dumps(rep['seg_tokens'][:L]).encode()).hexdigest()[:12]
            fid = f'{S}/{idx}/{g}/{L}/{ph}'
            clean = all(not (a['seg_wf'][L] & 3) for sub in ch.values() for a in sub)
            children = sorted([[t, len(sub), sum(bool(a['ok']) for a in sub), sub[0]['seg_logp'][L]] for t, sub in ch.items()],
                              key=lambda c: -c[1])
            fid_of[(idx, g, L, ph)] = fid
            forks.append(dict(fid=fid, idx=idx, group=g, L=L, rep=rep['gid'], plen=rep['phrase_len'], children=children,
                              n=len(atts), clean=clean))
    F = {f['fid']: f for f in forks}
    pairs = {}
    for d in Dv:
        w = byg[d['gid']]; L = d['b_tok']; g = group_of(w)
        ph = hashlib.md5(json.dumps(w['seg_tokens'][:L]).encode()).hexdigest()[:12]
        fid = fid_of[(w['idx'], g, L, ph)]
        r = byg[d['partner']]; tw, tr = w['seg_tokens'][L], r['seg_tokens'][L]
        key = (fid, tr, tw)
        if key in pairs: pairs[key]['n_wrong'] += 1; continue
        ch = {c[0]: c for c in F[fid]['children']}
        pairs[key] = dict(fid=fid, idx=w['idx'], group=g, L=L, tok_r=tr, tok_w=tw, lp_r=d['lp_r'], lp_w=d['lp_w'], n_wrong=1,
                          cat=d['cat'], b_vs_div=d.get('b_vs_div'), b_type=d['b_type'], b_rel=d['b_rel'], wtxt=d['b_wtok'],
                          rtxt=d['b_rtok'], clean=F[fid]['clean'] and not d['b_calc'] and not d['b_rcalc'],
                          v_r=ch[tr][2] / ch[tr][1], v_w=ch[tw][2] / ch[tw][1], n_r=ch[tr][1], n_w=ch[tw][1],
                          b_line=d.get('b_line', ''), wcont=d.get('b_wcont', ''), rcont=d.get('b_rcont', ''), gid_w=w['gid'], gid_r=r['gid'])
        F[fid]['on_branch'] = True
    pairs = list(pairs.values())
    # GPU jobs: every clean fork node -> its representative attempt; dev/dev2 forward all mixed attempts (controls)
    need = collections.defaultdict(list)
    for f in forks:
        if not f['clean']: continue
        need[f['rep']].append([f['fid'], f['L'], [c[0] for c in f['children']][:8]])
    bpos = {d['gid']: d['b_tok'] for d in Dv}
    ctrl = S != 'trainC'
    jobs = []
    for a in (mixed if ctrl else [byg[g] for g in need]):
        jobs.append(dict(gid=a['gid'], idx=a['idx'], q=a['q'], seg_tokens=a['seg_tokens'], seg_wf=a['seg_wf'], ok=int(bool(a['ok'])),
                         plen=a['phrase_len'], b=bpos.get(a['gid'], -1), forks=need.get(a['gid'], []), ctrl=ctrl,
                         seg_logp=a['seg_logp']))
    wr(f'{HERE}/tables/forks_{S}.jsonl.gz', forks); wr(f'{HERE}/tables/pairs_{S}.jsonl.gz', pairs)
    wr(f'{HERE}/job/tables/jobs_{S}.jsonl.gz', jobs)
    nf = sum(f['clean'] for f in forks)
    nfv = sum(f['clean'] and len({round(c[2] / c[1], 3) for c in f['children']}) > 1 for f in forks)
    print(S, 'mixed items', len({a['idx'] for a in mixed}), 'groups', len(G), 'forks', len(forks), 'clean', nf,
          'clean w/ value diff', nfv, '| branch pairs (wrong atts)', len(Dv), 'unique', len(pairs),
          'clean', sum(p['clean'] for p in pairs), '| job attempts', len(jobs), 'fork pos', sum(len(j['forks']) for j in jobs),
          'tokens', sum(len(j['seg_tokens']) for j in jobs), flush=True)


if __name__ == '__main__':
    for S in (sys.argv[1:] or ['trainC', 'dev', 'dev2']): run(S)
