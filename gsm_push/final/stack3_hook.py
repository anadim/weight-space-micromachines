# Forward-only single-call stack with the forget-mask sanity implant (scope 2026-11-20, user decision 2).
# exec'd inside dig_san/tw_dump_lean.py --eval_ckpt ... --fused 0.
# StackH = CompEngine (reading agent: compiled input-token sanity DFA + pointer-register phrase writer + forget mask)
#          over C2Engine (calc v2, mode 'fwd', policy 'preempt'), plus the stop-token MISSING-answer trigger routed through
#          the same attempt registers (fire count s_nf, pointer s_fp, DFA reset, forget range [s_qlen, s_fr)).
# FORGET backend: STK_FORGET=kernel (reading agent's masked suffix-attention kernel; dev only) | weights (weight-space
# version, plugged in via STK_FORGET_MOD when it exists).
#   S0            calc v2 off, sanity off
#   S1            calc v2 fwd
#   S2F_<phrase>  S1 + compiled sanity (neg / non-int on the input token) + missing-answer stop-token trigger, forget on
#   S2Fn_<phrase> same without the missing-answer trigger
#   S2P_<phrase>  S1 + compiled sanity + missing trigger, forget OFF (plain phrase)
# STK_MISSING_ONLY=1 (GSM-Hard): the neg / non-int classes are off, only the missing-answer trigger stays.
import json as _j, os as _o, sys as _s, time as _t, random as _r, re as _re
_s.path.insert(0, _o.environ['STK_DIR']); _o.environ.setdefault('SAN_DIR', _o.environ['STK_DIR'])
import engine as EM
import fastqwen as Q
from c2_engine import C2Engine
import san_engine as SE
import comp_sanity as CS
from comp_sanity import CompEngine
import rd_trig as RT
import triton, torch.nn.functional as F
OUT = _o.environ['STK_OUT']; _o.makedirs(OUT, exist_ok=True)
SET = _o.environ['STK_SET']; CFGS = _o.environ.get('STK_CFGS', 'S0,S1,S2F_fwntf,S2Fn_fwntf,S2P_fwntf').split(',')
NLIM = int(_o.environ.get('STK_N', 0)); CAP0 = a.max_new; EXTRA = int(_o.environ.get('SAN_EXTRA', 320)); MAXF = 2
MISS_ONLY = _o.environ.get('STK_MISSING_ONLY', '0') == '1'
FORGET = _o.environ.get('STK_FORGET', 'kernel')
GEN_MAX = CAP0 + MAXF * EXTRA
LOG = open(f'{OUT}/{SET}.log', 'a')
def P(*x):
    s = ' '.join(str(y) for y in x); print('STK', SET, s, flush=True); LOG.write(s + '\n'); LOG.flush()


def items_of(nm):
    f = lambda it: [r for r in it if len(C.q_ids(tok, r['q'])) + CAP0 <= 544]
    if nm in ('dev', 'dev2'):
        it = f(C.load_train('B', which=('gsm',), shot_q=shot_q)); _r.Random(0).shuffle(it)
        return it[:600] if nm == 'dev' else it[600:900]
    if nm in ('test', 'hard'):
        al = C.load_test(nm, shot_q=shot_q); P('n_all', len(al)); return f(al)
    raise ValueError(nm)


items = items_of(SET)
if NLIM: items = items[:NLIM]
P('items', len(items), 'S_max', eng.S_max, 'gen_max', GEN_MAX, 'missing_only', MISS_ONLY, 'forget', FORGET)
qids = [C.q_ids(tok, r['q']) for r in items]


class StackH(CompEngine, C2Engine):
    def stack_setup(self):
        TT, CLS, TC, CN, CF, S = self.c_tables
        self.m_FLAG = torch.tensor([SE.flag_of(s) for s in S], device=self.dev)
        V = self.W['E'].shape[0]
        self.m_eos = torch.zeros(V, device=self.dev, dtype=torch.bool); self.m_eos[self.eos] = True
        self.m_on = True
        if MISS_ONLY: self.c_CN = torch.zeros_like(self.c_CN); self.c_CF = torch.zeros_like(self.c_CF)
        self._miss_ptr()

    def set_phrases(self, phrase_set, gain=None):
        CompEngine.set_phrases(self, phrase_set, gain)
        if hasattr(self, 'm_FLAG'): self._miss_ptr()

    def _miss_ptr(self):            # bank 3 = the 'missing' bank: its first token and pointer index
        T = lambda s: self.tok(s, add_special_tokens=False)['input_ids']
        ph = CS.PHRASES[self.c_phrase]; b1, b2 = T(ph[1]), T(ph[2])
        self.m_st3 = 1 + len(b1) + len(b2); self.m_tok3 = int(self.c_ids[self.m_st3 - 1])

    def _prefill(self, qs, rows):
        out = EM.Engine._prefill(self, qs, rows)
        lens = out[1]; G = rows[0].numel(); n = len(qs) * G
        self.s_qlen[:n] = lens.repeat_interleave(G)
        return out

    def _wave(self, qs, G, T, max_new, greedy):
        self.s_dfa.zero_(); self.s_nf.zero_(); self.s_fr.fill_(-1); self.s_fp.zero_(); self.s_rst.zero_()
        return C2Engine._wave(self, qs, G, T, max_new, greedy)

    def _mut(self):
        return SE.SanEngine._mut(self)

    def _attn_suffix(self, i, B, q, Kc, Vc, slot, att, sc, ieee):
        if self.c_forget and FORGET == 'kernel':
            CS.attn_suffix_mask_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.s_qlen, self.s_fr, self.mp, self.lpf, self.accp, att,
                                                   Kc.stride(0), Kc.stride(1), Kc.stride(2), sc, H=Q.H, HD=Q.HD, BN=64 if not ieee else 32,
                                                   IEEE=ieee, num_warps=4, num_stages=1 if ieee else 2)
        else:
            EM.HK.attn_suffix_kernel[(B, Q.KVH)](q, Kc, Vc, slot, self.mp, self.lpf, self.accp, att,
                                                 Kc.stride(0), Kc.stride(1), Kc.stride(2), sc, H=Q.H, HD=Q.HD, BN=64 if not ieee else 32,
                                                 IEEE=ieee, num_warps=4, num_stages=1 if ieee else 2)

    def _step_triton(self, B, S):
        HK = EM.HK                      # calc v2's sampler proxy (ALT token) around hostkern
        W, tok = self.W, self.tokb[:B]
        slot = self.slot[:B]
        imp = self.imp if self.imp_on else None
        emb = W['E'][tok]
        write = wf = None
        if imp is not None:
            e = imp.step(emb, B, tok) * self.row_on[:B, None]
            if self.use_write:
                write = (e @ self.Dp) if self.Dp is not None else (imp.last_w if imp.fk is not None else e @ imp.D)
            wf = (e.abs().sum(-1) > 0).to(torch.int8)
        cw, wbits, p2 = self._comp(B, tok)
        x, h, q, att = self.xb[:B], self.hb[:B], self.qb[:B], self.ab[:B]
        x.copy_(emb)
        pos = self.Lp + slot
        ieee = x.dtype == torch.float32
        sc = 1.0 / (Q.HD ** 0.5)
        delta = None
        for i, lw in enumerate(W['L']):
            HK.add_rms_kernel[(B,)](x, delta if delta is not None else x, lw['ln1'], h, D=Q.D, HAS_DELTA=delta is not None, eps=Q.EPS)
            qkv = F.linear(h, lw['Wqkv'])
            Kc, Vc = self.Kc[i], self.Vc[i]
            HK.qk_rope_kernel[(B,)](qkv, lw['qn'], lw['kn'], self.cos_t, self.sin_t, pos, q, Kc, Vc, slot,
                                    Kc.stride(0), Kc.stride(1), Kc.stride(2), H=Q.H, KVH=Q.KVH, HD=Q.HD, eps=Q.EPS)
            rbk = 8 if ieee else 32; ns = 1 if ieee else 2
            HK.attn_prefix_kernel[(Q.KVH, triton.cdiv(B, rbk))](q, self.Kp[i], self.Vp[i], self.mp, self.lpf, self.accp, B,
                                                                self.Lp, sc, H=Q.H, HD=Q.HD, RBK=rbk, BN=64 if not ieee else 32,
                                                                IEEE=ieee, num_warps=4, num_stages=ns)
            self._attn_suffix(i, B, q, Kc, Vc, slot, att, sc, ieee)
            o = F.linear(att.view(B, Q.H * Q.HD), lw['Wo'])
            HK.add_rms_kernel[(B,)](x, o, lw['ln2'], h, D=Q.D, HAS_DELTA=True, eps=Q.EPS)
            gu = F.linear(h, lw['Wgu'])
            act = self.actb[:B]
            HK.silu_mul_kernel[(B, Q.FF // 1024)](gu, act, FF=Q.FF, BLK=1024)
            delta = F.linear(act, lw['Wd'])
        x += delta
        if write is not None: x += write.to(x.dtype)
        x += cw.to(x.dtype)
        HK.add_rms_kernel[(B,)](x, x, W['lnf'], h, D=Q.D, HAS_DELTA=False, eps=Q.EPS)
        logits = F.linear(h, W['E'])
        nxt, lp = self.nxt_b[:B], self.lp_b[:B]
        HK.sample_kernel[(B,)](logits, self.temp, self.greedy, self.seed, nxt, lp, logits.shape[1], BV=2048, num_warps=8)
        self.seed += 1
        wf = wbits if wf is None else (wf | wbits)
        if self.m_on:   # missing answer at the stop token: the same attempt registers as the compiled trigger
            fl = self.m_FLAG[self.s_dfa[:B]]
            stopping = self.m_eos[nxt] | self.has_nn[nxt] | (self.ends_nl[tok] & self.starts_nl[nxt])
            room = (self.glen[:B] + self.c_Lmax + 24) < (self.s_cap0 + self.s_extra * self.s_nf[:B])
            mf = (fl == 3) & stopping & (self.s_fp[:B] == 0) & (self.s_nf[:B] < self.c_maxf) & room & ~self.done[:B] & (self.c_on > 0)
            nxt.copy_(torch.where(mf, torch.full_like(nxt, self.m_tok3), nxt))
            self.s_fp[:B] = torch.where(mf, torch.full_like(self.s_fp[:B], self.m_st3), self.s_fp[:B])
            self.s_nf[:B] += mf.long()
            self.s_dfa[:B] = torch.where(mf, torch.zeros_like(self.s_dfa[:B]), self.s_dfa[:B])
            if self.c_forget: self.s_fr[:B] = torch.where(mf, slot + 1, self.s_fr[:B])
            wf = wf | (mf.to(torch.int8) * 10)
        g0 = self.glen[:B].clone()
        self._book(B, tok, nxt, lp, wf, imp if imp is not None else True, None, None)
        self.done[:B] |= (g0 + 2) >= (self.s_cap0 + self.s_extra * self.s_nf[:B])


if FORGET == 'weights':     # the reading agent's weight-space forget (attempt counter + extra Q/K dims), when available
    import importlib
    _fm = importlib.import_module(_o.environ['STK_FORGET_MOD'])
    StackH = _fm.wrap_stack(StackH)          # must replace _attn_suffix (and its registers) with the weight-space version

eng.__class__ = StackH
eng.c2_setup()
eng.comp_setup('fwntf', maxf=MAXF, cap0=CAP0, extra=EXTRA, cache=_o.environ.get('DFA_CACHE', f"{_o.environ['STK_DIR']}/dfa_tables.pt"))
eng.stack_setup()
assert type(EM.HK).__name__ == '_HKProxy'
assert _o.environ.get('C2_NOCOMPACT') == '1'


def flag_of(txt):
    v = RT.answer(txt)
    if v is None: return 3
    if v < 0: return 1
    if v != int(v): return 2
    return 0


def score(ids, wf, gold):
    fires = [i for i, w in enumerate(wf) if w & 8]
    if not fires:
        ok, fmt, txt = C.score(tok, ids, gold); return ok, txt, 0
    f = fires[-1]
    txt = tok.decode(ids).replace('<|endoftext|>', '').split('\n\n')[0]
    pre = tok.decode(ids[:f]).replace('<|endoftext|>', '')
    tail = txt[len(pre):] if txt.startswith(pre) else ''
    m = _re.search(C.NUM, tail); pred = m.group(1).replace(',', '') if m else None
    return float(C.correct(pred, gold)), tail, len(fires)


def mark(t, w):
    s = ''; on = 0
    for x, f in zip(t, w):
        if f & 8: s += {1: '⟧', 2: '»', 3: '›'}.get(on, '') + '【SAN】'; on = 0
        c = 2 if f & 2 else (3 if f & 4 else (1 if f & 1 else 0))
        if c != on:
            s += {1: '⟧', 2: '»', 3: '›'}.get(on, ''); s += {1: '⟦', 2: '«', 3: '‹'}.get(c, ''); on = c
        s += tok.decode([x])
    s += {1: '⟧', 2: '»', 3: '›'}.get(on, '')
    return s.replace('<|endoftext|>', '')


def nfires(c2): return sum(1 for v in c2['events'] if v.get('act') in ('preempt', 'rewrite', 'rewrite_restate'))


base = {}
for cfg in CFGS:
    eng.set_c2('off' if cfg == 'S0' else 'fwd', exact=False, policy='preempt')
    if cfg in ('S0', 'S1'):
        eng.set_comp(False, forget=False); eng.m_on = False
    else:
        kind, ph = cfg.split('_', 1)
        eng.set_phrases(ph); eng.set_comp(True, forget=kind in ('S2F', 'S2Fn')); eng.m_on = kind != 'S2Fn'
    eng.graphs = {}; eng.pool = None; eng._outshape = None      # python-level switches (m_on, forget) are baked into the graphs
    t0 = _t.time()
    o = eng.generate(qids, G=1, T=1.0, greedy=True, max_new=GEN_MAX)
    dt = _t.time() - t0
    rows = []
    for r, t, w, c in zip(items, o['tokens'], o['wflag'], o['c2']):
        ok, txt, sf = score(t, w, r['gold'])
        ok0, _, txt0 = C.score(tok, t, r['gold'])
        rows.append(dict(gold=r['gold'], ok=ok, pred=RT.answer(txt), flag=flag_of(txt0), ntok=len(t), c2fires=nfires(c), san_fires=sf,
                         c2events=[e for e in c['events'] if e.get('act') == 'preempt'],
                         tokens=t, wflag=w, txt=mark(t, w)))
    with open(f'{OUT}/{SET}_{cfg}.jsonl', 'w') as f:
        for x in rows: f.write(_j.dumps(x) + '\n')
    n = len(rows); acc = 100 * sum(x['ok'] for x in rows) / n
    msg = [cfg, f'acc={acc:.2f}', 'tok/q', round(sum(x['ntok'] for x in rows) / n, 1), f'sec={dt:.0f}', 'rollback_tok', o.get('rollback_tok'),
           'c2_fire_rows', sum(x['c2fires'] > 0 for x in rows), 'san_fire_rows', sum(x['san_fires'] > 0 for x in rows)]
    base[cfg] = rows
    for ref in ('S0', 'S1'):
        if ref in base and ref != cfg:
            b = base[ref]
            fx = sum(1 for x, y in zip(rows, b) if x['ok'] and not y['ok']); br = sum(1 for x, y in zip(rows, b) if not x['ok'] and y['ok'])
            msg += [f'vs{ref}', f'+{fx}/-{br}']
            sil = [k for k in range(n) if (rows[k]['c2fires'] == 0 if (cfg == 'S1' and ref == 'S0') else rows[k]['san_fires'] == 0)]
            if (cfg == 'S1' and ref == 'S0') or (cfg != 'S1' and ref == 'S1'):
                msg += ['silent_identical', f"{sum(rows[k]['tokens'] == b[k]['tokens'] for k in sil)}/{len(sil)}"]
    P(*msg)
P('DONE')
print('STKDONE', flush=True)
