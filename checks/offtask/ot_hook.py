# Off-task benchmarks with the rung-2 implants attached. exec'd in dig_san/tw_dump_lean.py (eval_ckpt mode,
# --arm 2 --fused 0 [--host_ckpt base | RUN/step_000300]) exactly like final_eval fe_hook.py, so `imp` (calc_s4) and
# the chain2 machine are built as at GSM8K eval (DualImplant = calc_s4 + chain2, chain2 gate).
# Every forward computes BOTH: host only, and host + DualImplant write (e @ D added to the final pre-norm residual at
# every position where the implant output e != 0, as Engine._step / _prefill do). Implant input = the eval host's
# token embeddings (W['E'][tok] in the engine). Serial layer (SerialWD writer, WPR, forget mask): its only trigger
# inside attempt 1 is END (answer closed in the '#### number' DFA | stop token eos / blank line / '\n' then '\n' |
# cap); we run that trigger on every continuation with the stack's own tables (san_engine DFA cache) and count it.
# WPR offset and forget mask are 0 / empty during attempt 1 by construction (they act only from attempt 2 on).
# env: OT_OUT dir, OT_TAG row tag, OT_DATA dir with *.jsonl, OT_TASKS (mmlu,arcc,hs,wg,nq), OT_LIM (items per task, 0 = all),
#      OT_TOKB (tokens per forward batch), HOST_MODEL.
import json as _j, os as _o, sys as _s, time as _t, re as _re, string as _str
_s.path.insert(0, _o.environ['STK_DIR'])
import torch.nn.functional as _F
E = _o.environ
OUT = E['OT_OUT']; TAG = E['OT_TAG']; DATA = E['OT_DATA']; _o.makedirs(OUT, exist_ok=True)
TASKS = E.get('OT_TASKS', 'mmlu,arcc,hs,wg,nq').split(','); LIM = int(E.get('OT_LIM', 0)); TOKB = int(E.get('OT_TOKB', 49152))
HOSTM = E.get('HOST_MODEL', 'Qwen/Qwen3-0.6B-Base')
dev = 'cuda'


def P(*x): print('OT', *x, flush=True)


# ---- implants exactly as fe_hook.py builds them
import chain2
chain2.register(tok)
from implant import CalcImplant
from implant2 import DualImplant
from transformers import AutoModelForCausalLM
_hf = AutoModelForCausalLM.from_pretrained(HOSTM, dtype=torch.bfloat16).to(dev).eval()
DUAL = DualImplant(imp, CalcImplant(_hf, tok, a.B_max, machine='chain2', fused=False))
del _hf
for _n in ('Kc', 'Vc'):
    if hasattr(eng, _n): setattr(eng, _n, None)
torch.cuda.empty_cache()
DW = DUAL.D.to(dev)                                 # (n_out, d) fp32 write directions incl. gain
KWIN = DUAL.K + DUAL.W
P('implant', 'n_out', DUAL.n_out, 'n_calc', DUAL.n_calc, 'K', DUAL.K, 'W', DUAL.W, 'D', tuple(DW.shape))
# positive control: the fire detector must see the calculator / chain on GSM-style arithmetic
for _s0 in ("Natalia sold 48/2 = 24 clips in May. So she sold 48 + 24 = 72 clips.",
            "The total is $585.80 + $565 + $599.20 + $567 = 2317"):
    _ids = torch.tensor([tok(_s0, add_special_tokens=False)['input_ids']], device=dev)
    _f = DUAL.full(W['E'][_ids]).abs().sum(-1)[0] > 0
    P('posctl', int(_f.sum()), 'of', _ids.shape[1], [tok.decode([int(_ids[0, j])]) for j in torch.nonzero(_f)[:, 0].tolist()][:12])

# ---- eval host = the engine's host (base or --host_ckpt), as an HF model
H = AutoModelForCausalLM.from_pretrained(HOSTM, dtype=torch.bfloat16).to(dev).eval()
if a.host_ckpt and a.host_ckpt != 'base':
    from safetensors.torch import load_file as _lf
    _sd = _lf(f'{a.host_ckpt}/model.safetensors', device='cpu')
    _mk, _uk = H.load_state_dict(_sd, strict=False)
    _mk = [k for k in _mk if k != 'lm_head.weight']
    assert not _mk and not _uk, (_mk[:5], _uk[:5])
    H.tie_weights(); del _sd
P('host', a.host_ckpt or 'base', 'tied', bool(H.lm_head.weight.data_ptr() == H.model.embed_tokens.weight.data_ptr()))
# consistency with the engine's host weights (W['E'] is the fused copy the stack used at GSM8K eval)
P('emb_vs_engine_maxabs', float((H.model.embed_tokens.weight - W['E']).abs().max()))
EMB = H.model.embed_tokens.weight
NORM = H.model.norm
LMW = H.lm_head.weight
_cap = {}
NORM.register_forward_pre_hook(lambda m, args: _cap.__setitem__('x', args[0]) or None)


def implant_e(ids):
    """ids (B,T) -> fire mask (B,T) bool and write (B,T,d) fp32 (teacher-forced: e at every position)."""
    e = DUAL.full(EMB[ids])
    fire = e.abs().sum(-1) > 0
    return fire, (e.float() @ DW) if bool(fire.any()) else None


# ---- the serial layer's END trigger (same fixed tables as comp_serial.sw_setup)
import san_engine as SE
V = EMB.shape[0]
_txt = [tok.decode([i]) for i in range(len(tok))] + [''] * max(0, V - len(tok))
_TT, _FLAG, _S = SE.build_tables(_txt, f"{E['STK_DIR']}/dfa_tables.pt")
TT = _TT.cpu().long(); DONE = [s[0] == 'X' for s in _S]
EOS = tok.convert_tokens_to_ids('<|endoftext|>')
STOP = [('\n\n' in t) for t in _txt]; STOP[EOS] = True
ENL = [t.endswith('\n') for t in _txt]; SNL = [t.startswith('\n') for t in _txt]
ACAP = a.max_new


def serial_end(cont):
    """first index j in cont where the attempt-1 END trigger fires reading cont[j] as input (-1: never)."""
    d = 0; pe = False
    for j, t in enumerate(cont):
        d1 = int(TT[t, d])
        closed = DONE[d1] and not DONE[d]
        stopin = STOP[t] or (pe and SNL[t])
        if closed or stopin or j + 2 >= ACAP: return j
        pe = ENL[t]; d = d1
    return -1


def batches(lens, tokb):
    order = sorted(range(len(lens)), key=lambda i: -lens[i])
    b = []; mx = 0
    for i in order:
        m = max(mx, lens[i])
        if b and m * (len(b) + 1) > tokb:
            yield b; b = []; m = lens[i]
        b.append(i); mx = m
    if b: yield b


@torch.no_grad()
def score(reqs):
    """reqs: list of (ctx_ids, cont_ids, cand) with cand None (score cont) or a list of single tokens scored at the last
    ctx position (cont empty). Returns per req dict: lp_h, lp_a (lists for cand), fsc (implant fires on scored
    positions), fall (fires anywhere in ctx+cont), fpos (n positions), end (serial END index in cont that affects a
    scored token, -1)."""
    res = [None] * len(reqs)
    seqs = [c + x for c, x, _ in reqs]
    for b in batches([len(s) for s in seqs], TOKB):
        T = max(len(seqs[i]) for i in b)
        ids = torch.full((len(b), T), EOS, dtype=torch.long)
        am = torch.zeros((len(b), T), dtype=torch.long)
        for r, i in enumerate(b):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i]); am[r, :len(seqs[i])] = 1
        ids = ids.to(dev); am = am.to(dev)
        H.model(input_ids=ids, attention_mask=am, use_cache=False)
        x = _cap.pop('x')                                           # (B,T,d) pre-norm residual (bf16)
        fire, wr = implant_e(ids)
        for r, i in enumerate(b):
            c, xc, cand = reqs[i]
            L = len(seqs[i]); p0 = len(c) - 1
            pos = list(range(p0, L - 1)) if cand is None else [p0]
            pt = torch.tensor(pos, device=dev)
            xs = x[r, pt]
            lh = torch.log_softmax(_F.linear(NORM(xs), LMW).float(), -1)
            fs = fire[r, pt]
            if wr is not None and bool(fs.any()):
                xa = xs + wr[r, pt].to(xs.dtype)
                la = torch.log_softmax(_F.linear(NORM(xa), LMW).float(), -1)
            else:
                la = lh
            if cand is None:
                tg = torch.tensor(xc, device=dev)[:, None]
                lp_h = float(lh.gather(-1, tg).sum()); lp_a = float(la.gather(-1, tg).sum())
                g_h = bool((lh.argmax(-1) == tg[:, 0]).all())
            else:
                ct = torch.tensor(cand, device=dev)
                lp_h = lh[0, ct].tolist(); lp_a = la[0, ct].tolist(); g_h = None
            en = serial_end(xc) if xc else -1
            res[i] = dict(lp_h=lp_h, lp_a=lp_a, fsc=int(fs.sum()), fall=int(fire[r, :L].sum()), npos=len(pos),
                          end=en if (0 <= en < len(xc) - 1) else -1, end_any=en, gh=g_h)
    return res


def rd(name):
    return [_j.loads(l) for l in open(f'{DATA}/{name}.jsonl')]


T_ = lambda s: tok(s, add_special_tokens=False)['input_ids']


def lim(x):
    return x[:LIM] if LIM else x


def dump(task, rows, extra):
    with open(f'{OUT}/{TAG}_{task}.jsonl', 'w') as fh:
        for x in rows: fh.write(_j.dumps(x) + '\n')
    P('done', task, TAG, _j.dumps(extra))


def mc_task(task, items):
    """items: list of (ctx, [conts], gold). acc (sum lp), acc_norm (lp / chars), host and attached."""
    reqs = []; idx = []
    for k, (ctx, conts, gold) in enumerate(items):
        ci = T_(ctx)
        for j, ct in enumerate(conts):
            reqs.append((ci, T_(ct), None)); idx.append((k, j))
    t0 = _t.time(); R = score(reqs); dt = _t.time() - t0
    rows = [dict(i=k, gold=g, lp_h=[], lp_a=[], nch=[len(ct) for ct in conts], fsc=0, fall=0, npos=0, end=0, end_any=0)
            for k, (_, conts, g) in enumerate(items)]
    for (k, j), r in zip(idx, R):
        x = rows[k]; x['lp_h'].append(r['lp_h']); x['lp_a'].append(r['lp_a'])
        x['fsc'] += r['fsc']; x['fall'] += r['fall']; x['npos'] += r['npos']; x['end'] += int(r['end'] >= 0); x['end_any'] += int(r['end_any'] >= 0)
    am = lambda v: max(range(len(v)), key=lambda q: v[q])
    for x in rows:
        x['ok_h'] = int(am(x['lp_h']) == x['gold']); x['ok_a'] = int(am(x['lp_a']) == x['gold'])
        x['okn_h'] = int(am([l / n for l, n in zip(x['lp_h'], x['nch'])]) == x['gold'])
        x['okn_a'] = int(am([l / n for l, n in zip(x['lp_a'], x['nch'])]) == x['gold'])
    n = len(rows)
    ex = dict(n=n, acc_h=100 * sum(x['ok_h'] for x in rows) / n, acc_a=100 * sum(x['ok_a'] for x in rows) / n,
              accn_h=100 * sum(x['okn_h'] for x in rows) / n, accn_a=100 * sum(x['okn_a'] for x in rows) / n,
              seq_fire=sum(1 for x in rows if x['fsc']) / n, pos_fire=sum(x['fsc'] for x in rows) / max(1, sum(x['npos'] for x in rows)),
              seq_fire_anywhere=sum(1 for x in rows if x['fall']) / n, end_items=sum(1 for x in rows if x['end']), sec=round(dt))
    dump(task, rows, ex)


# ---------------- tasks
if 'mmlu' in TASKS:
    dv = rd('mmlu_dev'); te = lim(rd('mmlu_test'))
    LET = ['A', 'B', 'C', 'D']
    fmt = lambda r: r['question'].strip() + '\n' + ''.join(f'{l}. {c}\n' for l, c in zip(LET, r['choices'])) + 'Answer:'
    shots = {}
    for r in dv: shots.setdefault(r['subject'], []).append(r)
    cand = [T_(' ' + l) for l in LET]; assert all(len(c) == 1 for c in cand); cand = [c[0] for c in cand]
    reqs = []
    for r in te:
        s = r['subject']
        ctx = f"The following are multiple choice questions (with answers) about {s.replace('_', ' ')}.\n\n"
        ctx += ''.join(fmt(d) + f" {LET[d['answer']]}\n\n" for d in shots[s][:5]) + fmt(r)
        reqs.append((T_(ctx), [], cand))
    t0 = _t.time(); R = score(reqs); dt = _t.time() - t0
    rows = []
    for r, x in zip(te, R):
        am = lambda v: max(range(4), key=lambda q: v[q])
        rows.append(dict(subject=r['subject'], gold=r['answer'], lp_h=x['lp_h'], lp_a=x['lp_a'], ok_h=int(am(x['lp_h']) == r['answer']),
                         ok_a=int(am(x['lp_a']) == r['answer']), fsc=x['fsc'], fall=x['fall'], npos=1, end=0))
    n = len(rows)
    dump('mmlu', rows, dict(n=n, acc_h=100 * sum(x['ok_h'] for x in rows) / n, acc_a=100 * sum(x['ok_a'] for x in rows) / n,
                            seq_fire=sum(1 for x in rows if x['fsc']) / n, seq_fire_anywhere=sum(1 for x in rows if x['fall']) / n,
                            pos_fire_anywhere=sum(x['fall'] for x in rows) / sum(len(q[0]) for q in reqs), sec=round(dt)))

if 'arcc' in TASKS:
    its = []
    for r in lim(rd('arcc_test')):
        lab = list(r['choices']['label']); its.append((f"Question: {r['question']}\nAnswer:", [' ' + c for c in r['choices']['text']], lab.index(r['answerKey'])))
    mc_task('arcc', its)

if 'hs' in TASKS:
    def _pp(t):
        t = t.strip().replace(' [title]', '. '); t = _re.sub('\\[.*?\\]', '', t); return t.replace('  ', ' ')
    its = []
    for r in lim(rd('hs_val')):
        ctx = r['ctx_a'] + ' ' + r['ctx_b'].capitalize()
        its.append((_pp(r['activity_label'] + ': ' + ctx), [' ' + _pp(e) for e in r['endings']], int(r['label'])))
    mc_task('hs', its)

if 'wg' in TASKS:
    # partial scoring: score the continuation after the blank under each option-filled context (lm-eval winogrande)
    reqs = []; meta = []
    for k, r in enumerate(lim(rd('wg_val'))):
        s = r['sentence']; i = s.index('_'); cont = ' ' + s[i + 1:].strip()
        for o in (r['option1'], r['option2']):
            reqs.append((T_(s[:i] + o), T_(cont), None))
        meta.append(int(r['answer']) - 1)
    t0 = _t.time(); R = score(reqs); dt = _t.time() - t0
    rows = []
    for k, g in enumerate(meta):
        r1, r2 = R[2 * k], R[2 * k + 1]
        lh = [r1['lp_h'], r2['lp_h']]; la = [r1['lp_a'], r2['lp_a']]
        rows.append(dict(i=k, gold=g, lp_h=lh, lp_a=la, ok_h=int((lh[1] > lh[0]) == (g == 1)), ok_a=int((la[1] > la[0]) == (g == 1)),
                         fsc=r1['fsc'] + r2['fsc'], fall=r1['fall'] + r2['fall'], npos=r1['npos'] + r2['npos'],
                         end=int(r1['end'] >= 0) + int(r2['end'] >= 0)))
    n = len(rows)
    dump('wg', rows, dict(n=n, acc_h=100 * sum(x['ok_h'] for x in rows) / n, acc_a=100 * sum(x['ok_a'] for x in rows) / n,
                          seq_fire=sum(1 for x in rows if x['fsc']) / n, pos_fire=sum(x['fsc'] for x in rows) / sum(x['npos'] for x in rows),
                          seq_fire_anywhere=sum(1 for x in rows if x['fall']) / n, end_items=sum(1 for x in rows if x['end']), sec=round(dt)))


# ---------------- generative: NQ-open, 5-shot, greedy, stop at '\n', EM (lm-eval nq_open normalisation)
def _norm(s):
    s = s.lower(); s = ''.join(ch for ch in s if ch not in set(_str.punctuation))
    s = _re.sub(r'\b(a|an|the)\b', ' ', s); return ' '.join(s.split())


@torch.no_grad()
def greedy(prompts, attach, max_new=24, bs=256):
    """left-padded greedy decode; attach: DualImplant write live at every step (window of the last K+W tokens, as
    Engine._prefill / step). Returns texts, implant fires on the produced answer's steps, serial END index."""
    NL_ = [i for i, t in enumerate(_txt) if '\n' in t]
    nlt = torch.zeros(V, dtype=torch.bool, device=dev); nlt[NL_] = True
    outs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for b0 in range(0, len(order), bs):
        b = order[b0:b0 + bs]; B = len(b); T = max(len(prompts[i]) for i in b)
        ids = torch.full((B, T), EOS, dtype=torch.long); am = torch.zeros((B, T), dtype=torch.long)
        for r, i in enumerate(b):
            ids[r, T - len(prompts[i]):] = torch.tensor(prompts[i]); am[r, T - len(prompts[i]):] = 1
        ids = ids.to(dev); am = am.to(dev); pid = (am.cumsum(-1) - 1).clamp(min=0)
        o = H.model(input_ids=ids, attention_mask=am, position_ids=pid, use_cache=True)
        pkv = o.past_key_values; x = _cap.pop('x')[:, -1]
        hist = ids; gen = []; fires = torch.zeros(B, dtype=torch.long, device=dev); stopped = torch.zeros(B, dtype=torch.bool, device=dev)
        for st in range(max_new):
            f, wr = implant_e(hist[:, -KWIN:])
            f = f[:, -1] & ~stopped; fires += f.long()
            xa = x + wr[:, -1].to(x.dtype) if (attach and wr is not None) else x
            nxt = _F.linear(NORM(xa), LMW).float().argmax(-1)
            nxt = torch.where(stopped, torch.full_like(nxt, EOS), nxt)
            gen.append(nxt); stopped |= nlt[nxt] | (nxt == EOS)
            if bool(stopped.all()): break
            hist = torch.cat([hist, nxt[:, None]], 1); am = torch.cat([am, torch.ones_like(am[:, :1])], 1); pid = pid[:, -1:] + 1
            o = H.model(input_ids=nxt[:, None], attention_mask=am, position_ids=pid, past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values; x = _cap.pop('x')[:, -1]
        G = torch.stack(gen, 1).tolist()
        for r, i in enumerate(b):
            g = G[r]; cut = len(g)
            for j, t in enumerate(g):
                if t == EOS or '\n' in _txt[t]: cut = j; break
            outs[i] = dict(txt=tok.decode(g[:cut]), fire=int(fires[r]), end=serial_end(g[:cut]), ntok=cut)
    return outs


if 'nq' in TASKS:
    tr = rd('nq_train')[:5]
    fs = ''.join(f"Q: {r['question']}?\nA: {r['answer'][0]}\n\n" for r in tr)
    te = lim(rd('nq_val'))
    prompts = [T_(fs + f"Q: {r['question']}?\nA:") for r in te]
    t0 = _t.time(); Oh = greedy(prompts, False); Oa = greedy(prompts, True); dt = _t.time() - t0
    rows = []
    for r, oh, oa in zip(te, Oh, Oa):
        gl = {_norm(x) for x in r['answer']}
        rows.append(dict(ok_h=int(_norm(oh['txt']) in gl), ok_a=int(_norm(oa['txt']) in gl), fsc=oa['fire'], fh=oh['fire'],
                         end=int(oa['end'] >= 0), same=int(oh['txt'] == oa['txt']), txt_h=oh['txt'], txt_a=oa['txt']))
    n = len(rows)
    dump('nq', rows, dict(n=n, acc_h=100 * sum(x['ok_h'] for x in rows) / n, acc_a=100 * sum(x['ok_a'] for x in rows) / n,
                          seq_fire=sum(1 for x in rows if x['fsc']) / n, end_items=sum(x['end'] for x in rows),
                          same_text=sum(x['same'] for x in rows) / n, sec=round(dt)))
print('OTDONE', flush=True)
