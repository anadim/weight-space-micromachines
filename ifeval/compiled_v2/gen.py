"""HF greedy generation of IFEval answers with the Phase-1 reference rule-keepers attached through the micromachine
interface (an additive vector on the last hidden state before the host's final norm; exactly zero when silent).

python gen.py OUT.jsonl --keys even|odd|all --rules none|all|no_comma,lower,... [--dirs dirs.pt] [--bs 48] [--data ../data/ifeval.jsonl]
"""
import argparse, json, re, time, torch
from common import load, chat, vocab_strings, EOS_IDS
from detect import detect
from rules import build, combine
R_ALL = ['repeat', 'quote', 'no_comma', 'lower', 'caps', 'forbid', 'two', 'paras', 'nth', 'words', 'sents_less', 'sents_atleast']

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--keys', default='even'); ap.add_argument('--rules', default='all')
ap.add_argument('--dirs', default='dirs.pt'); ap.add_argument('--bs', type=int, default=48); ap.add_argument('--max_new', type=int, default=2048)
ap.add_argument('--data', default='../data/ifeval.jsonl'); ap.add_argument('--beta', type=float, default=20.0)
ap.add_argument('--gain', type=float, default=1e4); ap.add_argument('--subset', type=int, default=1); ap.add_argument('--kscale', type=float, default=1.0); ap.add_argument('--compiled', type=int, default=0); ap.add_argument('--vscale', default=''); ap.add_argument('--kgate', type=int, default=1); ap.add_argument('--shard', default=''); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
enabled = 'all' if a.rules == 'all' else set(a.rules.split(',')) if a.rules != 'none' else set()

tok, model = load()
strs = vocab_strings(tok); lstrs = [s.lower() for s in strs]
docs = [json.loads(l) for l in open(a.data)]
if a.keys != 'all': docs = [d for d in docs if (d['key'] % 2 == 0) == (a.keys == 'even')]
if a.limit: docs = docs[:a.limit]
if a.shard: si, sn = map(int, a.shard.split('/')); docs = [d for j, d in enumerate(docs) if j % sn == si]
if a.subset and enabled != 'all' and enabled:          # only prompts where an enabled rule is detected (others are bit-identical: zero write)
    docs = [d for d in docs if set(detect(d['prompt'])) & enabled]
dirs = torch.load(a.dirs) if enabled else {'veto': {}, 'swap': {}}
g = model.model.norm.weight.float(); E = model.lm_head.weight                      # (bf16; rows are cast on use)
FIX = {t: d.cuda() for t, d in (dirs.get('force_fix') or {}).items()}               # tokens where E_t/g does not win (solved)


def Dn(t):                                                                          # rms-1 'emit t' direction
    if t in FIX: return FIX[t]
    d = E[t].float() / g; return d * torch.rsqrt(d.pow(2).mean())


KM = dirs.get('kill_M'); KM = KM.cuda() if KM is not None else None
VETO = {k: v.cuda() for k, v in dirs['veto'].items()}
for kv in filter(None, a.vscale.split(',')): k, x = kv.split('='); VETO[k] = VETO[k] * float(x)          # veto strength
SWAP = {k: v.cuda() for k, v in dirs["swap"].items()}
d_model = E.shape[1]
CIRC = None
COMPILED_RULES = {'no_comma', 'lower', 'caps', 'quote'}
if a.compiled:            # Phase 2: these four rules come from the compiled circuit, not from Python rule objects
    from compiled import RuleKeeperCircuit
    Ef = model.get_input_embeddings().weight.float()
    V = torch.stack([VETO['comma'], VETO['upper'], VETO['lower'], Dn(1) * a.gain, torch.zeros_like(VETO['comma'])])
    CIRC = RuleKeeperCircuit(tok, strs, Ef, V)
    print('circuit reader exact:', CIRC.check_reader(Ef), 'lookup groups:', list(CIRC.table), flush=True)

pending = {'v': None, 'swap': [], 'kill': []}
SWAP_TOK = {'eos>quote': 1, 'eos>nn': 271, 'eos>star6': 6824}
normw = model.model.norm.weight


def _hook(m, x):
    if pending['v'] is None: return None
    h = x[0]; W = pending['v'].clone()
    for i, t in pending['swap']:
        # end-of-turn probe on the host's own state at this position: if the host (with the other writes) would end its
        # turn here, the write becomes 'emit t'. (One readout unit: is <|im_end|> the top logit?)
        hl = (h[i, -1:] + W[i].to(h.dtype)).float()
        lg = model.lm_head((hl * torch.rsqrt(hl.pow(2).mean(-1, keepdim=True) + 1e-6) * normw.float()).to(h.dtype)).float()
        if lg.argmax(-1).item() in EOS_IDS: W[i, 0] = Dn(t) * a.gain
    for i, ks in pending['kill']:
        # gated kill: if the host's top token (with the other writes) would start/complete a forbidden word, add the kill
        # write M e_t for that token; up to 4 rounds (the next choice may be another variant of the word)
        for _ in range(4):
            hl = (h[i, -1:] + W[i].to(h.dtype)).float()
            top = model.lm_head((hl * torch.rsqrt(hl.pow(2).mean(-1, keepdim=True) + 1e-6) * normw.float()).to(h.dtype)).float().argmax(-1).item()
            if top not in ks: break
            W[i, 0] = W[i, 0] + a.kscale * (E[top].float() @ KM.T)
    return (h + W.to(h.dtype),)


hook = model.model.norm.register_forward_pre_hook(_hook)


class St:
    def __init__(self): self.ids, self.text, self.n = [], '', 0
    def dec(self, ids): return tok.decode(ids)
    def push(self, t): self.ids.append(t); self.text += strs[t]; self.n += 1


def user_text(ids):
    """The instructions are read from the prompt TOKENS: every user turn of the chat template (multi-turn: all of them)."""
    s = tok.decode(ids)
    return '\n'.join(p.split('<|im_end|>', 1)[0] for p in s.split('<|im_start|>user\n')[1:])


def write_vec(rules, st):
    if not rules: return None, None
    force, veto, swap, kill = combine(rules, st)
    if force is not None: return Dn(force) * a.gain, ('force', force)
    if not veto and not swap and not kill: return None, None
    if swap: swap = {min(swap, key=['eos>star6', 'eos>nn', 'eos>quote'].index)}
    v = torch.zeros(d_model, device='cuda')
    for k in veto: v += VETO[k]
    if kill and not a.kgate:
        kk = torch.tensor(sorted(set(kill)), device='cuda')
        v += a.kscale * (E[kk].float() @ KM.T).sum(0)                       # solved kill map: write = M e_t per killed token
    return v, ('mix', tuple(sorted(veto)), set(kill) if a.kgate else len(kill), SWAP_TOK[next(iter(swap))] if swap else None)


@torch.no_grad()
def run(batch):
    P = [tok(chat(tok, d['prompt'], d.get('history')), add_special_tokens=False).input_ids for d in batch]
    R = [detect(user_text(p)) for p in P]
    ref_enabled = enabled if not a.compiled else ({k for k in (enabled if enabled != 'all' else set(R_ALL)) if k not in COMPILED_RULES})
    rules = [build(r, tok, strs, lstrs, ref_enabled) for r in R]
    B = len(P); L = max(map(len, P))
    x = torch.tensor([[EOS_IDS[1]] * (L - len(p)) + p for p in P], device='cuda')
    att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in P], device='cuda')
    sts = [St() for _ in range(B)]; done = [False] * B; nfire = [0] * B

    cw = {'w': None, 'f': None}
    if CIRC is not None:                      # run the circuit over the prompt tokens (incremental form, exact)
        CIRC.reset(); emb = model.get_input_embeddings()
        for t in range(x.shape[1]): cw['w'], cw['f'] = CIRC.step(emb(x[:, t]).float(), x[:, t])
        on = [k for k in COMPILED_RULES if enabled == 'all' or k in enabled]
        mask = torch.tensor([['no_comma' in on, 'lower' in on, 'caps' in on, 'quote' in on, 'quote' in on]], device='cuda').float()

    def set_pending():
        W = torch.zeros(B, 1, d_model, device='cuda'); any_ = False; sw = []; kl = []
        for i in range(B):
            if done[i]: continue
            v, info = write_vec(rules[i], sts[i])
            forced = v is not None and info[0] == 'force'
            if CIRC is not None:
                f = cw['f'][i] * mask[0]
                if f.abs().sum() > 0 and not forced:
                    cv = f @ CIRC.V
                    if f[3] > 0.5: v, info = cv, ('force', 1)            # circuit's emit '"' (other writes dropped, as in the reference)
                    else:
                        v = cv if v is None else v + cv
                        if f[4] > 0.5 and (info is None or (info[0] == 'mix' and info[3] is None and 'eos' not in info[1])): info = ('mix', info[1] if info else (), info[2] if info else 0, 1)
                        elif info is None: info = ('mix', (), 0, None)
            if v is not None:
                W[i, 0] = v; any_ = True; nfire[i] += 1
                if info[0] == 'mix' and info[3] is not None: sw.append((i, info[3]))
                if info[0] == 'mix' and isinstance(info[2], set) and info[2]: kl.append((i, info[2]))
        pending['v'] = W if any_ else None; pending['swap'] = sw; pending['kill'] = kl

    set_pending()
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    # NOTE: model.model(...) already applies the final norm (with our hook) -> last_hidden_state is post-norm
    lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    for step in range(a.max_new):
        nxt = lg.argmax(-1); nl = nxt.tolist()
        for i in range(B):
            if done[i]: continue
            if nl[i] in EOS_IDS: done[i] = True
            else: sts[i].push(nl[i])
        if all(done): break
        nxt = torch.where(torch.tensor(done, device='cuda'), torch.full_like(nxt, EOS_IDS[1]), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        if CIRC is not None: cw['w'], cw['f'] = CIRC.step(model.get_input_embeddings()(nxt).float(), nxt)
        set_pending()
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    pending['v'] = None
    return [(d, tok.decode(s.ids), done[i], len(s.ids), R[i], [r.name for r in rules[i]] + ([n for j, n in enumerate(['no_comma', 'lower', 'caps', 'quote']) if CIRC is not None and CIRC.state['latch'][i, j] > 0.5 and (enabled == 'all' or n in enabled)]), nfire[i]) for i, (d, s) in enumerate(zip(batch, sts))]


order = sorted(range(len(docs)), key=lambda i: len(docs[i]['prompt']))
import os, pickle


def run_safe(batch, b, tries=0):
    """Shared GPU: on OOM split the batch in halves (outputs can then differ by bf16 batch numerics); size 1 -> wait, retry."""
    try:
        return run(batch)
    except torch.OutOfMemoryError:
        pending['v'] = None; torch.cuda.empty_cache(); print('OOM at batch', b, 'size', len(batch), flush=True)
        if len(batch) > 4 and tries >= 1:
            h = len(batch) // 2
            return run_safe(batch[:h], b) + run_safe(batch[h:], b + h)
        time.sleep(30); return run_safe(batch, b, tries + 1)
t0 = time.time(); out = []; part = a.out + '.partial'
if os.path.exists(part):                                  # resume: whole batches already done (same order, same batch size)
    out = pickle.load(open(part, 'rb')); print('resumed', len(out), flush=True)
for b in range(0, len(order), a.bs):
    if b < len(out): continue
    batch = [docs[i] for i in order[b:b + a.bs]]
    res = run_safe(batch, b)
    out += res; pickle.dump(out, open(part, 'wb'))
    print(f'{len(out)}/{len(docs)} {time.time() - t0:.0f}s', flush=True)
with open(a.out, 'w') as f:
    for d, resp, fin, n, R, names, nf in sorted(out, key=lambda r: r[0]['key']):
        f.write(json.dumps({'key': d['key'], 'prompt': d['prompt'], 'history': d.get('history'), 'response': resp, 'finish': 'stop' if fin else 'length', 'ntok': n,
                            'detected': {k: v for k, v in R.items() if k != 'repeat'} | ({'repeat': 1} if 'repeat' in R else {}),
                            'active': names, 'nfire': nf}) + '\n')
print('done', time.time() - t0)
os.remove(part)
