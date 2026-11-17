"""COMPILED-ONLY rule keepers: no_comma, lower (upper veto x2), quote (open + close). No Python rule objects, no regex,
no detect.py, no Python branches on token ids or on the host's argmax. Every prompt runs through the circuit (no subset).

Per token (incremental tensor ops, batched):
  reader (compiled.py): token embedding -> 25 symbol flags (linear reader, 0/1 lookup rows for 12 groups)
  stack1 (ReLU, K=16 window) -> pattern matches (no_comma/lower/caps/quote), role-header flags, first-answer-position, qclose
  latch heads (max over matches so far), turn head (role at most recent header)
  answer tracker (tensor cache): n = answer tokens so far; lnw_q = [most recent non-whitespace answer token rstrip ends with '"'];
     lnw_x = [that token is answer token 0 and its rstrip is exactly '"'] (tables NW/QE/QX over the vocabulary, as lookups)
  gates (ReLU on 0/1 wires): nc = AND(latch_nc, asst); lo = AND(latch_lo, asst); q = AND(latch_q, asst)
     open = AND(q, first); close = ReLU(q - first - ReLU(lnw_q - lnw_x))
     priority: force(open) suppresses vetoes and close:  W0 = open*F + (1-open)*(nc*Vc + lo*2Vu);  s = (1-open)*close
  end-of-turn readout (DISCLOSED second readout through the frozen LM head): logits of rmsnorm(h+W0); m = max EOS logit - max other;
     e = clamp(1e6*m, 0, 1);  W = (1-s*e)*W0 + s*e*F            (F = emit-'"' direction x gain)
python gen_c.py OUT.jsonl --keys even|odd [--bs 16] [--limit N] [--none 1]
"""
import argparse, json, time, os, pickle, torch
from common import load, chat, vocab_strings, EOS_IDS
from compiled import RuleKeeperCircuit, K, SYMS

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--keys', default='even'); ap.add_argument('--dirs', default='dirs.pt')
ap.add_argument('--bs', type=int, default=16); ap.add_argument('--max_new', type=int, default=2048)
ap.add_argument('--data', default='../data/ifeval.jsonl'); ap.add_argument('--gain', type=float, default=1e4)
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--keylist', default=''); ap.add_argument('--shard', default='')
a = ap.parse_args()

tok, model = load()
strs = vocab_strings(tok)
docs = [json.loads(l) for l in open(a.data)]
if a.keys != 'all': docs = [d for d in docs if (d['key'] % 2 == 0) == (a.keys == 'even')]
if a.keylist: ks = set(map(int, a.keylist.split(','))); docs = [d for d in docs if d['key'] in ks]
if a.limit: docs = docs[:a.limit]
if a.shard: si, sn = map(int, a.shard.split('/')); docs = [d for j, d in enumerate(docs) if j % sn == si]
dirs = torch.load(a.dirs)
g = model.model.norm.weight.float(); E = model.lm_head.weight; normw = model.model.norm.weight
Vd = E.shape[0]; d_model = E.shape[1]

# ---------------- fixed tensors (built offline from the frozen weights / vocabulary) ----------------
dq = E[1].float() / g; Fq = (dq * torch.rsqrt(dq.pow(2).mean())) * a.gain              # emit '"' direction (Phase-1 force)
Vc = dirs['veto']['comma'].cuda().float(); Vu2 = dirs['veto']['upper'].cuda().float() * 2.0   # upper veto x2 (frozen choice)
Ef = model.get_input_embeddings().weight.float()
C = RuleKeeperCircuit(tok, strs, Ef, torch.zeros(5, d_model, device='cuda'))
print('reader exact:', C.check_reader(Ef), 'lookup groups:', list(C.table), flush=True)
nw = torch.tensor([float(s.strip() != '') for s in strs], device='cuda')[:, None]
qe = torch.tensor([float(s.strip() != '' and s.rstrip().endswith('"')) for s in strs], device='cuda')[:, None]
qx = torch.tensor([float(s.rstrip() == '"') for s in strs], device='cuda')[:, None]
TAB = torch.cat([nw, qe, qx], 1)                                                        # (V, 3) lookup table
EOSMASK = torch.zeros(Vd, device='cuda'); EOSMASK[EOS_IDS] = 1.0                         # fixed 0/1 vector
NEG = -1e30


class Circuit:
    def reset(self, B):
        z = lambda *s: torch.zeros(*s, device='cuda')
        self.win = z(B, K, len(SYMS)); self.latch = z(B, 4); self.asst = z(B)
        self.n = z(B); self.lq = z(B); self.lx = z(B)

    @torch.no_grad()
    def step(self, emb, ids):
        s = C.flags(emb, ids); B = s.shape[0]
        self.win = torch.cat([s[:, None], self.win[:, :-1]], 1)
        o1 = C.s1(self.win.reshape(B, -1))
        m, hdr, hdr_a, first = o1[:, :4], o1[:, 4], o1[:, 5], o1[:, 6]
        self.latch = torch.maximum(self.latch, m)                                     # latch heads
        self.asst = hdr * hdr_a + (1 - hdr) * self.asst                                # turn head
        t = torch.nn.functional.embedding(ids, TAB); nwf, qef, qxf = t[:, 0], t[:, 1], t[:, 2]
        idx0 = torch.relu(1 - self.n)                                                  # [this answer token has index 0]
        lq = nwf * qef + (1 - nwf) * self.lq
        lx = nwf * qxf * idx0 + (1 - nwf) * self.lx
        self.lq = (1 - first) * lq; self.lx = (1 - first) * lx                          # reset at the answer start marker
        self.n = (1 - first) * (self.n + 1)
        ia = self.asst
        AND = lambda *xs: torch.relu(sum(xs) - (len(xs) - 1))
        nc, lo, q = AND(self.latch[:, 0], ia), AND(self.latch[:, 1], ia), AND(self.latch[:, 3], ia)
        op = AND(q, first)
        cl = torch.relu(q - first - torch.relu(self.lq - self.lx))
        W0 = op[:, None] * Fq + (1 - op)[:, None] * (nc[:, None] * Vc + lo[:, None] * Vu2)
        self.W0, self.s = W0, (1 - op) * cl
        self.fire = torch.stack([nc, lo, q, op, self.s], 1)


CIRC = Circuit()
pending = {'on': False}


def _hook(mod, x):
    if not pending['on']: return None                    # harness switch (on during generation); not a rule decision
    h = x[0]
    hl = (h[:, -1] + CIRC.W0.to(h.dtype)).float()
    lg = model.lm_head((hl * torch.rsqrt(hl.pow(2).mean(-1, keepdim=True) + 1e-6) * normw.float()).to(h.dtype)).float()
    me = (lg + NEG * (1 - EOSMASK)).max(-1).values; mo = (lg + NEG * EOSMASK).max(-1).values
    e = torch.clamp((me - mo) * 1e6, 0, 1)
    ge = (CIRC.s * e)[:, None]
    W = (1 - ge) * CIRC.W0 + ge * Fq
    return (h + W[:, None].to(h.dtype),)


hook = model.model.norm.register_forward_pre_hook(_hook)


@torch.no_grad()
def run(batch):
    P = [tok(chat(tok, d['prompt'], d.get('history')), add_special_tokens=False).input_ids for d in batch]
    B = len(P); L = max(map(len, P))
    x = torch.tensor([[EOS_IDS[1]] * (L - len(p)) + p for p in P], device='cuda')
    att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in P], device='cuda')
    ids_out = [[] for _ in range(B)]; done = [False] * B; nfire = torch.zeros(B, device='cuda'); fl = torch.zeros(B, 5, device='cuda')
    emb = model.get_input_embeddings()
    CIRC.reset(B)
    for t in range(L): CIRC.step(emb(x[:, t]).float(), x[:, t])
    pending['on'] = True
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    for step in range(a.max_new):
        nfire += (CIRC.fire[:, [0, 1, 3, 4]].sum(1) > 0).float(); fl = torch.maximum(fl, CIRC.fire)
        nxt = lg.argmax(-1); nl = nxt.tolist()
        for i in range(B):
            if done[i]: continue
            if nl[i] in EOS_IDS: done[i] = True
            else: ids_out[i].append(nl[i])
        if all(done): break
        nxt = torch.where(torch.tensor(done, device='cuda'), torch.full_like(nxt, EOS_IDS[1]), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        CIRC.step(emb(nxt).float(), nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        lg = model.lm_head(o.last_hidden_state[:, -1]).float()
    pending['on'] = False
    fl = fl.tolist(); nf = nfire.tolist()
    return [(d, tok.decode(ids_out[i]), done[i], len(ids_out[i]), [n for n, v in zip(['no_comma', 'lower', 'quote'], fl[i][:3]) if v > 0.5], nf[i])
            for i, d in enumerate(batch)]


order = sorted(range(len(docs)), key=lambda i: len(docs[i]['prompt']))


def run_safe(batch, b, tries=0):
    try:
        return run(batch)
    except torch.OutOfMemoryError:
        pending['on'] = False; torch.cuda.empty_cache(); print('OOM at batch', b, 'size', len(batch), flush=True)
        if len(batch) > 4 and tries >= 1:
            h = len(batch) // 2
            return run_safe(batch[:h], b) + run_safe(batch[h:], b + h)
        time.sleep(30); return run_safe(batch, b, tries + 1)


t0 = time.time(); out = []; part = a.out + '.partial'
if os.path.exists(part): out = pickle.load(open(part, 'rb')); print('resumed', len(out), flush=True)
for b in range(0, len(order), a.bs):
    if b < len(out): continue
    out += run_safe([docs[i] for i in order[b:b + a.bs]], b); pickle.dump(out, open(part, 'wb'))
    print(f'{len(out)}/{len(docs)} {time.time() - t0:.0f}s', flush=True)
with open(a.out, 'w') as f:
    for d, resp, fin, n, act, nf in sorted(out, key=lambda r: r[0]['key']):
        f.write(json.dumps({'key': d['key'], 'prompt': d['prompt'], 'history': d.get('history'), 'response': resp,
                            'finish': 'stop' if fin else 'length', 'ntok': n, 'active': act, 'nfire': nf}) + '\n')
print('done', time.time() - t0)
os.remove(part)
