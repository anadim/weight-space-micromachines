"""Causal latent write with the COMPILED trigger (ctrig.py tables in torch registers) instead of Python positions.
Modes:
  prompt : toy muladd mid, query '(a*b)+c?\\nA: '. Map Wt_all_ct_2k.pt (cw.py train all:ct, 97.3% Python-positioned).
           arms: gated (nsep>=4 = the old query-only positions) | ungated drop-in (fires in the 4 shots too) | ungated retrained
  gen    : generated 'A: (a*b)+c = ANS'. Map GEN_gall.pt (cw.py gen CW_GLOSS=all, 96.7% Python-positioned).
           arms: gated (colon + nsep>=4: only the generated expression) | colon-only drop-in (shot answers too) |
                 fully ungated drop-in (shot questions + answers + query + generated) | fully ungated retrained
Every write decision is the trigger's: the embedding pre-hook runs the table scan (prefill: token by token over the prompt;
decode: one step per generated token, registers carried), layer pre-hooks add onehot(slot) (x) feats(P) @ W_L at fired rows.
Also fuzzes the torch scan against ctrig.ref_find on every eval prompt / training sequence / generated stream.
"""
import os, sys, json, time, random, re
import torch, torch.nn.functional as Fn
import lwv1
from lwv1 import load, setup, digit_feats, parse_int, DEV
import ctrig as CT

MODE = sys.argv[1]
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
tok, model = load(); S = setup("muladd", "mid"); N = len(S["test"]); D = model.config.hidden_size; NL = model.config.num_hidden_layers
OUT = os.path.join(os.environ.get("LW_OUT", "out"), f"ct_{MODE}{os.environ.get('CW_TAG', '')}.json")
STEPS = int(os.environ.get("CW_STEPS", "2000")); TBS = 32; LR = float(os.environ.get("CW_LR", "3e-3")); BS = 50
V = model.get_input_embeddings().weight.shape[0]
txt = [tok.decode([i]) for i in range(len(tok))] + [""] * max(0, V - len(tok))


class TTrig:
    """torch registers + fixed tables (same tables as ctrig.Trig)."""
    def __init__(self, colon, tail, nsep):
        TC, ROP, FSLOT = CT.build(colon, tail); CLS, DIG = CT.vocab_tables(txt)
        self.TC = torch.tensor(TC, device=DEV); self.ROP = torch.tensor(ROP, device=DEV); self.FSLOT = torch.tensor(FSLOT, device=DEV)
        self.CLS = torch.tensor(CLS, device=DEV); self.DIG = torch.tensor(DIG, device=DEV); self.nsep = nsep
        self.cfg = dict(colon=colon, tail=tail, nsep=nsep); self.ref = CT.Trig(txt, **self.cfg)

    def reset(self, B):
        z = lambda: torch.zeros(B, dtype=torch.long, device=DEV)
        self.s, self.ra, self.rb, self.P, self.n = z(), z(), z(), z(), z()

    def step(self, t):
        c = self.CLS[t]; d = self.DIG[t]
        s = self.TC[c, self.s]; op = self.ROP[s]
        self.ra = torch.where(op == 1, d, torch.where(op == 2, 10 * self.ra + d, self.ra))
        self.rb = torch.where(op == 3, d, torch.where(op == 4, 10 * self.rb + d, self.rb))
        Pm = self.ra * self.rb; ok = (Pm >= 100) & (Pm <= 999)          # fixed product / validity tables
        close = s == CT.CLOSE
        self.P = torch.where(close & ok, Pm, self.P)
        s = torch.where(close & ~ok, torch.zeros_like(s), s)
        self.n = torch.clamp(self.n + (c == CT.K_SEP).long(), max=max(self.nsep, 1)); self.s = s
        slot = self.FSLOT[s]
        slot = torch.where(self.n < self.nsep, torch.full_like(slot, -1), slot)
        return slot, torch.where(slot >= 0, self.P, torch.zeros_like(self.P))

    def scan(self, ids, reset=True):
        if reset: self.reset(ids.shape[0])
        sl, Ps = zip(*[self.step(ids[:, j]) for j in range(ids.shape[1])])
        return torch.stack(sl, 1), torch.stack(Ps, 1)


FZ = {"seqs": 0, "fires": 0, "bad": 0}
def fuzz_check(trig, ids, slot, P, skip_pad=True):
    for b in range(ids.shape[0]):
        row = ids[b].tolist(); sl = slot[b].tolist(); pp = P[b].tolist()
        got = [(p, s, q) for p, (s, q) in enumerate(zip(sl, pp)) if s >= 0]
        want = CT.ref_find([txt[i] for i in row], **trig.cfg)
        FZ["seqs"] += 1; FZ["fires"] += len(want); FZ["bad"] += got != want


def feats_t(P):   # P (...,) long -> (..., 31): ones + 3 digit one-hots (= cw.feats)
    d = torch.stack([P // 100 % 10, P // 10 % 10, P % 10], -1)
    return torch.cat([torch.ones(*P.shape, 1, device=P.device), Fn.one_hot(d, 10).float().flatten(-2)], -1)


class Writer:
    """trigger-driven additive write. W: {L: (nslot, 31, D)}. emb pre-hook runs the trigger; layer pre-hooks add."""
    def __init__(self, trig, W, corr=None, check=False):
        self.trig, self.W, self.corr, self.check = trig, W, corr, check; self.G = None; self.hs = []; self.prefill_ids = None
    def _emb(self, mod, args, kwargs):
        x = args[0] if args else kwargs["input_ids"]
        if x.shape[1] > 1:
            slot, P = self.trig.scan(x)
            if self.check: fuzz_check(self.trig, x, slot, P)
            self.hist = [x]
        else:
            slot, P = self.trig.step(x[:, 0]); slot, P = slot[:, None], P[:, None]; self.hist.append(x)
        if self.corr is not None: P = torch.where(slot >= 0, self.corr[:, None].expand_as(P), P)
        ns = next(iter(self.W.values())).shape[0]
        on = (slot >= 0)
        G = Fn.one_hot(slot.clamp(min=0), ns).float()[..., :, None] * feats_t(P)[..., None, :]    # (B,T,ns,31)
        self.G = (G * on[..., None, None]).flatten(-2) if on.any() else None
    def _layer(self, L):
        def pre(mod, args, kwargs):
            if self.G is None: return None
            h = args[0] if args else kwargs["hidden_states"]
            h2 = h + (self.G @ self.W[L].reshape(-1, D)).to(h.dtype)
            if args: return (h2,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = h2; return args, kwargs
        return pre
    def __enter__(self):
        self.hs = [model.model.embed_tokens.register_forward_pre_hook(self._emb, with_kwargs=True)]
        self.hs += [model.model.layers[L].register_forward_pre_hook(self._layer(L), with_kwargs=True) for L in self.W]
        return self
    def __exit__(self, *a):
        for h in self.hs: h.remove()
        self.G = None
    def full_ids(self): return torch.cat(self.hist, 1)


def enc_batch(ps): return tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)


@torch.no_grad()
def run(prompts, trig, W, corr=None, max_new=6, check_gen=False):
    outs = []
    for i0 in range(0, len(prompts), BS):
        enc = enc_batch(prompts[i0:i0 + BS]); cr = None if corr is None else torch.tensor(corr[i0:i0 + BS], device=DEV)
        if W is None: g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        else:
            with Writer(trig, W, cr, check=True) as wr:
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
                if check_gen:   # the decode-step trigger outputs over the full stream must equal the reference too
                    full = g[:, :-1]                                   # the last sampled token is never fed back
                    sl, Ps = trig.scan(full); fuzz_check(trig, full, sl, Ps)
                    GEN["streams"].append(full.cpu())
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs
GEN = {"streams": []}


def score_prompt(outs):
    p = [parse_int(o) for o in outs]
    return {"acc": round(100 * sum(x == y for x, y in zip(p, S["ans_true"])) / N, 1),
            "follow": round(100 * sum(x == y for x, y in zip(p, S["ans_corr"])) / N, 1)}


def score_gen(outs):
    p = []; cp = 0
    for o, (a, b, c) in zip(outs, S["test"]):
        m = re.search(r"=\s*(-?\d+)", o); p.append(int(m.group(1)) if m else None); cp += o.strip().startswith(f"({a}*{b})+{c}")
    return {"acc": round(100 * sum(x == y for x, y in zip(p, S["ans_true"])) / N, 1),
            "follow": round(100 * sum(x == y for x, y in zip(p, S["ans_corr"])) / N, 1), "copy_ok": round(100 * cp / N, 1)}


def train(trig, ns, mk_ids, loss_span):
    test_set = set(S["test"]); r = random.Random(7); tr = []
    while len(tr) < 6000:
        a, b, c = r.randint(11, 29), r.randint(3, 9), r.randint(10, 99)
        if a * b >= 100 and (a, b, c) not in test_set: tr.append((a, b, c))
    Wt = {L: torch.zeros(ns, 31, D, device=DEV, requires_grad=True) for L in range(NL)}
    opt = torch.optim.Adam(list(Wt.values()), lr=LR)
    for step in range(STEPS):
        bi = [tr[(step * TBS + j) % len(tr)] for j in range(TBS)]
        x = mk_ids(bi)
        with Writer(trig, Wt, check=(step < 3)):
            logits = model(input_ids=x).logits
        lo, hi = loss_span
        lg = logits[:, lo - 1:hi - 1].float(); tgt = x[:, lo:hi]
        loss = Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == STEPS - 1: log("train step", step, "loss", round(loss.item(), 4))
    return {L: w.detach() for L, w in Wt.items()}, round(loss.item(), 4)


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): res["fuzz"] = dict(FZ); json.dump(res, open(OUT, "w"), indent=1)

if MODE == "prompt":
    Wpy = {L: w.to(DEV).float() for L, w in torch.load("out/Wt_all_ct_2k.pt").items()}
    ids0 = tok(S["P_task"][0], add_special_tokens=False).input_ids; LT = len(ids0)
    g = TTrig(colon=False, tail=True, nsep=4); u = TTrig(colon=False, tail=True, nsep=0)
    # exactness vs the OLD Python positions (query-only, from the end): gated trigger must fire exactly there
    for b, tg in [("gated", g), ("ungated", u)]:
        sl, Ps = tg.scan(torch.tensor([ids0], device=DEV)); f = [(LT - p, int(sl[0, p])) for p in range(LT) if sl[0, p] >= 0]
        res[f"positions_{b}"] = f; log(b, "fires (fe, slot):", f)
    assert sorted(res["positions_gated"]) == sorted([(7, 0), (4, 4), (3, 3), (2, 2), (1, 1)])
    res["plain"] = score_prompt(run(S["P_task"], g, None))
    for nm, tg in [("gated", g), ("ungated", u)]:
        res[f"{nm}_pymap"] = score_prompt(run(S["P_task"], tg, Wpy)); res[f"{nm}_pymap_corr"] = score_prompt(run(S["P_task"], tg, Wpy, corr=S["corr"]))
        log(nm, "py-trained map", res[f"{nm}_pymap"], "corr", res[f"{nm}_pymap_corr"]); save()
    prefix = S["prefix"]
    def mk(items):
        x = torch.tensor([tok(prefix + f"Q: What is ({a}*{b})+{c}?\nA: {a*b+c}", add_special_tokens=False).input_ids for a, b, c in items], device=DEV)
        assert x.shape[1] == LT + 3; return x
    Wu, ls = train(u, 5, mk, (LT, LT + 3))
    res["ungated_retrained"] = score_prompt(run(S["P_task"], u, Wu)); res["ungated_retrained_corr"] = score_prompt(run(S["P_task"], u, Wu, corr=S["corr"]))
    res["ungated_retrained_loss"] = ls; log("ungated retrained", res["ungated_retrained"], "corr", res["ungated_retrained_corr"])
    torch.save({L: w.cpu() for L, w in Wu.items()}, "out/CT_prompt_ungated.pt"); save()

elif MODE == "gen":
    Wpy = {L: w.to(DEV).float()[None] for L, w in torch.load("out/GEN_gall.pt").items()}    # (1, 31, D): slot 0 = ')+'
    rs = random.Random(1234); shots = []
    while len(shots) < 4:
        a, b, c = rs.randint(11, 29), rs.randint(3, 9), rs.randint(10, 99)
        shots.append(f"Q: What is ({a}*{b})+{c}?\nA: ({a}*{b})+{c} = {a*b+c}\n\n")
    gpre = "".join(shots)
    GP = [gpre + f"Q: What is ({a}*{b})+{c}?\nA:" for a, b, c in S["test"]]
    g = TTrig(colon=True, tail=False, nsep=4); cl = TTrig(colon=True, tail=False, nsep=0); u = TTrig(colon=False, tail=False, nsep=0)
    res["gen_plain"] = score_gen(run(GP, g, None, max_new=16))
    for nm, tg in [("gated", g), ("colon", cl), ("ungated", u)]:
        res[f"{nm}_pymap"] = score_gen(run(GP, tg, Wpy, max_new=16, check_gen=True))
        res[f"{nm}_pymap_corr"] = score_gen(run(GP, tg, Wpy, corr=S["corr"], max_new=16))
        log(nm, "py-trained map", res[f"{nm}_pymap"], "corr", res[f"{nm}_pymap_corr"]); save()
    # how the gated compiled trigger compares with the OLD Python decode-step finder on the real generated streams
    agree = tot = 0
    for full in GEN["streams"][:6]:
        plen = len(tok(GP[0], add_special_tokens=False).input_ids)
        for row in full.tolist():
            gtoks = row[plen:]; s = tok.decode(gtoks)
            py = [p for p in range(len(gtoks)) if txt[gtoks[p]] == ")+"]       # old hook: fires at generated ')+' if parse ok
            def pp(p):
                s_ = tok.decode(gtoks[:p + 1]); j = s_.rfind("(")
                try: a_, b_ = s_[j + 1:].split(")")[0].split("*"); return 100 <= int(a_) * int(b_) <= 999
                except Exception: return False
            py = [plen + p for p in py if pp(p)]
            cp = [p for p, s__, _ in CT.ref_find([txt[i] for i in row], **g.cfg)]
            tot += 1; agree += py == cp
    res["gated_vs_old_python_finder"] = f"{agree}/{tot} streams identical fire positions"; log(res["gated_vs_old_python_finder"])
    full0 = tok(GP[0] + f" ({S['test'][0][0]}*{S['test'][0][1]})+{S['test'][0][2]} = {S['ans_true'][0]}", add_special_tokens=False).input_ids
    GLT = len(full0); GCL = max(j for j, i in enumerate(full0) if txt[i] == ")+")
    def mk(items):
        x = torch.tensor([tok(gpre + f"Q: What is ({a}*{b})+{c}?\nA: ({a}*{b})+{c} = {a*b+c}", add_special_tokens=False).input_ids for a, b, c in items], device=DEV)
        assert x.shape[1] == GLT; return x
    Wu, ls = train(u, 1, mk, (GCL + 1, GLT))
    res["ungated_retrained"] = score_gen(run(GP, u, Wu, max_new=16, check_gen=True))
    res["ungated_retrained_corr"] = score_gen(run(GP, u, Wu, corr=S["corr"], max_new=16))
    res["ungated_retrained_loss"] = ls; log("ungated retrained", res["ungated_retrained"], "corr", res["ungated_retrained_corr"])
    torch.save({L: w.cpu() for L, w in Wu.items()}, "out/CT_gen_ungated.pt"); save()
log("fuzz", FZ); save(); log("done")
