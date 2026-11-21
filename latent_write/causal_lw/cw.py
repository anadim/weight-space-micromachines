"""Causal latent write, muladd mid (Qwen3-0.6B-Base). Writes ONLY at the closing token ')+' of the query span and/or at
the tail positions after c ('?\nA: '), i.e. at positions where (a, b, c) are all already behind (forward-only).
Python REFERENCE write (positions from the end of the left-padded prompt; query span only) - compile later if it works.

usage: python cw.py ridge [layers]        # V1: ridge REPLACE writes fitted on oracle '(P)+c' prompts
       python cw.py train [configs]       # V2: trained linear side map P-digits -> additive delta (CE on answer digits)
"""
import os, sys, json, time, random
import torch, torch.nn.functional as Fn
import lwv1
from lwv1 import load, setup, digit_feats, parse_int, collect, ridge, DEV

MODE = sys.argv[1]
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
tok, model = load(); S = setup("muladd", "mid"); N = len(S["test"]); D = model.config.hidden_size; NL = model.config.num_hidden_layers
OUT = os.path.join(os.environ.get("LW_OUT", "out"), f"cw_{MODE}{os.environ.get('CW_TAG', '')}.json"); os.makedirs(os.path.dirname(OUT), exist_ok=True)
BS = 50

ids = tok(S["P_task"][0], add_special_tokens=False).input_ids; toks = [tok.decode([i]) for i in ids]; LT = len(ids)
cj = max(j for j, t in enumerate(toks) if t == ")+"); k = cj + 1
while toks[k].isdigit(): k += 1
FE_CLOSE = LT - cj; FE_TAIL = list(range(1, LT - k + 1))
otoks = [tok.decode([i]) for i in tok(S["P_or_true"][0], add_special_tokens=False).input_ids]
assert otoks[-FE_CLOSE] == ")+" and otoks[-len(FE_TAIL):] == toks[-len(FE_TAIL):]
assert all(len(tok(p, add_special_tokens=False).input_ids) == LT for p in S["P_task"])
log("query tail tokens", toks[cj:], "FE_CLOSE", FE_CLOSE, "FE_TAIL", FE_TAIL)

Ptrue = [a * b for a, b, c in S["test"]]; Cs = [c for a, b, c in S["test"]]


def feats(P, C=None, mean=False):
    f = digit_feats(P, 3)
    if mean: f = f * 0 + 0.1
    x = [torch.ones(len(P), 1), f] + ([digit_feats(C, 2)] if C is not None else [])
    return torch.cat(x, 1)


class Writer:
    """pre-hook on block L: prefill only. 'rep': h[:, -fe] = X @ W ; 'add': h[:, -fe] += X @ W."""
    def __init__(self): self.hs = []
    def attach(self, L, items, kind, train=False):
        # items: list of (fe, X[B,F] (device), W[F,D])
        def pre(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            if h.shape[1] == 1: return None
            T = h.shape[1]; h2 = h.clone()
            for fe, X, W in items:
                p = T - fe if not train else LT - fe
                v = (X @ W).to(h.dtype)
                h2[:, p] = v if kind == "rep" else h2[:, p] + v
            if args: return (h2,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = h2; return args, kwargs
        self.hs.append(model.model.layers[L].register_forward_pre_hook(pre, with_kwargs=True))
    def clear(self):
        for h in self.hs: h.remove()
        self.hs = []


W = Writer()
def enc_batch(ps): return tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)


@torch.no_grad()
def gen(prompts, plan=None):
    """plan(i0, i1) -> list of (L, items, kind) to attach for rows i0:i1"""
    outs = []
    for i0 in range(0, len(prompts), BS):
        enc = enc_batch(prompts[i0:i0 + BS])
        if plan:
            for L, items, kind in plan(i0, i0 + BS): W.attach(L, items, kind)
        try: g = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        finally: W.clear()
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def score(outs):
    p = [parse_int(o) for o in outs]
    return {"acc": round(100 * sum(x == y for x, y in zip(p, S["ans_true"])) / N, 1),
            "follow": round(100 * sum(x == y for x, y in zip(p, S["ans_corr"])) / N, 1)}


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, "w"), indent=1)
if "plain" not in res:
    for name, ps in [("plain", "P_task"), ("text_oracle", "P_or_true"), ("text_tool_line", "P_tool")]:
        res[name] = score(gen(S[ps])); log(name, res[name])
    save()

if MODE == "ridge":
    LAYS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,4,8,10,12,14,16,18,20,22,24").split(",")]
    fP = [v for v, _ in S["fitv"]]; fC = [c for _, c in S["fitv"]]
    Ho = collect(S["P_fit"], LAYS, [FE_CLOSE] + FE_TAIL)
    Xc, Xt = feats(fP), feats(fP, fC)
    ENC = {l: [ridge(Xc, Ho[l][:, 0])] + [ridge(Xt, Ho[l][:, 1 + i]) for i in range(len(FE_TAIL))] for l in LAYS}
    del Ho
    # fit quality: R^2 on the fit set itself is not informative; report held-out on the first 300 test oracle prompts
    for L in LAYS:
        E = ENC[L]
        def plan_for(arm):
            def plan(i0, i1):
                P = Ptrue[i0:i1]; C = Cs[i0:i1]
                if arm.endswith("_corr"): P = S["corr"][i0:i1]
                mean = arm.endswith("_mean")
                xc = feats(P, mean=mean).to(DEV); xt = feats(P, C, mean=mean).to(DEV)
                items = []
                if arm.startswith(("close", "ct")): items.append((FE_CLOSE, xc, E[0]))
                if arm.startswith(("tail", "ct")): items += [(fe, xt, E[1 + i]) for i, fe in enumerate(FE_TAIL)]
                return [(L, items, "rep")]
            return plan
        for arm in ["close", "tail", "ct", "tail_mean", "tail_corr", "ct_corr"]:
            res[f"L{L}_{arm}"] = score(gen(S["P_task"], plan_for(arm)))
        log(f"L{L}", {a: res[f'L{L}_{a}'] for a in ["close", "tail", "ct", "tail_mean", "tail_corr", "ct_corr"]})
        save()

elif MODE == "train":
    CFGS = (sys.argv[2] if len(sys.argv) > 2 else "8:ct,16:ct,all:close,all:ct,all:tail").split(",")
    STEPS = int(os.environ.get("CW_STEPS", "300")); TBS = 32; LR = float(os.environ.get("CW_LR", "3e-3"))
    test_set = set(S["test"]); r = random.Random(7); tr = []
    while len(tr) < 6000:
        a, b, c = r.randint(11, 29), r.randint(3, 9), r.randint(10, 99)
        if a * b >= 100 and (a, b, c) not in test_set: tr.append((a, b, c))
    prefix = S["prefix"]
    def tr_ids(items):
        P = [prefix + f"Q: What is ({a}*{b})+{c}?\nA: " for a, b, c in items]
        full = [tok(p + str(a * b + c), add_special_tokens=False).input_ids for p, (a, b, c) in zip(P, items)]
        for f in full: assert len(f) == LT + 3, (len(f), LT)
        return torch.tensor(full, device=DEV)
    pid = tr_ids(tr[:1])[0, :LT].tolist(); assert pid == ids or True
    for cfg in CFGS:
        Ls, pos = cfg.split(":")
        Ls = list(range(NL)) if Ls == "all" else [int(x) for x in Ls.split("+")]
        FEs = ([FE_CLOSE] if pos in ("close", "ct") else []) + (FE_TAIL if pos in ("tail", "ct") else [])
        Wt = {L: torch.zeros(len(FEs), 31, D, device=DEV, requires_grad=True) for L in Ls}
        opt = torch.optim.Adam(list(Wt.values()), lr=LR)
        for step in range(STEPS):
            bi = [tr[(step * TBS + j) % len(tr)] for j in range(TBS)]
            x = tr_ids(bi); X = feats([a * b for a, b, c in bi]).to(DEV)
            for L in Ls: W.attach(L, [(fe, X, Wt[L][i]) for i, fe in enumerate(FEs)], "add", train=True)
            try:
                logits = model(input_ids=x).logits[:, LT - 1:LT + 2].float()
                loss = Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, LT:LT + 3].reshape(-1))
            finally: W.clear()
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 50 == 0 or step == STEPS - 1: log(cfg, "step", step, "loss", round(loss.item(), 4))
        Wd = {L: w.detach() for L, w in Wt.items()}
        def plan_for(corr):
            def plan(i0, i1):
                P = (S["corr"] if corr else Ptrue)[i0:i1]; X = feats(P).to(DEV)
                return [(L, [(fe, X, Wd[L][i]) for i, fe in enumerate(FEs)], "add") for L in Ls]
            return plan
        res[f"T_{cfg}"] = score(gen(S["P_task"], plan_for(False)))
        res[f"T_{cfg}_corr"] = score(gen(S["P_task"], plan_for(True)))
        res[f"T_{cfg}_loss"] = round(loss.item(), 4)
        log("TRAIN", cfg, res[f"T_{cfg}"], "corr", res[f"T_{cfg}_corr"])
        torch.save({L: w.cpu() for L, w in Wd.items()}, os.path.join(os.path.dirname(OUT), f"Wt_{cfg.replace(':','_')}{os.environ.get('CW_TAG', '')}.pt"))
        save()
elif MODE == "kv":
    # V3 (c): KV-entry write. Additive residual delta at ')+' on all layers (P-digit map, as V2 all:close) PLUS a learned
    # per-layer, per-head attention bias beta[l,h] from every LATER query to the ')+' key (= one extra un-rotated q/k dim:
    # q_extra = 1 at every position, k_extra = beta at the fired closing token). Needs LW_ATTN=eager.
    import transformers.models.qwen3.modeling_qwen3 as mq
    assert model.config._attn_implementation == "eager"
    _orig = mq.eager_attention_forward
    CL = LT - FE_CLOSE                      # absolute index of ')+' (equal-length prompts, left pad = 0)
    KV = {"beta": None}
    def patched(module, q, k, v, attention_mask, scaling, dropout=0.0, **kw):
        beta = KV["beta"]
        if beta is not None:
            T, S_ = q.shape[-2], k.shape[-2]
            if attention_mask is None:
                attention_mask = torch.full((T, S_), float("-inf"), device=q.device).triu(S_ - T + 1)[None, None].to(q.dtype)
            qpos = torch.arange(S_ - T, S_, device=q.device)
            col = torch.zeros(T, S_, device=q.device); col[:, CL] = (qpos > CL).float()
            attention_mask = attention_mask[:, :, :, :S_] + (beta[module.layer_idx][:, None, None] * col[None]).to(q.dtype)[None]
        return _orig(module, q, k, v, attention_mask, scaling, dropout, **kw)
    mq.eager_attention_forward = patched
    STEPS = int(os.environ.get("CW_STEPS", "300")); TBS = 32; LR = float(os.environ.get("CW_LR", "3e-3"))
    test_set = set(S["test"]); r = random.Random(7); tr = []
    while len(tr) < 6000:
        a, b, c = r.randint(11, 29), r.randint(3, 9), r.randint(10, 99)
        if a * b >= 100 and (a, b, c) not in test_set: tr.append((a, b, c))
    prefix = S["prefix"]; H = model.config.num_attention_heads
    def tr_ids(items):
        P = [prefix + f"Q: What is ({a}*{b})+{c}?\nA: " for a, b, c in items]
        return torch.tensor([tok(p + str(a * b + c), add_special_tokens=False).input_ids for p, (a, b, c) in zip(P, items)], device=DEV)
    for cfg in (sys.argv[2] if len(sys.argv) > 2 else "bias,nobias").split(","):
        Wt = {L: torch.zeros(1, 31, D, device=DEV, requires_grad=True) for L in range(NL)}
        beta = torch.zeros(NL, H, device=DEV, requires_grad=(cfg == "bias"))
        params = [{"params": list(Wt.values()), "lr": LR}] + ([{"params": [beta], "lr": float(os.environ.get("CW_BLR", "0.05"))}] if cfg == "bias" else [])
        opt = torch.optim.Adam(params)
        for step in range(STEPS):
            bi = [tr[(step * TBS + j) % len(tr)] for j in range(TBS)]
            x = tr_ids(bi); X = feats([a * b for a, b, c in bi]).to(DEV)
            for L in range(NL): W.attach(L, [(FE_CLOSE, X, Wt[L][0])], "add", train=True)
            KV["beta"] = beta
            try:
                logits = model(input_ids=x).logits[:, LT - 1:LT + 2].float()
                loss = Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, LT:LT + 3].reshape(-1))
            finally: W.clear(); KV["beta"] = None
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 50 == 0 or step == STEPS - 1: log("kv", cfg, "step", step, "loss", round(loss.item(), 4), "beta absmax", round(beta.abs().max().item(), 2))
        Wd = {L: w.detach() for L, w in Wt.items()}; bd = beta.detach()
        def plan_for(corr):
            def plan(i0, i1):
                P = (S["corr"] if corr else Ptrue)[i0:i1]; X = feats(P).to(DEV)
                return [(L, [(FE_CLOSE, X, Wd[L][0])], "add") for L in range(NL)]
            return plan
        KV["beta"] = bd
        try:
            res[f"KV_{cfg}"] = score(gen(S["P_task"], plan_for(False)))
            res[f"KV_{cfg}_corr"] = score(gen(S["P_task"], plan_for(True)))
        finally: KV["beta"] = None
        res[f"KV_{cfg}_loss"] = round(loss.item(), 4)
        log("KV", cfg, res[f"KV_{cfg}"], "corr", res[f"KV_{cfg}_corr"], "beta top", torch.topk(bd.flatten(), 5).values.tolist())
        torch.save({"W": {L: w.cpu() for L, w in Wd.items()}, "beta": bd.cpu()}, os.path.join(os.path.dirname(OUT), f"KV_{cfg}{os.environ.get('CW_TAG', '')}.pt"))
        save()
elif MODE == "gen":
    # V4 (d): the expression is GENERATED. Shots answer 'A: (a*b)+c = ANS'; the write fires at the generated ')+' token
    # (decode step, causal), all-layer additive delta from P-digit features (V2 all:close map, retrained on this format).
    # P is parsed in Python from the generated tokens (reference; the compiled backward-window circuit is the next step).
    STEPS = int(os.environ.get("CW_STEPS", "600")); TBS = 32; LR = float(os.environ.get("CW_LR", "3e-3"))
    rs = random.Random(1234); shots = []
    while len(shots) < 4:
        a, b, c = rs.randint(11, 29), rs.randint(3, 9), rs.randint(10, 99)
        shots.append(f"Q: What is ({a}*{b})+{c}?\nA: ({a}*{b})+{c} = {a*b+c}\n\n")
    gpre = "".join(shots)
    GP = [gpre + f"Q: What is ({a}*{b})+{c}?\nA:" for a, b, c in S["test"]]
    test_set = set(S["test"]); r = random.Random(7); tr = []
    while len(tr) < 6000:
        a, b, c = r.randint(11, 29), r.randint(3, 9), r.randint(10, 99)
        if a * b >= 100 and (a, b, c) not in test_set: tr.append((a, b, c))
    full0 = tok(GP[0] + f" ({S['test'][0][0]}*{S['test'][0][1]})+{S['test'][0][2]} = {S['ans_true'][0]}", add_special_tokens=False).input_ids
    ft = [tok.decode([i]) for i in full0]; GLP = len(tok(GP[0], add_special_tokens=False).input_ids)
    GCL = max(j for j, t in enumerate(ft) if t == ")+"); GLT = len(full0)
    assert GCL >= GLP and all(t.isdigit() for t in ft[-3:]), ft[GLP:]
    log("gen format answer tokens", ft[GLP:], "close idx", GCL, "prompt len", GLP)
    CLOSE_ID = full0[GCL]
    def tr_ids(items):
        x = [tok(gpre + f"Q: What is ({a}*{b})+{c}?\nA: ({a}*{b})+{c} = {a*b+c}", add_special_tokens=False).input_ids for a, b, c in items]
        for f in x: assert len(f) == GLT
        return torch.tensor(x, device=DEV)
    ST = {"hist": None, "plen": 0, "W": None, "corr": None, "i0": 0}
    def parseP(tokens):   # last '(' a '*' b ')+' in the generated text
        s = tok.decode(tokens); j = s.rfind("(")
        try: a, b = s[j + 1:].split(")")[0].split("*"); return int(a) * int(b)
        except Exception: return None
    def emb_hook(mod, args, kwargs):
        x = args[0] if args else kwargs["input_ids"]
        ST["hist"] = x.clone() if ST["hist"] is None or x.shape[1] > 1 else torch.cat([ST["hist"], x], 1)
    def mk_layer_hook(L):
        def pre(mod, args, kwargs):
            if ST["W"] is None: return None
            h = args[0] if args else kwargs["hidden_states"]
            if h.shape[1] != 1: return None                       # fire only on decode steps (generated tokens)
            hist = ST["hist"]; fired = (hist[:, -1] == CLOSE_ID).nonzero().flatten().tolist()
            if not fired: return None
            h2 = h.clone()
            for rI in fired:
                P = parseP(hist[rI, ST["plen"]:].tolist())
                if P is None or not (100 <= P <= 999): continue
                if ST["corr"] is not None: P = ST["corr"][ST["i0"] + rI]
                h2[rI, 0] = h2[rI, 0] + (feats([P]).to(DEV) @ ST["W"][L]).to(h.dtype)[0]
            if args: return (h2,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = h2; return args, kwargs
        return pre
    @torch.no_grad()
    def gen2(Wd=None, corr=None):
        outs = []; hs = [model.model.embed_tokens.register_forward_pre_hook(emb_hook, with_kwargs=True)]
        hs += [model.model.layers[L].register_forward_pre_hook(mk_layer_hook(L), with_kwargs=True) for L in range(NL)]
        try:
            for i0 in range(0, N, BS):
                enc = enc_batch(GP[i0:i0 + BS]); ST.update(hist=None, plen=enc.input_ids.shape[1], W=Wd, corr=corr, i0=i0)
                g = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
                outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        finally:
            for h in hs: h.remove()
            ST["W"] = None
        return outs
    def score2(outs):
        import re
        p = []; cp = 0
        for o, (a, b, c) in zip(outs, S["test"]):
            m = re.search(r"=\s*(-?\d+)", o); p.append(int(m.group(1)) if m else None); cp += o.strip().startswith(f"({a}*{b})+{c}")
        return {"acc": round(100 * sum(x == y for x, y in zip(p, S["ans_true"])) / N, 1),
                "follow": round(100 * sum(x == y for x, y in zip(p, S["ans_corr"])) / N, 1), "copy_ok": round(100 * cp / N, 1)}
    res["gen_plain"] = score2(gen2()); log("gen plain", res["gen_plain"])
    Wt = {L: torch.zeros(31, D, device=DEV, requires_grad=True) for L in range(NL)}
    opt = torch.optim.Adam(list(Wt.values()), lr=LR)
    for step in range(STEPS):
        bi = [tr[(step * TBS + j) % len(tr)] for j in range(TBS)]
        x = tr_ids(bi); X = feats([a * b for a, b, c in bi]).to(DEV)
        for L in range(NL): W.attach(L, [(GLT - GCL, X, Wt[L])], "add")
        try:
            if os.environ.get("CW_GLOSS", "ans") == "all":   # every token after the generated ')+' (c digits, ' =', answer)
                logits = model(input_ids=x).logits[:, GCL:GLT - 1].float(); tgt = x[:, GCL + 1:]
            else:
                logits = model(input_ids=x).logits[:, GLT - 4:GLT - 1].float(); tgt = x[:, GLT - 3:]
            loss = Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        finally: W.clear()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == STEPS - 1: log("gen train step", step, "loss", round(loss.item(), 4))
    Wd = {L: w.detach() for L, w in Wt.items()}
    res["gen_write"] = score2(gen2(Wd)); res["gen_write_corr"] = score2(gen2(Wd, S["corr"]))
    log("gen write", res["gen_write"], "corr", res["gen_write_corr"]); save()
    torch.save({L: w.cpu() for L, w in Wd.items()}, os.path.join(os.path.dirname(OUT), f"GEN{os.environ.get('CW_TAG', '')}.pt"))
log("done")
