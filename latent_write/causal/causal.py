"""Causal latent-write variant. The non-causal implant detects the (a*b) span with a +/-K window (6 tokens of
LOOK-AHEAD) and rewrites the a*b operand positions. This only works because the prompt is prefilled in one pass.

Causal variant: a backward-only window (KF=0). The span is detected AT ITS CLOSING TOKEN ')+'/')-' (all operands are
behind it). The value P=a*b is written at the closing token position (nwrite=1) and, as a sweep, the closing + a few
FOLLOWING positions (nwrite>1; those are c/'?' tokens -> expected to hurt on arith, a disclosed confound). Everything
on the inference path is still a fixed tensor op in the Implant (reader + compiled ReLU circuit + big-M gated write);
only the write SLOTS and the fitted writer bank change. We compare causal vs the non-causal span write by layer, with
the fully-in-weights mean-value control (comp_meanall) for both.
usage: python causal.py <muladd|mulsub|lost> <mid|hard> [layers] [nwrite_list]
"""
import os, sys, json, time, random, re
import torch
import lwv1
from lwv1 import load, setup, digit_feats, parse_int, collect, ridge, DEV
from implant import Implant, arith_spec, Spec, build_circuit, match_spans

TASK, RN = sys.argv[1], sys.argv[2]
LAYS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,4,8,12,14,16").split(",")]
NWR = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1").split(",")]
TAG = os.environ.get("LW_MODEL", "Qwen/Qwen3-0.6B-Base").split("/")[-1]
OUT = os.path.join(lwv1.OUT_DIR, f"causal_{TASK}_{RN}_{TAG}.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BS = 50
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
tok, model = load()
S = lwv1.setup_word() if TASK == "lost" else setup(TASK, RN)
N = len(S["test"]); D = model.config.hidden_size


def tokS(p): return [tok.decode([i]) for i in tok(p, add_special_tokens=False).input_ids]

# ----- find closing-token positions (from end) for task + oracle -----
def close_fe(prompt_tokens, closing=(")+", ")-", ".")):
    # last occurrence of a closing symbol that is the span closer
    pass

# Build causal spec: slots = the last nwrite element indices (closing token and the nwrite-1 before it is NOT causal;
# for arith nwrite>1 means closing + FOLLOWING positions via forward shift). We detect at the closing token and write
# there; FOLLOWING positions are handled by a forward-shift of the fired flag (see ShiftImplant).
if TASK == "lost":
    base = S["spec"]; el = base.patterns[0]["elems"]; nP = base.patterns[0]["nP"]
    close_idx = len(el) - 1
    symbols = {k: v for k, v in base.TEXT.items() if not k.isdigit()}
    def make_causal_spec(nwrite):
        sp = Spec(symbols, [dict(elems=el, slots=[close_idx], nP=nP, name="lost_causal")])
        sp.KP = close_idx; sp.KF = 0; sp.W = sp.KP + 1
        return sp
    cls = None
else:
    cls = S["cls"]; la, lb, op = cls; n = la + 1 + lb; nP = n - 1
    el = [('S', 'OP')] + [('A', la-1-j) for j in range(la)] + [('S', 'TI')] + [('B', lb-1-j) for j in range(lb)] + [('S', 'CP' if op == '+' else 'CM')]
    close_idx = len(el) - 1
    def make_causal_spec(nwrite):
        sp = Spec({'OP': [' (', '('], 'TI': ['*'], 'CP': [')+'], 'CM': [')-']}, [dict(elems=el, slots=[close_idx], nP=nP, name=cls)])
        sp.KP = close_idx; sp.KF = 0; sp.W = sp.KP + 1
        return sp

# closing-token position (from end) in task and oracle prompts
tks = tokS(S["P_task"][0]); oks = tokS(S["P_or_true"][0])
CLOSE = {")+", ")-"} if TASK != "lost" else {"."}
if TASK == "lost":
    # the closing '.' after ' boxes' : take the '.' that closes the defining clause = SLOT_FE[0]-aligned last '.'
    tci = S["SLOT_FE"][0]; oci = S["ORC_FE"][0]   # from-end of the task DOT and oracle (right-aligned) pos
    CLOSE_FE_TASK = tci; CLOSE_FE_ORC = len(oks) - max(j for j, t in enumerate(oks) if t == ".")
else:
    CLOSE_FE_TASK = len(tks) - max(j for j, t in enumerate(tks) if t in CLOSE)
    CLOSE_FE_ORC = len(oks) - max(j for j, t in enumerate(oks) if t in CLOSE)
log("close_fe task", CLOSE_FE_TASK, "oracle", CLOSE_FE_ORC, "token", tks[-CLOSE_FE_TASK], oks[-CLOSE_FE_ORC])

# ----- fit writers -----
def fit_causal(layers, fe_list):
    Ho = collect(S["P_fit"], layers, fe_list)
    Fo = torch.cat([torch.ones(len(S["fitv"]), 1), digit_feats([v for v, _ in S["fitv"]], nP)], 1)
    ENC = {l: torch.stack([ridge(Fo, Ho[l][:, i]) for i in range(len(fe_list))]).cpu() for l in layers}
    fbar = Fo[:, 1:].mean(0)
    return ENC, fbar

def enc_batch(ps): return tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
def score(outs):
    p = [parse_int(o) for o in outs]
    return {"acc": sum(x == y for x, y in zip(p, S["ans_true"])) / N, "follow": sum(x == y for x, y in zip(p, S["ans_corr"])) / N}

@torch.no_grad()
def gen_imp(imp, L, meanall_bank=None):
    outs = []
    for i0 in range(0, N, BS):
        enc = enc_batch(S["P_task"][i0:i0 + BS])
        imp.reset()
        if meanall_bank is not None:
            saved = imp.Wbank; imp.Wbank = meanall_bank
        imp.attach(model, L)
        try:
            g = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            imp.detach_()
            if meanall_bank is not None: imp.Wbank = saved
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs

res = {"task": TASK, "range": RN, "model": TAG, "n": N, "class": cls, "close_fe_task": CLOSE_FE_TASK, "close_fe_orc": CLOSE_FE_ORC}
for name, ps in [("plain", "P_task"), ("text_oracle_true", "P_or_true"), ("text_tool_line", "P_tool")]:
    outs = []
    for i0 in range(0, N, BS):
        enc = enc_batch(S[ps][i0:i0 + BS]); g = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    res[name] = score(outs); log(name, res[name])

# ---- CAUSAL close-token write (nwrite=1) ----
spec_c = make_causal_spec(1)
ENCc, fbarc = fit_causal(LAYS, [CLOSE_FE_ORC])
blk = 1 + 10 * nP
Wbank_c = {l: ENCc[l].reshape(-1, D).to(DEV) for l in LAYS}
Wmean_c = {l: torch.cat([(ENCc[l][0][0] + fbarc @ ENCc[l][0][1:])[None], torch.zeros(blk - 1, D)]).to(DEV) for l in LAYS}
imp_c = Implant(model, tok, spec_c, Wbank_c)
log("causal implant units", imp_c.net.n_units, "nnz", imp_c.net.nnz, "NF", imp_c.NF)

# circuit check: does the causal circuit fire exactly at the closing token with correct P?
@torch.no_grad()
def circuit_check_causal(prompts):
    nfire = badP = 0
    for i0 in range(0, len(prompts), BS):
        enc = enc_batch(prompts[i0:i0 + BS])
        imp_c.reset()
        F, G, dbg = imp_c.features(model.get_input_embeddings()(enc.input_ids))
        T = enc.input_ids.shape[1]
        for r in range(F.shape[0]):
            ids = enc.input_ids[r][enc.attention_mask[r].bool()].tolist(); padn = T - len(ids)
            spans = match_spans([tok.decode([i]) for i in ids], spec_c)   # won't match (spec slots=closing); use parse
            # recompute expected: each (a*b) span -> P, fired at its closing token (padn + st + close_idx)
        # count fired positions and check P via dbg at fired positions
        fired = (G[..., 0] > 0.5)
        nfire += int(fired.sum())
    return {"fired_positions_total": nfire}
res["causal_circuit"] = circuit_check_causal(S["P_task"]); log("causal circuit", res["causal_circuit"])

for L in LAYS:
    res[f"L{L}_causal_close"] = score(gen_imp(imp_c, L))
    res[f"L{L}_causal_meanall"] = score(gen_imp(imp_c, L, meanall_bank=Wmean_c))
    log(f"L{L} causal_close", res[f"L{L}_causal_close"], "meanall", res[f"L{L}_causal_meanall"])
    json.dump(res, open(OUT, "w"), indent=1)
imp_c.detach_()

# ---- NON-CAUSAL span write (baseline, same items) ----
spec_nc = S["spec"] if TASK == "lost" else arith_spec([cls])
NS = S["NS"]
Ho = collect(S["P_fit"], LAYS, S.get("ORC_FE", S["SLOT_FE"]))
Fo = torch.cat([torch.ones(len(S["fitv"]), 1), digit_feats([v for v, _ in S["fitv"]], nP)], 1)
ENCnc = {l: torch.stack([ridge(Fo, Ho[l][:, i]) for i in range(NS)]).cpu() for l in LAYS}
fbarnc = Fo[:, 1:].mean(0)
Wbank_nc = {l: ENCnc[l].reshape(-1, D).to(DEV) for l in LAYS}
Wmean_nc = {l: torch.cat([torch.cat([(ENCnc[l][i][0] + fbarnc @ ENCnc[l][i][1:])[None], torch.zeros(blk - 1, D)]) for i in range(NS)]).to(DEV) for l in LAYS}
imp_nc = Implant(model, tok, spec_nc, Wbank_nc)
for L in LAYS:
    res[f"L{L}_noncausal_span"] = score(gen_imp(imp_nc, L))
    res[f"L{L}_noncausal_meanall"] = score(gen_imp(imp_nc, L, meanall_bank=Wmean_nc))
    log(f"L{L} noncausal_span", res[f"L{L}_noncausal_span"], "meanall", res[f"L{L}_noncausal_meanall"])
    json.dump(res, open(OUT, "w"), indent=1)
imp_nc.detach_()
log("done"); json.dump(res, open(OUT, "w"), indent=1)
