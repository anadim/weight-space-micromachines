"""Usage: python run.py <task muladd|mulsub> <range mid|hard> [layers]
Builds the compiled implant, runs the machine check (compiled == Python reference), the silent check, and the eval.
"""
import os, sys, json, time, random, re
import torch
import lwv1
from lwv1 import load, setup, fit_writers, RefWriter, digit_feats, parse_int, DEV
from implant import Implant, arith_spec, match_spans

TASK, RN = sys.argv[1], sys.argv[2]
LAYS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,2,4,8,12,14,16,18").split(",")]
TAG = os.environ.get("LW_MODEL", "Qwen/Qwen3-0.6B-Base").split("/")[-1]
OUT = os.path.join(lwv1.OUT_DIR, f"{TASK}_{RN}_{TAG}.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
BS = 50
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

tok, model = load()
S = lwv1.setup_word() if TASK == "lost" else setup(TASK, RN)
N = len(S["test"]); cls = S.get("cls"); n = S["NS"]; nP = S["spec"].patterns[0]["nP"] if "spec" in S else n - 1; blk = 1 + 10 * nP
log("task", TASK, RN, "n_test", N, "class", cls, "slot from end", S["SLOT_FE"])
ENC, fbar = fit_writers(S, LAYS)
D = model.config.hidden_size
Wbank = {l: ENC[l].reshape(-1, D).to(DEV) for l in LAYS}                                # rows = [mu_i ; A_i] per slot
Wmean = {l: torch.cat([torch.cat([(ENC[l][i][0] + fbar @ ENC[l][i][1:])[None], torch.zeros(blk - 1, D)])
                       for i in range(n)]).to(DEV) for l in LAYS}                       # fixed-tensor no-value control
torch.save({"ENC": ENC, "fbar": fbar, "cls": cls, "model": TAG}, os.path.join(lwv1.OUT_DIR, f"writers_{TASK}_{RN}_{TAG}.pt"))
spec = S["spec"] if "spec" in S else arith_spec([cls]); PAT = spec.patterns[0]
imp = Implant(model, tok, spec, Wbank)
log("implant circuit units", imp.net.n_units, "nnz", imp.net.nnz, "layers", imp.net.L, "NF", imp.NF)
ref = RefWriter(ENC, fbar, spec)
res = {"task": TASK, "range": RN, "model": TAG, "n": N, "class": cls, "units": imp.net.n_units, "nnz": imp.net.nnz}


def enc_batch(ps):
    return tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)


def make_override(arm, i0):
    """CONTROL ONLY: replace the value features at the QUERY occurrence (last fired span) by P' or the mean."""
    if arm == "true": return None
    def ov(F, G, dbg):
        if F.shape[1] == 1: return F
        F = F.clone()
        for r in range(F.shape[0]):
            idx = (G[r, :, 0] > 0.5).nonzero().flatten()
            q = idx[-n:]
            assert len(q) == n and int(q[-1] - q[0]) == n - 1
            f = (digit_feats([S["corr"][i0 + r]], nP)[0] if arm == "corr" else fbar).to(F.device)
            for i, p in enumerate(q.tolist()): F[r, p, i * blk + 1:(i + 1) * blk] = f
        return F
    return ov


@torch.no_grad()
def gen(prompts, kind=None, L=None, arm="true", mode="all"):
    outs = []
    for i0 in range(0, len(prompts), BS):
        enc = enc_batch(prompts[i0:i0 + BS])
        if kind == "comp":
            imp.reset(); imp.override = make_override(arm, i0); imp.attach(model, L)
        elif kind == "comp_meanall":
            imp.reset(); imp.override = None; imp.Wbank = Wmean; imp.attach(model, L)
        elif kind == "ref":
            qv = None if arm == "true" else [S["corr"][i0 + r] if arm == "corr" else "mean" for r in range(enc.input_ids.shape[0])]
            ref.prepare(enc.input_ids, enc.attention_mask, mode=mode, qvals=qv); ref.attach(model, L)
        try:
            g = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            imp.detach_(); ref.detach_(); imp.Wbank = Wbank; imp.override = None
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def score(outs):
    p = [parse_int(o) for o in outs]
    return {"acc": sum(x == y for x, y in zip(p, S["ans_true"])) / N, "follow": sum(x == y for x, y in zip(p, S["ans_corr"])) / N}


# ------------------------------------------------------------------ 1. circuit check (fires/values == Python parse)
@torch.no_grad()
def circuit_check(prompts):
    bad_pos = bad_F = nfire = nP = badP = maxg = 0
    for i0 in range(0, len(prompts), BS):
        enc = enc_batch(prompts[i0:i0 + BS])
        imp.reset()
        F, G, dbg = imp.features(model.get_input_embeddings()(enc.input_ids))
        maxg = max(maxg, G.max().item())
        T = enc.input_ids.shape[1]
        Fpy = torch.zeros_like(F)
        for r in range(F.shape[0]):
            ids = enc.input_ids[r][enc.attention_mask[r].bool()].tolist(); padn = T - len(ids)
            for pi, st, P in match_spans([tok.decode([i]) for i in ids], spec):
                f = digit_feats([P], PAT["nP"])[0]
                for i, k in enumerate(PAT["slots"]):
                    Fpy[r, padn + st + k, i * blk] = 1; Fpy[r, padn + st + k, i * blk + 1:(i + 1) * blk] = f
                    if int(dbg[r, padn + st + k, 0].item()) != P: badP += 1
                    nP += 1
                nfire += 1
        bad_pos += int(((G[..., 0] > 0.5) != (Fpy[..., ::blk].sum(-1) > 0.5)).sum())
        bad_F += int((F != Fpy).any(-1).sum())
    return {"spans_fired": nfire, "slot_positions": nP, "pos_mismatch": bad_pos, "F_mismatch_positions": bad_F,
            "P_mismatch": badP, "max_G": maxg}


log("circuit check on task prompts")
res["circuit_check_task"] = circuit_check(S["P_task"]); log(res["circuit_check_task"])
res["circuit_check_text_oracle"] = circuit_check(S["P_or_true"]); log(res["circuit_check_text_oracle"])


# ------------------------------------------------------------------ 2. residual check at block L (compiled vs reference)
@torch.no_grad()
def residual_check(L, prompts):
    st = {"nonslot_bit_equal": True, "slot_pos_equal": True, "slot_max_abs_diff": 0.0, "slot_elems": 0, "slot_elems_bit_equal": 0, "max_abs_h": 0.0}
    for i0 in range(0, len(prompts), BS):
        enc = enc_batch(prompts[i0:i0 + BS])
        imp.reset(); imp.capture = []; imp.attach(model, L)
        model.model(**enc); imp.detach_()
        h_in, h_c = imp.capture[0]; imp.capture = None
        st["max_abs_h"] = max(st["max_abs_h"], imp.maxabs)
        ref.prepare(enc.input_ids, enc.attention_mask, mode="all"); ref.capture = []; ref.attach(model, L)
        model.model(**enc); ref.detach_()
        h_in2, h_r = ref.capture[0]; ref.capture = None
        assert torch.equal(h_in, h_in2)
        wr = (h_r != h_in).any(-1)                       # positions the reference changed
        wc = (h_c != h_in).any(-1)
        slotmask = torch.zeros_like(wr)
        for r, s0, v in ref.plan:
            for k in PAT["slots"]: slotmask[r, s0 + k] = True
        st["slot_pos_equal"] &= bool(torch.equal(wc, wr) or torch.equal(wc | slotmask, slotmask))
        st["nonslot_bit_equal"] &= bool(torch.equal(h_c[~slotmask], h_in[~slotmask]))
        d = (h_c[slotmask].float() - h_r[slotmask].float()).abs()
        st["slot_max_abs_diff"] = max(st["slot_max_abs_diff"], d.max().item())
        st["slot_elems"] += d.numel(); st["slot_elems_bit_equal"] += int((h_c[slotmask] == h_r[slotmask]).sum())
    return st


# ------------------------------------------------------------------ 3. silent check (no pattern -> bit-identical logits)
def silent_prompts():
    r = random.Random(7)
    shots_mul = re.sub(r"\((\d+)\*(\d+)\)", lambda m: f"({int(m.group(1))*int(m.group(2))})", S["prefix"])   # no pattern
    ps = [shots_mul + f"Q: What is {r.randint(11,49)}*{r.randint(3,29)}?\nA: " for _ in range(50)]         # a*b, no parens
    ps += [shots_mul + f"Q: What is ({r.randint(11,49)}+{r.randint(3,29)})*{r.randint(2,9)}?\nA: " for _ in range(50)]
    ps += [shots_mul + f"Q: What is ({r.randint(11,49)}*{r.randint(3,29)})*{r.randint(2,9)}?\nA: " for _ in range(50)]   # ')*' not ')+'
    ps += [shots_mul + f"Q: What is ({r.randint(1,9)}*{r.randint(100,999)})+{r.randint(1,9)}?\nA: " for _ in range(50)]  # other layout (1,3)
    txt = open(os.path.join(lwv1.HERE, "silent_text.txt")).read().split("\n\n")
    ps += [t for t in txt if len(t) > 40][:50]
    ps += [shots_mul + p[len(S["prefix"]):] for p in S["P_or_true"][:100]]           # '(P)+c' text oracle, neutral shots
    if TASK == "lost":   # oracle-sentence shots (no template), near misses (3 digits a, 2 digits b), text oracle
        osh = re.sub(r"Each box has (\d+) pens\. There are (\d+) boxes\.", lambda m: f"There are {int(m.group(1))*int(m.group(2))} pens.", S["prefix"])
        ps = [osh + p[len(S["prefix"]):] for p in S["P_or_true"][:100]]
        ps += [osh + f"Q: Each box has {r.randint(100,999)} pens. There are {r.randint(2,9)} boxes. 12 pens are lost. How many pens are left?\nA:" for _ in range(50)]
        ps += [osh + f"Q: Each box has {r.randint(11,29)} pens. There are {r.randint(10,30)} boxes. 12 pens are lost. How many pens are left?\nA:" for _ in range(50)]
        ps += [osh + f"Q: Each box has {r.randint(11,29)} pencils. There are {r.randint(3,9)} boxes. 12 pencils are lost. How many pencils are left?\nA:" for _ in range(50)]
        ps += [t for t in txt if len(t) > 40][:50]
    return ps


@torch.no_grad()
def silent_check(L, prompts):
    nb = 0; eq = 0; geq = 0; fired = 0.0
    for i0 in range(0, len(prompts), 25):
        enc = enc_batch(prompts[i0:i0 + 25])
        lp = model(**enc).logits
        imp.reset(); imp.attach(model, L)
        li = model(**enc).logits
        fired += imp.last[1].sum().item()
        imp.detach_()
        eq += int(torch.equal(lp, li)); nb += 1
    o_p = gen(prompts); o_i = gen(prompts, "comp", L)
    return {"batches": nb, "logits_bit_identical_batches": eq, "gen_identical": sum(a == b for a, b in zip(o_p, o_i)),
            "n_prompts": len(prompts), "fired_positions": fired}


SP = silent_prompts()
# the (1,3) layout sentences: the circuit only knows the eval class, so they must not fire either
for L in sorted({LAYS[0], 8} if 8 in LAYS else {LAYS[0]}):
    res[f"silent_L{L}"] = silent_check(L, SP); log("silent", L, res[f"silent_L{L}"])
json.dump(res, open(OUT, "w"), indent=1)

# ------------------------------------------------------------------ 4. eval
for name, ps in [("plain", S["P_task"]), ("text_oracle_true", S["P_or_true"]), ("text_oracle_corr", S["P_or_corr"]), ("text_tool_line", S["P_tool"])]:
    res[name] = score(gen(ps)); log(name, res[name])
json.dump(res, open(OUT, "w"), indent=1)

for L in LAYS:
    res[f"L{L}_residual_check"] = residual_check(L, S["P_task"]); log("L", L, "residual", res[f"L{L}_residual_check"])
    o = {}
    for arm in ["true", "corr", "mean"]:
        o["comp_" + arm] = gen(S["P_task"], "comp", L, arm)
        o["ref_" + arm] = gen(S["P_task"], "ref", L, arm, "all")
    o["comp_meanall"] = gen(S["P_task"], "comp_meanall", L)
    o["refq_true"] = gen(S["P_task"], "ref", L, "true", "query")
    o["refq_corr"] = gen(S["P_task"], "ref", L, "corr", "query")
    for k, v in o.items(): res[f"L{L}_{k}"] = score(v)
    res[f"L{L}_outputs_identical"] = {arm: sum(a == b for a, b in zip(o["comp_" + arm], o["ref_" + arm])) for arm in ["true", "corr", "mean"]}
    log(f"L{L:2d}", {k: (round(res[f'L{L}_{k}']['acc'] * 100, 1), round(res[f'L{L}_{k}']['follow'] * 100, 1)) for k in o}, "identical", res[f"L{L}_outputs_identical"])
    json.dump(res, open(OUT, "w"), indent=1)
log("done")
