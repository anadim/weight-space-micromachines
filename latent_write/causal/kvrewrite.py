"""Causal-trigger span rewrite (the 'rewrite the span's cached keys/values' option, honestly stated).
The span is detected CAUSALLY at its closing token (KF=0, all operands behind it); the value is then written into
the OPERAND positions' block-L residual by shifting the closing-token detection backward by (n-i) for operand slot i.
In a single prefill this edits the operand positions exactly as the non-causal span write does -> it reproduces the
non-causal accuracy. In a true token-by-token decode the operand residuals are already consumed when the closing token
arrives, so this is realizable only as an edit of those positions' CACHED K/V (a past-state edit), NOT a forward write.
That is the honest distinction from the forward-only 'causal_close' write (which never edits the past and fails).
usage: kvrewrite.py <muladd|mulsub> <mid|hard> [layers]"""
import os, sys, json, torch
import lwv1
from lwv1 import load, setup, digit_feats, parse_int, collect, ridge, DEV
from implant import Implant, Spec

TASK, RN = sys.argv[1], sys.argv[2]
LAYS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,4,8,12").split(",")]
tok, model = load(); S = setup(TASK, RN); N = len(S["test"]); D = model.config.hidden_size
cls = S["cls"]; la, lb, op = cls; n = la + 1 + lb; nP = n - 1; blk = 1 + 10 * nP
el = [('S', 'OP')] + [('A', la-1-j) for j in range(la)] + [('S', 'TI')] + [('B', lb-1-j) for j in range(lb)] + [('S', 'CP' if op == '+' else 'CM')]
close_idx = len(el) - 1
spec = Spec({'OP': [' (', '('], 'TI': ['*'], 'CP': [')+'], 'CM': [')-']}, [dict(elems=el, slots=[close_idx], nP=nP, name=cls)])
spec.KP = close_idx; spec.KF = 0; spec.W = spec.KP + 1

# non-causal operand-slot writer banks (same as the baseline), fit on oracle prompts
Ho = collect(S["P_fit"], LAYS, S["SLOT_FE"])
Fo = torch.cat([torch.ones(len(S["fitv"]), 1), digit_feats([v for v, _ in S["fitv"]], nP)], 1)
ENC = {l: torch.stack([ridge(Fo, Ho[l][:, i]) for i in range(S["NS"])]).cpu() for l in LAYS}   # [NS, blk, D]

class ShiftbackImplant(Implant):
    @torch.no_grad()
    def _write(self, mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        F, G = self.pending                       # F: (B,T,blk) nonzero at closing token
        hf = h.float(); B, T, _ = hf.shape
        banks = self.banks[self.L]                 # (NS, blk, D)
        off = lwv1 and 1
        out = hf.clone()
        NSc = banks.shape[0]
        for i in range(NSc):                       # operand slot i sits (n - i) tokens before the closing token
            shift = n - i
            Fsh = torch.zeros_like(F)
            if shift < T: Fsh[:, :T - shift] = F[:, shift:]
            wi = Fsh @ banks[i]                    # (B,T,D) value write at operand position i
            Gi = (Fsh[..., 0] > 0.5)               # big-M gated replacement at those positions
            BIGM = 1e6; offm = BIGM * (1 - Gi.float())[..., None]
            gh = torch.relu(hf - offm) - torch.relu(-hf - offm)
            out = torch.where(Gi[..., None], (hf - gh) + wi, out)
        out = out.to(h.dtype)
        if args: return (out,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = out; return args, kwargs

imp = ShiftbackImplant(model, tok, spec, {l: ENC[l].reshape(-1, D).to(DEV) for l in LAYS})
imp.banks = {l: ENC[l].to(DEV) for l in LAYS}
BS = 50
def score(outs):
    p = [parse_int(o) for o in outs]
    return sum(x == y for x, y in zip(p, S["ans_true"])) / N
@torch.no_grad()
def gen(L):
    outs = []
    for i0 in range(0, N, BS):
        enc = tok(S["P_task"][i0:i0+BS], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        imp.reset(); imp.attach(model, L)
        try: g = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        finally: imp.detach_()
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs
res = {}
for L in LAYS:
    res[L] = score(gen(L)); print(f"L{L} causal_KVrewrite acc {100*res[L]:.1f}", flush=True)
json.dump(res, open(os.path.join(lwv1.OUT_DIR, f"kvrewrite_{TASK}_{RN}.json"), "w"))
