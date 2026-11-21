"""latent_inweights_v1 engine: task setup (replicates the earlier E4 experiment script exactly: same seeds and prompts), writer fit (ridge on plain
oracle prompts with random values), the Python REFERENCE write (token parse + replace, the old method), and eval.
"""
import os, re, json, random, sys, time
from collections import Counter
import torch, torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WSM_ROOT') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)   # this folder + repo root (micromachines/)
OUT_DIR = os.environ.get("LW_OUT", os.path.join(HERE, "out"))   # new runs go here; the shipped results are in results/
MODEL = os.environ.get("LW_MODEL", "Qwen/Qwen3-0.6B-Base")
DEV = os.environ.get("LW_DEV", "cuda:0")
CAP = float(os.environ.get("LW_CAP_GB", "12"))
RANGES = {"mid": ((11, 29), (3, 9), (10, 99)), "hard": ((12, 49), (12, 29), (100, 899))}
_tok = _model = None


def load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _tok.padding_side = "left"
        if _tok.pad_token is None: _tok.pad_token = _tok.eos_token
        d = torch.device(DEV)
        tot = torch.cuda.get_device_properties(d).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, CAP * 1e9 / tot), d)
        _model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, **({"attn_implementation": os.environ["LW_ATTN"]} if os.environ.get("LW_ATTN") else {})).to(DEV).eval()
        for p in _model.parameters(): p.requires_grad_(False)
    return _tok, _model


def sample(rng, rn):
    (a0, a1), (b0, b1), (c0, c1) = RANGES[rn]
    return rng.randint(a0, a1), rng.randint(b0, b1), rng.randint(c0, c1)


def digit_feats(vals, width):
    f = torch.zeros(len(vals), width * 10)
    for i, v in enumerate(vals):
        for p, ch in enumerate(str(v).zfill(width)):
            f[i, p * 10 + int(ch)] = 1
    return f


# ------------------------------------------------------------------ task setup (= e4_general.py, same seeds)
def setup(task, rn, NTE=300):
    op = "-" if task == "mulsub" else "+"
    q_task = lambda a, b, c: (f"What is ({a}*{b}){op}{c}?", a * b - c if op == "-" else a * b + c)
    q_oracle = lambda vstr, c: f"What is ({vstr}){op}{c}?"
    ok_item = lambda a, b, c: task != "mulsub" or a * b > c
    rng = random.Random(11)
    rs = random.Random(1234)
    shots, shots_tool = [], []
    while len(shots) < 4:
        it = sample(rs, rn)
        if ok_item(*it):
            q, an = q_task(*it); shots.append(f"Q: {q}\nA: {an}\n\n")
            shots_tool.append(f"Q: {q}\nTool: {it[0]}*{it[1]} = {it[0]*it[1]}\nA: {an}\n\n")
    prefix = "".join(shots)
    allit = []
    r0 = random.Random(99)
    while len(allit) < 20000:
        it = sample(r0, rn)
        if ok_item(*it): allit.append(it)
    key = lambda it: (len(str(it[0])), len(str(it[1])), len(str(it[2])))
    common_key = Counter(key(it) for it in allit).most_common(1)[0][0]
    la, lb, lc = common_key
    NS = la + 1 + lb
    okP = lambda v: len(str(v)) == NS - 1
    test = [it for it in allit if key(it) == common_key and okP(it[0] * it[1])][:NTE]
    pool = [a * b for a, b, c in allit if key((a, b, c)) == common_key and okP(a * b)]
    corr = []
    for a, b, c in test:
        v = rng.choice(pool)
        while v == a * b or (task == "mulsub" and v <= c): v = rng.choice(pool)
        corr.append(v)
    fitv = [(rng.randint(10 ** (NS - 2), 10 ** (NS - 1) - 1), rng.choice([c for *_, c in test])) for _ in range(3000)]
    ans = lambda v, c: v - c if op == "-" else v + c
    S = dict(task=task, rn=rn, op=op, cls=(la, lb, op), NS=NS, prefix=prefix, test=test, corr=corr, fitv=fitv,
             P_task=[prefix + f"Q: {q_task(*it)[0]}\nA: " for it in test],
             P_or_true=[prefix + f"Q: {q_oracle(str(a*b), c)}\nA: " for a, b, c in test],
             P_or_corr=[prefix + f"Q: {q_oracle(str(v), c)}\nA: " for v, (a, b, c) in zip(corr, test)],
             P_tool=["".join(shots_tool) + f"Q: {q_task(a, b, c)[0]}\nTool: {a}*{b} = {a*b}\nA: " for a, b, c in test],
             P_fit=[prefix + f"Q: {q_oracle(str(v), c)}\nA: " for v, c in fitv],
             ans_true=[ans(a * b, c) for a, b, c in test], ans_corr=[ans(v, c) for v, (a, b, c) in zip(corr, test)])
    tok, _ = load()
    ids = tok(S["P_task"][0], add_special_tokens=False).input_ids
    toks = [tok.decode([i]) for i in ids]
    open_j = max(j for j, t in enumerate(toks) if t == " (")
    S["SLOT_FE"] = [len(ids) - j for j in range(open_j + 1, open_j + 1 + NS)]
    return S


# ------------------------------------------------------------------ writer fit (the only fitted tensors)
@torch.no_grad()
def collect(prompts, layer_ids, pos_from_end, bs=16):
    tok, model = load()
    res = {l: [] for l in layer_ids}
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        hs = model.model(**enc, output_hidden_states=True).hidden_states
        for l in layer_ids: res[l].append(hs[l][:, [-k for k in pos_from_end], :].float().cpu())
        del hs
    torch.cuda.empty_cache()
    return {l: torch.cat(v) for l, v in res.items()}


def ridge(X, Y, lam=1.0):
    Xd = X.to(DEV).double(); Yd = Y.to(DEV).double()
    return torch.linalg.solve(Xd.T @ Xd + lam * torch.eye(X.shape[1], device=DEV, dtype=torch.double), Xd.T @ Yd).float()


def fit_writers(S, layers):
    """ENC[L]: [n_slot, 1 + 10*(n-1), D] = per slot [mu ; A] (ridge, lam=1, on plain oracle prompts, random values)."""
    NS = S["NS"]
    Ho = collect(S["P_fit"], layers, S.get("ORC_FE", S["SLOT_FE"]))
    nP = S["spec"].patterns[0]["nP"] if "spec" in S else NS - 1
    Fo = torch.cat([torch.ones(len(S["fitv"]), 1), digit_feats([v for v, _ in S["fitv"]], nP)], 1)
    ENC = {l: torch.stack([ridge(Fo, Ho[l][:, i]) for i in range(NS)]).cpu() for l in layers}
    fbar = Fo[:, 1:].mean(0)
    return ENC, fbar


# ------------------------------------------------------------------ Python reference write (the old method)
def parse_spans(tok_strs, classes):
    """REFERENCE ONLY: all occurrences of ' (' a '*' b ')+'/')-' matching a class with len(a*b) == n-1.
    Returns list of (start_index_of_first_slot, class, P)."""
    out = []
    for j, t in enumerate(tok_strs):
        if t not in (" (", "("): continue
        k = j + 1; a = ""
        while k < len(tok_strs) and tok_strs[k].isdigit() and len(tok_strs[k]) == 1: a += tok_strs[k]; k += 1
        if k >= len(tok_strs) or tok_strs[k] != "*" or not a: continue
        k += 1; b = ""
        while k < len(tok_strs) and tok_strs[k].isdigit() and len(tok_strs[k]) == 1: b += tok_strs[k]; k += 1
        if k >= len(tok_strs) or tok_strs[k] not in (")+", ")-") or not b: continue
        c = (len(a), len(b), tok_strs[k][1])
        P = int(a) * int(b)
        if c in classes and len(str(P)) == slots(c) - 1: out.append((j + 1, c, P))
    return out


def slots(c): return c[0] + 1 + c[1]


class RefWriter:
    """REFERENCE (Python token match + replace, prefill only; the old method). spec with ONE pattern.
    mode: 'all' (every occurrence) or 'query' (last only). qvals: per-row value at the query occurrence:
    None (true P), int P', or 'mean'."""

    def __init__(self, ENC, fbar, spec):
        from implant import match_spans
        assert len(spec.patterns) == 1
        self.ENC, self.fbar, self.spec, self.match = ENC, fbar, spec, match_spans
        self.pat = spec.patterns[0]
        self.handle = None; self.plan = None; self.capture = None

    def prepare(self, enc_ids, enc_mask, mode="all", qvals=None):
        tok, _ = load()
        T = enc_ids.shape[1]
        plan = []
        for r in range(enc_ids.shape[0]):
            ids = enc_ids[r][enc_mask[r].bool()].tolist(); padn = T - len(ids)
            spans = self.match([tok.decode([i]) for i in ids], self.spec)
            if mode == "query": spans = spans[-1:]
            for si, (pi, st, P) in enumerate(spans):
                v = P
                if si == len(spans) - 1 and qvals is not None and qvals[r] is not None: v = qvals[r]
                plan.append((r, padn + st, v))
        self.plan = plan

    def attach(self, model, L):
        self.L = L
        self.handle = model.model.layers[L].register_forward_pre_hook(self._pre, with_kwargs=True)

    def detach_(self):
        if self.handle: self.handle.remove(); self.handle = None

    @torch.no_grad()
    def _pre(self, mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if h.shape[1] == 1: return None
        h2 = h.clone()
        E = self.ENC[self.L].to(h.device)
        nP = self.pat["nP"]
        for r, st, v in self.plan:
            f = self.fbar.to(h.device) if v == "mean" else digit_feats([v], nP)[0].to(h.device)
            for i, k in enumerate(self.pat["slots"]):
                h2[r, st + k] = (E[i][0] + f @ E[i][1:]).to(h.dtype)
        if self.capture is not None: self.capture.append((h.clone(), h2.clone()))
        if args: return (h2,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = h2; return args, kwargs


def parse_int(s):
    m = re.match(r"\s*(-?\d+)", s)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------ word problem 'lost' (= the earlier word-problem script, loc 'both', same seed)
def setup_word(NTE=300, NFIT=2000):
    from implant import Spec
    def sample_w(r):
        while True:
            a, b, c = r.randint(11, 29), r.randint(3, 9), r.randint(10, 99)
            if a * b >= 100: return a, b, c
    pool = [a * b for a in range(11, 30) for b in range(3, 10) if a * b >= 100]
    body = lambda a, b, c: f" Each box has {a} pens. There are {b} boxes. {c} pens are lost. How many pens are left?"
    orc = lambda P, c: f" There are {P} pens. {c} pens are lost. How many pens are left?"
    rng = random.Random(21)
    rs = random.Random(1234)
    shots = [sample_w(rs) for _ in range(4)]
    prefix = "".join(f"Q: {body(*s).strip()}\nA: {s[0]*s[1]-s[2]}\n\n" for s in shots)
    prefix_tool = "".join(f"Q: {body(*s).strip()}\nTool: {s[0]}*{s[1]} = {s[0]*s[1]}\nA: {s[0]*s[1]-s[2]}\n\n" for s in shots)
    Q = lambda txt: prefix + "Q:" + txt + "\nA:"
    test = [sample_w(rng) for _ in range(NTE)]
    corr = []
    for a, b, c in test:
        w = rng.choice(pool)
        while w == a * b: w = rng.choice(pool)
        corr.append(w)
    for _ in range(2 * NFIT): sample_w(rng); rng.randint(100, 999)      # the earlier script fitted sentA, sentB first (keeps the RNG stream identical)
    fit = []
    for _ in range(NFIT):
        it = sample_w(rng); v = rng.randint(100, 999); fit.append((v, it[2]))
    spec = Spec({'EACH': [' Each'], 'BOX': [' box'], 'HAS': [' has'], 'SPC': [' '], 'PENS': [' pens'], 'DOT': ['.'],
                 'THERE': [' There'], 'ARE': [' are'], 'BOXES': [' boxes']},
                [dict(elems=[('S', 'EACH'), ('S', 'BOX'), ('S', 'HAS'), ('S', 'SPC'), ('A', 1), ('A', 0), ('S', 'PENS'),
                             ('S', 'DOT'), ('S', 'THERE'), ('S', 'ARE'), ('S', 'SPC'), ('B', 0), ('S', 'BOXES'), ('S', 'DOT')],
                      slots=list(range(14)), nP=3, name='lost')])
    S = dict(task="lost", rn="mid", op="-", NS=14, prefix=prefix, test=test, corr=corr, fitv=fit, spec=spec,
             P_task=[Q(body(*it)) for it in test],
             P_or_true=[Q(orc(a * b, c)) for a, b, c in test],
             P_or_corr=[Q(orc(v, c)) for v, (a, b, c) in zip(corr, test)],
             P_tool=[prefix_tool + f"Q: {body(*it).strip()}\nTool: {it[0]}*{it[1]} = {it[0]*it[1]}\nA:" for it in test],
             P_fit=[Q(orc(v, c)) for v, c in fit],
             ans_true=[a * b - c for a, b, c in test], ans_corr=[v - c for v, (a, b, c) in zip(corr, test)])
    tok, _ = load()
    ti = [tok.decode([i]) for i in tok(S["P_task"][0], add_special_tokens=False).input_ids]
    oi = [tok.decode([i]) for i in tok(S["P_or_true"][0], add_special_tokens=False).input_ids]
    t0 = max(j for j, t in enumerate(ti) if t == " Each"); o0 = max(j for j, t in enumerate(oi) if t == " There")
    assert ti[t0 + 13] == "." and oi[o0 + 7] == "." and oi[o0 + 3:o0 + 6] == list(str(test[0][0] * test[0][1]))
    S["SLOT_FE"] = [len(ti) - (t0 + k) for k in range(14)]
    S["ORC_FE"] = [len(oi) - ((o0 + 7 - (13 - k)) if 13 - k < 8 else o0 - 1) for k in range(14)]   # right-aligned, lw2 rule
    return S
