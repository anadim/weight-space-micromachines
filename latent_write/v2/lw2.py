"""latent_write2 engine: write tool-computed values into the residual stream of a frozen LM at chosen token spans.

Convention (same as latent_write): "write at layer L" = replace the residual stream ENTERING block L (L=0 = embeddings).
Prompts are marked with [[...]] around spans; markers are stripped before tokenization.

A *condition* = task prompt with m marked spans + an *oracle* prompt (the same text with each written span replaced by
its value in text, marked with [[...]]).  Alignment per span: right-align the oracle value tokens with the task span
tokens; task span positions left over on the left get the state of the oracle token just before the value (the E3
' (' fix); if the oracle value is longer than the task span the alignment extends left into the task prefix.
Writer (the only fitted part): per layer and aligned position, a ridge map  [1, onehot(values)] -> residual state,
fitted on oracle prompts with random values (no task answers).  Modes: synth (writer), donor (real oracle states of
the same item; upper bound), mean (writer with mean features = no value information).
"""
import os, re, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("LW_MODEL", "Qwen/Qwen3-0.6B-Base")
DEV = os.environ.get("LW_DEV", "cuda:0")
_tok = _model = _cont = None


def load():
    global _tok, _model, _cont
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _tok.padding_side = "left"
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
        tot = torch.cuda.get_device_properties(torch.device(DEV)).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 7.5e9 / tot), torch.device(DEV))  # self-imposed 8 GB cap
        _model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
        for p in _model.parameters():
            p.requires_grad_(False)
        # token "continues an answer" iff its text starts with an alphanumeric char
        V = _model.lm_head.weight.shape[0]
        cont = torch.zeros(V, dtype=torch.bool)
        for t in range(min(V, len(_tok))):
            s = _tok.decode([t])
            cont[t] = len(s) > 0 and s[0].isalnum()
        _cont = cont.to(DEV)
    return _tok, _model


def nlayers():
    return len(load()[1].model.layers)


# ------------------------------------------------------------------ markup / tokenization
def unmark(s):
    spans, out, i = [], [], 0
    pos = 0
    for m in re.finditer(r"\[\[|\]\]", s):
        out.append(s[pos:m.start()]); i += m.start() - pos; pos = m.end()
        if m.group() == "[[":
            st = i
        else:
            spans.append((st, i))
    out.append(s[pos:])
    return "".join(out), spans


def tokinfo(marked):
    tok, _ = load()
    s, spans = unmark(marked)
    e = tok(s, return_offsets_mapping=True, add_special_tokens=False)
    off = e.offset_mapping
    sp = []
    for cs, ce in spans:
        idx = [j for j, (a, b) in enumerate(off) if a >= cs and b <= ce and b > a]
        assert idx, (marked, cs, ce)
        sp.append((idx[0], idx[-1] + 1))
    return s, e.input_ids, sp


def align(task_marked, orc_marked, which):
    """which: list of task span indices written; oracle marks are in the same order. Returns (pairs, ntask, norc)
    pairs = list of (task_from_end, oracle_from_end, k) with k = index in `which`."""
    _, ti, ts = tokinfo(task_marked)
    _, oi, os_ = tokinfo(orc_marked)
    assert len(os_) == len(which), (orc_marked, which)
    pairs = []
    for k, j in enumerate(which):
        (t0, t1), (o0, o1) = ts[j], os_[k]
        n, m = t1 - t0, o1 - o0
        for i in range(n):  # oracle longer than the task span: its leftmost tokens are dropped (keep spans wide enough)
            tp = t1 - 1 - i
            op = o1 - 1 - i if i < m else o0 - 1
            pairs.append((len(ti) - tp, len(oi) - op, k))
    return pairs, len(ti), len(oi)


def suffix_pairs(task_marked, orc_marked, which):
    """positions after the last written span (identical text in task and oracle), aligned from the end"""
    _, ti, ts = tokinfo(task_marked)
    _, oi, os_ = tokinfo(orc_marked)
    t_end = ts[which[-1]][1]
    return [(len(ti) - p, len(ti) - p, -1) for p in range(t_end, len(ti))]


# ------------------------------------------------------------------ hooks
class Injector:
    """edits: dict layer -> fn(h) -> h. Applied only on multi-token passes."""

    def __init__(self, edits):
        self.edits = edits or {}
        self.handles = []

    def __enter__(self):
        L = load()[1].model.layers
        for li, fn in self.edits.items():
            def pre(mod, args, kwargs, fn=fn):
                h = args[0] if args else kwargs["hidden_states"]
                if h.shape[1] == 1:
                    return None
                h2 = fn(h)
                if args:
                    return (h2,) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = h2
                return args, kwargs
            self.handles.append(L[li].register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()



import copy
_PCACHE = {}


@torch.no_grad()
def _prefix_len(ids, am, need=1):
    """longest common token prefix of the batch (no left padding), leaving >= 1 token"""
    eq = (ids == ids[:1]).all(0)
    nz = (~eq).nonzero()
    lcp = int(nz[0]) if len(nz) else ids.shape[1]
    lcp = min(lcp, ids.shape[1] - need)
    if lcp < 16 or not bool(am[:, :lcp].all()):
        return 0
    return lcp


@torch.no_grad()
def _cached_forward(ids, am, hooks_for_plen, hidden=False, need=1):
    """forward with a shared-prefix KV cache. hooks_for_plen(plen) -> dict layer->fn (fn sees only the new tokens)."""
    _, model = load()
    plen = _prefix_len(ids, am, need)
    kw = {}
    if plen:
        key = tuple(ids[0, :plen].tolist())
        if key not in _PCACHE:
            if len(_PCACHE) > 8:
                _PCACHE.clear()
            out = model.model(input_ids=ids[:1, :plen], use_cache=True)
            _PCACHE[key] = out.past_key_values
        c = copy.deepcopy(_PCACHE[key])
        c.batch_repeat_interleave(ids.shape[0])
        kw = dict(past_key_values=c, use_cache=True)
    with Injector(hooks_for_plen(plen)):
        out = model.model(input_ids=ids[:, plen:], attention_mask=am, output_hidden_states=hidden, **kw)
    return out, plen


def encode(prompts):
    tok, _ = load()
    return tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)


@torch.no_grad()
def collect(prompts, lays, fe, bs=32):
    """states entering block l at positions-from-end fe: dict l -> [N, len(fe), D] bf16 cpu"""
    _, model = load()
    res = {l: [] for l in lays}
    for i in range(0, len(prompts), bs):
        enc = encode(prompts[i:i + bs])
        out, plen = _cached_forward(enc.input_ids, enc.attention_mask, lambda p: {}, hidden=True, need=max(fe))
        hs = out.hidden_states
        assert max(fe) <= enc.input_ids.shape[1] - plen
        for l in lays:
            res[l].append(hs[l][:, [-k for k in fe], :].to(torch.bfloat16).cpu())
        del hs
    torch.cuda.empty_cache()
    return {l: torch.cat(v) for l, v in res.items()}


def ridge_multi(X, Y, lam=1.0):
    """X [N,f], Y [N,P,D] -> W [P,f,D]"""
    N, P, D = Y.shape
    Xd = X.to(DEV).double()
    A = Xd.T @ Xd + lam * torch.eye(X.shape[1], device=DEV, dtype=torch.double)
    out = []
    for p in range(P):
        out.append(torch.linalg.solve(A, Xd.T @ Y[:, p].to(DEV).double()).float())
    return torch.stack(out)


# ------------------------------------------------------------------ scoring
def cand_tokens(prompt, cand):
    tok, _ = load()
    a = tok(prompt, add_special_tokens=False).input_ids
    b = tok(prompt + cand, add_special_tokens=False).input_ids
    if b[:len(a)] == a:
        return b[len(a):]
    return tok(cand, add_special_tokens=False).input_ids


@torch.no_grad()
def tf_score(prompts, cands, edits=None, bs=None):
    """prompts: list (same token length assumed within a batch after padding is fine); cands: list of list of str.
    edits: None or list of (layer, fe_positions list, S [N_items, npos, D]) applied to rows of the item.
    Returns list of lists of bool: greedy output starts with exactly that candidate (then a non-alnum token)."""
    tok, model = load()
    out = []
    if bs is None:
        bs = max(1, 64 // max(len(c) for c in cands))
    for i0 in range(0, len(prompts), bs):
        pr = prompts[i0:i0 + bs]
        enc = encode(pr)
        END = enc.input_ids.shape[1]
        rows = []
        for j, p in enumerate(pr):
            for c in cands[i0 + j]:
                rows.append((j, cand_tokens(p, c)))
        maxc = max(len(c) for _, c in rows)
        ids = torch.full((len(rows), END + maxc), tok.pad_token_id, device=DEV)
        am = torch.zeros_like(ids)
        src = torch.tensor([j for j, _ in rows], device=DEV)
        ids[:, :END] = enc.input_ids[src]
        am[:, :END] = enc.attention_mask[src]
        for r, (j, c) in enumerate(rows):
            ids[r, END:END + len(c)] = torch.tensor(c, device=DEV)
            am[r, END:END + len(c)] = 1
        def mkhooks(plen):
            hooks = {}
            for (L, fe, S) in (edits or []):
                assert END - max(fe) >= plen
                def fn(h, fe=fe, S=S, prev=hooks.get(L)):
                    if prev is not None:
                        h = prev(h)
                    h = h.clone()
                    st = S[(src + i0).cpu()].to(DEV, h.dtype)  # [R, npos, D]
                    for q, k in enumerate(fe):
                        h[:, END - plen - k] = st[:, q]
                    return h
                hooks[L] = fn
            return hooks
        need = END + maxc - (END - max([max(fe) for (_, fe, _) in (edits or [])] or [1]))
        out_, plen = _cached_forward(ids, am, mkhooks, need=need)
        hid = out_.last_hidden_state
        pred = model.lm_head(hid[:, END - plen - 1:, :]).argmax(-1)
        del hid
        tgt = torch.full((len(rows), maxc + 1), -1, device=DEV)
        lens = torch.tensor([len(c) for _, c in rows], device=DEV)
        for r, (j, c) in enumerate(rows):
            tgt[r, :len(c)] = torch.tensor(c)
        ar = torch.arange(maxc + 1, device=DEV)[None]
        match = ((pred == tgt) | (ar >= lens[:, None])).all(1)
        nxt = pred.gather(1, lens[:, None])[:, 0]
        res = (match & ~_cont[nxt]).tolist()
        k = 0
        for j in range(len(pr)):
            n = len(cands[i0 + j]); out.append(res[k:k + n]); k += n
    return out


def rates(r):
    n = len(r)
    return [round(sum(x[i] for x in r) / n, 4) for i in range(len(r[0]))]


@torch.no_grad()
def generate(prompts, max_new=8, bs=32):
    tok, model = load()
    outs = []
    for i in range(0, len(prompts), bs):
        enc = encode(prompts[i:i + bs])
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


# ------------------------------------------------------------------ features
def f_num(v, width):
    s = str(v).zfill(width)
    assert len(s) == width, (v, width)
    f = [0.0] * (10 * width)
    for p, ch in enumerate(s):
        f[p * 10 + int(ch)] = 1.0
    return f


def f_cat(v, cats):
    f = [0.0] * len(cats)
    f[cats.index(v)] = 1.0
    return f


# ------------------------------------------------------------------ experiment core
class SpanWrite:
    """Writer for one span configuration.
    ref_task_m / ref_orc_m: a reference marked task prompt and its marked oracle (defines the token layout).
    which: task span ids written (oracle marks are in the same order). feat(values) -> list of floats.
    fit_set: list of (marked oracle prompt, values) with random values (no task answers)."""

    def __init__(self, ref_task_m, ref_orc_m, which, feat, fit_set, lays, with_suffix=False):
        self.which, self.lays, self.feat = which, lays, feat
        self.ref = (ref_task_m, ref_orc_m)
        self.pairs = align(ref_task_m, ref_orc_m, which)[0]
        if with_suffix:
            self.pairs = self.pairs + suffix_pairs(ref_task_m, ref_orc_m, which)
        self.tfe = [p[0] for p in self.pairs]
        self.ofe = [p[1] for p in self.pairs]
        osig = osig_of(ref_orc_m)
        fs = [(om, v) for (om, v) in fit_set if osig_of(om) == osig]
        self.nfit = len(fs)
        F = torch.tensor([[1.0] + feat(v) for _, v in fs])
        H = collect([unmark(om)[0] for om, _ in fs], lays, self.ofe)
        self.W = {l: ridge_multi(F, H[l]) for l in lays}  # [P, f, D]
        self.r2 = {}
        for l in lays:
            pred = torch.einsum("nf,pfd->npd", F.to(DEV), self.W[l]).cpu()
            y = H[l].float()
            self.r2[l] = round(float(1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()), 3)
        del H
        self.fmean = F.mean(0)

    def ok(self, task_m, orc_m):
        """does this (task, oracle) pair have the reference layout?"""
        return align(task_m, orc_m, self.which)[0] == align(*self.ref, self.which)[0]

    def _group(self, layer_of_span):
        groups = {}
        for q, (_, _, k) in enumerate(self.pairs):
            L = layer_of_span.get(k, layer_of_span.get(-1, max(layer_of_span.values())))
            groups.setdefault(L, []).append(q)
        return groups

    def synth(self, vals, layer_of_span):
        F = torch.tensor([[1.0] + self.feat(v) for v in vals]).to(DEV)
        return self._from_F(F, layer_of_span)

    def mean(self, n, layer_of_span):
        return self._from_F(self.fmean.expand(n, -1).to(DEV), layer_of_span)

    def _from_F(self, F, layer_of_span):
        ed = []
        for L, qs in self._group(layer_of_span).items():
            S = torch.einsum("nf,pfd->npd", F, self.W[L][qs]).to(torch.bfloat16).cpu()
            ed.append((L, [self.tfe[q] for q in qs], S))
        return ed

    def donor(self, orc_m_list, layer_of_span):
        ed = []
        prompts = [unmark(o)[0] for o in orc_m_list]
        for L, qs in self._group(layer_of_span).items():
            S = collect(prompts, [L], [self.ofe[q] for q in qs])[L]
            ed.append((L, [self.tfe[q] for q in qs], S))
        return ed


def osig_of(om):
    _, oi, os_ = tokinfo(om)
    return (len(oi), tuple(os_))


def tsig_of(tm):
    _, ti, ts = tokinfo(tm)
    return (len(ti), tuple(ts))


def shots_prefix(qa_list):
    return "".join(f"Q: {q}\nA:{a}\n\n" for q, a in qa_list)
